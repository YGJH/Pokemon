"""ShardDataset + collate_fn (TRANSFORMER_IL_SPEC.md C.1–C.4).

mmap-based dataset reading fixed-shape shard slices, joined with meta.parquet
for per-sample metadata.  ``sample_weight`` is derived at load time from meta
columns via the C.3 formula.

The shards on disk are ``np.savez_compressed`` archives, and ``np.load``'s
``mmap_mode`` is **silently ignored** for ``.npz``: it returns an ``NpzFile``,
and materialising it decompresses every array into anonymous RAM.  The
compression ratio is high (the padded feature tensors are mostly zeros), and
with ``num_workers=8`` every worker paid it independently.

So each shard is decompressed **once** into a ``.npy``-per-key directory under
``shards/.mmap-cache/`` and mmap'd from there.  Resident memory then comes from
the OS page cache: shared between workers, evictable under pressure, and
independent of corpus size.  The cache is built in ``__init__`` (parent process,
one array at a time) so the workers only ever mmap.

That trade only holds while the decompressed corpus fits in page cache, and it
stopped holding once shards stored the ``*_card_feat`` tensors: at ~0.5%
nonzero they compress ~240x, so a 26 MB shard became **6.2 GB** on decompression
and the train split's cache reached 47 GB against 30 GB of RAM.  A shuffled
epoch draws uniformly across every shard, so the working set is the whole split
— the kernel never leaves reclaim, and systemd-oomd kills the *entire* terminal
scope (bash, uv, python, all workers, no traceback) on memory **pressure**
rather than exhaustion.  The fix is upstream, in what gets stored: shards now
carry card *ids* and ``_rebuild_card_feats`` re-gathers from the 1.1 MB static
table, exactly as the ``bel_*`` labels are stored sparsely and densified here.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ptcg_il.featurizer import CARD_FEAT_SOURCES

logger = logging.getLogger(__name__)

#: Subdirectory of ``shards/`` holding the decompressed, mmap-able copies.
MMAP_CACHE_DIRNAME = ".mmap-cache"

#: Subdirectory of ``shards/`` holding one archetype's rows, densely packed.
#: One directory per (archetype, split) so the two datasets a training run
#: builds never share a stamp.
#:
#: The split comes **first** on purpose.  ``meta["shard"]`` is filtered with
#: ``str.startswith(split)`` and that prefix is the only thing separating the
#: 80/10/10 episode split once rows are renamed — a dense shard whose name did
#: not start with its split would leak val rows into a train dataset the next
#: time anything re-filtered.  It is also why the directory is not dot-hidden:
#: everything that scans ``shards/`` filters on ``*.npz``, so a visible
#: directory costs nothing, and a leading dot would break the prefix.
REPACK_DIRNAME_TEMPLATE = "{split}-repack-a{arch}"


def repack_dirname(archetype_self: int, split: str) -> str:
    """Directory name, under ``shards/``, of a deck's dense copy of *split*."""
    return REPACK_DIRNAME_TEMPLATE.format(arch=int(archetype_self), split=split)


#: Offset between the UUID epoch (1582-10-15) and the Unix epoch, in 100-ns
#: ticks — for decoding the timestamp embedded in a v1 UUID.
_GREGORIAN_OFFSET_100NS = 0x01B21DD213814000

_ID_RE = re.compile(rb'"id"\s*:\s*"([0-9a-fA-F-]{36})"')


def _episode_date_from_json(path: Path) -> datetime.date | None:
    """Play date embedded in an episode's UUIDv1 ``id``; None when unreadable.

    The episode JSON has no explicit play-date field, but Kaggle stamps every
    episode with a UUIDv1 at creation, and a v1 UUID embeds its own creation
    timestamp.  (Cross-checked against the download-day layout: 123/123
    sampled episodes landed on their directory's date.)

    The ``id`` sits within the first ~150 bytes, so a 4 KB head read avoids
    parsing megabytes of step history; a full parse is the fallback.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
        m = _ID_RE.search(head)
        if m is not None:
            raw_id = m.group(1).decode()
        else:
            with path.open() as fh:
                raw_id = json.load(fh).get("id")
            if not isinstance(raw_id, str):
                return None
        u = uuid.UUID(raw_id)
        if u.version != 1:
            return None
        return datetime.datetime.fromtimestamp(
            (u.time - _GREGORIAN_OFFSET_100NS) / 1e7, tz=datetime.timezone.utc
        ).date()
    except (OSError, ValueError):
        return None


def episode_dates(raw_dir: str | Path) -> dict[str, datetime.date]:
    """Map episode id → play date, read from each episode JSON's UUIDv1 id.

    Falls back to the download-day directory name (``raw/YYYY-MM-DD/``) when
    a file carries no usable id; non-date directory names are ignored.
    """
    out: dict[str, datetime.date] = {}
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return out
    for day_dir in raw_dir.iterdir():
        if not day_dir.is_dir():
            continue
        try:
            day = datetime.date.fromisoformat(day_dir.name)
        except ValueError:
            continue
        for f in day_dir.iterdir():
            out[f.stem] = _episode_date_from_json(f) or day
    return out


def arch_age_drop_mask(
    meta: pd.DataFrame,
    arch: int,
    cutoff: datetime.date,
    ep_dates: dict[str, datetime.date],
) -> pd.Series:
    """Boolean mask — True for rows of *arch* whose episode predates *cutoff*.

    Rows whose episode has no raw file (no known date) are kept: an unknown
    date is not an old date, and dropping on suspicion would silently shrink
    the corpus if raw/ is ever cleaned up before shards/ are rebuilt.
    """
    days = pd.to_datetime(meta["episode_id"].map(ep_dates))
    return (meta["archetype_self"] == arch) & (days < pd.Timestamp(cutoff))


# ``MADV_RANDOM`` on these mappings was tried and **rejected on measurement**.
# It does what it claims — physical reads fall ~10x, 1585 -> 156 KiB/sample —
# but it is slower in every paired comparison, because small synchronous random
# reads are latency-bound on NVMe while default readahead pipelines.  At the
# training configuration (B=1024, 8 workers, full epoch, cache evicted) it cost
# 170.5 vs 168.8 ms/step sparse and 165.0 vs 163.2 ms/step repacked; single
# threaded the gap is 4x.  The read-volume argument is real but the repack below
# buys it far more cheaply, by shrinking the working set instead of the reads.

# ============================================================
# Constants from C.3 / C.10
# ============================================================
ALPHA_CTX: float = 0.5
ALPHA_ARCH: float = 0.5
#: Signed outcome weight: a lost-game decision pushes its expert action's
#: probability *down* at 0.3x the rate a won-game decision pushes its up.
#: Kept small because per-game outcome is a noisy proxy for move quality
#: (TCG variance), so imitation stays dominant at 1 : 0.3.
W_LOST: float = -0.3

# Keys that should stay on CPU as int64 (indices, masks, types)
_INT_KEYS = frozenset({
    "tok_type", "tok_owner", "tok_zone",
    "opt_type", "opt_src_idx", "opt_tgt_idx", "opt_bench_idx", "opt_group",
    "sel_type", "sel_ctx",
    "action_idx", "minCount", "maxCount", "action_len", "stop_column",
} | {id_key for id_key, _ in CARD_FEAT_SOURCES.values()})

# Keys that should be float32 (can be cast to bf16)
_FLOAT_KEYS = frozenset({
    "cls_feat", "poke_feat", "hand_feat", "sum_feat",
    "stadium_present", "opt_scalar",
    "value_target", "sample_weight",
    "poke_card_feat", "hand_card_feat", "stadium_card_feat",
    "context_card_feat", "effect_card_feat",
    "discard_card_feat", "prize_card_feat",
    "opt_card_feat", "opt_attack_feat",
})

# Boolean masks
_BOOL_KEYS = frozenset({
    "tok_mask", "opt_mask", "discard_mask",
})


def compute_sample_weights(
    meta: pd.DataFrame,
    alpha_ctx: float = ALPHA_CTX,
    alpha_arch: float = ALPHA_ARCH,
    w_lost: float = W_LOST,
) -> np.ndarray:
    """Compute per-sample weights from meta.parquet columns (C.3 formula).

    ``w = w_ctx[sel_ctx] * w_arch[archetype_self] * w_outcome * w_skill``,
    normalized so that ``mean(|weights|) ≈ 1.0`` over the split.

    ``w_skill`` is the ``skill_w`` column — ``exp(20 * (wilson_lb - 0.5))`` of
    the row's team, see :func:`ptcg_mine.stats.skill_weight`.  It is what
    replaced the top-K expert filter, so on a corpus built without that filter
    it is the *only* thing keeping weak players from being imitated equally.

    A ``meta.parquet`` predating the column falls back to 1.0 with a warning
    rather than raising, and that fallback is *correct* for it: every row in a
    filtered corpus is an expert's, so uniform is what skill weighting would
    have produced anyway.  The combination that would be wrong — an unfiltered
    corpus read by code that ignores skill — cannot occur, since the column and
    the filter removal ship in the same change and ``stamp.py`` fingerprints
    both files.

    Parameters
    ----------
    meta : DataFrame
        Must contain columns ``sel_ctx``, ``archetype_self``, ``won``;
        ``skill_w`` when built by a post-filter-removal ``build_shards``.
    alpha_ctx : float
        Exponent for rare-context balancing (0.5).
    alpha_arch : float
        Exponent for archetype balancing (0.5).
    w_lost : float
        Weight multiplier for lost-game decisions (-0.3).  Negative values
        make the loss *decrease* the expert action's probability on those
        rows instead of imitating it; zero drops lost games from the loss.

    Returns
    -------
    weights : float32 ndarray[Nsamples]
    """
    n = len(meta)
    if n == 0:
        return np.array([], dtype=np.float32)

    sel_ctx = meta["sel_ctx"].to_numpy(dtype=np.int64)
    archetype_self = meta["archetype_self"].to_numpy(dtype=np.int64)
    won = meta["won"].to_numpy(dtype=bool)

    # Context weights
    ctx_vals, ctx_counts = np.unique(sel_ctx, return_counts=True)
    ctx_count_map = dict(zip(ctx_vals, ctx_counts))
    w_ctx = np.array([n / ctx_count_map[c] for c in sel_ctx], dtype=np.float64)
    w_ctx = w_ctx ** alpha_ctx

    # Archetype weights
    arch_vals, arch_counts = np.unique(archetype_self, return_counts=True)
    arch_count_map = dict(zip(arch_vals, arch_counts))
    w_arch = np.array([n / arch_count_map[a] for a in archetype_self], dtype=np.float64)
    w_arch = w_arch ** alpha_arch

    # Outcome weight
    w_out = np.where(won, 1.0, w_lost).astype(np.float64)

    # Skill weight (see docstring for why a missing column is not fatal)
    if "skill_w" in meta.columns:
        w_skill = meta["skill_w"].to_numpy(dtype=np.float64)
    else:
        logger.warning(
            "meta.parquet has no 'skill_w' column — treating every team as "
            "equally skilled. Correct for an expert-filtered corpus, wrong for "
            "one built by a current build_shards; rebuild shards if unsure."
        )
        w_skill = np.ones(n, dtype=np.float64)

    weights = w_ctx * w_arch * w_out * w_skill
    # Normalize by the mean *magnitude*.  With a signed w_lost the plain mean
    # can approach (or cross) zero, which would rescale the loss by a
    # near-infinite or sign-flipping factor; for non-negative weights this is
    # exactly the old mean-normalization.
    scale = np.abs(weights).mean()
    if scale > 0:
        weights /= scale  # normalize to mean-magnitude ≈ 1.0

    return weights.astype(np.float32)


def _cache_stamp(npz_path: Path) -> dict:
    """Identity of the shard a cache directory was built from."""
    st = npz_path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _cache_is_current(cache_dir: Path, npz_path: Path) -> bool:
    """True only if the cache was built from *this* shard and is complete.

    Completeness matters as much as freshness: an interrupted build leaves a
    directory that exists but is missing keys, and reusing it would fail deep
    inside ``__getitem__`` rather than here.
    """
    stamp_path = cache_dir / "_source.json"
    if not stamp_path.is_file():
        return False
    try:
        stamp = json.loads(stamp_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if stamp.get("source") != _cache_stamp(npz_path):
        return False
    present = {p.stem for p in cache_dir.glob("*.npy")}
    return present == set(stamp.get("keys", []))


def build_mmap_cache(npz_path: Path, cache_root: Path) -> Path | None:
    """Decompress *npz_path* into a ``.npy``-per-key directory, once.

    Returns the directory, or ``None`` when it cannot be written — a read-only
    or full corpus directory degrades to the old in-RAM path rather than
    failing training outright.

    Arrays are converted one at a time: ``dict(np.load(...))`` would hold all
    45 of them at once (528 MB), which is the peak this whole mechanism exists
    to avoid.  The build goes to a temporary sibling and is renamed into place,
    so an interrupted run never leaves a half-cache that looks valid.
    """
    npz_path = Path(npz_path)
    cache_dir = Path(cache_root) / npz_path.name

    if _cache_is_current(cache_dir, npz_path):
        return cache_dir

    tmp_dir = cache_dir.with_name(f"{cache_dir.name}.tmp-{os.getpid()}")
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        keys: list[str] = []
        with np.load(npz_path, allow_pickle=False) as z:
            for key in z.files:
                np.save(tmp_dir / f"{key}.npy", z[key])
                keys.append(key)
        (tmp_dir / "_source.json").write_text(
            json.dumps({"source": _cache_stamp(npz_path), "keys": keys})
        )
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(tmp_dir, cache_dir)
    except OSError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Another process may have finished the same cache while this one
        # failed; use it if it is valid, otherwise fall back to in-RAM loading.
        if _cache_is_current(cache_dir, npz_path):
            return cache_dir
        logger.warning(
            "Cannot build mmap cache for %s (%s) — falling back to loading the "
            "shard into RAM. Expect high memory use with num_workers > 0.",
            npz_path.name, exc,
        )
        return None
    return cache_dir


# ============================================================
# Per-archetype repack (locality)
# ============================================================
#
# MADV_RANDOM fixes how much the kernel reads per fault; it does nothing about
# how *scattered* the wanted rows are.  A specialist's rows are ~8% of the rows
# of the ~126 shards they appear in, so a shuffled epoch's working set is the
# entire split (44.7 GB traversed to consume 3.5 GB) and can never be cached.
#
# The rows themselves are a pure function of (shards, archetype filter), so they
# can be copied once into dense shards where every row is wanted.  The copy is
# written as bare ``.npy`` per key — not ``.npz`` — because the whole point is
# that reading it costs a page fault and no decompression.


def _repack_probe_keys(z) -> list[str]:
    return sorted(z.files)


def _gather_block(
    shards_dir: Path,
    keys: list[str],
    names: np.ndarray,
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Read the ``(shard, row)`` pairs in *names*/*rows* into dense arrays.

    Output position ``i`` holds the row named by ``names[i]``/``rows[i]``, so
    the block preserves meta order — which is what makes the rewritten
    ``(shard, row)`` pair pure arithmetic on the meta index.

    Reads are grouped by source shard and sorted by row within it: one open per
    source shard, and one decompression per key.  Only a single source array is
    materialised at a time, so peak memory is one key of one shard rather than
    the shard.
    """
    m = len(names)
    out: dict[str, np.ndarray] = {}
    order = np.argsort(names, kind="stable")
    sorted_names = names[order]
    for shard_name in dict.fromkeys(sorted_names.tolist()):
        dst = order[sorted_names == shard_name]
        src = rows[dst]
        within = np.argsort(src, kind="stable")
        dst, src = dst[within], src[within]
        with np.load(shards_dir / shard_name, allow_pickle=False) as z:
            if _repack_probe_keys(z) != keys:
                raise ValueError(
                    f"shard {shard_name} has a different key set than the first "
                    "shard of this split — the corpus mixes formats"
                )
            for key in keys:
                arr = z[key]
                if key not in out:
                    out[key] = np.empty((m, *arr.shape[1:]), dtype=arr.dtype)
                out[key][dst] = arr[src]
                del arr
    return out


def _verify_block(
    shards_dir: Path,
    keys: list[str],
    names: np.ndarray,
    rows: np.ndarray,
    block: dict[str, np.ndarray],
    n_probe: int = 8,
) -> None:
    """Re-read a spread of rows from the source and demand exact equality.

    The repack is a rearrange, so the only bug it can have is an index one —
    and an index bug is invisible downstream: training just fits the wrong
    labels.  Probing a bounded sample rather than the whole block keeps this
    affordable (it re-decompresses every key of every shard it touches) while
    still exercising the grouping, the within-shard sort and the destination
    scatter.
    """
    m = len(names)
    if m == 0:
        return
    probe = np.unique(np.linspace(0, m - 1, min(n_probe, m)).astype(np.int64))
    for i in probe:
        i = int(i)
        with np.load(shards_dir / str(names[i]), allow_pickle=False) as z:
            for key in keys:
                if not np.array_equal(z[key][int(rows[i])], block[key][i]):
                    raise ValueError(
                        f"repack mismatch at {names[i]}:{rows[i]} key {key!r}"
                    )


def _repack_is_current(repack_dir: Path, expect: dict) -> dict | None:
    """Return the recorded stamp when the dense copy is reusable, else None.

    Freshness and completeness both matter, for the same reasons as the mmap
    cache: card ids are vocab indices, so a dense copy of a rebuilt corpus is
    silently mislabelled, and an interrupted build leaves a directory that
    exists but is missing keys.
    """
    stamp_path = repack_dir / "_source.json"
    if not stamp_path.is_file():
        return None
    try:
        stamp = json.loads(stamp_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if any(stamp.get(k) != v for k, v in expect.items()):
        return None
    keys = set(stamp.get("keys") or ())
    sps = stamp.get("samples_per_shard") or 0
    shards = stamp.get("shards") or []
    if not keys or sps <= 0:
        return None
    if len(shards) != -(-int(stamp["n_rows"]) // int(sps)):
        return None
    for name in shards:
        shard_dir = repack_dir / name
        if not shard_dir.is_dir():
            return None
        if {p.stem for p in shard_dir.glob("*.npy")} != keys:
            return None
    return stamp


def build_repack(
    shards_dir: Path,
    meta: pd.DataFrame,
    repack_dir: Path,
    split: str,
    stamp: dict,
) -> dict | None:
    """Copy *meta*'s rows out of the sparse shards into dense ``.npy`` shards.

    Returns the completed stamp (the input plus ``samples_per_shard``, ``keys``
    and ``shards``), or ``None`` when the copy could not be written — a
    read-only or full corpus directory degrades to reading the originals rather
    than failing training outright.

    Rows per dense shard are inherited from the source shards rather than
    chosen here: the corpus already sized them from ``--mem-budget-gb``, and
    re-deriving would put a second, independently-stale constant in the loader.

    The build goes to a temporary sibling and is renamed into place, so an
    interrupted run never leaves a half-copy that looks valid.
    """
    src_names = meta["shard"].astype(str).to_numpy()
    src_rows = meta["row"].to_numpy(dtype=np.int64)
    n = len(meta)

    tmp_dir = repack_dir.with_name(f"{repack_dir.name}.tmp-{os.getpid()}")
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        with np.load(shards_dir / str(src_names[0]), allow_pickle=False) as z:
            keys = _repack_probe_keys(z)
            if not keys:
                raise ValueError(f"shard {src_names[0]} is empty")
            samples_per_shard = int(z[keys[0]].shape[0])
        if samples_per_shard <= 0:
            raise ValueError(f"shard {src_names[0]} has no rows")

        dense: list[str] = []
        for start in range(0, n, samples_per_shard):
            end = min(start + samples_per_shard, n)
            name = f"{split}-{len(dense):05d}"
            out_dir = tmp_dir / name
            out_dir.mkdir()
            block = _gather_block(
                shards_dir, keys, src_names[start:end], src_rows[start:end]
            )
            if not dense:
                _verify_block(
                    shards_dir, keys, src_names[start:end], src_rows[start:end], block
                )
            for key, arr in block.items():
                np.save(out_dir / f"{key}.npy", arr)
            del block
            dense.append(name)

        stamp = dict(stamp, samples_per_shard=samples_per_shard, keys=keys, shards=dense)
        (tmp_dir / "_source.json").write_text(json.dumps(stamp))
        if repack_dir.exists():
            shutil.rmtree(repack_dir)
        os.replace(tmp_dir, repack_dir)
    except (OSError, ValueError) as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.warning(
            "Cannot build the dense repack at %s (%s) — reading the archetype's "
            "rows from the original shards instead. Expect a loader-bound "
            "training step.",
            repack_dir, exc,
        )
        return None
    return stamp


class ShardDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset over pre-featurized .npz shards, mmap'd for OS page-cache reuse.

    Each ``__getitem__`` reads one sample (dict-of-tensors) by loading the
    appropriate row slice from the underlying mmap'd shard array.  Batch
    assembly happens in ``collate_fn`` via ``torch.stack``.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``shards/`` and ``meta.parquet``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    shuffle : bool
        If True, shuffle indices on init (used for training).
    seed : int
        RNG seed for shuffle.
    archetype_self : int, optional
        If given, keep only decision points whose ``archetype_self`` matches —
        i.e. train a per-deck specialist.  The self archetypes are near-disjoint
        decks (pairwise multiset-Jaccard <= 0.17, no card common to all), so a
        single model trained across all of them is fitting several unrelated
        policies at once.  ``None`` (default) keeps every deck, the original
        behaviour.
    exclude_arch_before : tuple[int, datetime.date], optional
        ``(archetype_self, cutoff)`` — drop that archetype's rows whose
        episode predates *cutoff* (play dates are decoded from the UUIDv1
        ``id`` inside each raw episode JSON).  Applied to whichever split this
        dataset loads; episode-level, so the 80/10/10 split is unaffected.
    raw_dir : str or Path, optional
        Where episode dates are read from.  Defaults to ``data_dir``'s
        sibling ``raw/``.
    w_lost : float
        Outcome weight for lost-game rows, forwarded to
        :func:`compute_sample_weights` (default ``W_LOST``).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        shuffle: bool = False,
        seed: int = 42,
        archetype_self: int | None = None,
        exclude_arch_before: tuple[int, datetime.date] | None = None,
        raw_dir: str | Path | None = None,
        w_lost: float = W_LOST,
    ):
        data_dir = Path(data_dir)
        self.split = split
        self.archetype_self = archetype_self

        # The dataset does *not* build the static card/attack tables.  It used
        # to, to re-gather the ``*_card_feat`` tensors per sample; that gather
        # now happens once per batch on the model's device
        # (``Policy._gather_card_feats``), which is the whole point — the
        # gathered tensors were 93% of a sample's bytes and never needed to
        # cross the DataLoader queue.  Samples carry ids; ``Policy`` owns the
        # tables.

        # Load meta
        meta_path = data_dir / "meta.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.parquet not found at {meta_path}")
        self.meta = pd.read_parquet(meta_path)

        # Filter to split
        self.meta = self.meta[self.meta["shard"].str.startswith(split)].reset_index(drop=True)
        if len(self.meta) == 0:
            raise ValueError(f"No samples found for split '{split}' in meta.parquet")

        # Optional per-archetype age filter, before the deck filter and the
        # sample weights so both see the same row set.  Whole episodes drop
        # together (they share a date), so the split stays episode-level.
        if exclude_arch_before is not None:
            arch, cutoff = exclude_arch_before
            ep_dates = episode_dates(
                raw_dir if raw_dir is not None else data_dir.parent / "raw"
            )
            drop = arch_age_drop_mask(self.meta, arch, cutoff, ep_dates)
            n_dropped = int(drop.sum())
            self.meta = self.meta[~drop].reset_index(drop=True)
            logger.info(
                "Excluded %d archetype-%d row(s) from the %s split (episodes "
                "before %s); %d row(s) kept.",
                n_dropped, arch, self.split, cutoff, len(self.meta),
            )
            if len(self.meta) == 0:
                raise ValueError(
                    f"exclude_arch_before={exclude_arch_before} removed every "
                    f"row of split '{split}'"
                )

        # Optional per-deck filter.  Applied after the split filter so the
        # 80/10/10 episode-level split still holds within each deck.
        if archetype_self is not None:
            if "archetype_self" not in self.meta.columns:
                raise ValueError(
                    "meta.parquet has no 'archetype_self' column — rebuild shards "
                    "with `ptcg_il.cli build-shards` to enable per-deck training"
                )
            self.meta = self.meta[
                self.meta["archetype_self"] == archetype_self
            ].reset_index(drop=True)
            if len(self.meta) == 0:
                raise ValueError(
                    f"No samples for archetype_self={archetype_self} in split "
                    f"'{split}'. Check `archetypes.json` for valid ids."
                )

        # Compute sample weights from meta columns (C.3)
        self.sample_weights = compute_sample_weights(self.meta, w_lost=w_lost)
        self.meta["sample_weight"] = self.sample_weights

        # Shuffle row order if requested
        self._indices = np.arange(len(self.meta))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self._indices)

        # Lazily open shards — mmap each unique shard once
        self._shard_cache: dict[str, dict[str, np.ndarray]] = {}
        self._shards_dir = data_dir / "shards"
        self._mmap_dirs: dict[str, Path] = {}
        self._repack_dir: Path | None = None

        # A specialist reads a few percent of every shard it touches, so it gets
        # a dense copy of just its rows.  That copy is read straight out of the
        # .npz archives, which means the generalist mmap cache — the whole
        # decompressed split, tens of GB — is never built for a specialist run.
        if archetype_self is None or not self._prepare_repack(
            archetype_self, meta_path
        ):
            self._prepare_mmap_cache()

        self._drop_keys: frozenset[str] = frozenset()
        self._check_card_id_coverage()

    #: Keys the model gathers for itself, from the ids in ``CARD_FEAT_SOURCES``.
    _DERIVED_FEAT_KEYS = frozenset(CARD_FEAT_SOURCES)

    def _check_card_id_coverage(self) -> None:
        """Refuse a shard that has ``*_card_feat`` but not the id behind it.

        Cards reach the model as ids now; ``Policy`` gathers the features from
        its static tables.  A shard written before that (features stored, no
        ids) has nothing for the gather to read, and **every layer downstream
        tolerates the absence** — ``TokenEmbedder`` skips a missing feature key,
        the pointer head would see zeros, and the run would train to convergence
        having never seen a card.  There is no later point at which this becomes
        visible, so it has to fail here, while the corpus is still named.

        A shard carrying *both* is merely redundant: the ids win and the stored
        features are dropped on read rather than shipped through the queue.
        """
        shard_name = str(self.meta.iloc[0]["shard"])
        keys = set(self._open_shard(shard_name))

        orphaned = sorted(
            feat_key for feat_key, (id_key, _) in CARD_FEAT_SOURCES.items()
            if feat_key in keys and id_key not in keys
        )
        if orphaned:
            raise ValueError(
                f"shard {shard_name} stores card features {orphaned} but not the "
                "card ids behind them. This corpus predates on-device card "
                "gathering and the model has no way to read it — rebuild shards "
                "with `ptcg_il.cli build-shards`."
            )
        self._drop_keys = frozenset(self._DERIVED_FEAT_KEYS & keys)

    def _prepare_repack(self, archetype_self: int, meta_path: Path) -> bool:
        """Build or adopt this deck's dense shards.  False => use the originals.

        Returning False rather than raising is the point: the repack is an
        optimisation, and a corpus directory that cannot be written must still
        train.
        """
        repack_dir = self._shards_dir / repack_dirname(archetype_self, self.split)
        source_names = sorted(set(self.meta["shard"].astype(str)))
        for name in source_names:
            if not (self._shards_dir / name).exists():
                raise FileNotFoundError(
                    f"Shard file not found: {self._shards_dir / name}"
                )

        # meta.parquet is stamped alongside the shards because it, not the
        # shards, decides *which* rows this deck owns.
        expect = {
            "archetype_self": int(archetype_self),
            "split": self.split,
            "n_rows": int(len(self.meta)),
            "sources": {n: _cache_stamp(self._shards_dir / n) for n in source_names},
            "meta": _cache_stamp(meta_path),
        }

        stamp = _repack_is_current(repack_dir, expect)
        if stamp is None:
            stamp = build_repack(
                self._shards_dir, self.meta, repack_dir, self.split, expect
            )
            if stamp is None:
                return False
            logger.info(
                "Repacked %d %s row(s) of archetype %d from %d sparse shard(s) "
                "into %d dense shard(s) at %s (%.1f GB); reused on later runs.",
                len(self.meta), self.split, archetype_self, len(source_names),
                len(stamp["shards"]), repack_dir,
                sum(p.stat().st_size for p in repack_dir.rglob("*.npy")) / 1e9,
            )

        self._adopt_repack(repack_dir, stamp)
        return True

    def _adopt_repack(self, repack_dir: Path, stamp: dict) -> None:
        """Point ``self.meta`` at the dense shards.

        The dense copy preserves meta order, so the new ``(shard, row)`` pair is
        arithmetic on the row's position — nothing has to be read back to
        recover it, which is what lets a cached repack skip touching the
        originals entirely.
        """
        samples_per_shard = int(stamp["samples_per_shard"])
        dirname = repack_dir.name
        names: list[str] = stamp["shards"]
        pos = np.arange(len(self.meta))
        labels = np.array([f"{dirname}/{n}" for n in names], dtype=object)

        self.meta = self.meta.copy()
        self.meta["shard"] = labels[pos // samples_per_shard]
        self.meta["row"] = pos % samples_per_shard

        self._repack_dir = repack_dir
        for name in names:
            self._mmap_dirs[f"{dirname}/{name}"] = repack_dir / name

    def _prepare_mmap_cache(self) -> None:
        """Decompress this split's shards to mmap-able form, before forking.

        Done here rather than on first access because DataLoader workers are
        forked after construction: a lazy build would have all 8 of them
        decompress the same shard at the same time, which is precisely the
        memory spike being avoided.  Only the shards this split references are
        touched, so a val dataset does not pay for the train shards.
        """
        cache_root = self._shards_dir / MMAP_CACHE_DIRNAME
        n_built = 0
        for shard_name in sorted(set(self.meta["shard"].astype(str))):
            path = self._shards_dir / shard_name
            if not path.exists():
                raise FileNotFoundError(f"Shard file not found: {path}")
            stale = not _cache_is_current(cache_root / path.name, path)
            cache_dir = build_mmap_cache(path, cache_root)
            if cache_dir is not None:
                self._mmap_dirs[shard_name] = cache_dir
                n_built += stale

        # Drop caches whose shard is gone — rebuilding the corpus with a
        # different shard count would otherwise strand GBs.  Keyed on the
        # source file existing, so a train dataset never evicts val caches.
        if cache_root.is_dir():
            for orphan in cache_root.iterdir():
                if orphan.is_dir() and not (self._shards_dir / orphan.name).exists():
                    shutil.rmtree(orphan, ignore_errors=True)

        if n_built:
            # Not silent: this is why the first run pauses before step 0, and
            # why shards/ grew by roughly the decompressed corpus size.
            logger.info(
                "Decompressed %d %s shard(s) to %s (%.1f GB total) for "
                "memory-mapped access; reused on later runs.",
                n_built, self.split, cache_root,
                sum(p.stat().st_size for p in cache_root.rglob("*.npy")) / 1e9,
            )

    def _open_shard(self, shard_name: str) -> dict[str, np.ndarray]:
        """Return the shard's arrays, mmap'd when a cache is available.

        ``_mmap_dirs`` holds both flavours of ``.npy``-per-key directory — a
        decompressed mmap cache and a dense repack — so there is one code path
        for reading them.

        The fallback branch reads the compressed archive into RAM — correct,
        but ~94x its on-disk size and per-worker.  It only runs when neither
        could be written.
        """
        if shard_name not in self._shard_cache:
            cache_dir = self._mmap_dirs.get(shard_name)
            if cache_dir is not None:
                self._shard_cache[shard_name] = {
                    p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
                    for p in cache_dir.glob("*.npy")
                }
            else:
                path = self._shards_dir / shard_name
                if not path.exists():
                    raise FileNotFoundError(f"Shard file not found: {path}")
                self._shard_cache[shard_name] = dict(
                    np.load(path, allow_pickle=False)
                )
        return self._shard_cache[shard_name]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one sample as a dict-of-tensors.

        Keys match the featurizer contract (Appendix A.4) plus
        ``sample_weight``, ``value_target``.
        """
        row_idx = int(self._indices[idx])
        row = self.meta.iloc[row_idx]
        shard_name = str(row["shard"])
        sample_row = int(row["row"])

        shard = self._open_shard(shard_name)

        sample: dict[str, torch.Tensor] = {}
        for key in shard:
            if key in self._drop_keys:
                continue  # redundant with the ids; Policy gathers these
            arr = shard[key][sample_row]  # slice one row
            # Convert to torch tensor
            if key in _BOOL_KEYS:
                sample[key] = torch.from_numpy(np.asarray(arr).copy()).bool()
            elif key in _INT_KEYS:
                sample[key] = torch.from_numpy(np.asarray(arr).copy()).long()
            else:
                sample[key] = torch.from_numpy(np.asarray(arr).copy()).float()

        # Backward compat: old shards lack stop_column
        if "stop_column" not in sample:
            sample["stop_column"] = torch.tensor(-1, dtype=torch.long)

        # Backward compat: shards written before opt_group existed.  Every
        # option becomes its own group, which makes group-marginal CE identical
        # to plain CE rather than silently merging unrelated options.
        if "opt_group" not in sample:
            g = torch.arange(sample["opt_mask"].shape[0], dtype=torch.long)
            sample["opt_group"] = torch.where(sample["opt_mask"], g, torch.full_like(g, -1))

        # Attach sample_weight and value_target from meta
        sample["sample_weight"] = torch.tensor(
            float(row["sample_weight"]), dtype=torch.float32
        )

        return sample

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for i in range(len(self)):
            yield self[i]

    def get_raw(self, idx: int) -> dict[str, np.ndarray]:
        """Return the raw numpy row without torch conversion (for testing)."""
        row_idx = int(self._indices[idx])
        row = self.meta.iloc[row_idx]
        shard_name = str(row["shard"])
        sample_row = int(row["row"])
        shard = self._open_shard(shard_name)
        return {k: np.asarray(shard[k][sample_row]).copy() for k in shard}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """torch.stack per-key along batch dim B (C.4).

    Derives ``encoder_padding_mask = ~tok_mask`` at collate time.

    Parameters
    ----------
    batch : list[dict]
        List of per-sample dicts from ShardDataset.__getitem__.

    Returns
    -------
    dict
        Batched dict with leading dim B for all tensors.
    """
    if not batch:
        return {}

    # torch.stack every key
    merged: dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for key in keys:
        tensors = [s[key] for s in batch]
        # Scalars need special handling — ensure they become [B] tensors
        shape0 = tensors[0].shape
        if shape0 == ():  # scalar
            merged[key] = torch.stack(tensors)  # → [B]
        else:
            merged[key] = torch.stack(tensors)

    # Derive encoder_padding_mask
    merged["encoder_padding_mask"] = ~merged["tok_mask"]

    return merged
