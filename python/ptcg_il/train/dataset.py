"""ShardDataset + collate_fn (TRANSFORMER_IL_SPEC.md C.1–C.4).

mmap-based dataset reading fixed-shape shard slices, joined with meta.parquet
for per-sample metadata.  ``sample_weight`` is derived at load time from meta
columns via the C.3 formula.

The shards on disk are ``np.savez_compressed`` archives, and ``np.load``'s
``mmap_mode`` is **silently ignored** for ``.npz``: it returns an ``NpzFile``,
and materialising it decompresses every array into anonymous RAM.  The
compression ratio is ~94x (5.6 MB on disk -> 528 MB resident for a 50k-sample
shard, because the padded feature tensors are mostly zeros), and with
``num_workers=8`` every worker paid it independently — ~14 GB for the real
187k-sample corpus.

So each shard is decompressed **once** into a ``.npy``-per-key directory under
``shards/.mmap-cache/`` and mmap'd from there.  Resident memory then comes from
the OS page cache: shared between workers, evictable under pressure, and
independent of corpus size.  The cache is built in ``__init__`` (parent process,
one array at a time) so the workers only ever mmap.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

#: Subdirectory of ``shards/`` holding the decompressed, mmap-able copies.
MMAP_CACHE_DIRNAME = ".mmap-cache"

# ============================================================
# Constants from C.3 / C.10
# ============================================================
ALPHA_CTX: float = 0.5
ALPHA_ARCH: float = 0.5
W_LOST: float = 0.6

# Keys that should stay on CPU as int64 (indices, masks, types)
_INT_KEYS = frozenset({
    "tok_type", "tok_owner", "tok_zone",
    "opt_type", "opt_src_idx", "opt_tgt_idx", "sel_type", "sel_ctx",
    "action_idx", "minCount", "maxCount", "action_len", "stop_column",
    "log_len",
})

# Keys that should be float32 (can be cast to bf16)
_FLOAT_KEYS = frozenset({
    "cls_feat", "poke_feat", "hand_feat", "sum_feat",
    "stadium_present", "opt_scalar",
    "value_target", "sample_weight", "log_feat",
    "poke_card_feat", "hand_card_feat", "stadium_card_feat",
    "context_card_feat", "effect_card_feat",
    "discard_card_feat", "prize_card_feat",
    "opt_card_feat", "opt_attack_feat",
    "log_card_feat",
})

# Boolean masks
_BOOL_KEYS = frozenset({
    "tok_mask", "opt_mask", "discard_mask",
    "bel_valid", "bel_hand_valid",
})

# Belief labels are stored sparsely in the shards (index/count pairs) because a
# 60-card decklist touches at most 26 of ~300 vocab slots, so dense float32[V]
# rows would be ~97% zeros and would dominate shard size.  They are densified
# per sample here, on the way into the batch.
_BELIEF_SPARSE = (("bel_deck", "bel_deck_idx", "bel_deck_cnt"),
                  ("bel_hidden", "bel_hidden_idx", "bel_hidden_cnt"),
                  ("bel_hand", "bel_hand_idx", "bel_hand_cnt"))


def compute_sample_weights(
    meta: pd.DataFrame,
    alpha_ctx: float = ALPHA_CTX,
    alpha_arch: float = ALPHA_ARCH,
    w_lost: float = W_LOST,
) -> np.ndarray:
    """Compute per-sample weights from meta.parquet columns (C.3 formula).

    ``w = w_ctx[sel_ctx] * w_arch[archetype_self] * w_outcome``,
    normalized so that ``mean(weights) ≈ 1.0`` over the split.

    Parameters
    ----------
    meta : DataFrame
        Must contain columns ``sel_ctx``, ``archetype_self``, ``won``.
    alpha_ctx : float
        Exponent for rare-context balancing (0.5).
    alpha_arch : float
        Exponent for archetype balancing (0.5).
    w_lost : float
        Weight multiplier for lost-game decisions (0.6).

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

    weights = w_ctx * w_arch * w_out
    weights /= weights.mean()  # normalize to mean ≈ 1.0

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
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        shuffle: bool = False,
        seed: int = 42,
        archetype_self: int | None = None,
    ):
        data_dir = Path(data_dir)
        self.split = split
        self.archetype_self = archetype_self

        # n_all_cards — total engine cards, needed to densify sparse belief
        # labels.  Read from engine_card_features.npy (max engine card id + 1).
        self.n_all_cards: int = 0
        ecf_path = data_dir / "engine_card_features.npy"
        if ecf_path.exists():
            import numpy as _np
            ecf = _np.load(ecf_path, allow_pickle=True).item()
            self.n_all_cards = max(ecf.keys()) + 1 if ecf else 0

        # Load meta
        meta_path = data_dir / "meta.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.parquet not found at {meta_path}")
        self.meta = pd.read_parquet(meta_path)

        # Filter to split
        self.meta = self.meta[self.meta["shard"].str.startswith(split)].reset_index(drop=True)
        if len(self.meta) == 0:
            raise ValueError(f"No samples found for split '{split}' in meta.parquet")

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
        self.sample_weights = compute_sample_weights(self.meta)
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
        self._prepare_mmap_cache()

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

        The fallback branch reads the compressed archive into RAM — correct,
        but ~94x its on-disk size and per-worker.  It only runs when the cache
        could not be written.
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

        # Attach sample_weight and value_target from meta
        sample["sample_weight"] = torch.tensor(
            float(row["sample_weight"]), dtype=torch.float32
        )

        self._densify_belief(sample)
        return sample

    def _densify_belief(self, sample: dict[str, torch.Tensor]) -> None:
        """Sparse belief labels -> normalised distributions over the vocab.

        Replaces each ``bel_*_idx``/``bel_*_cnt`` pair with a single
        ``float32[V]`` row summing to 1 (or to 0 when the label is absent, which
        the ``bel_*_valid`` masks flag).  Rows are *not* renormalised when empty
        -- an all-zero row must stay all-zero, or a missing label would silently
        become a uniform target.

        Shards written before belief labels existed simply have no ``bel_*``
        keys; those samples get zero-filled rows and ``bel_valid=False``, so an
        old corpus trains exactly as it did before.
        """
        V = self.n_all_cards
        for dense_key, idx_key, cnt_key in _BELIEF_SPARSE:
            # No vocab.json => no way to size the row.  Fall through to the
            # zero-fill branch and mark the label invalid below, rather than
            # emitting a length-1 row the belief head would reject.
            if idx_key not in sample or V is None:
                sample[dense_key] = torch.zeros(V or 1, dtype=torch.float32)
                sample.pop(idx_key, None)
                sample.pop(cnt_key, None)
                continue
            idx = sample.pop(idx_key).long().clamp_(0, V - 1)
            cnt = sample.pop(cnt_key).float()
            row = torch.zeros(V, dtype=torch.float32)
            row.scatter_add_(0, idx, cnt)
            row[0] = 0.0  # PAD slot: padded entries land here carrying count 0
            total = row.sum()
            if total > 0:
                row /= total
            sample[dense_key] = row

        for key, dtype in (("bel_valid", torch.bool), ("bel_hand_valid", torch.bool),
                           ("bel_arch", torch.long)):
            if key not in sample or V is None:
                fill = -1 if dtype is torch.long else False
                sample[key] = torch.tensor(fill, dtype=dtype)
            else:
                sample[key] = sample[key].to(dtype)

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
