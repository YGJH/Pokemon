"""Phase 3 shard writer + meta.parquet builder.

Bridges corpus-mining output (vocab.json, archetypes.json) to the training
pipeline.  Iterates kept (episode, player) pairs, calls featurize(), packs
samples into fixed-shape shard .npz files, and builds meta.parquet.

See docs/plans/corpus-mining-plan.md Appendix D.4 and
TRANSFORMER_IL_SPEC.md Appendix C.1 for the shard format contract.
"""

import hashlib
import json
import logging
import os
from rich.logging import RichHandler
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_il.belief_labels import (
    build_belief_labels,
    empty_belief_labels,
    hand_after,
    opp_hand_timeline,
)
from ptcg_il.featurizer import (
    CARD_FEAT_SOURCES,
    featurize,
    load_engine_tables,
    normalize_vocab,
)
from ptcg_mine.archetype import Archetype, assign_archetype
from ptcg_mine.config import MineConfig
from ptcg_mine.episode import (
    OK,
    _PARALLEL_SCAN_MIN_FILES,
    _SCAN_CHUNKSIZE,
    deck_of,
    load_episode,
    project_for_selection,
    rewards,
    scan_episode,
    scan_projections,
    teams,
    validate_episode,
)
from ptcg_mine.stats import team_skill_weights
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

#: Legacy fixed shard size.  No longer the default -- `build_shards` derives
#: `samples_per_shard` from `DEFAULT_MEM_BUDGET_GB` unless a caller pins it --
#: but kept because it is what every shard written before the budget existed
#: used, and reproducing an old corpus means passing it explicitly.
SAMPLES_PER_SHARD = 50000

#: Default ceiling on the writer's own resident memory, in GiB.  Only the
#: parent's shard buffers are sized from it -- see `samples_per_shard_for_budget`
#: for why that is the term worth bounding, and `_MetaWriter` for the other one.
#:
#: 2.0 keeps the whole writer near ~2.3 GiB on the 55k-episode corpus, which
#: leaves the page cache room to stream 240 GB of raw JSON without driving the
#: memory-pressure signal `systemd-oomd` kills on.  It is a budget, not a
#: measurement: raise it on a bigger box, lower it if the writer has to share.
DEFAULT_MEM_BUDGET_GB = 2.0

# Required meta.parquet columns (D.4)
META_COLUMNS = [
    "sample_uid",
    "shard",
    "row",
    "episode_id",
    "player",
    "team",
    "archetype_self",
    "archetype_opp",
    "sel_type",
    "sel_ctx",
    "minCount",
    "maxCount",
    "won",
    "skill_w",
]

#: Explicit arrow schema for meta.parquet.
#:
#: Streaming the file in row groups means pyarrow no longer sees every row
#: before choosing types, so the schema cannot be inferred from the first batch:
#: ``archetype_opp`` is legitimately all ``-1`` early in a corpus and
#: ``skill_w`` can be integral, either of which would infer a narrower type than
#: a later batch needs and raise mid-write.  Pinning it also freezes the dtypes
#: `pd.read_parquet` hands training, which `test_meta_columns_and_types` asserts.
#: ``large_string`` matches what pandas 3.x wrote when the whole frame was
#: materialised at once, so existing meta.parquet files stay type-compatible.
_META_SCHEMA = pa.schema([
    ("sample_uid", pa.large_string()),
    ("shard", pa.large_string()),
    ("row", pa.int64()),
    ("episode_id", pa.large_string()),
    ("player", pa.int64()),
    ("team", pa.large_string()),
    ("archetype_self", pa.int64()),
    ("archetype_opp", pa.int64()),
    ("sel_type", pa.int64()),
    ("sel_ctx", pa.int64()),
    ("minCount", pa.int64()),
    ("maxCount", pa.int64()),
    ("won", pa.bool_()),
    ("skill_w", pa.float64()),
])

#: Meta rows buffered before a row group is written.  At ~781 B/row this caps
#: the writer's meta footprint at ~40 MB regardless of corpus size.
_META_FLUSH_ROWS = 50_000

class _MetaWriter:
    """Streaming meta.parquet writer: buffers rows, flushes row groups.

    The list-of-dicts this replaces was the only term in the writer with no
    ceiling at all -- measured 781 B/row, so 2.67 GiB at the 3.67M-row corpus,
    plus ~0.94 GiB more transiently when ``pd.DataFrame`` built columns on top
    of a list that was still alive.  Rows are appended once and never read back,
    which is exactly the access pattern a row-group writer wants.

    ``split_counts`` is tallied here rather than recovered from the finished
    frame, because there is no longer a frame in memory to filter.

    Not a context manager on purpose: `close` must run on the success path
    *before* the summary is built, and `abort` on the failure path leaves the
    partial file to be overwritten by the next run rather than half-committed.
    """

    def __init__(self, path: Path, flush_rows: int | None = None):
        self.path = path
        # Resolved from the module global in the body, not as a default
        # argument: a default binds once at def time, so `_META_FLUSH_ROWS`
        # could never be overridden afterwards -- which is exactly how
        # `test_meta_is_written_in_row_groups_not_one_frame` first read one row
        # group out of a 60-row corpus it had asked to flush every 5.
        self.flush_rows = max(1, _META_FLUSH_ROWS if flush_rows is None else flush_rows)
        self._pending: list[dict] = []
        self._writer: pq.ParquetWriter | None = None
        self._opened = False
        self.n_rows = 0
        self.split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    def add(self, row: dict, split: str) -> None:
        self._pending.append(row)
        self.n_rows += 1
        self.split_counts[split] += 1
        if len(self._pending) >= self.flush_rows:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, _META_SCHEMA)
            self._opened = True
        batch = pa.RecordBatch.from_pylist(self._pending, schema=_META_SCHEMA)
        self._writer.write_batch(batch)
        self._pending.clear()

    def close(self) -> None:
        """Flush and finalise.  Always leaves a readable file, even with 0 rows."""
        self._flush()
        if self._writer is None:
            # No row ever arrived, so no row group was opened.  An empty file
            # still has to exist and still has to carry META_COLUMNS -- callers
            # read it unconditionally, and `test_no_kept_games_empty_meta`
            # asserts the column set.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(_META_SCHEMA.empty_table(), self.path)
            return
        self._writer.close()
        self._writer = None

    def abort(self) -> None:
        """Drop buffered rows and remove this run's partial file.

        Closing the handle is not enough: `ParquetWriter.close` writes the
        footer, so a half-streamed file would come back as a perfectly *valid*
        parquet naming shards the failed run never flushed -- which training
        would then read without complaint.  The handle still has to be closed
        first to release the fd before the unlink.

        Only a file this writer opened is removed.  If the run failed before
        the first flush, whatever is on disk belongs to an earlier build and is
        not this one's to delete.
        """
        self._pending.clear()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._opened:
            self.path.unlink(missing_ok=True)
            self._opened = False


#: Per-ndarray cost of living in a Python dict rather than in a stacked array:
#: the ndarray header plus its entry in the owning dict.  Measured by building
#: 2000 real sample dicts out of a shard -- 11.4 KiB of array payload came to
#: 17.21 KiB resident across 48 arrays, i.e. ~124 B of overhead each.  112 is
#: the bare ndarray header; the rest is dict slot and allocator rounding.
_NDARRAY_OVERHEAD_BYTES = 124

#: Flat per-sample cost of the dict object itself, independent of key count.
_SAMPLE_DICT_BYTES = 400


def sample_nbytes(sample: dict) -> tuple[int, int]:
    """Return ``(resident_bytes, payload_bytes)`` for one featurized sample.

    *payload* is what `_write_shard` will stack into a contiguous array;
    *resident* is what the sample actually costs while it sits in a buffer as a
    dict of small ndarrays, which is ~1.5x more and is the number that decides
    how many of them fit in a budget.
    """
    payload = 0
    n_arrays = 0
    for v in sample.values():
        payload += getattr(v, "nbytes", 8)
        n_arrays += 1
    resident = payload + n_arrays * _NDARRAY_OVERHEAD_BYTES + _SAMPLE_DICT_BYTES
    return resident, payload


def samples_per_shard_for_budget(mem_budget_gb: float, sample: dict) -> int:
    """Largest ``samples_per_shard`` whose buffers fit in *mem_budget_gb*.

    The writer holds **three** buffers at once -- train, val and test all fill
    concurrently and each is only emptied when it alone reaches the limit, so
    the worst case really is three full ones -- and adds a transient copy of one
    buffer's payload while `_write_shard` stacks it.  Hence::

        budget = samples_per_shard * (3 * resident + payload)

    Sizing from a *real* sample rather than a constant is what keeps this honest
    across featurizer changes: a new tensor widens the per-sample cost and the
    shard shrinks to match, instead of the constant going quietly stale.
    """
    resident, payload = sample_nbytes(sample)
    per_sample = 3 * resident + payload
    return max(1, int(mem_budget_gb * (1024 ** 3)) // per_sample)


# ============================================================
# Public helpers
# ============================================================


def _is_kept_game(
    ep: dict,
    p: int,
    experts: set[str] | None,
    self_ids: set[int],
    opp_ids: set[int] | None,
    archetypes: list[Archetype],
    jaccard_thresh: float = 0.90,
) -> bool:
    """Return True if player *p*'s episode should be included in the corpus.

    Conditions (D.3.5):
      - team(p) in *experts*, when a set is given
      - arch(deck_of(ep, p)) in self_ids
      - arch(deck_of(ep, 1-p)) in opp_ids, when a set is given

    Two of the three are ``None`` on the production path, for different reasons.

    ``experts=None`` applies **no** team filter.  Skill enters as a per-row
    weight instead (``ptcg_mine.stats.skill_weight``, carried to training in
    ``meta.parquet``'s ``skill_w`` column), because the top-K filter discarded
    92.3% of the corpus to buy 4 points of represented win rate.

    ``opp_ids=None`` accepts **any opponent deck**, including one that clusters
    to no archetype at all.  This is the cheaper half of the deck filter to
    give up: of the belief targets, only ``bel_arch`` is defined in terms of
    𝒟_opp, and it already carries ``-1`` for an off-list opponent, which the
    loss ignores.  ``bel_deck``/``bel_hidden``/``bel_hand`` are read from the
    opponent's actual decklist and stay exact.  So an off-list row trains the
    policy and three of the four belief heads, and merely abstains on the
    fourth -- against dropping the row, which trains nothing and also removes
    the hard matchups a specialist most needs to see.

    ``self_ids`` has no such escape and is still enforced: a row whose own deck
    is not one being trained is a different game entirely.
    """
    t0, t1 = teams(ep)
    team = t0 if p == 0 else t1
    if experts is not None and team not in experts:
        return False
    try:
        self_arch = assign_archetype(deck_of(ep, p), archetypes, jaccard_thresh)
        opp_arch = assign_archetype(deck_of(ep, 1 - p), archetypes, jaccard_thresh)
    except (KeyError, IndexError, ValueError):
        return False
    if self_arch not in self_ids:
        return False
    return opp_ids is None or opp_arch in opp_ids


def _active_decisions(ep: dict, p: int) -> Iterator[tuple[int, dict, list[int]]]:
    """Yield ``(step_index, observation_dict, action_list)`` for every ACTIVE
    decision of player *p*, using the off-by-one pairing rule (A.3/D.4).

    Only steps i where ``steps[i][p]["status"] == "ACTIVE"`` are considered;
    the action is read from ``steps[i+1][p]["action"]``.
    """
    steps = ep["steps"]
    for i in range(len(steps) - 1):
        rec = steps[i][p]
        if rec.get("status") != "ACTIVE":
            continue
        obs = rec.get("observation")
        if obs is None:
            continue
        action = steps[i + 1][p].get("action", [])
        yield i, obs, action


def _split_of(episode_id: str | int) -> str:
    """Deterministic train/val/test split by episode hash.

    SHA-256(episode_id) % 100 → 0-9 = "val", 10-19 = "test", 20-99 = "train"
    (an 80/10/10 split).  Splitting on *episode_id* keeps every decision point
    of a game — and both players' trajectories — inside one split, so
    consecutive near-duplicate states cannot leak across the boundary.

    The ratio matters for model selection: at the previous 96/2/2 this yielded
    a val split of ~477 samples drawn from only 6 episodes, which is far too
    noisy to drive early stopping or best-checkpoint selection.

    Uses hashlib for cross-run reproducibility (Python's hash() is randomized).
    """
    h = int(hashlib.sha256(str(episode_id).encode()).hexdigest(), 16) % 100
    if h < 10:
        return "val"
    elif h < 20:
        return "test"
    else:
        return "train"


# ============================================================
# Private helpers
# ============================================================


def _load_archetypes_from_json(path: str | Path) -> tuple[list[Archetype], list[int], list[int]]:
    """Reconstruct Archetype objects + self_ids/opp_ids from archetypes.json."""
    with open(path) as f:
        data = json.load(f)
    archetypes = [
        Archetype(
            id=d["id"],
            representative=tuple(d["representative"]),
            frequency=d.get("frequency", 0),
        )
        for d in data["archetypes"]
    ]
    return archetypes, data["self_ids"], data["opp_ids"]


def _load_vocab(path: str | Path) -> dict:
    """Load the frozen vocab artifact, with int-keyed id maps.

    ``normalize_vocab`` is mandatory: JSON keys are strings but engine card /
    attack ids are ints, so an un-normalized vocab maps every card to UNKNOWN.
    """
    from ptcg_il.featurizer import normalize_vocab

    with open(path) as f:
        return normalize_vocab(json.load(f))


def _load_all_episodes(raw_dir: Path) -> tuple[list[tuple[str, dict]], int, int]:
    """Glob ``*.json`` under *raw_dir*, validate, return ``[(episode_id, ep), ...]``.

    Episode IDs are extracted from filenames (stem).

    Returns ``(valid_episodes, n_loaded, n_invalid)``.
    """
    episodes: list[tuple[str, dict]] = []
    n_loaded = 0
    n_invalid = 0
    for eid, ep in _iter_episodes(raw_dir, counts := {"n_loaded": 0, "n_invalid": 0}):
        episodes.append((eid, ep))
    n_loaded = counts["n_loaded"]
    n_invalid = counts["n_invalid"]
    return episodes, n_loaded, n_invalid


def _iter_kept_games(source, keep: dict[str, list[tuple[int, bool]]]):
    """Yield ``(eid, ep, p, won)`` for kept games, streaming episodes from
    *source* and dropping each one as soon as its samples are emitted.

    *source* is a zero-arg callable returning a fresh ``(eid, ep)`` iterator,
    so this is the second pass over the corpus; *keep* is the decision map
    built from the cheap first pass.

    *source* is expected to yield only episodes named in *keep* — the pass-B
    source is path-filtered up front, because parsing an episode in order to
    discard it here costs the same 6 MB decode as one that is used, and the
    corpus keeps well under 10% of its episodes.  Anything else is still
    skipped, so an unfiltered source stays correct (just slow).
    """
    for eid, ep in source(keep):
        plans = keep.get(eid)
        if not plans:
            continue
        for p, won in plans:
            yield eid, ep, p, won


def _episode_paths(raw_dir) -> list[Path]:
    """Sorted ``*.json`` paths under *raw_dir*. Sorted for run-to-run stability."""
    return sorted(Path(raw_dir).rglob("*.json"))


def _iter_episodes(raw_dir, counts: dict | None = None, paths=None):
    """Stream ``(episode_id, ep)`` for each valid, fully-parsed episode.

    The generator holds exactly one parsed episode at a time (~14.5 MB), so a
    caller that does not retain them is memory-flat regardless of corpus size.
    ``_load_all_episodes`` retains everything and is therefore only safe for
    small/test corpora — the real pipeline goes through ``build_shards``, which
    makes two passes instead.

    *paths*, if given, replaces the ``rglob`` of *raw_dir*; pass B uses it to
    open only the episodes it kept.  *counts*, if given, is updated in place
    with ``n_loaded`` / ``n_invalid``.
    """
    def _bump(key):
        if counts is not None:
            counts[key] = counts.get(key, 0) + 1

    for path in (_episode_paths(raw_dir) if paths is None else paths):
        try:
            ep = load_episode(path)
        except (json.JSONDecodeError, OSError):
            _bump("n_loaded")
            _bump("n_invalid")
            continue
        _bump("n_loaded")
        if validate_episode(ep):
            yield path.stem, ep
        else:
            _bump("n_invalid")
        del ep


def _scan_projections(paths: list[Path], counts: dict, jobs: int | None):
    """Pass A: yield ``(episode_id, projection)`` for every valid episode.

    Thin accounting wrapper over `episode.scan_projections`.  Phase 3 counts
    every file it opened as "loaded", including ones that would not parse.
    """
    for eid, proj, status in scan_projections(paths, jobs):
        counts["n_loaded"] = counts.get("n_loaded", 0) + 1
        if status is OK:
            yield eid, proj
        else:
            counts["n_invalid"] = counts.get("n_invalid", 0) + 1


# ============================================================
# Pass A: scan + selection, in the same worker
# ============================================================

#: Read-only selection context for `_scan_and_keep`, published per worker by a
#: pool initializer for the same reason `_FEAT_CTX` is.
_SCAN_CTX: dict | None = None


def _init_scan_ctx(ctx: dict) -> None:
    """Pool initializer: publish *ctx* for this worker's `_scan_and_keep`."""
    global _SCAN_CTX
    _SCAN_CTX = ctx


def _keep_plans(ep: dict, ctx: dict) -> tuple[tuple[int, bool], ...]:
    """``((player, won), ...)`` for the players of *ep* the corpus keeps."""
    plans = []
    for p in (0, 1):
        if _is_kept_game(ep, p, ctx["experts"], ctx["self_ids"], ctx["opp_ids"],
                         ctx["archetypes"], ctx["jaccard_thresh"]):
            plans.append((p, rewards(ep)[p] == 1))
    return tuple(plans)


def _scan_and_keep_ctx(path_str: str, ctx: dict) -> tuple[str, tuple | None, object, tuple]:
    """Parse, validate, **and** decide what to keep, in one place.

    Returns ``(episode_id, team_reward_tuple | None, status, plans)``.

    The selection used to run in the parent, over projections the workers had
    already produced and shipped back.  It is 300-cluster multiset Jaccard per
    player and measured 39.8 s for 3000 episodes -- ~12.3 min over the full
    corpus, single-threaded, which made it a larger block than the parallel
    featurize pass it feeds.  The worker already holds the projection, so doing
    it here costs nothing extra and parallelises for free.

    Only ``(t0, t1, r0, r1)`` comes back rather than the projection itself.
    That is all the parent still needs -- the leaderboard reads teams and
    rewards, and `_is_kept_game` has already run -- and it drops the parent's
    pass-A residency from ~36 KB/episode (~2.0 GiB at 55k episodes, measured) to
    a four-field tuple.
    """
    eid, proj, status = scan_episode(path_str)
    if status is not OK:
        return eid, None, status, ()
    t0, t1 = teams(proj)
    r0, r1 = rewards(proj)
    return eid, (t0, t1, r0, r1), status, _keep_plans(proj, ctx)


def _scan_and_keep(path_str: str) -> tuple[str, tuple | None, object, tuple]:
    """Pool entry point: `_scan_and_keep_ctx` against this worker's `_SCAN_CTX`.

    Split from the implementation so the **serial** path can pass its context
    as an argument instead of assigning the module global, which pass B's
    `_serial_featurized` already does for the same reason.  Under `fork` a
    global set in the parent is inherited by every later worker, so a serial run
    would silently supply the context that a pool run had failed to ship --
    hiding a missing `initializer=` until it reached a process that had never
    run serially.
    """
    return _scan_and_keep_ctx(path_str, _SCAN_CTX)


def _scan_with_keep(paths, jobs: int | None, ctx: dict):
    """Pass A, in task order: ``(eid, (t0, t1, r0, r1) | None, status, plans)``.

    Mirrors `episode.scan_projections` -- same fork rationale, same chunking,
    same order guarantee -- but carries the selection context the plain scan has
    no business knowing about.

    Unlike pass B, order here is **not** load-bearing: every consumer of this
    output is order-free (the leaderboard is a tally, `keep` is a dict, the rest
    are counters), and pass B re-imposes corpus order from `all_paths` anyway.
    It is preserved because `imap_unordered` buys nothing over a scan this cheap
    and a reproducible log is worth more.  What *is* load-bearing is that every
    worker gets `ctx` -- a worker running with `_SCAN_CTX = None` would fail to
    select rather than silently keeping the wrong ones, which
    `test_parallel_pass_a_selects_exactly_what_serial_does` pins.
    """
    paths = [str(p) for p in paths]
    n_jobs = (os.cpu_count() or 1) if jobs is None else max(1, jobs)

    if n_jobs <= 1 or len(paths) < _PARALLEL_SCAN_MIN_FILES:
        for path in paths:
            yield _scan_and_keep_ctx(path, ctx)
        return

    import multiprocessing as mp

    try:
        mp_ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover - non-fork platforms
        mp_ctx = mp.get_context()

    with mp_ctx.Pool(processes=n_jobs, initializer=_init_scan_ctx,
                     initargs=(ctx,)) as pool:
        yield from pool.imap(_scan_and_keep, paths, chunksize=_SCAN_CHUNKSIZE)


def _leaderboard_from_counts(counts: dict[str, list[int]]) -> dict[str, dict]:
    """Build `stats.team_leaderboard`'s output from streamed ``[games, wins]``.

    Same shape and same numbers -- `test_streamed_leaderboard_matches_stats`
    pins that against the real function -- but accumulated one episode at a time
    so pass A never has to hold the corpus to compute it.
    """
    return {
        team: {
            "games": games,
            "wins": wins,
            "win_rate": wins / games if games else 0.0,
        }
        for team, (games, wins) in counts.items()
    }


#: Keys stored as float16 — currently none.
#:
#: The ``*_card_feat`` keys used to live here and were ~90% of a shard's
#: uncompressed bytes; they are no longer stored at all (see _DERIVED_KEYS), so
#: there is nothing left that halving would meaningfully shrink.  ``opt_scalar``
#: is the largest remaining float tensor and deliberately stays fp32:
#: ``option_groups`` compares it at fp32 to decide ``opt_group``, and storing it
#: at a different precision than the grouping saw would split or merge options
#: the label then disagrees with.
#:
#: Index arrays must never be added here — see _INT32_KEYS for those.
_FP16_KEYS: frozenset[str] = frozenset()

#: Feature tensors the writer drops: each is a gather from a frozen static
#: table, reproduced by ``ShardDataset`` from an id that costs ~200x less to
#: store.  Dropping them here rather than in ``_write_shard`` also keeps them
#: out of the shard buffer, which holds ``samples_per_shard`` samples in RAM —
#: at 50k samples these keys alone were ~5.8 GB of the writer's own footprint.
_DERIVED_KEYS = frozenset(CARD_FEAT_SOURCES) | {"log_card_feat"}

#: Card/attack ids are small non-negative integers; int64 doubles them for no
#: reason.  int32 is still 6 orders of margin over the largest engine id.
_INT32_KEYS = frozenset(id_key for id_key, _ in CARD_FEAT_SOURCES.values())


def _write_shard(split: str, shard_idx: int, buffer: list[dict], out_dir: Path) -> Path:
    """Stack *buffer* samples and save as a compressed .npz shard."""
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    path = shards_dir / f"{split}-{shard_idx:05d}.npz"

    # Collect all keys from first sample
    if not buffer:
        return path
    keys = sorted(buffer[0].keys())
    stacked = {}
    for k in keys:
        arrays = [s[k] for s in buffer]
        stacked_k = np.stack(arrays, axis=0)
        if k in _FP16_KEYS:
            stacked_k = stacked_k.astype(np.float16)
        elif k in _INT32_KEYS:
            stacked_k = stacked_k.astype(np.int32)
        stacked[k] = stacked_k
    np.savez_compressed(path, **stacked)
    return path


def _parse_episode_id(raw: str | int) -> str:
    """Normalise episode-id to string for meta storage."""
    return str(raw)


# ============================================================
# Pass B: per-episode featurization
# ============================================================

#: Read-only context every pass-B worker needs.  A pool *initializer* publishes
#: it once per worker rather than a `functools.partial` shipping it per task:
#: at 54k tasks the latter would pickle the engine card/attack tables 54k times.
_FEAT_CTX: dict | None = None

#: Below this many kept episodes, pass B runs in-process.  Mirrors
#: `episode._PARALLEL_SCAN_MIN_FILES` — forking 16 workers to featurize a
#: handful of test episodes costs more than it saves.
_PARALLEL_FEATURIZE_MIN_EPISODES = 256

#: Episodes allowed in flight per worker.  `Pool.imap` applies **no**
#: backpressure — its feeder thread drains the whole task iterable and buffers
#: every result the consumer has not taken yet.  Pass B's workers are ~16x
#: faster in aggregate than the parent that stacks and compresses their output,
#: so an unbounded imap would queue the corpus's ~125 GB of samples (measured
#: 2.31 MB/episode x 54k) in the parent — a `systemd-oomd` kill, not an
#: MemoryError.  Two per worker keeps every worker fed with one spare while
#: capping the backlog at ~74 MB.
_FEATURIZE_INFLIGHT_PER_WORKER = 2


def _init_feat_ctx(ctx: dict) -> None:
    """Pool initializer: publish *ctx* for this worker's `_featurize_episode`."""
    global _FEAT_CTX
    _FEAT_CTX = ctx


def _featurize_player(eid: str, ep: dict, p: int, won: bool,
                      ctx: dict) -> list[tuple[dict, dict]]:
    """Featurize every decision point of one kept (episode, player) pair.

    Returns ``[(meta_row, sample), ...]`` in emission order.  The meta rows
    deliberately carry **no** ``shard``, ``row`` or ``skill_w``: the first two
    depend on how full the writer's buffers are and the third on the corpus-wide
    leaderboard, none of which a worker can see.  The parent fills them in as it
    appends.  Everything else here is a pure function of this one episode, which
    is what makes the pass parallelisable at all.
    """
    out: list[tuple[dict, dict]] = []
    archetypes_data = ctx["archetypes_data"]
    jaccard_thresh = ctx["jaccard_thresh"]

    r = rewards(ep)[p]
    value_target = 1.0 if r == 1 else -1.0
    team_name = teams(ep)[p]
    try:
        self_arch = assign_archetype(deck_of(ep, p), archetypes_data, jaccard_thresh)
        opp_arch = assign_archetype(deck_of(ep, 1 - p), archetypes_data, jaccard_thresh)
    except (KeyError, IndexError, ValueError):
        return out
    if self_arch is None:
        return out
    # `opp_arch is None` means the opponent's decklist matched no cluster at
    # jaccard >= thresh -- not that the episode is malformed.  Pass A already
    # decided to keep this row (filter_opp is off), so dropping it here would
    # silently reinstate the filter one pass later.  -1 is the same sentinel
    # an off-list-but-clustered opponent gets from opp_arch_to_contig.

    # Belief supervision, gathered once per (episode, player):
    #   - the opponent's exact decklist is their step-0 action, constant all game
    #   - their hand is read off their own ACTIVE steps
    try:
        opp_deck = deck_of(ep, 1 - p)
    except (KeyError, IndexError, ValueError):
        opp_deck = None
    opp_hands = opp_hand_timeline(ep, 1 - p) if opp_deck is not None else []
    opp_arch_contig = ctx["opp_arch_to_contig"].get(opp_arch, -1)

    for step_i, obs, action in _active_decisions(ep, p):
        # Deck-selection steps are skipped (featurizer raises ValueError)
        if obs.get("select") is None:
            continue

        try:
            sample = featurize(obs, ctx["vocab"], action, value_target=value_target,
                               sample_weight=1.0,
                               engine_card_features=ctx["engine_card_features"],
                               engine_attack_features=ctx["engine_attack_features"],
                               evolution_map=ctx["evolution_map"])
        except (ValueError, TypeError, KeyError) as exc:
            logger.debug("Skipping sample %s/%d/%d: %s", eid, p, step_i, exc)
            continue

        # Belief labels are always written, valid or not — see
        # empty_belief_labels() for why the key set has to be uniform.
        state = obs.get("current") or {}
        your_index = state.get("yourIndex")
        if opp_deck is not None and your_index is not None:
            sample.update(
                build_belief_labels(
                    state=state,
                    your_index=int(your_index),
                    opp_deck=opp_deck,
                    n_all_cards=ctx["n_all_cards"],
                    opp_arch_index=opp_arch_contig,
                    opp_hand_ids=hand_after(opp_hands, step_i),
                )
            )
        else:
            sample.update(empty_belief_labels())

        # sample_weight is NOT written into shards (spec D.4);
        # it is derived from meta.parquet columns at training time (C.3).
        sample.pop("sample_weight", None)

        # Nor are the *_card_feat tensors: ShardDataset re-gathers them
        # from the ids that stay behind.  featurize still computes them —
        # option_groups needs opt_card_feat/opt_attack_feat to decide
        # opt_group, which *is* stored.
        for key in _DERIVED_KEYS:
            sample.pop(key, None)

        out.append((
            {
                "sample_uid": f"{eid}_{p}_{step_i}",
                "episode_id": _parse_episode_id(eid),
                "player": p,
                "team": team_name,
                "archetype_self": self_arch,
                # The *global* cluster id, which is deliberately not the same
                # thing as the shard's `bel_arch`: an opponent outside 𝒟_opp
                # keeps its real id here (the information is free to store and
                # useful for slicing eval) while `bel_arch` holds -1 because
                # the head has no class for it.  -1 here means only "matched no
                # cluster at all", and keeps the column integer-typed.
                "archetype_opp": -1 if opp_arch is None else opp_arch,
                "sel_type": int(sample["sel_type"]),
                "sel_ctx": int(sample["sel_ctx"]),
                "minCount": int(sample["minCount"]),
                "maxCount": int(sample["maxCount"]),
                "won": won,
            },
            sample,
        ))

    return out


def _featurize_episode(task: tuple[str, list[tuple[int, bool]]]) -> tuple[str, list[tuple[dict, dict]]]:
    """Pool worker: featurize every kept player of the episode at *task*'s path.

    Returns ``(episode_id, rows)``.  An episode that will not parse or validate
    yields an empty list rather than raising — that is exactly what
    `_iter_episodes` does on the serial path, and pass A has already counted it
    as loaded/invalid, so raising here would change the summary as well as kill
    the pool.
    """
    path_str, plans = task
    path = Path(path_str)
    try:
        ep = load_episode(path)
    except (json.JSONDecodeError, OSError):
        return path.stem, []
    if not validate_episode(ep):
        return path.stem, []

    rows: list[tuple[dict, dict]] = []
    for p, won in plans:
        rows.extend(_featurize_player(path.stem, ep, p, won, _FEAT_CTX))
    return path.stem, rows


def _serial_featurized(source, keep, ctx):
    """In-process pass B, yielding the same ``(eid, rows)`` as `_imap_featurized`.

    Yields once per kept *player* rather than per episode; the consumer keys
    only on ``eid`` (via `_split_of`), so the two groupings are equivalent.
    """
    for eid, ep, p, won in _iter_kept_games(source, keep):
        yield eid, _featurize_player(eid, ep, p, won, ctx)


def _imap_featurized(tasks: list, n_jobs: int, feat_ctx: dict):
    """Parallel pass B: ``(eid, rows)`` for each task, **in task order**.

    Order-preserving like `scan_projections`, so shard row order, shard
    boundaries and meta.parquet are byte-for-byte what the serial path produces
    whatever *n_jobs* is.

    The semaphore is the backpressure `Pool.imap` does not have: it is acquired
    as the feeder thread pulls a task and released as the consumer takes a
    result, so at most `_FEATURIZE_INFLIGHT_PER_WORKER * n_jobs` episodes' worth
    of samples exist outside the parent's shard buffers.  ``chunksize=1``
    because a task is ~90 ms — three orders of magnitude above per-task IPC
    overhead — and a bigger chunk would only coarsen that window.
    """
    import multiprocessing as mp
    import threading

    # fork for the same reason pass A does: workers only read files and numpy
    # arrays, and this runs before any engine handle or CUDA context exists.
    try:
        mp_ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover - non-fork platforms
        mp_ctx = mp.get_context()

    sem = threading.Semaphore(_FEATURIZE_INFLIGHT_PER_WORKER * n_jobs)
    aborted = threading.Event()

    def gated():
        # Polled rather than a bare `acquire()`: the blocked thread here is the
        # pool's own task handler, and `Pool.terminate()` *joins* it.  A plain
        # acquire would therefore turn any early exit — an exception in the
        # consumer, a `break`, GeneratorExit from an abandoned generator — into
        # a permanent hang instead of the error the caller was raising.
        for task in tasks:
            while not sem.acquire(timeout=0.1):
                if aborted.is_set():
                    return
            yield task

    pool = mp_ctx.Pool(processes=n_jobs, initializer=_init_feat_ctx,
                       initargs=(feat_ctx,))
    try:
        for result in pool.imap(_featurize_episode, gated(), chunksize=1):
            sem.release()
            yield result
        # Every task was consumed, so `gated()` has already returned and the
        # workers are idle: let them exit on their own.
        pool.close()
    except BaseException:
        # Includes GeneratorExit, which is how an abandoned generator arrives.
        # `aborted` must be set *before* terminate(), which joins the task
        # handler that may be parked in the semaphore above.
        aborted.set()
        pool.terminate()
        raise
    finally:
        aborted.set()
        pool.join()


# ============================================================
# Main orchestrator
# ============================================================


def build_shards(
    config: MineConfig | None = None,
    *,
    episodes: list[tuple[str, dict]] | None = None,
    vocab: dict | None = None,
    archetypes_data: list[Archetype] | None = None,
    self_ids: list[int] | None = None,
    opp_ids: list[int] | None = None,
    experts: set[str] | list[str] | None = None,
    filter_opp: bool = False,
    samples_per_shard: int | None = None,
    mem_budget_gb: float = DEFAULT_MEM_BUDGET_GB,
    jobs: int | None = None,
) -> dict:
    """Build shards and meta.parquet from the corpus.

    **Production path** — reads everything from files::

        build_shards(config)

    **Testing / programmatic path** — pass pre-loaded data::

        build_shards(episodes=..., vocab=..., archetypes_data=...,
                     self_ids=..., opp_ids=..., experts=...)

    Parameters
    ----------
    config : MineConfig | None
        If given, ``raw_dir``, ``out_dir`` and ``jaccard_thresh`` are read from
        it.  ``data/vocab.json`` and ``data/archetypes.json`` are loaded from
        ``out_dir``.  ``k_experts``/``g_min`` are **not** read: they select the
        experts that ``ptcg_mine`` uses to rank 𝒟_self/𝒟_opp, and no longer
        decide which rows are written here.
    episodes : list
        Pre-loaded ``[(episode_id, episode_dict), ...]``.  Used when *config*
        is None.
    vocab : dict
        Pre-loaded vocab dict.
    archetypes_data : list[Archetype]
        Pre-reconstructed Archetype objects.
    self_ids / opp_ids : list[int]
        Pre-loaded 𝒟_self / 𝒟_opp archetype id lists.
    experts : set[str] | list[str] | None
        Restrict the corpus to these teams.  **None (the default) keeps every
        team** and lets ``meta.parquet``'s ``skill_w`` column carry skill as a
        training weight; pass a set only to reproduce the old filtered corpus.
    filter_opp : bool
        Require the opponent's deck to be in *opp_ids*.  **False by default** —
        an off-list opponent is kept, with ``archetype_opp = -1`` in the meta
        and ``bel_arch = -1`` (ignored by that head) in the shard.  The other
        belief targets are read from the opponent's real decklist and are
        unaffected.  True reproduces the old corpus, which dropped ~53% of
        (episode, player) pairs on this condition.
    samples_per_shard : int | None
        Max samples per shard file.  **None (the default) derives it from
        *mem_budget_gb* and the measured size of the first real sample**, and
        logs what it picked.  Pass an int to pin the shard size and ignore the
        budget entirely.
    mem_budget_gb : float
        Ceiling on the writer's resident memory, in GiB (default 2.0).  Only
        consulted when *samples_per_shard* is None.  The shard buffers are what
        it sizes: they are the writer's largest term and the only one a caller
        can trade against shard count.  meta.parquet is streamed in row groups
        and costs ~40 MB whatever this is set to.
    jobs : int | None
        Worker processes for **both** corpus passes — the pass-A head scan and
        the pass-B featurize.  ``None`` uses ``os.cpu_count()``; ``1`` runs both
        in-process.  Ignored when *episodes* is supplied, since then there is
        nothing to read from disk and the corpus is already parsed in this
        process.  Both pools are order-preserving, so this never changes the
        output — verified array-by-array against the serial path.

    Returns
    -------
    dict
        Summary: ``{"split_counts": {"train": N, "val": N, "test": N},
        "n_shards": N, "total_samples": N, "n_kept_games": N,
        "meta_path": str}``.
    """
    # --- Resolve inputs ---
    if config is not None:
        raw_dir = Path(config.raw_dir)
        out_dir = Path(config.out_dir)
        jaccard_thresh = config.jaccard_thresh
    else:
        raw_dir = Path("raw")
        out_dir = Path("data")
        jaccard_thresh = 0.90

    if episodes is None and config is None:
        raise ValueError("build_shards: either config or episodes must be provided")

    # Archetypes are resolved *before* pass A because pass A now applies the
    # selection filter itself, and `_is_kept_game` needs them.  Vocab and the
    # engine tables stay below the empty-corpus check: only pass B reads them,
    # and loading them here would make an empty corpus demand artifacts it has
    # no use for.
    if archetypes_data is None or self_ids is None or opp_ids is None:
        arch_path = out_dir / "archetypes.json"
        if not arch_path.exists():
            raise FileNotFoundError(f"Archetypes not found at {arch_path}")
        archetypes_data, self_ids, opp_ids = _load_archetypes_from_json(arch_path)

    self_id_set = set(self_ids)
    # None = accept any opponent deck.  `opp_ids` still defines the belief
    # head's class order below; it just no longer decides which rows exist.
    opp_id_set = set(opp_ids) if filter_opp else None

    # Skill enters as a weight, not a filter.  `experts_set` stays None unless a
    # caller pins it, so no (episode, player) pair is excluded for who played it;
    # the leaderboard below still covers every team, including the ones a top-K
    # filter would have dropped, because each one needs a weight.
    experts_set = set(experts) if experts is not None else None

    # Archetype ids in archetypes.json are global cluster indices (0..157 on the
    # current corpus) but only a handful are retained as 𝒟_opp.  The belief head
    # classifies over the retained set, so map global id -> contiguous index
    # here; unmapped opponents get -1, which the loss ignores.
    opp_arch_to_contig = {gid: i for i, gid in enumerate(opp_ids)}

    # --- Pass A: scan, select, and tally the leaderboard in one sweep ---
    #
    # Two streaming passes over disk rather than one resident corpus: pass A
    # keeps nothing per episode beyond its teams, rewards and keep-decision, and
    # pass B (the featurize loop below) re-reads each kept episode to get its
    # full game log.  Holding whole episodes costs ~14.5 MB each, i.e. ~145 GB
    # at the 10k-episode target; holding even the *projections* measured ~36 KB
    # each, ~2.0 GiB over the 55k-episode corpus, which is why nothing but the
    # tuple survives the worker now.
    #
    # `keep` records only the decision (episode id -> [(player, won), ...]).
    keep: dict[str, list[tuple[int, bool]]] = {}
    team_counts: dict[str, list[int]] = {}
    n_valid = 0
    n_kept = 0

    def _tally(t0, t1, r0, r1):
        for team, r in ((t0, r0), (t1, r1)):
            c = team_counts.setdefault(team, [0, 0])
            c[0] += 1
            if r == 1:
                c[1] += 1

    if episodes is None:
        n_loaded = 0
        n_invalid = 0
        all_paths = _episode_paths(raw_dir)
        logger.info("Pass A: scanning + selecting %d episodes across %s workers",
                    len(all_paths),
                    (os.cpu_count() or 1) if jobs is None else max(1, jobs))
        scan_ctx = {
            "experts": experts_set,
            "self_ids": self_id_set,
            "opp_ids": opp_id_set,
            "archetypes": archetypes_data,
            "jaccard_thresh": jaccard_thresh,
        }
        for eid, tr, status, plans in _scan_with_keep(all_paths, jobs, scan_ctx):
            n_loaded += 1
            if status is not OK:
                n_invalid += 1
                continue
            n_valid += 1
            _tally(*tr)
            if plans:
                keep[eid] = list(plans)
                n_kept += len(plans)

        # Pass B opens only the episodes pass A kept.  Filtering *paths* (rather
        # than parsing and discarding) is the whole saving; keeping `all_paths`
        # order means the shard row order is unchanged by this optimisation.
        def _episode_source(_keep=None):
            wanted = all_paths if _keep is None else [p for p in all_paths if p.stem in _keep]
            return _iter_episodes(raw_dir, paths=wanted)
    else:
        # Pre-loaded episodes are already parsed dicts in this process, so there
        # is nothing to stream and no pool to run: select in place.
        ep_list = list(episodes)
        n_loaded = len(ep_list)
        n_invalid = 0
        n_valid = len(ep_list)
        for eid, ep in ep_list:
            _tally(*teams(ep), *rewards(ep))
            plans = _keep_plans(ep, {
                "experts": experts_set,
                "self_ids": self_id_set,
                "opp_ids": opp_id_set,
                "archetypes": archetypes_data,
                "jaccard_thresh": jaccard_thresh,
            })
            if plans:
                keep[eid] = list(plans)
                n_kept += len(plans)

        def _episode_source(_keep=None, _eps=ep_list):
            return iter(_eps)

    if not n_valid:
        logger.warning("No valid episodes found; nothing to featurize.")
        meta_path = out_dir / "meta.parquet"
        _MetaWriter(meta_path).close()
        return {
            "split_counts": {"train": 0, "val": 0, "test": 0},
            "n_shards": 0,
            "total_samples": 0,
            "n_kept_games": 0,
            "n_loaded": n_loaded,
            "n_invalid": n_invalid,
            "meta_path": str(meta_path),
        }

    # Load vocab
    if vocab is None:
        vocab_path = out_dir / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab not found at {vocab_path}")
        vocab = _load_vocab(vocab_path)

    # Load the static tables the pure-feature featurizer consumes.  The
    # evolution map is what makes hand_feat[3] (`can_evolve`) non-zero; it was
    # written to disk by mining but never loaded by anybody, so that column was
    # constant 0 across every shard ever built.
    engine_tables = load_engine_tables(out_dir)
    engine_card_features = engine_tables["engine_card_features"]
    engine_attack_features = engine_tables["engine_attack_features"]
    evolution_map = engine_tables["evolution_map"]
    _n_all_cards = max(engine_card_features.keys()) + 1 if engine_card_features else 0

    leaderboard = _leaderboard_from_counts(team_counts)
    skill_w_by_team = team_skill_weights(leaderboard)

    logger.info(
        "build_shards: %d valid episodes, %d teams (skill-weighted%s), "
        "%d self archs, %d opp archs",
        n_valid,
        len(leaderboard),
        "" if experts_set is None else f", filtered to {len(experts_set)} experts",
        len(self_ids),
        len(opp_ids),
    )
    logger.info("Kept %d (episode, player) pairs out of %d", n_kept, n_valid * 2)

    if not n_kept:
        logger.warning("No kept games; writing empty meta.parquet only.")
        meta_path = out_dir / "meta.parquet"
        _MetaWriter(meta_path).close()
        return {
            "split_counts": {"train": 0, "val": 0, "test": 0},
            "n_shards": 0,
            "total_samples": 0,
            "n_kept_games": 0,
            "n_loaded": n_loaded,
            "n_invalid": n_invalid,
            "meta_path": str(meta_path),
        }

    # --- Featurize ---
    shard_counters: dict[str, int] = defaultdict(int)
    buffers: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    meta_path = out_dir / "meta.parquet"
    meta = _MetaWriter(meta_path)

    # Pass B: re-read each kept episode's full game log, one at a time.
    #
    # Pass B is ~98% of build-shards' wall clock (measured 0.092 s/episode over
    # 54k episodes = ~85 min, against ~1.2 min for the already-parallel pass A),
    # and per-episode it is 57% orjson decode / 31% featurize / 7% belief labels
    # — all CPU, all independent per episode.  Workers therefore do the parsing
    # and featurizing; the parent keeps the buffer bookkeeping, because `shard`,
    # `row` and `skill_w` are the only three fields that are not a pure function
    # of one episode, and keeping them here is what makes the output identical
    # to the serial path's rather than merely equivalent.
    feat_ctx = {
        "vocab": vocab,
        "engine_card_features": engine_card_features,
        "engine_attack_features": engine_attack_features,
        "evolution_map": evolution_map,
        "archetypes_data": archetypes_data,
        "jaccard_thresh": jaccard_thresh,
        "opp_arch_to_contig": opp_arch_to_contig,
        "n_all_cards": _n_all_cards,
    }
    n_jobs = (os.cpu_count() or 1) if jobs is None else max(1, jobs)

    # Pre-loaded `episodes` never take the pool: they are already parsed dicts
    # in this process, so a worker would have to be *sent* the corpus it was
    # supposed to avoid holding.
    if episodes is None and n_jobs > 1 and len(keep) >= _PARALLEL_FEATURIZE_MIN_EPISODES:
        tasks = [(str(p), keep[p.stem]) for p in all_paths if p.stem in keep]
        logger.info("Pass B: featurizing %d episodes across %d workers", len(tasks), n_jobs)
        featurized = _imap_featurized(tasks, n_jobs, feat_ctx)
    else:
        logger.info("Pass B: featurizing %d episodes in-process", len(keep))
        featurized = _serial_featurized(_episode_source, keep, feat_ctx)

    try:
        for eid, rows in featurized:
            split = _split_of(eid)
            for meta_row, sample in rows:
                # Sized off the first sample the corpus actually produced, not a
                # constant: a featurizer change that widens a tensor shrinks the
                # shard to match, instead of quietly blowing the budget.  Fixed
                # for the whole run once set, because `row` indexes into it.
                if samples_per_shard is None:
                    samples_per_shard = samples_per_shard_for_budget(
                        mem_budget_gb, sample)
                    resident, _ = sample_nbytes(sample)
                    logger.info(
                        "Buffers capped at %d samples/shard "
                        "(%.1f GiB budget, %.1f KiB/sample)",
                        samples_per_shard, mem_budget_gb, resident / 1024,
                    )

                buf = buffers[split]
                meta_row["shard"] = f"{split}-{shard_counters[split]:05d}.npz"
                meta_row["row"] = len(buf) % samples_per_shard
                # Indexed, not `.get(..., 0.0)`: the leaderboard is built from
                # these same episodes, so a miss is a bug, and a defaulted 0.0
                # would silently drop the row's gradient instead of raising.
                meta_row["skill_w"] = float(skill_w_by_team[meta_row["team"]])
                meta.add(meta_row, split)

                buf.append(sample)

                # Flush when buffer fills
                if len(buf) >= samples_per_shard:
                    _write_shard(split, shard_counters[split], buf, out_dir)
                    shard_counters[split] += 1
                    buf.clear()

        # Flush remaining
        for split in ("train", "val", "test"):
            if buffers[split]:
                _write_shard(split, shard_counters[split], buffers[split], out_dir)
                shard_counters[split] += 1
                buffers[split].clear()

        meta.close()
    except BaseException:
        # A half-written meta.parquet that still parses is worse than none: it
        # would name shards that a failed run may never have flushed.  Drop it
        # and let the (unstamped) rerun rewrite the file from scratch.
        meta.abort()
        raise

    split_counts = dict(meta.split_counts)

    logger.info(
        "build_shards done: %d total samples, %d shards, meta at %s",
        meta.n_rows,
        sum(shard_counters.values()),
        meta_path,
    )

    return {
        "split_counts": split_counts,
        "n_shards": sum(shard_counters.values()),
        "total_samples": meta.n_rows,
        "n_kept_games": n_kept,
        "n_loaded": n_loaded,
        "n_invalid": n_invalid,
        "meta_path": str(meta_path),
    }
