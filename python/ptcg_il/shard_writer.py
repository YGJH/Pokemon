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
from rich.logging import RichHandler
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from ptcg_il.belief_labels import (
    build_belief_labels,
    empty_belief_labels,
    hand_after,
    opp_hand_timeline,
)
from ptcg_il.featurizer import featurize, normalize_vocab
from ptcg_mine.archetype import Archetype, assign_archetype
from ptcg_mine.config import MineConfig
from ptcg_mine.episode import (
    deck_of,
    load_episode,
    project_for_selection,
    rewards,
    teams,
    validate_episode,
)
from ptcg_mine.stats import select_experts, team_leaderboard
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

SAMPLES_PER_SHARD = 50000

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
]

# ============================================================
# Public helpers
# ============================================================


def _is_kept_game(
    ep: dict,
    p: int,
    experts: set[str],
    self_ids: set[int],
    opp_ids: set[int],
    archetypes: list[Archetype],
    jaccard_thresh: float = 0.90,
) -> bool:
    """Return True if player *p*'s episode should be included in the corpus.

    Conditions (D.3.5):
      - team(p) in EXPERTS
      - arch(deck_of(ep, p)) in self_ids
      - arch(deck_of(ep, 1-p)) in opp_ids
    """
    t0, t1 = teams(ep)
    team = t0 if p == 0 else t1
    if team not in experts:
        return False
    try:
        self_arch = assign_archetype(deck_of(ep, p), archetypes, jaccard_thresh)
        opp_arch = assign_archetype(deck_of(ep, 1 - p), archetypes, jaccard_thresh)
    except (KeyError, IndexError, ValueError):
        return False
    return self_arch in self_ids and opp_arch in opp_ids


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
    """
    for eid, ep in source():
        plans = keep.get(eid)
        if not plans:
            continue
        for p, won in plans:
            yield eid, ep, p, won


def _iter_episodes(raw_dir, counts: dict | None = None):
    """Stream ``(episode_id, ep)`` for each valid episode under *raw_dir*.

    The generator holds exactly one parsed episode at a time (~14.5 MB), so a
    caller that does not retain them is memory-flat regardless of corpus size.
    ``_load_all_episodes`` retains everything and is therefore only safe for
    small/test corpora — the real pipeline goes through ``build_shards``, which
    makes two streaming passes instead.

    *counts*, if given, is updated in place with ``n_loaded`` / ``n_invalid``.
    """
    def _bump(key):
        if counts is not None:
            counts[key] = counts.get(key, 0) + 1

    for path in sorted(Path(raw_dir).rglob("*.json")):
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
        stacked[k] = np.stack(arrays, axis=0)
    np.savez_compressed(path, **stacked)
    return path


def _parse_episode_id(raw: str | int) -> str:
    """Normalise episode-id to string for meta storage."""
    return str(raw)


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
    samples_per_shard: int = SAMPLES_PER_SHARD,
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
        If given, ``raw_dir``, ``out_dir``, ``k_experts``, ``g_min``,
        ``jaccard_thresh`` are read from it.  ``data/vocab.json`` and
        ``data/archetypes.json`` are loaded from ``out_dir``.
    episodes : list
        Pre-loaded ``[(episode_id, episode_dict), ...]``.  Used when *config*
        is None.
    vocab : dict
        Pre-loaded vocab dict.
    archetypes_data : list[Archetype]
        Pre-reconstructed Archetype objects.
    self_ids / opp_ids : list[int]
        Pre-loaded 𝒟_self / 𝒟_opp archetype id lists.
    experts : set[str] | list[str]
        Expert team names.  If None and *config* is given, experts are
        recomputed from loaded episodes.
    samples_per_shard : int
        Max samples per shard file (default 50 000).

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
        k_experts = config.k_experts
        g_min = config.g_min
    else:
        raw_dir = Path("raw")
        out_dir = Path("data")
        jaccard_thresh = 0.90
        k_experts = 10
        g_min = 50

    # Load episodes if not provided.
    #
    # Two streaming passes over disk rather than one resident corpus: pass A
    # (here) keeps only ~1 KB per episode -- enough to pick experts and decide
    # which (episode, player) pairs to keep -- and pass B (the featurize loop
    # below) re-reads each kept episode to get its full game log. Holding whole
    # episodes costs ~14.5 MB each, i.e. ~145 GB at the 10k-episode target.
    if episodes is None:
        if config is None:
            raise ValueError("build_shards: either config or episodes must be provided")
        counts: dict = {"n_loaded": 0, "n_invalid": 0}
        ep_list = [(eid, project_for_selection(ep))
                   for eid, ep in _iter_episodes(raw_dir, counts)]
        n_loaded = counts["n_loaded"]
        n_invalid = counts["n_invalid"]

        def _episode_source():
            return _iter_episodes(raw_dir)
    else:
        ep_list = list(episodes)
        n_loaded = len(ep_list)
        n_invalid = 0

        # Bound as a default so the later `del ep_list` cannot break it.
        def _episode_source(_eps=ep_list):
            return iter(_eps)

    if not ep_list:
        logger.warning("No valid episodes found; nothing to featurize.")
        return {
            "split_counts": {"train": 0, "val": 0, "test": 0},
            "n_shards": 0,
            "total_samples": 0,
            "n_kept_games": 0,
            "n_loaded": n_loaded,
            "n_invalid": n_invalid,
            "meta_path": str(out_dir / "meta.parquet"),
        }

    # Load vocab
    if vocab is None:
        vocab_path = out_dir / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab not found at {vocab_path}")
        vocab = _load_vocab(vocab_path)

    # Load archetypes
    if archetypes_data is None or self_ids is None or opp_ids is None:
        arch_path = out_dir / "archetypes.json"
        if not arch_path.exists():
            raise FileNotFoundError(f"Archetypes not found at {arch_path}")
        archetypes_data, self_ids, opp_ids = _load_archetypes_from_json(arch_path)

    self_id_set = set(self_ids)
    opp_id_set = set(opp_ids)

    # Archetype ids in archetypes.json are global cluster indices (0..157 on the
    # current corpus) but only a handful are retained as 𝒟_opp.  The belief head
    # classifies over the retained set, so map global id -> contiguous index
    # here; unmapped opponents get -1, which the loss ignores.
    opp_arch_to_contig = {gid: i for i, gid in enumerate(opp_ids)}

    # Compute experts if not provided
    if experts is None:
        raw_eps = [ep for _, ep in ep_list]
        leaderboard = team_leaderboard(raw_eps)
        experts_list = select_experts(leaderboard, k_experts, g_min)
        experts_set = set(experts_list)
    else:
        experts_set = set(experts)

    logger.info(
        "build_shards: %d valid episodes, %d experts, %d self archs, %d opp archs",
        len(ep_list),
        len(experts_set),
        len(self_ids),
        len(opp_ids),
    )

    # --- Filter kept games ---
    # `keep` records only the decision (episode id -> [(player, won), ...]).
    # The episode bodies are re-read from disk in the featurize pass, so this
    # filter never pins the corpus in memory.
    keep: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    n_kept = 0
    for eid, ep in ep_list:
        for p in (0, 1):
            if _is_kept_game(ep, p, experts_set, self_id_set, opp_id_set, archetypes_data, jaccard_thresh):
                r = rewards(ep)[p]
                won = r == 1
                keep[eid].append((p, won))
                n_kept += 1

    logger.info("Kept %d (episode, player) pairs out of %d", n_kept, len(ep_list) * 2)
    del ep_list

    if not n_kept:
        logger.warning("No kept games; writing empty meta.parquet only.")
        meta_path = out_dir / "meta.parquet"
        pd.DataFrame(columns=META_COLUMNS).to_parquet(meta_path, index=False)
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
    meta_rows: list[dict] = []

    # Pass B: re-read each kept episode's full game log, one at a time.
    for eid, ep, p, won in _iter_kept_games(_episode_source, keep):
        r = rewards(ep)[p]
        value_target = 1.0 if r == 1 else -1.0
        team_name = teams(ep)[p]
        try:
            self_arch = assign_archetype(deck_of(ep, p), archetypes_data, jaccard_thresh)
            opp_arch = assign_archetype(deck_of(ep, 1 - p), archetypes_data, jaccard_thresh)
        except (KeyError, IndexError, ValueError):
            continue
        if self_arch is None or opp_arch is None:
            continue

        split = _split_of(eid)

        # Belief supervision, gathered once per (episode, player):
        #   - the opponent's exact decklist is their step-0 action, constant all game
        #   - their hand is read off their own ACTIVE steps
        try:
            opp_deck = deck_of(ep, 1 - p)
        except (KeyError, IndexError, ValueError):
            opp_deck = None
        opp_hands = opp_hand_timeline(ep, 1 - p) if opp_deck is not None else []
        opp_arch_contig = opp_arch_to_contig.get(opp_arch, -1)

        for step_i, obs, action in _active_decisions(ep, p):
            # Deck-selection steps are skipped (featurizer raises ValueError)
            if obs.get("select") is None:
                continue

            try:
                sample = featurize(obs, vocab, action, value_target=value_target, sample_weight=1.0)
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
                        id_to_index=vocab["id_to_index"],
                        vocab_size=vocab["size"],
                        opp_arch_index=opp_arch_contig,
                        opp_hand_ids=hand_after(opp_hands, step_i),
                    )
                )
            else:
                sample.update(empty_belief_labels())

            # sample_weight is NOT written into shards (spec D.4);
            # it is derived from meta.parquet columns at training time (C.3).
            sample.pop("sample_weight", None)

            buf = buffers[split]
            row_in_shard = len(buf) % samples_per_shard

            sample_uid = f"{eid}_{p}_{step_i}"
            meta_rows.append({
                "sample_uid": sample_uid,
                "shard": f"{split}-{shard_counters[split]:05d}.npz",
                "row": row_in_shard,
                "episode_id": _parse_episode_id(eid),
                "player": p,
                "team": team_name,
                "archetype_self": self_arch,
                "archetype_opp": opp_arch,
                "sel_type": int(sample["sel_type"]),
                "sel_ctx": int(sample["sel_ctx"]),
                "minCount": int(sample["minCount"]),
                "maxCount": int(sample["maxCount"]),
                "won": won,
            })

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

    # --- Write meta.parquet ---
    meta_df = pd.DataFrame(meta_rows, columns=META_COLUMNS)
    meta_df["episode_id"] = meta_df["episode_id"].astype(str)
    meta_path = out_dir / "meta.parquet"
    meta_df.to_parquet(meta_path, index=False)

    split_counts = {s: int((meta_df["shard"].str.startswith(s)).sum()) for s in ("train", "val", "test")}

    logger.info(
        "build_shards done: %d total samples, %d shards, meta at %s",
        len(meta_df),
        sum(shard_counters.values()),
        meta_path,
    )

    return {
        "split_counts": split_counts,
        "n_shards": sum(shard_counters.values()),
        "total_samples": len(meta_df),
        "n_kept_games": n_kept,
        "n_loaded": n_loaded,
        "n_invalid": n_invalid,
        "meta_path": str(meta_path),
    }
