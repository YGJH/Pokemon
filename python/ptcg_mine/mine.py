"""CLI orchestrator for Phases 0-2 of the corpus-mining pipeline.

Phase 0 (sampling) + Phase 1 (download) are handled by `download_corpus`
(skippable via --skip-download for offline/dev runs against an already-
populated raw_dir). Phase 2 (stats -> archetypes -> vocab -> static tables
-> artifacts) always runs over the parsed episodes found under `raw_dir`.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ptcg_mine import stamp
from ptcg_mine.archetype import canon, cluster_decks, pick_fixed_deck, select_self_opp
from ptcg_mine.artifacts import write_archetypes_json, write_mining_report, write_vocab_json
from ptcg_mine.cards import (build_engine_attack_features,
                              build_engine_card_features,
                              build_evolution_map,
                              build_static_tables, load_engine)
from ptcg_mine.config import MineConfig
from ptcg_mine.episode import (
    OK,
    UNREADABLE,
    deck_of,
    load_episode,
    project_for_selection,
    scan_projections,
    validate_episode,
)
from ptcg_mine.progress import ProgressReporter
from ptcg_mine.stats import select_experts, team_leaderboard
from ptcg_mine.vocab import build_vocab


logger = logging.getLogger(__name__)


class InsufficientDataError(RuntimeError):
    """Raised when the parsed corpus is too thin to select any experts
    (i.e. no episodes / no teams at all). Reported cleanly by main()."""

def build_parser() -> argparse.ArgumentParser:
    d = MineConfig()
    p = argparse.ArgumentParser(
        prog="ptcg_mine.mine",
        description="Mine the Kaggle Pokemon TCG episode corpus: sample/download "
        "episodes, select experts and deck archetypes, build the card vocab and "
        "engine-derived static feature tables, and write the frozen artifacts.",
    )
    p.add_argument("--days", default=None, help="Comma-separated day list (default: auto-select recent)")
    p.add_argument("--n-days", type=int, default=d.n_days)
    p.add_argument("--target-episodes", type=int, default=d.target_episodes)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--k-experts", type=int, default=d.k_experts)
    p.add_argument("--g-min", type=int, default=d.g_min)
    p.add_argument("--jaccard-thresh", type=float, default=d.jaccard_thresh)
    p.add_argument("--n-self", type=int, default=d.n_self)
    p.add_argument("--n-opp", type=int, default=d.n_opp)
    p.add_argument("--w-lost", type=float, default=d.w_lost)
    p.add_argument("--vocab-mode", default=d.vocab_mode)
    p.add_argument("--n-vocab", type=int, default=d.n_vocab)
    p.add_argument("--h-max", type=int, default=d.h_max)
    p.add_argument("--o-max", type=int, default=d.o_max)
    p.add_argument("--d-max", type=int, default=d.d_max)
    p.add_argument("--jobs", "-j", type=int, default=None,
                   help="Worker processes for the Phase 2 episode scan "
                        "(default: os.cpu_count(); 1 disables the pool). "
                        "Order-preserving, so this cannot change the output.")
    p.add_argument("--raw-dir", default=str(d.raw_dir))
    p.add_argument("--out-dir", default=str(d.out_dir))
    p.add_argument("--manifest-csv", default=str(d.manifest_csv))
    p.add_argument("--dataset-prefix", default=d.dataset_prefix)
    p.add_argument(
        "--day-order",
        default=d.day_order,
        choices=["recent-first", "oldest-first", "as-selected"],
        help="Order Phase 1 fetches days in (default: recent-first, so a run cut "
        "short by rate limiting keeps the most recent days).",
    )
    p.add_argument(
        "--list-mode",
        default=d.list_mode,
        choices=["stream", "sample"],
        help="stream (default): queue downloads page-by-page, stop listing once the "
        "per-day quota is met — far fewer API calls, downloads start immediately. "
        "sample: list every page of the day first, then take a seeded random draw.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Phase 0/1 download; mine directly from episodes already present under --raw-dir.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute Phase 2 even when the corpus, config and code are unchanged "
             "since the last successful run (default: reuse the existing artifacts).",
    )
    p.add_argument(
        "--baseline-archetypes",
        default=None,
        help="A previous archetypes.json to seed clustering from, so cluster ids "
             "and 𝒟_opp belief slots are append-only. Defaults to "
             "<out-dir>/archetypes.json when it exists — stable ids are the "
             "default because losing them silently repoints every trained "
             "checkpoint at a different deck.",
    )
    p.add_argument(
        "--rebaseline",
        action="store_true",
        help="Ignore any baseline and renumber archetypes from scratch. This "
             "INVALIDATES every existing checkpoint's deck record and every "
             "𝒟_opp belief slot; --archetype-self N will mean a different deck. "
             "Use it when the meta has moved far enough that frozen "
             "representatives are worse than a clean re-cluster, and expect to "
             "retrain.",
    )
    return p


def config_from_args(args: argparse.Namespace) -> MineConfig:
    return MineConfig(
        days=args.days.split(",") if args.days else None,
        n_days=args.n_days,
        target_episodes=args.target_episodes,
        seed=args.seed,
        k_experts=args.k_experts,
        g_min=args.g_min,
        jaccard_thresh=args.jaccard_thresh,
        n_self=args.n_self,
        n_opp=args.n_opp,
        w_lost=args.w_lost,
        vocab_mode=args.vocab_mode,
        n_vocab=args.n_vocab,
        h_max=args.h_max,
        o_max=args.o_max,
        d_max=args.d_max,
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        manifest_csv=Path(args.manifest_csv),
        dataset_prefix=args.dataset_prefix,
        day_order=args.day_order,
        list_mode=args.list_mode,
        baseline_archetypes=(
            Path(args.baseline_archetypes) if args.baseline_archetypes else None
        ),
        rebaseline=args.rebaseline,
    )


def _lineage(baseline_path, baseline, out_dir) -> dict:
    """The ``lineage`` block for archetypes.json.

    ``generation`` counts re-baselines, not runs: it increments only when ids
    are reassigned from scratch, which is precisely when previously trained
    checkpoints stop being valid.  A seeded run inherits the number, so every
    artifact in one id generation shares it and a checkpoint can be matched to
    the artifacts it is compatible with by a single integer.
    """
    import hashlib
    import json as _json

    if baseline is None or baseline_path is None:
        prior = Path(out_dir) / "archetypes.json"
        generation = 0
        if prior.exists():
            try:
                old = _json.loads(prior.read_text()).get("lineage") or {}
                generation = int(old.get("generation", 0)) + 1
            except (ValueError, OSError):
                generation = 1
        return {"seeded": False, "baseline_sha1": None, "baseline_path": None,
                "generation": generation, "n_baseline_archetypes": 0}

    path = Path(baseline_path)
    generation = 0
    try:
        generation = int(
            (_json.loads(path.read_text()).get("lineage") or {}).get("generation", 0)
        )
    except (ValueError, OSError):
        pass
    return {
        "seeded": True,
        "baseline_sha1": hashlib.sha1(path.read_bytes()).hexdigest()[:12],
        "baseline_path": str(path),
        "generation": generation,
        "n_baseline_archetypes": len(baseline),
    }


def resolve_baseline(
    config: MineConfig,
) -> tuple[list | None, list[int], list[int], Path | None]:
    """The archetype generation this run continues, or ``None`` to start fresh.

    Precedence: ``--rebaseline`` wins over everything, then an explicit
    ``--baseline-archetypes``, then ``<out-dir>/archetypes.json`` if it is
    there.  The implicit case is the important one — re-mining into a directory
    that already has artifacts is the normal way this pipeline is run, and it is
    exactly where losing the ids does the damage, so it must not depend on
    anyone remembering a flag.

    An explicitly named baseline that does not exist is an error, not a silent
    fall-through to fresh ids: the caller asked to continue a generation.
    """
    from ptcg_mine.archetype import load_archetypes_json

    if config.rebaseline:
        return None, [], [], None

    path = config.baseline_archetypes
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"--baseline-archetypes {path} does not exist. Mining without "
                "it would renumber every archetype and repoint every trained "
                "checkpoint at a different deck; pass --rebaseline if that is "
                "what you want."
            )
        archetypes, self_ids, opp_ids = load_archetypes_json(path)
        if not archetypes:
            raise ValueError(
                f"--baseline-archetypes {path} holds no archetypes, so it "
                "cannot continue an id generation. Point it at a real "
                "archetypes.json, or pass --rebaseline."
            )
        return archetypes, self_ids, opp_ids, path

    path = Path(config.out_dir) / "archetypes.json"
    if not path.exists():
        return None, [], [], None

    archetypes, self_ids, opp_ids = load_archetypes_json(path)
    if not archetypes:
        # Nothing to preserve: an artifact with no clusters carries no ids, so
        # starting fresh cannot renumber anything anyone depends on.  Said out
        # loud because the *usual* reason this path runs is to keep ids stable.
        print(f"  baseline: {path} holds no archetypes — assigning ids fresh")
        return None, [], [], None
    return archetypes, self_ids, opp_ids, path


def load_raw_episodes(raw_dir, jobs: int | None = None) -> tuple[list[dict], int]:
    """Load every `*.json` episode under raw_dir (recursively), keeping only
    those that pass validate_episode. Returns (valid_episodes, n_loaded).

    Each retained episode is reduced to its selection projection (see
    `project_for_selection`), and only as much of the document as that
    projection needs is ever decoded — so the corpus is never fully resident.
    Peak RSS over the real 2304-episode corpus: 0.12 GB, against 33.5 GB before.

    The scan is fanned out over *jobs* processes (default ``os.cpu_count()``).
    Results stay in path order whatever the worker count, which matters because
    archetype cluster ids are assigned in corpus-frequency order downstream.
    """
    raw_dir = Path(raw_dir)
    episodes: list[dict] = []
    n_loaded = 0
    # The walk is materialised anyway (it is sorted), so the exact total is
    # free -- unlike Phase 1, this bar never has to guess.
    paths = sorted(raw_dir.rglob("*.json"))
    with ProgressReporter(
        "parsing episodes", len(paths), logger=logger, unit="episode"
    ) as reporter:
        for _eid, proj, status in scan_projections(paths, jobs):
            # Advance per result, not per completed episode: a file that will
            # not parse still took the time to read.
            reporter.advance()
            if status is UNREADABLE:
                continue
            n_loaded += 1
            if status is OK:
                episodes.append(proj)
    return episodes, n_loaded


def run(config: MineConfig, skip_download: bool, force: bool = False,
        jobs: int | None = None) -> dict:
    """Run Phases 0-2 for `config`; returns a summary dict. Side effects:
    writes vocab.json, archetypes.json, mining_report.md, and the .npy static
    tables into config.out_dir.

    Phase 2 re-parses every episode under raw_dir (~15 min over the real
    9910-episode corpus) to produce artifacts that are a pure function of
    (corpus, config, this package's code). When none of those changed since the
    last successful run it is skipped and the existing artifacts are reused;
    pass `force=True` to recompute regardless.

    The check runs *after* the download deliberately: Phase 1 is what changes
    the corpus, so a run that fetches new episodes still recomputes, while one
    that fetches nothing does not.
    """
    if not skip_download:
        from kaggle.api.kaggle_api_extended import KaggleApi

        from ptcg_mine.download import download_corpus
        from ptcg_mine.kaggle_adapter import PooledKaggleApi

        api = KaggleApi()
        api.authenticate()
        # Wrapped here, not inside download.py: that module never imports
        # kaggle, so that its tests can inject fakes with no network in reach.
        with PooledKaggleApi(api) as pooled:
            download_corpus(config, pooled)

    # The baseline decides the id space, so it belongs in the fingerprint: two
    # runs over the same corpus with different baselines produce different
    # archetypes.json, and without this the second reports itself cached and
    # keeps the first one's ids.
    #
    # The implicit baseline is the previous run's own output, so upgrading an
    # existing data-dir costs exactly one extra Phase 2: the first seeded run
    # sees a fingerprint it has never recorded, and the artifacts it writes are
    # identical to what it read, so every run after that is cached again.
    params = stamp.params_from_config("mine", config)
    if not force:
        fresh, reason = stamp.check(
            "mine", raw_dir=config.raw_dir, data_dir=config.out_dir,
            params=params, require_summary=True,
        )
        rec = stamp.read("mine", config.out_dir) if fresh else None
        if rec and rec.get("summary"):
            logger.info("Phase 2: artifacts up to date (%s) — skipping", reason)
            return {**rec["summary"], "cached": True}
        logger.info("Phase 2: recomputing — %s", reason)

    episodes, n_loaded = load_raw_episodes(config.raw_dir, jobs)

    # Check the corpus itself before blaming expert selection: an empty or
    # unparseable raw_dir is a Phase 1 problem, and saying so here saves the
    # user from debugging the miner when the real fault was the download.
    if n_loaded == 0:
        raise InsufficientDataError(
            f"no episode JSON found under {config.raw_dir!s}. Phase 1 downloaded "
            f"nothing — rerun without --skip-download, or point --raw-dir at an "
            f"existing corpus."
        )
    if not episodes:
        raise InsufficientDataError(
            f"all {n_loaded} episode file(s) under {config.raw_dir!s} failed "
            f"validation (need statuses == ['DONE','DONE'], >=2 steps, 2 rewards, "
            f"two 60-card decks). The downloads may be truncated or partial."
        )

    leaderboard = team_leaderboard(episodes)
    experts = select_experts(leaderboard, config.k_experts, config.g_min)
    if not experts:
        raise InsufficientDataError(
            f"no experts could be selected from {len(episodes)} valid episode(s) "
            f"across {len(leaderboard)} team(s): no team has the required "
            f"--g-min={config.g_min} games. Download more episodes, or lower --g-min."
        )

    deck_freq: Counter = Counter()
    for ep in episodes:
        for p in (0, 1):
            deck_freq[canon(deck_of(ep, p))] += 1

    baseline, base_self, base_opp, baseline_path = resolve_baseline(config)
    archetypes = cluster_decks(deck_freq, config.jaccard_thresh, baseline=baseline)

    self_ids, opp_ids = select_self_opp(
        episodes, experts, archetypes, config.n_self, config.n_opp,
        baseline_self_ids=base_self, baseline_opp_ids=base_opp,
    )
    if baseline is not None:
        n_new = len(archetypes) - len(baseline)
        print(f"  baseline: {baseline_path} ({len(baseline)} archetypes, ids "
              f"0..{max(a.id for a in baseline)}) → +{n_new} new cluster(s)")
        if len(opp_ids) != len(base_opp):
            # n_opp_arch just changed, so a policy built against the new
            # artifacts no longer has the same belief-head width as one built
            # against the old.  Loud, because --resume across this boundary is
            # a shape mismatch and warm-starting needs the widening path.
            print(f"  WARNING: 𝒟_opp grew {len(base_opp)} → {len(opp_ids)} slots. "
                  "Old checkpoints keep their own width and still evaluate, but "
                  "resuming one into the new artifacts needs "
                  "--allow-belief-widening.")
    else:
        print("  baseline: none — archetype ids are being assigned from scratch")

    fixed_deck = pick_fixed_deck(episodes, experts, self_ids, archetypes)

    vocab = build_vocab(episodes, mode=config.vocab_mode, n_vocab=config.n_vocab)

    cards, attacks = load_engine()
    card_table, attack_id_to_index, attack_table = build_static_tables(vocab, cards, attacks)

    expert_set = set(experts)
    expert_games = sum(
        1 for ep in episodes if any(t in expert_set for t in ep["info"]["TeamNames"])
    )
    counts = {
        "episodes_loaded": n_loaded,
        "episodes_valid": len(episodes),
        "n_experts": len(experts),
        "expert_games": expert_games,
        "n_archetypes": len(archetypes),
    }

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_vocab_json(out_dir / "vocab.json", vocab, attack_id_to_index, config)
    lineage = _lineage(baseline_path, baseline, out_dir)
    write_archetypes_json(out_dir / "archetypes.json", self_ids, opp_ids,
                          archetypes, fixed_deck, lineage=lineage)
    write_mining_report(
        out_dir / "mining_report.md", leaderboard, experts, archetypes, self_ids, opp_ids, vocab, counts
    )
    np.save(out_dir / "card_static_table.npy", card_table)
    np.save(out_dir / "attack_static_table.npy", attack_table)
    engine_card_features = build_engine_card_features(cards, attacks)
    np.save(out_dir / "engine_card_features.npy", engine_card_features)
    engine_attack_features = build_engine_attack_features(attacks)
    np.save(out_dir / "engine_attack_features.npy", engine_attack_features)

    evolution_map = build_evolution_map(cards)
    np.save(out_dir / "evolution_map.npy", evolution_map)

    summary = {
        "vocab_size": vocab["size"],
        "n_attacks": int(attack_table.shape[0]),
        "n_experts": len(experts),
        "n_archetypes": len(archetypes),
        "self_ids": list(self_ids),
        "opp_ids": list(opp_ids),
        "out_dir": str(out_dir),
        **counts,
    }

    # Stamped only now, with every artifact on disk: a run that died partway
    # must leave no stamp, so the next one recomputes rather than trusting it.
    stamp.write("mine", raw_dir=config.raw_dir, data_dir=out_dir,
                params=params, summary=summary)
    return {**summary, "cached": False}


def main(argv: list[str] | None = None) -> int:
    import kagglehub

    # Download latest version
    import shutil
    path = kagglehub.dataset_download("kaggle/pokemon-tcg-ai-battle-episodes-index")
    shutil.rmtree(path)
    path = kagglehub.dataset_download("kaggle/pokemon-tcg-ai-battle-episodes-index")

    print("Path to dataset files:", path)
    shutil.copy(Path(path)/'manifest.csv' , '/home/charles/Documents/Pokemon/python/archive/manifest.csv')
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from ptcg_mine.download import DownloadError

    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    try:
        summary = run(config, skip_download=args.skip_download, force=args.force,
                      jobs=getattr(args, "jobs", None))
    except DownloadError as exc:
        print(f"[ptcg_mine] download failed: {exc}", file=sys.stderr)
        return 2
    except InsufficientDataError as exc:
        print(f"[ptcg_mine] insufficient data: {exc}", file=sys.stderr)
        return 1

    print("=== Corpus mining summary ==="
          + (" (cached — Phase 2 skipped)" if summary.get("cached") else ""))
    print(f"episodes loaded:    {summary['episodes_loaded']}")
    print(f"episodes valid:     {summary['episodes_valid']}")
    print(f"experts:            {summary['n_experts']}")
    print(f"expert games:       {summary['expert_games']}")
    print(f"archetypes:         {summary['n_archetypes']}")
    print(f"D_self ids:         {summary['self_ids']}")
    print(f"D_opp ids:          {summary['opp_ids']}")
    print(f"vocab size (V):     {summary['vocab_size']}")
    print(f"attack table (A):   {summary['n_attacks']}")
    print(f"artifacts written to: {summary['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
