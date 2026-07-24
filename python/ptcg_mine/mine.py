"""CLI orchestrator for Phases 0-2 of the corpus-mining pipeline.

Phase 0 (sampling) + Phase 1 (download) are handled by `download_corpus`
(skippable via --skip-download for offline/dev runs against an already-
populated raw_dir). Phase 2 (stats -> archetypes -> vocab -> static tables
-> artifacts) always runs over the parsed episodes found under `raw_dir`.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ptcg_mine.archetype import canon, cluster_decks, pick_fixed_deck, select_self_opp
from ptcg_mine.artifacts import write_archetypes_json, write_mining_report, write_vocab_json
from ptcg_mine.cards import build_static_tables, load_engine
from ptcg_mine.config import MineConfig
from ptcg_mine.episode import deck_of, load_episode, validate_episode
from ptcg_mine.stats import select_experts, team_leaderboard
from ptcg_mine.vocab import build_vocab


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
    )


def load_raw_episodes(raw_dir) -> tuple[list[dict], int]:
    """Load every `*.json` episode under raw_dir (recursively), keeping only
    those that pass validate_episode. Returns (valid_episodes, n_loaded)."""
    raw_dir = Path(raw_dir)
    episodes: list[dict] = []
    n_loaded = 0
    for path in sorted(raw_dir.rglob("*.json")):
        try:
            ep = load_episode(path)
        except (json.JSONDecodeError, OSError):
            continue
        n_loaded += 1
        if validate_episode(ep):
            episodes.append(ep)
    return episodes, n_loaded


def run(config: MineConfig, skip_download: bool) -> dict:
    """Run Phases 0-2 for `config`; returns a summary dict. Side effects:
    writes vocab.json, archetypes.json, mining_report.md, and the .npy static
    tables into config.out_dir."""
    if not skip_download:
        from kaggle.api.kaggle_api_extended import KaggleApi

        from ptcg_mine.download import download_corpus

        api = KaggleApi()
        api.authenticate()
        download_corpus(config, api)

    episodes, n_loaded = load_raw_episodes(config.raw_dir)

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
    archetypes = cluster_decks(deck_freq, config.jaccard_thresh)

    self_ids, opp_ids = select_self_opp(episodes, experts, archetypes, config.n_self, config.n_opp)
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
    write_archetypes_json(out_dir / "archetypes.json", self_ids, opp_ids, archetypes, fixed_deck)
    write_mining_report(
        out_dir / "mining_report.md", leaderboard, experts, archetypes, self_ids, opp_ids, vocab, counts
    )
    np.save(out_dir / "card_static_table.npy", card_table)
    np.save(out_dir / "attack_static_table.npy", attack_table)

    return {
        "vocab_size": vocab["size"],
        "n_attacks": attack_table.shape[0],
        "n_experts": len(experts),
        "n_archetypes": len(archetypes),
        "self_ids": self_ids,
        "opp_ids": opp_ids,
        "out_dir": str(out_dir),
        **counts,
    }


def main(argv: list[str] | None = None) -> int:
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
        summary = run(config, skip_download=args.skip_download)
    except DownloadError as exc:
        print(f"[ptcg_mine] download failed: {exc}", file=sys.stderr)
        return 2
    except InsufficientDataError as exc:
        print(f"[ptcg_mine] insufficient data: {exc}", file=sys.stderr)
        return 1

    print("=== Corpus mining summary ===")
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
