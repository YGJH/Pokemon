"""Train CLI entry point — wires mine → featurize → train.

Provides the ``ptcg-il train`` subcommand with all hyperparameters from C.10
as CLI flags.

Usage::

    python -m ptcg_il.cli train --data-dir data --out-dir checkpoints

    # Resume from checkpoint
    python -m ptcg_il.cli train --resume checkpoints/ckpt-step-0004000.pt

    # Eval only (load a checkpoint, run offline + live eval)
    python -m ptcg_il.cli train --eval-only --resume checkpoints/ckpt-best.pt

    # With live eval opponents
    python -m ptcg_il.cli train --live-eval --live-eval-games 500
"""

from __future__ import annotations

import argparse
import json
import logging
from rich.logging import RichHandler
import os
import sys
from pathlib import Path
from typing import Any

# belief_labels is deliberately torch-free, so importing the weight defaults
# here does not drag torch into `--help`.
from ptcg_il.belief_labels import BELIEF_WEIGHTS
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


# ============================================================
# Default hyperparameters (C.10)
# ============================================================
DEFAULTS = {
    # Model
    "d_model": 256,
    "layers": 4,
    "heads": 8,
    "ff": 1024,
    "dropout": 0.1,
    # Training
    "batch_size": 1024,
    "epochs": 1000,
    "peak_lr": 8e-5,
    "warmup": 100,
    "min_lr": 1e-5,
    "weight_decay": 0.01,
    "label_smoothing": 0.05,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "alpha_ctx": 0.5,
    "alpha_arch": 0.5,
    "lambda_v": 0.5,
    "w_lost": 0.6,
    # Cadence
    "log_every": 50,
    "val_every": 100,
    "ckpt_every": 2000,
    "live_every": 1000000,
    # Data
    "num_workers": 8,
    # Precision
    "mixed_precision": True,
    # Patience
    "patience": 5000,
    # Live eval
    "live_eval_games": 500,
    # W&B.  Runs land in the `poken` team by default; override with
    # --wandb-entity, or set it to your personal entity for a scratch run.
    "wandb_project": "pokemon-tcg-il",
    "wandb_entity": "poken",
    "wandb_name": "pokemon-tcg-il",
    "wandb_mode": "online",
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``ptcg-il train``."""
    parser = argparse.ArgumentParser(
        prog="ptcg-il",
        description="Transformer IL: train an imitation-learning policy for Pokemon TCG",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- train ----
    train_parser = sub.add_parser(
        "train",
        help="Run training (shard → featurize → train → eval)",
    )

    # ---- build-shards ----
    bs_parser = sub.add_parser(
        "build-shards",
        help="Featurize kept expert games into training-ready .npz shards + meta.parquet (Phase 3)",
    )
    bs_paths = bs_parser.add_argument_group("Paths")
    bs_paths.add_argument("--raw-dir", type=str, default="raw",
                        help="Directory with downloaded episode JSONs (default: raw/)")
    bs_paths.add_argument("--out-dir", type=str, default="data",
                        help="Directory with vocab.json + archetypes.json; shards/ written here (default: data/)")
    bs_config = bs_parser.add_argument_group("Corpus config")
    bs_config.add_argument("--k-experts", type=int, default=10,
                         help="Number of expert teams to select (default: 10)")
    bs_config.add_argument("--g-min", type=int, default=50,
                         help="Min games for expert eligibility (default: 50)")
    bs_config.add_argument("--jaccard-thresh", type=float, default=0.90,
                         help="Jaccard threshold for archetype clustering (default: 0.90)")
    bs_config.add_argument("--samples-per-shard", type=int, default=None,
                         help="Max samples per .npz shard file. Default: derived "
                              "from --mem-budget-gb and the measured size of the "
                              "first real sample. Pass an int to pin it and "
                              "ignore the budget.")
    bs_config.add_argument("--mem-budget-gb", type=float, default=None,
                         help="Ceiling on the writer's resident memory in GiB "
                              "(default: 2.0). Sizes the three shard buffers, "
                              "which are its largest term; meta.parquet is "
                              "streamed and costs ~40 MB regardless. Ignored "
                              "when --samples-per-shard is given.")
    bs_parser.add_argument("--jobs", "-j", type=int, default=None,
                         help="Worker processes for both corpus passes — the "
                              "pass-A selection scan and the pass-B featurize "
                              "(default: os.cpu_count(); 1 disables both pools). "
                              "Both are order-preserving, so this cannot change "
                              "the output.")
    bs_parser.add_argument("--force", action="store_true",
                         help="Rebuild even when the corpus, config, code and "
                              "vocab/archetypes are unchanged since the last "
                              "successful build (default: reuse existing shards)")

    # ---- archetypes ----
    # The pipeline calls this to decide which specialists to train.  Hardcoding
    # ids in the shell is unsafe: they are cluster indices and get reassigned
    # whenever mining is re-run.
    arch_parser = sub.add_parser(
        "archetypes",
        help="Print the best-supported 𝒟_self archetype ids for this corpus",
    )
    arch_parser.add_argument("--data-dir", type=str, default="data",
                             help="Directory with archetypes.json + meta.parquet")
    arch_parser.add_argument("--top", type=int, default=2,
                             help="How many ids to print (default: 2)")
    arch_parser.add_argument("--describe", action="store_true",
                             help="Print a row-count table for every 𝒟_self archetype "
                                  "instead of the bare id list")

    # Paths
    paths = train_parser.add_argument_group("Paths")
    paths.add_argument("--data-dir", type=str, default="data",
                       help="Directory with shards/, meta.parquet, vocab.json, archetypes.json")
    paths.add_argument("--out-dir", type=str, default="checkpoints",
                       help="Directory for checkpoints and logs")
    paths.add_argument("--resume", type=str, default=None,
                       help="Resume from a checkpoint file")
    paths.add_argument("--ckpt", action="append", default=None,
                       help="Checkpoint path for eval. Pass multiple times for "
                            "ensemble eval (--eval-only with 2+ checkpoints).")
    paths.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation (offline + live), skip training")
    paths.add_argument("--live-eval", action="store_true",
                       help="Run live-engine evaluation during training")

    # Model architecture
    model = train_parser.add_argument_group("Model architecture (Appendix B)")
    model.add_argument("--d-model", type=int, default=DEFAULTS["d_model"],
                       help="Model dimension")
    model.add_argument("--layers", type=int, default=DEFAULTS["layers"],
                       help="Transformer encoder layers")
    model.add_argument("--heads", type=int, default=DEFAULTS["heads"],
                       help="Attention heads")
    model.add_argument("--ff", type=int, default=DEFAULTS["ff"],
                       help="Feed-forward hidden dim")
    model.add_argument("--dropout", type=float, default=DEFAULTS["dropout"],
                       help="Dropout rate")

    # Training hyperparameters
    train_hp = train_parser.add_argument_group("Training (C.10)")
    train_hp.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"],
                          help="Decision points per micro-batch")
    train_hp.add_argument("--grad-accum", type=int, default=1,
                          help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    train_hp.add_argument("--seed", type=int, default=42,
                          help="Random seed for model init and batch order. "
                               "Vary across ensemble members for decorrelation.")
    train_hp.add_argument("--archetype-self", type=int, default=None,
                          help="Train a per-deck specialist on this archetype id only "
                               "(see archetypes.json self_ids). The archetype decks are "
                               "near-disjoint, so a specialist avoids fitting several "
                               "unrelated policies at once. Default: all decks.")
    train_hp.add_argument("--epochs", type=int, default=DEFAULTS["epochs"],
                          help="Training epochs (overridden by --total-steps)")
    train_hp.add_argument("--total-steps", type=int, default=None,
                          help="Override total training steps (default: epochs * steps_per_epoch)")
    from ptcg_il.train.loop import MUON_LR

    train_hp.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw",
                          help="adamw (default) or muon: orthogonalized momentum "
                               "for the trunk matrices, AdamW for embeddings, "
                               "biases, norms and output heads")
    train_hp.add_argument("--muon-lr", type=float, default=MUON_LR,
                          help="Peak LR for the orthogonalized group. Does NOT "
                               "transfer from --peak-lr: an orthogonalized "
                               "update's size is set by the matrix shape, not "
                               "the gradient. Ignored unless --optimizer muon")
    train_hp.add_argument("--peak-lr", type=float, default=DEFAULTS["peak_lr"],
                          help="Peak learning rate (AdamW parameters)")
    train_hp.add_argument("--min-lr", type=float, default=DEFAULTS["min_lr"],
                          help="Minimum learning rate (cosine floor)")
    train_hp.add_argument("--warmup", type=int, default=DEFAULTS["warmup"],
                          help="Linear warmup steps")
    train_hp.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"],
                          help="AdamW weight decay")
    train_hp.add_argument("--beta1", type=float, default=0.9,
                          help="AdamW beta1")
    train_hp.add_argument("--beta2", type=float, default=0.95,
                          help="AdamW beta2")
    train_hp.add_argument("--grad-clip", type=float, default=DEFAULTS["grad_clip"],
                          help="Max global gradient norm")
    train_hp.add_argument("--label-smoothing", type=float, default=DEFAULTS["label_smoothing"],
                          help="Label smoothing for CE")
    train_hp.add_argument(
        "--no-group-marginal-ce", dest="group_marginal", action="store_false",
        default=True,
        help="Score each option label on its own instead of marginalising over "
             "byte-identical options (the pre-2026-08 behaviour).",
    )

    # Loss / weighting
    loss = train_parser.add_argument_group("Loss & weighting (C.3, C.10)")
    loss.add_argument("--alpha-ctx", type=float, default=DEFAULTS["alpha_ctx"],
                      help="Exponent for rare-context balancing")
    loss.add_argument("--alpha-arch", type=float, default=DEFAULTS["alpha_arch"],
                      help="Exponent for archetype balancing")
    loss.add_argument("--lambda-v", type=float, default=DEFAULTS["lambda_v"],
                      help="Value loss weight")
    loss.add_argument("--w-lost", type=float, default=DEFAULTS["w_lost"],
                      help="Weight multiplier for lost-game decisions")
    loss.add_argument("--ema-decay", type=float, default=DEFAULTS["ema_decay"],
                      help="EMA decay for parameter averaging")

    # Opponent-card belief (auxiliary supervision for MCTS determinization).
    # These heads always train: build-shards writes belief labels for every
    # decision point unconditionally, and the RL stage's determinizer needs the
    # heads to exist.  Set every weight to 0 to disable the term.
    bel = train_parser.add_argument_group("Belief heads")
    bel.add_argument("--no-belief", action="store_true",
                     help="Disable the auxiliary opponent-card belief heads "
                          "(archetype / deck / hidden pool / hand). They train by "
                          "default; disabling them saves the heads' [B, V] matmuls "
                          "but leaves the MCTS determinizer on its mirror-deck guess.")
    for name, default in BELIEF_WEIGHTS.items():
        bel.add_argument(f"--belief-{name[2:]}", type=float, default=default,
                         dest=f"belief_{name[2:]}",
                         help=f"Weight of the belief {name[2:]} term (default {default})")

    # Cadence
    cadence = train_parser.add_argument_group("Logging & eval cadence (C.8–C.10)")
    cadence.add_argument("--log-every", type=int, default=DEFAULTS["log_every"],
                         help="Log scalars every N steps")
    cadence.add_argument("--val-every", type=int, default=DEFAULTS["val_every"],
                         help="Run offline eval every N steps")
    cadence.add_argument("--ckpt-every", type=int, default=DEFAULTS["ckpt_every"],
                         help="Save checkpoint every N steps")
    cadence.add_argument("--live-every", type=int, default=DEFAULTS["live_every"],
                         help="Run live eval every N steps")

    # Data
    data = train_parser.add_argument_group("Data loading")
    data.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"],
                      help="DataLoader workers")
    data.add_argument("--mixed-precision", action="store_true", default=DEFAULTS["mixed_precision"],
                      help="Use bf16 autocast (default: True)")
    data.add_argument("--fp32", action="store_true",
                      help="Disable mixed precision (use fp32)")

    # Early stop
    early = train_parser.add_argument_group("Early stopping")
    early.add_argument("--patience", type=int, default=DEFAULTS["patience"],
                       help="Early-stop patience in val-eval cycles")

    # Live eval
    live = train_parser.add_argument_group("Live-eval (Section 5)")
    live.add_argument("--live-eval-games", type=int, default=DEFAULTS["live_eval_games"],
                      help="Games per opponent per live-eval run")
    live.add_argument("--live-eval-workers", type=int, default=None,
                      help="Number of concurrent game processes")

    # W&B
    wandb_group = train_parser.add_argument_group("W&B logging (C.9)")
    wandb_group.add_argument("--wandb-project", type=str, default=DEFAULTS["wandb_project"],
                             help="W&B project name")
    wandb_group.add_argument("--wandb-entity", type=str, default=DEFAULTS["wandb_entity"],
                             help="W&B team/entity")
    wandb_group.add_argument("--wandb-name", type=str, default=DEFAULTS["wandb_name"],
                             help="W&B run name")
    wandb_group.add_argument("--wandb-mode", type=str, default=DEFAULTS["wandb_mode"],
                             choices=["online", "offline", "disabled"],
                             help="W&B mode")
    wandb_group.add_argument("--no-wandb", action="store_true",
                             help="Disable W&B entirely")

    # QA
    qa = train_parser.add_argument_group("QA gates (D.5)")
    qa.add_argument("--skip-qa", action="store_true",
                    help="Skip QA gates (not recommended)")
    qa.add_argument("--allow-belief-widening", action="store_true",
                    help="Accept a --resume checkpoint whose belief archetype "
                         "head is narrower than the current artifacts require, "
                         "copying its rows into the leading 𝒟_opp slots. Sound "
                         "only when archetypes.json was SEEDED from the "
                         "checkpoint's generation (lineage.seeded), because "
                         "then slot i still means the archetype it did. After "
                         "a --rebaseline the leading rows describe different "
                         "decks and this makes an untrained head look trained.")

    # Baseline recording (pipeline stage 4c)
    base = train_parser.add_argument_group("IL baselines (RL_SPEC §10.2 cond. 3)")
    base.add_argument("--eval-split", type=str, default="val",
                      choices=["train", "val", "test"],
                      help="Split for --eval-only (default: val). Baselines are "
                           "recorded from 'test' — RL_SPEC §10.2 condition 3 is a "
                           "held-out check, and 'val' drove model selection.")
    base.add_argument("--ensemble-select", type=int, default=None, metavar="K",
                      help="With --eval-only and 2+ --ckpt: fit a greedy "
                           "forward selection of ensemble members on the val "
                           "split, record the full ordering into "
                           "<data-dir>/il_baselines.json, and evaluate the "
                           "best K on test. build_submit.sh --ensemble-top "
                           "reads that ordering. Requires --eval-split val: "
                           "greedy on test would fit the subset to the split "
                           "the recorded score claims to hold out.")
    base.add_argument("--record-baseline", action="store_true",
                      help="Write this checkpoint's offline-eval scores to "
                           "<data-dir>/il_baselines.json, SHA-1-pinned to the "
                           "checkpoint. The RL promotion gate reads this and "
                           "refuses to run against a baseline from a different "
                           "model. Use with --eval-only.")

    return parser


# ============================================================
# Main dispatch
# ============================================================


def _load_artifacts(data_dir: Path) -> dict:
    """Load vocab.json and archetypes.json from *data_dir*, returning a dict
    with keys ``vocab``, ``vocab_size``, ``attack_size``, ``fixed_deck``."""
    artifacts: dict[str, Any] = {}

    vocab_path = data_dir / "vocab.json"
    if vocab_path.exists():
        from ptcg_il.featurizer import normalize_vocab

        with open(vocab_path) as f:
            artifacts["vocab"] = normalize_vocab(json.load(f))
        artifacts["vocab_size"] = artifacts["vocab"].get("size", 0)
        artifacts["attack_size"] = len(artifacts["vocab"].get("attack_id_to_index", {})) + 1
    else:
        raise FileNotFoundError(f"vocab.json not found at {vocab_path}")

    arch_path = data_dir / "archetypes.json"
    if arch_path.exists():
        with open(arch_path) as f:
            arch = json.load(f)
        artifacts["fixed_deck"] = arch.get("fixed_deck", list(range(60)))
        # The belief head classifies over the retained 𝒟_opp set, in the same
        # contiguous order shard_writer used when it wrote bel_arch.
        artifacts["n_opp_arch"] = len(arch.get("opp_ids", []))
        artifacts["archetypes"] = arch
    else:
        raise FileNotFoundError(f"archetypes.json not found at {arch_path}")

    return artifacts


def _belief_weights(args: argparse.Namespace) -> dict[str, float] | None:
    """CLI flags → the ``belief_weights`` dict, or None when disabled.

    None (not a dict of zeros) is what turns the belief forward pass off
    entirely, so a run with ``--no-belief`` never pays for the heads' [B, V]
    matmuls.

    The heads are on by default.  They used to be opt-in via ``--belief``,
    guarding against shards mined without belief labels — but ``shard_writer``
    writes those labels for every decision point unconditionally, so the guard
    protected against a corpus that no longer exists, while the *default* left
    the MCTS determinizer on its mirror-deck guess.
    """
    if getattr(args, "no_belief", False):
        return None
    return {name: float(getattr(args, f"belief_{name[2:]}")) for name in BELIEF_WEIGHTS}


def _load_static_tables(data_dir: Path) -> Any:
    """``(card_table, attack_table)`` for the policy's on-device card gather.

    Thin wrapper over :func:`ptcg_il.model.policy.load_static_tables` so the
    import stays lazy — ``cli`` is imported for subcommands that never build a
    model.  The card table doubles as the belief heads' scoring matrix; the
    policy wires both from one call.
    """
    from ptcg_il.model.policy import load_static_tables

    return load_static_tables(data_dir)


def _build_policy(artifacts: dict, args: argparse.Namespace) -> Any:
    """Create a Policy module from artifact sizes and CLI args.

    There is no vocab width to pass: cards reach the model purely as static
    feature vectors (``CardFeaturizer``), so the only card-derived shape is
    ``n_all_cards`` for the belief heads' output width.
    """
    from ptcg_il.model.policy import Policy

    all_card_feat, all_attack_feat = _load_static_tables(Path(args.data_dir))
    n_all_cards = int(all_card_feat.shape[0]) if all_card_feat is not None else 0
    missing = [name for name, t in (("engine_card_features.npy", all_card_feat),
                                    ("engine_attack_features.npy", all_attack_feat))
               if t is None]
    if missing:
        # Every card would embed as all-zeros and the belief heads would have no
        # scoring matrix.  Refuse here rather than at the first forward pass,
        # several minutes into a run -- or worse, not at all.
        raise SystemExit(
            f"missing {', '.join(missing)} in {args.data_dir}; the policy cannot "
            "featurize cards without the engine static tables. Run "
            "`python -m ptcg_mine.mine` to write them."
        )
    logger.info("static tables: %d cards x %d, %d attacks x %d",
                n_all_cards, int(all_card_feat.shape[1]),
                int(all_attack_feat.shape[0]), int(all_attack_feat.shape[1]))

    policy = Policy(
        D=args.d_model,
        heads=args.heads,
        layers=args.layers,
        ff=args.ff,
        n_opp_arch=artifacts.get("n_opp_arch", 0),
        n_all_cards=n_all_cards,
        all_card_feat=all_card_feat,
        all_attack_feat=all_attack_feat,
    )
    policy.config["seed"] = args.seed
    # Apply spec B.8 weight init (trunc_normal std=0.02 for Linear/Embedding weights)
    from ptcg_il.model import init_weights
    init_weights(policy)
    return policy


def cmd_build_shards(args: argparse.Namespace) -> int:
    """Execute the ``build-shards`` subcommand (Phase 3 featurization).

    Featurizing the whole corpus takes ~35 min and is a pure function of
    (corpus, config, featurizer/selection code, vocab.json + archetypes.json),
    so an unchanged run reuses the existing shards.  ``--force`` recomputes.
    """
    from ptcg_mine import stamp
    from ptcg_mine.config import MineConfig
    from ptcg_il.shard_writer import DEFAULT_MEM_BUDGET_GB, build_shards

    config = MineConfig(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        k_experts=args.k_experts,
        g_min=args.g_min,
        jaccard_thresh=args.jaccard_thresh,
    )
    # getattr for the same reason `jobs` uses it below: callers build this
    # Namespace by hand, so a new flag must not be mandatory on it.
    mem_budget_gb = getattr(args, "mem_budget_gb", None)
    if mem_budget_gb is None:
        mem_budget_gb = DEFAULT_MEM_BUDGET_GB
    # Both knobs are stamped, because either one can change the shard size and a
    # shard size change repoints every `(shard, row)` pair in meta.parquet.  The
    # budget matters even though it only *derives* the size: the per-sample cost
    # it divides comes from the featurizer, whose source files the fingerprint
    # already covers, so budget + code together pin the result.
    params = stamp.params_from_config(
        "shards", config,
        **{"samples-per-shard": args.samples_per_shard,
           "mem-budget-gb": mem_budget_gb},
    )

    if not args.force:
        fresh, reason = stamp.check(
            "shards", raw_dir=config.raw_dir, data_dir=config.out_dir,
            params=params, require_summary=True,
        )
        rec = stamp.read("shards", config.out_dir) if fresh else None
        if rec and rec.get("summary"):
            summary = rec["summary"]
            logger.info(
                "Phase 3: shards up to date (%s) — skipping. %d samples in %d shards.",
                reason, summary["total_samples"], summary["n_shards"],
            )
            return 0
        logger.info("Phase 3: rebuilding shards — %s", reason)

    logger.info("Phase 3: building shards from %s → %s", config.raw_dir, config.out_dir)
    summary = build_shards(config, samples_per_shard=args.samples_per_shard,
                           mem_budget_gb=mem_budget_gb,
                           jobs=getattr(args, "jobs", None))

    logger.info("Shard build complete:")
    logger.info("  samples:    %d total (%s)", summary["total_samples"], summary["split_counts"])
    logger.info("  shards:     %d files", summary["n_shards"])
    logger.info("  kept games: %d", summary["n_kept_games"])
    logger.info("  loaded:     %d episodes (%d invalid)", summary["n_loaded"], summary["n_invalid"])
    logger.info("  meta:       %s", summary["meta_path"])

    if summary["total_samples"] == 0:
        logger.error("No samples produced! Check that experts, D_self/D_opp, and raw data are available.")
        return 1

    # Stamped only on a run that produced samples, so a failed build never
    # blesses a partial shards/ directory.
    stamp.write("shards", raw_dir=config.raw_dir, data_dir=config.out_dir,
                params=params, summary=summary)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Execute the ``train`` subcommand."""
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    logger.info("Loading artifacts from %s", data_dir)
    artifacts = _load_artifacts(data_dir)

    # QA gates
    if not args.skip_qa:
        logger.info("Running QA gates...")
        from ptcg_il.qa import run_qa_checks

        qa_results = run_qa_checks(
            shard_dir=data_dir / "shards",
            meta_path=data_dir / "meta.parquet",
            vocab=artifacts["vocab"],
            fixed_deck=artifacts["fixed_deck"],
            max_samples=2000,
        )
        logger.info("QA complete: coverage=%s, label_sanity=%s, deck_legality=%s",
                    qa_results.get("coverage_pass"),
                    qa_results.get("label_sanity_pass"),
                    qa_results.get("deck_legality_pass"))

    # Build policy
    logger.info(
        "Building Policy(D=%d, heads=%d, layers=%d, ff=%d, n_opp_arch=%d)",
        args.d_model,
        args.heads,
        args.layers,
        args.ff,
        artifacts.get("n_opp_arch", 0),
    )
    import torch as _torch
    _torch.manual_seed(args.seed)
    policy = _build_policy(artifacts, args)

    # Eval-only mode
    if args.eval_only:
        if args.ckpt and len(args.ckpt) >= 2:
            return _cmd_eval_only_ensemble(args.ckpt, artifacts, args)
        if args.resume is None and not args.ckpt:
            logger.error("--eval-only requires --resume or --ckpt to specify a checkpoint")
            return 1
        # Single checkpoint: use --ckpt if provided, else --resume
        ckpt_path = args.ckpt[0] if args.ckpt else args.resume
        # Override args.resume so _cmd_eval_only reads the right path
        args.resume = ckpt_path
        return _cmd_eval_only(policy, artifacts, args)

    os.environ["WANDB_MODE"] = _wandb_mode(args)

    # Determine total steps
    total_steps = args.total_steps

    # Run training
    from ptcg_il.train.loop import train

    logger.info("Starting training: %s → %s", data_dir, out_dir)
    trained_policy = train(
        policy,
        data_dir=data_dir,
        save_dir=out_dir,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        optimizer_name=args.optimizer,
        muon_lr=args.muon_lr,
        archetype_self=args.archetype_self,
        seed=args.seed,
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
        warmup=args.warmup,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        group_marginal=args.group_marginal,
        ema_decay=args.ema_decay,
        lambda_v=args.lambda_v,
        belief_weights=_belief_weights(args),
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        total_steps=total_steps,
        log_every=args.log_every,
        val_every=args.val_every,
        ckpt_every=args.ckpt_every,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision and not args.fp32,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=_wandb_run_name(args),
        resume_ckpt=args.resume,
        allow_belief_widening=args.allow_belief_widening,
        run_val=True,
        patience=args.patience,
    )

    # Live eval after training
    if args.live_eval:
        _run_live_eval(trained_policy, artifacts, args)

    logger.info("Training complete.  Best checkpoint at %s/ckpt-best.pt", out_dir)
    return 0


def _wandb_mode(args: argparse.Namespace) -> str:
    """Effective ``WANDB_MODE``.

    ``--no-wandb`` wins over ``--wandb-mode``: it was declared as an off switch
    but never read anywhere, so passing it used to log online regardless.
    """
    return "disabled" if getattr(args, "no_wandb", False) else args.wandb_mode


def _wandb_run_name(args: argparse.Namespace) -> str | None:
    """Run name, suffixed with the archetype when training a specialist.

    Pipeline stage 4a trains one model per 𝒟_self archetype in the same
    project, so the bare default would produce several identically-named runs
    that can only be told apart by opening their config.  An explicit
    ``--wandb-name`` is honoured verbatim — the suffix is only added to the
    default.
    """
    name = args.wandb_name
    if name is None or name != DEFAULTS["wandb_name"]:
        return name
    if getattr(args, "archetype_self", None) is None:
        return f"{name}-generalist"
    return f"{name}-a{args.archetype_self}"


def _cmd_eval_only(policy: Any, artifacts: dict, args: argparse.Namespace) -> int:
    """Run offline eval + optional live eval from a checkpoint."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    from ptcg_il.train.checkpoint import load_checkpoint
    ckpt = load_checkpoint(args.resume, device)
    from ptcg_il.model.policy import load_policy_state
    load_policy_state(policy, ckpt["model_state_dict"],
                      allow_belief_widening=args.allow_belief_widening)
    policy.to(device)
    policy.eval()

    logger.info("Loaded checkpoint from step %d", ckpt["step"])

    # Offline eval
    from ptcg_il.train.dataset import ShardDataset, collate_fn
    from ptcg_il.train.eval import offline_eval
    from torch.utils.data import DataLoader

    split = getattr(args, "eval_split", "val")
    # A baseline is a held-out claim, so it must not come from the split that
    # chose the checkpoint.  Fail rather than silently recording a val number
    # under a name the gate will read as "test".
    if getattr(args, "record_baseline", False) and split != "test":
        logger.error(
            "--record-baseline requires --eval-split test (got %r): RL_SPEC "
            "§10.2 condition 3 is a held-out check, and 'val' selected this "
            "checkpoint", split,
        )
        return 1

    eval_metrics: dict[str, Any] = {}
    try:
        eval_ds = ShardDataset(args.data_dir, split=split, shuffle=False,
                               archetype_self=args.archetype_self)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        eval_metrics = offline_eval(
            policy, eval_loader, device, lambda_v=args.lambda_v,
            belief=_belief_weights(args) is not None,
        )
        logger.info(
            "Offline eval (%s): top1_macro=%.4f, top1_micro=%.4f, "
            "top1_nontrivial=%.4f, value_corr=%.4f, value_std=%.4f",
            split,
            eval_metrics.get("val/top1_macro", 0.0),
            eval_metrics.get("val/top1_micro", 0.0),
            eval_metrics.get("val/top1_nontrivial", 0.0),
            eval_metrics.get("val/value_corr", 0.0),
            eval_metrics.get("val/value_std", 0.0),
        )
    except (ValueError, FileNotFoundError) as e:
        logger.warning("Cannot run offline eval: %s", e)

    if getattr(args, "record_baseline", False):
        if not eval_metrics:
            logger.error("--record-baseline: offline eval produced no metrics")
            return 1
        from ptcg_il.baselines import record_baseline

        record = record_baseline(
            args.data_dir, args.archetype_self, args.resume, eval_metrics
        )
        logger.info(
            "Recorded IL baseline for archetype %s: nontrivial_top1=%.4f "
            "(sha1 %s)",
            args.archetype_self if args.archetype_self is not None else "generalist",
            record.get("nontrivial_top1", 0.0),
            record["ckpt_sha1"][:12],
        )

    # Live eval
    if args.live_eval:
        _run_live_eval(policy, artifacts, args)

    return 0


def _cmd_eval_only_ensemble(
    ckpt_paths: list[str], artifacts: dict, args: argparse.Namespace
) -> int:
    """Ensemble eval mode: evaluate each member and the ensemble.

    On val split: eval each member + ensemble, print comparison table.
    On test split: eval ensemble + best single member only.
    """
    import torch
    from ptcg_il.ensemble import EnsemblePolicy
    from ptcg_il.train.dataset import ShardDataset, collate_fn
    from ptcg_il.train.eval import offline_eval
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_card_feat, all_attack_feat = _load_static_tables(Path(args.data_dir))

    # Build EnsemblePolicy from checkpoints
    logger.info("Building ensemble from %d checkpoints...", len(ckpt_paths))
    ensemble = EnsemblePolicy.from_checkpoints(
        ckpt_paths, all_card_feat=all_card_feat,
        all_attack_feat=all_attack_feat, device=str(device),
    )
    ensemble.to(device)
    ensemble.eval()

    split = getattr(args, "eval_split", "val")
    belief = _belief_weights(args) is not None

    select_k = getattr(args, "ensemble_select", None)
    if select_k is not None:
        if split != "val":
            logger.error(
                "--ensemble-select requires --eval-split val (got %r): the "
                "subset that maximises a test score is fit to test, and the "
                "ens-%d record written afterwards would then be a selection "
                "target wearing the name of a held-out claim",
                split, select_k,
            )
            return 1
        if not 1 <= select_k <= len(ckpt_paths):
            logger.error(
                "--ensemble-select %d is out of range for %d checkpoints",
                select_k, len(ckpt_paths),
            )
            return 1

    # Build eval loader
    eval_ds = ShardDataset(args.data_dir, split=split, shuffle=False,
                           archetype_self=args.archetype_self)
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )

    # Evaluate each member
    member_metrics = []
    for i, member in enumerate(ensemble.members):
        m = offline_eval(member, eval_loader, device, lambda_v=args.lambda_v,
                         belief=belief)
        member_metrics.append(m)
        logger.info(
            "Member %d: top1_macro=%.4f, top1_micro=%.4f, top1_nontrivial=%.4f, "
            "value_corr=%.4f, value_std=%.4f",
            i,
            m.get("val/top1_macro", 0.0),
            m.get("val/top1_micro", 0.0),
            m.get("val/top1_nontrivial", 0.0),
            m.get("val/value_corr", 0.0),
            m.get("val/value_std", 0.0),
        )

    # Evaluate ensemble
    ens_metrics = offline_eval(ensemble, eval_loader, device, lambda_v=args.lambda_v,
                               belief=belief)
    logger.info(
        "Ensemble: top1_macro=%.4f, top1_micro=%.4f, top1_nontrivial=%.4f, "
        "value_corr=%.4f, value_std=%.4f",
        ens_metrics.get("val/top1_macro", 0.0),
        ens_metrics.get("val/top1_micro", 0.0),
        ens_metrics.get("val/top1_nontrivial", 0.0),
        ens_metrics.get("val/value_corr", 0.0),
        ens_metrics.get("val/value_std", 0.0),
    )

    # Print comparison table
    header = (f"{'':>12} {'top1_macro':>12} {'top1_micro':>12} "
              f"{'top1_nontr':>12} {'val_corr':>10} {'val_std':>10}")
    print(f"\n{header}")
    print("-" * 70)
    for i, m in enumerate(member_metrics):
        print(f"Member {i:>5}: {m.get('val/top1_macro', 0):12.4f} "
              f"{m.get('val/top1_micro', 0):12.4f} "
              f"{m.get('val/top1_nontrivial', 0):12.4f} "
              f"{m.get('val/value_corr', 0):10.4f} "
              f"{m.get('val/value_std', 0):10.4f}")
    print(f"{'Ensemble':>12}: {ens_metrics.get('val/top1_macro', 0):12.4f} "
          f"{ens_metrics.get('val/top1_micro', 0):12.4f} "
          f"{ens_metrics.get('val/top1_nontrivial', 0):12.4f} "
          f"{ens_metrics.get('val/value_corr', 0):10.4f} "
          f"{ens_metrics.get('val/value_std', 0):10.4f}")

    # Find best single member by val top1_nontrivial
    best_idx = max(range(len(member_metrics)),
                   key=lambda i: member_metrics[i].get("val/top1_nontrivial", 0.0))
    best_top1 = member_metrics[best_idx].get("val/top1_nontrivial", 0.0)
    ens_top1 = ens_metrics.get("val/top1_nontrivial", 0.0)
    logger.info(
        "Best single member: %d (top1_nontrivial=%.4f), ensemble lift: %+.4f",
        best_idx, best_top1, ens_top1 - best_top1,
    )

    # Record baseline.  The per-member table rides along: these numbers were
    # just computed and printed, and discarding them is what forced a re-eval
    # every time somebody wanted to compare members.
    if getattr(args, "record_baseline", False):
        from ptcg_il.baselines import member_record, record_ensemble_baseline

        record_ensemble_baseline(
            args.data_dir, args.archetype_self, ckpt_paths,
            ens_metrics,
            members=[member_record(p, m)
                     for p, m in zip(ckpt_paths, member_metrics)],
        )
        logger.info("Recorded ensemble baseline (SHA-pinned to %d checkpoints)",
                    len(ckpt_paths))

    # Greedy subset selection (Kaggle caps submitted checkpoints)
    if select_k is not None:
        rc = _record_ensemble_selection(
            ensemble, ckpt_paths, member_metrics, eval_loader, device,
            select_k, split, artifacts, args,
        )
        if rc != 0:
            return rc

    # Live eval: ensemble vs. best single member head-to-head
    if args.live_eval:
        _run_ensemble_live_eval(ensemble, best_idx, artifacts, args)

    return 0


def _record_ensemble_selection(
    ensemble: Any,
    ckpt_paths: list[str],
    member_metrics: list[dict],
    eval_loader: Any,
    device: Any,
    select_k: int,
    split: str,
    artifacts: dict,
    args: argparse.Namespace,
) -> int:
    """Fit a greedy member ordering on *split*, then verify the top-K on test.

    Two artifacts come out of this, deliberately kept apart:  the ordering,
    recorded against the full member set and marked with the split it was fit
    on, and an ``ens-K`` baseline measured on **test** for the subset the
    ordering chose.  The second is the number worth quoting; the first is what
    ``build_submit.sh --ensemble-top`` reads.
    """
    import torch
    import torch.nn as nn
    from ptcg_il.baselines import (
        member_record, record_ensemble_baseline, record_ensemble_selection,
    )
    from ptcg_il.ensemble import EnsemblePolicy
    from ptcg_il.ensemble_select import (
        build_selection, collect_member_probs, greedy_order,
    )
    from ptcg_il.train.dataset import ShardDataset, collate_fn
    from ptcg_il.train.eval import offline_eval
    from torch.utils.data import DataLoader

    logger.info("Caching member probabilities for greedy selection (%s split)...", split)
    probs, targets = collect_member_probs(list(ensemble.members), eval_loader, device)
    logger.info(
        "Cached %d non-trivial single-select rows x %d members (%.1f MB)",
        probs.shape[1], probs.shape[0], probs.nbytes / 1e6,
    )

    order, scores = greedy_order(probs, targets)

    print(f"\nGreedy forward selection ({split} split, nontrivial_top1):")
    print(f"{'k':>3} {'added':>7} {'ensemble':>10} {'delta':>8}")
    print("-" * 32)
    for k, (idx, score) in enumerate(zip(order, scores), start=1):
        delta = score - scores[k - 2] if k > 1 else 0.0
        mark = "  <-- --ensemble-top" if k == select_k else ""
        print(f"{k:>3} {idx:>7} {score:>10.4f} {delta:>+8.4f}{mark}")
    best_k = int(max(range(len(scores)), key=lambda i: scores[i])) + 1
    if best_k != select_k:
        logger.info(
            "Greedy peaks at k=%d (%.4f) on %s, not the requested k=%d (%.4f)",
            best_k, scores[best_k - 1], split, select_k, scores[select_k - 1],
        )

    record_ensemble_selection(
        args.data_dir, args.archetype_self, ckpt_paths,
        [member_record(p, m) for p, m in zip(ckpt_paths, member_metrics)],
        build_selection(order, scores, ckpt_paths,
                        split=split, n_rows=int(probs.shape[1])),
    )

    # Verify the chosen subset on the held-out split.  Reusing the loaded
    # members rather than re-reading from disk: same weights, and
    # EnsemblePolicy holds nothing per-member that a subset would invalidate.
    chosen_idx = order[:select_k]
    chosen_paths = [ckpt_paths[i] for i in chosen_idx]
    logger.info("Verifying the top-%d subset on the test split: %s",
                select_k, ", ".join(str(i) for i in chosen_idx))

    try:
        test_ds = ShardDataset(args.data_dir, split="test", shuffle=False,
                               archetype_self=args.archetype_self)
    except (ValueError, FileNotFoundError) as e:
        logger.warning(
            "Selection recorded, but the test split is unavailable (%s) — no "
            "ens-%d baseline was written", e, select_k,
        )
        return 0

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )
    subset = EnsemblePolicy(nn.ModuleList([ensemble.members[i] for i in chosen_idx]))
    subset.to(device)
    subset.eval()
    subset_metrics = offline_eval(
        subset, test_loader, device, lambda_v=args.lambda_v,
        belief=_belief_weights(args) is not None,
    )
    logger.info(
        "Top-%d subset on test: top1_macro=%.4f, top1_micro=%.4f, "
        "top1_nontrivial=%.4f, value_corr=%.4f, value_std=%.4f",
        select_k,
        subset_metrics.get("val/top1_macro", 0.0),
        subset_metrics.get("val/top1_micro", 0.0),
        subset_metrics.get("val/top1_nontrivial", 0.0),
        subset_metrics.get("val/value_corr", 0.0),
        subset_metrics.get("val/value_std", 0.0),
    )
    record_ensemble_baseline(
        args.data_dir, args.archetype_self, chosen_paths, subset_metrics,
    )
    del subset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


def _run_ensemble_live_eval(
    ensemble: Any, best_idx: int, artifacts: dict, args: argparse.Namespace
) -> None:
    """Live eval head-to-head: ensemble vs. best single member."""
    from ptcg_il.live_eval import (
        LiveEvaluator, make_agent_from_policy,
    )

    # Build agents
    ensemble_agent = make_agent_from_policy(
        ensemble, artifacts["vocab"], artifacts["fixed_deck"], device="cpu",
        data_dir=args.data_dir,
    )
    best_member = ensemble.members[best_idx]
    best_single_agent = make_agent_from_policy(
        best_member, artifacts["vocab"], artifacts["fixed_deck"], device="cpu",
        data_dir=args.data_dir,
    )

    # Run head-to-head
    evaluator = LiveEvaluator(
        ensemble, artifacts["vocab"], artifacts["fixed_deck"],
        n_workers=args.live_eval_workers, data_dir=args.data_dir,
    )
    evaluator._agent_fn = ensemble_agent

    h2h_results = evaluator.eval_vs_opponent(
        best_single_agent, "best_single",
        n_games=args.live_eval_games,
    )
    logger.info(
        "Ensemble vs best single (member %d): win=%.1f%% [%.1f–%.1f%%], %d games",
        best_idx,
        h2h_results.win_rate_center * 100,
        h2h_results.win_rate_lo * 100,
        h2h_results.win_rate_hi * 100,
        h2h_results.n_games,
    )

    # Ship rule: Wilson lower bound must be above 50%
    if h2h_results.win_rate_lo > 0.50:
        logger.info(
            "Ensemble SHIPS: Wilson lower bound %.1f%% > 50%%",
            h2h_results.win_rate_lo * 100,
        )
    else:
        logger.warning(
            "Ensemble does NOT ship: Wilson lower bound %.1f%% ≤ 50%%. "
            "Train more members or tune config.",
            h2h_results.win_rate_lo * 100,
        )


def _run_live_eval(policy: Any, artifacts: dict, args: argparse.Namespace) -> None:
    """Run live-engine eval against all three baseline opponents.

    Opponents (per spec):
    - random: uniformly random valid option
    - frozen_ckpt: a previously-saved checkpoint (if --resume given)
    - search_planner: stub (v1 placeholder — uses first legal option)
    """
    from ptcg_il.live_eval import (
        LiveEvaluator,
        make_agent_from_policy,
        random_agent,
        search_planner_agent,
    )

    evaluator = LiveEvaluator(
        policy,
        artifacts["vocab"],
        artifacts["fixed_deck"],
        n_workers=args.live_eval_workers,
        data_dir=args.data_dir,
    )

    opponents: dict[str, Any] = {
        "random": random_agent,
        "search_planner": search_planner_agent,
    }

    # A second planner whose determinization uses the belief heads' predicted
    # opponent deck instead of the mirror assumption.  Kept alongside the plain
    # planner rather than replacing it, so the two numbers measure what the
    # belief model is actually worth.
    if _belief_weights(args) is not None:
        try:
            import copy

            from ptcg_il.belief_infer import OpponentDeckOracle
            from ptcg_il.live_eval import SearchPlannerAgent

            # A CPU copy: the workers are forked, and a CUDA tensor cannot
            # cross a fork.
            oracle_policy = copy.deepcopy(policy).to("cpu").eval()
            oracle = OpponentDeckOracle(
                oracle_policy, artifacts["vocab"], artifacts["archetypes"],
                device="cpu", data_dir=args.data_dir,
            )
            opponents["search_planner_belief"] = SearchPlannerAgent(oracle=oracle)
        except Exception as e:
            logger.warning("Could not build belief-backed search planner: %s", e)

    # Frozen checkpoint opponent — load from --resume if available
    if args.resume is not None:
        try:
            frozen_policy = _build_policy(artifacts, args)
            frozen_policy.eval()
            frozen_policy.to("cpu")
            from ptcg_il.train.checkpoint import load_checkpoint
            ckpt = load_checkpoint(args.resume, device="cpu")
            from ptcg_il.model.policy import load_policy_state
            load_policy_state(
                frozen_policy, ckpt["model_state_dict"],
                allow_belief_widening=getattr(args, "allow_belief_widening", False))
            frozen_agent = make_agent_from_policy(
                frozen_policy, artifacts["vocab"], artifacts["fixed_deck"],
                device="cpu", data_dir=args.data_dir,
            )
            opponents["frozen_ckpt"] = frozen_agent
            logger.info("Added frozen_ckpt opponent from %s (step %d)", args.resume, ckpt["step"])
        except Exception as e:
            logger.warning("Could not load frozen checkpoint opponent: %s", e)
    else:
        logger.info("No --resume checkpoint provided; skipping frozen_ckpt opponent")

    logger.info("Running live eval (%d games per opponent, %d opponents)...",
                args.live_eval_games, len(opponents))
    results = evaluator.eval_vs_opponents(opponents, n_games=args.live_eval_games)
    for name, r in results.items():
        # Surface failed games here too -- this is the line users actually read.
        log = logger.warning if r.errors else logger.info
        log(
            "Live eval vs %s: win=%.1f%% [%.1f–%.1f%%], %d/%d games (%d failed), "
            "%.1f mean steps, %d illegal actions, oov=%.3f",
            name,
            r.win_rate_center * 100,
            r.win_rate_lo * 100,
            r.win_rate_hi * 100,
            r.n_games,
            args.live_eval_games,
            r.errors,
            r.mean_game_length,
            r.total_illegal_actions,
            r.oov_rate,
        )
        if r.errors:
            logger.warning(
                "  ^ %d/%d games vs %s did not complete; treat this win rate as unreliable.",
                r.errors,
                args.live_eval_games,
                name,
            )


def cmd_archetypes(args: argparse.Namespace) -> int:
    """Print the archetype ids the pipeline should train specialists for.

    Writes to stdout only, so the pipeline can capture it directly:
    ``ARCHETYPES=$(uv run python -m ptcg_il.cli archetypes --top 2)``.
    Diagnostics go to stderr via the logger so they never pollute that capture.
    """
    from ptcg_il.archetype_select import describe, pick_archetypes

    if args.describe:
        print(describe(args.data_dir))
        return 0

    ids = pick_archetypes(args.data_dir, top=args.top)
    if not ids:
        logger.error(
            "No 𝒟_self archetype in %s has enough held-out data to train a "
            "specialist. Run `ptcg_il.cli archetypes --describe` to see the "
            "row counts.", args.data_dir,
        )
        return 1
    if len(ids) < args.top:
        logger.warning(
            "Only %d of the requested %d archetypes have enough held-out data; "
            "training %s", len(ids), args.top, ids,
        )
    print(" ".join(str(i) for i in ids))
    return 0


# ============================================================
# Entry point
# ============================================================


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return cmd_train(args)
    elif args.command == "build-shards":
        return cmd_build_shards(args)
    elif args.command == "archetypes":
        return cmd_archetypes(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
