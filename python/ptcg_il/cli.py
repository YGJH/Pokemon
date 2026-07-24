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
import os
import sys
from pathlib import Path
from typing import Any

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
    "batch_size": 2048,
    "epochs": 10,
    "peak_lr": 3e-4,
    "warmup": 1000,
    "min_lr": 3e-5,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "label_smoothing": 0.05,
    "ema_decay": 0.999,
    "alpha_ctx": 0.5,
    "alpha_arch": 0.5,
    "lambda_v": 0.5,
    "w_lost": 0.6,
    # Cadence
    "log_every": 50,
    "val_every": 1000,
    "ckpt_every": 2000,
    "live_every": 10000,
    # Data
    "num_workers": 8,
    # Precision
    "mixed_precision": True,
    # Patience
    "patience": 5,
    # Live eval
    "live_eval_games": 500,
    # W&B
    "wandb_project": "pokemon-tcg-il",
    "wandb_entity": None,
    "wandb_name": None,
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

    # Paths
    paths = train_parser.add_argument_group("Paths")
    paths.add_argument("--data-dir", type=str, default="data",
                       help="Directory with shards/, meta.parquet, vocab.json, archetypes.json")
    paths.add_argument("--out-dir", type=str, default="checkpoints",
                       help="Directory for checkpoints and logs")
    paths.add_argument("--resume", type=str, default=None,
                       help="Resume from a checkpoint file")
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
                          help="Decision points per step")
    train_hp.add_argument("--epochs", type=int, default=DEFAULTS["epochs"],
                          help="Training epochs (overridden by --total-steps)")
    train_hp.add_argument("--total-steps", type=int, default=None,
                          help="Override total training steps (default: epochs * steps_per_epoch)")
    train_hp.add_argument("--peak-lr", type=float, default=DEFAULTS["peak_lr"],
                          help="Peak learning rate")
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
        with open(vocab_path) as f:
            artifacts["vocab"] = json.load(f)
        artifacts["vocab_size"] = artifacts["vocab"].get("size", 0)
        artifacts["attack_size"] = len(artifacts["vocab"].get("attack_id_to_index", {})) + 1
    else:
        raise FileNotFoundError(f"vocab.json not found at {vocab_path}")

    arch_path = data_dir / "archetypes.json"
    if arch_path.exists():
        with open(arch_path) as f:
            arch = json.load(f)
        artifacts["fixed_deck"] = arch.get("FIXED_DECK", list(range(60)))
    else:
        raise FileNotFoundError(f"archetypes.json not found at {arch_path}")

    return artifacts


def _build_policy(artifacts: dict, args: argparse.Namespace) -> Any:
    """Create a Policy module from artifact sizes and CLI args."""
    import torch
    from ptcg_il.model.policy import Policy

    V = artifacts["vocab_size"]
    A = artifacts["attack_size"]

    # Try to load static tables from data_dir if available
    import numpy as np

    card_static = None
    attack_static = None
    card_table_path = Path(args.data_dir) / "card_static_table.npy"
    attack_table_path = Path(args.data_dir) / "attack_static_table.npy"
    if card_table_path.exists():
        card_static_np = np.load(card_table_path)
        card_static = torch.from_numpy(card_static_np).float()
    if attack_table_path.exists():
        attack_static_np = np.load(attack_table_path)
        attack_static = torch.from_numpy(attack_static_np).float()

    policy = Policy(
        V=V,
        A=A,
        D=args.d_model,
        heads=args.heads,
        layers=args.layers,
        ff=args.ff,
        card_static_table=card_static,
        attack_static_table=attack_static,
    )
    # Apply spec B.8 weight init (trunc_normal std=0.02 for Linear/Embedding weights)
    from ptcg_il.model import init_weights
    init_weights(policy)
    return policy


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
        "Building Policy(V=%d, A=%d, D=%d, heads=%d, layers=%d, ff=%d)",
        artifacts["vocab_size"],
        artifacts["attack_size"],
        args.d_model,
        args.heads,
        args.layers,
        args.ff,
    )
    policy = _build_policy(artifacts, args)

    # Eval-only mode
    if args.eval_only:
        if args.resume is None:
            logger.error("--eval-only requires --resume to specify a checkpoint")
            return 1
        return _cmd_eval_only(policy, artifacts, args)

    # Set W&B mode
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    elif args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

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
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
        warmup=args.warmup,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        ema_decay=args.ema_decay,
        lambda_v=args.lambda_v,
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
        wandb_name=args.wandb_name,
        resume_ckpt=args.resume,
        run_val=True,
        patience=args.patience,
    )

    # Live eval after training
    if args.live_eval:
        _run_live_eval(trained_policy, artifacts, args)

    logger.info("Training complete.  Best checkpoint at %s/ckpt-best.pt", out_dir)
    return 0


def _cmd_eval_only(policy: Any, artifacts: dict, args: argparse.Namespace) -> int:
    """Run offline eval + optional live eval from a checkpoint."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    from ptcg_il.train.checkpoint import load_checkpoint
    ckpt = load_checkpoint(args.resume, device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device)
    policy.eval()

    logger.info("Loaded checkpoint from step %d", ckpt["step"])

    # Offline eval
    from ptcg_il.train.dataset import ShardDataset, collate_fn
    from ptcg_il.train.eval import offline_eval
    from torch.utils.data import DataLoader

    try:
        val_ds = ShardDataset(args.data_dir, split="val", shuffle=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        eval_metrics = offline_eval(policy, val_loader, device, lambda_v=args.lambda_v)
        logger.info("Offline eval: top1_macro=%.4f, top1_micro=%.4f",
                    eval_metrics.get("val/top1_macro", 0.0),
                    eval_metrics.get("val/top1_micro", 0.0))
    except (ValueError, FileNotFoundError) as e:
        logger.warning("Cannot run offline eval: %s", e)

    # Live eval
    if args.live_eval:
        _run_live_eval(policy, artifacts, args)

    return 0


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
    )

    opponents: dict[str, Any] = {
        "random": random_agent,
        "search_planner": search_planner_agent,
    }

    # Frozen checkpoint opponent — load from --resume if available
    if args.resume is not None:
        try:
            frozen_policy = _build_policy(artifacts, args)
            frozen_policy.eval()
            frozen_policy.to("cpu")
            from ptcg_il.train.checkpoint import load_checkpoint
            ckpt = load_checkpoint(args.resume, device="cpu")
            frozen_policy.load_state_dict(ckpt["model_state_dict"])
            frozen_agent = make_agent_from_policy(
                frozen_policy, artifacts["vocab"], artifacts["fixed_deck"], device="cpu"
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
        logger.info(
            "Live eval vs %s: win=%.1f%% [%.1f–%.1f%%], %d games, %.1f mean steps, %d illegal actions, oov=%.3f",
            name,
            r.win_rate_center * 100,
            r.win_rate_lo * 100,
            r.win_rate_hi * 100,
            r.n_games,
            r.mean_game_length,
            r.total_illegal_actions,
            r.oov_rate,
        )


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
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
