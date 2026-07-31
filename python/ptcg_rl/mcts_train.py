"""AlphaZero-style MCTS self-play training on an IL-warmstarted policy.

Usage::

    cd python
    uv run python -m ptcg_rl.mcts_train \\
        --il-ckpt checkpoints_a0/ckpt-best.pt \\
        --data-dir data \\
        --out-dir checkpoints_a0_mcts

Flow:
1. Load IL checkpoint as θ_init
2. Self-play games: our deck vs sampled 𝒟_opp archetype (6 known)
3. ρ=5% of decisions get MCTS targets (π̃, Ṽ) — using the KNOWN opponent deck
4. Store (state, π̃, game outcome) in replay buffer
5. Train: CE to π̃ + MSE to outcome
6. Periodically evaluate against frozen θ_init
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(show_time=False)],
)
logger = logging.getLogger(__name__)


def fmt_dur(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptcg_rl.mcts_train",
        description="AlphaZero-style MCTS self-play training (Phase 3)",
    )
    p.add_argument("--il-ckpt", type=str, required=True,
                   help="Frozen IL policy as θ_init and evaluation baseline")
    p.add_argument("--data-dir", type=str, default="data",
                   help="Directory with vocab.json, archetypes.json, static tables")
    p.add_argument("--out-dir", type=str, default="checkpoints_mcts",
                   help="Where MCTS-trained checkpoints are written")
    p.add_argument("--deck-archetype", type=int, default=None,
                   help="Archetype id whose decklist θ plays.  Must match the "
                        "archetype --il-ckpt was trained on: the specialist "
                        "only ever saw that deck's cards.  Omitted = the "
                        "global fixed_deck from archetypes.json.")

    hp = p.add_argument_group("MCTS hyperparameters")
    hp.add_argument("--iterations", type=int, default=64,
                    help="PUCT iterations per tree")
    hp.add_argument("--c-puct", type=float, default=2.0,
                    help="PUCT exploration constant")
    hp.add_argument("--k-determinizations", type=int, default=4,
                    help="Determinizations per root (lower than training since opp deck is known)")
    hp.add_argument("--leaf-batch", type=int, default=512,
                    help="Max leaves per MCTS batch step")
    hp.add_argument("--rho", type=float, default=0.05,
                    help="Fraction of decisions tagged for MCTS")
    hp.add_argument("--mcts-distill", action="store_true", default=False,
                    help="Enable MCTS PUCT search distillation (ρ=5%% of decisions)")
    hp.add_argument("--mcts-enabled", action="store_true", default=False,
                    help=argparse.SUPPRESS)  # alias for --mcts-distill
    hp.add_argument("--n-engines", type=int, default=8,
                    help="libcg instances for MCTS forest")
    hp.add_argument("--n-workers", type=int, default=12,
                    help="RolloutPool parallel workers (libcg battles)")
    hp.add_argument("--forward-batch", type=int, default=512,
                    help="GPU forward batch size for rollout + MCTS")

    rp = p.add_argument_group("Replay buffer")
    rp.add_argument("--buffer-capacity", type=int, default=100_000,
                    help="Max decision points in replay buffer")
    rp.add_argument("--min-buffer", type=int, default=10_000,
                    help="Start training only after this many samples")

    tp = p.add_argument_group("Training")
    tp.add_argument("--total-games", type=int, default=2_000,
                    help="Total self-play games to run")
    tp.add_argument("--games-per-iter", type=int, default=50,
                    help="Self-play games per iteration before training")
    tp.add_argument("--train-steps-per-iter", type=int, default=200,
                    help="Training steps per iteration")
    tp.add_argument("--batch-size", type=int, default=256)
    tp.add_argument("--all-archetypes", action="store_true", default=False,
                    help="Train against ALL 179 archetypes (default: 6 𝒟_opp only).")
    tp.add_argument("--lr", type=float, default=1e-4,
                    help="Peak learning rate")
    tp.add_argument("--min-lr", type=float, default=1e-5,
                    help="Minimum learning rate (cosine floor)")
    tp.add_argument("--warmup-steps", type=int, default=500,
                    help="Linear warmup steps at start of training")
    tp.add_argument("--lr-schedule", type=str, default="cosine",
                    choices=["cosine", "constant"],
                    help="LR schedule: cosine decay or constant")
    tp.add_argument("--c-value", type=float, default=0.5,
                    help="Value loss weight")
    tp.add_argument("--c-pi", type=float, default=1.0,
                    help="Policy (CE to π̃) loss weight")
    tp.add_argument("--grad-clip", type=float, default=1.0)

    ep = p.add_argument_group("Evaluation & League gate")
    ep.add_argument("--eval-games", type=int, default=100,
                    help="Games per opponent in each league evaluation")
    ep.add_argument("--eval-every-games", type=int, default=250,
                    help="Evaluate every N self-play games")
    ep.add_argument("--gate-score", type=float, default=0.70,
                    help="Win rate required against EVERY past champion to save")
    ep.add_argument("--max-champions", type=int, default=10,
                    help="Max past champion checkpoints to keep and test against")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from an MCTS checkpoint .pt file")

    wb = p.add_argument_group("W&B logging")
    wb.add_argument("--wandb", action="store_true", default=False,
                    help="Log to Weights & Biases")
    wb.add_argument("--wandb-project", type=str, default="pokemon-tcg-mcts")
    wb.add_argument("--wandb-entity", type=str, default="poken")
    wb.add_argument("--wandb-name", type=str, default=None)
    wb.add_argument("--wandb-mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])
    wb.add_argument("--no-wandb", action="store_true",
                    help="Disable wandb (overrides --wandb)")
    return p


# CLI dest → the attribute name the MCTS code paths actually read.  The
# parser spells these flags without the `mcts_` prefix (`--rho`,
# `--iterations`, …) while `run_self_play_games` and `ptcg_rl.search` read
# `config.mcts_rho`, `config.mcts_iterations`, … off the same Namespace.
# Every one of those reads is a `getattr(..., default)`, so an unmapped flag
# silently took the default instead of raising.
_MCTS_ARG_ALIASES = {
    "rho": "mcts_rho",
    "iterations": "mcts_iterations",
    "c_puct": "mcts_c_puct",
    "k_determinizations": "mcts_k_determinizations",
    "leaf_batch": "mcts_leaf_batch",
    "n_engines": "mcts_n_engines",
}


def normalize_mcts_args(args: argparse.Namespace) -> argparse.Namespace:
    """Mirror the MCTS CLI flags onto the ``mcts_*`` names the code reads.

    Mutates and returns *args*.  Also unifies ``--mcts-distill`` with its
    hidden ``--mcts-enabled`` alias: ``mcts_train`` gates on the former,
    ``ptcg_rl.search`` on the latter, so either flag must set both.
    """
    for src, dst in _MCTS_ARG_ALIASES.items():
        if hasattr(args, src):
            setattr(args, dst, getattr(args, src))
    enabled = bool(getattr(args, "mcts_distill", False)) or \
        bool(getattr(args, "mcts_enabled", False))
    args.mcts_distill = enabled
    args.mcts_enabled = enabled
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The only supported way to build this CLI's config — parsing without
    the alias pass yields a Namespace on which every ``mcts_*`` read falls
    back to its default."""
    return normalize_mcts_args(build_parser().parse_args(argv))


def _load_vocab(data_dir: Path) -> dict:
    from ptcg_il.featurizer import normalize_vocab
    with open(data_dir / "vocab.json") as f:
        return normalize_vocab(json.load(f))


def _load_archetypes(data_dir: Path) -> dict:
    with open(data_dir / "archetypes.json") as f:
        return json.load(f)


def _static_tables(data_dir: Path) -> dict:
    out: dict[str, Any] = {}
    for key, fname in (("card_static_table", "card_static_table.npy"),
                       ("attack_static_table", "attack_static_table.npy")):
        path = data_dir / fname
        out[key] = torch.from_numpy(np.load(path)).float() if path.exists() else None
    return out


def _load_all_card_feat(data_dir: Path) -> torch.Tensor | None:
    """Load [n_all_cards, F_CARD] feature matrix for BeliefHeads."""
    from ptcg_il.model.cards import F_CARD

    ecf_path = data_dir / "engine_card_features.npy"
    if not ecf_path.exists():
        return None
    import numpy as _np
    ecf = _np.load(ecf_path, allow_pickle=True).item()
    if not ecf:
        return None
    max_id = max(ecf.keys())
    all_feat = torch.zeros(max_id + 1, F_CARD)
    for cid, feat in ecf.items():
        all_feat[int(cid)] = torch.from_numpy(_np.asarray(feat, dtype=_np.float32))
    return all_feat


def _load_policy(ckpt_path: str, data_dir: Path, device: torch.device) -> Any:
    """Load policy from a checkpoint (pure-feature format).

    Loads ``all_card_feat`` from the data directory for belief heads.
    """
    from ptcg_il.model.policy import load_policy_state, policy_from_config
    from ptcg_il.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(ckpt_path, device="cpu")
    config = ckpt.get("config") or {}
    if not config:
        raise ValueError(f"{ckpt_path} has no 'config' record")

    all_card_feat = _load_all_card_feat(data_dir)

    policy = policy_from_config(config, all_card_feat=all_card_feat)
    model_sd = policy.state_dict()
    ckpt_sd = ckpt["model_state_dict"]
    try:
        load_policy_state(policy, ckpt_sd)
    except RuntimeError:
        _load_lenient(policy, model_sd, ckpt_sd)

    policy.to(device)
    return policy


def _load_lenient(policy: Any, model_sd: dict, ckpt_sd: dict) -> None:
    """Load *ckpt_sd* into *policy*, handling size-mismatched ``*.static`` params.

    For any parameter whose checkpoint shape differs from the model, only
    the overlapping prefix is copied.  Everything else is loaded as-is.
    Missing belief-head keys are tolerated (pre-belief checkpoints).
    """
    import torch

    loaded = 0
    skipped = 0
    for key, model_param in model_sd.items():
        ckpt_param = ckpt_sd.get(key)
        if ckpt_param is None:
            if key.startswith("belief_heads."):
                skipped += 1
                continue
            logger.warning("  _load_lenient: missing key %s", key)
            skipped += 1
            continue
        if ckpt_param.shape != model_param.shape:
            # Copy overlapping prefix (handles vocab growth)
            slices = tuple(
                slice(0, min(cs, ms))
                for cs, ms in zip(ckpt_param.shape, model_param.shape)
            )
            model_param[slices].copy_(ckpt_param[slices])
            loaded += 1
            logger.debug("  _load_lenient: %s %s → %s (partial)",
                         key, list(ckpt_param.shape), list(model_param.shape))
        else:
            model_param.copy_(ckpt_param)
            loaded += 1
    logger.info(
        "  Lenient load: %d params loaded (%d skipped) from old checkpoint",
        loaded, skipped,
    )


def _load_frozen_anchor(ckpt_path: str, data_dir: Path, device: torch.device) -> Any:
    """Load a frozen copy for evaluation."""
    from ptcg_il.model.policy import policy_from_config
    from ptcg_il.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(ckpt_path, device="cpu")
    config = ckpt.get("config") or {}
    config = dict(config)
    all_card_feat = _load_all_card_feat(data_dir)
    policy = policy_from_config(config, all_card_feat=all_card_feat)
    _load_lenient(policy, policy.state_dict(), ckpt["model_state_dict"])
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def _fixed_deck(data_dir: Path) -> list[int]:
    with open(data_dir / "archetypes.json") as f:
        return [int(c) for c in json.load(f)["fixed_deck"]]


def _deck_for(data_dir: Path, deck_archetype: int | None) -> list[int]:
    """The 60-card decklist θ plays.

    ``archetypes.json``'s ``fixed_deck`` is one specific archetype's
    representative (the best 𝒟_self one), so using it for *every* specialist
    would hand, say, the a0 model a deck it has never seen.  That fails
    silently — cards outside the model's vocab just become all-zero feature
    rows — which is why the deck has to follow the checkpoint, exactly as
    ``ptcg_rl.train`` does it.

    ``None`` keeps the historical behaviour (the global ``fixed_deck``).
    """
    if deck_archetype is None:
        return _fixed_deck(data_dir)
    from ptcg_il.deck import build_deck_metadata

    deck = build_deck_metadata(data_dir, deck_archetype)
    decklist = deck.get("deck")
    if not isinstance(decklist, list) or not decklist:
        raise ValueError(
            f"archetype {deck_archetype} produced no decklist from {data_dir}"
        )
    return [int(c) for c in decklist]


def _check_ckpt_deck(il_ckpt: str, deck_archetype: int | None) -> None:
    """Fail loudly when --il-ckpt was trained on a different archetype.

    ``save_checkpoint`` stamps the deck record precisely because this pairing
    is otherwise invisible at runtime.
    """
    if deck_archetype is None:
        return
    try:
        blob = torch.load(il_ckpt, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 — the real load downstream reports properly
        return
    record = blob.get("deck") if isinstance(blob, dict) else None
    if not isinstance(record, dict):
        logger.warning(
            "%s carries no deck record — cannot verify it was trained on "
            "archetype %d", il_ckpt, deck_archetype,
        )
        return
    trained_on = record.get("archetype_self")
    if trained_on is not None and int(trained_on) != int(deck_archetype):
        raise SystemExit(
            f"--deck-archetype {deck_archetype} does not match --il-ckpt "
            f"{il_ckpt}, which was trained on archetype {trained_on}.  The "
            f"specialist has never seen this deck's cards; training it here "
            f"would silently feed it all-zero card features."
        )


def _discover_il_checkpoints(il_ckpt_path: str) -> list[tuple[str, str]]:
    """Discover all .pt checkpoints in the same directory as *il_ckpt_path*.

    Returns a list of (name, path) tuples.  Step checkpoints are sorted by
    step number; ``ckpt-best`` (if present) comes last.
    """
    ckpt_dir = Path(il_ckpt_path).resolve().parent
    step_ckpts = sorted(
        ckpt_dir.glob("ckpt-step-*.pt"),
        key=lambda p: int(p.stem.rsplit("-", 1)[-1]),
    )
    result: list[tuple[str, str]] = []
    for p in step_ckpts:
        result.append((p.stem, str(p)))
    best = ckpt_dir / "ckpt-best.pt"
    if best.exists():
        result.append(("ckpt-best", str(best)))
    return result




def run_self_play_games(
    policy: Any,
    vocab: dict,
    archetypes: dict,
    fixed_deck: list[int],
    opp_decks: list[dict],
    config: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    n_games: int,
    opp_weights: dict[int, float] | None = None,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
) -> tuple[list[dict], dict]:
    """Run *n_games* self-play games.  Returns (samples, timing_stats).

    If *opp_weights* is given, samples opponents with those relative
    weights (low win rate → high weight).  If None, uses uniform
    weights.  *opp_weights* is updated in-place with per-opponent
    win/loss stats.
    """
    from ptcg_rl.rollout import PolicyActor
    from ptcg_rl.rust_vec_env import RustVecEnv

    all_samples: list[dict] = []
    wins, losses, draws = 0, 0, 0
    total_decisions = 0
    total_mcts_decisions = 0
    opp_counts: dict[int, int] = {}
    opp_wins: dict[int, int] = {}
    opp_losses: dict[int, int] = {}

    # ── Sample opponents with adaptive weights ───────────────────────
    games: list[dict] = []
    for g in range(n_games):
        opp = _sample_weighted(opp_decks, rng, opp_weights)
        opp_counts[opp["id"]] = opp_counts.get(opp["id"], 0) + 1
        games.append({"opp": opp, "seed": config.seed + g})

    # Group by opponent id
    from collections import defaultdict
    by_opp: dict[int, list] = defaultdict(list)
    for g in games:
        by_opp[g["opp"]["id"]].append(g)

    t_start = time.perf_counter()
    game_i = 0
    n_workers = getattr(config, "n_workers", 12)
    perf_t_feat = 0.0; perf_t_fwd = 0.0; perf_n_actor = 0
    perf_t_sweep = 0.0; perf_t_act = 0.0; perf_n_act = 0

    # ── Progress bar (TTY only; falls back to text logging) ──────────
    import sys as _sys
    _is_tty = _sys.stderr.isatty()
    if _is_tty:
        from rich.progress import (
            BarColumn, Progress, TaskProgressColumn, TextColumn,
            TimeElapsedColumn,
        )
        pbar = Progress(
            TextColumn("  [bold]{task.description}[/]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("• wr={task.fields[wr]:.2f}"),
            TextColumn("• {task.fields[rate]:.1f} g/min"),
            TextColumn("• {task.fields[dec_per]:.0f} d/g"),
            TimeElapsedColumn(),
        )
        task = pbar.add_task(
            f"self-play {n_games}g", total=n_games,
            wr=0.0, rate=0.0, dec_per=0.0,
        )
        pbar.start()
    else:
        pbar = None
        task = None

    for opp_id, group in sorted(by_opp.items()):
        opp = group[0]["opp"]
        opp_decklist = opp["deck"]
        n_group = len(group)

        actor = PolicyActor(policy, vocab, device=str(device), bf16=True,
                            seed=group[0]["seed"],
                            engine_card_features=engine_card_features,
                            engine_attack_features=engine_attack_features)
        policy.eval()

        with RustVecEnv(
            deck_self=fixed_deck,
            deck_opp=opp_decklist,
            n_envs=n_workers,
            our_player=0,
            seed=group[0]["seed"],
        ) as pool:
            # Drain n_group complete games.  Loop because a single
            # collect() call may return fewer games than needed if the
            # decision budget runs out before all games finish.
            remaining = n_group
            while remaining > 0 and game_i < n_games:
                budget = remaining * 300  # ~200 dec/game + headroom
                trajectories = pool.collect_batch(actor, budget)
                if not trajectories:
                    break
                for traj in trajectories:
                    if remaining <= 0 or game_i >= n_games:
                        break
                    remaining -= 1
                    game_i += 1
                    t_game = time.perf_counter()
                    outcome = float(traj.reward)
                    decisions = traj.decisions
                    seed_for = group[n_group - remaining - 1]["seed"]

                    if traj.error:
                        logger.warning(
                            "  game %d ERROR opp=%d: %s",
                            game_i, opp_id, traj.error,
                        )
                    if len(decisions) < 10:
                        logger.warning(
                            "  game %d SHORT: %d dec outcome=%.0f "
                            "opp=%d timeout=%s",
                            game_i, len(decisions), outcome, opp_id,
                            traj.timeout,
                        )

                    if not decisions:
                        continue

                    if outcome > 0:
                        wins += 1
                        opp_wins[opp_id] = opp_wins.get(opp_id, 0) + 1
                    elif outcome < 0:
                        losses += 1
                        opp_losses[opp_id] = opp_losses.get(opp_id, 0) + 1
                    else:
                        draws += 1

                    total_decisions += len(decisions)

                    n = len(decisions)
                    n_tag = max(1, int(n * getattr(config, "mcts_rho", 0.05)))
                    local_rng = np.random.default_rng(seed_for)
                    tagged_idx = set(
                        local_rng.choice(n, size=min(n_tag, n), replace=False).tolist()
                    )
                    mcts_targets: dict[int, tuple] = {}
                    if tagged_idx and getattr(config, "mcts_distill", False):
                        mcts_targets = _run_mcts_on_decisions_pooled(
                            decisions, tagged_idx, policy, vocab,
                            fixed_deck, opp_decklist, config, device, seed_for,
                        )
                    total_mcts_decisions += len(mcts_targets)

                    for i, dec in enumerate(decisions):
                        pi, v = mcts_targets.get(i, (None, None))
                        all_samples.append({
                            "features": dec.features,
                            "action_idx": dec.action_idx,
                            "action_len": dec.action_len,
                            "logp": dec.logp,
                            "value": dec.value,
                            "mcts_pi": pi,
                            "mcts_value": v,
                            "game_outcome": outcome,
                            "opp_archetype": opp_id,
                        })

                    t_game = time.perf_counter() - t_game

                    elapsed = time.perf_counter() - t_start
                    if pbar is not None:
                        pbar.update(task, completed=game_i,
                                    wr=wins / max(game_i, 1),
                                    rate=game_i / elapsed * 60 if elapsed > 0 else 0,
                                    dec_per=total_decisions / max(game_i, 1))
                    elif game_i % 10 == 0:
                        logger.info(
                            "  game %d/%d | wr=%.2f %dW/%dL/%dD | "
                            "%.0f dec/game | %.1f g/min",
                            game_i, n_games,
                            wins / max(game_i, 1), wins, losses, draws,
                            total_decisions / max(game_i, 1),
                            game_i / elapsed * 60 if elapsed > 0 else 0,
                        )

            # Accumulate perf from this pool's actor
            if hasattr(actor, "perf_n_calls") and actor.perf_n_calls > 0:
                perf_t_feat += actor.perf_t_featurize
                perf_t_fwd += actor.perf_t_forward
                perf_n_actor += actor.perf_n_calls
            if hasattr(pool, "_perf_n_act") and pool._perf_n_act > 0:
                perf_t_sweep += pool._perf_t_sweep
                perf_t_act += pool._perf_t_act
                perf_n_act += pool._perf_n_act

    # ── Update adaptive weights ──────────────────────────────────────
    if opp_weights is not None:
        for d in opp_decks:
            oid = d["id"]
            w_total = opp_wins.get(oid, 0) + opp_losses.get(oid, 0)
            if w_total > 0:
                wr = opp_wins.get(oid, 0) / w_total
                # Weight ∝ (1 - win_rate) + ε → hard opponents get more samples
                opp_weights[oid] = max(0.05, 1.0 - wr)
            else:
                opp_weights[oid] = 1.0  # unknown opponent → prioritize
        # Normalize
        total_w = sum(opp_weights.values())
        if total_w > 0:
            for oid in opp_weights:
                opp_weights[oid] /= total_w

    if pbar is not None:
        pbar.stop()
    elapsed = time.perf_counter() - t_start
    perf_summary = {
        "t_featurize": perf_t_feat, "t_forward": perf_t_fwd,
        "n_actor_calls": perf_n_actor,
        "t_sweep": perf_t_sweep, "t_act": perf_t_act,
        "n_act_calls": perf_n_act,
        "t_other": elapsed - perf_t_feat - perf_t_fwd - perf_t_sweep,
    }
    if perf_n_actor > 0:
        logger.info(
            "PERF │ feat=%.1fs (%.1fms/×%d) | fwd=%.1fs (%.1fms/×%d) | "
            "sweep=%.1fs | act=%.1fs | other=%.1fs",
            perf_t_feat, perf_t_feat / perf_n_actor * 1000, perf_n_actor,
            perf_t_fwd, perf_t_fwd / perf_n_actor * 1000, perf_n_actor,
            perf_t_sweep, perf_t_act,
            elapsed - perf_t_feat - perf_t_fwd - perf_t_sweep,
        )

    return all_samples, {
        "sp_games": n_games,
        "sp_sec": elapsed,
        "sp_games_per_min": n_games / elapsed * 60 if elapsed > 0 else 0,
        "sp_decisions": total_decisions,
        "sp_mcts_decisions": total_mcts_decisions,
        "sp_dec_per_game": total_decisions / max(n_games, 1),
        "sp_win_rate": wins / max(n_games, 1),
        "sp_wins": wins, "sp_losses": losses, "sp_draws": draws,
        "sp_opp_distribution": opp_counts,
        "perf": perf_summary,
    }


def _run_mcts_on_decisions_pooled(
    decisions: list,
    tagged_idx: set,
    policy: Any,
    vocab: dict,
    fixed_deck: list[int],
    opp_deck: list[int],
    config: Any,
    device: Any,
    seed: int,
) -> dict[int, tuple]:
    """Same as _run_mcts_on_decisions but shares the MctsForest across calls."""
    import json as _json
    from ptcg_rl.search import MctsForest, batch_evaluate_leaves

    forest = MctsForest(
        n_engines=getattr(config, "mcts_n_engines", 4),
        libcg_path=None,
    )
    tree_map: dict[int, tuple[int, int]] = {}
    K = getattr(config, "mcts_k_determinizations", 4)
    n_attempted = 0
    n_rejected = 0

    try:
        for i in tagged_idx:
            dec = decisions[i]
            obs_json = getattr(dec, "obs_json", None)
            if not obs_json:
                continue
            obs_dict = _json.loads(obs_json)
            if not obs_dict.get("search_begin_input"):
                # libcg's SearchBegin dereferences the pointer unconditionally;
                # an empty search_begin_input takes down the whole process, so
                # drop the root here rather than learn about it from a SIGSEGV.
                logger.warning(
                    "decision %d has no search_begin_input — skipping MCTS root", i,
                )
                continue
            for k in range(K):
                tree_id = forest.add_root(
                    obs_dict, fixed_deck,
                    opp_deck_template=opp_deck,
                    iterations=getattr(config, "mcts_iterations", 64),
                    c_puct=getattr(config, "mcts_c_puct", 2.0),
                    seed=seed + i * 1000 + k,
                )
                n_attempted += 1
                if tree_id >= 0:
                    tree_map[tree_id] = (i, k)
                else:
                    n_rejected += 1

        # A handful of engine-refused roots per batch is normal; a wholesale
        # refusal means MCTS is silently contributing nothing, which is
        # exactly the failure mode this counter exists to make visible.
        if n_rejected:
            log = logger.error if n_rejected == n_attempted else logger.debug
            log("MCTS: %d/%d roots refused by the engine", n_rejected, n_attempted)

        if not tree_map:
            return {}

        leaf_batch = getattr(config, "mcts_leaf_batch", 512)
        while True:
            leaves = forest.select_batch(leaf_batch)
            if not leaves:
                break
            expansions = batch_evaluate_leaves(leaves, policy, vocab, device)
            forest.expand_batch(expansions)

        results = {r["tree_id"]: r for r in forest.results()}
    finally:
        forest.close()

    # Aggregate K determinizations
    targets: dict[int, tuple] = {}
    state_results: dict[int, list[dict]] = {}
    for tree_id, result in results.items():
        di, _k = tree_map[tree_id]
        state_results.setdefault(di, []).append(result)

    for di, k_results in state_results.items():
        agg_visits: dict[int, float] = {}
        agg_value = 0.0
        n_value = 0
        n_opts = 0
        for r in k_results:
            for opt, visits in r.get("visit_counts", []):
                agg_visits[opt] = agg_visits.get(opt, 0.0) + visits
                n_opts = max(n_opts, opt + 1)
            rv = r.get("root_value")
            if rv is not None:
                agg_value += rv
                n_value += 1
        if n_opts > 0 and agg_visits:
            total = sum(agg_visits.values())
            pi = np.zeros(n_opts, dtype=np.float32)
            for opt, v in agg_visits.items():
                if opt < n_opts and total > 0:
                    pi[opt] = float(v) / total
        else:
            pi = np.array([], dtype=np.float32)
        root_v = (agg_value / max(n_value, 1)) if n_value > 0 else None
        targets[di] = (pi, root_v)

    return targets


def _eval_game_batch(
    theta: Any,
    opponent: Any,
    vocab: dict,
    fixed_deck: list[int],
    opp_deck: list[int],
    our_player: int,
    n_games: int,
    config: argparse.Namespace,
    device: torch.device,
    seed: int,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
) -> tuple[int, int, int]:
    """Play *n_games* evaluation games.  Returns (wins, losses, draws).

    Uses RustVecEnv (no multiprocessing).  theta plays for our_player,
    opponent plays for the other side.  Both use greedy inference.
    """
    from ptcg_rl.rust_vec_env import RustVecEnv
    from ptcg_rl.rollout import PolicyActor

    theta.eval()
    opponent.eval()
    actor_ours = PolicyActor(theta, vocab, device=str(device), greedy=True, seed=seed,
                              engine_card_features=engine_card_features,
                              engine_attack_features=engine_attack_features)
    actor_theirs = PolicyActor(opponent, vocab, device=str(device), greedy=True, seed=seed + 1,
                                engine_card_features=engine_card_features,
                                engine_attack_features=engine_attack_features)

    wins, losses, draws = 0, 0, 0
    n_done = 0

    with RustVecEnv(
        deck_self=fixed_deck, deck_opp=opp_deck,
        n_envs=4, our_player=our_player, seed=seed,
    ) as env:
        while n_done < n_games:
            pending = env.poll()
            if not pending:
                for t in env.drain():
                    if t.reward > 0:
                        wins += 1
                    elif t.reward < 0:
                        losses += 1
                    else:
                        draws += 1
                    n_done += 1
                if not pending:
                    time.sleep(0.001)
                    continue

            # Route: our player uses theta, theirs uses opponent
            picks_list = []
            ours_reqs = []
            theirs_reqs = []
            ours_idx = []
            theirs_idx = []
            for pi, p in enumerate(pending):
                obs = json.loads(p["obs_json"])
                if p["select_player"] == our_player:
                    ours_reqs.append({"obs": obs, "actor": "ours"})
                    ours_idx.append(pi)
                else:
                    theirs_reqs.append({"obs": obs, "actor": "theirs"})
                    theirs_idx.append(pi)

            # Build full reply list
            replies = [None] * len(pending)
            if ours_reqs:
                for ri, rep in enumerate(actor_ours(ours_reqs)):
                    replies[ours_idx[ri]] = rep
            if theirs_reqs:
                for ri, rep in enumerate(actor_theirs(theirs_reqs)):
                    replies[theirs_idx[ri]] = rep

            for rep in replies:
                if rep is not None:
                    picks_list.append(rep["picks"])
            if picks_list:
                env.reply(picks_list)

            for t in env.drain():
                if t.reward > 0:
                    wins += 1
                elif t.reward < 0:
                    losses += 1
                else:
                    draws += 1
                n_done += 1

    return wins, losses, draws


def league_evaluate(
    theta: Any,
    opponents: list[tuple[str, Any]],
    vocab: dict,
    fixed_deck: list[int],
    opp_decks: list[dict],
    config: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
) -> dict[str, Any]:
    """League evaluation: θ vs every opponent, balanced seats.
    Uses RustVecEnv — no multiprocessing, no broken pipes."""
    import json as _json

    results: dict[str, dict] = {}
    all_passed = True

    for name, opp in opponents:
        # Load champion on CPU first, then move to GPU
        if isinstance(opp, str):
            opp = _load_policy(opp, Path(config.data_dir if hasattr(config, 'data_dir') else "data"),
                               torch.device("cpu"))
        opp.to(device)
        opp.eval()

        wins, losses, draws = 0, 0, 0
        total_games = 0

        for seat in (0, 1):
            opp_deck = opp_decks[rng.integers(0, len(opp_decks))]["deck"]
            per_seat = max(1, config.eval_games // 2)
            w, l, d = _eval_game_batch(
                theta, opp, vocab, fixed_deck, opp_deck,
                our_player=seat, n_games=per_seat,
                config=config, device=device,
                seed=config.seed + 10000 * seat,
                engine_card_features=engine_card_features,
                engine_attack_features=engine_attack_features,
            )
            wins += w
            losses += l
            draws += d
            total_games += per_seat

        # Move opponent back to CPU to free GPU memory (except frozen_il)
        if name != "IL_baseline":
            opp.to("cpu")

        win_rate = wins / max(total_games, 1)
        passed = win_rate >= config.gate_score
        if not passed:
            all_passed = False

        results[name] = {
            "win_rate": win_rate, "wins": wins, "losses": losses,
            "draws": draws, "total": total_games, "passed": passed,
        }
        logger.info("  vs %-40s  wr=%.3f  %dW/%dL/%dD  %s",
                    name[:40], win_rate, wins, losses, draws,
                    "PASS" if passed else "FAIL")

    return {"passed": all_passed, "results": results}


def _manage_champions(
    out_dir: Path,
    policy: Any,
    current_ckpts: list[Path],
    total_games: int,
    win_rate: float,
    max_champions: int,
    elo: EloTracker | None = None,
    deck_record: dict | None = None,
) -> list[Path]:
    """Save a new champion and prune by ELO if over capacity.

    When the pool exceeds *max_champions*, the bottom half (by ELO rating)
    is dropped to bound per-round evaluation time.  Files stay on disk for
    auditing; only the active league list shrinks.
    """
    tag = f"champion-{total_games:06d}"
    _save_ckpt(out_dir, policy, tag, total_games, win_rate, deck_record=deck_record)
    new_path = out_dir / f"ckpt-mcts-{tag}.pt"
    current_ckpts.append(new_path)

    if len(current_ckpts) > max_champions:
        n_drop = min(4, len(current_ckpts) - 1)  # remove 4 lowest, keep ≥1
        # Rank by ELO (lowest first), drop n_drop
        if elo is not None:
            current_ckpts.sort(
                key=lambda p: elo.ratings.get(p.stem, elo.initial)
            )
        else:
            current_ckpts.sort(key=lambda p: p.stat().st_mtime)
        removed = current_ckpts[:n_drop]
        current_ckpts = current_ckpts[n_drop:]
        for r in removed:
            logger.info(
                "  Dropped from league (ELO %.0f): %s",
                elo.ratings.get(r.stem, 0) if elo else 0, r.name,
            )

    return current_ckpts


def train_step(
    policy: Any,
    optimizer: torch.optim.Optimizer,
    batch: list[dict],
    config: argparse.Namespace,
    device: torch.device,
    frozen_il: Any = None,
) -> dict[str, float]:
    """One training step from a sampled batch."""
    import torch.nn.functional as F

    from ptcg_rl.search import _collate_feat_list

    if not batch:
        return {"loss": 0.0}

    policy.train()  # GRU backward needs cuDNN training mode

    feat_list = [s["features"] for s in batch]
    tensor_batch = _collate_feat_list(feat_list, device)

    h, _history_h = policy._encode(tensor_batch)
    logits, _ = policy.pointer(h, tensor_batch["tok_mask"],
                                policy.embed.card, tensor_batch)
    values = policy.value(h[:, 0])  # [B]

    # ── KL anchor to frozen IL ────────────────────────────────────────
    kl_sum = torch.zeros((), device=device)
    n_kl = 0
    if frozen_il is not None:
        frozen_il.to(device)
        with torch.no_grad():
            from ptcg_rl.actor import recompute_logp
            logp_il, _ = recompute_logp(frozen_il, tensor_batch)
            logp_theta, _ = recompute_logp(policy, tensor_batch, encoded=h)
            from ptcg_rl.actor import kl_to_reference
            kl = kl_to_reference(logp_theta, logp_il)  # [B]
            kl_sum = kl.sum()
            n_kl = kl.shape[0]

    total_loss = torch.zeros((), device=device)
    loss_pi = torch.zeros((), device=device)
    loss_v = torch.zeros((), device=device)
    n_pi = 0

    rets: list[float] = []
    preds: list[float] = []
    for i, sample in enumerate(batch):
        outcome = torch.tensor(sample["game_outcome"], device=device)
        v_loss = F.mse_loss(values[i], outcome)
        loss_v = loss_v + v_loss
        rets.append(float(sample["game_outcome"]))
        preds.append(float(values[i].detach()))

        mcts_pi = sample.get("mcts_pi")
        if mcts_pi is not None and len(mcts_pi) > 0:
            # MCTS distillation: CE to visit distribution π̃
            pi_tilde = torch.as_tensor(
                np.asarray(mcts_pi, dtype=np.float32)
            ).to(device)
            mask = tensor_batch["opt_mask"][i].bool()
            masked = logits[i].float().masked_fill(~mask, float("-inf"))
            logp = F.log_softmax(masked, dim=-1)
            n_legal = min(len(pi_tilde), int(mask.sum()))
            if n_legal > 0:
                loss_pi = loss_pi - (pi_tilde[:n_legal] * logp[:n_legal]).sum()
                n_pi += 1
        else:
            # Behavior cloning: maximize joint log-prob of the full AR
            # sequence (including STOP).  recompute_logp from actor.py
            # handles the GRU loop correctly for multi-select.
            if (sample.get("action_idx") is not None
                    and sample.get("action_len", 0) > 0):
                from ptcg_rl.actor import recompute_logp
                # Build a batch of size 1 for this sample
                x_i = {k: v[i:i+1] for k, v in tensor_batch.items()}
                x_i["action_idx"] = torch.as_tensor(
                    np.asarray(sample["action_idx"], dtype=np.int64)[None, :]
                ).to(device)
                x_i["action_len"] = torch.tensor(
                    [int(sample["action_len"])], device=device
                )
                logp, _ = recompute_logp(policy, x_i, encoded=h[i:i+1])
                loss_pi = loss_pi - logp[0]  # maximize logp
                n_pi += 1

    n_batch = len(batch)
    loss_pi = loss_pi / max(n_pi, 1)
    loss_v = loss_v / n_batch
    loss_kl = kl_sum / max(n_kl, 1)
    total_loss = config.c_pi * loss_pi + config.c_value * loss_v \
                 + getattr(config, "beta", 0.1) * loss_kl

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    # Pre-clip gradient norm
    raw_grad = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            raw_grad += p.grad.norm().item() ** 2
    raw_grad = raw_grad ** 0.5

    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
    optimizer.step()

    # Explained variance: 1 - Var(ret - pred) / Var(ret)
    rets_arr = np.array(rets, dtype=np.float32)
    var_ret = float(np.var(rets_arr))
    if var_ret > 1e-8:
        ev = 1.0 - float(np.var(rets_arr - np.array(preds, dtype=np.float32))) / var_ret
    else:
        ev = float("nan")

    return {
        "loss": float(total_loss.detach()),
        "pi_ce": float(loss_pi.detach()) if n_pi > 0 else 0.0,
        "v_mse": float(loss_v.detach()),
        "kl": float(loss_kl.detach()),
        "n_pi": n_pi,
        "grad_norm": raw_grad,
        "ev": ev,
        "pred_mean": float(np.mean(preds)),
        "pred_std": float(np.std(preds)),
        "ret_mean": float(np.mean(rets)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load artifacts ──────────────────────────────────────────────────
    logger.info("Loading artifacts from %s", data_dir)
    vocab = _load_vocab(data_dir)
    archetypes = _load_archetypes(data_dir)
    _check_ckpt_deck(args.il_ckpt, args.deck_archetype)
    fixed_deck = _deck_for(data_dir, args.deck_archetype)
    logger.info(
        "Deck: %s (%d cards)",
        "fixed_deck" if args.deck_archetype is None
        else f"archetype {args.deck_archetype}",
        len(fixed_deck),
    )
    opp_decks = _load_opp_decks(data_dir, all_archetypes=args.all_archetypes)
    # Load engine card/attack feature maps (required for pure-feature model)
    import numpy as _np
    engine_card_features = None
    engine_attack_features = None
    ecf_path = data_dir / "engine_card_features.npy"
    eaf_path = data_dir / "engine_attack_features.npy"
    if ecf_path.exists():
        engine_card_features = _np.load(ecf_path, allow_pickle=True).item()
        logger.info("Engine card features loaded (%d cards)", len(engine_card_features))
    if eaf_path.exists():
        engine_attack_features = _np.load(eaf_path, allow_pickle=True).item()
        logger.info("Engine attack features loaded (%d attacks)", len(engine_attack_features))
    opp_weights: dict[int, float] = {}  # adaptive: low wr → high weight

    logger.info(
        "Opponent archetypes: %s",
        ", ".join(f"id={d['id']}" for d in opp_decks),
    )

    # ── ELO tracker (must be before policy load — auto-resume needs it) ──
    elo = EloTracker()
    # Discover all IL checkpoints and register them as fixed-reference
    # opponents.  ckpt-best is the anchor at 1500 (fixed).
    il_checkpoints = _discover_il_checkpoints(args.il_ckpt)
    logger.info(
        "IL checkpoints in %s: %d found",
        Path(args.il_ckpt).resolve().parent,
        len(il_checkpoints),
    )
    for name, _path in il_checkpoints:
        elo.add_model(name, fixed=(name == "ckpt-best"))
    elo.add_model("IL_baseline")  # frozen π_IL (may duplicate a step ckpt)
    # Restore saved ELO ratings — must come after add_model so fixed
    # markers are re-applied, but overwrites fresh 1500s with saved values.
    _load_elo(out_dir, elo)
    elo_name = lambda g: f"ckpt-mcts-champion-{g:06d}"

    # ── Discover existing champions ─────────────────────────────────────
    champion_ckpts: list[Path] = sorted(
        out_dir.glob("ckpt-mcts-champion-*.pt")
    )[:args.max_champions]

    # ── Load policy ─────────────────────────────────────────────────────
    # Resolve what checkpoint to start from:
    #   1. --resume (explicit) → use that file
    #   2. highest-ELO champion in out_dir → auto-resume
    #   3. fall back → IL warm-start
    best_ckpt: Path | None = None
    if args.resume:
        logger.info("Resuming from %s", args.resume)
        policy = _load_policy(args.resume, data_dir, device)
    else:
        best_ckpt = _find_best_champion(champion_ckpts, elo)
        if best_ckpt is not None:
            best_elo_val = elo.ratings.get(best_ckpt.stem, 0)
            logger.info(
                "Auto-resuming from best champion: %s (ELO: %.0f)",
                best_ckpt.name, best_elo_val,
            )
            policy = _load_policy(str(best_ckpt), data_dir, device)
        else:
            logger.info("Loading IL warm-start from %s", args.il_ckpt)
            policy = _load_policy(args.il_ckpt, data_dir, device)

    frozen_il = _load_frozen_anchor(args.il_ckpt, data_dir, device)
    logger.info("Frozen π_IL loaded for evaluation")

    # ── Read deck label from IL checkpoint (needed for Kaggle submission) ─
    _deck_record = None
    try:
        import torch as _torch
        _il_raw = _torch.load(args.il_ckpt, map_location="cpu", weights_only=False)
        _deck_record = _il_raw.get("deck")
        if _deck_record is not None:
            logger.info("Deck label loaded from IL checkpoint")
        else:
            logger.warning("IL checkpoint has no 'deck' label — MCTS champions will lack it too")
    except Exception:
        logger.warning("Could not read deck from IL checkpoint — MCTS champions will lack deck label")

    # ── Startup diagnostic ──────────────────────────────────────────────
    logger.info("══════════════════════════════════════════════════════════")
    logger.info("IL checkpoint : %s", args.il_ckpt)
    logger.info("Output dir    : %s", out_dir)
    logger.info("Champions     : %d found (max %d)", len(champion_ckpts), args.max_champions)
    if champion_ckpts:
        for ckpt in sorted(champion_ckpts, key=lambda p: p.stat().st_mtime):
            r = elo.ratings.get(ckpt.stem)
            logger.info("  %-45s  ELO=%s", ckpt.name,
                        f"{r:.0f}" if r is not None else "?")
    else:
        logger.info("  (none — will start from IL warm-start)")
    logger.info("Policy source : %s",
                "explicit --resume" if args.resume
                else f"auto-resume {best_ckpt.name}" if best_ckpt is not None
                else "IL warm-start")
    logger.info("Frozen KL anchor : %s", args.il_ckpt)
    logger.info("ELO leaderboard:")
    for line in elo.leaderboard().split("\n"):
        logger.info("  %s", line)
    logger.info("══════════════════════════════════════════════════════════")

    # ── W&B logger ────────────────────────────────────────────────────
    wb_cfg = {k: str(v) for k, v in vars(args).items()}
    wb_cfg.update({"n_opp_archetypes": len(opp_decks), "il_ckpt": args.il_ckpt})
    wb = _MctsWandbLogger(args, wb_cfg)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    # ── Replay buffer ───────────────────────────────────────────────────
    from ptcg_rl.search import ReplayBuffer
    replay = ReplayBuffer(capacity=args.buffer_capacity)

    # ── League state (champion_ckpts already discovered above) ─────────

    # ── Training loop ───────────────────────────────────────────────────
    total_games = 0
    t_start = time.perf_counter()
    t_spent_sp = 0.0     # cumulative self-play time
    t_spent_train = 0.0  # cumulative training time
    t_spent_eval = 0.0   # cumulative evaluation time
    cum_train_loss = 0.0
    cum_grad_norm = 0.0
    n_train_steps = 0
    iter_count = 0

    def _print_summary():
        elapsed = time.perf_counter() - t_start
        logger.info(
            "ITER %-3d │ games %d │ wall %s (sp %s + train %s + eval %s) │ "
            "buf %d │ avg_loss %.4f │ avg_grad %.3f",
            iter_count, total_games,
            fmt_dur(elapsed), fmt_dur(t_spent_sp), fmt_dur(t_spent_train),
            fmt_dur(t_spent_eval),
            len(replay), cum_train_loss, cum_grad_norm,
        )

    while total_games < args.total_games:
        iter_count += 1
        n_games = min(args.games_per_iter, args.total_games - total_games)
        logger.info("━━━ iter %d  self-play %d games (total %d/%d) ━━━",
                    iter_count, n_games, total_games + n_games, args.total_games)

        # ── Phase 1: Self-play ────────────────────────────────────────
        t_phase = time.perf_counter()
        samples, sp_stats = run_self_play_games(
            policy, vocab, archetypes, fixed_deck, opp_decks,
            args, device, rng, n_games,
            opp_weights=opp_weights,
            engine_card_features=engine_card_features,
            engine_attack_features=engine_attack_features,
        )
        t_sp = time.perf_counter() - t_phase
        t_spent_sp += t_sp
        wb.log_sp(sp_stats, total_games)
        for s in samples:
            replay.push(
                features=s["features"],
                action_idx=s["action_idx"],
                action_len=s["action_len"],
                mcts_pi=s["mcts_pi"],
                mcts_value=s["mcts_value"],
                game_outcome=s["game_outcome"],
                opp_archetype=s["opp_archetype"],
            )
        del samples  # free transient list (features now owned by replay buffer)
        total_games += n_games
        # Print self-play summary
        mcts_pct = sp_stats["sp_mcts_decisions"] / max(sp_stats["sp_decisions"], 1) * 100
        logger.info(
            "SP  │ %d games in %s (%.1f/min) │ wr=%.2f %dW/%dL/%dD │ "
            "%d dec (%d MCTS, %.0f%%) │ %.0f dec/game",
            n_games, fmt_dur(t_sp), sp_stats["sp_games_per_min"],
            sp_stats["sp_win_rate"], sp_stats["sp_wins"],
            sp_stats["sp_losses"], sp_stats["sp_draws"],
            sp_stats["sp_decisions"], sp_stats["sp_mcts_decisions"], mcts_pct,
            sp_stats["sp_dec_per_game"],
        )

        # ── Phase 2: Train ────────────────────────────────────────────
        if len(replay) >= args.min_buffer:
            policy.train()
            logger.info("TRAIN │ %d steps  batch=%d  lr=%.1e",
                        args.train_steps_per_iter, args.batch_size, args.lr)
            t_phase = time.perf_counter()
            cum_pi = 0.0
            cum_v = 0.0
            cum_g = 0.0
            cum_ev = 0.0
            total_lr_steps = (args.total_games // args.games_per_iter) * args.train_steps_per_iter
            for step in range(args.train_steps_per_iter):
                if args.lr_schedule == "cosine":
                    lr = _cosine_lr(n_train_steps, total_lr_steps,
                                    args.lr, args.min_lr, args.warmup_steps)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                batch = replay.sample(args.batch_size, rng)
                stats = train_step(policy, optimizer, batch, args, device,
                                   frozen_il=frozen_il)
                cum_pi += stats["pi_ce"]
                cum_v += stats["v_mse"]
                cum_g += stats["grad_norm"]
                cum_ev += stats.get("ev", 0.0)
                n_train_steps += 1
                if (step + 1) % 100 == 0:
                    logger.info(
                        "  step %d/%d  lr=%.1e  loss=%.4f  pi=%.4f  v=%.4f  "
                        "kl=%.4f  ev=%.3f  |g|=%.2f→%.3f",
                        step + 1, args.train_steps_per_iter,
                        optimizer.param_groups[0]["lr"],
                        stats["loss"], stats["pi_ce"], stats["v_mse"],
                        stats.get("kl", 0),
                        stats.get("ev", float("nan")),
                        stats["grad_norm"],
                        min(stats["grad_norm"], args.grad_clip),                     )
            t_tr = time.perf_counter() - t_phase
            t_spent_train += t_tr
            avg_pi = cum_pi / args.train_steps_per_iter
            avg_v = cum_v / args.train_steps_per_iter
            avg_g = cum_g / args.train_steps_per_iter
            avg_ev = cum_ev / args.train_steps_per_iter
            cum_train_loss = (cum_train_loss * (n_train_steps - args.train_steps_per_iter)
                              + (avg_pi + args.c_value * avg_v) * args.train_steps_per_iter
                             ) / max(n_train_steps, 1)
            cum_grad_norm = (cum_grad_norm * (n_train_steps - args.train_steps_per_iter)
                             + avg_g * args.train_steps_per_iter) / max(n_train_steps, 1)
            logger.info(
                "TRAIN │ %d steps in %s │ pi_ce=%.4f  v_mse=%.4f  "
                "ev=%.3f  grad=%.3f",
                args.train_steps_per_iter, fmt_dur(t_tr), avg_pi, avg_v, avg_ev, avg_g,
            )
            wb.log_train({"pi_ce": avg_pi, "v_mse": avg_v, "ev": avg_ev,
                          "grad": avg_g,
                          "loss": avg_pi + args.c_value * avg_v,
                          "sec": t_tr, "steps": args.train_steps_per_iter},
                         total_games)
            if total_games % 500 < args.games_per_iter:
                wb.log_model(policy, total_games)
        else:
            logger.info(
                "TRAIN │ skipped (buffer %d < min %d)",
                len(replay), args.min_buffer,
            )

        # ── Phase 3: League evaluation ────────────────────────────────
        if total_games % args.eval_every_games < args.games_per_iter:
            opponents: list[tuple[str, Any]] = [
                ("IL_baseline", frozen_il),
            ]
            # Load IL checkpoints (skip the one that's already frozen_il)
            il_ckpt_resolved = str(Path(args.il_ckpt).resolve())
            for ckpt_name, ckpt_path in il_checkpoints:
                if str(Path(ckpt_path).resolve()) == il_ckpt_resolved:
                    continue  # already loaded as frozen_il / IL_baseline
                try:
                    ckpt_model = _load_policy(ckpt_path, data_dir, device)
                    opponents.append((ckpt_name, ckpt_model))
                except Exception as e:
                    logger.warning("  Skipping unloadable IL ckpt %s: %s",
                                   ckpt_name, e)
            for ckpt_path in sorted(champion_ckpts, key=lambda p: p.stat().st_mtime):
                try:
                    champ = _load_policy(str(ckpt_path), data_dir, device)
                    opponents.append((ckpt_path.stem, champ))
                except Exception as e:
                    logger.warning("  Skipping unloadable champion %s: %s",
                                   ckpt_path.name, e)

            logger.info(
                "EVAL │ %d opponents × %d games  gate ≥%.0f%%",
                len(opponents), args.eval_games, args.gate_score * 100,
            )
            t_phase = time.perf_counter()
            league_result = league_evaluate(
                policy, opponents, vocab, fixed_deck, opp_decks,
                args, device, rng,
                engine_card_features=engine_card_features,
                engine_attack_features=engine_attack_features,
            )
            t_ev = time.perf_counter() - t_phase
            t_spent_eval += t_ev

            all_wr = [r["win_rate"] for r in league_result["results"].values()]
            avg_wr = sum(all_wr) / max(len(all_wr), 1)
            logger.info(
                "EVAL │ %d opp in %s │ avg=%.3f min=%.3f max=%.3f",
                len(opponents), fmt_dur(t_ev), avg_wr, min(all_wr), max(all_wr),
            )
            for name, r in league_result["results"].items():
                logger.info(
                    "  %-40s  wr=%.3f  %s",
                    name[:40], r["win_rate"],
                    "PASS" if r["passed"] else "FAIL",
                )

            wb.log_eval(league_result, len(champion_ckpts), total_games)

            # ── ELO update ──────────────────────────────────────────────
            new_name = elo_name(total_games)
            elo.add_model(new_name)
            for name, r in league_result["results"].items():
                elo.update(new_name, name, r["wins"], r["losses"])
            new_elo = elo.ratings[new_name]
            threshold = elo.top_pct(0.70)
            best_name, best_elo = elo.best()

            logger.info("ELO │ %s: %.0f | top30%%=%.0f | best=%s(%.0f)",
                        new_name, new_elo, threshold, best_name, best_elo)
            logger.info("ELO leaderboard:\n%s", elo.leaderboard())

            # Persist ELO ratings
            _save_elo(out_dir, elo)

            if new_elo >= threshold:
                logger.info("ELO │ PASSED (%.0f ≥ top30%% %.0f) — saving", new_elo, threshold)
                champion_ckpts = _manage_champions(
                    out_dir, policy, champion_ckpts, total_games,
                    new_elo / 1000.0, args.max_champions, elo=elo,
                    deck_record=_deck_record,
                )
            else:
                logger.info("ELO │ FAILED (%.0f < top30%% %.0f)", new_elo, threshold)
        else:
            # Per-iteration summary when no eval
            _print_summary()

    # ── Final ──────────────────────────────────────────────────────────
    _save_ckpt(out_dir, policy, "last", total_games, None, deck_record=_deck_record)
    elapsed = time.perf_counter() - t_start
    logger.info(
        "DONE │ %d games in %s │ sp=%s train=%s eval=%s │ %d champions",
        total_games, fmt_dur(elapsed),
        fmt_dur(t_spent_sp), fmt_dur(t_spent_train), fmt_dur(t_spent_eval),
        len(champion_ckpts),
    )
    wb.log_model(policy, total_games)
    _save_elo(out_dir, elo)
    wb.finish()
    return 0


class _MctsWandbLogger:
    """Lightweight W&B logger for MCTS self-play training."""

    def __init__(self, args: argparse.Namespace, cfg: dict):
        self.active = False
        if args.no_wandb or not args.wandb:
            return
        try:
            import wandb
        except ImportError:
            return
        name = args.wandb_name or f"mcts-{Path(args.il_ckpt).parent.name}"
        try:
            self._run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=name,
                config=cfg,
                mode=args.wandb_mode,
                resume="allow",
            )
            self._run.define_metric("mcts/step")
            self._run.define_metric("mcts/*", step_metric="mcts/step")
            self._run.define_metric("sp/*", step_metric="mcts/step")
            self._run.define_metric("train/*", step_metric="mcts/step")
            self._run.define_metric("eval/*", step_metric="mcts/step")
            self.active = True
        except Exception as e:
            logger.warning("wandb.init failed: %s", e)

    def log(self, metrics: dict, step: int | None = None) -> None:
        if not self.active:
            return
        payload = {k: v for k, v in metrics.items() if v is not None}
        if step is not None:
            payload["mcts/step"] = step
        self._run.log(payload)

    def log_sp(self, stats: dict, step: int) -> None:
        self.log({
            "sp/win_rate": stats["sp_win_rate"],
            "sp/games": stats["sp_games"],
            "sp/sec": stats["sp_sec"],
            "sp/games_per_min": stats["sp_games_per_min"],
            "sp/dec_per_game": stats["sp_dec_per_game"],
            "sp/mcts_fraction": stats["sp_mcts_decisions"]
            / max(stats["sp_decisions"], 1),
            "sp/decisions": stats["sp_decisions"],
            "mcts/step": step,
        }, step)

    def log_train(self, stats: dict, step: int) -> None:
        self.log({"train/" + k: v for k, v in stats.items()
                  if isinstance(v, (int, float))}, step)

    def log_eval(self, league_result: dict, n_champions: int, step: int) -> None:
        wr_list = [r["win_rate"] for r in league_result["results"].values()]
        self.log({
            "eval/avg_wr": sum(wr_list) / max(len(wr_list), 1),
            "eval/min_wr": min(wr_list),
            "eval/max_wr": max(wr_list),
            "eval/n_opponents": len(league_result["results"]),
            "eval/n_champions": n_champions,
            "eval/passed": float(league_result["passed"]),
            "mcts/step": step,
        }, step)

    def log_model(self, policy, step: int) -> None:
        if not self.active:
            return
        import torch
        for name, p in policy.named_parameters():
            n = p.data.numel()
            self._run.log({
                f"model/{name}_mean": p.data.mean().item(),
                f"model/{name}_std": p.data.std(unbiased=False).item() if n > 1 else 0.0,
                "mcts/step": step,
            })
            if p.grad is not None:
                self._run.log({
                    f"grad/{name}_mean": p.grad.mean().item(),
                    f"grad/{name}_std": p.grad.std(unbiased=False).item() if n > 1 else 0.0,
                    "mcts/step": step,
                })

    def finish(self) -> None:
        if self.active:
            self._run.finish()


def _sample_weighted(
    opp_decks: list[dict],
    rng: np.random.Generator,
    weights: dict[int, float] | None = None,
) -> dict:
    """Sample an opponent, optionally with custom relative weights."""
    if weights is None:
        return opp_decks[int(rng.integers(0, len(opp_decks)))]
    w = np.array([weights.get(d["id"], 1.0) for d in opp_decks], dtype=np.float64)
    w = w / w.sum()
    return opp_decks[int(rng.choice(len(opp_decks), p=w))]


def _load_opp_decks(data_dir: Path, all_archetypes: bool = True) -> list[dict]:
    from ptcg_rl.search import load_opp_archetype_decks
    return load_opp_archetype_decks(str(data_dir), all_archetypes=all_archetypes)


def _save_elo(out_dir: Path, elo: EloTracker) -> None:
    """Persist ELO ratings to ``out_dir / elo_ratings.json``."""
    payload = {
        "ratings": elo.ratings,
        "fixed": sorted(elo.fixed),
    }
    path = out_dir / "elo_ratings.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("ELO ratings saved to %s", path)


def _load_elo(out_dir: Path, elo: EloTracker) -> bool:
    """Restore ELO ratings from ``out_dir / elo_ratings.json``.

    Returns True if a saved file was found and loaded.
    """
    path = out_dir / "elo_ratings.json"
    if not path.exists():
        return False
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load ELO ratings: %s", e)
        return False
    saved_ratings: dict = payload.get("ratings", {})
    saved_fixed: list = payload.get("fixed", [])
    for name, rating in saved_ratings.items():
        elo.ratings[name] = float(rating)
    for name in saved_fixed:
        elo.fixed.add(name)
    logger.info(
        "ELO ratings loaded from %s: %d models, %d fixed",
        path, len(saved_ratings), len(saved_fixed),
    )
    return True


def _find_best_champion(
    champion_ckpts: list[Path], elo: EloTracker,
) -> Path | None:
    """Return the champion checkpoint with the highest ELO rating.

    Returns None if *champion_ckpts* is empty or none have a recorded rating.
    """
    best_ckpt = None
    best_elo = -float("inf")
    for ckpt in champion_ckpts:
        if not ckpt.exists():
            continue
        rating = elo.ratings.get(ckpt.stem, 0)
        if rating > best_elo:
            best_elo = rating
            best_ckpt = ckpt
    return best_ckpt


def _cosine_lr(step: int, total: int, peak: float, floor: float, warmup: int) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup:
        return peak * step / max(warmup, 1)
    if step >= total:
        return floor
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + 0.5 * (peak - floor) * (1.0 + np.cos(np.pi * progress))


def _save_ckpt(
    out_dir: Path, policy: Any, tag: str, games: int, win_rate: float | None,
    deck_record: dict | None = None,
) -> None:
    import torch as _torch
    ckpt = {
        "model_state_dict": {k: v.cpu() for k, v in policy.state_dict().items()},
        "config": dict(getattr(policy, "config", {})),
        "mcts_games": games,
        "mcts_win_rate": win_rate,
        "tag": tag,
    }
    if deck_record is not None:
        ckpt["deck"] = deck_record
    path = out_dir / f"ckpt-mcts-{tag}.pt"
    _torch.save(ckpt, path)
    logger.info("Saved %s", path)


# ── ELO Rating System ────────────────────────────────────────────────────


class EloTracker:
    """ELO ratings across all models (IL checkpoints + saved champions).

    ``ckpt-best`` is the fixed reference at 1500 — its rating never changes.
    """

    def __init__(self, k: float = 32.0, initial: float = 1500.0):
        self.k = k; self.initial = initial
        self.ratings: dict[str, float] = {}
        self.fixed: set[str] = set()  # models whose ELO is frozen (reference)

    def add_model(self, name: str, fixed: bool = False) -> None:
        self.ratings[name] = self.initial
        if fixed:
            self.fixed.add(name)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, a: str, b: str, wins: int, losses: int) -> tuple[float, float]:
        ra, rb = self.ratings.get(a, self.initial), self.ratings.get(b, self.initial)
        total = wins + losses
        if total == 0: return ra, rb
        delta = self.k * (wins / total - self.expected(ra, rb))
        if a not in self.fixed:
            self.ratings[a] = ra + delta
        if b not in self.fixed:
            self.ratings[b] = rb - delta
        return self.ratings.get(a, ra), self.ratings.get(b, rb)

    def top_pct(self, pct: float = 0.70) -> float:
        """Rating above which a model is in the top (100-pct)%."""
        vals = sorted(self.ratings.values())
        idx = max(0, int(len(vals) * pct))
        return vals[idx]

    def best(self) -> tuple[str, float]:
        if not self.ratings: return ("?", self.initial)
        name = max(self.ratings, key=self.ratings.get)
        return name, self.ratings[name]

    def leaderboard(self) -> str:
        ranked = sorted(self.ratings.items(), key=lambda x: -x[1])
        return "\n".join(f"  {n:<35} {r:>6.0f}  {'[FIXED]' if n in self.fixed else ''}"
                         for n, r in ranked)


if __name__ == "__main__":
    sys.exit(main())
