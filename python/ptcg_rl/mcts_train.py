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
    hp.add_argument("--n-engines", type=int, default=4,
                    help="libcg instances for MCTS forest")

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
    tp.add_argument("--all-archetypes", action="store_true", default=True,
                    help="Train against ALL 179 archetypes (default). "
                         "Use --no-all-archetypes for 6 𝒟_opp only.")
    tp.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate")
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
    return p


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


def _load_policy(ckpt_path: str, data_dir: Path, device: torch.device) -> Any:
    """Load policy from an IL checkpoint (or MCTS checkpoint)."""
    from ptcg_il.model.policy import load_policy_state, policy_from_config
    from ptcg_il.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(ckpt_path, device="cpu")
    config = ckpt.get("config") or {}
    if not config:
        raise SystemExit(f"{ckpt_path} has no 'config' record")

    static = _static_tables(data_dir)
    policy = policy_from_config(config, **static)
    load_policy_state(policy, ckpt["model_state_dict"])
    policy.to(device)
    return policy


def _load_frozen_anchor(ckpt_path: str, data_dir: Path, device: torch.device) -> Any:
    """Load a frozen copy for evaluation."""
    import copy
    from ptcg_il.model.policy import load_policy_state, policy_from_config
    from ptcg_il.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(ckpt_path, device="cpu")
    config = ckpt.get("config") or {}
    static = _static_tables(data_dir)
    policy = policy_from_config(config, **static)
    load_policy_state(policy, copy.deepcopy(ckpt["model_state_dict"]))
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def _fixed_deck(data_dir: Path) -> list[int]:
    with open(data_dir / "archetypes.json") as f:
        return [int(c) for c in json.load(f)["fixed_deck"]]


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
) -> list[dict]:
    """Run *n_games* self-play games with MCTS distillation.

    Each game samples a random opponent archetype.  Uses the KNOWN opponent
    deck for perfect MCTS determinization.
    """
    from ptcg_rl.search import (
        ReplayBuffer,
        mcts_distill_game,
        sample_opp_archetype,
    )

    all_samples: list[dict] = []
    t_start = time.perf_counter()

    for g in range(n_games):
        opp = sample_opp_archetype(opp_decks, rng)
        decisions = mcts_distill_game(
            policy=policy,
            vocab=vocab,
            archetypes=opp_decks,
            fixed_deck=fixed_deck,
            opp_archetype=opp,
            config=config,
            device=device,
            seed=config.seed + g,
        )

        # Add game outcome to each decision
        for d in decisions:
            all_samples.append(d)

        if (g + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            decisions_total = sum(1 for d in all_samples)
            mcts_total = sum(1 for d in all_samples if d.get("mcts_pi") is not None)
            logger.info(
                "self-play: %d/%d games, %d decisions (%d MCTS) in %s",
                g + 1, n_games, decisions_total, mcts_total, fmt_dur(elapsed),
            )

    return all_samples


def _greedy_policy_actor(policy: Any, vocab: dict, device: torch.device, seed: int):
    """Build a greedy PolicyActor from a module (or checkpoint path)."""
    from ptcg_rl.rollout import PolicyActor
    if isinstance(policy, str):
        policy = _load_policy(policy, Path("data"), device)
    policy.eval()
    return PolicyActor(policy, vocab, device=str(device), greedy=True, seed=seed)


def _play_match(
    theta_actor: Any,
    opponent_actor: Any,
    fixed_deck: list[int],
    opp_deck: list[int],
    config: argparse.Namespace,
    seat: int,
) -> list[int]:
    """Play one batch of games from a given seat.  Returns [wins, losses, draws]."""
    from ptcg_rl.rollout import DuelActor
    from ptcg_rl.vec_env import RolloutPool

    with RolloutPool(
        fixed_deck, opp_deck,
        n_workers=getattr(config, "n_workers", 4),
        forward_batch=getattr(config, "forward_batch", 256),
        seed=config.seed + 10000 * seat,
        our_player=seat,
    ) as pool:
        trajs = pool.collect(
            DuelActor(theta_actor, opponent_actor, our_player=seat),
            n_decisions=config.eval_games * 150,
        )

    wins, losses, draws = 0, 0, 0
    for t in trajs[:config.eval_games]:
        if t.reward > 0:
            wins += 1
        elif t.reward < 0:
            losses += 1
        else:
            draws += 1
    return [wins, losses, draws]


def league_evaluate(
    theta: Any,
    opponents: list[tuple[str, Any]],  # [(name, policy_module_or_path), ...]
    vocab: dict,
    fixed_deck: list[int],
    opp_decks: list[dict],
    config: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """League evaluation: θ vs every opponent, balanced seats.

    opponents: list of (name, policy_or_ckpt_path).  The first should be
    the frozen IL baseline; the rest are past champions.

    Returns ``{passed, results: {name: {win_rate, wins, losses, draws, total}}}``.
    """
    theta_actor = _greedy_policy_actor(theta, vocab, device, config.seed)

    results: dict[str, dict] = {}
    all_passed = True

    for name, opp in opponents:
        opp_actor = _greedy_policy_actor(opp, vocab, device, config.seed + 9999)

        wins, losses, draws = 0, 0, 0
        total_games = 0

        for seat in (0, 1):
            opp_deck = opp_decks[rng.integers(0, len(opp_decks))]["deck"]
            per_seat = max(1, config.eval_games // 2)
            # Override eval_games temporarily
            saved = config.eval_games
            config.eval_games = per_seat
            w, l, d = _play_match(theta_actor, opp_actor, fixed_deck, opp_deck,
                                  config, seat)
            config.eval_games = saved
            wins += w
            losses += l
            draws += d
            total_games += per_seat

        win_rate = wins / max(total_games, 1)
        passed = win_rate >= config.gate_score
        if not passed:
            all_passed = False

        results[name] = {
            "win_rate": win_rate,
            "wins": wins, "losses": losses, "draws": draws,
            "total": total_games,
            "passed": passed,
        }

        logger.info(
            "  vs %-40s  wr=%.3f  %dW/%dL/%dD  %s",
            name[:40], win_rate, wins, losses, draws,
            "PASS" if passed else "FAIL",
        )

    return {"passed": all_passed, "results": results}


def _manage_champions(
    out_dir: Path,
    policy: Any,
    current_ckpts: list[Path],
    total_games: int,
    win_rate: float,
    max_champions: int,
) -> list[Path]:
    """Save a new champion and prune old ones if over capacity.

    Returns updated list of champion checkpoint paths.
    """
    tag = f"champion-{total_games:06d}"
    _save_ckpt(out_dir, policy, tag, total_games, win_rate)
    new_path = out_dir / f"ckpt-mcts-{tag}.pt"
    current_ckpts.append(new_path)

    # Prune to max_champions, keeping most recent
    if len(current_ckpts) > max_champions:
        # Remove oldest (by file index), keep last max_champions
        current_ckpts.sort(key=lambda p: p.stat().st_mtime)
        while len(current_ckpts) > max_champions:
            removed = current_ckpts.pop(0)
            # Don't delete — just drop from the active list.  The file stays
            # on disk for auditing.
            logger.info("  Dropped oldest champion from league: %s", removed.name)

    return current_ckpts


def train_step(
    policy: Any,
    optimizer: torch.optim.Optimizer,
    batch: list[dict],
    config: argparse.Namespace,
    device: torch.device,
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

    total_loss = torch.zeros((), device=device)
    loss_pi = torch.zeros((), device=device)
    loss_v = torch.zeros((), device=device)
    n_pi = 0

    for i, sample in enumerate(batch):
        outcome = torch.tensor(sample["game_outcome"], device=device)
        v_loss = F.mse_loss(values[i], outcome)
        loss_v = loss_v + v_loss

        mcts_pi = sample.get("mcts_pi")
        if mcts_pi is not None and len(mcts_pi) > 0:
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

    n_batch = len(batch)
    loss_pi = loss_pi / max(n_pi, 1)
    loss_v = loss_v / n_batch
    total_loss = config.c_pi * loss_pi + config.c_value * loss_v

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
    optimizer.step()

    return {
        "loss": float(total_loss.detach()),
        "pi_ce": float(loss_pi.detach()) if n_pi > 0 else 0.0,
        "v_mse": float(loss_v.detach()),
        "n_pi": n_pi,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load artifacts ──────────────────────────────────────────────────
    logger.info("Loading artifacts from %s", data_dir)
    vocab = _load_vocab(data_dir)
    archetypes = _load_archetypes(data_dir)
    fixed_deck = _fixed_deck(data_dir)
    opp_decks = _load_opp_decks(data_dir, all_archetypes=args.all_archetypes)

    logger.info(
        "Opponent archetypes: %s",
        ", ".join(f"id={d['id']}" for d in opp_decks),
    )

    # ── Load policy ─────────────────────────────────────────────────────
    if args.resume:
        logger.info("Resuming from %s", args.resume)
        policy = _load_policy(args.resume, data_dir, device)
    else:
        logger.info("Loading IL warm-start from %s", args.il_ckpt)
        policy = _load_policy(args.il_ckpt, data_dir, device)

    frozen_il = _load_frozen_anchor(args.il_ckpt, data_dir, device)
    logger.info("Frozen π_IL loaded for evaluation")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    # ── Replay buffer ───────────────────────────────────────────────────
    from ptcg_rl.search import ReplayBuffer
    replay = ReplayBuffer(capacity=args.buffer_capacity)

    # ── League state ────────────────────────────────────────────────────
    # Past champion checkpoints.  The frozen IL is always the first opponent.
    champion_ckpts: list[Path] = []
    if args.resume:
        # Discover existing champions in out_dir
        existing = sorted(out_dir.glob("ckpt-mcts-champion-*.pt"))
        champion_ckpts = existing[:args.max_champions]

    # ── Training loop ───────────────────────────────────────────────────
    total_games = 0
    t_start = time.perf_counter()

    while total_games < args.total_games:
        # ── Phase 1: Self-play ────────────────────────────────────────
        n_games = min(args.games_per_iter, args.total_games - total_games)
        logger.info("=== Self-play: %d games (total: %d/%d) ===",
                    n_games, total_games + n_games, args.total_games)

        samples = run_self_play_games(
            policy, vocab, archetypes, fixed_deck, opp_decks,
            args, device, rng, n_games,
        )
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
        total_games += n_games
        logger.info("Replay buffer: %d samples", len(replay))

        # ── Phase 2: Train ────────────────────────────────────────────
        if len(replay) >= args.min_buffer:
            policy.train()  # switch back from eval (self-play) to train mode
            logger.info("--- Training: %d steps ---", args.train_steps_per_iter)
            t_train = time.perf_counter()
            for step in range(args.train_steps_per_iter):
                batch = replay.sample(args.batch_size, rng)
                stats = train_step(policy, optimizer, batch, args, device)
                if (step + 1) % 50 == 0:
                    logger.info(
                        "  step %d/%d  loss=%.4f  pi_ce=%.4f  v_mse=%.4f",
                        step + 1, args.train_steps_per_iter,
                        stats["loss"], stats["pi_ce"], stats["v_mse"],
                    )
            logger.info("  Training took %s", fmt_dur(time.perf_counter() - t_train))
        else:
            logger.info(
                "--- Skipping training (buffer: %d < min: %d) ---",
                len(replay), args.min_buffer,
            )

        # ── Phase 3: League evaluation ────────────────────────────────
        if total_games % args.eval_every_games < args.games_per_iter:
            # Build opponent list: frozen IL + past champions
            opponents: list[tuple[str, Any]] = [
                ("IL_baseline", frozen_il),
            ]
            for ckpt_path in sorted(champion_ckpts, key=lambda p: p.stat().st_mtime):
                try:
                    champ = _load_policy(str(ckpt_path), data_dir, device)
                    opponents.append((ckpt_path.stem, champ))
                except Exception as e:
                    logger.warning("  Skipping unloadable champion %s: %s",
                                   ckpt_path.name, e)

            logger.info(
                "--- League evaluation vs %d opponents (gate ≥%.0f%%) ---",
                len(opponents), args.gate_score * 100,
            )
            t_eval = time.perf_counter()
            league_result = league_evaluate(
                policy, opponents, vocab, fixed_deck, opp_decks,
                args, device, rng,
            )

            all_wr = [r["win_rate"] for r in league_result["results"].values()]
            avg_wr = sum(all_wr) / max(len(all_wr), 1)
            logger.info(
                "  League avg wr=%.3f  min=%.3f  max=%.3f  took %s",
                avg_wr, min(all_wr), max(all_wr),
                fmt_dur(time.perf_counter() - t_eval),
            )

            if league_result["passed"]:
                logger.info("  LEAGUE GATE PASSED — saving champion checkpoint")
                champion_ckpts = _manage_champions(
                    out_dir, policy, champion_ckpts, total_games,
                    avg_wr, args.max_champions,
                )
            else:
                failed = [n for n, r in league_result["results"].items()
                          if not r["passed"]]
                logger.info("  League gate FAILED against: %s",
                            ", ".join(f[:30] for f in failed))

    # ── Final ──────────────────────────────────────────────────────────
    _save_ckpt(out_dir, policy, "last", total_games, None)
    elapsed = time.perf_counter() - t_start
    logger.info(
        "Done.  %d games, %d champions, total %s",
        total_games, len(champion_ckpts), fmt_dur(elapsed),
    )
    return 0


def _load_opp_decks(data_dir: Path, all_archetypes: bool = True) -> list[dict]:
    from ptcg_rl.search import load_opp_archetype_decks
    return load_opp_archetype_decks(str(data_dir), all_archetypes=all_archetypes)


def _save_ckpt(
    out_dir: Path, policy: Any, tag: str, games: int, win_rate: float | None
) -> None:
    import torch as _torch
    ckpt = {
        "model_state_dict": {k: v.cpu() for k, v in policy.state_dict().items()},
        "config": dict(getattr(policy, "config", {})),
        "mcts_games": games,
        "mcts_win_rate": win_rate,
        "tag": tag,
    }
    path = out_dir / f"ckpt-mcts-{tag}.pt"
    _torch.save(ckpt, path)
    logger.info("Saved %s", path)


if __name__ == "__main__":
    sys.exit(main())
