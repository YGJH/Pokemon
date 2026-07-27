"""``python -m ptcg_rl.train`` — pipeline stage 5.

Two sub-stages, run in order, with a gate between them:

* **5a / R1** — critic repair (§3).  Blocks 5b.  With a dead critic GAE
  degenerates to the raw terminal return and PPO becomes REINFORCE.
* **5b / R2** — PPO with a KL anchor to the frozen IL policy, one deck,
  self-play, no MCTS (§9).

R1's gate is enforced, not advisory: `--force` exists for debugging but prints
what it is overriding.  RL_SPEC §11 is explicit that starting a phase before the
previous one passes makes failures compound and become unattributable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from ptcg_rl.logger import DEFAULTS as LOG_DEFAULTS
from ptcg_rl.logger import NullRLLogger, build_logger

logger = logging.getLogger(__name__)


def fmt_dur(seconds: float) -> str:
    """``4h12m``/``12m30s``/``30.4s`` — readable at every scale stage 5 spans.

    R1 is minutes and R2 is hours, so a single unit is wrong for one of them.
    Seconds are dropped above an hour: at that scale they are noise, and the
    string ends up in log lines that are read for their order of magnitude.
    """
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class Timer:
    """Wall-clock for one named part, as a context manager.

    Wall-clock and not CPU time on purpose: R2 is dominated by worker processes
    driving ``libcg``, so ``process_time`` in the parent would report a rollout
    that costs hours as costing seconds.
    """

    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.elapsed += time.perf_counter() - self._t0
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptcg_rl.train",
        description="Self-play RL on top of an IL specialist (RL_SPEC R1–R2)",
    )
    p.add_argument("--data-dir", type=str, default="data",
                   help="Directory with shards/, vocab.json, archetypes.json, il_baselines.json")
    p.add_argument("--il-ckpt", type=str, required=True,
                   help="The frozen IL specialist: π_IL anchor, θ init, and gate opponent")
    p.add_argument("--out-dir", type=str, default="checkpoints_rl",
                   help="Where RL checkpoints are written")
    p.add_argument("--deck-archetype", type=int, required=True,
                   help="Which 𝒟_self archetype this specialist plays")

    phase = p.add_argument_group("Phases")
    phase.add_argument("--phase", choices=["r1", "r2", "all"], default="all",
                       help="r1 = critic repair only, r2 = PPO only (assumes a "
                            "repaired critic), all = both in order (default)")
    phase.add_argument("--force", action="store_true",
                       help="Run R2 even if R1's gate failed. For debugging only: "
                            "PPO on a dead critic produces an unattributable result.")

    hp = p.add_argument_group("Hyperparameters (RL_SPEC §9.4)")
    hp.add_argument("--total-steps", type=int, default=None,
                    help="PPO optimizer steps (default: RLConfig's 50000)")
    hp.add_argument("--critic-steps", type=int, default=None)
    hp.add_argument("--lr", type=float, default=None)
    hp.add_argument("--kappa", type=float, default=None,
                    help="KL budget in nats/decision. Fixed at 0.02 through R2 by "
                         "decision (§9.3) — changing it makes the go/no-go unattributable.")
    hp.add_argument("--rollout-buffer", type=int, default=None)
    hp.add_argument("--minibatch", type=int, default=None)
    hp.add_argument("--n-workers", type=int, default=None)
    hp.add_argument("--forward-batch", type=int, default=None)
    hp.add_argument("--seed", type=int, default=0)
    hp.add_argument("--no-anchor", action="store_true",
                    help="Drop the KL anchor. Ablation only — this removes the "
                         "catastrophic-forgetting guard entirely.")

    gate = p.add_argument_group("Gate (§10.2)")
    gate.add_argument("--gate-games", type=int, default=None,
                      help="Paired games against the frozen IL policy (default 400)")
    gate.add_argument("--skip-gate", action="store_true",
                      help="Skip the final evaluation. The run then produces no "
                           "evidence about whether it worked.")

    wb = p.add_argument_group("W&B logging")
    wb.add_argument("--wandb-project", type=str, default=LOG_DEFAULTS["wandb_project"],
                    help="RL runs go to their own project by default; IL's curves "
                         "share no axes with kl_to_il/beta/clip_fraction")
    wb.add_argument("--wandb-entity", type=str, default=LOG_DEFAULTS["wandb_entity"],
                    help="Set to your personal entity for a scratch run")
    wb.add_argument("--wandb-name", type=str, default=LOG_DEFAULTS["wandb_name"],
                    help="Default: pokemon-tcg-rl-a<deck-archetype>. Stage 5 trains "
                         "one run per deck into one project, so unnamed runs would "
                         "differ only by their config.")
    wb.add_argument("--wandb-mode", type=str, default=LOG_DEFAULTS["wandb_mode"],
                    choices=["online", "offline", "disabled"])
    wb.add_argument("--no-wandb", action="store_true",
                    help="Log metrics to the console only. Wins over --wandb-mode.")

    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def _config_from_args(args: argparse.Namespace):
    from ptcg_rl.config import RLConfig

    overrides: dict[str, Any] = {"deck_archetype": args.deck_archetype, "seed": args.seed}
    for name in ("total_steps", "critic_steps", "lr", "kappa", "rollout_buffer",
                 "minibatch", "n_workers", "forward_batch"):
        val = getattr(args, name, None)
        if val is not None:
            overrides[name] = val
    if args.gate_games is not None:
        overrides["gate_paired_games"] = args.gate_games
    return RLConfig(**overrides)


def _load_policies(args: argparse.Namespace, device):
    """Build θ (trainable) and π_IL (frozen) from the same IL checkpoint.

    Both come from the checkpoint's own ``config`` record rather than from
    inferred shapes.  A wrong ``n_opp_arch`` would not raise — the belief head
    just changes width and ``load_policy_state`` forgives missing belief keys —
    so a guessed architecture yields a plausible model with random heads.
    """
    import copy

    import torch

    from ptcg_il.model.policy import load_policy_state, policy_from_config
    from ptcg_il.train.checkpoint import load_checkpoint

    ckpt = load_checkpoint(args.il_ckpt, device="cpu")
    config = ckpt.get("config") or {}
    if not config:
        raise SystemExit(
            f"{args.il_ckpt} has no 'config' record, so its architecture cannot be "
            f"reconstructed without guessing. Retrain with the current pipeline "
            f"(stage 4 writes it) — RL loads two policies and cannot infer shapes."
        )

    static = _static_tables(Path(args.data_dir))
    policy = policy_from_config(config, **static)
    load_policy_state(policy, ckpt["model_state_dict"])
    policy.to(device)

    reference = None
    if not args.no_anchor:
        reference = policy_from_config(config, **static)
        load_policy_state(reference, copy.deepcopy(ckpt["model_state_dict"]))
        reference.to(device).eval()
        for p in reference.parameters():
            p.requires_grad_(False)

    return policy, reference, ckpt


def _static_tables(data_dir: Path) -> dict:
    import numpy as np
    import torch

    out: dict[str, Any] = {}
    for key, fname in (("card_static_table", "card_static_table.npy"),
                       ("attack_static_table", "attack_static_table.npy")):
        path = data_dir / fname
        out[key] = torch.from_numpy(np.load(path)).float() if path.exists() else None
    return out


def _loaders(args, cfg, batch_size: int = 256):
    """Train/val loaders over the IL shards for this archetype."""
    from torch.utils.data import DataLoader

    from ptcg_il.train.dataset import ShardDataset, collate_fn

    def make(split: str, shuffle: bool):
        ds = ShardDataset(args.data_dir, split=split, shuffle=shuffle,
                          archetype_self=cfg.deck_archetype)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=0, drop_last=False)

    return make("train", True), make("val", False)


def run_r1(args, cfg, policy, device, *, wb=None) -> dict[str, Any]:
    """Stage 5a — critic repair."""
    from ptcg_rl.critic import measure, repair

    wb = wb or NullRLLogger()
    t_start = time.perf_counter()

    train_loader, val_loader = _loaders(args, cfg)

    before = measure(policy, val_loader, device, max_batches=40)
    logger.info("R1 critic BEFORE: %s", before.summary())
    if before.collapsed:
        logger.info(
            "  Confirms RL_SPEC §3 on a correctly-loaded model: the head is "
            "constant, which is the MSE-minimising answer when h_CLS carries no "
            "outcome signal."
        )

    # ── Phase A: frozen trunk (§3 candidate 1, diagnostic as much as repair) ──
    with Timer("r1_phase_a") as t_a:
        diag, history_a = repair(
            policy, train_loader, val_loader, device,
            steps=cfg.critic_steps, lr=cfg.critic_lr,
            freeze_trunk=cfg.critic_freeze_trunk,
        )
    logger.info("R1 phase A (frozen trunk) took %s: %s", fmt_dur(t_a.elapsed), diag.summary())

    passed = diag.passes(cfg.critic_corr_target, cfg.critic_std_target)
    phase_a = _diag_record(diag)
    phase_a["elapsed_sec"] = t_a.elapsed
    wb.summary({"r1/before/corr": before.corr, "r1/before/pred_std": before.pred_std})
    wb.log_r1_curve(history_a, "a")
    wb.log_r1_diag(phase_a, "a")
    phase_b: dict[str, Any] | None = None

    # ── Phase B: joint fine-tune, if phase A stalled below the gate ──────────
    if not passed and cfg.critic_freeze_trunk and cfg.critic_joint_steps > 0:
        logger.info(
            "Phase A reached corr=%.4f std=%.4f, short of the %.2f/%.2f gate. That "
            "is §3 candidate 1 answering: h_CLS does not carry enough outcome "
            "signal, so the trunk is the bottleneck, not the head. Starting phase "
            "B — joint fine-tune with the IL cross-entropy holding the policy.",
            diag.corr, diag.pred_std, cfg.critic_corr_target, cfg.critic_std_target,
        )

        # The two top-1 measurements are timed separately from the repair: they
        # are full eval passes, so folding them into the phase time would make
        # phase B look slower than the training it actually does.
        with Timer("il_top1") as t_top1:
            top1_before = _measure_il_top1(args, cfg, policy, device)
        with Timer("r1_phase_b") as t_b:
            diag, history_b = repair(
                policy, train_loader, val_loader, device,
                steps=cfg.critic_joint_steps, lr=cfg.critic_joint_lr,
                freeze_trunk=False,
                value_weight=cfg.critic_value_weight,
                ce_weight=cfg.critic_ce_weight,
            )
        with t_top1:
            top1_after = _measure_il_top1(args, cfg, policy, device)
        logger.info("R1 phase B (joint) took %s (+%s in IL top-1 eval): %s",
                    fmt_dur(t_b.elapsed), fmt_dur(t_top1.elapsed), diag.summary())

        phase_b = _diag_record(diag)
        phase_b["elapsed_sec"] = t_b.elapsed
        phase_b["il_top1_eval_sec"] = t_top1.elapsed
        phase_b["il_top1_before"] = top1_before
        phase_b["il_top1_after"] = top1_after
        wb.log_r1_curve(history_b, "b")

        # The CE term exists to prevent exactly this; check rather than trust it.
        if top1_before is not None and top1_after is not None:
            drop_pts = (top1_before - top1_after) * 100.0
            phase_b["il_regression_pts"] = -drop_pts
            logger.info("  policy non-trivial top-1: %.4f → %.4f (%+.2f pts)",
                        top1_before, top1_after, -drop_pts)
            if drop_pts > cfg.critic_policy_regression_pts:
                logger.error(
                    "Phase B cost the policy %.2f pts of non-trivial top-1 (limit "
                    "%.2f). A critic bought by wrecking the actor is not progress: "
                    "raise critic_ce_weight or lower critic_joint_lr.",
                    drop_pts, cfg.critic_policy_regression_pts,
                )
                phase_b["aborted_on_policy_regression"] = True
                wb.log_r1_diag(phase_b, "b")
                out = {"phase": "R1", "passed": False,
                       "phase_a": phase_a, "phase_b": phase_b,
                       "before_corr": before.corr, "before_pred_std": before.pred_std,
                       "elapsed_sec": time.perf_counter() - t_start,
                       **_diag_record(diag)}
                logger.info("R1 aborted after %s", fmt_dur(out["elapsed_sec"]))
                wb.log_r1_result(out)
                return out

        # After the regression block, so `il_regression_pts` is in the record.
        wb.log_r1_diag(phase_b, "b")
        passed = diag.passes(cfg.critic_corr_target, cfg.critic_std_target)

    if not passed:
        logger.error(
            "R1 did not reach the gate. Both phases are recorded in rl_report.json; "
            "if phase B also stalled, the outcome may simply be less predictable "
            "from a single state than §3 assumed — every decision in a game shares "
            "one ±1 label, so early-game states are near-unpredictable by "
            "construction. Judge against the per-turn curve before lowering the bar."
        )

    out = {
        "phase": "R1",
        "passed": passed,
        "phase_a": phase_a,
        "phase_b": phase_b,
        "before_corr": before.corr,
        "before_pred_std": before.pred_std,
        "elapsed_sec": time.perf_counter() - t_start,
        **_diag_record(diag),
    }
    logger.info("R1 total: %s", fmt_dur(out["elapsed_sec"]))
    wb.log_r1_result(out)
    return out


def _resnapshot_anchor(policy, reference):
    """Reset the frozen π_IL to the current θ, after R1 has moved the trunk.

    RL_SPEC §9.3 defines π_IL as "the frozen ``ckpt-best.pt`` for that deck",
    written before R1 had a phase that also trains the trunk. The anchor's job is
    to stop R2 drifting away from a *known-good* policy; if the known-good policy
    has since improved, anchoring to the stale one inverts the mechanism — the KL
    term actively pulls θ backwards.

    Returns ``None`` unchanged when the anchor was disabled (``--no-anchor``).
    """
    import copy

    if reference is None:
        return None
    reference.load_state_dict(copy.deepcopy(policy.state_dict()))
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    return reference


def _r1_failure_message(r1: dict[str, Any], cfg) -> str:
    """Name the condition that actually failed.

    Printing the two numeric bars unconditionally is how a run that met both of
    them still read as "corr too low" — the failing criterion was the per-turn
    one, and the message pointed at the wrong place entirely.
    """
    failed = []
    if r1["corr"] < cfg.critic_corr_target:
        failed.append(f"corr={r1['corr']:.4f} < {cfg.critic_corr_target}")
    if r1["pred_std"] < cfg.critic_std_target:
        failed.append(f"pred_std={r1['pred_std']:.4f} < {cfg.critic_std_target}")
    if not r1.get("accuracy_rises_with_turn", False):
        failed.append(
            "accuracy does not rise with game progress "
            f"(buckets: {r1.get('per_turn_accuracy') or 'none measured'})"
        )
    return (
        f"R1 gate FAILED on: {'; '.join(failed) or 'unknown'}. "
        f"PPO on a dead critic is REINFORCE with no variance reduction."
    )


def _diag_record(diag) -> dict[str, Any]:
    return {
        "corr": diag.corr,
        "pred_std": diag.pred_std,
        "sign_agreement": diag.sign_agreement,
        "mse": diag.mse,
        "accuracy_rises_with_turn": diag.accuracy_rises_with_turn,
        "per_turn_accuracy": diag.per_turn_accuracy,
    }


def run_r2(args, cfg, policy, reference, device, *, wb=None) -> dict[str, Any]:
    """Stage 5b — PPO with a KL anchor.

    Rollout is mirror self-play: both sides act with the current ``π_θ``, and
    only our player's decisions are recorded.
    """
    import torch

    from ptcg_il.deck import build_deck_metadata
    from ptcg_rl.ppo import ppo_losses, update_beta
    from ptcg_rl.rollout import PolicyActor, build_batch
    from ptcg_rl.vec_env import RolloutPool

    wb = wb or NullRLLogger()
    vocab = _load_vocab(Path(args.data_dir))
    deck = build_deck_metadata(args.data_dir, cfg.deck_archetype)
    decklist = list(deck["deck"]) if isinstance(deck.get("deck"), list) else _fixed_deck(args)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    beta = cfg.beta_init
    step = 0
    history: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    t_rollout_total = 0.0
    t_update_total = 0.0

    actor = PolicyActor(policy, vocab, device=str(device), bf16=cfg.bf16_rollout,
                        seed=cfg.seed)

    with RolloutPool(
        decklist, decklist,
        n_workers=cfg.n_workers, forward_batch=cfg.forward_batch,
        seed=cfg.seed, our_player=0,
    ) as pool:
        while step < cfg.total_steps:
            policy.eval()
            with Timer("rollout") as t_roll:
                trajectories = pool.collect(actor, cfg.rollout_buffer)
                if not trajectories:
                    raise RuntimeError("rollout produced no completed games")
                batch = build_batch(trajectories, gamma=cfg.gamma, lam=cfg.gae_lambda)
            rollout = {
                "games": len(trajectories),
                "decisions": len(batch),
                "mean_reward": sum(t.reward for t in trajectories) / len(trajectories),
                "rollout_sec": t_roll.elapsed,
                "decisions_per_sec": len(batch) / t_roll.elapsed if t_roll.elapsed else 0.0,
            }

            policy.train()
            with Timer("update") as t_upd:
                stats = _ppo_epochs(policy, reference, batch, optimizer, beta, cfg, device)
            rollout["update_sec"] = t_upd.elapsed
            rollout["iter_sec"] = t_roll.elapsed + t_upd.elapsed
            t_rollout_total += t_roll.elapsed
            t_update_total += t_upd.elapsed
            # Rollout is expected to dominate; if it stops dominating, the update
            # path has regressed and the split is the only thing that shows it.
            logger.info(
                "iter: %d games, %d decisions, mean reward %+.3f | "
                "rollout %s (%.0f dec/s) + update %s = %s",
                rollout["games"], rollout["decisions"], rollout["mean_reward"],
                fmt_dur(t_roll.elapsed), rollout["decisions_per_sec"],
                fmt_dur(t_upd.elapsed), fmt_dur(rollout["iter_sec"]),
            )
            # β chases the k3 estimator, not the raw mean: the raw one is unbiased
            # but can be negative on a finite minibatch, which reads as "the anchor
            # is slack" and collapses β toward β_min.
            beta = update_beta(beta, stats["kl_to_il_k3"], cfg)
            step += stats["n_updates"]
            stats["step"] = step
            stats["beta_next"] = beta
            history.append(stats)
            wb.log_r2_iter(stats, rollout)
            logger.info(
                "step %d/%d  pi=%.4f v=%.4f H=%.3f kl=%.4f (k3 %.4f) beta=%.4g "
                "clip=%.3f ev=%.3f",
                step, cfg.total_steps, stats["policy_loss"], stats["value_loss"],
                stats["entropy"], stats["kl_to_il"], stats["kl_to_il_k3"], beta,
                stats["clip_fraction"], stats["explained_variance"],
            )

    elapsed = time.perf_counter() - t_start
    # `elapsed` exceeds rollout+update by the pool setup and teardown, so all
    # three are reported rather than leaving the remainder to be inferred.
    logger.info(
        "R2 total: %s  (rollout %s, update %s, other %s)  %d steps, %.0f steps/h",
        fmt_dur(elapsed), fmt_dur(t_rollout_total), fmt_dur(t_update_total),
        fmt_dur(elapsed - t_rollout_total - t_update_total),
        step, step / elapsed * 3600 if elapsed else 0.0,
    )
    timing = {
        "elapsed_sec": elapsed,
        "rollout_sec": t_rollout_total,
        "update_sec": t_update_total,
        "steps_per_hour": step / elapsed * 3600 if elapsed else 0.0,
    }
    wb.summary({f"r2/{k}": v for k, v in timing.items()})
    return {"phase": "R2", "steps": step, "beta": beta, "history": history, **timing}


def _ppo_epochs(policy, reference, batch, optimizer, beta, cfg, device) -> dict[str, Any]:
    """Run ``cfg.ppo_epochs`` over one rollout buffer.

    ``logp_old`` is recomputed over the **whole buffer up front** in this
    precision path, rather than restored from the rollout values (§7).  Doing it
    per-minibatch at epoch 0 instead is subtly wrong: the epoch loop drops a
    ragged tail, so any sample that lands in the tail at epoch 0 keeps its
    rollout log-prob — computed in the *rollout's* precision — and can then be
    drawn into a full minibatch at epoch 1. That is exactly the mixed-precision
    ratio §7 is about, on a small and shifting subset, which is the hardest
    version to notice.
    """
    import numpy as np
    import torch

    from ptcg_rl.ppo import ppo_losses
    from ptcg_rl.rollout import _collate

    n = len(batch)
    order = np.arange(n)
    agg: dict[str, list[float]] = {}
    n_updates = 0

    _recompute_buffer_logp(policy, batch, cfg, device)

    for epoch in range(cfg.ppo_epochs):
        np.random.shuffle(order)
        for start in range(0, n - cfg.minibatch + 1, cfg.minibatch):
            idx = order[start:start + cfg.minibatch]
            mb = _collate([batch.features[i] for i in idx], device)
            mb["action_idx"] = torch.as_tensor(batch.action_idx[idx]).to(device)
            mb["action_len"] = torch.as_tensor(batch.action_len[idx]).to(device)

            logp_old = torch.as_tensor(batch.logp_old[idx]).to(device).float()

            loss, stats = ppo_losses(
                policy, reference, mb,
                advantage=torch.as_tensor(batch.advantage[idx]).to(device).float(),
                value_target=torch.as_tensor(batch.value_target[idx]).to(device).float(),
                value_old=torch.as_tensor(batch.value_old[idx]).to(device).float(),
                logp_old=logp_old,
                beta=beta, cfg=cfg,
                # Only the very first minibatch: after one optimizer step θ has
                # genuinely moved, so a ratio of 1 is no longer expected and
                # checking further would fire on correct behaviour.
                check_canary=(n_updates == 0),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            optimizer.step()
            n_updates += 1

            for k in ("policy_loss", "value_loss", "entropy", "kl_to_il",
                      "kl_to_il_k3", "clip_fraction", "approx_kl_step",
                      "explained_variance", "ratio_p99"):
                agg.setdefault(k, []).append(getattr(stats, k))

    out = {k: float(sum(v) / len(v)) for k, v in agg.items() if v}
    out["n_updates"] = n_updates
    return out


def _recompute_buffer_logp(policy, batch, cfg, device) -> None:
    """Overwrite ``batch.logp_old`` with log-probs from the update's own path.

    One forward over the whole buffer, in order, before any optimizer step — so
    every sample's reference point comes from the same weights *and* the same
    precision path, and all epochs of this update share a single ``π_θ_old``.
    Recomputing per epoch instead would make the ratio measure only the last
    epoch's step rather than the whole update's, which is a different objective.
    """
    import numpy as np
    import torch

    from ptcg_rl.actor import recompute_logp
    from ptcg_rl.rollout import _collate

    n = len(batch)
    out = np.empty(n, dtype=np.float32)
    # train(), *not* eval(), despite this being a pure inference pass.
    # nn.TransformerEncoderLayer takes a fused fast path only when the module is
    # in eval mode AND grad is disabled — both true here, neither true in the
    # grad-enabled PPO update.  The two kernels disagree by ~2e-4 on the encoder
    # output, which reaches the epoch-0 ratio as p99 |ratio-1| = 3.7e-3 and
    # trips the canary at 1e-3.  Matching the update's mode is what makes this
    # function's "same precision path" promise true; it is safe because the
    # policy has no dropout (encoder.py forces it to 0) and no batchnorm, so
    # train mode differs from eval *only* in that kernel choice.
    policy.train()
    with torch.no_grad():
        for start in range(0, n, cfg.minibatch):
            sl = slice(start, min(start + cfg.minibatch, n))
            mb = _collate(batch.features[sl], device)
            mb["action_idx"] = torch.as_tensor(batch.action_idx[sl]).to(device)
            mb["action_len"] = torch.as_tensor(batch.action_len[sl]).to(device)
            logp, _ = recompute_logp(policy, mb)
            out[sl] = logp.float().cpu().numpy()
    policy.train()
    batch.logp_old = out


def _load_vocab(data_dir: Path) -> dict:
    from ptcg_il.featurizer import normalize_vocab

    with open(data_dir / "vocab.json") as f:
        return normalize_vocab(json.load(f))


def _fixed_deck(args) -> list[int]:
    with open(Path(args.data_dir) / "archetypes.json") as f:
        return [int(c) for c in json.load(f)["fixed_deck"]]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = _config_from_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("RL stage: archetype %d, device %s, %s",
                cfg.deck_archetype, device, args.phase)
    policy, reference, il_ckpt = _load_policies(args, device)

    wb = build_logger(args, cfg, {
        "il_ckpt": str(args.il_ckpt),
        "phase": args.phase,
        "device": str(device),
    })
    try:
        return _run_phases(args, cfg, policy, reference, il_ckpt, device, out_dir, wb)
    finally:
        # A crashed run still has to close its W&B run, or the next stage-5 deck
        # inherits a live process and the failed run stays "running" forever.
        wb.finish()


def _run_phases(args, cfg, policy, reference, il_ckpt, device, out_dir, wb) -> int:
    report: dict[str, Any] = {"config": cfg.to_dict(), "il_ckpt": str(args.il_ckpt)}
    t_start = time.perf_counter()

    if args.phase in ("r1", "all"):
        report["r1"] = run_r1(args, cfg, policy, device, wb=wb)

        # R1's phase B trains the shared trunk, so θ is no longer the checkpoint
        # the anchor and the baseline were taken from. Measured on this corpus it
        # moved the policy +19.3 pts of non-trivial top-1 — anchoring R2 to the
        # pre-R1 weights would pull θ *back* toward a materially worse policy,
        # and the gate's IL-regression check would carry 19 pts of slack before
        # it noticed anything. Re-derive both from the post-R1 policy.
        if report["r1"].get("phase_b") is not None:
            reference = _resnapshot_anchor(policy, reference)
            report["r1"]["anchor_resnapshotted"] = True
            logger.info(
                "R1 phase B changed the trunk; re-snapshotted π_IL and the "
                "IL-regression baseline from the post-R1 policy."
            )

        _write_report(out_dir, report)
        if not report["r1"]["passed"]:
            msg = _r1_failure_message(report["r1"], cfg)
            if args.phase == "all" and not args.force:
                logger.error("%s Stopping before R2. Use --force to override.", msg)
                report["timing"] = _timing_summary(report, time.perf_counter() - t_start)
                _write_report(out_dir, report)
                logger.info("RL stage stopped after %s%s",
                            fmt_dur(report["timing"]["total_sec"]),
                            _timing_breakdown(report["timing"]))
                return 1
            if args.phase == "all":
                logger.warning("%s Continuing into R2 anyway (--force).", msg)
            else:
                logger.warning("%s (R1-only run; nothing downstream was started.)", msg)

    if args.phase in ("r2", "all"):
        report["r2"] = run_r2(args, cfg, policy, reference, device, wb=wb)
        _write_report(out_dir, report)

    if not args.skip_gate and args.phase in ("r2", "all"):
        report["gate"] = _run_gate(
            args, cfg, policy, reference, device,
            post_r1_top1=(report.get("r1") or {}).get("phase_b", {}).get("il_top1_after")
            if (report.get("r1") or {}).get("phase_b") else None,
            wb=wb,
        )
        _write_report(out_dir, report)
        logger.info("Gate: %s", report["gate"].get("summary", "n/a"))

    _save_checkpoint(out_dir, policy, il_ckpt, cfg, report)
    report["timing"] = _timing_summary(report, time.perf_counter() - t_start)
    _write_report(out_dir, report)
    logger.info("RL stage done in %s%s", fmt_dur(report["timing"]["total_sec"]),
                _timing_breakdown(report["timing"]))
    wb.summary({"timing/total_sec": report["timing"]["total_sec"]})
    return 0


def _timing_summary(report: dict[str, Any], total_sec: float) -> dict[str, Any]:
    """Per-part seconds, collected in one place for ``rl_report.json``.

    Kept out of the phase records so a reader comparing two decks does not have
    to walk three nesting levels to find where the hours went.
    """
    out: dict[str, Any] = {"total_sec": total_sec}
    r1, r2, gate = report.get("r1"), report.get("r2"), report.get("gate")
    if r1:
        out["r1_sec"] = r1.get("elapsed_sec")
        out["r1_phase_a_sec"] = (r1.get("phase_a") or {}).get("elapsed_sec")
        out["r1_phase_b_sec"] = (r1.get("phase_b") or {}).get("elapsed_sec")
    if r2:
        out["r2_sec"] = r2.get("elapsed_sec")
        out["r2_rollout_sec"] = r2.get("rollout_sec")
        out["r2_update_sec"] = r2.get("update_sec")
        out["r2_steps_per_hour"] = r2.get("steps_per_hour")
    if gate:
        out["gate_sec"] = gate.get("elapsed_sec")
    return out


def _timing_breakdown(timing: dict[str, Any]) -> str:
    """`` (R1 12m30s, R2 4h02m [rollout 3h51m + update 9m40s], gate 21m)``."""
    parts = []
    if timing.get("r1_sec") is not None:
        parts.append(f"R1 {fmt_dur(timing['r1_sec'])}")
    if timing.get("r2_sec") is not None:
        inner = ""
        if timing.get("r2_rollout_sec") is not None:
            inner = (f" [rollout {fmt_dur(timing['r2_rollout_sec'])}"
                     f" + update {fmt_dur(timing['r2_update_sec'])}]")
        parts.append(f"R2 {fmt_dur(timing['r2_sec'])}{inner}")
    if timing.get("gate_sec") is not None:
        parts.append(f"gate {fmt_dur(timing['gate_sec'])}")
    return f"  ({', '.join(parts)})" if parts else ""


def _run_gate(args, cfg, policy, reference, device,
              post_r1_top1: float | None = None, *, wb=None) -> dict[str, Any]:
    """Evaluate θ against the frozen π_IL and apply §10.2's conditions.

    The IL-regression baseline is read through ``ptcg_il.baselines``, which
    refuses a record whose SHA-1 does not match ``--il-ckpt``.  A baseline from a
    different model is worse than none: it looks like a check.
    """
    from ptcg_rl.gate import evaluate_gate, load_il_baseline

    t_start = time.perf_counter()

    baseline = None
    if post_r1_top1 is not None:
        # R1's phase B trained the trunk, so the recorded stage-4 baseline no
        # longer describes the policy R2 started from. Gate against where R2
        # actually began, or the regression check carries R1's whole improvement
        # as slack and stops detecting anything.
        baseline = post_r1_top1
        logger.info("IL-regression baseline: %.4f (post-R1 policy, not the "
                    "stage-4 record)", baseline)
    else:
        try:
            baseline = load_il_baseline(args.data_dir, cfg.deck_archetype, args.il_ckpt)
        except (FileNotFoundError, KeyError) as e:
            logger.warning("No usable IL baseline (%s); the regression check will be skipped.", e)
        except Exception as e:
            logger.error("IL baseline rejected: %s", e)
            raise

    measured = _measure_il_top1(args, cfg, policy, device)

    result = evaluate_gate(
        _play_gate_games(args, cfg, policy, reference, device),
        min_score=cfg.gate_score,
        min_wilson_lb=cfg.gate_wilson_lb,
        il_nontrivial_top1=measured,
        il_baseline=baseline,
        max_il_regression_pts=cfg.gate_il_regression_pts,
    )
    out = result.__dict__.copy()
    out["summary"] = result.summary()
    out["elapsed_sec"] = time.perf_counter() - t_start
    logger.info("Gate took %s", fmt_dur(out["elapsed_sec"]))
    (wb or NullRLLogger()).log_gate(out)
    return out


def _measure_il_top1(args, cfg, policy, device) -> float | None:
    """Non-trivial top-1 on the held-out IL test split, for §10.2 condition 3."""
    from torch.utils.data import DataLoader

    from ptcg_il.train.dataset import ShardDataset, collate_fn
    from ptcg_il.train.eval import offline_eval

    try:
        ds = ShardDataset(args.data_dir, split="test", shuffle=False,
                          archetype_self=cfg.deck_archetype)
    except (ValueError, FileNotFoundError) as e:
        logger.warning("Cannot measure IL regression: %s", e)
        return None
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn)
    metrics = offline_eval(policy, loader, device)
    return float(metrics.get("val/top1_nontrivial", 0.0))


def _play_gate_games(args, cfg, policy, reference, device) -> list[dict[str, Any]]:
    """Greedy games of θ against the frozen π_IL, balanced across both seats.

    Two pools, one per seat, with equal game counts. Balancing the seat is the
    point of §10.2's pairing: first-player advantage is large, and an unbalanced
    sample spends most of its games measuring that rather than the policies.

    **Honest limitation:** this balances the seat but does not reproduce the
    spec's *seed-level* pairing. The engine shuffles internally, so the two games
    of a "pair" do not share a deal — only the seat is controlled. That makes the
    comparison unbiased but higher-variance than the spec assumes, so the Wilson
    bound (which does not know this) is if anything optimistic at a given n.
    """
    from ptcg_rl.rollout import DuelActor, PolicyActor
    from ptcg_rl.vec_env import RolloutPool

    if reference is None:
        logger.warning("No frozen reference policy (--no-anchor); gate skipped.")
        return []

    vocab = _load_vocab(Path(args.data_dir))
    decklist = _fixed_deck(args)
    results: list[dict[str, Any]] = []

    # Greedy on both sides: this is the deployment-time procedure, so the gate
    # measures the artifact that ships (§8.3, §10.2).
    theta = PolicyActor(policy, vocab, device=str(device), greedy=True, seed=cfg.seed)
    frozen = PolicyActor(reference, vocab, device=str(device), greedy=True, seed=cfg.seed)

    # ~1 game is ~150 decisions; ask for enough decisions to finish the games.
    per_seat = max(1, cfg.gate_paired_games)
    for seat in (0, 1):
        with RolloutPool(decklist, decklist, n_workers=cfg.n_workers,
                         forward_batch=cfg.forward_batch,
                         seed=cfg.seed + 1000 * seat, our_player=seat) as pool:
            trajs = pool.collect(
                DuelActor(theta, frozen, our_player=seat),
                n_decisions=per_seat * 150,
            )
        for t in trajs[:per_seat]:
            if t.reward > 0:
                winner = seat
            elif t.reward < 0:
                winner = 1 - seat
            else:
                winner = -1
            results.append({"winner": winner, "our_player": seat})

    logger.info("Gate played %d games (%d per seat)", len(results), per_seat)
    return results


def _save_checkpoint(out_dir: Path, policy, il_ckpt: dict, cfg, report: dict) -> None:
    """Write the RL checkpoint, extending — never replacing — the deck record.

    The ``deck`` record's artifact pinning (``vocab_sha1``, ``archetypes_sha1``)
    is as load-bearing under RL as under IL, and RL checkpoints must stay
    loadable by ``scripts/build_submission.py``.
    """
    import torch

    from ptcg_il.deck import DECK_KEY

    ckpt: dict[str, Any] = {
        "step": report.get("r2", {}).get("steps", 0),
        "model_state_dict": {k: v.cpu() for k, v in policy.state_dict().items()},
        "config": dict(getattr(policy, "config", {})),
        "rl": {
            "generation": 1,
            "parent": report.get("il_ckpt"),
            "phase": "R2" if "r2" in report else "R1",
            "kappa": cfg.kappa,
            "steps": report.get("r2", {}).get("steps", 0),
            "r1": {k: v for k, v in report.get("r1", {}).items()
                   if k != "per_turn_accuracy"},
            "gate": report.get("gate"),
        },
    }
    if il_ckpt.get(DECK_KEY):
        ckpt[DECK_KEY] = il_ckpt[DECK_KEY]

    path = out_dir / "ckpt-rl-last.pt"
    torch.save(ckpt, path)
    logger.info("Saved %s", path)


def _write_report(out_dir: Path, report: dict) -> None:
    with open(out_dir / "rl_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
        f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
