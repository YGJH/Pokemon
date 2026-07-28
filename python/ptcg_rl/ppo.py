"""PPO with a KL anchor to the frozen IL policy (RL_SPEC §9).

```
L = L_clip + c_v · L_value − c_H · H(π_θ) + β · KL(π_θ ‖ π_IL)
```

``L_search`` is absent: MCTS distillation is R3, and §11 makes R2 the go/no-go.

Three things here are load-bearing and easy to get subtly wrong.

**The ratio is on the joint sequence.**  Not per AR step.  Per-step ratios
optimise a different objective, and the difference does not show up as an error —
it shows up as a policy that slowly prefers short selections.  All log-probs come
from :mod:`ptcg_rl.actor`, which is the only place masking lives.

**`logp_old` is recomputed, not restored.**  §7 measured that the *same weights*
produce log-probs differing by 0.02 mean / 0.69 p99 between bf16 and fp32, with
the tail concentrated in ties (p99 ≈ ln 2, max ≈ ln 8, ratio floor exactly 1/8 —
bf16 rounds near-equal logits to exactly equal).  At p99 that consumes 500% of
the ε = 0.2 clip range, so clipping would be driven by rounding rather than by
policy change.  Recomputing ``logp_old`` in the update's own precision at epoch 0
removes the entire class of error for the cost of one extra forward pass.
Upcasting ``log_softmax`` alone does **not** fix it — the error is born in the
network's bf16 matmuls.

**β adapts, κ does not.**  β is a dual variable chasing a fixed KL budget.  A
fixed β is either inert or dominant depending on scale; a moving κ makes a
failed run unattributable, which is why it stays at 0.02 through R2 by decision
(§9.3).
"""

from __future__ import annotations

import logging
from rich.logging import RichHandler
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ptcg_rl.actor import recompute_logp
from ptcg_rl.config import RLConfig
from ptcg_rl.rollout import normalize_advantages
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_time=False)])
logger = logging.getLogger(__name__)


class RatioCanaryError(RuntimeError):
    """Epoch-0 ratios were not ~1, so the two log-prob paths disagree.

    Raised loudly rather than warned about.  At epoch 0 the policy has not been
    updated yet, so ``π_θ`` *is* ``π_θ_old`` and every ratio must be 1 to
    floating-point tolerance.  Anything else means the precision path, the
    masking, or the batching differs between rollout and update — and each of
    those corrupts training silently if allowed through.
    """


@dataclass
class PPOStats:
    """Per-update diagnostics.  Every field here has a failure it detects."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_to_il: float = 0.0
    """Unbiased single-sample estimate ``E[log π_θ(a) − log π_IL(a)]``. This is the
    quantity in the loss, and on a finite minibatch it can come out **negative** —
    it is unbiased for a non-negative KL, not itself non-negative."""
    kl_to_il_k3: float = 0.0
    """Schulman's k3 estimator, ``E[exp(−d) − 1 + d]``, which is non-negative by
    construction and much lower variance. Drive β off this one: feeding a
    spuriously negative KL into the dual update tells it the anchor is slack when
    it may not be, and β collapses toward β_min."""
    beta: float = 0.0
    clip_fraction: float = 0.0
    """Fraction of samples where the ratio hit the clip. Persistently >0.3 means
    the step size is too large for ε."""
    approx_kl_step: float = 0.0
    """KL between θ and θ_old — the policy's own step size, distinct from the
    anchor's KL to π_IL."""
    ratio_p99: float = 0.0
    explained_variance: float = 0.0
    """1 − Var(target − pred)/Var(target). Near 0 means the critic is no better
    than predicting the mean — the R1 failure, visible during R2."""
    n_samples: int = 0
    extra: dict = field(default_factory=dict)


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """``1 − Var(target − pred) / Var(target)``, or 0.0 for a constant target.

    A better critic diagnostic than MSE: with ±1 labels a dead head scores
    MSE ≈ 1.0, which reads as mediocre rather than as carrying no signal.
    """
    target_var = target.var(unbiased=False)
    if float(target_var) <= 0:
        return 0.0
    return float(1.0 - (target - pred).var(unbiased=False) / target_var)


def update_beta(beta: float, measured_kl: float, cfg: RLConfig) -> float:
    """Dual update on the KL penalty: ``β ← clip(β·exp((KL−κ)/κ·η_β), min, max)``.

    Multiplicative because β spans four orders of magnitude; the exponent is
    relative to κ so the step is scale-free.
    """
    if cfg.kappa <= 0:
        return beta
    ratio = (measured_kl - cfg.kappa) / cfg.kappa
    # Clamp the exponent before exponentiating: one pathological KL (a NaN
    # upstream, an empty minibatch) would otherwise saturate β permanently.
    scaled = max(-10.0, min(10.0, ratio * cfg.beta_lr))
    return float(min(cfg.beta_max, max(cfg.beta_min, beta * math.exp(scaled))))


def check_ratio_canary(ratio: torch.Tensor, tol: float) -> float:
    """Assert epoch-0 ratios are 1.  Returns the measured p99 ``|ratio − 1|``.

    Cheap, decisive, and it catches more than precision drift: a masking change,
    a reordered batch, or a stale feature cache all show up here first.
    """
    dev = (ratio.detach() - 1.0).abs()
    p99 = float(torch.quantile(dev.float(), 0.99)) if dev.numel() > 1 else float(dev.max())
    if not math.isfinite(p99) or p99 > tol:
        raise RatioCanaryError(
            f"epoch-0 p99 |ratio - 1| = {p99:.3e} exceeds {tol:.1e}. At epoch 0 "
            f"the policy is unchanged, so every ratio must be 1. Something "
            f"differs between the rollout and update paths — precision (§7), "
            f"option masking, or batch ordering."
        )
    return p99


def ppo_losses(
    policy,
    # 🌟 修正 1：拔除 reference_policy 參數，直接使用下面傳進來的 logp_ref
    batch: dict[str, torch.Tensor],
    *,
    advantage: torch.Tensor,
    value_target: torch.Tensor,
    value_old: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,  # 🌟 已經在外部算好傳進來的
    beta: float,
    cfg: RLConfig,
    check_canary: bool = False,
) -> tuple[torch.Tensor, PPOStats]:
    """One minibatch's total loss and its diagnostics."""
    h, _history = policy._encode(batch)
    logp, entropy = recompute_logp(policy, batch, encoded=h)

    # --- Clipped surrogate on the joint sequence ---
    ratio = (logp - logp_old).exp()
    stats = PPOStats(n_samples=int(logp.shape[0]))
    if check_canary:
        stats.ratio_p99 = check_ratio_canary(ratio, cfg.ratio_canary_tol)

    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantage
    policy_loss = -torch.min(unclipped, clipped).mean()

    # --- Value loss, clipped to ±value_clip around V_old (§9.2) ---
    value = policy.value(h[:, 0]).float()
    v_clipped = value_old + (value - value_old).clamp(-cfg.value_clip, cfg.value_clip)
    value_loss = torch.max(
        F.mse_loss(value, value_target, reduction="none"),
        F.mse_loss(v_clipped, value_target, reduction="none"),
    ).mean()

    entropy_mean = entropy.mean()
    
    # 🌟 修正 2：這是最終的 loss！不會再往裡面加未 clip 的 KL Penalty 了
    total = policy_loss + cfg.c_value * value_loss - cfg.c_entropy * entropy_mean

    # --- KL metrics (只做統計，不參與梯度計算) ---
    kl = torch.zeros((), device=logp.device)
    if logp_ref is not None:
        if logp_ref.shape != logp.shape:
            raise ValueError(
                f"reference policy produced {tuple(logp_ref.shape)} log-probs for "
                f"{tuple(logp.shape)} decision points"
            )
        # 🌟 修正 3：全部包在 no_grad 裡面，單純計算 KL 數據，不加進 total
        with torch.no_grad():
            d = logp - logp_ref
            kl = d.mean()
            stats.kl_to_il_k3 = float(((-d).exp() - 1.0 + d).mean())

    with torch.no_grad():
        stats.policy_loss = float(policy_loss)
        stats.value_loss = float(value_loss)
        stats.entropy = float(entropy_mean)
        stats.kl_to_il = float(kl)
        stats.beta = beta
        stats.clip_fraction = float(
            ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
        )
        log_ratio = logp - logp_old
        stats.approx_kl_step = float((log_ratio.exp() - 1.0 - log_ratio).mean())
        stats.explained_variance = explained_variance(value, value_target)

    return total, stats


# ── Search distillation loss (RL_SPEC §8.3, Phase 3d) ──────────────────────

def search_distillation_loss(
    policy,
    batch: dict[str, torch.Tensor],
    search_targets: list,
    cfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute ``L_search`` for states that have MCTS distillation targets.

    Only the states whose index is in ``search_targets`` contribute to the
    loss; all other states carry zero contribution.

    Parameters
    ----------
    policy : Policy
    batch : dict
        Batched featurized tensors (the full rollout buffer).
    search_targets : list of SearchTarget or None
        One ``SearchTarget`` per tagged decision point.  Any entry that is
        ``None`` or missing is skipped.

    Returns
    -------
    loss : Tensor
        Scalar loss, differentiable w.r.t. policy parameters.
    info : dict
        ``{"search/pi_ce": ..., "search/v_mse": ...}`` for logging.
    """
    if not search_targets or all(t is None for t in search_targets):
        return torch.zeros((), device=next(policy.parameters()).device), {}

    import torch.nn.functional as F
    import numpy as np

    device = next(policy.parameters()).device
    h, _history_h = policy._encode(batch)
    logits, _ = policy.pointer(h, batch["tok_mask"],
                                policy.embed.card, batch)
    value = policy.value(h[:, 0])  # [B, 1]

    total_pi = torch.zeros((), device=device)
    total_v = torch.zeros((), device=device)
    n_pi = 0
    n_v = 0

    for i, target in enumerate(search_targets):
        if target is None:
            continue

        # -- Policy distillation: CE to visit distribution π̃ ────────
        if target.visit_distribution is not None and len(target.visit_distribution) > 0:
            pi_tilde = torch.as_tensor(
                np.asarray(target.visit_distribution, dtype=np.float32)
            ).to(device)

            # Mask and softmax the model's logits
            mask = batch["opt_mask"][i].bool()
            masked_logits = logits[i].float().masked_fill(~mask, float("-inf"))
            logp = F.log_softmax(masked_logits, dim=-1)

            # CE over only the legal options populated by π̃
            n_legal = min(len(pi_tilde), int(mask.sum()))
            if n_legal > 0:
                ce = -(pi_tilde[:n_legal] * logp[:n_legal]).sum()
                total_pi = total_pi + ce
                n_pi += 1

        # -- Value distillation: MSE to root value Ṽ ──────────────────
        if target.root_value is not None:
            v_tilde = torch.tensor(target.root_value, device=device)
            mse = F.mse_loss(value[i, 0], v_tilde)
            total_v = total_v + mse
            n_v += 1

    loss_pi = total_pi / max(n_pi, 1) if n_pi > 0 else torch.zeros((), device=device)
    loss_v = total_v / max(n_v, 1) if n_v > 0 else torch.zeros((), device=device)

    loss = cfg.mcts_c_pi * loss_pi + cfg.mcts_c_v * loss_v

    info = {}
    if n_pi > 0:
        info["search/pi_ce"] = float(loss_pi.detach())
    if n_v > 0:
        info["search/v_mse"] = float(loss_v.detach())
    info["search/n_tagged"] = len([t for t in search_targets if t is not None])

    return loss, info