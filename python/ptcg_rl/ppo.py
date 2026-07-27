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
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ptcg_rl.actor import recompute_logp
from ptcg_rl.config import RLConfig
from ptcg_rl.rollout import normalize_advantages

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
    reference_policy,
    batch: dict[str, torch.Tensor],
    *,
    advantage: torch.Tensor,
    value_target: torch.Tensor,
    value_old: torch.Tensor,
    logp_old: torch.Tensor,
    beta: float,
    cfg: RLConfig,
    check_canary: bool = False,
) -> tuple[torch.Tensor, PPOStats]:
    """One minibatch's total loss and its diagnostics.

    *reference_policy* is the frozen π_IL for this deck: loaded once, never
    updated, evaluated under ``no_grad``.  Pass ``None`` to drop the anchor —
    useful for ablations, but it removes the forgetting guard entirely.

    *logp_old* must have been produced by :func:`recompute_logp` under **this**
    precision path, not restored from the rollout buffer (§7).
    """
    # One encode serves both heads. `recompute_logp` and the value head each
    # need the encoded state, and running the transformer twice would double the
    # cost of every PPO minibatch for nothing.
    h, _history = policy._encode(batch)
    logp, entropy = recompute_logp(policy, batch, encoded=h)

    # --- Clipped surrogate on the joint sequence ---
    ratio = (logp - logp_old).exp()
    stats = PPOStats(n_samples=int(logp.shape[0]))
    if check_canary:
        stats.ratio_p99 = check_ratio_canary(ratio, cfg.ratio_canary_tol)

    adv = normalize_advantages(advantage)
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()

    # --- Value loss, clipped to ±value_clip around V_old (§9.2) ---
    value = policy.value(h[:, 0]).float()
    v_clipped = value_old + (value - value_old).clamp(-cfg.value_clip, cfg.value_clip)
    value_loss = torch.max(
        F.mse_loss(value, value_target, reduction="none"),
        F.mse_loss(v_clipped, value_target, reduction="none"),
    ).mean()

    entropy_mean = entropy.mean()

    total = policy_loss + cfg.c_value * value_loss - cfg.c_entropy * entropy_mean

    # --- KL anchor to the frozen IL policy ---
    kl = torch.zeros((), device=logp.device)
    if reference_policy is not None:
        with torch.no_grad():
            logp_ref, _ = recompute_logp(reference_policy, batch)
        # Sample estimate of the mode-seeking KL(π_θ ‖ π_IL) on the actions
        # taken.  Same mask, same precision path, same decision points — §9.3
        # says assert that rather than assume it.
        if logp_ref.shape != logp.shape:
            raise ValueError(
                f"reference policy produced {tuple(logp_ref.shape)} log-probs for "
                f"{tuple(logp.shape)} decision points"
            )
        d = logp - logp_ref
        kl = d.mean()
        total = total + beta * kl
        with torch.no_grad():
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
        # Schulman's low-variance estimator for KL(θ_old ‖ θ); always ≥ 0,
        # unlike the naive -mean(logp - logp_old).
        log_ratio = logp - logp_old
        stats.approx_kl_step = float((log_ratio.exp() - 1.0 - log_ratio).mean())
        stats.explained_variance = explained_variance(value, value_target)

    return total, stats
