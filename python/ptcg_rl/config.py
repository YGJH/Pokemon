"""``RLConfig`` — every knob in RL_SPEC §9.4, §8.3 and §10, in one place.

Defaults are the spec's starting points, not tuned values.  Two of them are
load-bearing enough to restate here:

* ``lr`` is **an order of magnitude below IL's 3e-4**.  This is fine-tuning a
  policy that already plays, not fitting one from scratch.
* ``kappa`` is fixed at 0.02 nats/decision through R2 by decision (§9.3).  ``beta``
  adapts to *hold* that budget; the budget itself does not move.  A moving κ and
  a moving lr are the two things most likely to be blamed for a failure, and a
  curriculum would make R2's go/no-go unattributable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RLConfig:
    """Hyperparameters for the PPO stage."""

    # ── Identity ────────────────────────────────────────────────────────────
    deck_archetype: int = 0
    """Which 𝒟_self archetype's specialist is being trained."""

    # ── Rollout ─────────────────────────────────────────────────────────────
    rollout_buffer: int = 16_384
    """Decision points collected per PPO iteration (~110 games).

    Distinct from ``forward_batch``.  Revision 1 of the spec conflated the two
    and arrived at a 16,384-wide forward pass, which is 33 ms of latency for no
    throughput gain (§2.3).
    """
    forward_batch: int = 512
    """Inference batch size.  Throughput saturates at 256–512 (§2.3, measured)."""
    n_workers: int = 6
    """Rollout worker processes.  Six saturate the GPU's 61.6k dec/s (§6.5)."""
    max_decisions_per_game: int = 600
    """Safety valve; real games run ~150 decisions."""

    # ── Returns ─────────────────────────────────────────────────────────────
    gamma: float = 1.0
    """No discounting: episodes are finite and a win now is a win later."""
    gae_lambda: float = 0.95

    # ── PPO ─────────────────────────────────────────────────────────────────
    ppo_epochs: int = 2
    """Low, because the KL anchor is already regularising."""
    minibatch: int = 1_024
    lr: float = 1e-5
    clip_eps: float = 0.2
    value_clip: float = 0.2
    c_value: float = 0.5
    c_entropy: float = 0.003
    """Masked AR entropy sums over steps, so it exceeds single-token entropy.
    Start low."""
    grad_clip: float = 1.0
    total_steps: int = 50_000

    # ── KL anchor to the frozen IL policy (§9.3) ────────────────────────────
    kappa: float = 0.02
    """Target KL in nats per decision point.  Fixed through R2."""
    beta_init: float = 0.1
    beta_min: float = 1e-4
    beta_max: float = 10.0
    beta_lr: float = 0.1
    """η_β in β ← clip(β · exp((KL − κ)/κ · η_β), β_min, β_max)."""

    # ── Precision (§7) ──────────────────────────────────────────────────────
    bf16_rollout: bool = True
    """bf16 is 1.57× and preserves 98.7% of top-1. Safe *within* one precision."""
    ratio_canary_tol: float = 1e-3
    """Epoch-0 p99 |ratio − 1| must be under this. Fails loudly (§7)."""

    # ── Gate (§10.2) ────────────────────────────────────────────────────────
    gate_paired_games: int = 400
    gate_score: float = 0.60
    """R2's exit bar is ≥60% vs frozen π_IL. The ≥70% of §10.2 is R4's promotion
    gate, which this phase does not reach."""
    gate_wilson_lb: float = 0.55
    gate_il_regression_pts: float = 5.0
    """Non-trivial top-1 may not fall more than this many points below baseline."""
    gate_kl_multiple: float = 4.0
    """KL at evaluation temperature may not exceed ``kappa × this``."""

    # ── Critic repair (R1, §3) ──────────────────────────────────────────────
    critic_steps: int = 5_000
    critic_lr: float = 3e-4
    """Higher than the PPO lr: this fits a head from scratch, it does not
    fine-tune one."""
    critic_freeze_trunk: bool = True
    """§3 candidate 1: if a frozen-trunk head still collapses, the trunk is the
    problem and no amount of head training will fix it."""
    critic_corr_target: float = 0.35
    critic_std_target: float = 0.30

    # -- Phase B: joint fine-tune, when the frozen-trunk head saturates short --
    critic_joint_steps: int = 4_000
    """Measured on this corpus, the frozen-trunk head plateaus at corr ≈ 0.17–0.20:
    16× more steps bought +0.04. That is §3 candidate 1 answering *yes, the trunk
    is the bottleneck* — so phase B unfreezes it."""
    critic_joint_lr: float = 5e-5
    """Well below the head-only lr. This is fine-tuning a trained trunk, and the
    trunk is shared with the pointer head."""
    critic_value_weight: float = 5.0
    """Far above IL's λ_v = 0.5. That weighting is *why* h_CLS never learned to
    carry the outcome: a dominant CE left the value term with no say in shaping
    CLS."""
    critic_ce_weight: float = 1.0
    """Keeps the IL objective in the loss. Without it, fine-tuning the shared
    trunk on MSE alone destroys the policy — catastrophic forgetting before RL
    has even started, which would invalidate both the KL anchor and the gate."""
    critic_policy_regression_pts: float = 3.0
    """Abort phase B if non-trivial top-1 falls more than this far. A repaired
    critic bought by wrecking the actor is not progress."""

    # ── Bookkeeping ─────────────────────────────────────────────────────────
    seed: int = 0
    log_every: int = 10
    save_every: int = 500
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rollout_buffer % self.minibatch != 0:
            raise ValueError(
                f"rollout_buffer ({self.rollout_buffer}) must be a multiple of "
                f"minibatch ({self.minibatch}); a ragged tail minibatch changes "
                f"the advantage normalisation for those samples only"
            )
        if not 0.0 < self.kappa:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if not self.beta_min <= self.beta_init <= self.beta_max:
            raise ValueError(
                f"beta_init {self.beta_init} outside [{self.beta_min}, {self.beta_max}]"
            )
        if self.gamma > 1.0 or self.gamma <= 0.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")

    def to_dict(self) -> dict:
        return asdict(self)
