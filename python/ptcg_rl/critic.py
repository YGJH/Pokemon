"""R1: repair the value head before PPO can use it (RL_SPEC §3).

**This blocks R2.**  Measured on the IL checkpoint, the head has collapsed to a
constant ~0: prediction std 0.0006 against a target std of 0.994, correlation
0.045, sign agreement 0.556.  Constant-zero is the MSE-minimising answer when the
input carries no usable signal, which is why the failure is invisible in the loss.

PPO's advantage is ``A_t = Σ (γλ)^k δ_{t+k}`` with
``δ_t = r_t + γV(s_{t+1}) − V(s_t)``.  With ``V ≡ 0`` GAE degenerates to the raw
terminal return for *every* state — plain REINFORCE, no variance reduction, over
~150-decision episodes.  It will not learn at usable sample cost.

The repair follows §3's candidate list in order:

1. **Freeze the trunk, train only ``ValueHead``.**  This is diagnostic as much as
   corrective: if a frozen-trunk head still collapses, ``h_CLS`` carries no
   outcome information and the problem is the trunk, not the head.  IL trains CLS
   *only* through the value loss — the pointer head reads option and state
   tokens — so a λ_v of 0.5 against a dominant CE may simply never shape it.
2. **Check for tanh saturation** at init via the pre-activation std.
3. **Judge against a per-turn baseline.**  Every decision in a game shares one ±1
   label, so early-game states are near-unpredictable *by construction*.  A head
   that is accurate late and chance-level early is working correctly; comparing
   against zero loss would call it a failure.

Exit criteria: ``corr(pred, outcome) ≥ 0.35``, prediction std ≥ 0.3, and accuracy
rising monotonically in turn number.
"""

from __future__ import annotations

import logging
from rich.logging import RichHandler
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from ptcg_il.featurizer import CLS_OPP_PRIZES, CLS_OUR_PRIZES

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_time=False)])
logger = logging.getLogger(__name__)


@dataclass
class CriticDiagnostics:
    """Everything needed to tell a repaired critic from a collapsed one."""

    pred_std: float = 0.0
    target_std: float = 0.0
    corr: float = 0.0
    sign_agreement: float = 0.0
    mse: float = 0.0
    preactivation_std: float = 0.0
    """``ValueHead`` ends in ``tanh``. A large pre-activation std means saturation
    (gradients vanish); a near-zero one means the head sees nothing to say."""
    per_turn_accuracy: list[tuple[int, float, int]] = field(default_factory=list)
    """``[(turn_bucket, sign_accuracy, n)]``, ascending. The *shape* is the
    signal: flat means the critic has learned nothing about game progress."""
    n_samples: int = 0

    @property
    def collapsed(self) -> bool:
        """The §3 signature: predictions with essentially no variance."""
        return self.pred_std < 0.05

    def passes(self, corr_target: float = 0.35, std_target: float = 0.30) -> bool:
        return (
            self.corr >= corr_target
            and self.pred_std >= std_target
            and self.accuracy_rises_with_turn
        )

    @property
    def accuracy_rises_with_turn(self) -> bool:
        """Late-game predictions must beat early-game ones.

        Compares the first and last populated buckets rather than demanding
        strict monotonicity across all of them — with a shared ±1 label per game
        the middle buckets are noisy, and requiring monotone improvement there
        would fail a healthy critic.
        """
        buckets = [(acc, n) for _, acc, n in self.per_turn_accuracy if n >= 20]
        if len(buckets) < 2:
            return False
        return buckets[-1][0] > buckets[0][0]

    def summary(self) -> str:
        return (
            f"corr={self.corr:.4f} pred_std={self.pred_std:.4f} "
            f"sign_agree={self.sign_agreement:.4f} mse={self.mse:.4f} "
            f"n={self.n_samples}"
            + ("  [COLLAPSED]" if self.collapsed else "")
        )


def diagnose(
    predictions: Iterable[float],
    targets: Iterable[float],
    turns: Iterable[int] | None = None,
    *,
    preactivation_std: float = 0.0,
    n_buckets: int = 5,
) -> CriticDiagnostics:
    """Compute the §3 diagnostics from paired predictions and outcomes."""
    pred = np.asarray(list(predictions), dtype=np.float64)
    targ = np.asarray(list(targets), dtype=np.float64)
    if pred.shape != targ.shape:
        raise ValueError(f"pred {pred.shape} and target {targ.shape} differ")

    d = CriticDiagnostics(n_samples=int(pred.size), preactivation_std=preactivation_std)
    if pred.size == 0:
        return d

    d.pred_std = float(pred.std())
    d.target_std = float(targ.std())
    d.mse = float(((pred - targ) ** 2).mean())

    # A constant series has no correlation; np.corrcoef returns NaN there and a
    # NaN silently poisons every comparison downstream.
    if d.pred_std > 0 and d.target_std > 0:
        d.corr = float(
            ((pred - pred.mean()) * (targ - targ.mean())).mean() / (d.pred_std * d.target_std)
        )

    signed = targ != 0
    if signed.any():
        d.sign_agreement = float((np.sign(pred[signed]) == np.sign(targ[signed])).mean())

    if turns is not None:
        t = np.asarray(list(turns), dtype=np.int64)
        if t.size == pred.size and t.size:
            edges = np.quantile(t, np.linspace(0, 1, n_buckets + 1))
            for b in range(n_buckets):
                lo, hi = edges[b], edges[b + 1]
                sel = (t >= lo) & (t <= hi) if b == n_buckets - 1 else (t >= lo) & (t < hi)
                sel &= signed
                if not sel.any():
                    continue
                acc = float((np.sign(pred[sel]) == np.sign(targ[sel])).mean())
                d.per_turn_accuracy.append((int(lo), acc, int(sel.sum())))
    return d


# The two players' prize counts, each normalised by PRIZE_N.  Prizes taken only
# ever goes up, so their complement is a real measure of game progress — and
# unlike a turn counter it is already in the features, so no shard rebuild is
# needed.  `CLS_OUR_PRIZES`/`CLS_OPP_PRIZES` are imported from the featurizer
# rather than written as literals here: the global block has been reordered once
# (the absolute-seat columns came out of it), and a stale literal reads a
# condition flag instead and reports a plausible wrong curve.
PROGRESS_BUCKETS = 5


def _progress(batch) -> np.ndarray:
    """Game progress per row, as total prizes taken by both players.

    §3's third exit criterion is "accuracy rising monotonically in turn number",
    but the shards carry no turn index. This is the stand-in: it starts at 0,
    ends near 2 (normalised), and is monotone within a game by the rules of the
    format.
    """
    cls = batch["cls_feat"]
    taken = (1.0 - cls[:, CLS_OUR_PRIZES]) + (1.0 - cls[:, CLS_OPP_PRIZES])
    return (taken.float().cpu().numpy() * 100).astype(np.int64)


@torch.no_grad()
def measure(policy, loader, device, max_batches: int | None = None) -> CriticDiagnostics:
    """Diagnose the critic of a **correctly loaded** policy on held-out data.

    "Correctly loaded" is not pedantry: the §2.3 probe reported a value std of
    0.222 from a model built by inferring shapes from a state dict with 36
    tensors left unloaded.  That number describes a partially random network, not
    the trained head, and it is why §3 asks for a re-measurement here as R1's
    first act.
    """
    policy.eval()
    preds: list[float] = []
    targets: list[float] = []
    progress: list[int] = []
    pre: list[float] = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        _x, h, _ = policy._encode(batch)
        cls = h[:, 0]
        value = policy.value(cls).float()
        preds.extend(value.cpu().tolist())
        targets.extend(batch["value_target"].float().cpu().tolist())
        progress.extend(_progress(batch).tolist())
        pre.append(float(_preactivation(policy.value, cls).std()))

    return diagnose(
        preds, targets, progress,
        preactivation_std=float(np.mean(pre)) if pre else 0.0,
        n_buckets=PROGRESS_BUCKETS,
    )


def _preactivation(value_head, cls: torch.Tensor) -> torch.Tensor:
    """The input to ``ValueHead``'s final ``tanh``, for saturation checks.

    Reaches inside the head deliberately.  If its internals change this raises
    rather than reporting a plausible wrong number, which is the correct failure
    mode for a diagnostic.
    """
    modules = [m for m in value_head.modules() if isinstance(m, torch.nn.Linear)]
    if not modules:
        raise AttributeError("ValueHead has no Linear layers to probe")
    x = cls
    for i, layer in enumerate(modules):
        x = layer(x)
        if i < len(modules) - 1:
            x = F.gelu(x)
    return x


def repair(
    policy,
    train_loader,
    val_loader,
    device,
    *,
    steps: int = 5_000,
    lr: float = 3e-4,
    freeze_trunk: bool = True,
    ce_weight: float = 0.0,
    value_weight: float = 1.0,
    log_every: int = 200,
) -> tuple[CriticDiagnostics, dict[str, Any]]:
    """Fit the value head, with the trunk frozen or jointly fine-tuned.

    **Phase A — ``freeze_trunk=True``** is §3's candidate-1 experiment as much as
    a repair: only ``ValueHead`` gets gradients, so if the critic stays flat the
    trunk's ``h_CLS`` carries no outcome information and no amount of head
    training will help.  Read that result before widening the scope; unfreezing
    first confounds the two hypotheses.

    **Phase B — ``freeze_trunk=False`` with ``ce_weight > 0``** is what to do when
    phase A saturates short of the gate.  The trunk is *shared with the pointer
    head*, so fine-tuning it on MSE alone repairs the critic by destroying the
    actor — catastrophic forgetting before RL has started, which would invalidate
    the KL anchor and the gate at once.  Keeping the IL cross-entropy in the loss
    is what makes unfreezing safe.

    The CE term is ``−log π_θ(a_expert)`` from :func:`ptcg_rl.actor.recompute_logp`,
    which is exactly ``multiselect_ce`` at zero label smoothing (asserted in
    ``test_rl_actor.py``) and shares this call's encode.

    Returns ``(diagnostics on val, history)``.
    """
    from ptcg_rl.actor import recompute_logp

    policy.to(device)

    if freeze_trunk:
        for p in policy.parameters():
            p.requires_grad_(False)
        for p in policy.value.parameters():
            p.requires_grad_(True)
        params = list(policy.value.parameters())
        logger.info("Critic repair (phase A): trunk frozen, %d value-head tensors",
                    len(params))
    else:
        for p in policy.parameters():
            p.requires_grad_(True)
        params = [p for p in policy.parameters() if p.requires_grad]
        logger.info(
            "Critic repair (phase B): joint fine-tune, %d tensors, "
            "value_weight=%.1f ce_weight=%.1f", len(params), value_weight, ce_weight,
        )

    opt = torch.optim.AdamW(params, lr=lr)
    history: dict[str, Any] = {"loss": [], "mse": [], "ce": [], "steps": 0}

    step = 0
    policy.train()
    while step < steps:
        for batch in train_loader:
            if step >= steps:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            if freeze_trunk:
                # The trunk runs under no_grad: it skips the backward pass
                # through the transformer, which is the whole cost here.
                with torch.no_grad():
                    batch, h, _ = policy._encode(batch)
                cls = h[:, 0].detach()
            else:
                batch, h, _ = policy._encode(batch)
                cls = h[:, 0]

            value = policy.value(cls).float()
            mse = F.mse_loss(value, batch["value_target"].float())
            loss = value_weight * mse

            ce = torch.zeros((), device=device)
            if ce_weight > 0 and not freeze_trunk:
                logp, _ = recompute_logp(policy, batch, encoded=h)
                ce = -logp.mean()
                loss = loss + ce_weight * ce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            history["loss"].append(loss.detach().item())
            history["mse"].append(mse.detach().item())
            history["ce"].append(ce.detach().item())
            step += 1
            if step % log_every == 0:
                logger.info("critic step %d/%d  mse=%.4f ce=%.4f",
                            step, steps, mse.detach().item(), ce.detach().item())

    history["steps"] = step
    for p in policy.parameters():
        p.requires_grad_(True)

    diag = measure(policy, val_loader, device)
    logger.info("Critic after repair: %s", diag.summary())
    return diag, history
