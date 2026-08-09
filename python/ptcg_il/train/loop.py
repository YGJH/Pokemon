"""Training step, optimizer/scheduler builder, main training loop (C.5–C.6).

Implements:
- ``create_optimizer``: AdamW with per-param no-decay set
- ``create_schedule``: linear warmup → cosine decay
- ``train_step``: one optimizer step (forward + backward + opt + ema)
- ``train``: full training loop with logging, checkpointing, eval
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ptcg_il.deck import build_deck_metadata
from ptcg_il.deck import describe as describe_deck
from ptcg_il.deck import require_deck_record, update_sidecar, write_deck_csv
from ptcg_il.belief_labels import BELIEF_WEIGHTS  # noqa: F401  (re-exported for the CLI)
from ptcg_il.model.belief import belief_loss
from ptcg_il.model.policy import Policy, load_policy_state, multiselect_ce
from ptcg_il.train.checkpoint import save_checkpoint
from ptcg_il.train.dataset import ShardDataset, collate_fn
from ptcg_il.train.eval import offline_eval
from ptcg_il.train.logger import WandbLogger

logger = logging.getLogger(__name__)

# ============================================================
# Default hyperparameters (C.10)
# ============================================================
BATCH: int = 2048
PEAK_LR: float = 3e-4
MIN_LR: float = 3e-5
WARMUP: int = 1000
GRAD_CLIP: float = 1.0
LABEL_SMOOTH: float = 0.05
EMA_DECAY: float = 0.999
LAMBDA_V: float = 0.5
WEIGHT_DECAY: float = 0.01
BETAS: tuple[float, float] = (0.9, 0.95)

#: Peak LR for the orthogonalized (Muon) parameter group.  Deliberately a
#: separate constant from PEAK_LR: an orthogonalized update's magnitude is set
#: by the matrix shape, not by the gradient, so the two groups live on different
#: scales and this value does not follow PEAK_LR when that is tuned.
#:
#: Swept on archetype 25, 5000 steps, B=1024, seed 42, scored on the best
#: ``val/top1_nontrivial`` reached (the metric that selects ``ckpt-best``):
#:
#:   0.003 -> 0.6654 | 0.007 -> 0.6679 | **0.015 -> 0.6697** | 0.03 -> 0.6454 |
#:   0.05 -> 0.6010  (AdamW baseline 0.6420)
#:
#: The optimum is interior to the grid and the curve falls off on both sides, so
#: this is a peak rather than an edge.  Re-sweep if the model width, batch size
#: or corpus changes — none of those leave an orthogonalized update's scale
#: alone.
MUON_LR: float = 0.015
MUON_MOMENTUM: float = 0.95
LOG_EVERY: int = 50
VAL_EVERY: int = 1000
CKPT_EVERY: int = 2000


# No-decay parameter name patterns
_NO_DECAY_PATTERNS = (
    "bias",
    "LayerNorm",
    "layer_norm",
    "layernorm",
    "norm",
    "ln_",
    "Embedding",
    "embedding",
    "emb",
    "null_token",
    "no_stadium",
)


def _no_decay(n: str, p: nn.Parameter) -> bool:
    """Return True if parameter *p* named *n* should NOT be weight-decayed."""
    return any(pat in n for pat in _NO_DECAY_PATTERNS)


def masked_label_smoothed_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.05,
    target_group: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy with label smoothing that respects an option mask.

    Unlike ``F.cross_entropy(logits, targets, label_smoothing=...)``, the
    smoothing distribution is uniform over the **valid** (masked) positions
    only — padded positions contribute zero.  This avoids the catastrophic
    log-prob penalty when padded logits are ``-1e9``.

    Parameters
    ----------
    logits : Tensor[B, O]
        Per-option logits (can have ``-1e9`` for padded positions).
    targets : int64 Tensor[B]
        True option index for each sample.
    mask : bool Tensor[B, O]
        ``True`` = valid option.
    label_smoothing : float
    target_group : bool Tensor[B, O] or None
        When given, the NLL is taken over the *summed* probability of the
        target's equivalence class instead of the target alone.  Options in a
        class are byte-identical inputs to the pointer head, so a plain CE asks
        the model to rank one above the others using information it does not
        have; on this corpus that is 13.9% of samples.  Each row must have at
        least one True (the target itself) or the logsumexp underflows to -inf.
        None reproduces the ungrouped behaviour exactly.

    Returns
    -------
    ce : Tensor[B]
        Per-sample cross-entropy (not reduced).
    """
    # Masked log-softmax — softmax is computed only over valid options
    logits_inf = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(logits_inf, dim=-1)  # [B, O]; -inf at padded slots

    # NLL
    if target_group is None:
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    else:
        nll = -torch.logsumexp(
            log_probs.masked_fill(~target_group, float("-inf")), dim=-1
        )

    if label_smoothing > 0:
        n_valid = mask.sum(dim=-1).float().clamp(min=1)  # [B]
        # Sum of log-probs over valid positions
        log_prob_sum = log_probs.masked_fill(~mask, 0.0).sum(dim=-1)  # [B]
        avg_log_prob = log_prob_sum / n_valid
        # Smoothed CE: (1-eps)*NLL - eps * avg_log_prob
        ce = (1.0 - label_smoothing) * nll - label_smoothing * avg_log_prob
    else:
        ce = nll

    return ce


def target_group_mask(
    opt_group: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """[B, O] bool marking options indistinguishable from each row's target.

    ``opt_group`` is -1 on padded slots, so those never join a group.  The
    target's own slot is always included because it is valid by construction.
    """
    g = opt_group.gather(1, targets.unsqueeze(1))          # [B, 1]
    return (opt_group == g) & mask & (opt_group >= 0)


#: Smallest out-features a 2D parameter may have and still be orthogonalized.
#:
#: The partition is on ``p.size(-2)`` (out-features for an ``nn.Linear`` weight)
#: rather than on ``min(shape)``, because those disagree on exactly the cases
#: that matter.  ``embed.hand_mlp.0.weight`` is ``[256, 7]`` — a real 7->256 map
#: whose seven singular values are worth flattening, and whose narrowness the
#: aspect-ratio scale factor already accounts for.  ``arch_head.2.weight`` is
#: ``[9, 256]`` — nine logits, where "orthogonalize" means "rescale the head".
#: A ``min(shape)`` threshold cannot separate 7 from 9; out-features separates
#: 256 from 9 with room to spare.
#:
#: This is a guard as much as a rule: a head added later is caught by it instead
#: of being silently orthogonalized.
MIN_MUON_OUT_FEATURES: int = 16


def stacked_slice_counts(policy: nn.Module) -> dict[int, int]:
    """``id(param) -> number of independent maps concatenated inside it``.

    Two module types in this model fuse several linear maps into one tensor on
    the output axis, and on the D=256/8-layer configuration they are **28.6% of
    all parameters**:

    * ``nn.MultiheadAttention.in_proj_weight`` — ``[3D, D]``, i.e. Q, K and V
      (9 of them, 19.8%);
    * ``nn.GRU`` / ``nn.GRUCell`` ``weight_ih*`` / ``weight_hh*`` — ``[3H, *]``,
      i.e. the reset, update and new gates (4 of them, 8.8%).

    Orthogonalizing such a tensor whole is not what Muon means: the stack has
    rank at most its width, so Newton-Schulz flattens a *joint* spectrum and the
    three blocks stop being independently conditioned.  Detection is by module
    type rather than by shape, because ``[768, 256]`` is only "three stacked
    maps" by virtue of what owns it — a plain ``nn.Linear`` of the same shape is
    one map and must not be split.
    """
    counts: dict[int, int] = {}
    for m in policy.modules():
        if isinstance(m, nn.MultiheadAttention):
            if getattr(m, "in_proj_weight", None) is not None:
                counts[id(m.in_proj_weight)] = 3
        elif isinstance(m, (nn.GRU, nn.GRUCell)):
            for n, p in m.named_parameters(recurse=False):
                if n.startswith(("weight_ih", "weight_hh")):
                    counts[id(p)] = 3
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            for n, p in m.named_parameters(recurse=False):
                if n.startswith(("weight_ih", "weight_hh")):
                    counts[id(p)] = 4
    return counts


def partition_parameters(policy: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Split parameters into ``muon`` / ``adamw_decay`` / ``adamw_no_decay``.

    Muon takes 2D parameters that are genuine linear maps.  Everything else
    stays on AdamW: ``nn.Embedding`` tables (rows are lookups, not a map), 1D
    biases and norm gains (no spectrum), and output heads (see
    :data:`MIN_MUON_OUT_FEATURES`).

    Every parameter lands in exactly one list.  That is not automatic here —
    one ``CardEncoder`` is aliased into ``pointer.card``, ``belief.card_emb``
    and ``belief_heads``, so the same tensors are reachable under several names
    — and a tensor in two groups would be updated twice per step.  Dedup is by
    ``id``, not by name.
    """
    embedding_ids = {
        id(p) for m in policy.modules() if isinstance(m, nn.Embedding)
        for p in m.parameters()
    }

    groups: dict[str, list[nn.Parameter]] = {
        "muon": [], "adamw_decay": [], "adamw_no_decay": [],
    }
    seen: set[int] = set()
    for n, p in policy.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))

        eligible = (
            p.ndim == 2
            and id(p) not in embedding_ids
            and p.size(-2) >= MIN_MUON_OUT_FEATURES
        )
        if eligible:
            groups["muon"].append(p)
        elif _no_decay(n, p):
            groups["adamw_no_decay"].append(p)
        else:
            groups["adamw_decay"].append(p)
    return groups


def create_optimizer(
    policy: nn.Module,
    peak_lr: float = PEAK_LR,
    weight_decay: float = WEIGHT_DECAY,
    betas: tuple[float, float] = BETAS,
    optimizer: str = "adamw",
    muon_lr: float = MUON_LR,
    muon_momentum: float = MUON_MOMENTUM,
) -> torch.optim.Optimizer:
    """Build the optimizer with the per-param no-decay set (C.5).

    ``optimizer="adamw"`` (default) is the original two-group AdamW and is the
    baseline every Muon result is measured against, so it is left byte-for-byte
    as it was.

    ``optimizer="muon"`` puts the trunk matrices on orthogonalized momentum and
    everything else on AdamW, inside a single :class:`~ptcg_il.train.muon.Muon`
    object — one optimizer means one ``LambdaLR``, one state dict and no changes
    to ``train_step`` or the resume path.  The two halves get **independent base
    learning rates**: an orthogonalized update's size is set by the matrix shape
    rather than by the gradient, so ``muon_lr`` does not transfer from
    ``peak_lr`` and is swept separately.  The schedule scales both.

    Parameters
    ----------
    policy : nn.Module
        The Policy module.
    peak_lr : float
        Peak learning rate for the AdamW parameters.
    weight_decay : float
        Weight decay for decay-eligible params.
    betas : tuple
        AdamW beta coefficients.
    optimizer : str
        ``"adamw"`` or ``"muon"``.
    muon_lr : float
        Peak learning rate for the orthogonalized group.
    muon_momentum : float
        Momentum for the orthogonalized group.
    """
    if optimizer not in ("adamw", "muon"):
        raise ValueError(
            f"unknown optimizer {optimizer!r}; expected 'adamw' or 'muon'"
        )

    if optimizer == "adamw":
        decay_params = []
        no_decay_params = []

        for n, p in policy.named_parameters():
            if not p.requires_grad:
                continue
            if _no_decay(n, p):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(param_groups, lr=peak_lr, betas=betas)

    from ptcg_il.train.muon import Muon

    groups = partition_parameters(policy)
    slices = stacked_slice_counts(policy)

    # One Muon group per distinct slice count.  They share a learning rate and
    # differ only in how the update is carved up before orthogonalization.
    by_split: dict[int, list[nn.Parameter]] = {}
    for p in groups["muon"]:
        by_split.setdefault(slices.get(id(p), 1), []).append(p)

    muon_groups = [
        {"params": ps, "use_muon": True, "lr": muon_lr,
         "weight_decay": weight_decay, "momentum": muon_momentum, "split": n}
        for n, ps in sorted(by_split.items())
    ]

    n_muon = sum(p.numel() for p in groups["muon"])
    n_total = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_stacked = sum(p.numel() for n, ps in by_split.items() if n > 1 for p in ps)
    logger.info(
        "Muon: %d/%d parameters (%.1f%%) on orthogonalized momentum at lr=%.4g "
        "(%.1f%% of them in stacked QKV/GRU tensors, split per map); the rest on "
        "AdamW at lr=%.4g",
        n_muon, n_total, 100.0 * n_muon / max(n_total, 1), muon_lr,
        100.0 * n_stacked / max(n_muon, 1), peak_lr,
    )

    return Muon(
        muon_groups + [
            {"params": groups["adamw_decay"], "use_muon": False,
             "lr": peak_lr, "weight_decay": weight_decay, "betas": betas},
            {"params": groups["adamw_no_decay"], "use_muon": False,
             "lr": peak_lr, "weight_decay": 0.0, "betas": betas},
        ],
        lr=peak_lr,
        betas=betas,
    )


def require_matching_optimizer(ckpt: dict, optimizer: str) -> None:
    """Refuse to resume a checkpoint whose optimizer state is a different shape.

    Muon's trunk groups keep a single ``momentum_buffer``; AdamW keeps
    ``exp_avg``/``exp_avg_sq`` for everything.  ``Optimizer.load_state_dict``
    pairs groups by *position*, so the failure is not reliably an exception —
    with the group counts lined up it can load AdamW moments into slots a Muon
    step never reads, and the run continues from what looks like a resumed
    state and is really a cold optimizer.

    A checkpoint written before this field existed is treated as AdamW, which is
    what every checkpoint on disk was written by.
    """
    was = ckpt.get("optimizer", "adamw")
    if was != optimizer:
        raise ValueError(
            f"checkpoint was trained with --optimizer {was}, this run uses "
            f"--optimizer {optimizer}. Optimizer state cannot be carried across "
            f"the two: pass --optimizer {was} to continue the run, or start a "
            f"fresh one (drop --resume) to change optimizer."
        )


def create_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup: int = WARMUP,
    peak_lr: float = PEAK_LR,
    min_lr: float = MIN_LR,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup → cosine decay schedule (C.5).

    Parameters
    ----------
    optimizer : Optimizer
    total_steps : int
        Total number of training steps.
    warmup : int
        Number of linear warmup steps.
    peak_lr : float
        Peak learning rate.
    min_lr : float
        Minimum learning rate at end of cosine decay.

    Returns
    -------
    LRScheduler
    """

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        # Cosine decay
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        return (min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + np.cos(np.pi * progress))) / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class _EMA:
    """Exponential moving average of model parameters.

    Keeps a shadow copy of model parameters and updates with ``decay``.
    """

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.data, alpha=1.0 - d)

    def apply(self, model: nn.Module) -> None:
        """Copy EMA weights into *model* (in-place).

        Destructive — the raw training weights are lost unless they were
        captured first with :meth:`backup`.  Prefer :meth:`applied`.
        """
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def backup(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Snapshot *model*'s current (raw) weights so they can be restored."""
        return {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if n in self.shadow
        }

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Copy a :meth:`backup` back into *model* (in-place)."""
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    @contextmanager
    def applied(self, model: nn.Module):
        """Temporarily swap EMA weights into *model*, restoring on exit.

        Checkpointing and eval want the EMA weights, but training must resume
        from the raw weights — the optimizer's Adam moment estimates correspond
        to those, so overwriting them in place corrupts the trajectory.
        """
        backup = self.backup(model)
        self.apply(model)
        try:
            yield model
        finally:
            self.restore(model, backup)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict, model: nn.Module | None = None) -> None:
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]
        if model is not None:
            # A checkpoint written before a head existed carries no shadow
            # entry for its parameters.  `update`/`apply` both skip unknown
            # names, so without this back-fill those weights would never be
            # averaged and `applied()` would evaluate a half-EMA model.
            for n, p in model.named_parameters():
                if p.requires_grad and n not in self.shadow:
                    self.shadow[n] = p.data.clone().detach()


def _compute_loss(
    policy: Policy,
    batch: dict[str, torch.Tensor],
    lambda_v: float = LAMBDA_V,
    label_smoothing: float = LABEL_SMOOTH,
    use_amp: bool = True,
    belief_weights: dict[str, float] | None = None,
    group_marginal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Forward pass + loss only (no backward, no optimizer step).

    Returns ``(loss, ce, value_mse_loss, belief_parts)`` where *loss* is a
    scalar tensor ready for ``.backward()`` and *belief_parts* holds the
    per-term belief scalars for logging (empty when the belief loss is off).
    Caller is responsible for scaling, backward, gradient clipping, and
    optimizer step.
    """
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    with (
        torch.amp.autocast(device_type, dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    ):
        # forward_with_belief shares the encode pass; calling policy() and then
        # policy.belief_logits() would run the transformer twice per step.
        want_belief = bool(belief_weights) and any(belief_weights.values())
        if want_belief:
            logits, value, _history_h, belief_preds = policy.forward_with_belief(batch)
        else:
            logits, value, _history_h = policy(batch)  # [B, O], [B], [B, D]
            belief_preds = None

        maxcount = batch["maxCount"]
        single = maxcount == 1

        # Single-select rows need exactly one pointer pass; multi-select rows
        # need an autoregressive loop.  Run the AR loop on the multi-select
        # *subset* only — routing the whole batch through it (the previous
        # behaviour whenever any row was multi-select, i.e. nearly always) costs
        # max(action_len) full-batch pointer passes with all activations
        # retained, which is both wasted compute for the ~95% single-select
        # majority and the reason batch_size=1024 exhausts a 16GB GPU.
        # Optional single-select rows (minCount == 0) now carry a STOP column,
        # so an expert who declined is labelled with it and trains like any
        # other target.  ``has_target`` still guards the gather: shards written
        # before that change record declining as action_idx[:, 0] == -1, and a
        # CE gather with -1 trips a device-side assert rather than failing
        # cleanly.  ``multiselect_ce`` guards the same way via ``target >= 0``.
        has_target = batch["action_idx"][:, 0] >= 0
        single_ok = single & has_target

        opt_group = batch.get("opt_group")
        use_groups = group_marginal and opt_group is not None

        if single.all() and bool(has_target.all()):
            tgt = batch["action_idx"][:, 0]
            ce = masked_label_smoothed_ce(
                logits, tgt, batch["opt_mask"],
                label_smoothing=label_smoothing,
                target_group=(
                    target_group_mask(opt_group, tgt, batch["opt_mask"])
                    if use_groups else None
                ),
            )
        else:
            ce = logits.new_zeros(logits.shape[0])
            if single_ok.any():
                s_idx = torch.where(single_ok)[0]
                tgt = batch["action_idx"][s_idx, 0]
                ce = ce.index_put(
                    (s_idx,),
                    masked_label_smoothed_ce(
                        logits[s_idx], tgt, batch["opt_mask"][s_idx],
                        label_smoothing=label_smoothing,
                        target_group=(
                            target_group_mask(opt_group[s_idx], tgt, batch["opt_mask"][s_idx])
                            if use_groups else None
                        ),
                    ),
                )
            m_idx = torch.where(~single)[0]
            if m_idx.numel() > 0:
                sub = {
                    k: (
                        v[m_idx]
                        if isinstance(v, torch.Tensor) and v.shape[:1] == single.shape
                        else v
                    )
                    for k, v in batch.items()
                }
                ce = ce.index_put(
                    (m_idx,),
                    multiselect_ce(
                        policy, sub,
                        label_smoothing=label_smoothing,
                        group_marginal=group_marginal,
                    ),
                )

        loss = (batch["sample_weight"] * ce).mean()
        if lambda_v > 0:
            value_mse_loss = F.mse_loss(value, batch["value_target"])
            loss = loss + lambda_v * value_mse_loss
        else:
            value_mse_loss = torch.tensor(0.0, device=loss.device)

        belief_parts: dict[str, float] = {}
        if belief_preds is not None:
            # Every term is masked by bel_valid, so shards without labels
            # contribute exactly 0 rather than a bogus gradient.
            b_loss, belief_parts = belief_loss(belief_preds, batch, **belief_weights)
            loss = loss + b_loss
            belief_parts["belief/loss"] = float(b_loss.detach())

    return loss, ce, value_mse_loss, belief_parts


def train_step(
    policy: Policy,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    ema: _EMA,
    device: torch.device,
    grad_scaler: torch.amp.GradScaler | None = None,
    lambda_v: float = LAMBDA_V,
    label_smoothing: float = LABEL_SMOOTH,
    grad_clip: float = GRAD_CLIP,
    belief_weights: dict[str, float] | None = None,
    group_marginal: bool = True,
) -> dict[str, float]:
    """Run one optimizer step on *batch* (C.6).

    Parameters
    ----------
    policy : Policy
    batch : dict
        Batched featurizer tensors (from collate_fn).
    optimizer : Optimizer
    ema : _EMA
    device : torch.device
    grad_scaler : GradScaler or None
        Mixed-precision gradient scaler; None = fp32.
    lambda_v : float
        Value loss weight.
    label_smoothing : float
        Label smoothing for CE.
    grad_clip : float
        Max global gradient norm.

    Returns
    -------
    Dict with ``loss, ce_loss, value_mse, grad_norm`` for logging.
    """
    use_amp = grad_scaler is not None

    loss, ce, value_mse_loss, belief_parts = _compute_loss(
        policy, batch, lambda_v=lambda_v,
        label_smoothing=label_smoothing, use_amp=use_amp,
        belief_weights=belief_weights,
        group_marginal=group_marginal,
    )

    # Backward
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

    ema.update(policy)

    return {
        "loss": loss.item(),
        "ce": ce.mean().item(),
        "value_mse": value_mse_loss.item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        **belief_parts,
    }


def train(
    policy: Policy,
    data_dir: str | Path,
    save_dir: str | Path,
    *,
    # Hyperparameters
    batch_size: int = BATCH,
    peak_lr: float = PEAK_LR,
    min_lr: float = MIN_LR,
    warmup: int = WARMUP,
    grad_clip: float = GRAD_CLIP,
    label_smoothing: float = LABEL_SMOOTH,
    ema_decay: float = EMA_DECAY,
    lambda_v: float = LAMBDA_V,
    belief_weights: dict[str, float] | None = None,
    group_marginal: bool = True,
    weight_decay: float = WEIGHT_DECAY,
    betas: tuple[float, float] = BETAS,
    total_steps: int | None = None,
    # Cadence
    log_every: int = LOG_EVERY,
    val_every: int = VAL_EVERY,
    ckpt_every: int = CKPT_EVERY,
    # Data
    num_workers: int = 8,
    optimizer_name: str = "adamw",
    muon_lr: float = MUON_LR,
    grad_accum: int = 1,
    archetype_self: int | None = None,
    # Precision
    mixed_precision: bool = True,
    # W&B
    wandb_logger: WandbLogger | None = None,
    wandb_project: str = "pokemon-tcg-il",
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    # Resume
    resume_ckpt: str | Path | None = None,
    allow_belief_widening: bool = False,
    # Eval
    run_val: bool = True,
    # Early-stop (C.5/C.8)
    patience: int = 5,
    # Seed
    seed: int = 42,
) -> Policy:
    """Full training loop (C.6).

    Parameters
    ----------
    policy : Policy
        Model to train.
    data_dir : Path
        Directory with ``shards/`` and ``meta.parquet``.
    save_dir : Path
        Directory for checkpoints.
    batch_size : int
        Decision points per step (2048).
    peak_lr : float
        Peak learning rate (3e-4).
    min_lr : float
        Minimum LR (3e-5).
    warmup : int
        Linear warmup steps (1000).
    grad_clip : float
        Max global gradient norm (1.0).
    label_smoothing : float
        Label smoothing (0.05).
    ema_decay : float
        EMA decay (0.999).
    lambda_v : float
        Value loss weight (0.5).
    weight_decay : float
        Weight decay (0.01).
    betas : tuple
        AdamW betas.
    total_steps : int or None
        Total training steps.  If None, compute as 10 epochs over train split.
    log_every : int
        Log scalars every N steps.
    val_every : int
        Run offline eval every N steps.
    ckpt_every : int
        Save checkpoint every N steps.
    num_workers : int
        DataLoader workers.
    grad_accum : int
        Gradient accumulation steps (1 = no accumulation).  Loss is scaled
        by ``1/grad_accum`` so the effective batch size is
        ``batch_size × grad_accum`` while peak VRAM usage corresponds to
        ``batch_size`` alone.
    mixed_precision : bool
        Use bf16 autocast + GradScaler.
    wandb_logger : WandbLogger or None
        Pre-initialized logger; if None, one is created.
    wandb_project, wandb_entity, wandb_name : str
        W&B init params (only used if wandb_logger is None).
    resume_ckpt : Path or None
        Path to checkpoint to resume from.
    run_val : bool
        Whether to run offline eval (requires a non-empty val split).

    Returns
    -------
    Policy
        EMA-averaged model (best val checkpoint reloaded if val was run).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # Deck identity — stamped into every checkpoint and the decks.json sidecar
    # so a .pt always says which archetype deck it plays.
    # Built before the first batch, not at the first save: a run that cannot
    # label its checkpoints has nothing shippable to produce, and finding that
    # out at save time throws away the training that preceded it.  The
    # generalist used to be exempt (warn, `deck_meta = None`), but its record
    # is the one that says `is_fixed_deck` — without it a generalist .pt is as
    # unusable downstream as a specialist's.
    deck_meta = require_deck_record(
        build_deck_metadata(data_dir, archetype_self),
        f"train(archetype_self={archetype_self})",
    )
    logger.info("Training %s", describe_deck(deck_meta))
    logger.info('total_steps %s' , total_steps)
    # Build datasets
    train_ds = ShardDataset(
        data_dir, split="train", shuffle=True, seed=seed, archetype_self=archetype_self
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # ShardDataset handles shuffle via indices
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = None
    val_ds = None
    if run_val:
        try:
            val_ds = ShardDataset(
                data_dir, split="val", shuffle=False, archetype_self=archetype_self
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
                persistent_workers=(num_workers > 0),
            )
        except (ValueError, FileNotFoundError) as e:
            logger.warning("Val split not available, skipping eval: %s", e)
            run_val = False

    # Record the realised sample counts now, before the first checkpoint is
    # written, so every .pt carries them and not just the sidecar.
    deck_meta["samples_used"] = {
        "train": len(train_ds),
        "val": len(val_ds) if val_ds is not None else 0,
    }

    # Steps
    if total_steps is None:
        steps_per_epoch = len(train_ds) // batch_size
        total_steps = 10 * steps_per_epoch

    # Clamp the schedule/cadence knobs to the actual run length.  The module
    # defaults (warmup=1000, val_every=1000, ckpt_every=2000) assume a large
    # corpus; on a small one total_steps can be a few hundred, and unclamped
    # they silently break the run — LR never leaves warmup, and offline eval /
    # best-checkpoint selection / early stopping never fire at all.
    opt_steps = max(1, total_steps // max(1, grad_accum))

    if warmup >= opt_steps:
        new_warmup = max(1, int(0.05 * opt_steps))
        logger.warning(
            "warmup=%d >= optimizer steps=%d — LR would never reach peak_lr. "
            "Clamping warmup to %d (5%% of the run).",
            warmup, opt_steps, new_warmup,
        )
        warmup = new_warmup

    if run_val and val_every >= total_steps:
        new_val_every = max(1, total_steps // 10)
        logger.warning(
            "val_every=%d >= total_steps=%d — offline eval would never run "
            "(no ckpt-best.pt, no early stopping). Clamping val_every to %d.",
            val_every, total_steps, new_val_every,
        )
        val_every = new_val_every

    if ckpt_every >= total_steps:
        new_ckpt_every = max(1, total_steps // 4)
        logger.warning(
            "ckpt_every=%d >= total_steps=%d — no periodic checkpoints would "
            "be written. Clamping ckpt_every to %d.",
            ckpt_every, total_steps, new_ckpt_every,
        )
        ckpt_every = new_ckpt_every

    # Move model to device BEFORE creating optimizer/EMA so their
    # internal state lands on the correct device.
    policy.to(device)

    # Optimizer & schedule.  ``scheduler.step()`` only fires on accumulation
    # boundaries, so the cosine horizon is measured in *optimizer* steps —
    # passing total_steps here would leave the decay unfinished (and never
    # reach min_lr) whenever grad_accum > 1.
    optimizer = create_optimizer(
        policy, peak_lr, weight_decay, betas,
        optimizer=optimizer_name, muon_lr=muon_lr,
    )
    scheduler = create_schedule(optimizer, opt_steps, warmup, peak_lr, min_lr)

    # EMA
    ema = _EMA(policy, ema_decay)

    # Grad scaler
    grad_scaler = torch.amp.GradScaler(device.type) if mixed_precision else None

    # Resume
    start_step = 0
    best_val_metric = -1.0
    patience_counter = 0
    if resume_ckpt is not None:
        from ptcg_il.train.checkpoint import load_checkpoint
        ckpt = load_checkpoint(resume_ckpt, device)
        stale = load_policy_state(
            policy, ckpt["model_state_dict"],
            allow_belief_widening=allow_belief_widening,
        )
        if stale:
            logger.warning(
                "Checkpoint predates the belief heads; %d belief parameters "
                "start from scratch.", len(stale),
            )
        require_matching_optimizer(ckpt, optimizer_name)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"], model=policy)
        start_step = ckpt["step"]
        logger.info("Resumed from step %d", start_step)

    # W&B
    if wandb_logger is None:
        wandb_logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config={
                "batch_size": batch_size,
                "peak_lr": peak_lr,
                "min_lr": min_lr,
                "warmup": warmup,
                "grad_clip": grad_clip,
                "label_smoothing": label_smoothing,
                "group_marginal_ce": group_marginal,
                "ema_decay": ema_decay,
                "lambda_v": lambda_v,
                "weight_decay": weight_decay,
                "betas": betas,
                "total_steps": total_steps,
                "mixed_precision": mixed_precision,
                "data_dir": str(data_dir),
            },
        )

    best_val_metric = float(wandb_logger._best_macro) if wandb_logger._best_macro > 0 else best_val_metric

    policy.train()

    train_iter = iter(train_loader)
    data_start = time.perf_counter()

    # ---- gradient accumulation state ----
    use_amp = grad_scaler is not None
    accum_loss = 0.0
    accum_ce = 0.0
    accum_value_mse = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, total_steps):
        # Fetch next batch — cycle loader infinitely
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        data_wait = time.perf_counter() - data_start

        # Move to device
        batch_gpu: dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if k in ("encoder_padding_mask",):
                continue  # kept on CPU as bool
            batch_gpu[k] = v.to(device, non_blocking=(device.type == "cuda"))

        # ---- micro-batch forward ----
        loss, ce, value_mse_loss, belief_parts = _compute_loss(
            policy, batch_gpu,
            lambda_v=lambda_v,
            label_smoothing=label_smoothing,
            use_amp=use_amp,
            belief_weights=belief_weights,
            group_marginal=group_marginal,
        )

        # Scale loss for gradient accumulation
        if grad_accum > 1:
            loss = loss / grad_accum

        # Backward (accumulate gradients)
        if use_amp:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()
        accum_ce += ce.mean().item()
        accum_value_mse += value_mse_loss.item()

        # ---- optimizer step (every grad_accum micro-batches) ----
        is_accum_boundary = (step + 1) % grad_accum == 0

        if is_accum_boundary:
            if use_amp:
                grad_scaler.unscale_(optimizer)
            grad_norm_val = nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            if use_amp:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            ema.update(policy)
            optimizer.zero_grad(set_to_none=True)

        lr = scheduler.get_last_lr()[0]

        # Build metrics from accumulated losses
        n = grad_accum if is_accum_boundary else (step + 1) % grad_accum
        metrics = {
            "loss": accum_loss / n,
            "ce": accum_ce / n,
            "value_mse": accum_value_mse / n,
            "grad_norm": (
                grad_norm_val.item() if isinstance(grad_norm_val, torch.Tensor) else float(grad_norm_val)
            ) if is_accum_boundary else 0.0,
            # Belief scalars come from the last micro-batch rather than the
            # accumulation average: they are diagnostics, not the objective,
            # and each is already a batch mean.
            **belief_parts,
        }

        if is_accum_boundary:
            accum_loss = 0.0
            accum_ce = 0.0
            accum_value_mse = 0.0

        # Log / eval / checkpoint — only on accumulation boundaries
        if not is_accum_boundary:
            continue

        # Log
        if (step + 1) % log_every == 0 or step == 0:
            wandb_logger.log_train(
                step=step + 1,
                loss=metrics["loss"],
                ce=metrics["ce"],
                value_mse=metrics["value_mse"],
                grad_norm=metrics["grad_norm"],
                lr=lr,
                samples_per_sec=batch_size / max(data_wait, 0.001),
                extra={k: v for k, v in metrics.items() if k.startswith("belief/")},
            )

        # Offline eval
        if run_val and val_loader is not None and (step + 1) % val_every == 0:
            eval_metrics = offline_eval(
                policy, val_loader, device,
                ema=ema,
                lambda_v=lambda_v,
                label_smoothing=label_smoothing,
                belief=belief_weights is not None,
            )
            eval_metrics["step"] = step + 1
            wandb_logger.log_eval(**eval_metrics)

            # Select on non-trivial micro top-1, not top1_macro: the macro
            # average weights near-degenerate sel_ctx buckets (a handful of
            # forced 1-2 option decisions scoring 1.0) equally with the buckets
            # holding almost all real decisions, so it tracks noise.
            sel_metric = eval_metrics.get("val/top1_nontrivial")
            if sel_metric is None:
                sel_metric = eval_metrics.get("val/top1_macro", 0.0)
            if sel_metric > best_val_metric + 1e-4:
                best_val_metric = sel_metric
                patience_counter = 0
                wandb_logger.mark_best(step + 1, sel_metric)

                with ema.applied(policy):
                    ckpt_path = save_checkpoint(
                        policy, optimizer, scheduler, ema,
                        step=step + 1,
                        save_dir=save_dir,
                        tag="best",
                        deck=deck_meta,
                    )
                policy.train()
                # Log checkpoint as W&B artifact (C.9)
                wandb_logger.log_artifact(str(ckpt_path), artifact_type="model", aliases=["best"])
                # Attach vocab + archetypes
                _data_dir = Path(data_dir)
                for fname in ("vocab.json", "archetypes.json"):
                    fp = _data_dir / fname
                    if fp.exists():
                        wandb_logger.log_artifact(str(fp), artifact_type=fname.split(".")[0])
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early-stop: patience %d reached (best=%.4f)", patience, best_val_metric)
                    break

        # Checkpoint
        if (step + 1) % ckpt_every == 0:
            with ema.applied(policy):
                save_checkpoint(
                    policy, optimizer, scheduler, ema,
                    step=step + 1,
                    save_dir=save_dir,
                    tag=f"step-{step+1:07d}",
                    deck=deck_meta,
                )
            policy.train()

        data_start = time.perf_counter()

    # Final checkpoint (last)
    ema.apply(policy)
    last_ckpt_path = save_checkpoint(
        policy, optimizer, scheduler, ema,
        step=total_steps,
        save_dir=save_dir,
        tag="last",
        deck=deck_meta,
    )
    # Log last checkpoint as W&B artifact (C.9)
    wandb_logger.log_artifact(str(last_ckpt_path), artifact_type="model", aliases=["last"])
    # Attach vocab + archetypes
    _data_dir = Path(data_dir)
    for fname in ("vocab.json", "archetypes.json"):
        fp = _data_dir / fname
        if fp.exists():
            wandb_logger.log_artifact(str(fp), artifact_type=fname.split(".")[0])

    # Deck sidecar — same record as the one inside the .pt, but readable
    # without torch, and also emits deck.csv for the submission bundle.
    deck_meta["best_val_metric"] = float(best_val_metric)
    available = [
        p.name for p in sorted(save_dir.glob("ckpt-*.pt"))
    ]
    sidecar = update_sidecar(save_dir, deck_meta, checkpoints=available)
    write_deck_csv(deck_meta, save_dir / "deck.csv")
    logger.info("Wrote deck sidecar %s and deck.csv", sidecar)

    # Reload best if we had val
    if best_val_metric >= 0:
        best_path = save_dir / "ckpt-best.pt"
        if best_path.exists():
            from ptcg_il.train.checkpoint import load_checkpoint
            ckpt = load_checkpoint(best_path, device)
            load_policy_state(policy, ckpt["model_state_dict"])
            ema.load_state_dict(ckpt["ema_state_dict"], model=policy)
            ema.apply(policy)

    wandb_logger.finish()
    return policy
