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
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ptcg_il.model.policy import Policy, multiselect_ce
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

    Returns
    -------
    ce : Tensor[B]
        Per-sample cross-entropy (not reduced).
    """
    # Masked log-softmax — softmax is computed only over valid options
    logits_inf = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(logits_inf, dim=-1)  # [B, O]; -inf at padded slots

    # NLL
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

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


def create_optimizer(
    policy: nn.Module,
    peak_lr: float = PEAK_LR,
    weight_decay: float = WEIGHT_DECAY,
    betas: tuple[float, float] = BETAS,
) -> AdamW:
    """Build AdamW with per-param no-decay set (C.5).

    Parameters
    ----------
    policy : nn.Module
        The Policy module.
    peak_lr : float
        Peak learning rate.
    weight_decay : float
        Weight decay for decay-eligible params.
    betas : tuple
        AdamW beta coefficients.

    Returns
    -------
    AdamW optimizer.
    """
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
        """Copy EMA weights into *model* (in-place)."""
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]


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
    device_type = device.type if device.type in ("cuda", "cpu") else "cpu"

    with (
        torch.amp.autocast(device_type, dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    ):
        logits, value = policy(batch)  # [B, O], [B]

        # Check if all samples in batch are single-select
        maxcount = batch["maxCount"]
        all_single = (maxcount == 1).all().item()

        if all_single:
            ce = masked_label_smoothed_ce(
                logits,
                batch["action_idx"][:, 0],
                batch["opt_mask"],
                label_smoothing=label_smoothing,
            )
        else:
            ce = multiselect_ce(policy, batch, label_smoothing=label_smoothing)

        loss = (batch["sample_weight"] * ce).mean()
        if lambda_v > 0:
            value_mse_loss = F.mse_loss(value, batch["value_target"])
            loss = loss + lambda_v * value_mse_loss
        else:
            value_mse_loss = torch.tensor(0.0, device=loss.device)

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
    weight_decay: float = WEIGHT_DECAY,
    betas: tuple[float, float] = BETAS,
    total_steps: int | None = None,
    # Cadence
    log_every: int = LOG_EVERY,
    val_every: int = VAL_EVERY,
    ckpt_every: int = CKPT_EVERY,
    # Data
    num_workers: int = 8,
    # Precision
    mixed_precision: bool = True,
    # W&B
    wandb_logger: WandbLogger | None = None,
    wandb_project: str = "pokemon-tcg-il",
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    # Resume
    resume_ckpt: str | Path | None = None,
    # Eval
    run_val: bool = True,
    # Early-stop (C.5/C.8)
    patience: int = 5,
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

    # Build datasets
    train_ds = ShardDataset(data_dir, split="train", shuffle=True, seed=42)
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
    if run_val:
        try:
            val_ds = ShardDataset(data_dir, split="val", shuffle=False)
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

    # Steps
    if total_steps is None:
        steps_per_epoch = len(train_ds) // batch_size
        total_steps = 10 * steps_per_epoch

    # Optimizer & schedule
    optimizer = create_optimizer(policy, peak_lr, weight_decay, betas)
    scheduler = create_schedule(optimizer, total_steps, warmup, peak_lr, min_lr)

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
        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
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

    policy.to(device)
    policy.train()

    train_iter = iter(train_loader)
    data_start = time.perf_counter()

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

        # Forward + backward
        metrics = train_step(
            policy, batch_gpu, optimizer, ema, device,
            grad_scaler=grad_scaler,
            lambda_v=lambda_v,
            label_smoothing=label_smoothing,
            grad_clip=grad_clip,
        )

        # Scheduler step
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

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
            )

        # Offline eval
        if run_val and val_loader is not None and (step + 1) % val_every == 0:
            eval_metrics = offline_eval(
                policy, val_loader, device,
                ema=ema,
                lambda_v=lambda_v,
                label_smoothing=label_smoothing,
            )
            eval_metrics["step"] = step + 1
            wandb_logger.log_eval(**eval_metrics)

            macro_top1 = eval_metrics.get("val/top1_macro", 0.0)
            if macro_top1 > best_val_metric + 1e-4:
                best_val_metric = macro_top1
                patience_counter = 0
                wandb_logger.mark_best(step + 1, macro_top1)

                ema.apply(policy)
                ckpt_path = save_checkpoint(
                    policy, optimizer, scheduler, ema,
                    step=step + 1,
                    save_dir=save_dir,
                    tag="best",
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
            ema.apply(policy)
            save_checkpoint(
                policy, optimizer, scheduler, ema,
                step=step + 1,
                save_dir=save_dir,
                tag=f"step-{step+1:07d}",
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
    )
    # Log last checkpoint as W&B artifact (C.9)
    wandb_logger.log_artifact(str(last_ckpt_path), artifact_type="model", aliases=["last"])
    # Attach vocab + archetypes
    _data_dir = Path(data_dir)
    for fname in ("vocab.json", "archetypes.json"):
        fp = _data_dir / fname
        if fp.exists():
            wandb_logger.log_artifact(str(fp), artifact_type=fname.split(".")[0])

    # Reload best if we had val
    if best_val_metric >= 0:
        best_path = save_dir / "ckpt-best.pt"
        if best_path.exists():
            from ptcg_il.train.checkpoint import load_checkpoint
            ckpt = load_checkpoint(best_path, device)
            policy.load_state_dict(ckpt["model_state_dict"])
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.apply(policy)

    wandb_logger.finish()
    return policy
