"""Training pipeline: dataset, loop, checkpointing, eval, W&B logging.

Implements TRANSFORMER_IL_SPEC.md Appendix C.
"""

from ptcg_il.train.dataset import ShardDataset, collate_fn, compute_sample_weights
from ptcg_il.train.checkpoint import save_checkpoint, load_checkpoint, build_submission_bundle
from ptcg_il.train.eval import offline_eval
from ptcg_il.train.logger import WandbLogger
from ptcg_il.train.loop import (
    create_optimizer,
    create_schedule,
    masked_label_smoothed_ce,
    train_step,
    train,
)

__all__ = [
    "ShardDataset",
    "collate_fn",
    "compute_sample_weights",
    "masked_label_smoothed_ce",
    "create_optimizer",
    "create_schedule",
    "train_step",
    "train",
    "save_checkpoint",
    "load_checkpoint",
    "build_submission_bundle",
    "offline_eval",
    "WandbLogger",
]
