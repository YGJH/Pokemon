"""W&B integration (C.9).

Thin wrapper around ``wandb``.  Tolerates ``wandb`` being absent (no-op mode)
or offline (``WANDB_MODE=offline``).  Handles init, per-step scalar logging,
per-eval table logging, checkpoint artifacts, alerts, and final summary.

``wandb`` is a training-only dependency — it is never imported by the
submitted ``agent()`` / ``main.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import so wandb is optional
_wandb = None


def _get_wandb():
    global _wandb
    if _wandb is None:
        try:
            import wandb as _w  # type: ignore[no-redef]
            _wandb = _w
        except ImportError:
            _wandb = False  # type: ignore[assignment]
    return _wandb


class WandbLogger:
    """W&B experiment tracker.

    One instance per training run.  Methods are no-ops when wandb is not
    available or ``WANDB_MODE=offline``.

    Parameters
    ----------
    project : str
        W&B project name.
    entity : str or None
        W&B team/entity.
    name : str or None
        Run name (auto-generated if None).
    config : dict
        Hyperparameter + provenance dict logged to run config.
    mode : str or None
        Override ``WANDB_MODE``; "offline" / "disabled" / "online".
    """

    def __init__(
        self,
        project: str = "pokemon-tcg-il",
        entity: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        mode: str | None = None,
    ):
        self._active = False
        self._best_macro: float = -1.0
        self._best_step: int = 0

        wandb = _get_wandb()
        if wandb is False or wandb is None:
            logger.info("wandb not available — logging disabled")
            return

        # Respect WANDB_MODE env var; allow override
        run_mode = mode or os.environ.get("WANDB_MODE", "online")

        # ``mode="disabled"`` makes wandb.init() succeed while dropping every
        # metric.  Treating that as "active" would silence the console fallback
        # too, so `--no-wandb` would print no train loss and no eval numbers at
        # all.  Stay inactive instead and log to the console.
        if run_mode == "disabled":
            logger.info("wandb disabled — logging metrics to console")
            self._run = None
            return

        try:
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config or {},
                mode=run_mode,
            )
            self._active = True
        except Exception as e:
            logger.warning("wandb.init failed: %s — logging disabled", e)
            self._run = None

    def log_train(
        self,
        step: int,
        loss: float,
        ce: float,
        value_mse: float,
        grad_norm: float,
        lr: float,
        samples_per_sec: float,
        extra: dict[str, float] | None = None,
    ) -> None:
        """Log per-step training scalars (C.9 ``LOG_EVERY``).

        *extra* carries already-namespaced scalars from optional loss terms
        (``belief/*``), so a new auxiliary head does not need a new keyword
        argument here.
        """
        # Console fallback when wandb is unavailable
        if not self._active:
            tail = "".join(f"  {k}={v:.4f}" for k, v in sorted((extra or {}).items()))
            logger.info(
                "step %d  loss=%.4f  ce=%.4f  grad=%.3f  lr=%.2e  samp/s=%d%s",
                step, loss, ce, grad_norm, lr, int(samples_per_sec), tail,
            )
            return
        wandb = _get_wandb()
        if wandb is False:
            return
        wandb.log(
            {
                "train/loss": loss,
                "train/ce": ce,
                "train/value_mse": value_mse,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "perf/samples_per_sec": samples_per_sec,
                **(extra or {}),
            },
            step=step,
        )

    def log_eval(self, **metrics: Any) -> None:
        """Log offline-eval results (C.9 ``VAL_EVERY``).

        Accepts the flat dict from ``offline_eval()``.  Scalars are logged
        directly; per-context lists are converted to ``wandb.Table``.
        """
        if not self._active:
            # Console fallback
            step = metrics.get("step", 0)
            top1_macro = metrics.get("val/top1_macro", float("nan"))
            top1_micro = metrics.get("val/top1_micro", float("nan"))
            nt = metrics.get("val/top1_nontrivial", float("nan"))
            nt_base = metrics.get("val/top1_nontrivial_firstlegal", float("nan"))
            n_nt = metrics.get("val/n_nontrivial", 0)
            logger.info(
                "eval step %d  top1_macro=%.4f  top1_micro=%.4f  "
                "nontrivial: top1=%.4f vs first-legal=%.4f (lift %+.4f, n=%d)",
                step, top1_macro, top1_micro,
                nt, nt_base, nt - nt_base, n_nt,
            )
            return
        wandb = _get_wandb()
        if wandb is False:
            return

        step = metrics.pop("step", 0)
        scalars: dict[str, float] = {}
        tables: dict[str, Any] = {}

        for key, value in metrics.items():
            if key in ("val/top1_by_sel_type", "val/top1_by_sel_ctx"):
                if isinstance(value, list) and len(value) > 0:
                    tables[key] = self._make_accuracy_table(
                        value, "context" if "ctx" in key else "type"
                    )
            elif isinstance(value, (int, float, np.floating)):
                scalars[key] = float(value)

        wandb.log(scalars, step=step)
        for name, table in tables.items():
            wandb.log({name: table}, step=step)

    @staticmethod
    def _make_accuracy_table(
        rows: list[tuple[int, float, int]],
        label: str,
    ) -> Any:
        """Build a wandb.Table from ``[(id, accuracy, count), ...]``."""
        wandb = _get_wandb()
        if wandb is False:
            return None
        table = wandb.Table(
            columns=[f"{label}_id", "top1_accuracy", "count"]
        )
        for row in rows:
            table.add_data(*row)
        return table

    def mark_best(self, step: int, macro_top1: float) -> None:
        """Record best-val step for the summary."""
        self._best_macro = macro_top1
        self._best_step = step
        if self._active:
            wandb = _get_wandb()
            if wandb is not False:
                wandb.run.summary["val/best_top1_macro"] = macro_top1
                wandb.run.summary["val/best_step"] = step

    def log_artifact(
        self,
        path: str,
        artifact_type: str = "model",
        name: str | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Log a checkpoint artifact to W&B (C.9).

        Parameters
        ----------
        path : str
            Path to the file/directory.
        artifact_type : str
            "model", "vocab", etc.
        name : str or None
            Artifact name; defaults to the filename.
        aliases : list[str] or None
            E.g. ``["best"]`` for the promoted checkpoint.
        """
        if not self._active:
            return
        wandb = _get_wandb()
        if wandb is False:
            return
        import os as _os
        name = name or _os.path.basename(path)
        artifact = wandb.Artifact(name=name, type=artifact_type)
        if _os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        wandb.log_artifact(artifact, aliases=aliases or [])

    def alert(
        self,
        title: str,
        text: str,
        level: str = "WARN",
    ) -> None:
        """Send a W&B alert (C.9)."""
        if not self._active:
            return
        wandb = _get_wandb()
        if wandb is False:
            return
        try:
            wandb.alert(title=title, text=text, level=level)
        except Exception:
            logger.warning("wandb.alert failed", exc_info=True)

    def finish(self) -> None:
        """Close the W&B run."""
        if self._active:
            wandb = _get_wandb()
            if wandb is not False and wandb.run is not None:
                wandb.finish()
            self._active = False

    @property
    def active(self) -> bool:
        """True if W&B is connected and logging."""
        return self._active
