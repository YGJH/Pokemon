"""W&B logging for pipeline stage 5 (RL_SPEC R1–R2).

RL runs go to their own project, **not** ``pokemon-tcg-il``.  The two stages
share no axes — IL plots loss and top-1 against optimizer steps, RL plots
``kl_to_il``, ``beta`` and ``clip_fraction`` against PPO steps on a policy that
is already trained — so mixing them makes both projects' default panels
unreadable.  A deck's IL and RL runs are matched by the ``-a<N>`` suffix
instead.

R1 and R2 advance on **different clocks**: critic-repair steps and PPO optimizer
steps.  Both are declared against their own ``step_metric`` so wandb's global
step, which just counts ``log`` calls, never becomes the x axis of either.
"""

from __future__ import annotations

import logging
from rich.logging import RichHandler
from typing import Any

from ptcg_il.train.logger import WandbLogger
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_time=False)])
logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "wandb_project": "pokemon-tcg-rl",
    "wandb_entity": "poken",
    # None, not a literal: the run name is derived from --deck-archetype, so a
    # default string here would make every deck's run identically named.
    "wandb_name": None,
    "wandb_mode": "online",
}


class RLWandbLogger(WandbLogger):
    """RL-shaped metrics on top of IL's ``WandbLogger`` init and teardown.

    Subclasses rather than reimplements, because that init path carries three
    behaviours the IL side already found the hard way: the missing-``wandb``
    guard, ``mode="disabled"`` falling back to console rather than silently
    dropping metrics, and a failed ``wandb.init`` degrading to a warning instead
    of killing a run that is about to spend hours in rollout.  Only the metric
    vocabulary is new here.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # `WandbLogger.__init__` returns *before* assigning `_run` when the
        # wandb import itself fails, so this cannot be a plain attribute read.
        self._run = getattr(self, "_run", None)
        self._r1_step = 0
        self._declare_axes()

    def _declare_axes(self) -> None:
        if not self.active:
            return
        self._run.define_metric("r1/step")
        self._run.define_metric("r1/*", step_metric="r1/step")
        self._run.define_metric("r2/step")
        self._run.define_metric("r2/*", step_metric="r2/step")

    def log(self, metrics: dict[str, Any]) -> None:
        """One ``wandb.log`` on wandb's auto-incrementing global step.

        The global step is deliberately never passed explicitly: R1 and R2 would
        each want to own it and one of them would have to lose.  ``define_metric``
        puts both on their real x axis instead.
        """
        if not self.active:
            return
        self._run.log({k: v for k, v in metrics.items() if v is not None})

    # ── R1: critic repair ────────────────────────────────────────────────────

    def log_r1_curve(self, history: dict[str, Any], phase: str) -> None:
        """Replay the loss curve ``critic.repair`` recorded.

        ``repair`` records rather than calling back, so these points land after
        the phase finishes — the whole curve, just late.  Threading a callback
        through ``critic.py`` would buy live plotting of a chart that is read
        after the fact anyway.

        ``r1/step`` continues across phase A into phase B so the two are one
        continuous curve; ``r1/phase`` marks which is which.
        """
        loss = history.get("loss") or []
        mse = history.get("mse") or []
        ce = history.get("ce") or []
        for i in range(len(loss)):
            self._r1_step += 1
            self.log({
                "r1/step": self._r1_step,
                "r1/phase": 0 if phase == "a" else 1,
                "r1/loss": loss[i],
                "r1/mse": mse[i] if i < len(mse) else None,
                "r1/ce": ce[i] if i < len(ce) else None,
            })

    def log_r1_diag(self, record: dict[str, Any], phase: str) -> None:
        """Post-phase critic diagnostics (§3): ``corr``, ``pred_std``, …

        ``per_turn_accuracy`` is a list and is skipped — it is in the JSON
        report, and a per-turn curve does not belong on a step axis.
        """
        payload = {
            f"r1/{phase}/{k}": (float(v) if isinstance(v, bool) else v)
            for k, v in record.items()
            if isinstance(v, (int, float, bool))
        }
        payload["r1/step"] = self._r1_step
        self.log(payload)
        self.summary(payload)

    def log_r1_result(self, report: dict[str, Any]) -> None:
        self.summary({"r1/passed": float(bool(report.get("passed")))})

    # ── R2: PPO ──────────────────────────────────────────────────────────────

    def log_r2_iter(self, stats: dict[str, Any], rollout: dict[str, Any]) -> None:
        """One point per rollout buffer — the same numbers as the console line.

        ``stats`` is already averaged over the ``ppo_epochs × n_minibatch``
        updates in this iteration, so ``r2/step`` is the cumulative optimizer
        step and not the iteration index.
        """
        payload = {f"r2/{k}": v for k, v in stats.items() if isinstance(v, (int, float))}
        payload.update({f"r2/{k}": v for k, v in rollout.items()})
        payload["r2/step"] = stats.get("step", 0)
        payload.pop("r2/beta_next", None)
        payload["r2/beta"] = stats.get("beta_next")
        self.log(payload)

    # ── Gate ─────────────────────────────────────────────────────────────────

    def log_gate(self, result: dict[str, Any]) -> None:
        """Terminal §10.2 verdict.  Summary metrics so the runs table sorts on it."""
        payload: dict[str, Any] = {
            f"gate/{k}": (float(v) if isinstance(v, bool) else v)
            for k, v in result.items()
            if isinstance(v, (int, float, bool))
        }
        for name, ok in (result.get("conditions") or {}).items():
            payload[f"gate/cond_{name}"] = float(bool(ok))
        self.log(payload)
        self.summary(payload)
        if result.get("summary"):
            self.summary({"gate/summary": result["summary"]})

    # ── Plumbing ─────────────────────────────────────────────────────────────

    def summary(self, metrics: dict[str, Any]) -> None:
        """Pin values to the run summary so they show in the runs table."""
        if not self.active:
            return
        for k, v in metrics.items():
            if v is not None:
                self._run.summary[k] = v


class NullRLLogger:
    """No-op stand-in so call sites need no ``if wb is not None``.

    Every public method of :class:`RLWandbLogger` must exist here, or a caller
    that opted out of W&B raises where a logging call was added.
    """

    active = False

    def log(self, metrics: dict[str, Any]) -> None: ...
    def log_r1_curve(self, history: dict[str, Any], phase: str) -> None: ...
    def log_r1_diag(self, record: dict[str, Any], phase: str) -> None: ...
    def log_r1_result(self, report: dict[str, Any]) -> None: ...
    def log_r2_iter(self, stats: dict[str, Any], rollout: dict[str, Any]) -> None: ...
    def log_gate(self, result: dict[str, Any]) -> None: ...
    def summary(self, metrics: dict[str, Any]) -> None: ...
    def finish(self) -> None: ...


def run_name(args: Any) -> str | None:
    """``pokemon-tcg-rl-a<N>``, unless ``--wandb-name`` said otherwise.

    Stage 5 trains one RL run per deck into one project, so unnamed runs would
    differ only by their config — the same trap IL's run naming already fixes.
    An explicit ``--wandb-name`` is honoured verbatim.
    """
    name = getattr(args, "wandb_name", None)
    if name:
        return name
    return f"pokemon-tcg-rl-a{args.deck_archetype}"


def build_logger(args: Any, cfg: Any, extra: dict[str, Any] | None = None):
    """``RLWandbLogger`` per *args*, or :class:`NullRLLogger` if opted out.

    ``--no-wandb`` wins over ``--wandb-mode``, matching ``ptcg_il.cli``: it was
    declared as an off switch, so it has to switch things off.
    """
    if getattr(args, "no_wandb", False):
        logger.info("wandb disabled (--no-wandb) — RL metrics go to the console only")
        return NullRLLogger()

    config = dict(cfg.to_dict())
    config.update(extra or {})
    return RLWandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name(args),
        config=config,
        mode=args.wandb_mode,
    )
