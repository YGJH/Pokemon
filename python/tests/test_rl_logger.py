"""W&B logging for the RL stage (`ptcg_rl.logger`).

The point of these tests is the *shape* of what reaches wandb, not that wandb
works.  A fake run records every `log`/`summary` call, so the assertions are on
metric names and their x axis — which is the part that silently goes wrong:
a metric logged under the wrong prefix or against wandb's global step still
"works", it just plots nonsense.
"""

from __future__ import annotations

import argparse

import pytest

from ptcg_il.train import logger as il_logger
from ptcg_rl.logger import DEFAULTS, NullRLLogger, RLWandbLogger, build_logger, run_name


class FakeRun:
    def __init__(self):
        self.logged: list[dict] = []
        self.summary: dict = {}
        self.axes: dict[str, str | None] = {}
        self.finished = False

    def log(self, metrics):
        self.logged.append(dict(metrics))

    def define_metric(self, name, step_metric=None):
        self.axes[name] = step_metric


class FakeWandb:
    def __init__(self, run):
        self._run = run
        self.run = run

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self._run

    def finish(self):
        self._run.finished = True
        self.run = None


@pytest.fixture
def fake(monkeypatch):
    run = FakeRun()
    wandb = FakeWandb(run)
    monkeypatch.setattr(il_logger, "_get_wandb", lambda: wandb)
    return run, wandb


@pytest.fixture
def wb(fake):
    run, _ = fake
    return RLWandbLogger(project="p", entity="e", name="n", mode="online"), run


def _args(**kw):
    ns = argparse.Namespace(
        deck_archetype=0, wandb_name=None, wandb_project=DEFAULTS["wandb_project"],
        wandb_entity=DEFAULTS["wandb_entity"], wandb_mode="online", no_wandb=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class _Cfg:
    def to_dict(self):
        return {"lr": 3e-4, "deck_archetype": 0}


# ── run naming ───────────────────────────────────────────────────────────────

def test_run_name_is_suffixed_per_deck():
    assert run_name(_args(deck_archetype=3)) == "pokemon-tcg-rl-a3"


def test_explicit_wandb_name_is_honoured_verbatim():
    assert run_name(_args(deck_archetype=3, wandb_name="scratch")) == "scratch"


def test_rl_default_project_is_not_the_il_project():
    # Separate projects is the decision; a regression here silently merges two
    # metric vocabularies into one workspace.
    assert DEFAULTS["wandb_project"] != "pokemon-tcg-il"


# ── opting out ───────────────────────────────────────────────────────────────

def test_no_wandb_returns_the_null_logger(fake):
    assert isinstance(build_logger(_args(no_wandb=True), _Cfg()), NullRLLogger)


def test_no_wandb_beats_wandb_mode(fake):
    _, wandb = fake
    build_logger(_args(no_wandb=True, wandb_mode="online"), _Cfg())
    assert not hasattr(wandb, "init_kwargs"), "--no-wandb must not open a run"


def test_null_logger_implements_every_public_method():
    """A logging call added to `RLWandbLogger` alone would raise for `--no-wandb`."""
    methods = [
        n for n in dir(RLWandbLogger)
        if not n.startswith("_") and callable(getattr(RLWandbLogger, n))
        and n in RLWandbLogger.__dict__
    ]
    assert methods, "found no public methods to check — the test would pass vacuously"
    missing = [n for n in methods if not hasattr(NullRLLogger, n)]
    assert not missing, f"NullRLLogger is missing {missing}"


def test_disabled_mode_logs_nothing_and_does_not_raise(fake):
    run, _ = fake
    wb = RLWandbLogger(project="p", mode="disabled")
    assert not wb.active
    wb.log_r1_curve({"loss": [1.0], "mse": [1.0], "ce": [0.0]}, "a")
    wb.log_r2_iter({"step": 1, "policy_loss": 0.0}, {"games": 1})
    wb.log_gate({"passed": True, "score": 0.5})
    wb.finish()
    assert run.logged == []


# ── axes ─────────────────────────────────────────────────────────────────────

def test_r1_and_r2_get_their_own_step_axis(wb):
    logger, run = wb
    assert run.axes["r1/*"] == "r1/step"
    assert run.axes["r2/*"] == "r2/step"


# ── R1 ───────────────────────────────────────────────────────────────────────

def test_r1_curve_logs_one_point_per_recorded_step(wb):
    logger, run = wb
    logger.log_r1_curve({"loss": [3.0, 2.0, 1.0], "mse": [3.0, 2.0, 1.0],
                         "ce": [0.1, 0.2, 0.3], "steps": 3}, "a")
    points = [m for m in run.logged if "r1/loss" in m]
    assert len(points) == 3
    assert [m["r1/step"] for m in points] == [1, 2, 3]
    assert {m["r1/phase"] for m in points} == {0}


def test_r1_step_continues_from_phase_a_into_phase_b(wb):
    logger, run = wb
    curve = {"loss": [1.0, 1.0], "mse": [1.0, 1.0], "ce": [0.0, 0.0]}
    logger.log_r1_curve(curve, "a")
    logger.log_r1_curve(curve, "b")
    points = [m for m in run.logged if "r1/loss" in m]
    steps = [m["r1/step"] for m in points]
    assert steps == sorted(steps) and len(set(steps)) == len(steps), \
        "phase B restarting the x axis would draw the two phases on top of each other"
    assert [m["r1/phase"] for m in points] == [0, 0, 1, 1]


def test_r1_diag_is_prefixed_by_phase_and_pinned_to_the_summary(wb):
    logger, run = wb
    logger.log_r1_diag({"corr": 0.42, "pred_std": 0.9, "per_turn_accuracy": [0.1, 0.2]}, "a")
    logged = run.logged[-1]
    assert logged["r1/a/corr"] == 0.42
    assert run.summary["r1/a/pred_std"] == 0.9
    # A list has no meaning on a step axis and would be logged as a bare sequence.
    assert not any(k.endswith("per_turn_accuracy") for k in logged)


def test_r1_result_records_pass_fail(wb):
    logger, run = wb
    logger.log_r1_result({"passed": False})
    assert run.summary["r1/passed"] == 0.0


# ── R2 ───────────────────────────────────────────────────────────────────────

def test_r2_iter_plots_against_cumulative_optimizer_steps(wb):
    logger, run = wb
    logger.log_r2_iter(
        {"policy_loss": -0.01, "value_loss": 0.2, "kl_to_il": 0.018,
         "beta_next": 0.5, "n_updates": 16, "step": 2048},
        {"games": 64, "decisions": 4096, "mean_reward": 0.125},
    )
    m = run.logged[-1]
    assert m["r2/step"] == 2048, "the x axis must be optimizer steps, not the iteration index"
    assert m["r2/policy_loss"] == -0.01
    assert m["r2/mean_reward"] == 0.125
    assert m["r2/games"] == 64


def test_r2_beta_is_logged_under_one_name(wb):
    logger, run = wb
    logger.log_r2_iter({"step": 1, "beta_next": 0.25}, {})
    m = run.logged[-1]
    assert m["r2/beta"] == 0.25
    assert "r2/beta_next" not in m, "two names for beta gives two half-populated charts"


def test_r2_ratio_canary_reaches_wandb(wb):
    """`ratio_p99` is how the §7 precision bug shows up as drift rather than a raise."""
    logger, run = wb
    logger.log_r2_iter({"step": 1, "ratio_p99": 4.2e-07}, {})
    assert run.logged[-1]["r2/ratio_p99"] == pytest.approx(4.2e-07)


# ── Gate ─────────────────────────────────────────────────────────────────────

def test_gate_logs_verdict_conditions_and_summary_text(wb):
    logger, run = wb
    logger.log_gate({
        "passed": True, "score": 0.56, "wilson_lb": 0.52, "n_paired": 400,
        "il_baseline": 0.82, "conditions": {"score": True, "wilson": False},
        "notes": ["ok"], "summary": "score=0.5600 wilson_lb=0.5200",
    })
    m = run.logged[-1]
    assert m["gate/passed"] == 1.0
    assert m["gate/score"] == 0.56
    assert m["gate/cond_score"] == 1.0 and m["gate/cond_wilson"] == 0.0
    assert run.summary["gate/wilson_lb"] == 0.52
    assert run.summary["gate/summary"].startswith("score=")
    assert "gate/notes" not in m


def test_none_valued_metrics_are_dropped(wb):
    logger, run = wb
    logger.log({"gate/kl_to_il": None, "gate/score": 0.5})
    assert run.logged[-1] == {"gate/score": 0.5}
