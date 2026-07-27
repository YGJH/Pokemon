"""Per-part execution timing for the RL stage.

Stage 5 runs for hours per deck, so "where did the time go" has to survive the
run: `rl_report.json` carries the split, not just the log.  These tests pin the
report shape and the formatting, not the durations themselves.
"""

from __future__ import annotations

import pytest

from ptcg_rl.train import _timing_breakdown, _timing_summary, fmt_dur


class TestFmtDur:
    @pytest.mark.parametrize("seconds,expected", [
        (0.0, "0.0s"),
        (0.44, "0.4s"),
        (59.9, "59.9s"),
        (60, "1m00s"),
        (750, "12m30s"),
        (3599, "59m59s"),
        (3600, "1h00m"),
        (15130, "4h12m"),
    ])
    def test_unit_switches_with_scale(self, seconds, expected):
        # R1 is minutes and R2 is hours; one unit would be wrong for one of them.
        assert fmt_dur(seconds) == expected

    def test_negative_is_clamped_not_rendered(self):
        # Clock skew must not produce "-1m00s" in a log line.
        assert fmt_dur(-5) == "0.0s"


class TestTimingSummary:
    def _report(self):
        return {
            "r1": {
                "elapsed_sec": 800.0,
                "phase_a": {"elapsed_sec": 300.0},
                "phase_b": {"elapsed_sec": 450.0, "il_top1_eval_sec": 50.0},
            },
            "r2": {
                "elapsed_sec": 14520.0, "rollout_sec": 13860.0,
                "update_sec": 580.0, "steps_per_hour": 12400.0,
            },
            "gate": {"elapsed_sec": 1260.0},
        }

    def test_flattens_every_part_to_one_level(self):
        t = _timing_summary(self._report(), 16600.0)
        assert t["total_sec"] == 16600.0
        assert t["r1_phase_a_sec"] == 300.0
        assert t["r1_phase_b_sec"] == 450.0
        assert t["r2_rollout_sec"] == 13860.0
        assert t["r2_update_sec"] == 580.0
        assert t["gate_sec"] == 1260.0

    def test_partial_run_omits_the_phases_that_did_not_run(self):
        # --phase r1, or an R1 that failed its gate: no r2/gate keys to read.
        t = _timing_summary({"r1": {"elapsed_sec": 800.0}}, 800.0)
        assert t["r1_sec"] == 800.0
        assert "r2_sec" not in t and "gate_sec" not in t

    def test_r1_without_phase_b_records_none_rather_than_raising(self):
        t = _timing_summary({"r1": {"elapsed_sec": 300.0, "phase_a": {"elapsed_sec": 300.0}}}, 300.0)
        assert t["r1_phase_b_sec"] is None


class TestTimingBreakdown:
    def test_reports_the_rollout_update_split(self):
        line = _timing_breakdown({
            "r1_sec": 750, "r2_sec": 14520, "r2_rollout_sec": 13860,
            "r2_update_sec": 580, "gate_sec": 1260,
        })
        # The split is the actionable number: rollout is expected to dominate,
        # and a regression in the update path shows up nowhere else.
        assert "rollout 3h51m" in line and "update 9m40s" in line
        assert "R1 12m30s" in line and "gate 21m00s" in line

    def test_empty_when_nothing_ran(self):
        assert _timing_breakdown({}) == ""

    def test_r2_without_a_split_still_reports_its_total(self):
        assert _timing_breakdown({"r2_sec": 60}) == "  (R2 1m00s)"
