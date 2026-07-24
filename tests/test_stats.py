"""Tests for ptcg_mine.stats.select_experts, incl. best-effort relaxation of
g_min when too little data exists to fill k_experts at the configured floor."""

import logging

from ptcg_mine.stats import select_experts


def _lb(**teams):
    """Build a leaderboard dict from name=(games, win_rate) kwargs."""
    return {
        name: {"games": games, "wins": round(games * wr), "win_rate": wr}
        for name, (games, wr) in teams.items()
    }


def test_select_experts_returns_top_k_by_win_rate_when_enough_qualify():
    # A, B, C all clear g_min=50; D has a great win-rate but too few games.
    lb = _lb(A=(60, 0.55), B=(55, 0.70), C=(50, 0.60), D=(10, 0.99))
    assert select_experts(lb, k_experts=2, g_min=50) == ["B", "C"]


def test_select_experts_relaxes_threshold_when_too_few_qualify():
    # Nobody reaches g_min=50; relax to admit the 3 most-played teams
    # (games >= 5: A, B, C) then rank that pool by win-rate. D (2 games) excluded.
    lb = _lb(A=(10, 0.4), B=(8, 0.9), C=(5, 0.7), D=(2, 1.0))
    assert select_experts(lb, k_experts=3, g_min=50) == ["B", "C", "A"]


def test_select_experts_relaxes_to_all_teams_when_fewer_than_k():
    lb = _lb(A=(3, 0.5), B=(1, 0.8))
    assert select_experts(lb, k_experts=10, g_min=50) == ["B", "A"]


def test_select_experts_empty_leaderboard_returns_empty():
    assert select_experts({}, k_experts=10, g_min=50) == []


def test_select_experts_warns_when_relaxing_below_g_min(caplog):
    lb = _lb(A=(3, 0.33))
    with caplog.at_level(logging.WARNING):
        select_experts(lb, k_experts=10, g_min=50)
    assert any("g_min" in r.message for r in caplog.records)
