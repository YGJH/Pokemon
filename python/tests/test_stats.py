"""Tests for ptcg_mine.stats: team_leaderboard, select_experts."""

import pytest

from ptcg_mine.stats import select_experts, team_leaderboard


def _make_ep(t0, t1, r0, r1):
    """A minimal well-formed synthetic episode (see episode.py docstring)."""
    deck0 = list(range(60))
    deck1 = list(range(60, 120))
    return {
        "info": {"TeamNames": [t0, t1]},
        "rewards": [r0, r1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [{"action": None}, {"action": None}],
            [{"action": deck0}, {"action": deck1}],
        ],
    }


def test_leaderboard_counts_wins_and_draw():
    episodes = [
        _make_ep("A", "B", 1, -1),
        _make_ep("A", "B", -1, 1),
        _make_ep("A", "B", 0, 0),  # draw: counts toward games, not wins
    ]
    lb = team_leaderboard(episodes)
    assert lb["A"]["games"] == 3
    assert lb["A"]["wins"] == 1
    assert lb["A"]["win_rate"] == pytest.approx(1 / 3)
    assert lb["B"]["games"] == 3
    assert lb["B"]["wins"] == 1
    assert lb["B"]["win_rate"] == pytest.approx(1 / 3)


def test_leaderboard_separate_teams():
    episodes = [
        _make_ep("A", "B", 1, -1),
        _make_ep("C", "D", 1, -1),
    ]
    lb = team_leaderboard(episodes)
    assert set(lb.keys()) == {"A", "B", "C", "D"}
    assert lb["A"]["games"] == 1 and lb["A"]["wins"] == 1
    assert lb["D"]["games"] == 1 and lb["D"]["wins"] == 0


def test_select_experts_threshold_and_topk():
    leaderboard = {
        "A": {"games": 100, "wins": 90, "win_rate": 0.9},
        "B": {"games": 100, "wins": 80, "win_rate": 0.8},
        "C": {"games": 30, "wins": 29, "win_rate": 29 / 30},  # below g_min, excluded
        "D": {"games": 100, "wins": 70, "win_rate": 0.7},
    }
    experts = select_experts(leaderboard, k_experts=2, g_min=50)
    assert experts == ["A", "B"]


def test_select_experts_tie_break_by_name():
    leaderboard = {
        "Z": {"games": 100, "wins": 50, "win_rate": 0.5},
        "A": {"games": 100, "wins": 50, "win_rate": 0.5},
    }
    experts = select_experts(leaderboard, k_experts=2, g_min=50)
    assert experts == ["A", "Z"]


def test_select_experts_respects_g_min_exactly():
    leaderboard = {
        "A": {"games": 50, "wins": 40, "win_rate": 0.8},
        "B": {"games": 49, "wins": 45, "win_rate": 45 / 49},
    }
    experts = select_experts(leaderboard, k_experts=10, g_min=50)
    assert experts == ["A"]
