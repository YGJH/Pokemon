"""`SearchPlannerAgent`'s opponent model, now prior-driven.

The agent must reset its predictor between games: `live_eval` reuses one agent
object across a whole match series, and carrying game 1's observations into
game 2 is evidence about a deck that is no longer in play.
"""
from __future__ import annotations

import pytest

from ptcg_il.deck_prior import OpponentDeckPredictor
from ptcg_il.live_eval import SearchPlannerAgent

ARCHETYPES = {
    "opp_ids": [0],
    "archetypes": [
        {"id": 0, "representative": [100] * 60, "frequency": 90},
        {"id": 1, "representative": [200] * 60, "frequency": 10},
    ],
}


def _obs(discard_ids: list[int]) -> dict:
    return {
        "select": {"selectType": 0},
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [],
                 "discard": [{"id": i} for i in discard_ids]},
            ],
        },
    }


@pytest.fixture
def captured(monkeypatch):
    """Capture the `opp_deck` handed to the Rust planner."""
    seen: list = []

    def fake_rust(obs_dict, opp_deck=None, iterations=200, seed=42):
        seen.append(opp_deck)
        return [0]

    monkeypatch.setattr("ptcg_il.live_eval._call_rust_search_planner", fake_rust)
    return seen


def test_passes_a_real_decklist_not_none(captured):
    agent = SearchPlannerAgent(predictor=OpponentDeckPredictor(ARCHETYPES))
    agent(_obs([]))
    assert captured[0] is not None, "None is the mirror fallback being removed"
    assert len(captured[0]) == 60


def test_observations_steer_the_template(captured):
    agent = SearchPlannerAgent(predictor=OpponentDeckPredictor(ARCHETYPES))
    agent(_obs([]))
    assert set(captured[0]) == {100}
    agent(_obs([200] * 4))
    assert set(captured[1]) == {200}


def test_a_new_game_resets_the_predictor(captured):
    """`select is None` is the deck-selection step: a new game.  Without a
    reset, the previous game's opponent is still being inferred."""
    agent = SearchPlannerAgent(predictor=OpponentDeckPredictor(ARCHETYPES))
    agent(_obs([200] * 4))
    assert set(captured[0]) == {200}

    agent({"select": None})          # new game
    agent(_obs([]))
    assert set(captured[-1]) == {100}, "stale evidence carried across games"


def test_no_predictor_still_plays(captured):
    """The plain mirror-determinized planner stays available as the baseline
    the prior-driven one is measured against."""
    agent = SearchPlannerAgent(predictor=None)
    assert agent(_obs([])) == [0]
    assert captured[0] is None
