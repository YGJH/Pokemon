"""Tests for ptcg_mine.episode: load/validate/teams/rewards/deck_of."""

import copy
import json
from pathlib import Path

import pytest

from ptcg_mine.episode import deck_of, load_episode, rewards, teams, validate_episode

FIXTURE_PATH = Path(__file__).parent.parent / "archive/sample_episodes/80169582.json"


@pytest.fixture(scope="module")
def episode():
    return load_episode(FIXTURE_PATH)


def test_load_episode_returns_dict(episode):
    assert isinstance(episode, dict)
    assert "steps" in episode


def test_teams(episode):
    assert teams(episode) == ("shikisoukan", "cocoaAI")


def test_rewards(episode):
    assert rewards(episode) == (-1, 1)


def test_validate_episode_true_for_real_fixture(episode):
    assert validate_episode(episode) is True


def test_deck_of_player0_has_60_cards(episode):
    deck = deck_of(episode, 0)
    assert len(deck) == 60


def test_deck_of_player1_has_60_cards(episode):
    deck = deck_of(episode, 1)
    assert len(deck) == 60


def _minimal_valid_episode():
    """A minimal, well-formed synthetic episode matching the real shape."""
    deck0 = list(range(60))
    deck1 = list(range(100, 160))
    return {
        "info": {"TeamNames": ["alice", "bob"]},
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [{"action": None}, {"action": None}],
            [{"action": deck0}, {"action": deck1}],
        ],
    }


def test_validate_episode_true_for_minimal_synthetic():
    ep = _minimal_valid_episode()
    assert validate_episode(ep) is True


def test_validate_episode_false_for_bad_statuses():
    ep = copy.deepcopy(_minimal_valid_episode())
    ep["statuses"] = ["ERROR", "DONE"]
    assert validate_episode(ep) is False


def test_validate_episode_false_for_short_deck():
    ep = copy.deepcopy(_minimal_valid_episode())
    ep["steps"][1][0]["action"] = list(range(59))  # 59 cards, not 60
    assert validate_episode(ep) is False


def test_validate_episode_false_for_too_few_steps():
    ep = copy.deepcopy(_minimal_valid_episode())
    ep["steps"] = ep["steps"][:1]
    assert validate_episode(ep) is False


def test_validate_episode_false_for_wrong_rewards_length():
    ep = copy.deepcopy(_minimal_valid_episode())
    ep["rewards"] = [1, -1, 0]
    assert validate_episode(ep) is False


def test_deck_of_raises_on_malformed_deck():
    ep = copy.deepcopy(_minimal_valid_episode())
    ep["steps"][1][0]["action"] = list(range(59))
    with pytest.raises(ValueError):
        deck_of(ep, 0)
