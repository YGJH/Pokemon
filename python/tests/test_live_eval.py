"""Unit tests for live_eval OOV tracking and game-end detection.

Tests use real-shaped observation dicts (Pokemon/Card dicts with ``id`` keys,
not ``cardId`` — matching the live engine's cg.api dataclass schemas).
"""

import sys
from pathlib import Path

import pytest

# Ensure cg is importable (for enum references if needed)
_ENGINE_DIR = (
    Path(__file__).resolve().parent.parent
    / "pokemon-tcg-ai-battle"
    / "sample_submission"
    / "sample_submission"
)
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

from ptcg_il.live_eval import _count_oov_opponent_cards, wilson_interval


# ---------------------------------------------------------------------------
# Helpers: build real-shaped observation dicts
# ---------------------------------------------------------------------------

def _make_opp_active(*card_ids: int | None) -> list:
    """Build an active list: ``[{"id": cid, ...}]`` or ``[None]``."""
    if not card_ids or card_ids[0] is None:
        return [None]
    return [{"id": cid, "serial": 100 + i, "playerIndex": 1} for i, cid in enumerate(card_ids)]


def _make_pokemon(*card_ids: int) -> list[dict]:
    """Build a bench list of Pokemon dicts."""
    return [
        {"id": cid, "serial": 200 + i, "hp": 100, "maxHp": 150,
         "appearThisTurn": False, "energies": [], "energyCards": [],
         "tools": [], "preEvolution": []}
        for i, cid in enumerate(card_ids)
    ]


def _make_discard(*card_ids: int) -> list[dict]:
    """Build a discard list of Card dicts (cg.api.Card: id, serial, playerIndex)."""
    return [
        {"id": cid, "serial": 300 + i, "playerIndex": 1}
        for i, cid in enumerate(card_ids)
    ]


def _make_obs(your_index: int, opp_active, opp_bench, opp_discard) -> dict:
    """Build a minimal observation dict with opponent card data."""
    my_player = {
        "active": _make_opp_active(None),  # irrelevant for OOV counting
        "bench": [],
        "discard": [],
        "deckCount": 40,
        "handCount": 5,
        "prize": [None] * 6,
        "hand": None,
        "benchMax": 5,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }
    opp_player = {
        "active": opp_active,
        "bench": opp_bench,
        "discard": opp_discard,
        "deckCount": 40,
        "handCount": 5,
        "prize": [None] * 6,
        "hand": None,
        "benchMax": 5,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }
    players = [my_player, opp_player] if your_index == 0 else [opp_player, my_player]
    return {"players": players}


# ---------------------------------------------------------------------------
# _count_oov_opponent_cards tests
# ---------------------------------------------------------------------------

class TestOovTracking:
    """Verify _count_oov_opponent_cards reads ``id`` (not ``cardId``) from
    real-shaped engine observation dicts."""

    def test_all_in_vocab(self):
        """Active (1), bench (2), discard (3 cards): all in vocab → oov=0, total=6."""
        id_to_index = {10: 2, 20: 3, 30: 4, 40: 5, 50: 6, 60: 7}
        obs = _make_obs(
            your_index=0,
            opp_active=_make_opp_active(10),
            opp_bench=_make_pokemon(20, 30),
            opp_discard=_make_discard(40, 50, 60),
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=0, id_to_index=id_to_index)
        assert oov == 0
        assert total == 6  # 1 active + 2 bench + 3 discard

    def test_some_oov(self):
        """Cards 99 and 88 not in vocab → oov=2, total=4."""
        id_to_index = {10: 2, 20: 3}
        obs = _make_obs(
            your_index=0,
            opp_active=_make_opp_active(10),
            opp_bench=_make_pokemon(99),
            opp_discard=_make_discard(20, 88),
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=0, id_to_index=id_to_index)
        assert oov == 2  # 99 and 88
        assert total == 4

    def test_face_down_active_skipped(self):
        """Face-down active (None) adds nothing to counts."""
        id_to_index = {10: 2, 20: 3}
        obs = _make_obs(
            your_index=0,
            opp_active=_make_opp_active(None),  # face-down
            opp_bench=_make_pokemon(10, 20),
            opp_discard=[],
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=0, id_to_index=id_to_index)
        assert oov == 0
        assert total == 2  # bench only

    def test_empty_opponent(self):
        """Opponent has no visible cards → oov=0, total=0."""
        id_to_index = {10: 2}
        obs = _make_obs(
            your_index=0,
            opp_active=[],
            opp_bench=[],
            opp_discard=[],
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=0, id_to_index=id_to_index)
        assert oov == 0
        assert total == 0

    def test_none_id_to_index_skips(self):
        """id_to_index=None → return (0,0) without iterating (fast path)."""
        obs = _make_obs(
            your_index=0,
            opp_active=_make_opp_active(10),
            opp_bench=_make_pokemon(20),
            opp_discard=_make_discard(30),
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=0, id_to_index=None)
        assert oov == 0
        assert total == 0

    def test_your_index_1(self):
        """Works when the opponent is player 0 (yourIndex=1)."""
        id_to_index = {10: 2}
        obs = _make_obs(
            your_index=1,  # we are player 1, opponent is player 0
            opp_active=_make_opp_active(10),
            opp_bench=[],
            opp_discard=[],
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=1, id_to_index=id_to_index)
        assert oov == 0
        assert total == 1


# ---------------------------------------------------------------------------
# Wilson interval tests (existing behavior, kept for coverage)
# ---------------------------------------------------------------------------

class TestWilsonInterval:
    def test_perfect_winrate(self):
        center, lo, hi = wilson_interval(500, 500)
        assert 0.99 < center < 1.0
        assert lo > 0.99

    def test_zero_games(self):
        center, lo, hi = wilson_interval(0, 0)
        assert center == 0.0 and lo == 0.0 and hi == 0.0

    def test_even_split(self):
        center, lo, hi = wilson_interval(250, 500)
        assert 0.45 < center < 0.55
        assert lo < center < hi
