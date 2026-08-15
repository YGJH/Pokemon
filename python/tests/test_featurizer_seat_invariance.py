"""The featurizer must not be able to tell which *seat* the actor is sitting in.

Relabelling the two players (seat 0 <-> seat 1) is a symmetry of the game: it
changes no legal move, no card, no outcome.  Everything the featurizer emits is
already built relative to ``yourIndex`` -- except the global block, which used to
carry ``yourIndex`` and the absolute ``firstPlayer`` one-hot verbatim.

That mattered because seat 0 wins the coin toss in essentially every corpus
episode, so ``yourIndex`` was 96% collinear with "am I going second" and the
policy learned the seat as a proxy for the turn order.  Measured on a real
Kaggle replay before the fix: flipping those columns alone changed the agent's
chosen action on 11 of 126 decisions.

These tests pin the fix: the only turn-order information in ``cls_feat`` is the
*derived* ``am_i_first`` bit, which is invariant under the relabelling.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ptcg_il.featurizer import (
    CLS_AM_I_FIRST,
    CLS_HAS_CONTEXT_CARD,
    CLS_HAS_EFFECT,
    CLS_OPP_PRIZES,
    CLS_OUR_PRIZES,
    CLS_TOSS_UNDECIDED,
    PRIZE_N,
    featurize,
)

SAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "archive"
    / "sample_episodes"
    / "80169582.json"
)


def _load_episode() -> dict:
    with open(SAMPLE_PATH) as f:
        return json.load(f)


def _build_vocab(ep: dict) -> dict:
    from ptcg_mine.vocab import build_vocab

    return build_vocab([ep], mode="all_corpus")


def _seat_swap(obs: dict) -> dict:
    """Relabel seat 0 <-> seat 1 throughout an observation.

    Swaps the ``players`` list, flips ``yourIndex``/``firstPlayer``, and flips
    every nested ``playerIndex`` in the state, the select and the logs.  The
    result describes the *same* game position from the *same* actor's point of
    view -- only the seat numbering differs.
    """
    obs = copy.deepcopy(obs)

    def flip_player_indices(node):
        if isinstance(node, dict):
            pi = node.get("playerIndex")
            if isinstance(pi, int) and pi in (0, 1):
                node["playerIndex"] = 1 - pi
            for v in node.values():
                flip_player_indices(v)
        elif isinstance(node, list):
            for v in node:
                flip_player_indices(v)

    state = obs["current"]
    state["players"] = [state["players"][1], state["players"][0]]
    state["yourIndex"] = 1 - state["yourIndex"]
    if state.get("firstPlayer") in (0, 1):
        state["firstPlayer"] = 1 - state["firstPlayer"]
    flip_player_indices(state)
    flip_player_indices(obs.get("select"))
    flip_player_indices(obs.get("logs"))
    return obs


def _active_decisions(ep: dict):
    """Yield ``(obs, action)`` for every ACTIVE decision with a select."""
    steps = ep["steps"]
    for i in range(len(steps) - 1):
        for p in range(2):
            rec = steps[i][p]
            if rec.get("status") != "ACTIVE":
                continue
            obs = rec.get("observation")
            if obs is None or obs.get("select") is None:
                continue
            yield obs, steps[i + 1][p].get("action", [])


def test_featurize_is_invariant_under_seat_swap():
    """Every emitted tensor must be identical for a seat-relabelled position."""
    ep = _load_episode()
    vocab = _build_vocab(ep)

    examined = 0
    for obs, action in _active_decisions(ep):
        base = featurize(obs, vocab, action)
        swapped = featurize(_seat_swap(obs), vocab, action)

        assert set(base) == set(swapped)
        for key in base:
            a = np.asarray(base[key])
            b = np.asarray(swapped[key])
            assert a.shape == b.shape, f"{key}: shape {a.shape} vs {b.shape}"
            if not np.array_equal(a, b):
                diff = np.flatnonzero(np.asarray(a != b).ravel())
                pytest.fail(
                    f"{key} differs under a pure seat swap at flat indices "
                    f"{diff[:8].tolist()} (n={diff.size})"
                )
        examined += 1

    assert examined > 0, "no ACTIVE decisions examined -- fixture is empty"


def test_am_i_first_is_derived_not_the_raw_seat():
    """``cls_feat[CLS_AM_I_FIRST]`` is 1 exactly when the actor moves first.

    Runs over the fixture *and* its seat-relabelled twin on purpose.  The
    fixture alone has ``firstPlayer == 1``, which makes ``am_i_first`` and the
    old raw ``yourIndex`` numerically identical -- the assertion would hold
    against the very code it is meant to reject.  The relabelled world has
    ``firstPlayer == 0``, where the two definitions disagree on every row.
    """
    ep = _load_episode()
    vocab = _build_vocab(ep)

    seen_first, seen_second, discriminating = 0, 0, 0
    for obs, action in _active_decisions(ep):
        for view in (obs, _seat_swap(obs)):
            state = view["current"]
            fp = state["firstPlayer"]
            if fp not in (0, 1):
                continue
            expected = float(state["yourIndex"] == fp)
            got = featurize(view, vocab, action)["cls_feat"][CLS_AM_I_FIRST]
            assert got == expected, (
                f"yourIndex={state['yourIndex']} firstPlayer={fp}: "
                f"am_i_first={got}, expected {expected}"
            )
            if expected != float(state["yourIndex"]):
                discriminating += 1
            if expected:
                seen_first += 1
            else:
                seen_second += 1

    assert seen_first > 0, "covered no first-player decisions"
    assert seen_second > 0, "covered no second-player decisions"
    assert discriminating > 0, (
        "every row had am_i_first == yourIndex, so this test cannot tell the "
        "derived bit from the raw seat -- it would pass against the old code"
    )


def test_toss_undecided_flag_and_am_i_first_abstains():
    """Before the toss resolves, ``am_i_first`` claims nothing."""
    ep = _load_episode()
    vocab = _build_vocab(ep)

    examined = 0
    for obs, action in _active_decisions(ep):
        if obs["current"]["firstPlayer"] != -1:
            continue
        cls = featurize(obs, vocab, action)["cls_feat"]
        assert cls[CLS_TOSS_UNDECIDED] == 1.0
        assert cls[CLS_AM_I_FIRST] == 0.0
        examined += 1

    assert examined > 0, "fixture covered no pre-toss decisions"


def _poke(card_id: int, serial: int) -> dict:
    return {"id": card_id, "serial": serial, "hp": 100, "maxHp": 100,
            "appearThisTurn": False, "energies": [], "energyCards": [],
            "tools": [], "preEvolution": []}


def _obs(my_prizes: int, opp_prizes: int, context_card, effect) -> dict:
    """Minimal observation with each named column set to a discriminating value.

    Built by hand rather than drawn from the archived episode because that
    fixture cannot separate these columns: over its 17 decisions it never has a
    ``contextCard`` and the two prize counts are equal on every row, so
    ``CLS_OUR_PRIZES`` and ``CLS_OPP_PRIZES`` could be swapped undetected.
    """
    def player(prizes):
        return {"active": [_poke(104, 13)], "bench": [], "benchMax": 5,
                "deckCount": 40, "discard": [], "prize": [None] * prizes,
                "handCount": 0, "hand": [], "poisoned": False, "burned": False,
                "asleep": False, "paralyzed": False, "confused": False}

    return {
        "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                   "option": [{"type": 14}], "contextCard": context_card,
                   "effect": effect},
        "logs": [],
        "current": {
            "turn": 5, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False,
            "energyAttached": False, "retreated": False, "result": -1,
            "stadium": [], "looking": None,
            "players": [player(my_prizes), player(opp_prizes)],
        },
    }


def test_named_prize_columns_are_not_swapped():
    """``CLS_OUR_PRIZES``/``CLS_OPP_PRIZES`` name the columns the featurizer writes.

    The expected values come from the observation, never from the constants, so
    moving either constant off its column fails here.
    """
    vocab = {"id_to_index": {}, "attack_id_to_index": {}, "index_to_id": {}}
    cls = featurize(_obs(2, 5, None, None), vocab, [0])["cls_feat"]

    assert cls[CLS_OUR_PRIZES] == pytest.approx(2 / PRIZE_N)
    assert cls[CLS_OPP_PRIZES] == pytest.approx(5 / PRIZE_N)


def test_named_context_and_effect_columns_track_the_select():
    """``CLS_HAS_CONTEXT_CARD``/``CLS_HAS_EFFECT`` flag presence, at those columns.

    ``embed.py`` scales the context-card and effect-card embeddings by exactly
    these two columns, so a constant pointing one slot away silently multiplies
    them by a reserved zero -- the card is embedded and then thrown away.
    """
    vocab = {"id_to_index": {}, "attack_id_to_index": {}, "index_to_id": {}}
    card = {"id": 104, "serial": 1}

    both = featurize(_obs(3, 3, card, card), vocab, [0])["cls_feat"]
    assert both[CLS_HAS_CONTEXT_CARD] == 1.0
    assert both[CLS_HAS_EFFECT] == 1.0

    neither = featurize(_obs(3, 3, None, None), vocab, [0])["cls_feat"]
    assert neither[CLS_HAS_CONTEXT_CARD] == 0.0
    assert neither[CLS_HAS_EFFECT] == 0.0

    # Set apart so the two cannot be confused with each other.
    ctx_only = featurize(_obs(3, 3, card, None), vocab, [0])["cls_feat"]
    assert ctx_only[CLS_HAS_CONTEXT_CARD] == 1.0
    assert ctx_only[CLS_HAS_EFFECT] == 0.0


def test_toss_undecided_is_zero_once_resolved():
    examined = 0
    ep = _load_episode()
    vocab = _build_vocab(ep)
    for obs, action in _active_decisions(ep):
        if obs["current"]["firstPlayer"] == -1:
            continue
        cls = featurize(obs, vocab, action)["cls_feat"]
        assert cls[CLS_TOSS_UNDECIDED] == 0.0
        examined += 1

    assert examined > 0, "fixture covered no post-toss decisions"
