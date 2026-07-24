"""Tests for ptcg_mine.archetype: canon, jaccard, clustering, self/opp, fixed deck."""

from collections import Counter

import pytest

from ptcg_mine.archetype import (
    assign_archetype,
    canon,
    cluster_decks,
    jaccard_multiset,
    multiset_counts,
    pick_fixed_deck,
    select_self_opp,
)

DECK_A = list(range(60))  # 0..59
DECK_B_3DIFF = list(range(57)) + [1000, 1001, 1002]  # 57/60 shared with A
DECK_C_10DIFF = list(range(50)) + [2000 + i for i in range(10)]  # 50/60 shared with A
DECK_DISJOINT = list(range(1000, 1060))


def _make_ep(t0, t1, r0, r1, deck0, deck1):
    return {
        "info": {"TeamNames": [t0, t1]},
        "rewards": [r0, r1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [{"action": None}, {"action": None}],
            [{"action": deck0}, {"action": deck1}],
        ],
    }


def test_canon_sorts_deck():
    assert canon([3, 1, 2]) == (1, 2, 3)


def test_multiset_counts():
    assert multiset_counts([1, 1, 2]) == Counter({1: 2, 2: 1})


def test_jaccard_identical_is_one():
    assert jaccard_multiset(canon(DECK_A), canon(DECK_A)) == pytest.approx(1.0)


def test_jaccard_disjoint_is_zero():
    assert jaccard_multiset(canon(DECK_A), canon(DECK_DISJOINT)) == pytest.approx(0.0)


def test_jaccard_empty_is_zero():
    assert jaccard_multiset((), ()) == 0.0


def test_jaccard_57_of_60_shared_is_at_least_point_nine():
    score = jaccard_multiset(canon(DECK_A), canon(DECK_B_3DIFF))
    assert score >= 0.9
    assert score == pytest.approx(57 / 63)


def test_jaccard_50_of_60_shared_is_below_point_nine():
    score = jaccard_multiset(canon(DECK_A), canon(DECK_C_10DIFF))
    assert score < 0.9
    assert score == pytest.approx(50 / 70)


def test_cluster_merges_close_decks_and_separates_far_deck():
    deck_freq = {
        canon(DECK_A): 100,
        canon(DECK_B_3DIFF): 50,
        canon(DECK_C_10DIFF): 40,
    }
    archetypes = cluster_decks(deck_freq, thresh=0.90)
    assert len(archetypes) == 2

    merged = next(a for a in archetypes if a.representative == canon(DECK_A))
    assert merged.frequency == 150
    assert canon(DECK_B_3DIFF) in merged.members

    separate = next(a for a in archetypes if a.representative == canon(DECK_C_10DIFF))
    assert separate.frequency == 40


def test_assign_archetype_matches_close_deck_and_none_for_disjoint():
    deck_freq = {canon(DECK_A): 100, canon(DECK_C_10DIFF): 40}
    archetypes = cluster_decks(deck_freq, thresh=0.90)
    close_id = assign_archetype(DECK_B_3DIFF, archetypes)
    assert close_id is not None

    # Close deck should join the DECK_A cluster.
    a_arch = next(a for a in archetypes if a.representative == canon(DECK_A))
    assert close_id == a_arch.id

    assert assign_archetype(DECK_DISJOINT, archetypes) is None


DECK_X = list(range(60))  # 0..59
DECK_Y = list(range(1000, 1060))  # fully disjoint from DECK_X


def _self_opp_fixture():
    archetypes = cluster_decks({canon(DECK_X): 10, canon(DECK_Y): 10}, thresh=0.90)
    # Deterministic tie-break puts DECK_X's cluster first (lower canon tuple).
    arch_x = next(a for a in archetypes if a.representative == canon(DECK_X))
    arch_y = next(a for a in archetypes if a.representative == canon(DECK_Y))

    experts = ["expertA", "expertB"]
    episodes = [
        # expertA plays DECK_X, wins.
        _make_ep("expertA", "opponentZ", 1, -1, DECK_X, DECK_Y),
        # expertA plays DECK_X again, loses (still counts toward self freq).
        _make_ep("expertA", "opponentZ", -1, 1, DECK_X, DECK_Y),
        # expertB plays DECK_Y, wins.
        _make_ep("expertB", "opponentW", 1, -1, DECK_Y, DECK_X),
    ]
    return episodes, experts, archetypes, arch_x, arch_y


def test_select_self_opp_counts_own_and_opponent_decks():
    episodes, experts, archetypes, arch_x, arch_y = _self_opp_fixture()

    self_ids, opp_ids = select_self_opp(episodes, experts, archetypes, n_self=2, n_opp=2)
    # self: arch_x played twice (expertA), arch_y played once (expertB) -> arch_x first.
    assert self_ids == [arch_x.id, arch_y.id]
    # opp: arch_y seen twice (as opponentZ's deck), arch_x seen once -> arch_y first.
    assert opp_ids == [arch_y.id, arch_x.id]


def test_select_self_opp_respects_n_self_limit():
    episodes, experts, archetypes, arch_x, arch_y = _self_opp_fixture()
    self_ids, _ = select_self_opp(episodes, experts, archetypes, n_self=1, n_opp=2)
    assert self_ids == [arch_x.id]


def test_pick_fixed_deck_returns_highest_winrate_representative():
    episodes, experts, archetypes, arch_x, arch_y = _self_opp_fixture()
    self_ids = [arch_x.id, arch_y.id]

    # arch_x (DECK_X): 1 win / 2 games = 0.5 win rate as expert's own deck.
    # arch_y (DECK_Y): 1 win / 1 game = 1.0 win rate as expert's own deck.
    fixed_deck = pick_fixed_deck(episodes, experts, self_ids, archetypes)
    assert fixed_deck == list(arch_y.representative)
    assert len(fixed_deck) == 60
    assert all(isinstance(c, int) for c in fixed_deck)
