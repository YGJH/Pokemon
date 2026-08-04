"""The point of the whole change: cards must stop being indistinguishable.

Thresholds are floors, set under the measured values so a pattern edit that
regresses separation goes red without any refactor tripping the gate.
"""

import collections

import pytest

from ptcg_mine.cards import build_engine_card_features, load_engine

CARD_TYPE_FLOORS = {1: ("ITEM", 25), 2: ("TOOL", 12), 3: ("SUPPORTER", 25), 4: ("STADIUM", 12)}


@pytest.fixture(scope="module")
def engine_rows():
    cards, attacks = load_engine()
    return cards, build_engine_card_features(cards, attacks)


def test_trainer_cards_are_separated(engine_rows):
    cards, feats = engine_rows
    checked = 0
    for ctype, (name, floor) in CARD_TYPE_FLOORS.items():
        sub = [c for c in cards if int(c.cardType) == ctype]
        assert sub, f"no {name} cards in the engine tables"
        distinct = len({feats[c.cardId].tobytes() for c in sub})
        assert distinct >= floor, (
            f"{name}: {distinct} distinct rows over {len(sub)} cards, floor {floor}"
        )
        checked += 1
    assert checked == 4, "fixture drift: expected 4 trainer cardTypes"


def test_no_whole_card_type_is_uncovered(engine_rows):
    """An all-zero keyword row is correct for a vanilla card, but a whole
    cardType coming back empty means the patterns missed that card class."""
    from ptcg_mine.keywords import ability_keyword_row, attack_keyword_row

    cards, _ = engine_rows
    _, attacks = load_engine()
    by_aid = {a.attackId: a for a in attacks}
    covered = collections.Counter()
    total = collections.Counter()
    for c in cards:
        t = int(c.cardType)
        total[t] += 1
        hit = bool(ability_keyword_row(c).any()) or bool(any(
            attack_keyword_row(by_aid[int(a)]).any()
            for a in c.attacks if int(a) in by_aid
        ))
        covered[t] += bool(hit)
    assert total, "no cards examined"
    for t in (0, 1, 2, 3, 4):   # POKEMON, ITEM, TOOL, SUPPORTER, STADIUM
        assert covered[t] > 0, f"cardType {t}: 0 of {total[t]} cards matched any keyword"
