"""Effect-keyword extraction from card and attack oracle text."""

import numpy as np
import pytest

from ptcg_mine import keywords as kw


class _Skill:
    def __init__(self, text):
        self.name = "s"
        self.text = text


class _Card:
    def __init__(self, *texts):
        self.skills = [_Skill(t) for t in texts]


class _Attack:
    def __init__(self, text):
        self.text = text


def test_k_effect_matches_list_length():
    assert kw.K_EFFECT == len(kw.KEYWORDS) == len(kw.KEYWORD_NAMES) == 29


def test_keyword_order_is_pinned():
    """KEYWORDS index IS the feature column. Inserting in the middle silently
    repoints every later column in every trained checkpoint."""
    assert kw.KEYWORD_NAMES[:6] == (
        "draw", "search_deck", "deck_look",
        "discard_own", "discard_opp", "hand_disrupt",
    )
    assert kw.KEYWORD_NAMES[-1] == "recover_discard"


def test_row_is_float32_binary_of_correct_width():
    row = kw.effect_keyword_row(["Draw 3 cards."])
    assert row.shape == (29,)
    assert row.dtype == np.float32
    assert set(np.unique(row)) <= {0.0, 1.0}


def test_empty_text_gives_all_zero_row():
    assert not kw.effect_keyword_row([]).any()
    assert not kw.effect_keyword_row(["", None]).any()


def test_matching_is_case_insensitive():
    """Oracle text capitalises the leading verb. A case-sensitive `draw \\d+ card`
    misses 'Draw 3 cards.' — i.e. misses Cheren entirely."""
    i = kw.KEYWORD_NAMES.index("draw")
    assert kw.effect_keyword_row(["Draw 3 cards."])[i] == 1.0
    assert kw.effect_keyword_row(["draw 3 cards."])[i] == 1.0


@pytest.mark.parametrize("name,positive,negative", [
    ("draw", "Draw 3 cards.", "Discard your hand."),
    ("search_deck", "Search your deck for a Trainer card.", "Look at the top 7 cards of your deck."),
    ("deck_look", "Look at the top 7 cards of your deck.", "Search your deck for a Trainer card."),
    ("gust", "Switch in 1 of your opponent's Benched Pokemon to the Active Spot.",
             "Switch this Pokemon with 1 of your Benched Pokemon."),
    ("switch_own", "Switch this Pokemon with 1 of your Benched Pokemon.",
                   "Switch in 1 of your opponent's Benched Pokemon to the Active Spot."),
    ("heal", "Heal 70 damage from your Active Pokemon.", "Draw 3 cards."),
    ("status_offensive", "Your opponent's Active Pokemon is now Poisoned.",
                         "Your opponent's Active Pokemon is now Confused."),
    ("status_lock", "Your opponent's Active Pokemon is now Confused.",
                    "Your opponent's Active Pokemon is now Poisoned."),
    ("coin_flip", "Flip a coin. If heads, this attack does 30 more damage.",
                  "This attack does 30 more damage."),
    ("ability_lock", "Pokemon in play have no Abilities.", "Draw 3 cards."),
    ("once_per_turn", "Once during your turn, you may draw a card.", "Draw 3 cards."),
    ("recover_discard", "Put a Pokemon from your discard pile into your hand.",
                        "Put a Pokemon from your deck into your hand."),
])
def test_keyword_positive_and_near_miss(name, positive, negative):
    i = kw.KEYWORD_NAMES.index(name)
    assert kw.effect_keyword_row([positive])[i] == 1.0, f"{name} missed a positive"
    assert kw.effect_keyword_row([negative])[i] == 0.0, f"{name} fired on a near-miss"


def test_ability_row_reads_every_skill():
    card = _Card("Draw 3 cards.", "Heal 20 damage from this Pokemon.")
    row = kw.ability_keyword_row(card)
    assert row[kw.KEYWORD_NAMES.index("draw")] == 1.0
    assert row[kw.KEYWORD_NAMES.index("heal")] == 1.0


def test_card_with_no_skills_is_all_zero():
    assert not kw.ability_keyword_row(_Card()).any()


def test_attack_row_reads_attack_text():
    row = kw.attack_keyword_row(_Attack("Flip a coin. If heads, your opponent's Active Pokemon is now Paralyzed."))
    assert row[kw.KEYWORD_NAMES.index("coin_flip")] == 1.0
    assert row[kw.KEYWORD_NAMES.index("status_lock")] == 1.0
