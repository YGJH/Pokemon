"""Tests for ptcg_mine.cards: load_engine, card_static_row, attack_static_row,
build_static_tables.

Layout is exact Appendix A.3 / A.2 (TRANSFORMER_IL_SPEC.md):
  card_static_row[94] = base[52] + 3 × attack_static_row[14]
  base[52]            = [hp/400, retreat/4, cardType(7), stage(3), energyType(12),
                         weakness(12), resistance(12), [ex,megaEx,tera,aceSpec](4)]
  attack_static_row[14] = [damage/350, energy-cost histogram(12), len(energies)/5]
"""

import numpy as np
import pytest

from ptcg_mine.cards import attack_static_row, build_static_tables, card_static_row, load_engine


@pytest.fixture(scope="module")
def engine():
    return load_engine()


@pytest.fixture(scope="module")
def attacks_by_id(engine):
    """``{attackId: Attack}`` — ``card.attacks`` holds ids, not Attack objects."""
    _, attacks = engine
    return {a.attackId: a for a in attacks}


def test_load_engine_returns_cards_and_attacks(engine):
    cards, attacks = engine
    assert len(cards) > 0
    assert len(attacks) > 0


def test_card_static_row_shape(engine, attacks_by_id):
    cards, _ = engine
    row = card_static_row(cards[0], attacks_by_id)
    assert row.shape == (218,)  # 52 base + 29 ability kw + 2 counts + 3 attacks × 45
    assert row.dtype == np.float32


def test_card_static_row_attack_slice_matches_attack_static_row(engine, attacks_by_id):
    """The 3 attack slots repeat attack_static_row for the card's own attacks.

    ``card.attacks`` holds attack *ids*; resolving them through *attacks_by_id*
    is what makes the 52:94 half carry anything at all.
    """
    cards, _ = engine
    card = next(c for c in cards
                if len(getattr(c, "attacks", []) or []) > 0
                and int((c.attacks or [0])[0]) in attacks_by_id)
    row = card_static_row(card, attacks_by_id)

    from ptcg_mine.cards import CARD_ATTACK_BLOCK_START, F_ATK
    for ai, aid in enumerate(card.attacks[:3]):
        expected = attack_static_row(attacks_by_id[int(aid)])
        np.testing.assert_array_equal(
            row[CARD_ATTACK_BLOCK_START + ai * F_ATK:
                CARD_ATTACK_BLOCK_START + (ai + 1) * F_ATK],
            expected)

    # Unused attack slots are zero-padded
    for ai in range(len(card.attacks[:3]), 3):
        np.testing.assert_array_equal(
            row[CARD_ATTACK_BLOCK_START + ai * F_ATK:
                CARD_ATTACK_BLOCK_START + (ai + 1) * F_ATK],
            np.zeros(F_ATK, dtype=np.float32)
        )


def test_card_static_row_known_card_basic_grass_energy(engine, attacks_by_id):
    """Card id 1 = "Basic {G} Energy": cardType=BASIC_ENERGY(5), energyType=GRASS(1),
    hp=0, retreatCost=0, no weakness/resistance, not basic/stage1/stage2, no ex/megaEx/tera/aceSpec."""
    cards, _ = engine
    card = next(c for c in cards if c.cardId == 1)
    row = card_static_row(card, attacks_by_id)

    assert row[0] == pytest.approx(0.0)  # hp / 400
    assert row[1] == pytest.approx(0.0)  # retreatCost / 4

    card_type_onehot = row[2:9]
    expected_type = np.zeros(7, dtype=np.float32)
    expected_type[5] = 1.0  # BASIC_ENERGY
    np.testing.assert_array_equal(card_type_onehot, expected_type)

    stage = row[9:12]
    np.testing.assert_array_equal(stage, np.zeros(3, dtype=np.float32))

    energy_onehot = row[12:24]
    expected_energy = np.zeros(12, dtype=np.float32)
    expected_energy[1] = 1.0  # GRASS
    np.testing.assert_array_equal(energy_onehot, expected_energy)

    weakness_onehot = row[24:36]
    np.testing.assert_array_equal(weakness_onehot, np.zeros(12, dtype=np.float32))

    resistance_onehot = row[36:48]
    np.testing.assert_array_equal(resistance_onehot, np.zeros(12, dtype=np.float32))

    flags = row[48:52]
    np.testing.assert_array_equal(flags, np.zeros(4, dtype=np.float32))


def test_attack_static_row_shape(engine):
    _, attacks = engine
    row = attack_static_row(attacks[0])
    assert row.shape == (45,)
    assert row.dtype == np.float32


def test_attack_static_row_known_values(engine):
    _, attacks = engine
    attack = next(a for a in attacks if a.attackId == 1)
    row = attack_static_row(attack)
    assert row[0] == pytest.approx(attack.damage / 350.0)
    hist = row[1:13]
    expected_hist = np.zeros(12, dtype=np.float32)
    for e in attack.energies:
        expected_hist[int(e)] += 1.0
    np.testing.assert_allclose(hist, expected_hist / 5.0, rtol=1e-6)
    assert row[13] == pytest.approx(len(attack.energies) / 5.0)
    # The histogram sums to the same value the length term reports -- they are
    # two views of one cost and must not drift onto different scales.
    assert hist.sum() == pytest.approx(row[13], rel=1e-6)


def test_attack_static_row_is_normalized(engine):
    """No attack feature exceeds 1.0.

    The energy-cost histogram used to be raw counts (up to 5.0) while every
    other feature was in [0, 1]; card_static_row repeats it three times, so the
    unnormalized version put 36 of 94 card dims on a 5x scale.
    """
    _, attacks = engine
    n_with_cost = 0
    worst = 0.0
    for attack in attacks:
        row = attack_static_row(attack)
        assert row.min() >= 0.0
        worst = max(worst, float(row.max()))
        if len(attack.energies) > 0:
            n_with_cost += 1
    assert n_with_cost > 0, "no attack in the engine has an energy cost"
    assert worst <= 1.0, f"attack feature exceeds 1.0 (max {worst})"


def test_card_static_row_is_normalized(engine, attacks_by_id):
    """The assembled 94-dim card row stays in [0, 1] too."""
    cards, _ = engine
    n_with_attacks = 0
    worst = 0.0
    for card in cards:
        row = card_static_row(card, attacks_by_id)
        worst = max(worst, float(row.max()))
        if getattr(card, "attacks", None):
            n_with_attacks += 1
    assert n_with_attacks > 0, "no card in the engine has attacks"
    assert worst <= 1.0, f"card feature exceeds 1.0 (max {worst})"


def test_build_static_tables_shapes_and_pad_row(engine):
    from ptcg_mine.vocab import build_vocab

    cards, attacks = engine
    # synthetic corpus: decks using a handful of real card ids
    sample_ids = [c.cardId for c in cards[:5]]
    deck0 = (sample_ids * 12)[:60]
    ep = {
        "info": {"TeamNames": ["A", "B"]},
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [{"action": None}, {"action": None}],
            [{"action": deck0}, {"action": deck0}],
        ],
    }
    vocab = build_vocab([ep], mode="all_corpus")

    card_table, attack_id_to_index, attack_table = build_static_tables(vocab, cards, attacks)

    from ptcg_mine.cards import F_CARD
    assert card_table.shape == (vocab["size"], F_CARD)
    assert card_table.dtype == np.float32
    np.testing.assert_array_equal(card_table[0], np.zeros(F_CARD, dtype=np.float32))

    # row 1 (UNKNOWN) = mean of in-vocab card rows (indices >= 2)
    expected_unknown = card_table[2:].mean(axis=0)
    np.testing.assert_allclose(card_table[1], expected_unknown, rtol=1e-5, atol=1e-6)

    assert attack_table.shape[1] == 45
    assert attack_table.dtype == np.float32
    np.testing.assert_array_equal(attack_table[0], np.zeros(45, dtype=np.float32))
    assert 0 not in attack_id_to_index.values()  # PAD row reserved, no attackId maps to it
    assert attack_table.shape[0] == len(attack_id_to_index) + 1


def test_evolution_map_resolves_names_to_ids():
    from ptcg_mine.cards import build_evolution_map

    class _C:
        def __init__(self, cid, name, evolves_from):
            self.cardId, self.name, self.evolvesFrom = cid, name, evolves_from

    cards = [_C(1, "Charmander", None), _C(2, "Charmeleon", "Charmander"),
             _C(3, "Charmander", None)]     # reprint: same name, different id
    m = build_evolution_map(cards)
    assert sorted(m[2]) == [1, 3], "a reprint of the pre-evolution must also count"
    assert m[1] == []


def test_evolution_map_handles_unresolvable_names(caplog):
    from ptcg_mine.cards import build_evolution_map

    class _C:
        def __init__(self, cid, name, evolves_from):
            self.cardId, self.name, self.evolvesFrom = cid, name, evolves_from

    m = build_evolution_map([_C(1, "Charmeleon", "Charmander")])
    assert m[1] == []
    assert "Charmander" in caplog.text


def test_evolution_map_covers_the_real_engine():
    from ptcg_mine.cards import build_evolution_map, load_engine

    cards, _ = load_engine()
    m = build_evolution_map(cards)
    evolvers = [c for c in cards if getattr(c, "evolvesFrom", None)]
    assert evolvers, "no evolution cards in the engine tables"
    unresolved = [c.name for c in evolvers if not m.get(c.cardId)]
    assert not unresolved, f"unresolved pre-evolution names: {unresolved[:10]}"
