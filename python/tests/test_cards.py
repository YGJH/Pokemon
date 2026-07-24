"""Tests for ptcg_mine.cards: load_engine, card_static_row, attack_static_row,
build_static_tables.

Layout is exact Appendix A.3 / A.2 (TRANSFORMER_IL_SPEC.md):
  card_static_row[52] = [hp/400, retreat/4, cardType(7), stage(3), energyType(12),
                         weakness(12), resistance(12), [ex,megaEx,tera,aceSpec](4)]
  attack_static_row[14] = [damage/350, energy-cost histogram(12), len(energies)/5]
"""

import numpy as np
import pytest

from ptcg_mine.cards import attack_static_row, build_static_tables, card_static_row, load_engine


@pytest.fixture(scope="module")
def engine():
    return load_engine()


def test_load_engine_returns_cards_and_attacks(engine):
    cards, attacks = engine
    assert len(cards) > 0
    assert len(attacks) > 0


def test_card_static_row_shape(engine):
    cards, _ = engine
    row = card_static_row(cards[0])
    assert row.shape == (52,)
    assert row.dtype == np.float32


def test_card_static_row_known_card_basic_grass_energy(engine):
    """Card id 1 = "Basic {G} Energy": cardType=BASIC_ENERGY(5), energyType=GRASS(1),
    hp=0, retreatCost=0, no weakness/resistance, not basic/stage1/stage2, no ex/megaEx/tera/aceSpec."""
    cards, _ = engine
    card = next(c for c in cards if c.cardId == 1)
    row = card_static_row(card)

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
    assert row.shape == (14,)
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
    np.testing.assert_array_equal(hist, expected_hist)
    assert row[13] == pytest.approx(len(attack.energies) / 5.0)


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

    assert card_table.shape == (vocab["size"], 52)
    assert card_table.dtype == np.float32
    np.testing.assert_array_equal(card_table[0], np.zeros(52, dtype=np.float32))

    # row 1 (UNKNOWN) = mean of in-vocab card rows (indices >= 2)
    expected_unknown = card_table[2:].mean(axis=0)
    np.testing.assert_allclose(card_table[1], expected_unknown, rtol=1e-5, atol=1e-6)

    assert attack_table.shape[1] == 14
    assert attack_table.dtype == np.float32
    np.testing.assert_array_equal(attack_table[0], np.zeros(14, dtype=np.float32))
    assert 0 not in attack_id_to_index.values()  # PAD row reserved, no attackId maps to it
    assert attack_table.shape[0] == len(attack_id_to_index) + 1
