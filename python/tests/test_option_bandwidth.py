"""Tests for per-option decision bandwidth (2026-08-09 spec)."""

import numpy as np
import pytest

from ptcg_il.featurizer import (
    F_OPT,
    F_ATK,
    F_CARD,
    CARD_ATTACK_BLOCK_START,
    DRAW_N,
    O_MAX,
    _best_bench_damage_ratio,
    _best_bench_hp_ratio,
    option_groups,
)


# ============================================================
# TestBestBenchDamageRatio
# ============================================================

class TestBestBenchDamageRatio:
    def test_empty_bench_returns_zero(self):
        """Empty bench (all HP=0) returns ratio 0.0."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        opp = np.zeros(F_CARD, dtype=np.float32)
        opp[0] = 100.0 / 400.0  # 100 HP
        ratio, slot = _best_bench_damage_ratio(poke_card, opp)
        assert ratio == 0.0

    def test_bench_with_damage_attack(self):
        """A bench Pokemon with a damaging attack returns the correct ratio."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        # Bench slot 2 (array index 3, bench position 2) has 200 HP and one
        # attack doing 60 damage.
        poke_card[3, 0] = 200.0 / 400.0  # HP
        poke_card[3, CARD_ATTACK_BLOCK_START + 0 * F_ATK] = 60.0 / 350.0  # damage
        # Opponent active: 120 HP
        opp = np.zeros(F_CARD, dtype=np.float32)
        opp[0] = 120.0 / 400.0
        ratio, slot = _best_bench_damage_ratio(poke_card, opp)
        # 60 / (120 + 10) = 60/130 ≈ 0.4615
        assert ratio == pytest.approx(60.0 / (120.0 + 10.0), rel=0.01)
        assert slot == 2  # bench position 2 (0-based)

    def test_tie_break_lowest_slot(self):
        """Equal damage, equal HP — lowest array index (lowest bench position) wins."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        # Two bench slots with same HP and same damage.
        for slot in (1, 3):
            poke_card[slot, 0] = 150.0 / 400.0
            poke_card[slot, CARD_ATTACK_BLOCK_START] = 50.0 / 350.0
        opp = np.zeros(F_CARD, dtype=np.float32)
        opp[0] = 100.0 / 400.0
        _, slot = _best_bench_damage_ratio(poke_card, opp)
        # Index 1 is visited before index 3, so bench position 0 wins.
        assert slot == 0

    def test_opp_hp_zero_or_negative_returns_zero(self):
        """If opponent active HP is zero (normalised), bail out early."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        poke_card[1, 0] = 200.0 / 400.0
        poke_card[1, CARD_ATTACK_BLOCK_START] = 60.0 / 350.0
        opp = np.zeros(F_CARD, dtype=np.float32)
        opp[0] = 0.0  # zero HP — the guard trips at 0 · 400 + 10 ≤ 10
        ratio, slot = _best_bench_damage_ratio(poke_card, opp)
        assert ratio == 0.0

    def test_zero_damage_attacks_skipped(self):
        """Attacks with zero damage are skipped; only damaging attacks count."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        # Bench slot has HP but only an attack with zero damage.
        poke_card[1, 0] = 150.0 / 400.0
        poke_card[1, CARD_ATTACK_BLOCK_START] = 0.0 / 350.0  # zero damage
        opp = np.zeros(F_CARD, dtype=np.float32)
        opp[0] = 100.0 / 400.0
        ratio, _ = _best_bench_damage_ratio(poke_card, opp)
        assert ratio == 0.0


# ============================================================
# TestBestBenchHpRatio
# ============================================================

class TestBestBenchHpRatio:
    def test_returns_ratio(self):
        """Bench Pokemon with more HP than active returns capped ratio 1.0."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        poke_card[2, 0] = 200.0 / 400.0  # array index 2, bench position 1
        active = np.zeros(F_CARD, dtype=np.float32)
        active[0] = 100.0 / 400.0
        ratio, slot = _best_bench_hp_ratio(poke_card, active)
        # 200/100 = 2.0, clipped to 1.0
        assert ratio == 1.0
        assert slot == 1  # bench position 1 (0-based)

    def test_empty_bench_returns_zero(self):
        """When no bench Pokemon has HP > 0, returns 0.0."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        active = np.zeros(F_CARD, dtype=np.float32)
        active[0] = 100.0 / 400.0
        ratio, slot = _best_bench_hp_ratio(poke_card, active)
        assert ratio == 0.0

    def test_active_zero_hp_returns_zero(self):
        """If the active has zero HP, bail out early."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        poke_card[1, 0] = 200.0 / 400.0
        active = np.zeros(F_CARD, dtype=np.float32)
        active[0] = 0.0
        ratio, slot = _best_bench_hp_ratio(poke_card, active)
        assert ratio == 0.0

    def test_tie_break_lowest_slot(self):
        """Equal HP ratios — lowest array index wins."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        for slot in (1, 4):
            poke_card[slot, 0] = 150.0 / 400.0
        active = np.zeros(F_CARD, dtype=np.float32)
        active[0] = 100.0 / 400.0
        _, slot = _best_bench_hp_ratio(poke_card, active)
        assert slot == 0  # index 1 wins over index 4


# ============================================================
# TestDrawCountParsing
# ============================================================

class TestDrawCountParsing:
    def test_all_draw_attacks_parse(self):
        """Every attack whose text mentions 'draw' must parse draw_fixed or draw_to_hand."""
        from ptcg_mine.keywords import draw_fixed, draw_to_hand
        from ptcg_mine.cards import load_engine

        _, atks = load_engine()
        import re

        draw_atks = [
            a for a in atks
            if re.search(r'draw', getattr(a, 'text', '') or '', re.I)
        ]
        assert len(draw_atks) > 0, "no draw attacks examined — test is vacuous"
        parsed = 0
        for a in draw_atks:
            f = draw_fixed(a)
            h = draw_to_hand(a)
            if f > 0 or h > 0:
                parsed += 1
        # At least the explicit-number and draw-a-card forms must parse
        assert parsed >= 16, (
            f"only {parsed}/{len(draw_atks)} draw attacks parsed; "
            f"expected >= 16"
        )

    def test_draw_a_card_is_fixed_1(self):
        """'Draw a card.' must parse as draw_fixed=1 and draw_to_hand=0."""
        from ptcg_mine.keywords import draw_fixed, draw_to_hand
        from ptcg_mine.cards import load_engine

        _, atks = load_engine()
        found = False
        for a in atks:
            t = getattr(a, 'text', '') or ''
            if 'draw a card' in t.lower() and 'draw a card.' in t.lower():
                assert draw_fixed(a) == 1, f"'Draw a card.' parsed as {draw_fixed(a)}"
                assert draw_to_hand(a) == 0
                found = True
                break
        assert found, "no 'Draw a card.' attack found — test is vacuous"

    def test_draw_to_hand_is_parsed(self):
        """'draw cards until you have 7 cards' must parse as draw_to_hand=7."""
        from ptcg_mine.keywords import draw_fixed, draw_to_hand
        from ptcg_mine.cards import load_engine

        _, atks = load_engine()
        found = False
        for a in atks:
            t = getattr(a, 'text', '') or ''
            if 'draw cards until you have 7 cards' in t.lower():
                assert draw_to_hand(a) == 7, f"draw_to_hand={draw_to_hand(a)}"
                found = True
                break
        assert found, "no 'draw cards until you have 7 cards' attack found"

    def test_draw_fixed_and_to_hand_are_exclusive(self):
        """An attack should never have both draw_fixed > 0 and draw_to_hand > 0."""
        from ptcg_mine.keywords import draw_fixed, draw_to_hand
        from ptcg_mine.cards import load_engine

        _, atks = load_engine()
        for a in atks:
            f = draw_fixed(a)
            h = draw_to_hand(a)
            assert f == 0 or h == 0, (
                f"attack has both draw_fixed={f} and draw_to_hand={h}: "
                f"{getattr(a, 'text', '')[:80]}"
            )


# ============================================================
# TestDimAssertions
# ============================================================

class TestDimAssertions:
    def test_cards_assertions_pass(self):
        """Importing attack_static_row must succeed (module-level assertions)."""
        from ptcg_mine.cards import attack_static_row  # noqa: F401
        assert True  # reached without AssertionError

    def test_k_effect_consistency(self, monkeypatch):
        """F_ATK and F_CARD must be consistent with K_EFFECT."""
        from ptcg_mine.keywords import K_EFFECT
        from ptcg_il.featurizer import F_ATK, F_CARD

        # +1 for bench damage at col 16 (the Active `damage` field is col 0).
        assert F_ATK == 17 + K_EFFECT, (
            f"F_ATK={F_ATK}, expected {17 + K_EFFECT}"
        )
        # +2 for the card-level draw counts at 83:85 (deck-out features).
        assert F_CARD == 52 + K_EFFECT + 2 + 2 + 3 * F_ATK, (
            f"F_CARD={F_CARD}, expected {52 + K_EFFECT + 2 + 2 + 3 * F_ATK}"
        )


# ============================================================
# TestOptionGroups
# ============================================================

class TestOptionGroups:
    def test_bench_idx_affects_groups(self):
        """Two options differing only in opt_bench_idx get different groups."""
        O = 64
        opt_type = np.zeros(O, dtype=np.int64)
        opt_src = np.full(O, -1, dtype=np.int64)
        opt_tgt = np.full(O, -1, dtype=np.int64)
        opt_bench = np.full(O, -1, dtype=np.int64)
        opt_bench[0] = 3  # different bench ref
        opt_bench[1] = 5  # different bench ref
        opt_card = np.zeros((O, F_CARD), dtype=np.float32)
        opt_atk = np.zeros((O, F_ATK), dtype=np.float32)
        opt_scalar = np.zeros((O, F_OPT), dtype=np.float32)
        opt_mask = np.zeros(O, dtype=bool)
        opt_mask[0] = True
        opt_mask[1] = True
        groups = option_groups(
            opt_type, opt_src, opt_tgt, opt_bench,
            opt_card, opt_atk, opt_scalar, opt_mask,
        )
        assert groups[0] != groups[1], (
            f"options with different bench refs got same group {groups[0]}"
        )

    def test_bench_idx_not_in_key_merges(self):
        """Without opt_bench_idx in the key, two bench-different options merge.

        This is the mutation test — reproduce the old grouping key (ints
        without opt_bench_idx) and verify the groups become equal.
        """
        O = 64
        opt_type = np.zeros(O, dtype=np.int64)
        opt_src = np.full(O, -1, dtype=np.int64)
        opt_tgt = np.full(O, -1, dtype=np.int64)
        opt_bench = np.full(O, -1, dtype=np.int64)
        opt_bench[0] = 3
        opt_bench[1] = 5
        opt_card = np.zeros((O, F_CARD), dtype=np.float32)
        opt_atk = np.zeros((O, F_ATK), dtype=np.float32)
        opt_scalar = np.zeros((O, F_OPT), dtype=np.float32)
        opt_mask = np.zeros(O, dtype=bool)
        opt_mask[0] = True
        opt_mask[1] = True
        # Simulate old behaviour: key without opt_bench_idx
        valid = np.flatnonzero(opt_mask)
        ints_old = np.stack(
            [opt_type[valid], opt_src[valid], opt_tgt[valid]], axis=1
        ).astype(np.int64)
        feats = np.concatenate(
            [opt_card[valid].astype(np.float32),
             opt_atk[valid].astype(np.float32)], axis=1
        )
        scal = opt_scalar[valid].astype(np.float32)
        lookup = {}
        for pos, slot in enumerate(valid):
            key = (ints_old[pos].tobytes(), feats[pos].tobytes(), scal[pos].tobytes())
            lookup.setdefault(key, len(lookup))
        # Without bench_idx, slots 0 and 1 are byte-identical — same group
        assert len(lookup) == 1, (
            f"without bench_idx, 2 options should be 1 group, got {len(lookup)}"
        )

    def test_masked_slots_are_negative_one(self):
        """Masked option slots always receive group -1."""
        O = 64
        opt_type = np.zeros(O, dtype=np.int64)
        opt_src = np.full(O, -1, dtype=np.int64)
        opt_tgt = np.full(O, -1, dtype=np.int64)
        opt_bench = np.full(O, -1, dtype=np.int64)
        opt_card = np.zeros((O, F_CARD), dtype=np.float32)
        opt_atk = np.zeros((O, F_ATK), dtype=np.float32)
        opt_scalar = np.zeros((O, F_OPT), dtype=np.float32)
        opt_mask = np.zeros(O, dtype=bool)
        # Only slot 5 is valid, all others masked.
        opt_mask[5] = True
        groups = option_groups(
            opt_type, opt_src, opt_tgt, opt_bench,
            opt_card, opt_atk, opt_scalar, opt_mask,
        )
        assert groups[5] == 0  # first valid slot
        for i in range(O):
            if i != 5:
                assert groups[i] == -1, f"masked slot {i} got group {groups[i]}"

    def test_identical_options_same_group(self):
        """Two valid options with identical features get the same group."""
        O = 64
        opt_type = np.ones(O, dtype=np.int64)
        opt_src = np.full(O, 2, dtype=np.int64)
        opt_tgt = np.full(O, 5, dtype=np.int64)
        opt_bench = np.full(O, 1, dtype=np.int64)
        opt_card = np.zeros((O, F_CARD), dtype=np.float32)
        opt_card[0, 0] = 1.0
        opt_card[1, 0] = 1.0  # identical to slot 0
        opt_atk = np.zeros((O, F_ATK), dtype=np.float32)
        opt_scalar = np.zeros((O, F_OPT), dtype=np.float32)
        opt_mask = np.zeros(O, dtype=bool)
        opt_mask[0] = True
        opt_mask[1] = True
        groups = option_groups(
            opt_type, opt_src, opt_tgt, opt_bench,
            opt_card, opt_atk, opt_scalar, opt_mask,
        )
        assert groups[0] == groups[1], (
            f"identical options got different groups: {groups[0]} vs {groups[1]}"
        )

    def test_different_options_different_groups(self):
        """Two valid options with different features get different groups."""
        O = 64
        opt_type = np.zeros(O, dtype=np.int64)
        opt_type[0] = 0
        opt_type[1] = 1  # different type
        opt_src = np.full(O, -1, dtype=np.int64)
        opt_tgt = np.full(O, -1, dtype=np.int64)
        opt_bench = np.full(O, -1, dtype=np.int64)
        opt_card = np.zeros((O, F_CARD), dtype=np.float32)
        opt_atk = np.zeros((O, F_ATK), dtype=np.float32)
        opt_scalar = np.zeros((O, F_OPT), dtype=np.float32)
        opt_mask = np.zeros(O, dtype=bool)
        opt_mask[0] = True
        opt_mask[1] = True
        groups = option_groups(
            opt_type, opt_src, opt_tgt, opt_bench,
            opt_card, opt_atk, opt_scalar, opt_mask,
        )
        assert groups[0] != groups[1], (
            "different-type options got the same group"
        )

    def test_no_valid_options_all_negative_one(self):
        """When no options are valid, all groups should be -1."""
        O = 64
        opt_type = np.zeros(O, dtype=np.int64)
        opt_src = np.full(O, -1, dtype=np.int64)
        opt_tgt = np.full(O, -1, dtype=np.int64)
        opt_bench = np.full(O, -1, dtype=np.int64)
        opt_card = np.zeros((O, F_CARD), dtype=np.float32)
        opt_atk = np.zeros((O, F_ATK), dtype=np.float32)
        opt_scalar = np.zeros((O, F_OPT), dtype=np.float32)
        opt_mask = np.zeros(O, dtype=bool)
        groups = option_groups(
            opt_type, opt_src, opt_tgt, opt_bench,
            opt_card, opt_atk, opt_scalar, opt_mask,
        )
        assert np.all(groups == -1), "all-masked should have all -1 groups"
