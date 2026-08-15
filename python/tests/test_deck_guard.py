"""Tests for ptcg_il/deck_guard.py — inference-time anti-deck-out guard.

Mechanism A hard-masks options whose resolution empties the deck before the
next-turn draw (engine RESULT reason 2 loss), with an exemption for
prize-winning attacks.  Mechanism B re-ranks near-tie options by net deck
delta, gated on (deck low, hand large) and never overriding a KO attack.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------- card table

class TestCardDeckDelta:
    """Net deck delta per card, derived from EN_Card_Data.csv oracle text.

    ``card_draw_counts`` (static cols 83:85) cannot carry these: it only
    parses "draw" phrasing, so search cards (Poké Pad et al.) read 0, Judge is
    deliberately skipped, and Lillie's Determination reads draw-6 while
    missing the shuffle-back — the wrong sign exactly when the hand is big.
    The explicit table is therefore the primary source, cols 83:85 the
    fallback for unlisted cards.
    """

    @staticmethod
    def delta(cid, hand=5, prizes=3, row=None):
        from ptcg_il.deck_guard import card_deck_delta

        return card_deck_delta(cid, hand=hand, prizes=prizes, card_row=row)

    def test_search_cards_burn_their_search_count(self):
        assert self.delta(1152) == -1   # Poké Pad: search a Pokémon
        assert self.delta(1121) == -1   # Ultra Ball: search a Pokémon
        assert self.delta(1086) == -2   # Buddy-Buddy Poffin: up to 2 Basics
        assert self.delta(1198) == -2   # Crispin: 2 energy (1 hand, 1 attached)
        assert self.delta(1231) == -3   # Dawn: Basic + Stage 1 + Stage 2

    def test_abilities_burn_their_draw(self):
        assert self.delta(140) == -3    # Fezandipiti ex: draw 3
        assert self.delta(120) == -1    # Drakloak: top-2 keep 1

    def test_shuffle_back_cards_are_positive_with_big_hands(self):
        assert self.delta(1227, hand=9) == 3    # Lillie's: +hand − 6
        assert self.delta(1213, hand=7) == 3    # Judge: +hand − 4
        assert self.delta(1080, hand=8) == 3    # Unfair Stamp: +hand − 5

    def test_shuffle_back_cards_can_still_burn(self):
        assert self.delta(1213, hand=2) == -2   # Judge with a tiny hand
        assert self.delta(1227, hand=4) == -2

    def test_lillie_draws_eight_at_six_prizes(self):
        # "If you have exactly 6 Prize cards remaining, draw 8 cards instead."
        assert self.delta(1227, hand=9, prizes=6) == 1

    def test_unlisted_card_without_row_is_zero(self):
        assert self.delta(999999) == 0

    def test_fallback_uses_static_draw_columns(self):
        from ptcg_il.deck_guard import (CARD_DRAW_FIXED_COL,
                                        CARD_DRAW_TO_HAND_COL, DRAW_N)

        row = np.zeros(223, dtype=np.float32)
        row[CARD_DRAW_FIXED_COL] = 3.0 / DRAW_N
        assert self.delta(999999, row=row) == -3

        row2 = np.zeros(223, dtype=np.float32)
        row2[CARD_DRAW_TO_HAND_COL] = 5.0 / DRAW_N
        assert self.delta(999999, hand=2, row=row2) == -3   # shortfall only
        assert self.delta(999999, hand=9, row=row2) == 0    # no draw happens

    def test_table_ids_resolve_in_the_engine_card_data(self):
        """Guards against id typos -- fails on zero examined."""
        from ptcg_mine.cards import load_engine

        cd, _ = load_engine()
        cards = list(cd.values()) if hasattr(cd, "values") else list(cd)
        by_id = {int(c.cardId): c for c in cards}
        from ptcg_il.deck_guard import BURN_TABLE

        n = 0
        for cid in BURN_TABLE:
            assert cid in by_id, f"table id {cid} not an engine card"
            n += 1
        assert n >= 10


# ------------------------------------------------------------- option deltas

class TestOptionDeckDeltas:
    """Per-option delta vector; only my own plays/abilities/attacks score."""

    @staticmethod
    def deltas(opt_type, opt_card_id, opt_atk_risk, deck=10, hand=5, prizes=3):
        from ptcg_il.deck_guard import option_deck_deltas

        return option_deck_deltas(
            opt_type=np.asarray(opt_type, dtype=np.int64),
            opt_card_id=np.asarray(opt_card_id, dtype=np.int64),
            opt_atk_risk=np.asarray(opt_atk_risk, dtype=np.float32),
            deck=deck, hand=hand, prizes=prizes,
            engine_card_features=None,
        )

    def test_play_and_skill_score_from_the_table(self):
        d = self.deltas([7, 15, 14], [1152, 140, 0], [0, 0, 0])
        assert d.tolist() == [-1, -3, 0]     # PLAY Poké Pad, SKILL Fez, END

    def test_attack_scores_from_the_attack_risk_column(self):
        d = self.deltas([13], [0], [0.5], deck=4)
        assert d.tolist() == [-2]            # risk 0.5 × deck 4 = 2 draws

    def test_attack_at_full_risk_empties_the_deck(self):
        d = self.deltas([13], [0], [1.0], deck=4)
        assert d.tolist() == [-4]

    def test_card_options_are_untouched(self):
        # CARD(3) options never occur in MAIN selects; even with a burn-table
        # id they must score 0 so search picks are never double-charged.
        d = self.deltas([3], [1152], [0])
        assert d.tolist() == [0]


# ---------------------------------------------------------------- mechanism A

class TestDeckoutKillMask:
    """Options leaving the deck empty before the next-turn draw are illegal."""

    @staticmethod
    def kill(opt_type, opt_card_id, opt_mask, deck, hand=5, prizes=3,
             opt_ko=None, opt_atk_risk=None):
        from ptcg_il.deck_guard import deckout_kill_mask

        n = len(opt_type)
        return deckout_kill_mask(
            opt_type=np.asarray(opt_type, dtype=np.int64),
            opt_card_id=np.asarray(opt_card_id, dtype=np.int64),
            opt_ko=np.asarray(opt_ko if opt_ko is not None else [0] * n,
                              dtype=np.float32),
            opt_atk_risk=np.asarray(
                opt_atk_risk if opt_atk_risk is not None else [0] * n,
                dtype=np.float32),
            opt_mask=np.asarray(opt_mask, dtype=bool),
            deck=deck, hand=hand, prizes=prizes,
            engine_card_features=None,
        )

    def test_poke_pad_at_deck_one_is_masked(self):
        k = self.kill([7, 14], [1152, 0], [True, True], deck=1)
        assert k.tolist() == [True, False]

    def test_poke_pad_at_deck_two_survives(self):
        k = self.kill([7, 14], [1152, 0], [True, True], deck=2)
        assert k.tolist() == [False, False]

    def test_lillie_survives_because_the_hand_refills_the_deck(self):
        k = self.kill([7, 7], [1227, 1152], [True, True], deck=1, hand=9)
        assert k.tolist() == [False, True]

    def test_prize_winning_ko_attack_is_exempt(self):
        k = self.kill([13], [0], [True], deck=2, prizes=1,
                      opt_ko=[1.0], opt_atk_risk=[1.0])
        assert k.tolist() == [False]

    def test_ko_attack_without_the_last_prize_is_not_exempt(self):
        k = self.kill([13, 14], [0, 0], [True, True], deck=2, prizes=2,
                      opt_ko=[1.0, 0], opt_atk_risk=[1.0, 0])
        assert k.tolist() == [True, False]

    def test_non_ko_attack_is_not_exempt(self):
        k = self.kill([13, 14], [0, 0], [True, True], deck=2, prizes=1,
                      opt_ko=[0.0, 0], opt_atk_risk=[1.0, 0])
        assert k.tolist() == [True, False]

    def test_a_lone_lethal_option_survives_the_floor(self):
        # The only legal option decks me out: masking it would leave argmax
        # over an all-dead mask (an arbitrary engine-legal pick), and the
        # outcome is identical anyway — the floor deliberately spares it.
        k = self.kill([13], [0], [True], deck=2, prizes=2,
                      opt_ko=[1.0], opt_atk_risk=[1.0])
        assert k.tolist() == [False]

    def test_never_masks_every_legal_option(self):
        # Both plays deck out; the least-bad (highest post) must survive.
        k = self.kill([7, 7], [1152, 1231], [True, True], deck=1)
        assert k.tolist() == [False, True]   # Poké Pad −1 survives over Dawn −3

    def test_dead_options_stay_dead_and_do_not_count_as_survivors(self):
        # END is already mask-dead; Poké Pad must still be masked even though
        # nothing else is legal -- the survivor floor counts *legal* options.
        k = self.kill([7, 14], [1152, 0], [True, False], deck=1)
        assert k.tolist() == [False, False]


# ---------------------------------------------------------------- mechanism B

class TestTiebreakPick:
    """Near-tie re-ranking by deck delta, gated on (deck low, hand large)."""

    @staticmethod
    def pick(logits, opt_type, opt_card_id, deck=5, hand=9, prizes=3,
             opt_ko=None, margin=0.5, deck_low=10, hand_gate=6):
        from ptcg_il.deck_guard import GuardConfig, tiebreak_pick

        n = len(opt_type)
        cfg = GuardConfig(deck_low=deck_low, hand_gate=hand_gate,
                          margin=margin)
        return tiebreak_pick(
            logits=np.asarray(logits, dtype=np.float32),
            opt_mask=np.ones(n, dtype=bool),
            opt_type=np.asarray(opt_type, dtype=np.int64),
            opt_card_id=np.asarray(opt_card_id, dtype=np.int64),
            opt_ko=np.asarray(opt_ko if opt_ko is not None else [0] * n,
                              dtype=np.float32),
            opt_atk_risk=np.zeros(n, dtype=np.float32),
            deck=deck, hand=hand, prizes=prizes,
            engine_card_features=None, config=cfg,
        )

    # options: 0 = Poké Pad (−1), 1 = Lillie's (+3 at hand 9), 2 = END (0)

    def test_within_margin_higher_delta_wins(self):
        p = self.pick([3.0, 2.8, 1.0], [7, 7, 14], [1152, 1227, 0])
        assert p == 1

    def test_beyond_margin_top_is_kept(self):
        p = self.pick([3.0, 2.0, 1.0], [7, 7, 14], [1152, 1227, 0])
        assert p == 0

    def test_gate_closed_when_deck_is_high(self):
        p = self.pick([3.0, 2.8, 1.0], [7, 7, 14], [1152, 1227, 0], deck=30)
        assert p == 0

    def test_gate_closed_when_hand_is_small(self):
        p = self.pick([3.0, 2.8, 1.0], [7, 7, 14], [1152, 1227, 0], hand=6)
        assert p == 0

    def test_ko_attack_on_top_is_never_overridden(self):
        p = self.pick([3.0, 2.9, 1.0], [13, 7, 14], [0, 1227, 0],
                      opt_ko=[1.0, 0, 0])
        assert p == 0

    def test_equal_delta_keeps_the_top(self):
        # END and Poké Pad… make two same-delta candidates within margin.
        p = self.pick([3.0, 2.9], [7, 7], [1152, 1121])  # both −1
        assert p == 0


# ------------------------------------------------------------- torch adapters

class TestDeckGuardTorch:
    """The batch-dict adapters the submission template and PolicyAgent call.

    ``sel_type``/``max_count`` are explicit arguments, not batch keys:
    live_eval's ``_sample_to_batch`` drops 0-d scalars, so a batch-keyed read
    would KeyError — or worse, silently disable the guard.
    """

    @staticmethod
    def _batch(opt_type, opt_card_id, opt_ko):
        torch = pytest.importorskip("torch")
        n = len(opt_type)
        opt_scalar = np.zeros((1, 64, 14), dtype=np.float32)
        opt_scalar[0, :n, 7] = opt_ko
        opt_mask = np.zeros((1, 64), dtype=bool)
        opt_mask[0, :n] = True
        return {
            "opt_type": torch.from_numpy(
                np.pad(opt_type, (0, 64 - n)).astype(np.int64)).unsqueeze(0),
            "opt_card_id": torch.from_numpy(
                np.pad(opt_card_id, (0, 64 - n)).astype(np.int64)).unsqueeze(0),
            "opt_scalar": torch.from_numpy(opt_scalar),
            "opt_mask": torch.from_numpy(opt_mask),
        }

    def test_apply_mask_edits_opt_mask_in_place(self):
        from ptcg_il.deck_guard import DeckGuard, GuardConfig

        g = DeckGuard(GuardConfig())
        batch = self._batch([7, 14], [1152, 0], [0, 0])
        g.apply_mask(batch, sel_type=0, deck=1, hand=9, prizes=3)
        assert batch["opt_mask"][0, 0].item() is False
        assert batch["opt_mask"][0, 1].item() is True
        assert g.stats.a_masked == 1

    def test_apply_mask_ignores_non_main_selects(self):
        from ptcg_il.deck_guard import DeckGuard, GuardConfig

        g = DeckGuard(GuardConfig())
        batch = self._batch([3], [1152], [0])
        g.apply_mask(batch, sel_type=1, deck=1, hand=9, prizes=3)
        assert batch["opt_mask"][0, 0].item() is True

    def test_pick_returns_an_int_and_logs_changes(self):
        from ptcg_il.deck_guard import DeckGuard, GuardConfig

        torch = pytest.importorskip("torch")
        g = DeckGuard(GuardConfig(deck_low=10, margin=0.5))
        batch = self._batch([7, 7, 14], [1152, 1227, 0], [0, 0, 0])
        logits = torch.zeros(1, 64)
        logits[0, 0], logits[0, 1], logits[0, 2] = 3.0, 2.8, 1.0
        p = g.pick(logits, batch, sel_type=0, max_count=1,
                   deck=5, hand=9, prizes=3)
        assert p == 1
        assert g.stats.b_gated == 1
        assert g.stats.b_changed == 1
        assert g.stats.recent[-1]["new"] == 1
        assert g.stats.recent[-1]["old"] == 0

    def test_pick_ungated_counts_nothing(self):
        from ptcg_il.deck_guard import DeckGuard, GuardConfig

        torch = pytest.importorskip("torch")
        g = DeckGuard(GuardConfig())
        batch = self._batch([7, 7], [1152, 1227], [0, 0])
        logits = torch.zeros(1, 64)
        logits[0, 0], logits[0, 1] = 3.0, 2.9
        p = g.pick(logits, batch, sel_type=0, max_count=1,
                   deck=50, hand=9, prizes=3)
        assert p == 0
        assert g.stats.b_gated == 0
        assert g.stats.b_changed == 0


# ------------------------------------------------------- option scope rules

class TestOptionScope:
    """A card's burn applies only where its effect actually resolves.

    Caught by the episode replay: Drakloak's −1 belongs to its Recon
    Directive *ability* (SKILL) — playing Drakloak from hand to the bench
    touches no deck cards, and charging −1 there masks real plays at low deck.
    The same applies to the fallback columns: their draw text describes a
    Pokémon's ability, so an unlisted Pokémon played from hand burns nothing,
    while an unlisted trainer's text *is* its play effect.
    """

    @staticmethod
    def deltas(opt_type, opt_card_id, ecf=None, deck=10, hand=5):
        from ptcg_il.deck_guard import option_deck_deltas

        return option_deck_deltas(
            opt_type=np.asarray(opt_type, dtype=np.int64),
            opt_card_id=np.asarray(opt_card_id, dtype=np.int64),
            opt_atk_risk=np.zeros(len(opt_type), dtype=np.float32),
            deck=deck, hand=hand, prizes=3, engine_card_features=ecf,
        )

    def test_playing_an_ability_pokemon_to_bench_burns_nothing(self):
        d = self.deltas([7, 15, 7, 15], [120, 120, 140, 140])
        assert d.tolist() == [0, -1, 0, -3]

    def test_play_fallback_only_applies_to_trainer_cards(self):
        from ptcg_il.deck_guard import CARD_DRAW_FIXED_COL, DRAW_N

        poke_row = np.zeros(223, dtype=np.float32)
        poke_row[2] = 1.0                       # cardType POKEMON (0)
        poke_row[CARD_DRAW_FIXED_COL] = 3.0 / DRAW_N
        item_row = np.zeros(223, dtype=np.float32)
        item_row[3] = 1.0                       # cardType ITEM (1)
        item_row[CARD_DRAW_FIXED_COL] = 3.0 / DRAW_N
        ecf = {991: poke_row, 992: item_row}

        d = self.deltas([7, 7, 15], [991, 992, 991], ecf=ecf)
        assert d.tolist() == [0, -3, -3]


class TestZeroDeckSemantics:
    """The mask charges *burns*, not states: at deck 0 a delta-0 option is
    not what kills you (you are already dead), and masking everything leaves
    argmax noise.  Only options with a genuinely negative delta may be killed.
    """

    def test_zero_delta_options_survive_at_zero_deck(self):
        from ptcg_il.deck_guard import deckout_kill_mask

        k = deckout_kill_mask(
            opt_type=np.asarray([7, 7, 14], dtype=np.int64),
            opt_card_id=np.asarray([721, 1152, 0], dtype=np.int64),
            opt_ko=np.zeros(3, dtype=np.float32),
            opt_atk_risk=np.zeros(3, dtype=np.float32),
            opt_mask=np.ones(3, dtype=bool),
            deck=0, hand=9, prizes=3, engine_card_features=None)
        # unlisted PLAY (delta 0) and END survive; Poké Pad (−1) still dies.
        assert k.tolist() == [False, True, False]
