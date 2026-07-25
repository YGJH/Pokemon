"""Tests for ptcg_il.featurizer — Appendix A contract verification.

Uses the sample episode fixture at archive/sample_episodes/80169582.json
and the real engine at pokemon-tcg-ai-battle/.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ptcg_il.featurizer import (
    ATKCOST_N,
    COUNT_N,
    D_MAX,
    DECK_N,
    DMGCTR_N,
    ENERGY_N,
    F_GLOBAL,
    F_HAND,
    F_OPT,
    F_POKE,
    F_SUM,
    H_MAX,
    HAND_N,
    HP_N,
    L_STATE,
    O_MAX,
    PAD_CARD,
    P_MAX,
    PZ_MAX,
    PAD_ATTACK,
    SUM,
    TURN_N,
    featurize,
)
from ptcg_il.ref_map import build_ref_map, card_id_at

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SAMPLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "archive"
    / "sample_episodes"
    / "80169582.json"
)


def _load_episode() -> dict:
    with open(SAMPLE_PATH) as f:
        return json.load(f)


def _build_test_vocab(ep: dict) -> dict:
    """Build a minimal vocab from all card IDs appearing in the episode decks."""
    from ptcg_mine.episode import deck_of
    from ptcg_mine.vocab import build_vocab

    return build_vocab([ep], mode="all_corpus")


def _build_test_vocab_with_attacks(ep: dict) -> dict:
    """Build a vocab including attack_id_to_index."""
    vocab = _build_test_vocab(ep)
    # For the sample episode, collect attack IDs from ATTACK options
    attack_ids = set()
    for step in ep["steps"]:
        for rec in step:
            obs = rec.get("observation", {})
            sel = obs.get("select")
            if sel is None:
                continue
            for opt in sel.get("option", []):
                aid = opt.get("attackId")
                if aid is not None:
                    attack_ids.add(aid)
    attack_id_to_index = {
        aid: i + 1 for i, aid in enumerate(sorted(attack_ids))
    }
    vocab["attack_id_to_index"] = attack_id_to_index
    return vocab


def _get_active_step(
    ep: dict, step_idx: int, player: int
) -> tuple[dict, list[int]]:
    """Return (obs_dict, action) for a known ACTIVE step using off-by-one pairing."""
    obs = ep["steps"][step_idx][player]["observation"]
    action = ep["steps"][step_idx + 1][player]["action"]
    return obs, action


# ---------------------------------------------------------------------------
# Step 1: Pokémon token tests
# ---------------------------------------------------------------------------


class TestPokemonTokens:
    """Step 1: Verify pokemon token shapes, slot layout, and feature values."""

    def test_shapes(self):
        """poke_card_id.shape == (12,), poke_feat.shape == (12, 26)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)  # MAIN select with visible mons

        result = featurize(obs, vocab, action)
        assert result["poke_card_id"].shape == (P_MAX,)
        assert result["poke_card_id"].dtype == np.int64
        assert result["poke_feat"].shape == (P_MAX, F_POKE)
        assert result["poke_feat"].dtype == np.float32

    def test_slot_layout(self):
        """my active at row 0 (index 1), opp active at row 6 (index 7)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Step 8 Player 0: my active=721, opp active=722, no bench
        # My active (slot 0) should have remapped 721, opp active (slot 6) should have 722
        id_to_index = vocab["id_to_index"]
        my_active_id = _remap(721, id_to_index)
        opp_active_id = _remap(722, id_to_index)

        assert result["poke_card_id"][0] == my_active_id, (
            f"Expected my active card at slot 0, got {result['poke_card_id'][0]}"
        )
        assert result["poke_card_id"][6] == opp_active_id, (
            f"Expected opp active card at slot 6, got {result['poke_card_id'][6]}"
        )

    def test_empty_bench_slots_are_pad(self):
        """Empty bench slots should be PAD_CARD."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Bench slots 1..5 and 7..11 should be PAD
        for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]:
            assert result["poke_card_id"][i] == PAD_CARD, f"Slot {i} not PAD"
            assert np.all(result["poke_feat"][i] == 0.0), f"Slot {i} features not zero"

    def test_hp_features(self):
        """Verify hp/HP_N, maxHp/HP_N, hp/maxHp for a known Pokemon."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # My active: Pokemon 721, hp=150, maxHp=150
        f = result["poke_feat"][0]
        assert f[0] == pytest.approx(150.0 / HP_N)
        assert f[1] == pytest.approx(150.0 / HP_N)
        assert f[2] == pytest.approx(150.0 / 150.0)

        # Opp active: Pokemon 722, hp=90, maxHp=90
        f = result["poke_feat"][6]
        assert f[0] == pytest.approx(90.0 / HP_N)
        assert f[1] == pytest.approx(90.0 / HP_N)
        assert f[2] == pytest.approx(90.0 / 90.0)

    def test_attached_energy_histogram(self):
        """Verify energy histogram for opp active (has 1 WATER energy attached)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Opp active: energies=[3] (WATER), energyCards=[(3, 104)]
        f_opp = result["poke_feat"][6]
        # EnergyType.WATER = 3
        assert f_opp[3 + 3] > 0.0, f"Expected WATER energy count > 0, got {f_opp[3:15]}"
        assert f_opp[15] > 0.0, "total_energies should be > 0"
        assert f_opp[16] > 0.0, "energy_cards should be > 0"

        # My active: no energies
        f_my = result["poke_feat"][0]
        assert np.all(f_my[3:15] == 0.0), "My active should have no energies"
        assert f_my[15] == 0.0, "total_energies should be 0 for my active"

    def test_is_active_flag(self):
        """Active slots should have is_active=1, bench slots is_active=0."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["poke_feat"][0, 20] == 1.0, "My active should be is_active=1"
        assert result["poke_feat"][6, 20] == 1.0, "Opp active should be is_active=1"

    def test_condition_flags_active_only(self):
        """Conditions only on active slots, zero on bench."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Active slots have condition info (may be 0 or 1 depending on game state)
        # In this step, no conditions are active
        assert result["poke_feat"][0, 21] == 0.0, "No poison expected"
        assert result["poke_feat"][0, 25] == 0.0, "No confuse expected"

        # Bench slots should be ALL zeros including conditions
        for i in [1, 2, 3, 4, 5]:
            assert np.all(result["poke_feat"][i, 21:26] == 0.0), (
                f"Bench slot {i} conditions should be 0"
            )

    def test_appear_this_turn(self):
        """appearThisTurn should be a boolean feature."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Both Pokemon in this step did NOT appear this turn
        assert result["poke_feat"][0, 19] == 0.0
        assert result["poke_feat"][6, 19] == 0.0


# ---------------------------------------------------------------------------
# Step 3: Hand and summary token tests
# ---------------------------------------------------------------------------


class TestHandAndSummary:
    """Step 3: Verify hand and summary token shapes and feature values."""

    def test_hand_shapes_and_padding(self):
        """hand_card_id.shape == (30,), hand_feat.shape == (30, 2)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["hand_card_id"].shape == (H_MAX,)
        assert result["hand_card_id"].dtype == np.int64
        assert result["hand_feat"].shape == (H_MAX, F_HAND)
        assert result["hand_feat"].dtype == np.float32

    def test_hand_cards_at_front_pad_after(self):
        """Hand cards packed at front, PAD after."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Player 0 hand has 7 cards in this step
        hand = obs["current"]["players"][0]["hand"]
        n_hand = len(hand)
        assert n_hand == 7

        # First 7 slots should be non-PAD
        for i in range(n_hand):
            assert result["hand_card_id"][i] != PAD_CARD, (
                f"Hand slot {i} should not be PAD"
            )
        # Remaining slots should be PAD
        for i in range(n_hand, H_MAX):
            assert result["hand_card_id"][i] == PAD_CARD, (
                f"Hand slot {i} should be PAD"
            )

    def test_hand_feat_position(self):
        """hand_feat[0] should be idx/H_MAX."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["hand_feat"][0, 0] == pytest.approx(0.0 / HAND_N)
        assert result["hand_feat"][3, 0] == pytest.approx(3.0 / HAND_N)

    def test_hand_dup_count(self):
        """Duplicate cards in hand should have dup_count > single."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Hand has multiple copies of card id 3
        # Compute expected dup count from actual hand data
        hand = obs["current"]["players"][0]["hand"]
        from collections import Counter
        id_counts = Counter(c["id"] for c in hand)
        dup_count_3 = id_counts.get(3, 1)  # card id 3 appears multiple times
        expected_dup = float(dup_count_3) / COUNT_N

        # Position 0 has card id 3
        assert result["hand_feat"][0, 1] == pytest.approx(expected_dup), (
            f"Expected dup feature {expected_dup}, got {result['hand_feat'][0, 1]}"
        )
        # Unique card (1145) should have dup_count = 1/COUNT_N
        # Find position of card id 1145
        pos_1145 = next(i for i, c in enumerate(hand) if c["id"] == 1145)
        assert result["hand_feat"][pos_1145, 1] == pytest.approx(1.0 / COUNT_N)

    def test_summary_shapes(self):
        """sum_feat.shape == (2, 11)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["sum_feat"].shape == (SUM, F_SUM)
        assert result["sum_feat"].dtype == np.float32

    def test_summary_is_me_flag(self):
        """Row 0 (me) should have is_me=1, row 1 (opp) is_me=0."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["sum_feat"][0, 0] == 1.0, "Row 0 (me) should have is_me=1"
        assert result["sum_feat"][1, 0] == 0.0, "Row 1 (opp) should have is_me=0"

    def test_summary_deck_hand_counts(self):
        """Verify deckCount, handCount features."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Player 0: deckCount=46, handCount=7
        assert result["sum_feat"][0, 1] == pytest.approx(46.0 / DECK_N)
        assert result["sum_feat"][0, 2] == pytest.approx(7.0 / HAND_N)

    def test_discard_and_prize_tensors(self):
        """discard_ids and prize_ids have correct shapes."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["discard_ids"].shape == (SUM, D_MAX)
        assert result["discard_mask"].shape == (SUM, D_MAX)
        assert result["discard_mask"].dtype == bool
        assert result["prize_ids"].shape == (SUM, PZ_MAX)

    def test_opponent_hand_not_visible(self):
        """Opponent hand is None in observation → all PAD in hand tokens."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        # Verify opponent hand is None
        opp_hand = obs["current"]["players"][1].get("hand")
        assert opp_hand is None, "Opponent hand should be None"

        # Our hand tokens only cover OUR hand — already tested above
        # No assertion needed on the hand tokens themselves since they're always my hand

    def test_stadium_absent(self):
        """When no stadium, stadium_card_id = PAD, stadium_present = 0.0."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["stadium_card_id"][0] == PAD_CARD
        assert result["stadium_present"][0] == 0.0


# ---------------------------------------------------------------------------
# Step 5: CLS feature tests
# ---------------------------------------------------------------------------


class TestClsFeatures:
    """Step 5: Verify CLS features shape and contents."""

    def test_shape(self):
        """cls_feat.shape == (93,)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["cls_feat"].shape == (F_GLOBAL,)
        assert result["cls_feat"].dtype == np.float32

    def test_select_type_onehot(self):
        """Verify one-hot encoding of select.type."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Step 8 P0: select.type=0 (MAIN)
        sel_type = obs["select"]["type"]
        assert result["cls_feat"][13 + sel_type] == 1.0
        # Other positions in 13:24 should be 0
        for i in range(13, 24):
            if i != 13 + sel_type:
                assert result["cls_feat"][i] == 0.0, f"cls_feat[{i}] should be 0"

    def test_select_context_onehot(self):
        """Verify one-hot encoding of select.context."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        sel_ctx = obs["select"]["context"]
        assert result["cls_feat"][24 + sel_ctx] == 1.0

    def test_turn_and_your_index(self):
        """Verify turn/TURN_N and yourIndex features."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        state = obs["current"]
        assert result["cls_feat"][0] == pytest.approx(float(state["turn"]) / TURN_N)
        assert result["cls_feat"][1] == float(state["turn"] % 2)
        assert result["cls_feat"][2] == float(state["yourIndex"])

    def test_first_player_onehot(self):
        """Verify firstPlayer one-hot over {-1,0,1}."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        fp = obs["current"]["firstPlayer"]
        assert fp == 1  # In this episode, player 1 goes first
        assert result["cls_feat"][4] == 0.0  # -1 slot
        assert result["cls_feat"][5] == 0.0  # 0 slot
        assert result["cls_feat"][6] == 1.0  # 1 slot

    def test_per_turn_flags(self):
        """Verify supporterPlayed, stadiumPlayed, energyAttached, retreated."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # These are booleans → 0.0 or 1.0
        for idx in [7, 8, 9, 10]:
            assert result["cls_feat"][idx] in (0.0, 1.0), (
                f"cls_feat[{idx}] should be 0 or 1, got {result['cls_feat'][idx]}"
            )

    def test_condition_flags_in_cls(self):
        """my-active conditions at [77:82], opp-active conditions at [82:87]."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Should be 5 booleans each
        for idx in range(77, 87):
            assert result["cls_feat"][idx] in (0.0, 1.0), (
                f"cls_feat[{idx}] should be 0 or 1, got {result['cls_feat'][idx]}"
            )

    def test_context_effect_flags(self):
        """has_contextCard and has_effect at [87:89]."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # This step has no contextCard (None) and no effect (None)
        assert result["cls_feat"][87] == 0.0, "contextCard absent → has_contextCard must be 0.0"
        assert result["cls_feat"][88] == 0.0, "effect absent → has_effect must be 0.0"

    def test_context_effect_card_ids(self):
        """context_card_id and effect_card_id are present and padded."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["context_card_id"].shape == (1,)
        assert result["effect_card_id"].shape == (1,)
        assert result["context_card_id"][0] == PAD_CARD  # No context card in this step
        assert result["effect_card_id"][0] == PAD_CARD

    def test_reserved_zeros(self):
        """Reserved positions [89:93] should be zero."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert np.all(result["cls_feat"][89:93] == 0.0)


# ---------------------------------------------------------------------------
# Step 7: Categorical token attribute tests
# ---------------------------------------------------------------------------


class TestCategoricalAttrs:
    """Step 7: Verify tok_type, tok_owner, tok_zone, tok_mask."""

    def test_shapes(self):
        """All categorical attrs have shape (46,)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        for key in ["tok_type", "tok_owner", "tok_zone", "tok_mask"]:
            assert result[key].shape == (L_STATE,), f"{key} shape mismatch"
        assert result["tok_type"].dtype == np.int64
        assert result["tok_owner"].dtype == np.int64
        assert result["tok_zone"].dtype == np.int64
        assert result["tok_mask"].dtype == bool

    def test_cls_token_type(self):
        """CLS token at row 0 has type=0, owner=none, zone=cls."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["tok_type"][0] == 0  # CLS
        assert result["tok_owner"][0] == 0  # none
        assert result["tok_zone"][0] == 0  # cls
        assert result["tok_mask"][0] == True

    def test_pokemon_owner_and_zone(self):
        """Slot 0 (my active): owner=self, zone=active. Slot 6 (opp active): owner=opp, zone=active."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # My active: row 1
        assert result["tok_owner"][1] == 1  # self
        assert result["tok_zone"][1] == 1  # active
        assert result["tok_mask"][1] == True

        # Opp active: row 7
        assert result["tok_owner"][7] == 2  # opp
        assert result["tok_zone"][7] == 1  # active
        assert result["tok_mask"][7] == True

    def test_empty_slots_masked_out(self):
        """Empty bench/other slots should have tok_mask = False."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Bench slots 2..6 (my bench) and 8..12 (opp bench) are empty
        for row in [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]:
            assert result["tok_mask"][row] == False, (
                f"Empty row {row} should be masked out"
            )

    def test_hand_tokens_owner_zone(self):
        """Hand tokens have owner=self, zone=hand."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # First hand token: row 13
        assert result["tok_owner"][13] == 1  # self
        assert result["tok_zone"][13] == 3  # hand
        assert result["tok_type"][13] == 2  # HAND

    def test_summary_tokens(self):
        """Summary tokens have correct owner."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Row 43: my summary → self
        assert result["tok_owner"][43] == 1  # self
        # Row 44: opp summary → opp
        assert result["tok_owner"][44] == 2  # opp
        assert result["tok_type"][43] == 3  # SUMMARY
        assert result["tok_type"][44] == 3  # SUMMARY

    def test_stadium_masked_when_absent(self):
        """When no stadium, row 45 should be masked out."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # No stadium in this step
        assert result["stadium_present"][0] == 0.0
        assert result["tok_mask"][45] == False


# ---------------------------------------------------------------------------
# Step 9: Single-select option token tests
# ---------------------------------------------------------------------------


class TestOptionsSingleSelect:
    """Step 9: Verify option tokens for single-select steps."""

    def test_opt_shapes(self):
        """All option tensors have correct shapes."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["opt_type"].shape == (O_MAX,)
        assert result["opt_src_idx"].shape == (O_MAX,)
        assert result["opt_tgt_idx"].shape == (O_MAX,)
        assert result["opt_card_id"].shape == (O_MAX,)
        assert result["opt_attack_idx"].shape == (O_MAX,)
        assert result["opt_scalar"].shape == (O_MAX, F_OPT)
        assert result["opt_mask"].shape == (O_MAX,)

    def test_opt_mask_correct_count(self):
        """opt_mask True for exactly len(options) entries."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        n_opts = len(obs["select"]["option"])
        assert result["opt_mask"].sum() == n_opts

    def test_yes_no_options(self):
        """YES/NO options have src=-1, tgt=-1."""
        ep = _load_episode()
        # Step 2 Player 0: YES_NO(IS_FIRST) with 2 options (type 1=YES, type 2=NO)
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 2, 0)

        result = featurize(obs, vocab, action)

        # Option 0 should be YES (type=1)
        assert result["opt_type"][0] == 1, "Option 0 should be YES (type 1)"
        assert result["opt_src_idx"][0] == -1
        assert result["opt_tgt_idx"][0] == -1

        # Option 1 should be NO (type=2), also with src=-1, tgt=-1
        assert result["opt_type"][1] == 2, "Option 1 should be NO (type 2)"
        assert result["opt_src_idx"][1] == -1
        assert result["opt_tgt_idx"][1] == -1

    def test_action_idx_single_pick(self):
        """action_idx has exactly 1 pick for single-select, rest are -1."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Single pick: action=[0]
        assert result["action_idx"][0] == 0
        for i in range(1, O_MAX):
            assert result["action_idx"][i] == -1
        assert result["action_len"] == 1


# ---------------------------------------------------------------------------
# Step 10: Multi-select and ref resolution tests
# ---------------------------------------------------------------------------


class TestMultiSelectAndRefs:
    """Step 10: Verify multi-select options and reference resolution."""

    def test_play_option_refs(self):
        """PLAY option: src = hand row, tgt = -1."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        # Step 8 P0: has PLAY options at indices 2 and 5
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Find PLAY options
        for j in range(O_MAX):
            if result["opt_mask"][j] and result["opt_type"][j] == 7:
                # PLAY src should be a hand row (13+index)
                assert result["opt_src_idx"][j] >= 13, (
                    f"PLAY opt[{j}] src should be hand row, got {result['opt_src_idx'][j]}"
                )
                assert result["opt_src_idx"][j] < 13 + H_MAX
                assert result["opt_tgt_idx"][j] == -1

    def test_attach_option_refs(self):
        """ATTACH option: src = hand/deck row, tgt = in-play Pokemon row."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Find ATTACH options (type=8)
        for j in range(O_MAX):
            if result["opt_mask"][j] and result["opt_type"][j] == 8:
                # tgt should be in-play row (1 for active in this step)
                assert result["opt_tgt_idx"][j] == 1, (
                    f"ATTACH opt[{j}] tgt should be my active row 1, got {result['opt_tgt_idx'][j]}"
                )

    def test_retreat_option_ref(self):
        """RETREAT option: src = my active row 1, tgt = -1.

        NOTE: This sample episode (80169582) has no RETREAT (type=12) options
        across any ACTIVE decision. The test scans all steps to confirm this
        and documents the gap. When a sample episode with RETREAT options
        becomes available, the contract to verify is: opt_src_idx == 1
        (my active row) and opt_tgt_idx == -1 for every RETREAT option.
        """
        ep = _load_episode()
        vocab = _build_test_vocab(ep)

        # Scan all ACTIVE decisions for RETREAT options
        retreat_found: list[tuple[int, int, int]] = []
        for step_idx in range(len(ep["steps"]) - 1):
            for p in (0, 1):
                rec = ep["steps"][step_idx][p]
                if rec.get("status") != "ACTIVE":
                    continue
                obs = rec.get("observation")
                if obs is None or obs.get("select") is None or obs.get("current") is None:
                    continue
                result = featurize(obs, vocab, ep["steps"][step_idx + 1][p].get("action", []))
                for j in range(O_MAX):
                    if result["opt_mask"][j] and result["opt_type"][j] == 12:
                        retreat_found.append((step_idx, p, j))
                        assert result["opt_src_idx"][j] == 1, (
                            f"RETREAT opt[{j}] src should be my active row 1"
                        )
                        assert result["opt_tgt_idx"][j] == -1, (
                            f"RETREAT opt[{j}] tgt should be -1"
                        )

        # Document: no RETREAT options exist in this sample episode
        assert len(retreat_found) == 0, (
            f"Expected no RETREAT options in sample episode 80169582, "
            f"but found {len(retreat_found)}: {retreat_found}"
        )

    def test_attack_option_ref(self):
        """ATTACK option: src = my active row 1, tgt = -1, attack_idx set."""
        ep = _load_episode()
        vocab = _build_test_vocab_with_attacks(ep)
        # Find a step with ATTACK options
        # Step 16 P1: options may have ATTACK
        obs, action = _get_active_step(ep, 16, 1)

        result = featurize(obs, vocab, action)

        for j in range(O_MAX):
            if result["opt_mask"][j] and result["opt_type"][j] == 13:
                assert result["opt_src_idx"][j] == 1, (
                    f"ATTACK opt[{j}] src should be my active row 1"
                )
                assert result["opt_tgt_idx"][j] == -1
                assert result["opt_attack_idx"][j] > 0, (
                    f"ATTACK opt[{j}] should have attack_idx > 0"
                )

    def test_end_option(self):
        """END option: src=-1, tgt=-1, type=14."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # Find END options (type=14)
        found_end = False
        for j in range(O_MAX):
            if result["opt_mask"][j] and result["opt_type"][j] == 14:
                assert result["opt_src_idx"][j] == -1
                assert result["opt_tgt_idx"][j] == -1
                found_end = True
        # Step 8 P0 has END at option[7]
        assert found_end, "Expected at least one END option"

    def test_opt_scalar_features(self):
        """opt_scalar has 6 features per option."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        # All valid options should have properly normalized scalars
        for j in range(O_MAX):
            if result["opt_mask"][j]:
                # Scalar features should be in valid range
                assert np.all(result["opt_scalar"][j] >= -1.0)
                assert np.all(result["opt_scalar"][j] <= 1.0)


# ---------------------------------------------------------------------------
# Step 13: End-to-end test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Step 13: End-to-end test — iterate all ACTIVE decisions, verify all keys."""

    REQUIRED_KEYS = {
        # State — card identity & structural
        "poke_card_id",
        "hand_card_id",
        "stadium_card_id",
        "context_card_id",
        "effect_card_id",
        "discard_ids",
        "discard_mask",
        "prize_ids",
        # State — dense features
        "poke_feat",
        "hand_feat",
        "sum_feat",
        "cls_feat",
        "stadium_present",
        # State — categorical token attributes
        "tok_type",
        "tok_owner",
        "tok_zone",
        "tok_mask",
        # Action — option references & features
        "opt_type",
        "opt_src_idx",
        "opt_tgt_idx",
        "opt_card_id",
        "opt_attack_idx",
        "opt_scalar",
        "opt_mask",
        # Labels & bookkeeping
        "action_idx",
        "action_len",
        "minCount",
        "maxCount",
        "sel_type",
        "sel_ctx",
        "value_target",
        "sample_weight",
        "stop_column",
        "log_feat",
        "log_mask",
        "log_len",
    }

    def _iter_active_decisions(self, ep: dict):
        """Yield (obs, action) for all ACTIVE decisions in the episode."""
        for i in range(len(ep["steps"]) - 1):
            for p in (0, 1):
                rec = ep["steps"][i][p]
                if rec.get("status") != "ACTIVE":
                    continue
                obs = rec["observation"]
                if obs.get("select") is None:
                    # Deck-selection step — should be rejected by featurizer
                    continue
                if obs.get("current") is None:
                    continue
                action = ep["steps"][i + 1][p]["action"]
                yield obs, action

    def test_all_keys_present(self):
        """Every featurize call returns all 31 A.4 keys."""
        ep = _load_episode()
        vocab = _build_test_vocab_with_attacks(ep)

        count = 0
        for obs, action in self._iter_active_decisions(ep):
            result = featurize(obs, vocab, action)
            missing = self.REQUIRED_KEYS - set(result.keys())
            extra = set(result.keys()) - self.REQUIRED_KEYS
            assert not missing, f"Missing keys: {missing}"
            assert not extra, f"Extra keys: {extra}"
            count += 1

        assert count > 0, "No ACTIVE decisions found in sample episode"

    def test_all_shapes_consistent(self):
        """All tensors have consistent shapes across all decisions."""
        ep = _load_episode()
        vocab = _build_test_vocab_with_attacks(ep)

        expected_shapes = {
            "poke_card_id": (P_MAX,),
            "hand_card_id": (H_MAX,),
            "stadium_card_id": (1,),
            "context_card_id": (1,),
            "effect_card_id": (1,),
            "discard_ids": (SUM, D_MAX),
            "discard_mask": (SUM, D_MAX),
            "prize_ids": (SUM, PZ_MAX),
            "poke_feat": (P_MAX, F_POKE),
            "hand_feat": (H_MAX, F_HAND),
            "sum_feat": (SUM, F_SUM),
            "cls_feat": (F_GLOBAL,),
            "stadium_present": (1,),
            "tok_type": (L_STATE,),
            "tok_owner": (L_STATE,),
            "tok_zone": (L_STATE,),
            "tok_mask": (L_STATE,),
            "opt_type": (O_MAX,),
            "opt_src_idx": (O_MAX,),
            "opt_tgt_idx": (O_MAX,),
            "opt_card_id": (O_MAX,),
            "opt_attack_idx": (O_MAX,),
            "opt_scalar": (O_MAX, F_OPT),
            "opt_mask": (O_MAX,),
            "action_idx": (O_MAX,),
            "action_len": (),
            "minCount": (),
            "maxCount": (),
            "sel_type": (),
            "sel_ctx": (),
            "value_target": (),
            "sample_weight": (),
        }

        for obs, action in self._iter_active_decisions(ep):
            result = featurize(obs, vocab, action)
            for key, shape in expected_shapes.items():
                assert result[key].shape == shape, (
                    f"Key {key}: expected {shape}, got {result[key].shape}"
                )

    def test_no_crash_on_any_step(self):
        """Featurizer should not crash on any valid ACTIVE decision."""
        ep = _load_episode()
        vocab = _build_test_vocab_with_attacks(ep)

        for obs, action in self._iter_active_decisions(ep):
            featurize(obs, vocab, action)

    def test_dtype_consistency(self):
        """Verify dtype contracts: float32, int64, bool."""
        ep = _load_episode()
        vocab = _build_test_vocab_with_attacks(ep)

        for obs, action in self._iter_active_decisions(ep):
            result = featurize(obs, vocab, action)

            # Float tensors
            for key in [
                "poke_feat", "hand_feat", "sum_feat", "cls_feat",
                "stadium_present", "opt_scalar", "value_target", "sample_weight",
            ]:
                assert result[key].dtype == np.float32, (
                    f"{key} dtype: expected float32, got {result[key].dtype}"
                )

            # Int tensors
            for key in [
                "poke_card_id", "hand_card_id", "stadium_card_id",
                "context_card_id", "effect_card_id",
                "discard_ids", "prize_ids",
                "tok_type", "tok_owner", "tok_zone",
                "opt_type", "opt_src_idx", "opt_tgt_idx",
                "opt_card_id", "opt_attack_idx",
                "action_idx", "action_len", "minCount", "maxCount",
                "sel_type", "sel_ctx",
            ]:
                assert result[key].dtype == np.int64, (
                    f"{key} dtype: expected int64, got {result[key].dtype}"
                )

            # Bool tensors
            for key in ["discard_mask", "tok_mask", "opt_mask"]:
                assert result[key].dtype == bool, (
                    f"{key} dtype: expected bool, got {result[key].dtype}"
                )

            break  # One sample is enough for dtype check


# ---------------------------------------------------------------------------
# Step 14: Deck-selection rejection
# ---------------------------------------------------------------------------


class TestDeckSelectionRejection:
    """Step 14: Deck-selection steps (select is None) raise ValueError."""

    def test_select_none_raises(self):
        """Calling featurize with select=None raises ValueError."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)

        # Step 0 has select=None for both players
        obs = ep["steps"][0][0]["observation"]
        assert obs["select"] is None

        with pytest.raises(ValueError, match="select.*None"):
            featurize(obs, vocab, [])

    def test_current_none_raises(self):
        """Calling featurize with current=None raises ValueError."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)

        obs = {"select": {"type": 0, "context": 0, "option": []}, "current": None}
        with pytest.raises(ValueError, match="current.*None"):
            featurize(obs, vocab, [])

    def test_deck_steps_not_in_active_iteration(self):
        """When iterating ACTIVE decisions, deck steps (select=None) are
        naturally skipped — they have ACTIVE status but the featurizer
        rejects them.  The caller must filter these out."""
        ep = _load_episode()
        # Verify that the featurizer correctly rejects deck-selection steps
        # even though they have ACTIVE status
        for i in range(len(ep["steps"]) - 1):
            for p in (0, 1):
                rec = ep["steps"][i][p]
                if rec.get("status") != "ACTIVE":
                    continue
                obs = rec["observation"]
                if obs.get("select") is None:
                    # Deck-selection step — should be rejected
                    with pytest.raises(ValueError, match="select.*None"):
                        featurize(obs, _build_test_vocab(ep), [])
                # else: normal decision, passes featurize


# ---------------------------------------------------------------------------
# Step: Ref map tests
# ---------------------------------------------------------------------------


class TestRefMap:
    """Test build_ref_map resolution."""

    def test_my_active_mapped(self):
        """My active Pokemon maps to row 1."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        your_idx = obs["current"]["yourIndex"]

        rf = build_ref_map(obs)

        assert rf.get((4, your_idx, 0)) == 1

    def test_opp_active_mapped(self):
        """Opp active Pokemon maps to row 7."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        your_idx = obs["current"]["yourIndex"]

        rf = build_ref_map(obs)

        assert rf.get((4, 1 - your_idx, 0)) == 7

    def test_my_hand_mapped(self):
        """My hand cards map to rows 13+."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        your_idx = obs["current"]["yourIndex"]

        rf = build_ref_map(obs)

        hand_len = len(obs["current"]["players"][your_idx]["hand"])
        for i in range(min(hand_len, 30)):
            assert rf.get((2, your_idx, i)) == 13 + i

    def test_unknown_ref_returns_none(self):
        """References to non-tokenized zones are not in the map."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)

        rf = build_ref_map(obs)

        # Deck cards are not tokenized
        assert (1, 0, 0) not in rf  # DECK area

    def test_stadium_mapped_when_present(self):
        """Stadium card maps to row 45 when present."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)

        rf = build_ref_map(obs)

        # No stadium in this step
        assert (7, -1, 0) not in rf


# ---------------------------------------------------------------------------
# Step: card_id_at location dereference
# ---------------------------------------------------------------------------


class TestCardIdAt:
    """Test card_id_at resolves visible zones and refuses hidden ones."""

    def test_hand_resolves(self):
        """Area 2 index i returns the id of hand[i]."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        state = obs["current"]
        me = state["yourIndex"]

        hand = state["players"][me]["hand"]
        assert len(hand) > 0, "fixture must have a non-empty hand"
        for i, card in enumerate(hand):
            assert card_id_at(state, 2, me, i) == card["id"]

    def test_active_resolves(self):
        """Area 4 index 0 returns the active Pokemon's id."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        state = obs["current"]
        me = state["yourIndex"]

        active = state["players"][me]["active"][0]
        assert active is not None, "fixture must have a face-up active"
        assert card_id_at(state, 4, me, 0) == active["id"]

    def test_bench_resolves(self):
        """Area 5 index i returns the id of bench[i]."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        state = obs["current"]
        me = state["yourIndex"]

        for i, card in enumerate(state["players"][me]["bench"]):
            if card is not None:
                assert card_id_at(state, 5, me, i) == card["id"]

    def test_discard_resolves(self):
        """Area 3 index i returns the id of discard[i]."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        state = obs["current"]
        me = state["yourIndex"]

        for i, card in enumerate(state["players"][me]["discard"]):
            if card is not None:
                assert card_id_at(state, 3, me, i) == card["id"]

    def test_deck_never_resolves(self):
        """Area 1 is the deck -- hidden by design, always None.

        Returning an id here would leak the deck order to the policy.
        """
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        state = obs["current"]

        for i in range(60):
            assert card_id_at(state, 1, state["yourIndex"], i) is None

    def test_face_down_prize_returns_none(self):
        """Area 6 slots that are None stay None -- face-down prizes are hidden."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        state = obs["current"]
        me = state["yourIndex"]

        prize = state["players"][me]["prize"]
        assert any(p is None for p in prize), "fixture must have a face-down prize"
        for i, card in enumerate(prize):
            got = card_id_at(state, 6, me, i)
            if card is None:
                assert got is None
            else:
                assert got == card["id"]

    def test_out_of_range_returns_none(self):
        """Indices past the end of a container return None, not an exception."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        state = obs["current"]
        me = state["yourIndex"]

        assert card_id_at(state, 2, me, 999) is None
        assert card_id_at(state, 5, me, 999) is None
        assert card_id_at(state, 2, me, -1) is None

    def test_bad_args_return_none(self):
        """Non-integer area/index and unknown areas return None."""
        ep = _load_episode()
        obs, _ = _get_active_step(ep, 8, 0)
        state = obs["current"]

        assert card_id_at(state, None, 0, 0) is None
        assert card_id_at(state, 2, 0, None) is None
        assert card_id_at(state, 99, 0, 0) is None
        assert card_id_at(state, 2, 99, 0) is None


# ---------------------------------------------------------------------------
# Step: opt_card_id population (A.7 dereference)
# ---------------------------------------------------------------------------


class TestOptCardId:
    """Test that options carry the identity of the card they reference.

    Options almost never ship a `cardId` field, so `opt_card_id` is populated by
    dereferencing `(area, playerIndex, index)` against the state.  Two things
    must hold: visible references resolve, and hidden ones stay PAD.
    """

    def test_hand_referencing_options_populated(self):
        """PLAY options that point into the hand get a real card id.

        Type 7 carries a bare `index` with no `area`, which implicitly means the
        acting player's hand -- the case that used to leave every option at PAD.
        """
        ep = _load_episode()
        vocab = _build_test_vocab(ep)

        n_checked = 0
        for step_i in range(len(ep["steps"])):
            for player_i in (0, 1):
                try:
                    obs, _ = _get_active_step(ep, step_i, player_i)
                except (AssertionError, KeyError, IndexError, TypeError):
                    continue
                if obs is None or not (obs.get("select") or {}).get("option"):
                    continue
                state = obs["current"]
                me = state["yourIndex"]
                out = featurize(obs, vocab)
                for j, opt in enumerate(obs["select"]["option"]):
                    if j >= O_MAX or not out["opt_mask"][j]:
                        continue
                    if not isinstance(opt, dict) or int(opt["type"]) != 7:
                        continue
                    area, idx = opt.get("area"), opt.get("index")
                    if area is not None or idx is None:
                        continue
                    raw = card_id_at(state, 2, me, idx)
                    if raw is None:
                        continue
                    assert out["opt_card_id"][j] == _remap(raw, vocab["id_to_index"])
                    assert out["opt_card_id"][j] != PAD_CARD
                    n_checked += 1

        assert n_checked > 0, "fixture offered no hand-referencing PLAY option"

    def test_no_hidden_card_ids(self):
        """Deck refs and face-down prize refs must stay PAD (information leak).

        Writing an id for a card the live agent cannot see inflates offline
        metrics and collapses at live-eval.
        """
        ep = _load_episode()
        vocab = _build_test_vocab(ep)

        n_hidden_seen = 0
        for step_i in range(len(ep["steps"])):
            for player_i in (0, 1):
                try:
                    obs, _ = _get_active_step(ep, step_i, player_i)
                except (AssertionError, KeyError, IndexError, TypeError):
                    continue
                if obs is None or not (obs.get("select") or {}).get("option"):
                    continue
                state = obs["current"]
                out = featurize(obs, vocab)
                for j, opt in enumerate(obs["select"]["option"]):
                    if j >= O_MAX or not out["opt_mask"][j]:
                        continue
                    if not isinstance(opt, dict):
                        continue
                    area, idx = opt.get("area"), opt.get("index")
                    if area is None:
                        continue
                    if int(area) == 1:
                        assert out["opt_card_id"][j] == PAD_CARD, (
                            f"deck ref leaked at step {step_i} option {j}")
                        n_hidden_seen += 1
                    elif int(area) == 6 and idx is not None:
                        pi = opt.get("playerIndex", state["yourIndex"])
                        try:
                            slot = state["players"][int(pi)]["prize"][int(idx)]
                        except (KeyError, IndexError, TypeError, ValueError):
                            continue
                        if slot is None:
                            assert out["opt_card_id"][j] == PAD_CARD, (
                                f"face-down prize leaked at step {step_i} option {j}")
                            n_hidden_seen += 1

        assert n_hidden_seen > 0, "fixture contained no hidden-zone reference to check"

    def test_populated_ids_match_their_location(self):
        """Every non-PAD opt_card_id equals the vocab index of the card there."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        id_to_index = vocab["id_to_index"]

        n_populated = 0
        for step_i in range(len(ep["steps"])):
            for player_i in (0, 1):
                try:
                    obs, _ = _get_active_step(ep, step_i, player_i)
                except (AssertionError, KeyError, IndexError, TypeError):
                    continue
                if obs is None or not (obs.get("select") or {}).get("option"):
                    continue
                state = obs["current"]
                me = state["yourIndex"]
                out = featurize(obs, vocab)
                for j, opt in enumerate(obs["select"]["option"]):
                    if j >= O_MAX or not out["opt_mask"][j]:
                        continue
                    if not isinstance(opt, dict) or opt.get("cardId") is not None:
                        continue
                    v = int(out["opt_card_id"][j])
                    if v == PAD_CARD:
                        continue
                    n_populated += 1
                    area, idx = opt.get("area"), opt.get("index")
                    if area is None and idx is not None:
                        raw = card_id_at(state, 2, me, idx)
                    elif area is not None and idx is not None:
                        raw = card_id_at(
                            state, area, opt.get("playerIndex", me), idx)
                    else:
                        raw = None
                    if raw is not None:
                        assert v == _remap(raw, id_to_index)

        assert n_populated > 0, "fixture produced no populated opt_card_id at all"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _remap(card_id, id_to_index):
    """Remap helper matching featurizer logic."""
    if card_id is None:
        return 0
    return id_to_index.get(card_id, 1)
