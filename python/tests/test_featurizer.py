"""Tests for ptcg_il.featurizer — Appendix A contract verification.

Uses the sample episode fixture at archive/sample_episodes/80169582.json
and the real engine at pokemon-tcg-ai-battle/.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ptcg_il.featurizer import featurize as _featurize_raw
from ptcg_il.featurizer import (
    ATKCOST_N,
    COUNT_N,
    D_MAX,
    DECK_N,
    DMGCTR_N,
    ENERGY_N,
    E_MAX,
    F_ATK,
    F_CARD,
    F_GLOBAL,
    F_HAND,
    F_OPT,
    F_POKE,
    F_SUM,
    H_MAX,
    HAND_N,
    HP_N,
    L_LOG_MAX,
    LOG_FEAT_DIM,
    L_STATE,
    O_MAX,
    PAD_CARD,
    P_MAX,
    PZ_MAX,
    PAD_ATTACK,
    SUM,
    T_MAX,
    TURN_N,
    featurize,
    option_groups,
)
from ptcg_il.ref_map import build_ref_map, card_id_at

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# parents[2] is the repo root, not `python/`.  This expression was copied from
# `tests/test_featurizer.py`, which sits one level shallower, so the inherited
# `.parent.parent` resolved to `python/archive/` — a directory that exists but
# holds only manifest.csv.  Every test in this file then died in its fixture
# with FileNotFoundError, which reads as 70 broken featurizer assertions.
SAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
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


_ECF_CACHE: dict[int, np.ndarray] | None = None
_EAF_CACHE: dict[int, np.ndarray] | None = None


def _engine_card_features() -> dict[int, np.ndarray]:
    """``{engine_card_id: float32[F_CARD]}`` for every engine card, built once.

    The identity and information-leak tests below *must* pass this to
    ``featurize``.  Without it every ``*_card_feat`` tensor is all zeros, and
    an assertion that a hidden card's row is zero would hold vacuously.
    """
    global _ECF_CACHE
    if _ECF_CACHE is None:
        from ptcg_mine.cards import build_engine_card_features, load_engine

        card_data, attack_data = load_engine()
        _ECF_CACHE = build_engine_card_features(card_data, attack_data)
    return _ECF_CACHE


def _engine_attack_features() -> dict[int, np.ndarray]:
    """``{engine_attack_id: float32[F_ATK]}`` for every engine attack."""
    global _EAF_CACHE
    if _EAF_CACHE is None:
        from ptcg_mine.cards import build_engine_attack_features, load_engine

        _, attack_data = load_engine()
        _EAF_CACHE = build_engine_attack_features(attack_data)
    return _EAF_CACHE


def _feat_of(card_id: int) -> np.ndarray:
    """Expected ``*_card_feat`` row for *card_id*."""
    return np.asarray(_engine_card_features()[card_id], dtype=np.float32)


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
        """poke_card_feat.shape == (12, F_CARD), poke_feat.shape == (12, 26)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)  # MAIN select with visible mons

        result = featurize(obs, vocab, action)
        assert result["poke_card_feat"].shape == (P_MAX, F_CARD)
        assert result["poke_card_feat"].dtype == np.float32
        assert result["poke_feat"].shape == (P_MAX, F_POKE)
        assert result["poke_feat"].dtype == np.float32

    def test_slot_layout(self):
        """my active at row 0 (index 1), opp active at row 6 (index 7)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action,
                           engine_card_features=_engine_card_features())

        # Step 8 Player 0: my active=721, opp active=722, no bench.  Card
        # identity now reaches the model as static features, not as an id.
        assert np.array_equal(result["poke_card_feat"][0], _feat_of(721)), (
            "Expected my active card's features at slot 0"
        )
        assert np.array_equal(result["poke_card_feat"][6], _feat_of(722)), (
            "Expected opp active card's features at slot 6"
        )

    def test_empty_bench_slots_are_pad(self):
        """Empty bench slots should be PAD (all-zero features)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action,
                           engine_card_features=_engine_card_features())

        # Bench slots 1..5 and 7..11 should be PAD
        for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]:
            assert np.all(result["poke_card_feat"][i] == 0.0), f"Slot {i} not PAD"
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
        """hand_card_feat.shape == (30, F_CARD), hand_feat.shape == (30, 2)."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["hand_card_feat"].shape == (H_MAX, F_CARD)
        assert result["hand_card_feat"].dtype == np.float32
        assert result["hand_feat"].shape == (H_MAX, F_HAND)
        assert result["hand_feat"].dtype == np.float32

    def test_hand_cards_at_front_pad_after(self):
        """Hand cards packed at front, PAD after."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action,
                           engine_card_features=_engine_card_features())

        # Player 0 hand has 7 cards in this step
        hand = obs["current"]["players"][0]["hand"]
        n_hand = len(hand)
        assert n_hand == 7

        # First 7 slots carry the real card's features (no engine card has an
        # all-zero row, so non-zero here is exactly "a card is present").
        for i in range(n_hand):
            assert np.array_equal(
                result["hand_card_feat"][i], _feat_of(hand[i]["id"])
            ), f"Hand slot {i} does not carry its card's features"
            assert np.any(result["hand_card_feat"][i] != 0.0), (
                f"Hand slot {i} should not be PAD"
            )
        # Remaining slots should be PAD
        for i in range(n_hand, H_MAX):
            assert np.all(result["hand_card_feat"][i] == 0.0), (
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
        """discard_card_feat and prize_card_feat have correct shapes."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action)

        assert result["discard_card_feat"].shape == (SUM, D_MAX, F_CARD)
        assert result["discard_card_feat"].dtype == np.float32
        assert result["discard_mask"].shape == (SUM, D_MAX)
        assert result["discard_mask"].dtype == bool
        assert result["prize_card_feat"].shape == (SUM, PZ_MAX, F_CARD)
        assert result["prize_card_feat"].dtype == np.float32

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
        """When no stadium, stadium_card_feat = zeros, stadium_present = 0.0."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action,
                           engine_card_features=_engine_card_features())

        assert result["stadium_card_feat"].shape == (1, F_CARD)
        assert np.all(result["stadium_card_feat"][0] == 0.0)
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
        """context_card_feat and effect_card_feat are present and padded."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)

        result = featurize(obs, vocab, action,
                           engine_card_features=_engine_card_features())

        assert result["context_card_feat"].shape == (1, F_CARD)
        assert result["effect_card_feat"].shape == (1, F_CARD)
        # No context card and no effect in this step → both PAD (all zeros)
        assert np.all(result["context_card_feat"][0] == 0.0)
        assert np.all(result["effect_card_feat"][0] == 0.0)

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
        assert result["opt_card_feat"].shape == (O_MAX, F_CARD)
        assert result["opt_card_feat"].dtype == np.float32
        assert result["opt_attack_feat"].shape == (O_MAX, F_ATK)
        assert result["opt_attack_feat"].dtype == np.float32
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
        """ATTACK grounds src/tgt/card on the board, not on constants.

        An attack option carries only ``{type, attackId}``, so everything else
        has to be dereferenced from the state.  Before that was done, ``src``
        was the literal 1, ``tgt`` was -1 and ``opt_card_id`` was PAD for every
        attack — which made ``opt_type_emb``, ``src``, ``tgt`` and ``card_enc``
        byte-identical across the options and left the pointer head choosing
        between attacks on ``opt_attack_feat`` alone.
        """
        ep = _load_episode()
        vocab = _build_test_vocab_with_attacks(ep)
        # Find a step with ATTACK options
        # Step 16 P1: options may have ATTACK
        obs, action = _get_active_step(ep, 16, 1)

        result = featurize(obs, vocab, action,
                           engine_attack_features=_engine_attack_features())

        options = obs["select"]["option"]
        assert len(options) <= O_MAX, "options were reordered; j is not the raw index"

        state = obs["current"]
        me = state["yourIndex"]
        my_active = state["players"][me]["active"][0]
        opp_active = state["players"][1 - me]["active"][0]
        expected_card = int(my_active["id"])

        n_attacks = 0
        for j, opt in enumerate(options):
            if not result["opt_mask"][j] or int(opt["type"]) != 13:
                continue
            assert result["opt_src_idx"][j] == 1, (
                f"ATTACK opt[{j}] src should be my active row 1"
            )
            # No snipe target named, so the default is the defending Active —
            # the same slot the damage-preview scalar already scores against.
            assert opt.get("inPlayIndex") is None, "fixture assumption: no snipe"
            expected_tgt = 7 if opp_active is not None else -1
            assert result["opt_tgt_idx"][j] == expected_tgt, (
                f"ATTACK opt[{j}] tgt should be the defending Active row "
                f"{expected_tgt}, got {result['opt_tgt_idx'][j]}"
            )
            assert result["opt_card_id"][j] == expected_card, (
                f"ATTACK opt[{j}] should carry the attacker's card id "
                f"{expected_card}, got {result['opt_card_id'][j]}"
            )
            # The attack reaches the model as static features, not an index.
            assert np.array_equal(
                result["opt_attack_feat"][j],
                _engine_attack_features()[opt["attackId"]],
            ), f"ATTACK opt[{j}] does not carry its attack's features"
            n_attacks += 1

        assert n_attacks > 0, "step 16 P1 offered no ATTACK option to check"

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
        # Card identity travels as *ids* (listed below); the ``*_card_feat``
        # tensors they gather are produced by ``Policy`` on device and must not
        # appear here.
        "discard_mask",
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
        "opt_bench_idx",
        "opt_scalar",
        "opt_mask",
        "opt_group",
        # Card and attack ids — what the model actually consumes.  It gathers
        # the feature rows itself from the frozen static tables
        # (CARD_FEAT_SOURCES), a ~200x larger tensor that no longer crosses the
        # DataLoader queue.
        "poke_card_id",
        "poke_tool_ids",
        "poke_energy_ids",
        "hand_card_id",
        "stadium_card_id",
        "context_card_id",
        "effect_card_id",
        "discard_ids",
        "prize_ids",
        "opt_card_id",
        "opt_attack_idx",
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
        # Logs (belief module)
        "log_feat",
        "log_mask",
        "log_len",
    }

    def test_out_of_vocab_cards_still_carry_their_features(self):
        """A card absent from the vocab must still arrive with real features.

        Regression: identity used to be routed card_id → vocab index → engine
        id.  An out-of-vocab card hit ``UNKNOWN_CARD`` (index 1), which
        dereferences to the string ``"UNKNOWN"`` and has no engine features, so
        it reached the model as an all-zero row — byte-identical to an empty
        slot.  ``engine_card_features`` covers every engine card, so nothing
        justified dropping it.  Invisible offline (the vocab is built from the
        whole corpus) and only reachable in live play, which is exactly why it
        needs a test.
        """
        ep = _load_episode()
        ecf = _engine_card_features()

        n_checked = 0
        for obs, action in self._iter_active_decisions(ep):
            hand = obs["current"]["players"][obs["current"]["yourIndex"]].get("hand")
            if not hand:
                continue
            # Build a vocab that deliberately does *not* know the hand's cards.
            vocab = _build_test_vocab_with_attacks(ep)
            evicted = {int(c["id"]) for c in hand}
            vocab["id_to_index"] = {
                k: v for k, v in vocab["id_to_index"].items() if int(k) not in evicted
            }

            result = featurize(obs, vocab, action, engine_card_features=ecf)

            for i, card in enumerate(hand[:H_MAX]):
                cid = int(card["id"])
                assert cid not in vocab["id_to_index"], "fixture did not evict the card"
                assert np.array_equal(result["hand_card_feat"][i], _feat_of(cid)), (
                    f"out-of-vocab card {cid} lost its features"
                )
                assert np.any(result["hand_card_feat"][i] != 0.0), (
                    f"out-of-vocab card {cid} arrived as an all-zero row"
                )
                n_checked += 1
            if n_checked:
                break

        assert n_checked > 0, "fixture offered no hand card to evict"

    def test_log_card_feat_shape(self):
        """log_card_feat is [L_LOG_MAX, F_CARD] — one card-feature row per log entry.

        Regression: the card-id column was sliced with the *batched* form
        ``log_feat[:, :, 2]`` (as ``model/belief.py`` does on ``[B, L, 6]``),
        but ``featurize`` emits per-sample ``log_feat[L_LOG_MAX, LOG_FEAT_DIM]``,
        so the slice raised IndexError for every decision point.
        """
        ep = _load_episode()
        vocab = _build_test_vocab_with_attacks(ep)

        count = 0
        for obs, action in self._iter_active_decisions(ep):
            result = featurize(obs, vocab, action)
            assert result["log_feat"].shape == (L_LOG_MAX, LOG_FEAT_DIM)
            assert result["log_card_feat"].shape == (L_LOG_MAX, F_CARD)
            assert result["log_card_feat"].dtype == np.float32
            count += 1

        assert count > 0, "No ACTIVE decisions found in sample episode"

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
        """Every featurize call returns exactly the A.4 keys — ids, no features.

        Uses the unwrapped featurizer on purpose: the module-level ``featurize``
        here adds the model-side gather, and this is the one test whose subject
        is the key set the featurizer actually produces.
        """
        ep = _load_episode()
        vocab = _build_test_vocab_with_attacks(ep)

        count = 0
        for obs, action in self._iter_active_decisions(ep):
            result = _featurize_raw(obs, vocab, action)
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
            "poke_card_feat": (P_MAX, F_CARD),
            "poke_tool_feat": (P_MAX, T_MAX, F_CARD),
            "poke_energy_feat": (P_MAX, E_MAX, F_CARD),
            "hand_card_feat": (H_MAX, F_CARD),
            "stadium_card_feat": (1, F_CARD),
            "context_card_feat": (1, F_CARD),
            "effect_card_feat": (1, F_CARD),
            "discard_card_feat": (SUM, D_MAX, F_CARD),
            "discard_mask": (SUM, D_MAX),
            "prize_card_feat": (SUM, PZ_MAX, F_CARD),
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
            "opt_card_feat": (O_MAX, F_CARD),
            "opt_attack_feat": (O_MAX, F_ATK),
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
            "log_feat": (L_LOG_MAX, LOG_FEAT_DIM),
            "log_mask": (L_LOG_MAX,),
            "log_card_feat": (L_LOG_MAX, F_CARD),
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
                "poke_card_feat", "hand_card_feat", "stadium_card_feat",
                "context_card_feat", "effect_card_feat",
                "discard_card_feat", "prize_card_feat",
                "opt_card_feat", "opt_attack_feat",
                "log_feat", "log_card_feat",
            ]:
                assert result[key].dtype == np.float32, (
                    f"{key} dtype: expected float32, got {result[key].dtype}"
                )

            # Int tensors
            for key in [
                "tok_type", "tok_owner", "tok_zone",
                "opt_type", "opt_src_idx", "opt_tgt_idx",
                "action_idx", "action_len", "minCount", "maxCount",
                "sel_type", "sel_ctx", "log_len",
            ]:
                assert result[key].dtype == np.int64, (
                    f"{key} dtype: expected int64, got {result[key].dtype}"
                )

            # Bool tensors
            for key in ["discard_mask", "tok_mask", "opt_mask", "log_mask"]:
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

    Options almost never ship a `cardId` field, so `opt_card_feat` is populated
    by dereferencing `(area, playerIndex, index)` against the state.  Two things
    must hold: visible references resolve, and hidden ones stay PAD.

    Every test here passes ``engine_card_features``.  Without it every
    ``opt_card_feat`` row is zeros and the leak assertions below would hold no
    matter what the featurizer did.  ``test_pad_row_is_distinguishable`` pins
    the invariant that makes an all-zero row mean "PAD".
    """

    def test_pad_row_is_distinguishable(self):
        """No real engine card has an all-zero feature row.

        The leak tests read "all-zero row" as "no card here".  That is only
        sound while no real card's features are all zero.
        """
        rows = np.stack([np.asarray(v, dtype=np.float32)
                         for v in _engine_card_features().values()])
        assert rows.shape[1] == F_CARD
        n_zero = int((np.abs(rows).sum(axis=1) == 0).sum())
        assert n_zero == 0, (
            f"{n_zero} engine cards have an all-zero feature row, so a zero row "
            f"no longer distinguishes PAD from a real card"
        )

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
                out = featurize(obs, vocab,
                                engine_card_features=_engine_card_features())
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
                    assert np.array_equal(out["opt_card_feat"][j], _feat_of(raw))
                    assert np.any(out["opt_card_feat"][j] != 0.0)
                    n_checked += 1

        assert n_checked > 0, "fixture offered no hand-referencing PLAY option"

    def test_no_hidden_card_ids(self):
        """Genuinely hidden refs stay PAD; a searched deck is not hidden.

        Writing an id for a card the live agent cannot see inflates offline
        metrics and collapses at live-eval.  The converse is a defect too: a
        deck being *searched* is handed to the acting player in
        ``select["deck"]``, and the live agent receives that same payload, so
        refusing to read it denies the model information it legitimately has —
        and leaves every option in the search byte-identical.

        So the invariant is conditional on the payload, not on the area:
          * ``area == DECK`` **with** ``select["deck"]``  -> resolves
          * ``area == DECK`` **without** it               -> PAD
          * face-down prize                               -> PAD, always
        """
        ep = _load_episode()
        vocab = _build_test_vocab(ep)

        n_hidden_seen = 0
        n_deck_resolved = 0
        for step_i in range(len(ep["steps"])):
            for player_i in (0, 1):
                try:
                    obs, _ = _get_active_step(ep, step_i, player_i)
                except (AssertionError, KeyError, IndexError, TypeError):
                    continue
                if obs is None or not (obs.get("select") or {}).get("option"):
                    continue
                state = obs["current"]
                deck_payload = obs["select"].get("deck")
                out = featurize(obs, vocab,
                                engine_card_features=_engine_card_features())
                for j, opt in enumerate(obs["select"]["option"]):
                    if j >= O_MAX or not out["opt_mask"][j]:
                        continue
                    if not isinstance(opt, dict):
                        continue
                    area, idx = opt.get("area"), opt.get("index")
                    if area is None:
                        continue
                    if int(area) == 1:
                        if deck_payload and idx is not None and 0 <= idx < len(deck_payload):
                            assert out["opt_card_id"][j] == deck_payload[idx]["id"], (
                                f"searched deck card unresolved at step {step_i} "
                                f"option {j}")
                            n_deck_resolved += 1
                        else:
                            assert np.all(out["opt_card_feat"][j] == 0.0), (
                                f"deck ref leaked at step {step_i} option {j}")
                            n_hidden_seen += 1
                    elif int(area) == 6 and idx is not None:
                        pi = opt.get("playerIndex", state["yourIndex"])
                        try:
                            slot = state["players"][int(pi)]["prize"][int(idx)]
                        except (KeyError, IndexError, TypeError, ValueError):
                            continue
                        if slot is None:
                            assert np.all(out["opt_card_feat"][j] == 0.0), (
                                f"face-down prize leaked at step {step_i} option {j}")
                            n_hidden_seen += 1

        assert n_deck_resolved > 0, "fixture contained no searched-deck reference"
        assert n_hidden_seen > 0, "fixture contained no hidden-zone reference to check"

    def test_populated_ids_match_their_location(self):
        """Every non-PAD opt_card_feat equals the features of the card there."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)

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
                out = featurize(obs, vocab,
                                engine_card_features=_engine_card_features())
                for j, opt in enumerate(obs["select"]["option"]):
                    if j >= O_MAX or not out["opt_mask"][j]:
                        continue
                    if not isinstance(opt, dict) or opt.get("cardId") is not None:
                        continue
                    row = out["opt_card_feat"][j]
                    if not np.any(row != 0.0):  # PAD
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
                        assert np.array_equal(row, _feat_of(raw))

        assert n_populated > 0, "fixture produced no populated opt_card_feat at all"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: The ``*_card_feat`` keys, and the ids they are gathered from.
#:
#: ``featurize`` emits only the ids -- ``Policy`` gathers the features on its
#: own device, because materialising them per sample was 93% of a sample's
#: bytes and 128 of 174 ms per training step.  These tests are about what the
#: features *contain*, so the wrapper below runs exactly the gather the model
#: runs and the assertions are unchanged.  That ``featurize`` itself emits no
#: feature key is a separate contract, pinned in
#: ``test_card_feature_reconstruction.py``.


def _with_card_feats(result, engine_card_features=None, engine_attack_features=None):
    """Add the ``*_card_feat`` tensors ``Policy._gather_card_feats`` would."""
    from ptcg_il.featurizer import (
        CARD_FEAT_SOURCES, LOG_CARD_ID_COLUMN, build_static_table,
        gather_static_feats,
    )

    tables = {
        "card": build_static_table(engine_card_features or {}, F_CARD),
        "attack": build_static_table(engine_attack_features or {}, F_ATK),
    }
    for feat_key, (id_key, kind) in CARD_FEAT_SOURCES.items():
        if id_key in result:
            result[feat_key] = gather_static_feats(result[id_key], tables[kind])
    if "log_feat" in result:
        result["log_card_feat"] = gather_static_feats(
            result["log_feat"][:, LOG_CARD_ID_COLUMN].astype(np.int64), tables["card"]
        )
    return result


def featurize(*args, **kwargs):
    """``ptcg_il.featurizer.featurize`` plus the model-side card gather."""
    return _with_card_feats(
        _featurize_raw(*args, **kwargs),
        kwargs.get("engine_card_features"),
        kwargs.get("engine_attack_features"),
    )


def _remap(card_id, id_to_index):
    """Remap helper matching featurizer logic."""
    if card_id is None:
        return 0
    return id_to_index.get(card_id, 1)


class TestEngineTablesCarryCardIdentity:
    """Card identity reaches the model *only* as static features looked up by
    engine card id (``featurizer._raw_card``), so omitting the tables is not a
    degradation — it erases the cards entirely.  ``_ids_to_feat`` returns an
    all-zero block for a ``None`` table, and zeros are a legal feature row,
    indistinguishable from an empty slot, so nothing raises.

    Pinned here because ``ptcg_rl.search.batch_evaluate_leaves`` omitted them
    for every MCTS leaf: the search that produced ``mcts_pi`` was blind to
    which cards were in play while the rollout actor stayed sighted.
    """

    def _sums(self, **kw):
        ep = _load_episode()
        obs, action = _get_active_step(ep, 8, 0)
        feats = featurize(obs, _build_test_vocab(ep), action, **kw)
        keys = [k for k in feats if k.endswith("card_feat")]
        assert keys, "fixture produced no card-feature tensors to examine"
        return {k: float(np.abs(feats[k]).sum()) for k in keys}

    def test_tables_present_gives_nonzero_card_features(self):
        sums = self._sums(engine_card_features=_engine_card_features())
        nonzero = {k: v for k, v in sums.items() if v > 0.0}
        assert nonzero, f"no card features carried any signal: {sums}"

    def test_tables_absent_zeroes_every_card_feature(self):
        sums = self._sums()
        assert sums, "fixture examined nothing"
        assert all(v == 0.0 for v in sums.values()), (
            f"expected an all-zero card block without the engine table: {sums}"
        )


# ---------------------------------------------------------------------------
# option_groups() tests
# ---------------------------------------------------------------------------


def _opt_arrays(n_valid: int):
    """Blank option tensors with the first n_valid slots marked valid."""
    return {
        "opt_type": np.zeros(O_MAX, dtype=np.int64),
        "opt_src_idx": np.full(O_MAX, -1, dtype=np.int64),
        "opt_tgt_idx": np.full(O_MAX, -1, dtype=np.int64),
        "opt_bench_idx": np.full(O_MAX, -1, dtype=np.int64),
        "opt_card_feat": np.zeros((O_MAX, F_CARD), dtype=np.float32),
        "opt_attack_feat": np.zeros((O_MAX, F_ATK), dtype=np.float32),
        "opt_scalar": np.zeros((O_MAX, F_OPT), dtype=np.float32),
        "opt_mask": np.array([i < n_valid for i in range(O_MAX)]),
    }


def test_option_groups_merges_identical_options():
    a = _opt_arrays(3)
    a["opt_type"][:3] = 3
    a["opt_card_feat"][:3, 7] = 1.0          # all three are the same card
    g = option_groups(**a)
    assert g[0] == g[1] == g[2]
    assert (g[3:] == -1).all(), "masked slots must be -1"


def test_option_groups_separates_on_each_field():
    for field, setter in [
        ("opt_type", lambda a: a["opt_type"].__setitem__(1, 5)),
        ("opt_src_idx", lambda a: a["opt_src_idx"].__setitem__(1, 4)),
        ("opt_tgt_idx", lambda a: a["opt_tgt_idx"].__setitem__(1, 4)),
        ("opt_card_feat", lambda a: a["opt_card_feat"].__setitem__((1, 3), 1.0)),
        ("opt_attack_feat", lambda a: a["opt_attack_feat"].__setitem__((1, 2), 1.0)),
        ("opt_scalar", lambda a: a["opt_scalar"].__setitem__((1, 2), 0.25)),
    ]:
        a = _opt_arrays(2)
        setter(a)
        g = option_groups(**a)
        assert g[0] != g[1], f"{field} must split the group"


def test_option_groups_ignores_fp32_noise_below_fp16_resolution():
    """Shards store card features as fp16, so the model cannot see a smaller
    difference than fp16 resolution — grouping must not either."""
    a = _opt_arrays(2)
    a["opt_card_feat"][0, 0] = 1.0
    a["opt_card_feat"][1, 0] = 1.0 + 1e-8
    g = option_groups(**a)
    assert g[0] == g[1]


def test_option_groups_all_masked_returns_all_minus_one():
    g = option_groups(**_opt_arrays(0))
    assert (g == -1).all()


def test_featurize_emits_opt_group():
    ep = _load_episode()
    vocab = _build_test_vocab(ep)
    obs, action = _get_active_step(ep, 8, 0)   # the MAIN select used elsewhere in this file
    out = featurize(obs, vocab, action)
    assert out["opt_group"].dtype == np.int64
    assert out["opt_group"].shape == (O_MAX,)
    valid = out["opt_mask"]
    assert (out["opt_group"][~valid] == -1).all()
    assert (out["opt_group"][valid] >= 0).all()
    assert valid.sum() > 0, "fixture produced no options — the assertions above are vacuous"


# ============================================================
# Independently-verifiable guards for the 2026-08-07 change set
# ============================================================


class TestEvolutionMapIsWired:
    """A1 — ``hand_feat[3]`` (``can_evolve``) was constant 0 in every shard.

    ``featurize`` has always accepted ``evolution_map`` and ``ptcg_mine`` has
    always written ``evolution_map.npy``, but nothing loaded it, so the column
    never fired once across 436 564 hand-card rows.  These tests fail if the
    argument stops reaching the flag, in either direction.
    """

    def test_can_evolve_is_zero_without_the_map(self):
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)
        out = featurize(obs, vocab, action,
                        engine_card_features=_engine_card_features())
        assert out["hand_feat"][:, 3].max() == 0.0

    def test_can_evolve_fires_when_the_map_is_supplied(self):
        """Needs a hand card whose pre-evolution is in play and not new."""
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        ecf = _engine_card_features()

        # evolution_map maps card id -> [pre-evolution card ids].  Build one
        # from the fixture itself so the test does not depend on data/.
        n_checked = 0
        for step in range(4, 60):
            for player in (0, 1):
                try:
                    obs, action = _get_active_step(ep, step, player)
                except (AssertionError, IndexError, KeyError, ValueError):
                    continue
                state = obs.get("current")
                if not state:
                    continue
                me = state["yourIndex"]
                p = state["players"][me]
                hand = p.get("hand") or []
                in_play = [q for q in (list(p["active"] or []) + list(p["bench"] or []))
                           if q is not None and not q.get("appearThisTurn", False)]
                if not hand or not in_play:
                    continue
                # Declare the first hand card an evolution of the first in-play one.
                evo = {int(hand[0]["id"]): [int(in_play[0]["id"])]}
                out = featurize(obs, vocab, action,
                                engine_card_features=ecf, evolution_map=evo)
                n_checked += 1
                assert out["hand_feat"][0, 3] == 1.0, (
                    "can_evolve stayed 0 with a matching evolution_map entry — "
                    "the argument is not reaching _build_hand_tokens"
                )
                return
        assert n_checked > 0, "fixture produced no evolvable state — test is vacuous"


class TestGatherMatchesTheLoop:
    """A5 — ``featurize`` now gathers static features vectorised.

    The vectorised path must be *bit-identical* to the ``_ids_to_feat`` loop it
    replaced; a divergence here silently redefines every card the model sees.
    """

    def test_featurize_matches_ids_to_feat_elementwise(self):
        from ptcg_il.featurizer import CARD_FEAT_SOURCES, _ids_to_feat

        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        ecf, eaf = _engine_card_features(), _engine_attack_features()
        obs, action = _get_active_step(ep, 16, 1)
        out = featurize(obs, vocab, action,
                        engine_card_features=ecf, engine_attack_features=eaf)

        n_checked = 0
        for feat_key, (id_key, which) in CARD_FEAT_SOURCES.items():
            table, dim = (ecf, F_CARD) if which == "card" else (eaf, F_ATK)
            expected = _ids_to_feat(out[id_key], table, dim, None)
            assert np.array_equal(out[feat_key], expected), (
                f"{feat_key} diverged from the _ids_to_feat reference"
            )
            n_checked += 1
        assert n_checked == len(CARD_FEAT_SOURCES) > 0

    def test_missing_tables_still_give_zeros(self):
        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 16, 1)
        out = featurize(obs, vocab, action)
        assert out["poke_card_feat"].shape == (P_MAX, F_CARD)
        assert not out["poke_card_feat"].any()

    def test_two_dicts_do_not_share_a_cached_table(self):
        """The cache is keyed on ``id()``; a stale hit would be silent."""
        from ptcg_il.featurizer import _static_table_for

        a = {5: np.ones(F_CARD, dtype=np.float32)}
        b = {5: np.full(F_CARD, 2.0, dtype=np.float32)}
        ta, tb = _static_table_for(a, F_CARD), _static_table_for(b, F_CARD)
        assert ta[5][0] == 1.0 and tb[5][0] == 2.0
        assert _static_table_for(a, F_CARD) is ta, "cache did not hit for the same dict"


class TestEnergyScaling:
    """A3 — energy counts entered the Pokémon token ~10x weaker than its flags."""

    def test_energy_divisor_covers_the_measured_maximum(self):
        # Max energies observed on one Pokémon over the corpus is 7.
        assert ENERGY_N == 8.0
        assert ENERGY_N > 7.0, "divisor must not clip a real board state"

    def test_ko_pressure_recovers_raw_counts(self):
        """``_ko_pressure`` multiplies by ENERGY_N; the two must stay in step."""
        from ptcg_il.featurizer import _energy_histogram

        hist = _energy_histogram([0, 0, 3])
        assert np.isclose(hist[0] * ENERGY_N, 2.0)
        assert np.isclose(hist[3] * ENERGY_N, 1.0)


def _poke(card_id: int, serial: int) -> dict:
    return {"id": card_id, "serial": serial, "hp": 100, "maxHp": 100,
            "appearThisTurn": False, "energies": [], "energyCards": [],
            "tools": [], "preEvolution": []}


def _obs_with_bench(same_card_id: int = 104) -> dict:
    """Minimal observation: I have an Active plus one Bench of the *same* card."""
    def player(active, bench):
        return {"active": active, "bench": bench, "benchMax": 5, "deckCount": 40,
                "discard": [], "prize": [None] * 6, "handCount": 0, "hand": [],
                "poisoned": False, "burned": False, "asleep": False,
                "paralyzed": False, "confused": False}

    return {
        "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                   "option": [{"type": 14}]},
        "logs": [],
        "current": {
            "turn": 5, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False,
            "energyAttached": False, "retreated": False, "result": -1,
            "stadium": [], "looking": None,
            "players": [
                player([_poke(same_card_id, 13)], [_poke(same_card_id, 73)]),
                player([_poke(200, 91)], []),
            ],
        },
    }


class TestSkillAndRetreatRefs:
    """B2 — SKILL options differ only by ``serial``; RETREAT never duplicates."""

    def test_skill_options_resolve_serial_to_distinct_rows(self):
        from ptcg_il.ref_map import serial_row

        # Synthetic: the sample episode is 19 steps and never benches a Pokémon
        # for the acting player, so a fixture-driven version of this test would
        # pass vacuously.  Two copies of the *same card* is exactly the case
        # that matters — they share cardId and differ only by serial.
        obs = _obs_with_bench(same_card_id=104)
        ref_map = build_ref_map(obs)

        assert serial_row(ref_map, 13) == 1, "active should resolve to row 1"
        assert serial_row(ref_map, 73) == 2, "first bench slot should resolve to row 2"
        assert serial_row(ref_map, 13) != serial_row(ref_map, 73), (
            "two copies of the same card must resolve to different rows — this is "
            "the only thing that separates their SKILL options"
        )
        assert serial_row(ref_map, 91) == 7, "opponent active should resolve to row 7"

    def test_unknown_serial_falls_back_to_the_null_token(self):
        from ptcg_il.ref_map import serial_row

        ref_map = build_ref_map(_obs_with_bench())
        assert serial_row(ref_map, 999999) == -1
        assert serial_row(ref_map, None) == -1

    def test_skill_options_get_distinct_src_rows_end_to_end(self):
        """The whole point: two same-card SKILLs must not be one option group."""
        obs = _obs_with_bench(same_card_id=104)
        obs["select"]["option"] = [
            {"type": 15, "cardId": 104, "serial": 13},
            {"type": 15, "cardId": 104, "serial": 73},
        ]
        out = featurize(obs, _build_test_vocab(_load_episode()), [0],
                        engine_card_features=_engine_card_features())

        assert out["opt_src_idx"][0] == 1 and out["opt_src_idx"][1] == 2
        assert out["opt_card_id"][0] == out["opt_card_id"][1] == 104
        groups = out["opt_group"][out["opt_mask"]]
        assert len(set(groups.tolist())) == 2, (
            "the two SKILL options collapsed into one equivalence class, so the "
            "group-marginal CE would train them at zero loss"
        )

    def test_retreat_resolves_the_active_rather_than_hardcoding_row_1(self):
        obs = _obs_with_bench()
        obs["select"]["option"] = [{"type": 12}]
        out = featurize(obs, _build_test_vocab(_load_episode()), [0],
                        engine_card_features=_engine_card_features())
        assert out["opt_src_idx"][0] == 1
        assert out["opt_card_id"][0] == 104

        # With no Active in play the row does not exist; pointing at it would
        # gather a PAD token as though it were a real Pokémon.
        obs2 = _obs_with_bench()
        obs2["current"]["players"][0]["active"] = []
        obs2["select"]["option"] = [{"type": 12}]
        out2 = featurize(obs2, _build_test_vocab(_load_episode()), [0],
                         engine_card_features=_engine_card_features())
        assert out2["opt_src_idx"][0] == -1


# ---------------------------------------------------------------------------
# CARD-option damage preview (opt_scalar dims 6/7)
# ---------------------------------------------------------------------------


def _poke_e(card_id: int, serial: int, hp: int, energies: list[int] | None = None) -> dict:
    """A Pokémon in play with explicit current HP and attached energy types."""
    energies = list(energies or [])
    return {
        "id": card_id, "serial": serial, "hp": hp, "maxHp": hp,
        "appearThisTurn": False, "energies": energies,
        "energyCards": [{"id": 1, "serial": 900 + i, "playerIndex": 0}
                        for i in range(len(energies))],
        "tools": [], "preEvolution": [],
    }


def _obs_with_boards(my_active, my_bench, opp_active, opp_bench, options) -> dict:
    """Minimal observation with both boards populated and a CARD select."""
    def player(active, bench):
        return {"active": active, "bench": bench, "benchMax": 5, "deckCount": 40,
                "discard": [], "prize": [None] * 6,
                "handCount": len(_HAND_STUB), "hand": list(_HAND_STUB),
                "poisoned": False, "burned": False, "asleep": False,
                "paralyzed": False, "confused": False}

    return {
        "select": {"type": 1, "context": 3, "minCount": 1, "maxCount": 1,
                   "option": options},
        "logs": [],
        "current": {
            "turn": 5, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False,
            "energyAttached": False, "retreated": False, "result": -1,
            "stadium": [], "looking": None,
            "players": [player(my_active, my_bench), player(opp_active, opp_bench)],
        },
    }


_HAND_STUB = [{"id": 1121, "serial": 500, "playerIndex": 0}]

# Engine ids used below, with the static data the expectations are derived from:
#   677 Riolu       {F} 80hp  weak {P}   atk 30 for {F}
#   305 Dunsparce   {C} 70hp  weak {F}   atk 0 for {C}, 20 for {C}{C}
#   675 Lunatone    {F} 110hp weak {G}   atk 50 for {F}{F}
#   676 Solrock     {F} 110hp weak {G}   atk 70 for {F}
_F_ENERGY = 6  # EnergyType index for Fighting


class TestCardOptionDamagePreview:
    """CARD options that name a Pokémon in play must carry the damage preview.

    Before this, ``opt_scalar`` was all-zero for every CARD option: dims 6-9 are
    filled only in the ATTACK branch and 10-12 only in RETREAT.  A Boss's Orders
    gust is a KO-evaluation decision with no KO feature, leaving ``src`` and
    ``card_enc`` as the only 2 of PointerHead's 6 base terms that separate the
    options.
    """

    def _featurize(self, obs):
        return featurize(obs, _build_test_vocab(_load_episode()), [0],
                         engine_card_features=_engine_card_features())

    def test_opponent_targets_get_damage_ratio_over_their_own_hp(self):
        """Gust targets are scored by my Active's best affordable attack."""
        obs = _obs_with_boards(
            my_active=[_poke_e(677, 10, 80, [_F_ENERGY])],   # Riolu, 1 {F}
            my_bench=[],
            opp_active=[_poke_e(676, 20, 110)],              # Solrock
            opp_bench=[_poke_e(305, 21, 50),                 # Dunsparce, weak {F}
                       _poke_e(675, 22, 110)],               # Lunatone, weak {G}
            options=[{"type": 3, "area": 5, "playerIndex": 1, "index": 0},
                     {"type": 3, "area": 5, "playerIndex": 1, "index": 1}],
        )
        out = self._featurize(obs)

        # Riolu's 30 doubles to 60 on Dunsparce's {F} weakness -> 60/50 = 1.2
        assert np.isclose(out["opt_scalar"][0, 6], 60.0 / 50.0), (
            "opponent-bench target did not get my Active's weakness-adjusted "
            "damage over its current HP"
        )
        assert out["opt_scalar"][0, 7] == 1.0, "60 damage into 50 HP is a KO"

        # Lunatone is weak to {G}, not {F} -> 30 undoubled, 30/110
        assert np.isclose(out["opt_scalar"][1, 6], 30.0 / 110.0)
        assert out["opt_scalar"][1, 7] == 0.0, "30 damage into 110 HP is not a KO"

    def test_weakness_is_what_separates_the_two_gust_targets(self):
        """The KO flag must come from weakness, not from raw damage."""
        obs = _obs_with_boards(
            my_active=[_poke_e(677, 10, 80, [_F_ENERGY])],
            my_bench=[],
            opp_active=[_poke_e(676, 20, 110)],
            opp_bench=[_poke_e(305, 21, 50), _poke_e(675, 22, 110)],
            options=[{"type": 3, "area": 5, "playerIndex": 1, "index": 0},
                     {"type": 3, "area": 5, "playerIndex": 1, "index": 1}],
        )
        out = self._featurize(obs)
        # Undoubled, 30 into 50 HP would be 0.6 and no KO.  The flag flipping
        # is the whole signal a gust decision needs.
        assert out["opt_scalar"][0, 6] > 1.0 > out["opt_scalar"][1, 6]
        assert out["opt_scalar"][0, 7] > out["opt_scalar"][1, 7]

    def test_my_own_targets_get_offense_and_an_incoming_ko_flag(self):
        """Promote/switch candidates: what it does, and whether it survives."""
        obs = _obs_with_boards(
            my_active=[],                                    # just got KO'd
            my_bench=[_poke_e(677, 10, 80, [_F_ENERGY]),     # Riolu, 1 {F}
                      _poke_e(305, 11, 70)],                 # Dunsparce, 0 energy
            opp_active=[_poke_e(676, 20, 110)],              # Solrock, 70 for {F}
            opp_bench=[],
            options=[{"type": 3, "area": 5, "playerIndex": 0, "index": 0},
                     {"type": 3, "area": 5, "playerIndex": 0, "index": 1}],
        )
        out = self._featurize(obs)

        # Riolu: 30 into Solrock (weak {G}, so undoubled) over 110 HP
        assert np.isclose(out["opt_scalar"][0, 6], 30.0 / 110.0)
        # Solrock's 70 does not KO an 80 HP Riolu (Riolu is weak to {P})
        assert out["opt_scalar"][0, 7] == 0.0

        # Dunsparce has no energy, so no attack is affordable
        assert out["opt_scalar"][1, 6] == 0.0, (
            "an unaffordable attack must not count as offense"
        )
        # Solrock's 70 doubles on Dunsparce's {F} weakness -> 140 into 70 HP
        assert out["opt_scalar"][1, 7] == 1.0, (
            "incoming weakness ignored — this candidate dies on promotion"
        )

    def test_hand_targets_stay_zero(self):
        """CARD options naming a hand card must not read a poke slot."""
        obs = _obs_with_boards(
            my_active=[_poke_e(677, 10, 80, [_F_ENERGY])],
            my_bench=[],
            opp_active=[_poke_e(676, 20, 110)],
            opp_bench=[],
            options=[{"type": 3, "area": 2, "playerIndex": 0, "index": 0}],
        )
        out = self._featurize(obs)
        assert out["opt_src_idx"][0] >= 13, "hand cards live on rows 13..42"
        assert out["opt_scalar"][0, 6] == 0.0
        assert out["opt_scalar"][0, 7] == 0.0

    def test_missing_card_tables_leave_the_preview_at_zero(self):
        """No static features means no damage information, not a garbage ratio."""
        obs = _obs_with_boards(
            my_active=[_poke_e(677, 10, 80, [_F_ENERGY])],
            my_bench=[],
            opp_active=[_poke_e(676, 20, 110)],
            opp_bench=[_poke_e(305, 21, 50)],
            options=[{"type": 3, "area": 5, "playerIndex": 1, "index": 0}],
        )
        out = featurize(obs, _build_test_vocab(_load_episode()), [0])
        assert out["opt_scalar"][0, 6] == 0.0
        assert out["opt_scalar"][0, 7] == 0.0

    def test_attack_options_keep_their_own_preview(self):
        """Dims 6/7 are shared with ATTACK; the ATTACK reading must not regress."""
        ep = _load_episode()
        vocab = _build_test_vocab_with_attacks(ep)
        ecf, eaf = _engine_card_features(), _engine_attack_features()
        n_checked = 0
        for step_idx in range(len(ep["steps"]) - 1):
            for player in (0, 1):
                rec = ep["steps"][step_idx][player]
                if rec.get("status") != "ACTIVE":
                    continue
                sel = rec["observation"].get("select")
                if sel is None:
                    continue
                atk = [j for j, o in enumerate(sel["option"])
                       if int(o["type"]) == 13]
                if not atk:
                    continue
                obs, action = _get_active_step(ep, step_idx, player)
                out = featurize(obs, vocab, action, engine_card_features=ecf,
                                engine_attack_features=eaf)
                for j in atk:
                    assert 0.0 <= out["opt_scalar"][j, 6] <= 2.0
                    assert out["opt_scalar"][j, 7] in (0.0, 1.0)
                    n_checked += 1
        assert n_checked > 0, "fixture offered no ATTACK options — test is vacuous"


class TestAffordabilityTolerance:
    """Energy counts and attack costs round-trip through *different* divisors.

    ``poke_feat``'s histogram is normalised by ``ENERGY_N`` and an attack's cost
    block by ``ATKCOST_N``, so "1 energy" and "costs 1" come back as 1.0 and
    1.00000001.  A bare ``<`` then calls an exactly-paid attack unaffordable —
    which is the single most common board state in the game.
    """

    def test_exactly_paid_cost_is_affordable(self):
        from ptcg_il.featurizer import _attack_is_affordable

        cost = np.zeros(12)
        cost[_F_ENERGY] = 1.00000001   # as denormalised from the static table
        have = np.zeros(12)
        have[_F_ENERGY] = 1.0          # as denormalised from poke_feat
        assert _attack_is_affordable(cost, have), (
            "an attack whose cost is exactly met read as unaffordable"
        )

    def test_genuinely_short_energy_is_still_unaffordable(self):
        """The tolerance must not swallow a whole missing energy."""
        from ptcg_il.featurizer import _attack_is_affordable

        cost = np.zeros(12)
        cost[_F_ENERGY] = 2.0
        have = np.zeros(12)
        have[_F_ENERGY] = 1.0
        assert not _attack_is_affordable(cost, have)

    def test_wrong_colour_is_still_unaffordable(self):
        from ptcg_il.featurizer import _attack_is_affordable

        cost = np.zeros(12)
        cost[_F_ENERGY] = 1.0
        have = np.zeros(12)
        have[1] = 1.0  # a Grass energy does not pay a Fighting cost
        assert not _attack_is_affordable(cost, have)

    def test_ko_pressure_sees_an_exactly_paid_attack(self):
        """The CLS offense features were reading 0.0 on the common case."""
        from ptcg_il.featurizer import _build_poke_tokens, _feat_gatherer, _ko_pressure

        obs = _obs_with_boards(
            my_active=[_poke_e(677, 10, 80, [_F_ENERGY])],   # Riolu, exactly 1 {F}
            my_bench=[],
            opp_active=[_poke_e(676, 20, 110)],              # Solrock
            opp_bench=[],
            options=[],
        )
        poke_id, poke_feat, _, _ = _build_poke_tokens(obs["current"], 0)
        poke_card_feat = _feat_gatherer(_engine_card_features(), F_CARD)(poke_id)
        my_ratio, can_ko, _, _ = _ko_pressure(poke_card_feat, poke_feat)
        assert np.isclose(my_ratio, 30.0 / 110.0), (
            "my_ratio stayed 0.0 with an exactly-affordable attack"
        )
        assert can_ko == 0.0
