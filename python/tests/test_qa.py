"""Tests for ptcg_il.qa — QA gates (D.5)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptcg_il.featurizer import F_CARD
from ptcg_il.qa import (
    CARDTYPE_BASIC_ENERGY,
    balance_report,
    check_attachment_collision,
    check_coverage,
    check_deck_legality,
    check_label_sanity,
    check_oov_coverage_curve,
    check_outcome_balance,
    check_reference_roundtrip,
    check_variable_length_multi_select,
    run_qa_checks,
)


def _card_row(seed: int) -> np.ndarray:
    """A distinct, reproducible ``opt_card_feat`` row standing in for one card.

    The collision check compares feature *rows*, so what matters is that
    different cards get different rows and the same card gets the same one.
    """
    return np.full(F_CARD, float(seed), dtype=np.float32)


# ============================================================
# check_coverage
# ============================================================


def test_coverage_all_covered():
    id_to_index = {1: 2, 2: 3, 3: 4}
    ok, missing, total = check_coverage([1, 2, 3], id_to_index)
    assert ok
    assert missing == []
    assert total == 3


def test_coverage_missing_raises():
    id_to_index = {1: 2, 2: 3}
    with pytest.raises(AssertionError, match="not in vocab"):
        check_coverage([1, 2, 3], id_to_index)


def test_coverage_empty():
    ok, missing, total = check_coverage([], {1: 2})
    assert ok
    assert missing == []
    assert total == 0


# ============================================================
# check_oov_coverage_curve
# ============================================================


def test_oov_curve_empty():
    curve = check_oov_coverage_curve([], [200, 300])
    assert curve == {200: 0.0, 300: 0.0}


def test_oov_curve_all_covered():
    # 100 copies of each of 3 cards → N_VOCAB=3 covers everything
    cards = [1] * 100 + [2] * 100 + [3] * 100
    curve = check_oov_coverage_curve(cards, [2, 3, 5])
    assert curve[3] == 0.0  # all covered
    assert curve[2] > 0.0   # only 2/3 covered
    assert curve[5] == 0.0


def test_oov_curve_single_card():
    cards = [42] * 1000
    curve = check_oov_coverage_curve(cards, [0, 1])
    assert curve[0] == 1.0  # N_VOCAB=0 covers nothing
    assert curve[1] == 0.0  # N_VOCAB=1 covers all


# ============================================================
# check_variable_length_multi_select
# ============================================================


def test_variable_length_none(tmp_path):
    meta = pd.DataFrame({
        "maxCount": [1, 1, 1],
        "minCount": [1, 1, 1],
    })
    result = check_variable_length_multi_select(tmp_path, meta)
    assert result["n_variable_length"] == 0
    assert result["variable_length_share"] == 0.0


def test_variable_length_some(tmp_path):
    meta = pd.DataFrame({
        "maxCount": [1, 3, 3, 3, 1],
        "minCount": [1, 1, 3, 1, 1],
    })
    result = check_variable_length_multi_select(tmp_path, meta)
    # 3 multi-select rows, 2 have minCount < maxCount
    assert result["n_variable_length"] == 2
    assert result["variable_length_share"] == pytest.approx(2 / 3)


def test_variable_length_all_same(tmp_path):
    meta = pd.DataFrame({
        "maxCount": [3, 3, 5, 5],
        "minCount": [3, 3, 5, 5],
    })
    result = check_variable_length_multi_select(tmp_path, meta)
    assert result["n_variable_length"] == 0


# ============================================================
# check_attachment_collision
# ============================================================


def test_attachment_collision_uses_opt_group(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    O = 64
    # Sample 0: slots 0 and 1 identical (group 0), slot 2 distinct (group 1).
    opt_group = np.full((1, O), -1, dtype=np.int64)
    opt_group[0, :3] = [0, 0, 1]
    opt_mask = np.zeros((1, O), dtype=bool)
    opt_mask[0, :3] = True
    opt_type = np.zeros((1, O), dtype=np.int64)
    opt_type[0, :3] = 3
    np.savez(shard_dir / "train-00000.npz",
             opt_group=opt_group, opt_mask=opt_mask, opt_type=opt_type)

    rep = check_attachment_collision(shard_dir, pd.DataFrame({"shard": ["train-00000.npz"]}))
    assert rep["n_options_total"] == 3
    assert rep["n_indistinguishable_options"] == 2
    assert rep["indistinguishable_share"] == pytest.approx(2 / 3)
    assert rep["collision_by_opt_type"] == {3: 2}


def test_attachment_collision_raises_on_zero_options(tmp_path):
    """An audit that examined nothing must not report a clean 0.0 share."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    with pytest.raises(ValueError, match="no options"):
        check_attachment_collision(shard_dir, pd.DataFrame({"shard": []}))


def test_attachment_collision_no_collisions(tmp_path):
    """Synthetic shard with attachment options that all differ."""
    S, O = 2, 64
    opt_type = np.zeros((S, O), dtype=np.int64)
    opt_src = np.full((S, O), -1, dtype=np.int64)
    opt_card = np.zeros((S, O, F_CARD), dtype=np.float32)
    opt_mask = np.zeros((S, O), dtype=bool)

    # Sample 0: two ENERGY options on different pokemon (same card features,
    # different source — distinguishable)
    opt_type[0, 0] = 6  # ENERGY
    opt_type[0, 1] = 6
    opt_src[0, 0] = 1
    opt_src[0, 1] = 7
    opt_card[0, 0] = _card_row(42)
    opt_card[0, 1] = _card_row(42)
    opt_mask[0, :2] = True

    # Sample 1: CARD attachment
    opt_type[1, 0] = 3  # CARD
    opt_src[1, 0] = 1
    opt_card[1, 0] = _card_row(99)
    opt_mask[1, :1] = True

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    # opt_group: sample 0 slots 0,1 differ (src 1 vs 7), sample 1 has one option.
    opt_group_arr = np.full((S, O), -1, dtype=np.int64)
    opt_group_arr[0, :2] = [0, 1]
    opt_group_arr[1, :1] = [0]
    np.savez_compressed(shard_dir / "train-00000.npz",
                        opt_type=opt_type, opt_src_idx=opt_src,
                        opt_card_feat=opt_card, opt_mask=opt_mask,
                        opt_group=opt_group_arr)

    meta = pd.DataFrame({"shard": ["train-00000.npz"] * S, "row": [0, 1]})
    result = check_attachment_collision(shard_dir, meta)
    assert result["n_collision_options"] == 0
    assert result["n_attachment_options"] == 3


def test_attachment_collision_with_collisions(tmp_path):
    """Two ENERGY options on the same pokemon with the same card features."""
    S, O = 1, 64
    opt_type = np.zeros((S, O), dtype=np.int64)
    opt_src = np.full((S, O), -1, dtype=np.int64)
    opt_card = np.zeros((S, O, F_CARD), dtype=np.float32)
    opt_mask = np.zeros((S, O), dtype=bool)

    # Two identical options: same src, same card features
    opt_type[0, 0] = 6
    opt_type[0, 1] = 6
    opt_src[0, 0] = 1
    opt_src[0, 1] = 1  # same source!
    opt_card[0, 0] = _card_row(9)
    opt_card[0, 1] = _card_row(9)  # same card!
    opt_mask[0, :2] = True

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    # opt_group: both identical options get the same group.
    opt_group_arr = np.full((S, O), -1, dtype=np.int64)
    opt_group_arr[0, :2] = [0, 0]
    np.savez_compressed(shard_dir / "train-00000.npz",
                        opt_type=opt_type, opt_src_idx=opt_src,
                        opt_card_feat=opt_card, opt_mask=opt_mask,
                        opt_group=opt_group_arr)

    meta = pd.DataFrame({"shard": ["train-00000.npz"], "row": [0]})
    result = check_attachment_collision(shard_dir, meta)
    assert result["n_collision_options"] == 1  # 2 options, 1 unique pair
    assert result["n_attachment_options"] == 2


# ============================================================
# check_label_sanity
# ============================================================


def test_label_sanity_passing(tmp_path):
    """Valid single-select labels."""
    S, O = 3, 64
    action_idx = np.full((S, O), -1, dtype=np.int64)
    action_idx[0, 0] = 1
    action_idx[1, 0] = 0
    action_idx[2, 0] = 3
    action_len = np.array([1, 1, 1], dtype=np.int64)
    opt_mask = np.zeros((S, O), dtype=bool)
    opt_mask[:, :5] = True  # 5 valid options per sample
    min_count = np.array([1, 1, 1], dtype=np.int64)
    max_count = np.array([1, 1, 1], dtype=np.int64)

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez_compressed(shard_dir / "train-00000.npz",
                        action_idx=action_idx, action_len=action_len,
                        opt_mask=opt_mask, minCount=min_count, maxCount=max_count)

    ok, details = check_label_sanity(shard_dir)
    assert ok
    assert details["samples_checked"] == 3


def test_label_sanity_out_of_bounds(tmp_path):
    """Action index beyond option count."""
    S, O = 1, 64
    action_idx = np.full((S, O), -1, dtype=np.int64)
    action_idx[0, 0] = 10  # only 3 valid options
    action_len = np.array([1], dtype=np.int64)
    opt_mask = np.zeros((S, O), dtype=bool)
    opt_mask[0, :3] = True
    min_count = np.array([1], dtype=np.int64)
    max_count = np.array([1], dtype=np.int64)

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez_compressed(shard_dir / "train-00000.npz",
                        action_idx=action_idx, action_len=action_len,
                        opt_mask=opt_mask, minCount=min_count, maxCount=max_count)

    with pytest.raises(AssertionError, match="Label sanity failure"):
        check_label_sanity(shard_dir)


def test_label_sanity_duplicate(tmp_path):
    """Multi-select with duplicate picks."""
    S, O = 1, 64
    action_idx = np.full((S, O), -1, dtype=np.int64)
    action_idx[0, 0] = 1
    action_idx[0, 1] = 1  # duplicate
    action_len = np.array([2], dtype=np.int64)
    opt_mask = np.zeros((S, O), dtype=bool)
    opt_mask[0, :5] = True
    min_count = np.array([2], dtype=np.int64)
    max_count = np.array([2], dtype=np.int64)

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez_compressed(shard_dir / "train-00000.npz",
                        action_idx=action_idx, action_len=action_len,
                        opt_mask=opt_mask, minCount=min_count, maxCount=max_count)

    with pytest.raises(AssertionError, match="Label sanity failure"):
        check_label_sanity(shard_dir)


def test_label_sanity_wrong_length(tmp_path):
    """action_len outside [minCount, maxCount]."""
    S, O = 1, 64
    action_idx = np.full((S, O), -1, dtype=np.int64)
    action_idx[0, 0] = 1
    action_len = np.array([1], dtype=np.int64)
    opt_mask = np.zeros((S, O), dtype=bool)
    opt_mask[0, :5] = True
    min_count = np.array([2], dtype=np.int64)
    max_count = np.array([2], dtype=np.int64)

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez_compressed(shard_dir / "train-00000.npz",
                        action_idx=action_idx, action_len=action_len,
                        opt_mask=opt_mask, minCount=min_count, maxCount=max_count)

    with pytest.raises(AssertionError, match="Label sanity failure"):
        check_label_sanity(shard_dir)


# ============================================================
# check_outcome_balance
# ============================================================


def test_outcome_balance():
    meta = pd.DataFrame({"won": [True, True, True, False, False]})
    result = check_outcome_balance(meta)
    assert result["won_count"] == 3
    assert result["lost_count"] == 2


def test_outcome_balance_all_won():
    meta = pd.DataFrame({"won": [True] * 100})
    result = check_outcome_balance(meta)
    assert result["won_count"] == 100
    assert result["lost_count"] == 0


# ============================================================
# check_deck_legality
# ============================================================


def test_deck_legality_not_60():
    ok, failures = check_deck_legality([1] * 59)
    assert not ok
    assert any("59 cards" in f for f in failures)


def test_deck_legality_too_many_copies():
    # 5 copies of a non-basic-energy card (use a high ID unlikely to be basic energy)
    deck = [999] * 5 + [1] * 55
    ok, failures = check_deck_legality(deck)
    assert not ok
    assert any("5 copies" in f for f in failures)


def test_deck_legality_basic_energy_exempt():
    """Basic energies (cardType=5) can exceed 4 copies.
    Card ids 1-8 are recognized as basic energies by the engine.
    """
    # Use card id 1 which is typically a Grass basic energy
    deck = [1] * 59 + [2]
    # This should pass the copy-limit check (id 1 is basic energy, id 2 too)
    # But it will fail on basic pokemon requirement since we have only energies
    ok, failures = check_deck_legality(deck)
    assert not ok  # fails on "0 basic Pokémon"
    assert any("0 basic" in f for f in failures)


# ============================================================
# balance_report
# ============================================================


def test_balance_report():
    meta = pd.DataFrame({
        "sel_ctx": [0, 0, 0, 1, 1, 2],
        "archetype_self": [0, 0, 0, 0, 1, 1],
    })
    result = balance_report(meta)
    assert result["balance_sel_ctx"] == {0: 3, 1: 2, 2: 1}
    assert result["balance_arch_self"] == {0: 4, 1: 2}
    # All < 100 so they're "rare"
    assert len(result["rare_contexts"]) == 3


# ============================================================
# check_reference_roundtrip
# ============================================================


def _roundtrip_shard(tmp_path, *, src_row, card_row_value, state_slot_value):
    """One-sample shard whose only option points at a hand row."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(exist_ok=True)
    O, F = 64, 8
    opt_mask = np.zeros((1, O), dtype=bool); opt_mask[0, 0] = True
    opt_src = np.full((1, O), -1, dtype=np.int64); opt_src[0, 0] = src_row
    opt_card = np.zeros((1, O, F), dtype=np.float16); opt_card[0, 0, 0] = card_row_value
    hand = np.zeros((1, 30, F), dtype=np.float16); hand[0, src_row - 13, 0] = state_slot_value
    np.savez(shard_dir / "train-00000.npz",
             opt_mask=opt_mask, opt_src_idx=opt_src, opt_card_feat=opt_card,
             hand_card_feat=hand,
             poke_card_feat=np.zeros((1, 12, F), dtype=np.float16),
             stadium_card_feat=np.zeros((1, 1, F), dtype=np.float16))
    return shard_dir


def test_reference_roundtrip_passes_when_pointer_resolves(tmp_path):
    shard_dir = _roundtrip_shard(tmp_path, src_row=15, card_row_value=1.0, state_slot_value=1.0)
    passed, details = check_reference_roundtrip(shard_dir)
    assert passed is True
    assert details["n_compared"] == 1
    assert details["n_mismatch"] == 0


def test_reference_roundtrip_catches_off_by_one(tmp_path):
    """The pointer names hand row 15 but the card there is a different card."""
    shard_dir = _roundtrip_shard(tmp_path, src_row=15, card_row_value=1.0, state_slot_value=0.5)
    passed, details = check_reference_roundtrip(shard_dir)
    assert passed is False
    assert details["n_mismatch"] == 1


def test_reference_roundtrip_flags_impossible_source_rows(tmp_path):
    """Row 0 is CLS and row 43-44 are summary tokens — no option may point there."""
    shard_dir = _roundtrip_shard(tmp_path, src_row=15, card_row_value=1.0, state_slot_value=1.0)
    d = dict(np.load(shard_dir / "train-00000.npz"))
    d["opt_src_idx"][0, 0] = 43
    np.savez(shard_dir / "train-00000.npz", **d)
    passed, details = check_reference_roundtrip(shard_dir)
    assert passed is False
    assert details["n_src_out_of_range"] == 1


def test_reference_roundtrip_returns_none_when_nothing_compared(tmp_path):
    """Zero comparisons is 'skipped', never 'passed'."""
    shard_dir = _roundtrip_shard(tmp_path, src_row=15, card_row_value=0.0, state_slot_value=0.0)
    passed, details = check_reference_roundtrip(shard_dir)
    assert passed is None
    assert details["n_compared"] == 0


# ============================================================
# run_qa_checks integration
# ============================================================


def test_run_qa_checks_minimal(tmp_path):
    """Integration with a small synthetic shard and meta."""
    # Create shard
    S, O = 5, 64
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()

    action_idx = np.full((S, O), -1, dtype=np.int64)
    for s in range(S):
        action_idx[s, 0] = s
    action_len = np.ones(S, dtype=np.int64)
    opt_mask = np.zeros((S, O), dtype=bool)
    opt_mask[:, :8] = True
    min_count = np.ones(S, dtype=np.int64)
    max_count = np.ones(S, dtype=np.int64)
    opt_type = np.zeros((S, O), dtype=np.int64)
    opt_src = np.full((S, O), -1, dtype=np.int64)
    opt_card = np.zeros((S, O), dtype=np.int64)

    np.savez_compressed(shard_dir / "train-00000.npz",
                        action_idx=action_idx, action_len=action_len,
                        opt_mask=opt_mask, minCount=min_count, maxCount=max_count,
                        opt_type=opt_type, opt_src_idx=opt_src, opt_card_id=opt_card)

    # Create meta
    meta = pd.DataFrame({
        "sample_uid": [f"x_{i}" for i in range(S)],
        "shard": ["train-00000.npz"] * S,
        "row": list(range(S)),
        "episode_id": ["ep1"] * S,
        "player": [0] * S,
        "team": ["expert"] * S,
        "archetype_self": [0] * S,
        "archetype_opp": [1] * S,
        "sel_type": [1] * S,
        "sel_ctx": [0] * S,
        "minCount": [1] * S,
        "maxCount": [1] * S,
        "won": [True, True, True, False, False],
    })
    meta.to_parquet(tmp_path / "meta.parquet", index=False)

    # Create vocab with coverage for card ids 1-2
    vocab = {
        "id_to_index": {1: 2, 2: 3},
        "index_to_id": ["PAD", "UNKNOWN", 1, 2],
        "freq": {1: 5, 2: 3},
        "size": 4,
        "freq_coverage": [(1, 5, 0.625), (2, 3, 1.0)],
    }

    # Use a deck that passes all legality checks:
    # card 22 = basic Pokemon (no ACE SPEC), cards 1-8 = basic energies
    valid_deck = [22] + [1] * 59  # 60 cards, 1 basic pokemon, no ACE SPEC, basic energies exempt from 4-copy rule

    results = run_qa_checks(
        shard_dir=shard_dir,
        meta_path=tmp_path / "meta.parquet",
        vocab=vocab,
        fixed_deck=valid_deck,
        all_episode_deck_cards=[1, 2, 1, 2],
    )

    assert results["coverage_pass"] is True
    assert results["n_variable_length"] == 0
    assert results["label_sanity_pass"] is True
    assert results["won_count"] == 3
    assert results["lost_count"] == 2
    assert results["deck_legality_pass"] is True
    assert "balance_sel_ctx" in results
