"""Tests for ptcg_il.qa — QA gates (D.5)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_variable_length_none():
    meta = pd.DataFrame({
        "maxCount": [1, 1, 1],
        "minCount": [1, 1, 1],
    })
    result = check_variable_length_multi_select(meta)
    assert result["n_variable_length"] == 0
    assert result["variable_length_share"] == 0.0


def test_variable_length_some():
    meta = pd.DataFrame({
        "maxCount": [1, 3, 3, 3, 1],
        "minCount": [1, 1, 3, 1, 1],
    })
    result = check_variable_length_multi_select(meta)
    # 3 multi-select rows, 2 have minCount < maxCount
    assert result["n_variable_length"] == 2
    assert result["variable_length_share"] == pytest.approx(2 / 3)


def test_variable_length_all_same():
    meta = pd.DataFrame({
        "maxCount": [3, 3, 5, 5],
        "minCount": [3, 3, 5, 5],
    })
    result = check_variable_length_multi_select(meta)
    assert result["n_variable_length"] == 0


# ============================================================
# check_attachment_collision
# ============================================================


def test_attachment_collision_no_collisions(tmp_path):
    """Synthetic shard with attachment options that all differ."""
    S, O = 2, 64
    opt_type = np.zeros((S, O), dtype=np.int64)
    opt_src = np.full((S, O), -1, dtype=np.int64)
    opt_card = np.zeros((S, O), dtype=np.int64)
    opt_mask = np.zeros((S, O), dtype=bool)

    # Sample 0: two ENERGY options on different pokemon
    opt_type[0, 0] = 6  # ENERGY
    opt_type[0, 1] = 6
    opt_src[0, 0] = 1
    opt_src[0, 1] = 7
    opt_card[0, 0] = 42
    opt_card[0, 1] = 42
    opt_mask[0, :2] = True

    # Sample 1: CARD attachment
    opt_type[1, 0] = 3  # CARD
    opt_src[1, 0] = 1
    opt_card[1, 0] = 99
    opt_mask[1, :1] = True

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez_compressed(shard_dir / "train-00000.npz",
                        opt_type=opt_type, opt_src_idx=opt_src,
                        opt_card_id=opt_card, opt_mask=opt_mask)

    meta = pd.DataFrame({"shard": ["train-00000.npz"] * S, "row": [0, 1]})
    result = check_attachment_collision(shard_dir, meta)
    assert result["n_collision_options"] == 0
    assert result["n_attachment_options"] == 3


def test_attachment_collision_with_collisions(tmp_path):
    """Two ENERGY options on the same pokemon with the same card id."""
    S, O = 1, 64
    opt_type = np.zeros((S, O), dtype=np.int64)
    opt_src = np.full((S, O), -1, dtype=np.int64)
    opt_card = np.zeros((S, O), dtype=np.int64)
    opt_mask = np.zeros((S, O), dtype=bool)

    # Two identical options: same src, same card id
    opt_type[0, 0] = 6
    opt_type[0, 1] = 6
    opt_src[0, 0] = 1
    opt_src[0, 1] = 1  # same source!
    opt_card[0, 0] = 9
    opt_card[0, 1] = 9  # same card!
    opt_mask[0, :2] = True

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez_compressed(shard_dir / "train-00000.npz",
                        opt_type=opt_type, opt_src_idx=opt_src,
                        opt_card_id=opt_card, opt_mask=opt_mask)

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


def test_reference_roundtrip_skips_without_obs_dir(tmp_path):
    """Gate skips (returns None) when no obs_dir is provided."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    # Create empty shard
    S, O = 1, 64
    opt_src = np.full((S, O), -1, dtype=np.int64)
    opt_card = np.zeros((S, O), dtype=np.int64)
    opt_mask = np.zeros((S, O), dtype=bool)
    np.savez_compressed(shard_dir / "train-00000.npz",
                        opt_src_idx=opt_src, opt_card_id=opt_card, opt_mask=opt_mask)

    passed, details = check_reference_roundtrip(shard_dir, obs_dir=None)
    assert passed is None
    assert details["skipped"] is True


def test_reference_roundtrip_skips_without_shard_dir():
    passed, details = check_reference_roundtrip(shard_dir=None, obs_dir=None)
    assert passed is None
    assert details["skipped"] is True


def test_reference_roundtrip_validates_correctly(tmp_path):
    """Gate passes when opt_src_idx / opt_card_id resolve correctly."""
    import json

    # Create observation dicts
    obs_dir = tmp_path / "observations"
    obs_dir.mkdir()
    obs = {
        "yourIndex": 0,
        "players": [
            {
                "hand": [{"cardId": 42}, {"cardId": 99}],
                "bench": [{"cardId": 7}],
                "active": {"cardId": 100},
                "discard": [{"cardId": 55}, {"cardId": 66}],
            },
            {"hand": [], "bench": [], "active": None, "discard": []},
        ],
    }
    with open(obs_dir / "train-00000_obs.jsonl", "w") as f:
        json.dump(obs, f)
        f.write("\n")

    # Create shard with one sample: opt_src_idx pointing to valid positions
    S, O = 1, 64
    opt_src = np.full((S, O), -1, dtype=np.int64)
    opt_card = np.full((S, O), -1, dtype=np.int64)
    opt_mask = np.zeros((S, O), dtype=bool)

    # idx 0 = hand[0] = cardId 42
    opt_src[0, 0] = 0
    opt_card[0, 0] = 42
    opt_mask[0, 0] = True
    # idx 1 = hand[1] = cardId 99
    opt_src[0, 1] = 1
    opt_card[0, 1] = 99
    opt_mask[0, 1] = True
    # idx 2 = bench[0] = cardId 7
    opt_src[0, 2] = 2
    opt_card[0, 2] = 7
    opt_mask[0, 2] = True
    # idx 3 = active = cardId 100
    opt_src[0, 3] = 3
    opt_card[0, 3] = 100
    opt_mask[0, 3] = True

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez_compressed(shard_dir / "train-00000.npz",
                        opt_src_idx=opt_src, opt_card_id=opt_card, opt_mask=opt_mask)

    passed, details = check_reference_roundtrip(shard_dir, obs_dir)
    assert passed is True
    assert details["checked"] == 1
    assert details["errors"] == 0


def test_reference_roundtrip_detects_mismatch(tmp_path):
    """Gate fails when opt_card_id doesn't match the card at opt_src_idx."""
    import json

    obs_dir = tmp_path / "observations"
    obs_dir.mkdir()
    obs = {
        "yourIndex": 0,
        "players": [
            {
                "hand": [{"cardId": 42}],
                "bench": [],
                "active": None,
                "discard": [],
            },
            {"hand": [], "bench": [], "active": None, "discard": []},
        ],
    }
    with open(obs_dir / "train-00000_obs.jsonl", "w") as f:
        json.dump(obs, f)
        f.write("\n")

    S, O = 1, 64
    opt_src = np.full((S, O), -1, dtype=np.int64)
    opt_card = np.full((S, O), -1, dtype=np.int64)
    opt_mask = np.zeros((S, O), dtype=bool)

    # src_idx 0 points to hand[0] = cardId 42, but opt_card_id says 999 — MISMATCH
    opt_src[0, 0] = 0
    opt_card[0, 0] = 999
    opt_mask[0, 0] = True

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez_compressed(shard_dir / "train-00000.npz",
                        opt_src_idx=opt_src, opt_card_id=opt_card, opt_mask=opt_mask)

    passed, details = check_reference_roundtrip(shard_dir, obs_dir)
    assert passed is False
    assert details["errors"] == 1


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
