"""Tests for ptcg_mine.sampling: select_days, deterministic_pick."""

import random

import pandas as pd

from ptcg_mine.sampling import deterministic_pick, select_days


def _manifest_df(dates):
    return pd.DataFrame(
        {
            "date": dates,
            "top_avg_score": [float(i) for i in range(len(dates))],
        }
    )


def test_select_days_returns_n_most_recent_by_date():
    dates = [
        "2026-06-16",
        "2026-06-17",
        "2026-06-18",
        "2026-06-19",
        "2026-06-20",
    ]
    df = _manifest_df(dates)
    result = select_days(df, 3)
    assert result == ["2026-06-18", "2026-06-19", "2026-06-20"]


def test_select_days_handles_unsorted_input():
    dates = ["2026-06-19", "2026-06-16", "2026-06-20", "2026-06-18", "2026-06-17"]
    df = _manifest_df(dates)
    result = select_days(df, 2)
    assert result == ["2026-06-19", "2026-06-20"]


def test_select_days_clamps_to_available_rows():
    df = _manifest_df(["2026-06-16", "2026-06-17"])
    result = select_days(df, 10)
    assert result == ["2026-06-16", "2026-06-17"]


def test_select_days_returns_strings():
    df = _manifest_df(["2026-06-16", "2026-06-17"])
    result = select_days(df, 1)
    assert all(isinstance(d, str) for d in result)


def test_deterministic_pick_reproducible():
    ids = [f"ep{i}" for i in range(50)]
    a = deterministic_pick(ids, 10, seed=42)
    b = deterministic_pick(ids, 10, seed=42)
    assert a == b


def test_deterministic_pick_order_independent():
    ids = [f"ep{i}" for i in range(50)]
    shuffled = ids[:]
    random.Random(1).shuffle(shuffled)
    a = deterministic_pick(ids, 10, seed=7)
    b = deterministic_pick(shuffled, 10, seed=7)
    assert a == b


def test_deterministic_pick_correct_count():
    ids = [f"ep{i}" for i in range(50)]
    result = deterministic_pick(ids, 10, seed=1)
    assert len(result) == 10
    assert len(set(result)) == 10
    assert set(result) <= set(ids)


def test_deterministic_pick_clamps_k_to_len():
    ids = [f"ep{i}" for i in range(3)]
    result = deterministic_pick(ids, 100, seed=1)
    assert sorted(result) == sorted(ids)


def test_deterministic_pick_different_seed_different_order():
    ids = [f"ep{i}" for i in range(50)]
    a = deterministic_pick(ids, 50, seed=1)
    b = deterministic_pick(ids, 50, seed=2)
    assert a != b


def test_deterministic_pick_empty_ids():
    assert deterministic_pick([], 5, seed=0) == []
