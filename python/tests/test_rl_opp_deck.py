"""`ptcg_rl.train`'s determinization templates.

The function this replaces constructed `OpponentDeckOracle(policy=None)`,
which raises inside `__init__`, and swallowed it in a bare `except Exception:
return []`.  Every one of K determinizations therefore ran on the Rust mirror
heuristic, silently, for the lifetime of the code.  These tests exist so that
cannot recur: a failure must raise, and the templates must actually differ.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ptcg_rl.train import _opp_deck_templates

ARCHETYPES = {
    "opp_ids": [0],
    "fixed_deck": [100] * 60,
    "archetypes": [
        {"id": 0, "representative": [100] * 60, "frequency": 40},
        {"id": 1, "representative": [200] * 60, "frequency": 30},
        {"id": 2, "representative": [300] * 60, "frequency": 30},
    ],
}


@pytest.fixture
def args(tmp_path):
    with open(tmp_path / "archetypes.json", "w") as f:
        json.dump(ARCHETYPES, f)
    return SimpleNamespace(data_dir=str(tmp_path))


def test_returns_k_full_decklists(args):
    decks = _opp_deck_templates(args, {}, [], k=8, seed=0)
    assert len(decks) == 8
    assert all(len(d) == 60 for d in decks)


def test_no_template_is_empty(args):
    """An empty list is the Rust mirror fallback — the thing being removed."""
    assert all(_opp_deck_templates(args, {}, [], k=8, seed=0))


def test_templates_vary_when_the_posterior_is_diffuse(args):
    """K identical worlds would collapse `mcts_k_determinizations` to 1 while
    leaving the config knob looking effective."""
    decks = _opp_deck_templates(args, {}, [], k=32, seed=0)
    assert len({tuple(d) for d in decks}) > 1


def test_observed_cards_steer_the_templates(args):
    decks = _opp_deck_templates(args, {}, [200] * 4, k=8, seed=0)
    assert {tuple(d) for d in decks} == {tuple([200] * 60)}


def test_the_same_seed_gives_the_same_worlds(args):
    a = _opp_deck_templates(args, {}, [], k=8, seed=3)
    b = _opp_deck_templates(args, {}, [], k=8, seed=3)
    assert a == b


def test_different_seeds_give_different_worlds(args):
    a = _opp_deck_templates(args, {}, [], k=16, seed=1)
    b = _opp_deck_templates(args, {}, [], k=16, seed=2)
    assert a != b


def test_a_missing_artifact_raises_rather_than_returning_mirrors(tmp_path):
    """The regression guard.  The predecessor's bare `except Exception` is
    exactly what hid a permanently-broken path behind plausible behaviour.
    """
    args = SimpleNamespace(data_dir=str(tmp_path / "nonexistent"))
    with pytest.raises(Exception):
        _opp_deck_templates(args, {}, [], k=8, seed=0)


def test_the_predictor_is_built_once_per_data_dir(args, monkeypatch):
    """It used to reload vocab.json and archetypes.json on every one of
    K x batch calls."""
    import ptcg_rl.train as train_mod

    train_mod._deck_predictor.cache_clear()   # the lru_cache is on this one
    calls = []
    real_open = open

    def counting_open(path, *a, **k):
        if "archetypes.json" in str(path):
            calls.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", counting_open)
    for seed in range(5):
        _opp_deck_templates(args, {}, [], k=2, seed=seed)
    assert len(calls) == 1, f"artifact re-read {len(calls)} times"
