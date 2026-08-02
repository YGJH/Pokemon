"""Tests for the submission agent's MCTS path (``ptcg_il.search_infer``).

Two failures motivated these, both of which presented as "MCTS unavailable":
the determinizer guessed a non-Basic card as the opponent's face-down active
and the engine refused the search root, and the warning printed exactly once
per reason so a single transient rejection read as total failure.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ptcg_il import search_infer
from ptcg_il.featurizer import CARD_FEAT_BASIC_COL
from ptcg_il.model.policy import Policy


# ── The Basic-Pokémon set handed to the Rust determinizer ────────────────

class _FakeLib:
    """Stands in for libptcg_search.so — records what it was handed."""

    def __init__(self, retval: int = 0):
        self.calls: list[bytes] = []
        self._retval = retval

    def puct_set_basic_pokemon(self, payload: bytes) -> int:
        self.calls.append(payload)
        return len(payload) if self._retval == 0 else self._retval


def _row(basic: float, stage1: float = 0.0, stage2: float = 0.0) -> np.ndarray:
    """A card static row with the stage flags at their real columns.

    Deliberately writes literal 9:12 rather than slicing from
    ``CARD_FEAT_BASIC_COL``: indexing the fixture by the same constant the code
    reads makes the test move with the bug and pass for any column value.  This
    pins ``ptcg_mine.cards.card_static_row``'s actual layout.
    """
    row = np.zeros(94, dtype=np.float32)
    row[9:12] = (basic, stage1, stage2)
    return row


def test_basic_column_constant_matches_the_card_row_layout():
    """``card_static_row`` writes ``row[9:12] = (basic, stage1, stage2)``."""
    assert CARD_FEAT_BASIC_COL == 9


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """search_infer caches the lib and the registration flag process-wide."""
    monkeypatch.setattr(search_infer, "_LIB", None, raising=False)
    monkeypatch.setattr(search_infer, "_BASICS_REGISTERED", False, raising=False)
    monkeypatch.setattr(search_infer, "_engine_card_features", None, raising=False)
    monkeypatch.setattr(search_infer, "_FALLBACK_WARNED", set(), raising=False)
    monkeypatch.setattr(search_infer, "_FALLBACK_COUNTS", {}, raising=False)


def test_registers_only_basic_pokemon(monkeypatch):
    """Anything but a Basic gets the root refused with SearchBegin error 2."""
    import json

    monkeypatch.setattr(search_infer, "_engine_card_features", {
        11: _row(1.0),                  # Basic
        12: _row(0.0, stage1=1.0),      # Stage 1 — a Pokémon, still illegal here
        13: _row(0.0, stage2=1.0),      # Stage 2
        14: _row(0.0),                  # Trainer / Energy
        15: _row(1.0),                  # Basic
    })
    lib = _FakeLib()
    search_infer._register_basic_pokemon(lib)

    assert len(lib.calls) == 1, "must register exactly once"
    assert json.loads(lib.calls[0].decode()) == [11, 15]


def test_registration_is_idempotent(monkeypatch):
    """Called on every decision; must not re-cross the FFI each time."""
    monkeypatch.setattr(search_infer, "_engine_card_features", {1: _row(1.0)})
    lib = _FakeLib()
    for _ in range(5):
        search_infer._register_basic_pokemon(lib)
    assert len(lib.calls) == 1


def test_no_card_features_registers_nothing(monkeypatch):
    """Without the feature table there is no set to assert; Rust keeps its
    previous whole-template behaviour rather than being handed an empty one."""
    lib = _FakeLib()
    search_infer._register_basic_pokemon(lib)
    assert lib.calls == []


def test_registration_failure_is_not_fatal(monkeypatch):
    """A decision must never be lost to this bookkeeping."""
    monkeypatch.setattr(search_infer, "_engine_card_features", {1: _row(1.0)})

    class _Boom:
        def puct_set_basic_pokemon(self, payload):
            raise OSError("symbol not found")

    search_infer._register_basic_pokemon(_Boom())  # must not raise
    assert search_infer._BASICS_REGISTERED is False


# ── The fallback warning ─────────────────────────────────────────────────

def test_fallback_warning_reports_a_count(capsys):
    """Printing strictly once made one transient rejection out of hundreds of
    decisions look like search never ran at all."""
    for _ in range(10):
        search_infer._warn_fallback("SearchBegin error code 2")
    out = capsys.readouterr().out

    assert out.count("MCTS fell back to greedy") == 2, (
        "expected the first occurrence and the 10th, got:\n" + out
    )
    assert "10 times so far" in out
    assert search_infer._FALLBACK_COUNTS["SearchBegin error code 2"] == 10


def test_fallback_warning_is_per_reason(capsys):
    search_infer._warn_fallback("reason A")
    search_infer._warn_fallback("reason B")
    out = capsys.readouterr().out
    assert "reason A" in out and "reason B" in out


# ── The tied CardEncoder the submission loader must not cry wolf over ────

def test_card_encoder_is_shared_across_heads():
    """``build_submission``'s missing-weight check compares parameter identity
    and reads ``named_parameters(remove_duplicate=False)``.  Both halves of
    that depend on this tie; if Policy ever stops sharing, the check silently
    becomes a no-op and a real missing weight ships unnoticed."""
    m = Policy(D=64, heads=4, layers=1, ff=128, n_opp_arch=2, n_all_cards=10,
               all_card_feat=torch.zeros(10, 94))
    assert m.pointer.card is m.embed.card
    assert m.belief_heads.card_emb is m.embed.card

    dedup = dict(m.named_parameters())
    full = dict(m.named_parameters(remove_duplicate=False))
    assert "pointer.card.mlp.0.weight" not in dedup, (
        "named_parameters() must de-duplicate — the loader's filter exists "
        "precisely because the alias is absent here"
    )
    assert "pointer.card.mlp.0.weight" in full
    assert full["pointer.card.mlp.0.weight"] is full["embed.card.mlp.0.weight"]


def test_loading_without_alias_keys_still_populates_them():
    """The packaged state dict omits the aliases (EMA de-duplicates); loading
    ``embed.card.*`` must be enough, or the submission would run a randomly
    initialised card encoder inside the pointer head."""
    src = Policy(D=64, heads=4, layers=1, ff=128, n_opp_arch=2, n_all_cards=10,
                 all_card_feat=torch.zeros(10, 94))
    sd = {k: v for k, v in src.state_dict().items()
          if not (k.startswith("pointer.card.") or "card_emb.mlp" in k)}
    for k in sd:
        if k.startswith("embed.card.mlp"):
            sd[k] = torch.full_like(sd[k], 0.1234)

    dst = Policy(D=64, heads=4, layers=1, ff=128, n_opp_arch=2, n_all_cards=10,
                 all_card_feat=torch.zeros(10, 94))
    dst.load_state_dict(sd, strict=False)

    assert (dst.embed.card.mlp[0].weight == 0.1234).all()
    assert torch.equal(dst.pointer.card.mlp[0].weight,
                       dst.embed.card.mlp[0].weight)


# ── Tier 3: the belief `deck` head is engine-id indexed ──────────────────
#
# `belief_labels.deck_counts_dense` writes `out[cid]` for a raw engine card id
# and `cli` sizes the head from the engine feature matrix, so position i of
# `deck` means engine card i — there is no vocab hop.  Tier 3 used to route it
# through `_index_to_id(vocab)`, a ~311-entry table against a 1268-wide head.


def test_deck_from_distribution_treats_none_as_engine_ids():
    probs = np.zeros(1268, dtype=np.float64)
    probs[900] = 0.5   # past the end of any plausible vocab
    probs[1200] = 0.5
    deck = search_infer._deck_from_distribution(probs, None, deck_size=60)
    assert len(deck) == 60, "engine-id positions must not be dropped"
    assert set(deck) == {900, 1200}


def test_deck_from_distribution_still_maps_when_given_a_table():
    """The explicit-table path is unchanged — None is opt-in, not a rewrite."""
    probs = np.zeros(4, dtype=np.float64)
    probs[2] = 1.0
    deck = search_infer._deck_from_distribution(probs, [-1, -1, 77, 88],
                                                deck_size=60)
    assert set(deck) == {77}


def test_tier3_keeps_full_deck_for_high_engine_ids(monkeypatch):
    """End-to-end through `predict_opponent_deck`, with vocab far smaller than
    the head.  The old code returned a short deck of relabelled cards."""
    n_all_cards = 1268
    probs = np.zeros(n_all_cards, dtype=np.float32)
    for cid in (700, 850, 1100):
        probs[cid] = 1 / 3

    class _FakePolicy:
        def belief_logits(self, batch):
            # No confident arch head, no reps -> falls through to Tier 3.
            return {"arch": torch.zeros(1, 1), "deck": torch.tensor(probs)[None]}

    # `predict_opponent_deck` does `from model.featurizer import featurize` —
    # a name that only exists inside the packaged bundle.  Unstubbed it raises
    # ImportError into the function's bare `except Exception`, so the test
    # would pass vacuously on an empty deck for the wrong reason.
    import sys
    import types
    pkg = types.ModuleType("model")
    pkg.__path__ = []
    feat_mod = types.ModuleType("model.featurizer")
    feat_mod.featurize = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "model", pkg)
    monkeypatch.setitem(sys.modules, "model.featurizer", feat_mod)
    monkeypatch.setattr(search_infer, "_dict_to_batch", lambda *a, **k: {})

    vocab = {"size": 311, "id_to_index": {str(i): i for i in range(311)}}
    deck = search_infer.predict_opponent_deck(
        {}, _FakePolicy(), vocab, {"self_ids": [], "clusters": []}, "cpu")

    assert len(deck) == 60, f"got {len(deck)} cards: {sorted(set(deck))}"
    assert set(deck) == {700, 850, 1100}
