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
    m = Policy(D=64, heads=4, layers=1, ff=128, n_all_cards=10,
               all_card_feat=torch.zeros(10, 94))
    assert m.pointer.card is m.embed.card

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
    src = Policy(D=64, heads=4, layers=1, ff=128, n_all_cards=10,
                 all_card_feat=torch.zeros(10, 94))
    sd = {k: v for k, v in src.state_dict().items()
          if not (k.startswith("pointer.card.") or "card_emb.mlp" in k)}
    for k in sd:
        if k.startswith("embed.card.mlp"):
            sd[k] = torch.full_like(sd[k], 0.1234)

    dst = Policy(D=64, heads=4, layers=1, ff=128, n_all_cards=10,
                 all_card_feat=torch.zeros(10, 94))
    dst.load_state_dict(sd, strict=False)

    assert (dst.embed.card.mlp[0].weight == 0.1234).all()
    assert torch.equal(dst.pointer.card.mlp[0].weight,
                       dst.embed.card.mlp[0].weight)


# ── Ensemble MCTS support: _policy_evaluate_leaf must use forward(), ───────
# not the Policy-specific low-level API (_encode/pointer/value), so that
# EnsemblePolicy — which has none of those methods — can serve as the MCTS
# leaf evaluator.


def _policy_evaluate_leaf_ast():
    """Parse the function source so assertions move with edits."""
    import ast
    import inspect

    src = inspect.getsource(search_infer._policy_evaluate_leaf)
    # De-dent: inspect.getsource returns the function body indented; the
    # try/except block is easier to parse as module-level.
    lines = src.splitlines()
    if lines and lines[0].startswith((" ", "\t")):
        indent = len(lines[0]) - len(lines[0].lstrip())
        src = "\n".join(line[indent:] for line in lines)
    return ast.parse(src)


def test_policy_evaluate_leaf_uses_forward_not_low_level_api():
    """After ensemble-MCTS: _policy_evaluate_leaf calls policy(batch), not
    policy._encode() + policy.pointer() + policy.value()."""
    import ast

    tree = _policy_evaluate_leaf_ast()

    attr_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                attr_calls.append(node.func.attr)

    # The old code had three Policy-specific attribute calls.
    assert "_encode" not in attr_calls, (
        "_policy_evaluate_leaf still calls policy._encode(); "
        "use policy(batch) so EnsemblePolicy works")
    assert "pointer" not in attr_calls, (
        "_policy_evaluate_leaf still calls policy.pointer(); "
        "EnsemblePolicy has no pointer attribute")
    # value-head access is now through forward()'s return tuple
    assert "value" not in attr_calls, (
        "_policy_evaluate_leaf still calls policy.value(); "
        "read value from forward()'s return tuple instead")


def test_policy_evaluate_leaf_calls_policy_as_callable():
    """The policy is invoked as ``policy(batch)``, which dispatches to
    Policy.forward() or EnsemblePolicy.forward() depending on type."""
    import ast

    tree = _policy_evaluate_leaf_ast()

    policy_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            policy_calls.append(node)

    # Find the ``policy(batch)`` call.
    batch_calls = [c for c in policy_calls
                   if len(c.args) >= 1
                   and isinstance(c.args[0], ast.Name)
                   and c.args[0].id == "batch"]
    assert batch_calls, (
        "_policy_evaluate_leaf must contain ``policy(batch)``; "
        "found calls: " + ", ".join(
            f"{c.func.id}({', '.join(a.id if isinstance(a, ast.Name) else '...' for a in c.args)})"
            for c in policy_calls))


def test_policy_evaluate_leaf_works_with_ensemble_like_object(monkeypatch):
    """An object with only forward() — no _encode/pointer/value — must work."""
    import json
    import sys
    import types

    import numpy as np

    # Mock _dict_to_batch to return batch-shaped torch tensors.
    # _policy_evaluate_leaf indexes opt_mask with [0], which fails on raw numpy.
    def _mock_dict_to_batch(feats, device):
        out = {}
        for k, v in feats.items():
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v).unsqueeze(0)
                out[k] = t.bool() if v.dtype == np.bool_ else t.float()
            else:
                out[k] = v
        return out

    monkeypatch.setattr(search_infer, "_dict_to_batch", _mock_dict_to_batch)

    # Mock model.featurizer.featurize so the lazy import inside
    # _policy_evaluate_leaf returns our stub.
    _fake_featurize = types.ModuleType("model.featurizer")
    _fake_featurize.featurize = lambda obs_dict, vocab, **kw: {
        "opt_mask": np.ones(3, dtype=bool),
        "stop_column": None,
    }
    _fake_model = types.ModuleType("model")
    _fake_model.featurizer = _fake_featurize
    # setitem, not raw assignment: a bare `sys.modules["model"] = ...` outlives
    # this test, and `model` is also the name of the real vendored Kaggle-bundle
    # package.  `test_vendored_sync` sorts after this file, so it imported this
    # stub instead and failed on the whole suite while passing in isolation --
    # a guard that only works when run alone is not a guard.
    monkeypatch.setitem(sys.modules, "model", _fake_model)
    monkeypatch.setitem(sys.modules, "model.featurizer", _fake_featurize)

    # An ensemble-like object: only __call__, no _encode/pointer/value.
    calls_log: list[str] = []

    class EnsembleLike:
        def __call__(self, batch):
            calls_log.append("forward")
            n_opt = int(batch["opt_mask"].sum())
            return (
                torch.zeros(1, n_opt),   # logits
                torch.zeros(1),           # value
                torch.zeros(1, 1),        # history_h
            )

    policy = EnsembleLike()
    obs_json = json.dumps({"select": {"option": [{}]}})

    priors, val = search_infer._policy_evaluate_leaf(
        obs_json, n_options=3, is_terminal=False,
        policy=policy, vocab={}, device="cpu",
    )

    assert "forward" in calls_log, (
        "policy.__call__ (forward) was not invoked; ensemble cannot work")
    assert len(priors) == 3, f"expected 3 priors, got {len(priors)}"
    assert isinstance(val, float)


def test_policy_evaluate_leaf_uses_policy_not_pointer_attribute():
    """The function body must not dereference ``policy.pointer`` or
    ``policy.embed`` — those are Policy-specific and would fail on
    EnsemblePolicy."""
    import ast
    import inspect

    src = inspect.getsource(search_infer._policy_evaluate_leaf)
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "policy":
                attr = node.attr
                assert attr not in ("pointer", "embed", "value", "_encode"), (
                    f"_policy_evaluate_leaf still references policy.{attr}; "
                    "EnsemblePolicy has none of these — use policy(batch) instead"
                )




# ── Who calls GameInitialize is a question, not an assumption ────────────
#
# `puct_init`'s last argument tells the Rust tree whether this process has
# already called `GameInitialize` on libcg.so.  Getting it wrong is fatal in
# both directions: a second call aborts the process, a skipped one drives an
# uninitialised engine.  It was hardcoded to 1 with the comment "Python always
# calls GameInitialize first" — true from the repo, where importing `cg.sim`
# does it, and false in the Kaggle bundle, which ships no `cg` package at all
# and falls back to a shim `to_observation_class`.

def test_host_initialized_is_derived_from_whether_cg_sim_was_imported(monkeypatch):
    import sys
    import types

    recorded: list[tuple] = []

    class _Lib:
        def puct_init(self, *args):
            recorded.append(args)
            return 0  # null handle -> greedy fallback; we only want the args

    monkeypatch.setattr(search_infer, "_load_search_lib", lambda: _Lib())
    monkeypatch.setattr(search_infer, "_register_basic_pokemon", lambda lib: None)
    monkeypatch.setattr(search_infer, "_greedy_action",
                        lambda *a, **k: {"indices": []})

    obs = {"select": {"maxCount": 1}}

    # The bundle: no cg package anywhere, so nothing has initialised the engine
    # and the Rust side must do it.
    monkeypatch.delitem(sys.modules, "cg.sim", raising=False)
    search_infer.mcts_search(obs, [], [], None, {}, None)
    assert recorded[-1][-1] == 0, (
        "no cg.sim in the process means no GameInitialize has run; the tree "
        "must call it or it drives an uninitialised engine")

    # The repo (live_eval, RL, tournament): cg.sim's import already did it.
    monkeypatch.setitem(sys.modules, "cg.sim", types.ModuleType("cg.sim"))
    search_infer.mcts_search(obs, [], [], None, {}, None)
    assert recorded[-1][-1] == 1, (
        "cg.sim imported means GameInitialize already ran; a second call "
        "aborts the process")

    assert len(recorded) == 2


# ── Decision time budget ──────────────────────────────────────────────────
#
# Kaggle gives one 600 s bank per game (`actTimeout: 0`), and a single call
# that exceeds what is left is a TIMEOUT, i.e. a loss.  These pin the two
# properties that matter: the budget must never bust the bank in a long game,
# and it must not collapse to nothing in one.


def test_no_budget_without_a_clock():
    """A caller off Kaggle's clock passes no remaining time and gets no
    deadline — the iteration ceiling applies instead."""
    assert search_infer.decision_time_budget(None) is None
    assert search_infer.decision_time_budget("not a number") is None


def test_first_decision_budget_is_the_p99_slice():
    """~588 s usable after import, spread over the p99 game length."""
    budget = search_infer.decision_time_budget(588.0, decisions_made=0)
    expected = (588.0 - search_infer.DECISION_BUDGET_RESERVE_S) / search_infer.DECISION_BUDGET_TOTAL
    assert budget == pytest.approx(expected)
    assert 2.5 < budget < 4.0, "p99 sizing should land near 3 s/decision"


def test_budget_grows_as_the_game_turns_out_short():
    """Spending less than budgeted leaves more for each remaining move, which
    is the point of reading `remaining` every call instead of dividing once."""
    early = search_infer.decision_time_budget(500.0, decisions_made=10)
    later = search_infer.decision_time_budget(500.0, decisions_made=100)
    assert later > early


def test_budget_is_clamped_at_both_ends():
    assert search_infer.decision_time_budget(1e9, 0) == search_infer.DECISION_BUDGET_MAX_S
    # Bank below the reserve: the raw slice is negative, the floor is not.
    assert search_infer.decision_time_budget(1.0, 500) == search_infer.DECISION_BUDGET_MIN_S


@pytest.mark.parametrize("n_decisions", [91, 166, 280, 600])
def test_a_full_game_never_busts_the_bank(n_decisions):
    """The safety property, simulated over real game lengths.

    91 = mean, 166 = p99, 280 = longest archetype-1 game in meta.parquet, and
    600 is well past anything measured.  Each decision is assumed to spend its
    whole budget, which is the worst case; `remaining` is decremented by it.
    """
    remaining = 600.0 - 12.0  # bank less the one-off import cost
    spent_at_all = 0
    for i in range(n_decisions):
        budget = search_infer.decision_time_budget(remaining, i)
        assert budget is not None
        assert budget > 0, f"decision {i} got a non-positive budget"
        if budget > search_infer.DECISION_BUDGET_MIN_S:
            spent_at_all += 1
        remaining -= budget
        assert remaining > 0, (
            f"bank exhausted after {i + 1}/{n_decisions} decisions — "
            f"a call with no bank left is a TIMEOUT, which is a loss")
    assert spent_at_all > 0, "budget never exceeded the floor; nothing was searched"


def test_the_deadline_stops_the_search_loop(monkeypatch):
    """A budget smaller than one iteration still expands once, then stops.

    Breaking *before* the first expand would leave `visit_counts` empty and
    silently take the greedy fallback — legal moves, no search.
    """
    expands = []

    class _Lib(_FakeLib):
        def puct_init(self, *a, **k):
            return 1234

        def puct_select(self, handle):
            # The real tree reports `tree_done` once the iteration ceiling is
            # reached; mirror that, so removing the deadline makes this test
            # fail with a count instead of hanging forever.
            if len(expands) >= 50:
                return b'{"tree_done": true}'
            return b'{"leaf_obs_json": "{}", "n_options": 2, "is_terminal": false, "player_role": 0}'

        def puct_expand(self, handle, priors, value):
            expands.append(value)
            return 0

        def puct_result(self, handle):
            return b'{"visit_counts": [[0, 1]]}'

        def puct_free(self, handle):
            return None

        def puct_free_result(self, ptr):
            return None

    lib = _Lib()
    monkeypatch.setattr(search_infer, "_load_search_lib", lambda: lib)
    monkeypatch.setattr(search_infer, "_register_basic_pokemon", lambda _l: None)
    monkeypatch.setattr(search_infer, "_read_and_free", lambda _l, raw: raw)
    monkeypatch.setattr(
        search_infer, "_policy_evaluate_leaf", lambda *a, **k: ([0.5, 0.5], 0.0))

    res = search_infer.mcts_search(
        {"select": {"minCount": 1, "maxCount": 1, "option": [{}, {}]}},
        fixed_deck=[1] * 60, opp_deck=[1] * 60, policy=None, vocab={},
        device="cpu", iterations=10_000, time_budget_s=0.0,
    )
    assert len(expands) == 1, (
        f"expected exactly one expansion before the deadline, got {len(expands)}")
    assert res.get("indices") == [0], "a one-node tree must still yield a move"
