"""Tests for the submission builder's checkpoint/artifact pairing guard.

The failure being guarded is silent end to end: a checkpoint trained against an
earlier mining run has a vocab of the same *length* but a different id→index
assignment, so the packaged agent loads, imports, plays every game to the end,
and loses nearly all of them without raising anywhere.  `save_checkpoint` pins
`vocab_sha1`/`archetypes_sha1` exactly so this is detectable; the builder used
to print them and compare nothing.
"""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_submission",
    Path(__file__).resolve().parent.parent / "scripts" / "build_submission.py",
)


@pytest.fixture(scope="module")
def bs():
    mod = importlib.util.module_from_spec(_SPEC)
    sys.modules["build_submission"] = mod
    _SPEC.loader.exec_module(mod)
    return mod


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "vocab.json").write_text('{"size": 296}')
    (d / "archetypes.json").write_text('{"self_ids": [0, 1]}')
    return d


def _sha12(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _exec_width_fragment(bs, cfg, row_width, template="MAIN_PY_TEMPLATE_GREEDY"):
    """Exec just the card-feature-table statements of *template*.

    Pulls the top-level statements that build ``_all_card_feat`` out of the
    real template — so the test reads the shipped source, not a paraphrase of
    it — and runs them against a stub checkpoint config and feature table.
    """
    import ast

    import numpy as np
    import torch

    src = getattr(bs, template)
    tree = ast.parse(src)
    wanted = ("_all_card_feat", "_max_cid", "_card_feat_dim", "_table_dim")
    chunks = []
    for node in tree.body:
        seg = ast.get_source_segment(src, node) or ""
        if any(w in seg for w in wanted) and "Policy(" not in seg and "torch.load" not in seg:
            chunks.append(seg)
    assert chunks, f"{template} no longer builds _all_card_feat at top level"

    ns = {
        "np": np,
        "torch": torch,
        "sys": sys,
        "_cfg": cfg,
        "_engine_card_features": {
            7: np.zeros(row_width, dtype=np.float32),
            11: np.ones(row_width, dtype=np.float32),
        },
    }
    exec(compile("\n".join(chunks), "main.py", "exec"), ns)
    return ns


def _record(data_dir, **over):
    rec = {
        "archetype_self": 0,
        "vocab_sha1": _sha12(data_dir / "vocab.json"),
        "archetypes_sha1": _sha12(data_dir / "archetypes.json"),
    }
    rec.update(over)
    return rec


def test_matching_pair_is_accepted(bs, data_dir):
    bs.check_artifact_pairing(_record(data_dir), data_dir)


def test_each_pin_explains_its_own_consequence(bs, data_dir):
    """The two pins mean different things; one shared sentence described neither.

    The old text promised "scrambled card identities" for the vocab pin, which
    requires learned card-id embeddings — `model/cards.py` has none.
    """
    rec = _record(data_dir, vocab_sha1="deadbeefdead",
                  archetypes_sha1="deadbeefdead")
    with pytest.raises(SystemExit) as exc:
        bs.check_artifact_pairing(rec, data_dir, mcts=True)
    msg = str(exc.value)

    assert bs._PIN_CONSEQUENCE["vocab.json"] in msg
    assert bs._PIN_CONSEQUENCE["archetypes.json"] in msg
    assert (bs._PIN_CONSEQUENCE["vocab.json"]
            != bs._PIN_CONSEQUENCE["archetypes.json"])
    assert "scrambled card identities" not in msg


def test_only_the_mismatched_pin_is_explained(bs, data_dir, capsys):
    """A vocab-only mismatch must not lecture about the belief arch head."""
    rec = _record(data_dir, vocab_sha1="deadbeefdead")
    bs.check_artifact_pairing(rec, data_dir, mcts=True)
    msg = capsys.readouterr().out
    assert bs._PIN_CONSEQUENCE["vocab.json"] in msg
    assert bs._PIN_CONSEQUENCE["archetypes.json"] not in msg


# ── Severity follows what the bundle actually reads ──────────────────────
#
# The guard used to abort on either pin for every build type.  That refused to
# package bundles whose correctness the mismatch could not affect: no build
# routes card identity through the vocab, and a --no-mcts bundle ships no
# archetypes.json at all.  Severity is now derived from `_PIN_CONSUMED_BY`.


def test_vocab_only_mismatch_never_aborts(bs, data_dir, capsys):
    """The real case: checkpoints_a2 pinned d43bb2e2 while data/ held f9470f51.

    Still reported — it is real evidence of a mismatched pair — but the vocab
    is read by neither build type, so it cannot make a bundle wrong.
    """
    rec = _record(data_dir, vocab_sha1="d43bb2e28112")
    for mcts in (True, False):
        bs.check_artifact_pairing(rec, data_dir, mcts=mcts)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "vocab.json" in out and "d43bb2e28112" in out
    assert not bs._PIN_CONSUMED_BY["vocab.json"], (
        "this test asserts the vocab is consumed by nothing")


def test_stale_archetypes_aborts_only_for_mcts(bs, data_dir, capsys):
    rec = _record(data_dir, archetypes_sha1="c6b5e71baaaa")

    with pytest.raises(SystemExit) as exc:
        bs.check_artifact_pairing(rec, data_dir, mcts=True)
    assert "READS this file" in str(exc.value)

    # --no-mcts ships no archetypes.json and never calls the belief heads.
    bs.check_artifact_pairing(rec, data_dir, mcts=False)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "does not read this file" in out


def test_force_downgrades_a_fatal_mismatch_to_a_warning(bs, data_dir, capsys):
    rec = _record(data_dir, archetypes_sha1="c6b5e71baaaa")
    bs.check_artifact_pairing(rec, data_dir, force=True, mcts=True)
    assert "WARNING" in capsys.readouterr().out


def test_default_build_type_is_mcts(bs, data_dir):
    """`mcts` defaults to True, so an omitted argument keeps the strict path."""
    rec = _record(data_dir, archetypes_sha1="c6b5e71baaaa")
    with pytest.raises(SystemExit):
        bs.check_artifact_pairing(rec, data_dir)


def test_truncated_pin_matches_full_digest(bs, data_dir):
    """ptcg_il.deck._sha1 stores 12 chars; comparing against the full 40-char
    digest made every checkpoint look stale, including good ones."""
    full = hashlib.sha1((data_dir / "vocab.json").read_bytes()).hexdigest()
    assert len(full) == 40
    bs.check_artifact_pairing(_record(data_dir, vocab_sha1=full[:12]), data_dir)


def test_unlabelled_checkpoint_is_left_to_the_deck_check(bs, data_dir):
    """build_data_files already refuses a checkpoint with no deck record; this
    guard must not raise a second, more confusing error first."""
    bs.check_artifact_pairing(None, data_dir)


def test_absent_pins_are_not_treated_as_mismatches(bs, data_dir):
    """Checkpoints predating the deck record carry no SHAs to compare."""
    bs.check_artifact_pairing({"archetype_self": 0}, data_dir)


# ── --no-mcts bundle ──────────────────────────────────────────────────────
#
# The greedy bundle ships no libptcg_search.so, so anything left in it that
# still reaches for MCTS degrades to `search_infer`'s own greedy fallback:
# same moves, but the failure prints as a warning and reads like search that
# merely underperformed.  These assert the two templates and the file lists
# actually diverge.


def _imports_and_calls(source: str) -> tuple[set[str], set[str]]:
    """(imported names, called names) for a template, parsed rather than grepped.

    Both main.py templates are checked for what they do and do not reference,
    and their prose mentions the very symbols being asserted absent — so a
    substring search over the template text reports false positives.
    """
    import ast

    tree = ast.parse(source)
    imported, called = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Call):
            fn = node.func
            called.add(fn.id if isinstance(fn, ast.Name)
                       else fn.attr if isinstance(fn, ast.Attribute) else "")
    return imported, called


def test_mcts_main_py_uses_the_prior_predictor(bs):
    imported, called = _imports_and_calls(bs.MAIN_PY_TEMPLATE)

    assert "OpponentDeckPredictor" in imported
    assert "OpponentDeckPredictor" in called
    assert "mcts_search" in called
    assert "observe" in called, "the predictor must be fed each observation"
    assert "template" in called
    assert "reset" in called, "a new game must clear the previous opponent"
    # The retired path
    assert "predict_opponent_deck" not in called
    assert "extract_opp_visible_cards" not in called
    assert "belief_logits" not in called


def test_mcts_package_bundles_deck_prior(bs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = Path(bs.__file__).resolve().parents[1]
    bs.build_model_package(src, tmp_path / "sub", mcts=True)
    assert (tmp_path / "sub" / "model" / "deck_prior.py").exists()
    assert not (tmp_path / "sub" / "model" / "belief_posterior.py").exists()


def test_bundled_deck_prior_has_no_unrewritten_imports(bs, tmp_path, monkeypatch):
    """A missed rewrite rule fails only at agent runtime, on Kaggle.

    Checks actual import statements, not a bare substring: deck_prior.py's
    docstring legitimately cross-references ``ptcg_il.belief_labels`` in prose
    (comparing this module's behavior to a validated definition elsewhere),
    which is not an unrewritten import and must not fail this guard.
    """
    import re

    monkeypatch.chdir(tmp_path)
    src = Path(bs.__file__).resolve().parents[1]
    bs.build_model_package(src, tmp_path / "sub", mcts=True)
    for name in ("deck_prior.py", "search_infer.py"):
        text = (tmp_path / "sub" / "model" / name).read_text()
        assert not re.search(r"(?m)^\s*(from|import)\s+ptcg_il\b", text), (
            f"{name} has an unrewritten ptcg_il import")
        assert not re.search(r"(?m)^\s*(from|import)\s+ptcg_rl\b", text), (
            f"{name} has an unrewritten ptcg_rl import")


def test_greedy_main_py_calls_no_search(bs):
    imported, called = _imports_and_calls(bs.MAIN_PY_TEMPLATE_GREEDY)

    assert not any("search_infer" in m for m in imported), imported
    assert not any("deck_prior" in m for m in imported), imported
    assert "mcts_search" not in called
    assert "OpponentDeckPredictor" not in called
    assert "select_multi" in called, "multi-select decisions still need the AR path"
    assert "featurize" in called


def test_build_main_py_selects_the_template(bs, tmp_path):
    bs.build_main_py(tmp_path, mcts=False)
    assert "mcts_search" not in (tmp_path / "main.py").read_text()
    bs.build_main_py(tmp_path, mcts=True)
    assert "mcts_search" in (tmp_path / "main.py").read_text()


def test_greedy_package_omits_the_search_modules(bs, tmp_path, monkeypatch):
    """EXTRA_FILES (search_infer.py, belief_posterior.py) must not be bundled."""
    src = tmp_path / "src"
    model_src = src / "python" / "ptcg_il" / "model"
    model_src.mkdir(parents=True)
    for fname in bs.MODEL_FILES:
        (model_src / fname).write_text("# stub\n")
    (src / "python" / "ptcg_il" / "ref_map.py").write_text("# stub\n")
    (src / "python" / "ptcg_il" / "featurizer.py").write_text("# stub\n")
    n_extra = 0
    for rel, _dest in bs.EXTRA_FILES:
        p = src / "python" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub\n")
        n_extra += 1
    assert n_extra, "EXTRA_FILES is empty — this test would pass vacuously"

    bs.build_model_package(src, tmp_path / "greedy", mcts=False)
    bs.build_model_package(src, tmp_path / "with_mcts", mcts=True)

    for _rel, dest in bs.EXTRA_FILES:
        assert not (tmp_path / "greedy" / "model" / dest).exists(), dest
        assert (tmp_path / "with_mcts" / "model" / dest).exists(), dest
    # The shared model files still ship in both.
    for fname in bs.MODEL_FILES:
        assert (tmp_path / "greedy" / "model" / fname).exists(), fname


# ── The Kaggle runner does not define __file__ ───────────────────────────
#
# `kaggle_environments.agent` execs main.py into a namespace without
# ``__file__``, so a template that reads it raises NameError.  `_find_libcg`
# is only reached from `agent()`, i.e. on the first *decision*, which is why
# `verify_model_imports`' `import main` (which does bind ``__file__``) passed
# a bundle that died on move one of every game.


@pytest.mark.parametrize("template", ["MAIN_PY_TEMPLATE", "MAIN_PY_TEMPLATE_GREEDY"])
def test_main_py_never_reads_bare_dunder_file(bs, template):
    """`_cgsim.__file__` is fine — an imported module has one.  A bare read is not."""
    import ast

    tree = ast.parse(getattr(bs, template))
    bare = [n for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id == "__file__"]
    assert not bare, (
        f"{template} reads __file__ as a bare name at line(s) "
        f"{[n.lineno for n in bare]}; the Kaggle runner does not define it")


def test_find_libcg_runs_without_dunder_file(bs, tmp_path, monkeypatch):
    """Exec the real template fragments the way Kaggle does: no ``__file__``."""
    import ast

    src = bs.MAIN_PY_TEMPLATE
    tree = ast.parse(src)
    wanted = {"_AGENT_DIR", "_libcg_name", "_find_libcg"}
    chunks = []
    for node in tree.body:
        names = ({t.id for t in node.targets if isinstance(t, ast.Name)}
                 if isinstance(node, ast.Assign)
                 else {node.name} if isinstance(node, ast.FunctionDef) else set())
        if names & wanted:
            chunks.append(ast.get_source_segment(src, node))
            wanted -= names
    assert not wanted, f"template no longer defines {wanted}"

    # cwd is the fallback when neither /kaggle_simulations/agent nor __file__
    # exists, so put a plausible engine there and confirm it is found.
    monkeypatch.chdir(tmp_path)
    ns = {"os": __import__("os"), "__name__": "main"}
    assert "__file__" not in ns
    exec(compile("\n".join(chunks), "main.py", "exec"), ns)

    name = ns["_libcg_name"]()
    assert not (tmp_path / name).exists()
    assert ns["_find_libcg"]() == name, "missing engine should degrade, not raise"

    (tmp_path / name).write_bytes(b"\x7fELF")
    assert ns["_find_libcg"]() == str(tmp_path / name)


def _exec_find_libcg(bs, ns_extra=None):
    """Exec the template's engine-discovery fragments into a fresh namespace."""
    import ast

    src = bs.MAIN_PY_TEMPLATE
    tree = ast.parse(src)
    wanted = {"_AGENT_DIR", "_libcg_name", "_find_libcg"}
    chunks = []
    for node in tree.body:
        names = ({t.id for t in node.targets if isinstance(t, ast.Name)}
                 if isinstance(node, ast.Assign)
                 else {node.name} if isinstance(node, ast.FunctionDef) else set())
        if names & wanted:
            chunks.append(ast.get_source_segment(src, node))
            wanted -= names
    assert not wanted, f"template no longer defines {wanted}"
    ns = {"os": __import__("os"), "__name__": "main"}
    ns.update(ns_extra or {})
    exec(compile("\n".join(chunks), "main.py", "exec"), ns)
    return ns


def test_find_libcg_finds_the_engine_the_builder_bundles(bs, tmp_path, monkeypatch):
    """The builder puts the engine in ``data/`` — next to libptcg_search.so —
    so discovery has to look there.

    It did not, and that is the whole bug: on Kaggle the agent process has no
    importable ``cg`` package (the ``import cg.sim`` branch never fires), so
    every candidate missed, `_find_libcg` returned the bare name, `puct_init`
    got a path dlopen could not resolve, and every decision fell to greedy.
    """
    monkeypatch.chdir(tmp_path)
    ns = _exec_find_libcg(bs)
    name = ns["_libcg_name"]()

    data = tmp_path / "data"
    data.mkdir()
    (data / name).write_bytes(b"\x7fELF")
    assert ns["_find_libcg"]() == str(data / name), (
        "the engine sits in data/ in the bundle; discovery must look there")


# ── Kaggle picks the entry point by position, not by name ────────────────
#
# `kaggle_environments.agent.get_last_callable` execs main.py and returns
# ``[v for v in env.values() if callable(v)][-1]`` — the *last* callable bound
# at module level.  Nothing looks for the name `agent`.  `MAIN_PY_TEMPLATE`
# defined `_libcg_name`/`_find_libcg` after `agent`, so Kaggle called
# `_find_libcg`, whose `co_argcount` is 0 (`agent.py` truncates the args to the
# callee's arity, so it does not even raise).  It returned a *path string*,
# which failed the `"type": "array"` action schema, so the action became the
# default `[]` and cabt's interpreter reported
# ``Player 1's deck does not have 60 cards.`` — with both players INVALID at
# step 0, an empty stdout, and no traceback anywhere.
#
# `verify_model_imports` cannot see this: it reaches the agent by name.


def _exec_module_level_defs(bs, template):
    """Bind the template's module-level `def`/`class` statements, in order.

    Only definitions — executing the real top level would load torch and the
    packaged checkpoints.  That is enough to reproduce Kaggle's choice, because
    a `def` binds its name at exactly the point it appears, which is the order
    `env.values()` preserves.  Annotations are deferred so a definition needing
    a name the fragment never binds still compiles.
    """
    src = getattr(bs, template)
    chunks = [
        ast.get_source_segment(src, node)
        for node in ast.parse(src).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert chunks, f"{template} defines nothing at module level"

    ns = {"__name__": "main"}
    exec(compile("from __future__ import annotations\n" + "\n".join(chunks),
                 "main.py", "exec"), ns)
    return ns


@pytest.mark.parametrize("template", ["MAIN_PY_TEMPLATE", "MAIN_PY_TEMPLATE_GREEDY"])
def test_agent_is_the_last_callable_kaggle_sees(bs, template):
    ns = _exec_module_level_defs(bs, template)

    callables = [v for v in ns.values() if callable(v)]  # kaggle agent.py:64
    assert callables, f"{template} bound no callable"
    picked = callables[-1]

    assert getattr(picked, "__name__", None) == "agent", (
        f"{template}: Kaggle would call {picked.__name__}(), not agent(). "
        f"Move it above `def agent` — the runner takes the last callable bound "
        f"at module level, and re-assigning `agent = agent` does not help "
        f"(a dict keeps a re-bound key in its original position).")


@pytest.mark.parametrize("template", ["MAIN_PY_TEMPLATE", "MAIN_PY_TEMPLATE_GREEDY"])
def test_no_callable_is_bound_after_agent(bs, template):
    """The `def` check above misses `f = lambda: ...` and `f = partial(...)`,
    which bind a callable just as well and would take the slot back."""
    src = getattr(bs, template)
    body = ast.parse(src).body

    agent_at = [i for i, n in enumerate(body)
                if isinstance(n, ast.FunctionDef) and n.name == "agent"]
    assert len(agent_at) == 1, f"{template} defines agent {len(agent_at)} times"

    trailing = [
        t.id
        for node in body[agent_at[0] + 1:] if isinstance(node, ast.Assign)
        for t in node.targets if isinstance(t, ast.Name)
        if isinstance(node.value, (ast.Lambda, ast.Call))
    ]
    assert not trailing, (
        f"{template} binds {trailing} after `agent`; if any of them is callable "
        f"Kaggle calls it instead of the agent")


# ── Feature widths belong to the checkpoint, not to a literal ────────────
#
# `F_CARD` moved 94 → 212 when the ability/attack keyword features landed.
# Every other architecture dim in the template is read from the checkpoint's
# `config` (`D`, `heads`, `layers`, `ff`, `n_all_cards`), and the
# packaged `.npy` tables are rebuilt from `ptcg_mine.cards` at packaging time,
# so both ends moved on their own — a hardcoded width is the one thing that
# cannot.  It fails at `verify_model_imports`, which is the good case; the same
# literal being *too large* would broadcast a short row into a padded slot and
# ship a silently wrong agent.


@pytest.mark.parametrize("template", ["MAIN_PY_TEMPLATE", "MAIN_PY_TEMPLATE_GREEDY"])
def test_card_feature_width_is_not_hardcoded(bs, template):
    """`_all_card_feat = torch.zeros(_max_cid + 1, <width>)` must derive <width>."""
    import ast

    tree = ast.parse(getattr(bs, template))
    allocs = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_all_card_feat" for t in n.targets)
        and isinstance(n.value, ast.Call)
    ]
    assert allocs, f"{template} no longer allocates _all_card_feat via a call"
    for call in allocs:
        width = call.args[-1]
        assert not isinstance(width, ast.Constant), (
            f"{template} hardcodes the card feature width as {width.value!r} at "
            f"line {width.lineno}; derive it from the checkpoint's "
            f"config['feat_dims']['F_CARD'] like every other dim on those lines")


def test_packaged_width_matches_the_trained_model(bs):
    """The width the template derives is the one the checkpoint was trained at.

    Guards the direction `verify_model_imports` cannot see: a table and a model
    that disagree by *broadcastable* shapes load without raising.
    """
    from ptcg_il.featurizer import F_CARD

    ns = _exec_width_fragment(bs, cfg={"feat_dims": {"F_CARD": F_CARD}}, row_width=F_CARD)
    assert ns["_all_card_feat"].shape[1] == F_CARD


def test_width_falls_back_to_the_table_for_unpinned_checkpoints(bs):
    """Checkpoints predating the `feat_dims` pin carry no width; the table does."""
    ns = _exec_width_fragment(bs, cfg={}, row_width=94)
    assert ns["_all_card_feat"].shape[1] == 94


@pytest.mark.parametrize("row_width", [94, 1])
def test_table_disagreeing_with_the_checkpoint_is_fatal(bs, row_width):
    """A table that disagrees with the checkpoint must abort, not broadcast.

    ``row_width=94`` is the stale-table case, which torch would reject anyway
    (opaquely).  ``row_width=1`` is why the check has to exist: assigning a
    1-wide row into a 212-wide slot **broadcasts silently**, filling every
    feature of every card with one value, and the agent then loads, imports,
    plays every game to the end and loses.
    """
    with pytest.raises(SystemExit):
        _exec_width_fragment(bs, cfg={"feat_dims": {"F_CARD": 212}}, row_width=row_width)


# ── Static feature tables ────────────────────────────────────────────────
# Cards and attacks reach the model as *ids*: `Policy._gather_card_feats` turns
# every `*_card_feat` key back into features from a dense table, one per kind
# (`featurizer.CARD_FEAT_SOURCES`).  A template that builds only the card table
# constructs, loads its weights and imports fine, then raises on the first real
# decision -- where main.py's greedy-inference `except` downgrades it to a
# warning and plays the first legal option for the rest of every game.

_AGENT_TEMPLATES = ["MAIN_PY_TEMPLATE", "MAIN_PY_TEMPLATE_GREEDY"]

#: The names whose top-level statements build the static tables.  Pulled out of
#: the real template, so the test reads the shipped source rather than a
#: paraphrase of it.
_TABLE_NAMES = ("_all_card_feat", "_all_attack_feat", "_max_cid", "_max_aid",
                "_card_feat_dim", "_attack_feat_dim", "_table_dim",
                "_attack_row_dim")


def _exec_table_fragment(bs, template):
    """Exec the static-table statements of *template* against stub engine data."""
    import ast

    import numpy as np
    import torch

    from ptcg_il.featurizer import F_ATK, F_CARD

    src = getattr(bs, template)
    tree = ast.parse(src)
    chunks = []
    for node in tree.body:
        seg = ast.get_source_segment(src, node) or ""
        if any(w in seg for w in _TABLE_NAMES) and "Policy(" not in seg and "torch.load" not in seg:
            chunks.append(seg)
    assert chunks, f"{template} no longer builds the static tables at top level"

    ns = {
        "np": np,
        "torch": torch,
        "sys": sys,
        "_cfg": {"D": 32, "heads": 2, "layers": 1, "ff": 64,
                 "feat_dims": {"F_CARD": F_CARD, "F_ATK": F_ATK}},
        "_engine_card_features": {
            7: np.zeros(F_CARD, dtype=np.float32),
            11: np.ones(F_CARD, dtype=np.float32),
        },
        "_engine_attack_features": {
            3: np.zeros(F_ATK, dtype=np.float32),
            9: np.ones(F_ATK, dtype=np.float32),
        },
    }
    exec(compile("\n".join(chunks), "main.py", "exec"), ns)
    return ns


def _template_call(bs, template, predicate):
    """Source of the first call node in *template* matching *predicate*."""
    src = getattr(bs, template)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and predicate(node):
            return ast.get_source_segment(src, node)
    return None


@pytest.mark.parametrize("template", _AGENT_TEMPLATES)
def test_agent_policy_can_gather_every_card_feature(bs, template):
    """The Policy each template builds must be able to gather every feature key.

    Built exactly as the template builds it -- its own table statements, its own
    `Policy(...)` call -- then asked for the gather that every forward begins
    with.  Constructing it proves nothing: the tables are non-persistent
    buffers, so a policy missing one loads every weight and only raises once the
    ids it has no table for reach `_gather_card_feats`.

    Driven off `CARD_FEAT_SOURCES` rather than a list of keys, so a feature key
    added against a third table fails here instead of after submission.
    """
    import torch

    from ptcg_il.featurizer import CARD_FEAT_SOURCES, F_ATK, F_CARD
    from ptcg_il.model.policy import Policy

    ns = _exec_table_fragment(bs, template)
    call = _template_call(
        bs, template,
        lambda n: isinstance(n.func, ast.Name) and n.func.id == "Policy")
    assert call, f"{template} no longer constructs Policy directly"

    ns["Policy"] = Policy
    policy = eval(call, ns)  # noqa: S307 - the template's own source

    x = {id_key: torch.ones(1, 4, dtype=torch.long)
         for id_key, _kind in CARD_FEAT_SOURCES.values()}
    out = policy._gather_card_feats(x)

    widths = {"card": F_CARD, "attack": F_ATK}
    assert set(CARD_FEAT_SOURCES) <= set(out), "gather dropped a feature key"
    for feat_key, (_id_key, kind) in CARD_FEAT_SOURCES.items():
        assert out[feat_key].shape[-1] == widths[kind], (
            f"{template} builds a {kind} table {out[feat_key].shape[-1]} wide, "
            f"not {widths[kind]}")


def _template_func(bs, template, name):
    """Source of the top-level function *name* defined in *template*."""
    src = getattr(bs, template)
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def _greedy_agent_ns(bs, featurize):
    """`agent` from the greedy template, wired to stubs, ready to call."""
    src = _template_func(bs, "MAIN_PY_TEMPLATE_GREEDY", "agent")
    assert src, "greedy template no longer defines agent()"

    def _unused(*a, **kw):
        raise AssertionError("should not be reached")

    ns = {
        "to_observation_class": lambda d: type("O", (), {"select": d.get("select")})(),
        "featurize": featurize,
        "_vocab": {},
        "_engine_card_features": {},
        "_engine_attack_features": {},
        "_evolution_map": None,
        "_fixed_deck": [11, 22, 33],
        "_to_batch": lambda f: f,
        "_model": _unused,
        "select_multi": _unused,
        "torch": __import__("torch"),
        "_legal": lambda idx, n, lo, hi: list(idx),
    }
    exec(compile(src, "main.py", "exec"), ns)
    return ns


def test_inference_failure_is_raised_not_swallowed(bs):
    """A policy that cannot answer must fail the run, not play the first option.

    The handler this replaces turned a dead policy -- a missing static table, a
    shape mismatch, a renamed key -- into one line on stderr and `minCount`
    legal options for every decision of every game after it.  Nothing
    downstream distinguishes that from a policy that simply plays badly, so it
    survives a full experiment and is read as a training result.
    """
    def _boom(*a, **kw):
        raise RuntimeError("no attack static table")

    ns = _greedy_agent_ns(bs, _boom)
    obs = {"select": {"option": [{}, {}, {}], "minCount": 1, "maxCount": 1}}
    with pytest.raises(RuntimeError, match="no attack static table"):
        ns["agent"](obs)


def test_deck_selection_still_answers_before_any_inference(bs):
    """The `select is None` step must stay ahead of the model.

    It is the one decision with no options to score, so it must keep working
    without touching the featurizer -- otherwise raising on inference failure
    would also break deck submission.
    """
    def _never(*a, **kw):
        raise AssertionError("deck selection must not featurize")

    ns = _greedy_agent_ns(bs, _never)
    assert ns["agent"]({"select": None}) == [11, 22, 33]


def test_ensemble_members_are_given_both_static_tables(bs):
    """`EnsemblePolicy.from_checkpoints` builds every member, so it needs both.

    The members are `policy_from_config`'d inside it and never seen by the
    template again, so a table omitted here cannot be attached afterwards.
    """
    ns = _exec_table_fragment(bs, "MAIN_PY_TEMPLATE_GREEDY")
    call = _template_call(
        bs, "MAIN_PY_TEMPLATE_GREEDY",
        lambda n: isinstance(n.func, ast.Attribute)
        and n.func.attr == "from_checkpoints")
    assert call, "MAIN_PY_TEMPLATE_GREEDY no longer builds an EnsemblePolicy"

    seen = {}

    class _Recorder:
        @classmethod
        def from_checkpoints(cls, paths, all_card_feat=None, all_attack_feat=None,
                             device="cpu"):
            seen["card"] = all_card_feat
            seen["attack"] = all_attack_feat
            return cls()

    ns.update({"EnsemblePolicy": _Recorder, "_member_paths": [], "_device": "cpu"})
    eval(call, ns)  # noqa: S307 - the template's own source

    assert seen["card"] is ns["_all_card_feat"], "members get no card table"
    assert seen["attack"] is ns["_all_attack_feat"], (
        "members get no attack table; every ensemble decision then raises "
        "inside the agent's greedy-inference except clause and falls back to "
        "the first legal option")


def test_featurizer_imports_are_rewritten_for_the_bundle(bs):
    """model/*.py import dims from ptcg_il.featurizer; the bundle vendors it as
    model/featurizer.py.  Without a rewrite rule the bundle ships a literal
    `from ptcg_il.featurizer import ...`, which only fails after submission."""
    src = "from ptcg_il.featurizer import F_CARD, F_ATK\n"
    assert bs.rewrite_imports(src) == "from model.featurizer import F_CARD, F_ATK\n"


def test_module_object_import_form_is_rewritten(bs):
    """`from ptcg_il import featurizer as _fz` is the other way to spell it.

    Every rule matches `from ptcg_il.<mod> import ...`; `policy.py`'s
    `current_feature_dims` uses the module-object form, so the bundle shipped a
    literal `from ptcg_il import featurizer` and died on `ModuleNotFoundError`.
    """
    src = "    from ptcg_il import featurizer as _fz\n"
    assert bs.rewrite_imports(src) == "    from model import featurizer as _fz\n"


#: Imports of `ptcg_il` that are allowed to survive into the bundle, because
#: the code holding them is training-only and the greedy/MCTS agents never call
#: it.  `multiselect_ce` is the teacher-forced training loss; inference goes
#: through `select_multi`.  The import is lazy (function body, not module
#: level), so it costs nothing until called — and if it ever is, it raises
#: rather than answering wrong.  Anything added here needs the same argument.
_DEAD_IN_BUNDLE = {"ptcg_il.train.loop"}


def test_no_packaged_module_still_imports_ptcg_il(bs):
    """Whole-bundle backstop: no rewrite rule may be *missing*.

    Per-form rules only cover the forms someone remembered, and a form nobody
    remembered is invisible until the bundle runs.  This parses every file the
    packager rewrites and asserts no `ptcg_il` import survives — including ones
    inside function bodies, which `verify_model_imports` cannot reach (the same
    blind spot that let `_find_libcg`'s bare `__file__` ship).
    """
    import ast

    src_dir = Path(__file__).resolve().parent.parent
    model_dir = src_dir / "python" / "ptcg_il" / "model"
    checked, survivors = 0, []
    for fname in bs.MODEL_FILES:
        path = model_dir / fname
        if not path.exists():
            path = src_dir / "python" / "ptcg_il" / fname
        if not path.exists():
            continue
        checked += 1
        tree = ast.parse(bs.rewrite_imports(path.read_text()))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            else:
                continue
            for mod in mods:
                if mod.split(".")[0] == "ptcg_il" and mod not in _DEAD_IN_BUNDLE:
                    survivors.append(f"{fname}:{node.lineno} imports {mod}")
    assert checked, "examined no packaged modules; MODEL_FILES lookup is broken"
    assert not survivors, (
        "these imports survive into the bundle, which has no ptcg_il package:\n  "
        + "\n  ".join(survivors)
        + "\nAdd a REWRITE_RULES entry for the import form, or justify it in "
          "_DEAD_IN_BUNDLE.")


# ── Ensemble + MCTS support ───────────────────────────────────────────────
#
# Before the ensemble+MCTS change, MAIN_PY_TEMPLATE had no ensemble detection
# and `is_ensemble` forced `mcts=False`.  Now ensemble mode also supports MCTS,
# and the MCTS main.py template detects and loads EnsemblePolicy when the
# bundle carries an ensemble.json manifest.


def test_mcts_template_now_detects_ensemble(bs):
    """MAIN_PY_TEMPLATE must reference _ENSEMBLE_MANIFEST and EnsemblePolicy
    so that an ensemble+MCTS bundle loads the right model."""
    imported, called = _imports_and_calls(bs.MAIN_PY_TEMPLATE)

    assert "EnsemblePolicy" in imported, (
        "MAIN_PY_TEMPLATE must import EnsemblePolicy for ensemble+MCTS support")
    assert "OpponentDeckPredictor" in called, (
        "ensemble+MCTS must still use the opponent deck prior")
    assert "mcts_search" in called, (
        "ensemble+MCTS must still run tree search")


def test_mcts_template_ensemble_path_loads_from_checkpoints(bs):
    """The EnsemblePolicy.load path in MAIN_PY_TEMPLATE must be present."""
    # Parse the template to find the from_checkpoints call
    call = _template_call(
        bs, "MAIN_PY_TEMPLATE",
        lambda n: isinstance(n.func, ast.Attribute)
        and n.func.attr == "from_checkpoints")
    assert call, (
        "MAIN_PY_TEMPLATE must call EnsemblePolicy.from_checkpoints "
        "when an ensemble manifest is present")


def test_ensemble_mcts_package_bundles_search_modules(bs, tmp_path, monkeypatch):
    """Ensemble+MCTS must bundle search_infer.py, deck_prior.py — the files
    that the old ensemble=greedy-only code deliberately dropped."""
    import shutil

    monkeypatch.chdir(tmp_path)
    # Build a fake repo under tmp_path so we never touch real source files.
    # build_model_package reads from src/python/ptcg_il/model/ etc.
    src = tmp_path / "repo"
    model_src = src / "python" / "ptcg_il" / "model"
    model_src.mkdir(parents=True)
    for fname in bs.MODEL_FILES:
        (model_src / fname).write_text("# stub\n")
    (src / "python" / "ptcg_il" / "ref_map.py").write_text("# stub\n")
    (src / "python" / "ptcg_il" / "featurizer.py").write_text("# stub\n")
    # ensemble.py lives in ptcg_il/ directly (not model/)
    (src / "python" / "ptcg_il" / "ensemble.py").write_text("# stub\n")
    for rel, _dest in bs.EXTRA_FILES:
        p = src / "python" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub\n")

    # Ensemble + MCTS
    bs.build_model_package(src, tmp_path / "ens_mcts", mcts=True)
    for _rel, dest in bs.EXTRA_FILES:
        assert (tmp_path / "ens_mcts" / "model" / dest).exists(), (
            f"{dest} must be bundled for ensemble+MCTS")

    # Ensemble + greedy still drops them
    bs.build_model_package(src, tmp_path / "ens_greedy", mcts=False)
    for _rel, dest in bs.EXTRA_FILES:
        assert not (tmp_path / "ens_greedy" / "model" / dest).exists(), (
            f"{dest} must NOT be bundled for ensemble+greedy")


def test_ensemble_mcts_data_includes_archetypes(bs, tmp_path, monkeypatch):
    """Ensemble+MCTS builds must copy archetypes.json (the opponent prior needs it)."""
    monkeypatch.chdir(tmp_path)
    # An MCTS build now refuses to package without the engine, so the fake repo
    # has to carry one (see test_missing_engine_aborts_an_mcts_build).
    src = _fake_repo_with_engine(bs, tmp_path)

    data = tmp_path / "data"
    data.mkdir()
    (data / "vocab.json").write_text('{"size": 10}')
    archetypes = {
        "archetypes": [{"id": 0, "representative": list(range(60)), "frequency": 100}],
        "self_ids": [0],
        "opp_ids": [0],
        "fixed_deck": list(range(60)),
    }
    (data / "archetypes.json").write_text(json.dumps(archetypes))

    # Provide pre-built engine feature files so _build_engine_features_from_engine
    # is skipped (it writes to dst, not data, and the test doesn't need it).
    import numpy as np
    np.save(data / "engine_card_features.npy", {})
    np.save(data / "engine_attack_features.npy", {})
    np.save(data / "evolution_map.npy", {})

    # Mock _build_engine_features_from_engine so we don't need the real engine.
    monkeypatch.setattr(bs, "_build_engine_features_from_engine",
                        lambda src_dir, dst_dir: None)

    deck = list(range(60))

    # Ensemble + MCTS: must copy archetypes.json
    mcts_data = tmp_path / "mcts_data"
    bs.build_data_files(data, mcts_data, deck=deck, src_dir=src, mcts=True)
    assert (mcts_data / "archetypes.json").exists(), (
        "archetypes.json must be in the bundle for the opponent prior")

    # Ensemble + greedy: must NOT copy archetypes.json
    greedy_data = tmp_path / "greedy_data"
    bs.build_data_files(data, greedy_data, deck=deck, src_dir=src, mcts=False)
    assert not (greedy_data / "archetypes.json").exists(), (
        "archetypes.json must NOT be in a greedy bundle")


def _fake_repo_with_engine(bs, tmp_path, engine: bool = True):
    """A repo tree holding just what `build_data_files` reads."""
    src = tmp_path / "repo"
    if engine:
        cg = src.joinpath(*bs.ENGINE_DIR_PARTS)
        cg.mkdir(parents=True)
        (cg / "libcg.so").write_bytes(b"\x7fELF" + b"\0" * 64)
    return src


def test_mcts_bundle_carries_the_engine(bs, tmp_path, monkeypatch):
    """`libcg.so` must be copied into the bundle for an MCTS build.

    The Rust tree dlopens the engine by path, and the Kaggle agent process has
    no `cg` package to borrow one from — so an engine missing from the bundle
    means `puct_init` returns null and the agent plays greedy for the whole
    competition.  Nothing at build or import time notices, because the repo
    path this copies *from* exists locally.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bs, "_build_engine_features_from_engine",
                        lambda src_dir, dst_dir: None)
    src = _fake_repo_with_engine(bs, tmp_path)

    data = tmp_path / "data"
    data.mkdir()
    (data / "vocab.json").write_text('{"size": 10}')
    (data / "archetypes.json").write_text(json.dumps({
        "archetypes": [{"id": 0, "representative": list(range(60)), "frequency": 1}],
        "self_ids": [0], "opp_ids": [0], "fixed_deck": list(range(60)),
    }))
    deck = list(range(60))

    mcts_data = tmp_path / "mcts_data"
    bs.build_data_files(data, mcts_data, deck=deck, src_dir=src, mcts=True)
    assert (mcts_data / "libcg.so").exists(), (
        "MCTS bundles must carry libcg.so; the Rust tree cannot dlopen what "
        "is not there")

    # The greedy agent dlopens nothing, so it must not pay the 1.3 MB.
    greedy_data = tmp_path / "greedy_data"
    bs.build_data_files(data, greedy_data, deck=deck, src_dir=src, mcts=False)
    assert not (greedy_data / "libcg.so").exists(), (
        "a greedy bundle loads no engine; libcg.so must not be in it")


def test_missing_engine_aborts_an_mcts_build(bs, tmp_path, monkeypatch):
    """A missing engine must be fatal, not a warning.

    Unlike libptcg_search.so — which is absent whenever cargo is — the engine
    is vendored in the repo, so it can only be missing if the build is wrong.
    Warning and continuing is what shipped the greedy-fallback submission.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bs, "_build_engine_features_from_engine",
                        lambda src_dir, dst_dir: None)
    src = _fake_repo_with_engine(bs, tmp_path, engine=False)

    data = tmp_path / "data"
    data.mkdir()
    (data / "vocab.json").write_text('{"size": 10}')
    (data / "archetypes.json").write_text(json.dumps({
        "archetypes": [{"id": 0, "representative": list(range(60)), "frequency": 1}],
        "self_ids": [0], "opp_ids": [0], "fixed_deck": list(range(60)),
    }))

    with pytest.raises(FileNotFoundError, match="libcg.so"):
        bs.build_data_files(data, tmp_path / "out", deck=list(range(60)),
                            src_dir=src, mcts=True)


def test_build_main_py_ensemble_mcts_uses_mcts_template(bs, tmp_path):
    """When mcts=True, build_main_py writes the MCTS template (now with
    ensemble support), not the greedy one."""
    bs.build_main_py(tmp_path, mcts=True)
    text = (tmp_path / "main.py").read_text()
    assert "mcts_search" in text
    assert "OpponentDeckPredictor" in text
    assert "from model.search_infer import mcts_search" in text

    bs.build_main_py(tmp_path, mcts=False)
    text = (tmp_path / "main.py").read_text()
    assert "mcts_search" not in text
    assert "OpponentDeckPredictor" not in text


def test_greedy_template_still_has_no_search(bs):
    """MAIN_PY_TEMPLATE_GREEDY must NOT gain MCTS imports — it is the
    escape hatch for when search is unavailable."""
    imported, called = _imports_and_calls(bs.MAIN_PY_TEMPLATE_GREEDY)
    assert "mcts_search" not in called
    assert "OpponentDeckPredictor" not in called
    assert not any("search_infer" in m for m in imported), imported


# ── Deck-out guard (mechanism A/B) wiring ─────────────────────────────────


def test_greedy_template_wires_the_deck_guard(bs):
    """The greedy submission must mask guaranteed deck-out options.

    Mechanism A edits ``opt_mask`` before the forward pass (so both the
    single-select masked_fill and select_multi's picked-mask honor it);
    mechanism B re-ranks near-ties by deck delta in the single-select path.
    Both live in model/deck_guard.py so live_eval shares one source.
    """
    imported, _called = _imports_and_calls(bs.MAIN_PY_TEMPLATE_GREEDY)
    assert any("deck_guard" in m for m in imported), imported

    src = _template_func(bs, "MAIN_PY_TEMPLATE_GREEDY", "agent")
    assert "_guard.apply_mask" in src, "mechanism A is not applied in agent()"
    assert "_guard.pick" in src, "mechanism B is not applied in agent()"


def test_mcts_template_stays_unguarded(bs):
    """Scope decision: only the greedy build carries the guard."""
    imported, _called = _imports_and_calls(bs.MAIN_PY_TEMPLATE)
    assert not any("deck_guard" in m for m in imported), imported


def test_deck_guard_shipped_in_both_packages(bs, tmp_path):
    """deck_guard.py vendors verbatim (like ref_map.py) in greedy AND mcts."""
    src = tmp_path / "src"
    model_src = src / "python" / "ptcg_il" / "model"
    model_src.mkdir(parents=True)
    for fname in bs.MODEL_FILES:
        (model_src / fname).write_text("# stub\n")
    (src / "python" / "ptcg_il" / "ref_map.py").write_text("# stub\n")
    (src / "python" / "ptcg_il" / "featurizer.py").write_text("# stub\n")
    (src / "python" / "ptcg_il" / "deck_guard.py").write_text(
        "MARKER = 1\n")

    bs.build_model_package(src, tmp_path / "greedy", mcts=False)
    bs.build_model_package(src, tmp_path / "with_mcts", mcts=True)

    for build in ("greedy", "with_mcts"):
        out = tmp_path / build / "model" / "deck_guard.py"
        assert out.exists(), f"deck_guard.py missing from {build} build"
        assert "MARKER" in out.read_text(), "deck_guard.py was not copied verbatim"
