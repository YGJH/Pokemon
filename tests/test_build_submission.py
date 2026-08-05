"""Tests for the submission builder's checkpoint/artifact pairing guard.

The failure being guarded is silent end to end: a checkpoint trained against an
earlier mining run has a vocab of the same *length* but a different id→index
assignment, so the packaged agent loads, imports, plays every game to the end,
and loses nearly all of them without raising anywhere.  `save_checkpoint` pins
`vocab_sha1`/`archetypes_sha1` exactly so this is detectable; the builder used
to print them and compare nothing.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

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


def test_greedy_main_py_calls_no_search(bs):
    """Parsed, not grepped: the template's prose mentions the search it drops."""
    import ast

    tree = ast.parse(bs.MAIN_PY_TEMPLATE_GREEDY)
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

    assert not any("search_infer" in m for m in imported), imported
    assert "mcts_search" not in called
    assert "predict_opponent_deck" not in called
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


# ── Feature widths belong to the checkpoint, not to a literal ────────────
#
# `F_CARD` moved 94 → 212 when the ability/attack keyword features landed.
# Every other architecture dim in the template is read from the checkpoint's
# `config` (`D`, `heads`, `layers`, `ff`, `n_opp_arch`, `n_all_cards`), and the
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
