"""Feature dimensions are defined once, in ptcg_il.featurizer."""

import ast
import pathlib

import ptcg_il.featurizer as fz
import ptcg_il.model.cards as model_cards
import ptcg_il.model.embed as embed
import ptcg_il.model.pointer as pointer
import ptcg_mine.artifacts as artifacts
import ptcg_mine.cards as mine_cards

_PY_ROOT = pathlib.Path(fz.__file__).resolve().parent.parent


def test_card_dims_agree_everywhere():
    assert mine_cards.F_CARD == fz.F_CARD
    assert model_cards.F_CARD == fz.F_CARD
    assert artifacts.F_CARD == fz.F_CARD
    assert mine_cards.F_ATK == fz.F_ATK
    assert model_cards.F_ATK == fz.F_ATK


def test_model_dims_agree_with_featurizer():
    assert embed.F_POKE == fz.F_POKE
    assert embed.F_HAND == fz.F_HAND
    assert embed.F_SUM == fz.F_SUM
    assert embed.F_GLOBAL == fz.F_GLOBAL
    assert pointer.F_OPT == fz.F_OPT


def _assigned_names(path):
    """Top-level `NAME = <literal int>` assignments in a module."""
    tree = ast.parse(pathlib.Path(path).read_text())
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, int):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def test_only_the_featurizer_defines_the_dims():
    """A literal redefinition elsewhere is how three-of-four edits happen."""
    owned = {"F_CARD", "F_ATK", "F_POKE", "F_HAND", "F_SUM", "F_GLOBAL", "F_OPT"}
    offenders = {}
    checked = 0
    for rel in ("ptcg_mine/cards.py", "ptcg_mine/artifacts.py",
                "ptcg_il/model/cards.py", "ptcg_il/model/embed.py",
                "ptcg_il/model/pointer.py"):
        checked += 1
        clash = _assigned_names(_PY_ROOT / rel) & owned
        if clash:
            offenders[rel] = sorted(clash)
    assert checked == 5, "fixture drift: expected to examine 5 modules"
    assert offenders == {}, f"dims redefined outside ptcg_il.featurizer: {offenders}"
