# Card-Effect Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the IL policy access to what cards actually *do* — currently all 61 Supporters share one identical feature vector — by deriving effect-keyword, playability, and threat features at mining time.

**Architecture:** A new `ptcg_mine/keywords.py` turns oracle text (`CardData.skills[].text`, `Attack.text`) into a frozen 29-column binary multihot. That block is appended to `attack_static_row` (14→43) and `card_static_row` (94→212), which propagates to all nine `*_card_feat` tensors for free because every consumer already reads the `engine_card_features.npy` mining artifact. Three smaller deterministic blocks follow: hand playability flags (`F_HAND` 2→7), KO-pressure scalars (`F_GLOBAL` 93→97), and option damage preview (`F_OPT` 6→8). Shard card features move to float16 to pay for the size increase.

**Tech Stack:** Python 3.11, uv, NumPy, PyTorch, pytest. Engine access via the bundled `cg` package wrapping `libcg.so`.

**Spec:** `docs/superpowers/specs/2026-08-04-card-effect-features-design.md`

## Global Constraints

- **Python 3.11 via uv.** Every command is `uv run …`. Never call `pip` or bare `python`.
- **Run from `python/`.** `ptcg_mine/` and `ptcg_il/` live under `python/`; `pyproject.toml`'s `pythonpath` applies to pytest only.
- **pytest needs `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`** (a ROS install on this machine hijacks plugin autoload).
- **The two test directories must run separately.** `tests/` and `python/tests/` share basenames; running both in one invocation fails collection with "import file mismatch".
- **`ptcg_mine` must not import PyTorch.** It may import `ptcg_il.featurizer` (pure NumPy).
- **`ptcg_il/featurizer.py` must not import `ptcg_mine`.** It is vendored into the Kaggle bundle where `ptcg_mine` does not exist.
- **`KEYWORDS` order is frozen and append-only.** The tuple index *is* the feature column.
- **Fixed divisors only, never z-scoring.** All-zero rows are the PAD sentinel and masks read it.
- **Tests that count occurrences must fail on zero examined.** Validate new guards by mutation: break the thing deliberately, confirm the test goes red.
- **Final dimensions:** `K_EFFECT = 29`, `F_ATK = 43`, `F_CARD = 212`, `F_HAND = 7`, `F_GLOBAL = 97`, `F_OPT = 8`.

---

### Task 1: `ptcg_mine/keywords.py`

**Files:**
- Create: `python/ptcg_mine/keywords.py`
- Test: `python/tests/test_keywords.py`

**Interfaces:**
- Consumes: nothing (leaf module; `re` and `numpy` only).
- Produces:
  - `KEYWORDS: tuple[tuple[str, str], ...]` — 29 `(name, pattern_source)` pairs.
  - `KEYWORD_NAMES: tuple[str, ...]` — 29 names, index == feature column.
  - `K_EFFECT: int` — 29.
  - `effect_keyword_row(texts: Iterable[str]) -> np.ndarray` — `float32[29]`.
  - `ability_keyword_row(card) -> np.ndarray` — over `card.skills[].text`.
  - `attack_keyword_row(attack) -> np.ndarray` — over `attack.text`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_keywords.py`:

```python
"""Effect-keyword extraction from card and attack oracle text."""

import numpy as np
import pytest

from ptcg_mine import keywords as kw


class _Skill:
    def __init__(self, text):
        self.name = "s"
        self.text = text


class _Card:
    def __init__(self, *texts):
        self.skills = [_Skill(t) for t in texts]


class _Attack:
    def __init__(self, text):
        self.text = text


def test_k_effect_matches_list_length():
    assert kw.K_EFFECT == len(kw.KEYWORDS) == len(kw.KEYWORD_NAMES) == 29


def test_keyword_order_is_pinned():
    """KEYWORDS index IS the feature column. Inserting in the middle silently
    repoints every later column in every trained checkpoint."""
    assert kw.KEYWORD_NAMES[:6] == (
        "draw", "search_deck", "deck_look",
        "discard_own", "discard_opp", "hand_disrupt",
    )
    assert kw.KEYWORD_NAMES[-1] == "recover_discard"


def test_row_is_float32_binary_of_correct_width():
    row = kw.effect_keyword_row(["Draw 3 cards."])
    assert row.shape == (29,)
    assert row.dtype == np.float32
    assert set(np.unique(row)) <= {0.0, 1.0}


def test_empty_text_gives_all_zero_row():
    assert not kw.effect_keyword_row([]).any()
    assert not kw.effect_keyword_row(["", None]).any()


def test_matching_is_case_insensitive():
    """Oracle text capitalises the leading verb. A case-sensitive `draw \\d+ card`
    misses 'Draw 3 cards.' — i.e. misses Cheren entirely."""
    i = kw.KEYWORD_NAMES.index("draw")
    assert kw.effect_keyword_row(["Draw 3 cards."])[i] == 1.0
    assert kw.effect_keyword_row(["draw 3 cards."])[i] == 1.0


@pytest.mark.parametrize("name,positive,negative", [
    ("draw", "Draw 3 cards.", "Discard your hand."),
    ("search_deck", "Search your deck for a Trainer card.", "Look at the top 7 cards of your deck."),
    ("deck_look", "Look at the top 7 cards of your deck.", "Search your deck for a Trainer card."),
    ("gust", "Switch in 1 of your opponent's Benched Pokemon to the Active Spot.",
             "Switch this Pokemon with 1 of your Benched Pokemon."),
    ("switch_own", "Switch this Pokemon with 1 of your Benched Pokemon.",
                   "Switch in 1 of your opponent's Benched Pokemon to the Active Spot."),
    ("heal", "Heal 70 damage from your Active Pokemon.", "Draw 3 cards."),
    ("status_offensive", "Your opponent's Active Pokemon is now Poisoned.",
                         "Your opponent's Active Pokemon is now Confused."),
    ("status_lock", "Your opponent's Active Pokemon is now Confused.",
                    "Your opponent's Active Pokemon is now Poisoned."),
    ("coin_flip", "Flip a coin. If heads, this attack does 30 more damage.",
                  "This attack does 30 more damage."),
    ("ability_lock", "Pokemon in play have no Abilities.", "Draw 3 cards."),
    ("once_per_turn", "Once during your turn, you may draw a card.", "Draw 3 cards."),
    ("recover_discard", "Put a Pokemon from your discard pile into your hand.",
                        "Put a Pokemon from your deck into your hand."),
])
def test_keyword_positive_and_near_miss(name, positive, negative):
    i = kw.KEYWORD_NAMES.index(name)
    assert kw.effect_keyword_row([positive])[i] == 1.0, f"{name} missed a positive"
    assert kw.effect_keyword_row([negative])[i] == 0.0, f"{name} fired on a near-miss"


def test_ability_row_reads_every_skill():
    card = _Card("Draw 3 cards.", "Heal 20 damage from this Pokemon.")
    row = kw.ability_keyword_row(card)
    assert row[kw.KEYWORD_NAMES.index("draw")] == 1.0
    assert row[kw.KEYWORD_NAMES.index("heal")] == 1.0


def test_card_with_no_skills_is_all_zero():
    assert not kw.ability_keyword_row(_Card()).any()


def test_attack_row_reads_attack_text():
    row = kw.attack_keyword_row(_Attack("Flip a coin. If heads, your opponent's Active Pokemon is now Paralyzed."))
    assert row[kw.KEYWORD_NAMES.index("coin_flip")] == 1.0
    assert row[kw.KEYWORD_NAMES.index("status_lock")] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_keywords.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'ptcg_mine.keywords'`.

- [ ] **Step 3: Write the implementation**

Create `python/ptcg_mine/keywords.py`. The patterns below are validated against
all 1267 engine cards — do not rewrite them from scratch:

```python
"""Effect keywords mined from card and attack oracle text.

``card_static_row`` is otherwise entirely numeric, so two cards with the same HP,
type and attack damage are *bit-identical* to the model — and the policy has no
learned card-id embedding to fall back on.  Measured on the real engine tables,
all 61 Supporters share one feature row: ``Boss's Orders`` is indistinguishable
from ``Cheren``.  This module turns the text those cards do carry into a fixed
binary multihot so the encoder can tell them apart.

**The tuple index is the feature column.**  ``KEYWORDS`` is therefore frozen and
append-only: inserting a keyword in the middle silently repoints every later
column of every trained checkpoint, exactly like renumbering archetype cluster
ids.  Retire a keyword by leaving its slot in place as a dead always-zero column.

Patterns are matched case-insensitively.  This is load-bearing, not cosmetic:
oracle text capitalises the leading verb, so a case-sensitive ``draw \\d+ card``
misses ``"Draw 3 cards."``  A prototype without ``re.IGNORECASE`` left 33 of the
61 Supporters with an all-zero row.
"""

import re
from typing import Iterable

import numpy as np

#: ``(name, pattern)`` pairs.  Index == feature column.  APPEND ONLY.
KEYWORDS: tuple[tuple[str, str], ...] = (
    ("draw",             r"draw \d+ (?:more )?card|draw a card|draw cards|draw that many|draw up to|draw cards until"),
    ("search_deck",      r"search your deck"),
    ("deck_look",        r"look at the top \d+ card|look at the top card"),
    ("discard_own",      r"discard your hand|discard (?:a|an|another|\d+|all|up to|the top|that|this|those|your|the other)\b|discard it\b|discard them\b"),
    ("discard_opp",      r"your opponent discards|opponent[’']?s? hand.{0,60}discard|discard .{0,60}from (?:\d+ of )?your opponent"),
    ("hand_disrupt",     r"shuffles? their hand into their deck|shuffle your hand into your deck|opponent[’']?s? hand.{0,40}(?:shuffle|reveal)"),
    ("gust",             r"switch in \d+ of your opponent|switch out your opponent|opponent.{0,40}benched pok.mon to the active"),
    ("switch_own",       r"switch (?:this|your) (?:pok.mon|active)|switch \d+ of your"),
    ("heal",             r"\bheal\b|remove.{0,25}damage counter"),
    ("place_damage",     r"put \d+ damage counter|put damage counter|place \d+ damage counter"),
    ("bench_damage",     r"benched pok.mon"),
    ("bench_accel",      r"onto your bench|put .{0,40}onto (?:your|that) bench"),
    ("evolve_effect",    r"to evolve it|evolve.{0,30}during your first turn|devolve"),
    ("status_offensive", r"\bpoisoned\b|\bburned\b"),
    ("status_lock",      r"\basleep\b|\bparalyzed\b|\bconfused\b"),
    ("energy_accel",     r"attach (?:a|an|\d+|up to|basic|the other|that|it|them)\b.{0,90}(?:energy|pok.mon)|attach .{0,30}energy card.{0,60}(?:to|onto)"),
    ("energy_deny",      r"discard.{0,60}energy from (?:\d+ of )?your opponent|opponent.{0,80}discard.{0,30}energy|move an energy|discard .{0,30}special energy"),
    ("prevent_damage",   r"prevent all damage|takes? no damage|prevent all effects|damage done to this pok.mon.{0,40}reduced|-\d+ damage from attacks|do(?:es)? \d+ less damage"),
    ("damage_scaling",   r"do(?:es)? \d+ more damage|damage for each|does \d+ damage times|\d+ more damage"),
    ("coin_flip",        r"flip \d+ coins|flip a coin"),
    ("ability_lock",     r"have no abilities|can[’']?t use.{0,25}abilit"),
    ("prize_effect",     r"prize card"),
    ("retreat_effect",   r"retreat cost|retreating|retreat for"),
    ("once_per_turn",    r"once during your turn"),
    ("target_pokemon",   r"(?:search|look at|reveal|put|shuffle)[^.]{0,80}\bpok.mon\b"),
    ("target_energy",    r"(?:search|look at|reveal|put|attach)[^.]{0,80}energy card"),
    ("target_trainer",   r"\btrainer card|\bsupporter card|\bitem card|\bstadium card|pok.mon tool"),
    ("to_deck",          r"into your deck|on top of (?:it|your deck)|bottom of your deck|back into your deck"),
    ("recover_discard",  r"from your discard pile"),
)

KEYWORD_NAMES: tuple[str, ...] = tuple(name for name, _ in KEYWORDS)
K_EFFECT: int = len(KEYWORDS)

_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for _, pattern in KEYWORDS
)

assert len(KEYWORD_NAMES) == len(set(KEYWORD_NAMES)), "duplicate keyword name"


def effect_keyword_row(texts: Iterable[str]) -> np.ndarray:
    """float32[K_EFFECT] binary multihot over the concatenation of *texts*.

    Binary rather than a count: a card that draws twice is not twice the draw
    card, and a count would need a normalizer no other column in the row uses.
    """
    blob = "\n".join(t for t in texts if t)
    row = np.zeros(K_EFFECT, dtype=np.float32)
    if not blob:
        return row
    for i, pattern in enumerate(_PATTERNS):
        if pattern.search(blob):
            row[i] = 1.0
    return row


def ability_keyword_row(card) -> np.ndarray:
    """float32[K_EFFECT] over every ``card.skills[].text``."""
    return effect_keyword_row(s.text for s in (getattr(card, "skills", None) or []))


def attack_keyword_row(attack) -> np.ndarray:
    """float32[K_EFFECT] over ``attack.text``."""
    return effect_keyword_row([getattr(attack, "text", "") or ""])
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_keywords.py -v
```

Expected: all PASS.

- [ ] **Step 5: Mutation-check the order guard**

Temporarily swap the first two entries of `KEYWORDS`. Re-run
`test_keyword_order_is_pinned` and confirm it FAILS. Revert the swap and confirm
it passes again. A guard that does not go red is not a guard.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_mine/keywords.py python/tests/test_keywords.py
git commit -m "feat(mine): effect-keyword extraction from card and attack text"
```

---

### Task 2: Single source of truth for feature dimensions

Pure refactor. **No dimension values change in this task** — this exists so that
Task 4's one-line edit propagates everywhere instead of needing four edits that
can be made in three places.

**Files:**
- Modify: `python/ptcg_il/featurizer.py:52-58` (becomes the owner — unchanged values)
- Modify: `python/ptcg_mine/cards.py:31-32`
- Modify: `python/ptcg_mine/artifacts.py:12`
- Modify: `python/ptcg_il/model/cards.py:20-21`
- Modify: `python/ptcg_il/model/embed.py:22-25`
- Modify: `python/ptcg_il/model/pointer.py:17`
- Modify: `scripts/build_submission.py:29-43`
- Test: `python/tests/test_feature_dims.py` (create), `tests/test_build_submission.py` (extend)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `ptcg_il.featurizer` exports `F_CARD`, `F_ATK`, `F_POKE`, `F_HAND`, `F_SUM`, `F_GLOBAL`, `F_OPT` as the only definitions in the repo. Every other module imports them.

**Why the featurizer owns it, not `ptcg_mine/cards.py`:** `scripts/build_submission.py:642`
copies `ptcg_il/featurizer.py` into the Kaggle bundle, but **`ptcg_mine` is never
shipped** (line 873 imports it on the host at build time only). A
`from ptcg_mine.cards import F_CARD` inside `featurizer.py` would import fine in
the repo and `ImportError` inside the bundle — a failure that only surfaces after
submission.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_feature_dims.py`:

```python
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
```

Extend `tests/test_build_submission.py` with:

```python
def test_featurizer_imports_are_rewritten_for_the_bundle(bs):
    """model/*.py import dims from ptcg_il.featurizer; the bundle vendors it as
    model/featurizer.py.  Without a rewrite rule the bundle ships a literal
    `from ptcg_il.featurizer import ...`, which only fails after submission."""
    src = "from ptcg_il.featurizer import F_CARD, F_ATK\n"
    assert bs.rewrite_imports(src) == "from model.featurizer import F_CARD, F_ATK\n"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_feature_dims.py -v
cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_build_submission.py::test_featurizer_imports_are_rewritten_for_the_bundle -v
```

Expected: `test_only_the_featurizer_defines_the_dims` FAILS listing all five
modules; the bundle test FAILS because `rewrite_imports` returns the input
unchanged.

- [ ] **Step 3: Make the featurizer the owner**

In `python/ptcg_il/featurizer.py`, leave lines 52–58 as they are but add a
docstring comment above the block:

```python
# ============================================================
# Feature dims (A.1)
#
# THIS BLOCK IS THE ONLY DEFINITION IN THE REPO.  ptcg_mine.cards,
# ptcg_mine.artifacts, ptcg_il.model.{cards,embed,pointer} import from here.
#
# It lives in the featurizer, not in ptcg_mine.cards, because
# scripts/build_submission.py vendors this file into the Kaggle bundle while
# ptcg_mine is never shipped -- so importing ptcg_mine here would work in the
# repo and ImportError in the bundle.
# ============================================================
F_CARD = 94  # 52 base + 3 attacks × 14
F_ATK = 14
F_POKE = 26
F_HAND = 2
F_SUM = 11
F_GLOBAL = 93
F_OPT = 6
```

- [ ] **Step 4: Replace the five duplicate definitions with imports**

`python/ptcg_mine/cards.py` — delete lines 31–32 (`F_CARD = 94`, `F_ATK = 14`)
and add near the other imports:

```python
from ptcg_il.featurizer import F_ATK, F_CARD
```

`python/ptcg_mine/artifacts.py` — delete line 12 (`F_CARD = 94`) and add:

```python
from ptcg_il.featurizer import F_CARD
```

`python/ptcg_il/model/cards.py` — delete lines 20–21 and add:

```python
from ptcg_il.featurizer import F_ATK, F_CARD
```

`python/ptcg_il/model/embed.py` — delete lines 22–25 (`F_POKE`, `F_HAND`,
`F_SUM`, `F_GLOBAL`) and add:

```python
from ptcg_il.featurizer import F_GLOBAL, F_HAND, F_POKE, F_SUM
```

`python/ptcg_il/model/pointer.py` — delete line 17 (`F_OPT = 6`) and add:

```python
from ptcg_il.featurizer import F_OPT
```

- [ ] **Step 5: Add the bundle rewrite rule**

In `scripts/build_submission.py`, append to `REWRITE_RULES` (after the
`ref_map` rule at line 42):

```python
    # featurizer (copied into model/ so modules import it as model.featurizer).
    # model/{cards,embed,pointer}.py import the feature dims from it.
    (r"from ptcg_il\.featurizer import", r"from model.featurizer import"),
```

- [ ] **Step 6: Run the full test suites**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q
cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q
```

Expected: all PASS, including the two new files. No dimension changed, so every
pre-existing test must still pass unmodified. If any test fails, the refactor
introduced a circular import — check that nothing in `ptcg_il/featurizer.py`
imports `ptcg_mine`.

- [ ] **Step 7: Commit**

```bash
git add python/ptcg_il/featurizer.py python/ptcg_mine/cards.py \
        python/ptcg_mine/artifacts.py python/ptcg_il/model/cards.py \
        python/ptcg_il/model/embed.py python/ptcg_il/model/pointer.py \
        scripts/build_submission.py python/tests/test_feature_dims.py \
        tests/test_build_submission.py
git commit -m "refactor: single source of truth for feature dims in ptcg_il.featurizer"
```

---

### Task 3: Verify the weakness multiplier against the engine

The only unknown in the design. Weakness resolution lives in `libcg.so` and
nothing in the Python bindings states whether it is ×2 (standard TCG) or a flat
bonus. Tasks 7 and 8 both consume it. **Assuming a value and being wrong poisons
six feature dims with no error anywhere**, so this task produces a *measured*
constant.

**Files:**
- Create: `python/ptcg_mine/damage.py`
- Test: `python/tests/test_damage.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `WEAKNESS_MULT: float` — measured, with the measurement recorded in the docstring.
  - `RESISTANCE_DELTA: float` — measured the same way.
  - `effective_damage(base: float, attacker_energy_type: int, defender_weakness: int | None, defender_resistance: int | None) -> float`
  - `attack_is_affordable(cost_hist: np.ndarray, attached_hist: np.ndarray) -> bool`

- [ ] **Step 1: Write the measurement spike**

Create `/tmp/claude-1000/-home-charles-Documents-Pokemon/scratch-weakness.py` (a
throwaway, not committed):

```python
"""Drive one attack into a known weakness and read the damage off the log."""
import sys
sys.path.insert(0, "pokemon-tcg-ai-battle/sample_submission/sample_submission")
from cg.api import all_card_data, all_attack

cards = all_card_data()
attacks = {a.attackId: a for a in all_attack()}

# Find (attacker, defender) where defender.weakness == attacker.energyType
# and the attacker has a vanilla attack (empty text => damage is not modified).
for atk_card in cards:
    if int(atk_card.cardType) != 0 or not atk_card.attacks:
        continue
    a = attacks.get(int(atk_card.attacks[0]))
    if a is None or a.text.strip() or a.damage <= 0:
        continue
    for dfn in cards:
        if int(dfn.cardType) != 0 or dfn.weakness is None:
            continue
        if int(dfn.weakness) == int(atk_card.energyType):
            print(f"attacker={atk_card.name} type={int(atk_card.energyType)} "
                  f"base={a.damage} attack={a.name!r}")
            print(f"defender={dfn.name} hp={dfn.hp} weakness={int(dfn.weakness)}")
            raise SystemExit
```

Run it, then play the matchup through `cg.sim` (or `ptcg_il/live_eval.py`'s
harness) and read the resulting `Log` entry's `value` field against `a.damage`.
`Log.value` is the applied damage.

- [ ] **Step 2: Record the measured constants**

Create `python/ptcg_mine/damage.py`:

```python
"""Deterministic damage arithmetic shared by the CLS threat block and the
option damage preview.

``WEAKNESS_MULT`` and ``RESISTANCE_DELTA`` are **measured**, not assumed.
Weakness resolution lives in libcg.so and no Python binding states the rule, so
the values below come from driving a vanilla attack (empty ``Attack.text``, so
nothing modifies the damage) into a defender whose ``weakness`` equals the
attacker's ``energyType`` and reading the applied damage off ``Log.value``.

Record of the measurement -- update this block if it is ever redone:
    attacker  : <name> (energyType=<n>, base damage=<d>)
    defender  : <name> (weakness=<n>)
    Log.value : <observed>
    => WEAKNESS_MULT = <observed> / <d>

**Base damage is not real damage.**  315 of 1556 attacks carry
``does N more damage`` or ``damage for each ...`` text, so any KO prediction
built on ``Attack.damage`` alone is systematically wrong on those.  Callers must
treat the output as a graded signal, not a verdict -- which is why the features
that consume it are continuous ratios with the booleans derived from them, and
why ``keywords.damage_scaling`` sits in the same feature row so the model can
learn to discount the ratio when it fires.
"""

import numpy as np

#: MEASURED -- see module docstring. Do not change without redoing the measurement.
WEAKNESS_MULT: float = 2.0      # <-- replace with the measured value
RESISTANCE_DELTA: float = -30.0  # <-- replace with the measured value

COLORLESS = 0


def effective_damage(base, attacker_energy_type, defender_weakness,
                     defender_resistance) -> float:
    """Base damage adjusted for weakness and resistance. Never negative."""
    dmg = float(base)
    if dmg <= 0.0:
        return 0.0
    if defender_weakness is not None and int(defender_weakness) == int(attacker_energy_type):
        dmg *= WEAKNESS_MULT
    if defender_resistance is not None and int(defender_resistance) == int(attacker_energy_type):
        dmg += RESISTANCE_DELTA
    return max(dmg, 0.0)


def attack_is_affordable(cost_hist: np.ndarray, attached_hist: np.ndarray) -> bool:
    """Can a Pokemon holding *attached_hist* energy pay *cost_hist*?

    Both are length-12 histograms over ``EnergyType`` with COLORLESS at index 0.
    A colorless requirement accepts any energy, so it is checked only against the
    total; every colored requirement must be met in its own colour.

    Exact except for the 12 special Energy cards that provide multiple or
    arbitrary types (Prism, Legacy, Neo Upper, ...), which the engine resolves
    and this does not.
    """
    cost = np.asarray(cost_hist, dtype=np.float64)
    have = np.asarray(attached_hist, dtype=np.float64)
    if cost[1:].sum() > 0 and np.any(have[1:] < cost[1:]):
        return False
    return have.sum() >= cost.sum()
```

- [ ] **Step 3: Write the tests**

Create `python/tests/test_damage.py`:

```python
import numpy as np
import pytest

from ptcg_mine import damage as dmg


def test_no_weakness_no_change():
    assert dmg.effective_damage(100, 2, None, None) == 100.0


def test_weakness_applies_only_on_type_match():
    assert dmg.effective_damage(100, 2, 2, None) == 100.0 * dmg.WEAKNESS_MULT
    assert dmg.effective_damage(100, 2, 3, None) == 100.0


def test_resistance_applies_only_on_type_match():
    assert dmg.effective_damage(100, 2, None, 2) == 100.0 + dmg.RESISTANCE_DELTA
    assert dmg.effective_damage(100, 2, None, 3) == 100.0


def test_damage_never_goes_negative():
    assert dmg.effective_damage(10, 2, None, 2) == 0.0


def test_zero_base_stays_zero_even_against_weakness():
    """Status-only attacks have damage 0; weakness must not manufacture damage."""
    assert dmg.effective_damage(0, 2, 2, None) == 0.0


def _hist(**kw):
    h = np.zeros(12, dtype=np.float32)
    for k, v in kw.items():
        h[int(k[1:])] = v
    return h


def test_colorless_cost_accepts_any_energy():
    assert dmg.attack_is_affordable(_hist(t0=2), _hist(t5=2))


def test_colored_cost_needs_its_own_colour():
    """Two Psychic does not pay a two-Fire cost even though the total matches."""
    assert not dmg.attack_is_affordable(_hist(t2=2), _hist(t5=2))
    assert dmg.attack_is_affordable(_hist(t2=2), _hist(t2=2))


def test_mixed_cost_checks_colour_then_total():
    cost = _hist(t2=1, t0=2)          # 1 Fire + 2 Colorless
    assert not dmg.attack_is_affordable(cost, _hist(t2=1, t5=1))   # total 2 < 3
    assert dmg.attack_is_affordable(cost, _hist(t2=1, t5=2))       # total 3, Fire met


def test_empty_cost_is_always_affordable():
    assert dmg.attack_is_affordable(np.zeros(12), np.zeros(12))


@pytest.mark.engine
def test_weakness_mult_matches_the_engine():
    """Pin the measured constant to the live engine.

    This is the guard for the whole KO-pressure block: if libcg's weakness rule
    ever changes, six feature dims go silently wrong and nothing else notices.
    """
    observed = _measure_weakness_multiplier_via_engine()   # see Step 1's spike
    assert observed == pytest.approx(dmg.WEAKNESS_MULT)
```

Port the spike from Step 1 into `_measure_weakness_multiplier_via_engine()` in
the test module. Mark it `@pytest.mark.engine` and register the marker in
`pyproject.toml` so it can be deselected on machines without `libcg.so`.

- [ ] **Step 4: Run the tests**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_damage.py -v
```

Expected: all PASS. If `test_weakness_mult_matches_the_engine` fails, the
constant in `damage.py` is wrong — **fix the constant, not the test.**

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_mine/damage.py python/tests/test_damage.py python/pyproject.toml
git commit -m "feat(mine): engine-measured weakness/resistance arithmetic and attack affordability"
```

---

### Task 4: Extend the card and attack static rows to 212 / 43

**Files:**
- Modify: `python/ptcg_mine/cards.py` (`attack_static_row`, `card_static_row`, docstrings)
- Modify: `python/ptcg_il/featurizer.py` (`F_CARD` 94→212, `F_ATK` 14→43)
- Test: `python/tests/test_cards.py` (extend), `python/tests/test_card_separation.py` (create)

**Interfaces:**
- Consumes: `ptcg_mine.keywords.{K_EFFECT, ability_keyword_row, attack_keyword_row}` (Task 1); the dim imports from Task 2.
- Produces: `card_static_row(card, attacks_by_id) -> float32[212]`, `attack_static_row(attack) -> float32[43]`. Layout:
  - `[0:52]` unchanged base
  - `[52:81]` ability keyword multihot
  - `[81]` `n_abilities / 3.0`
  - `[82]` `n_attacks / 3.0`
  - `[83:126]`, `[126:169]`, `[169:212]` — three `attack_static_row` blocks
  - `attack_static_row`: `[0:14]` unchanged, `[14:43]` attack keyword multihot

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_cards.py`:

```python
from ptcg_mine.keywords import K_EFFECT, KEYWORD_NAMES


def test_card_row_width_is_212():
    from ptcg_mine.cards import F_CARD
    assert F_CARD == 212 == 52 + K_EFFECT + 2 + 3 * (14 + K_EFFECT)


def test_attack_row_width_is_43():
    from ptcg_mine.cards import F_ATK
    assert F_ATK == 43 == 14 + K_EFFECT


def test_base_block_is_unchanged(sample_card, attacks_by_id):
    """The first 52 dims must be bit-identical to the pre-change layout, or every
    hardcoded slice (CARD_FEAT_BASIC_COL) silently repoints."""
    from ptcg_mine.cards import card_static_row
    row = card_static_row(sample_card, attacks_by_id)
    assert row[0] == pytest.approx(sample_card.hp / 400.0)
    assert row[9] == float(sample_card.basic)


def test_attack_blocks_embed_attack_static_row_verbatim(sample_card, attacks_by_id):
    """card_static_row embeds attack_static_row so the two views cannot drift."""
    from ptcg_mine.cards import F_ATK, attack_static_row, card_static_row
    row = card_static_row(sample_card, attacks_by_id)
    checked = 0
    for i, aid in enumerate(sample_card.attacks[:3]):
        atk = attacks_by_id.get(int(aid))
        if atk is None:
            continue
        off = 83 + i * F_ATK
        assert np.allclose(row[off:off + F_ATK], attack_static_row(atk))
        checked += 1
    assert checked > 0, "fixture has no resolvable attacks"


def test_ability_keywords_land_in_their_block(attacks_by_id):
    from ptcg_mine.cards import card_static_row

    class _S:
        name = "x"; text = "Draw 3 cards."

    class _C:
        cardId = 1; hp = 60; retreatCost = 1; cardType = 0
        basic = True; stage1 = False; stage2 = False
        energyType = 1; weakness = None; resistance = None
        ex = megaEx = tera = aceSpec = False
        skills = [_S()]; attacks = []

    row = card_static_row(_C(), attacks_by_id)
    assert row[52 + KEYWORD_NAMES.index("draw")] == 1.0
    assert row[81] == pytest.approx(1 / 3.0)   # n_abilities
    assert row[82] == 0.0                       # n_attacks


def test_every_card_feat_key_propagates_the_new_width():
    """The nine *_card_feat tensors all come from the same lookup, so one width
    change must reach all of them.  A key left at 94 fails only at the first
    forward pass, far from here."""
    from ptcg_il.featurizer import F_ATK, F_CARD, featurize
    obs, action = _get_active_step(_load_episode())
    t = featurize(obs, vocab={}, action=action,
                  engine_card_features=_engine_card_features(),
                  engine_attack_features=_engine_attack_features())
    card_keys = [k for k in t if k.endswith("_card_feat")]
    assert len(card_keys) == 9, f"expected 9 *_card_feat keys, got {card_keys}"
    for k in card_keys:
        assert t[k].shape[-1] == F_CARD, f"{k} is {t[k].shape[-1]}, expected {F_CARD}"
    assert t["opt_attack_feat"].shape[-1] == F_ATK
```

Import `_get_active_step`, `_load_episode`, `_engine_card_features` and
`_engine_attack_features` from `tests/test_featurizer.py`, or move this test
into that module — it needs a real observation, not a synthetic card.

Create `python/tests/test_card_separation.py`:

```python
"""The point of the whole change: cards must stop being indistinguishable.

Thresholds are floors, set under the measured values so a pattern edit that
regresses separation goes red without any refactor tripping the gate.
"""

import collections

import pytest

from ptcg_mine.cards import build_engine_card_features, load_engine

CARD_TYPE_FLOORS = {1: ("ITEM", 25), 2: ("TOOL", 12), 3: ("SUPPORTER", 25), 4: ("STADIUM", 12)}


@pytest.fixture(scope="module")
def engine_rows():
    cards, attacks = load_engine()
    return cards, build_engine_card_features(cards, attacks)


def test_trainer_cards_are_separated(engine_rows):
    cards, feats = engine_rows
    checked = 0
    for ctype, (name, floor) in CARD_TYPE_FLOORS.items():
        sub = [c for c in cards if int(c.cardType) == ctype]
        assert sub, f"no {name} cards in the engine tables"
        distinct = len({feats[c.cardId].tobytes() for c in sub})
        assert distinct >= floor, (
            f"{name}: {distinct} distinct rows over {len(sub)} cards, floor {floor}"
        )
        checked += 1
    assert checked == 4, "fixture drift: expected 4 trainer cardTypes"


def test_no_whole_card_type_is_uncovered(engine_rows):
    """An all-zero keyword row is correct for a vanilla card, but a whole
    cardType coming back empty means the patterns missed that card class."""
    from ptcg_mine.keywords import ability_keyword_row, attack_keyword_row

    cards, _ = engine_rows
    _, attacks = load_engine()
    by_aid = {a.attackId: a for a in attacks}
    covered = collections.Counter()
    total = collections.Counter()
    for c in cards:
        t = int(c.cardType)
        total[t] += 1
        hit = ability_keyword_row(c).any() or any(
            attack_keyword_row(by_aid[int(a)]).any() for a in c.attacks if int(a) in by_aid
        )
        covered[t] += bool(hit)
    assert total, "no cards examined"
    for t in (0, 1, 2, 3, 4):   # POKEMON, ITEM, TOOL, SUPPORTER, STADIUM
        assert covered[t] > 0, f"cardType {t}: 0 of {total[t]} cards matched any keyword"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_cards.py tests/test_card_separation.py -v
```

Expected: width assertions FAIL (`212 != 94`); separation FAILS (SUPPORTER has 1
distinct row).

- [ ] **Step 3: Bump the dims in the featurizer**

In `python/ptcg_il/featurizer.py`:

```python
F_CARD = 212  # 52 base + 29 ability keywords + 2 counts + 3 attacks × 43
F_ATK = 43    # 14 numeric + 29 attack keywords
```

- [ ] **Step 4: Extend the row builders**

In `python/ptcg_mine/cards.py`, add the import and the agreement asserts below
the existing imports:

```python
from ptcg_il.featurizer import F_ATK, F_CARD
from ptcg_mine.keywords import K_EFFECT, ability_keyword_row, attack_keyword_row

# ptcg_il.featurizer owns the dims but cannot import K_EFFECT (it is vendored
# into the Kaggle bundle, where ptcg_mine does not exist).  Assert agreement
# here instead: appending a keyword without bumping the featurizer would
# otherwise emit a short row that every downstream shape check accepts.
assert F_ATK == 14 + K_EFFECT, (
    f"F_ATK={F_ATK} in ptcg_il.featurizer disagrees with K_EFFECT={K_EFFECT} "
    f"in ptcg_mine.keywords (expected {14 + K_EFFECT})"
)
assert F_CARD == 52 + K_EFFECT + 2 + 3 * F_ATK, (
    f"F_CARD={F_CARD} in ptcg_il.featurizer disagrees with K_EFFECT={K_EFFECT} "
    f"(expected {52 + K_EFFECT + 2 + 3 * F_ATK})"
)

#: Offset of the first embedded attack block inside ``card_static_row``.
CARD_ATTACK_BLOCK_START = 52 + K_EFFECT + 2   # == 83
```

Replace the tail of `attack_static_row` (currently ends at `row[13]`):

```python
    row[13] = len(attack.energies) / ATKCOST_N
    # Effect keywords from the attack's oracle text (14:43).  Living here rather
    # than only in card_static_row is deliberate: opt_attack_feat is built from
    # this row, so ATTACK options gain their effect text for free -- the decision
    # where text matters most.
    row[14:14 + K_EFFECT] = attack_keyword_row(attack)
    return row
```

Replace the attack loop in `card_static_row`:

```python
    # Ability keywords + counts (52:83)
    row[52:52 + K_EFFECT] = ability_keyword_row(card)
    skills = getattr(card, "skills", None) or []
    attack_ids = getattr(card, "attacks", []) or []
    row[52 + K_EFFECT] = min(len(skills), 3) / 3.0
    row[52 + K_EFFECT + 1] = min(len(attack_ids), 3) / 3.0
    # Attack features (83:212) -- up to 3 attacks, each F_ATK, same layout as
    # attack_static_row so the two views of an attack cannot drift apart.
    for ai in range(min(len(attack_ids), 3)):
        atk = attacks_by_id.get(int(attack_ids[ai]))
        if atk is None:
            continue
        offset = CARD_ATTACK_BLOCK_START + ai * F_ATK
        row[offset:offset + F_ATK] = attack_static_row(atk)
    return row
```

Update the module docstring's layout diagram to the new offsets.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_cards.py tests/test_card_separation.py -v
```

Expected: all PASS. Separation should measure SUPPORTER 43, ITEM 57, STADIUM 16,
TOOL 18 — comfortably above the floors.

- [ ] **Step 6: Mutation-check the separation gate**

Temporarily make `ability_keyword_row` return a zero row. Confirm
`test_trainer_cards_are_separated` goes red for SUPPORTER. Revert.

- [ ] **Step 7: Commit**

```bash
git add python/ptcg_mine/cards.py python/ptcg_il/featurizer.py \
        python/tests/test_cards.py python/tests/test_card_separation.py
git commit -m "feat: effect keywords in card and attack static rows (F_CARD 94->212)"
```

---

### Task 5: `evolution_map.npy` mining artifact

`can_evolve` in Task 6 needs it: `CardData.evolvesFrom` is a **name string** and
names are not in the feature row.

**Files:**
- Modify: `python/ptcg_mine/cards.py` (add `build_evolution_map`)
- Modify: `python/ptcg_mine/mine.py:419-420` (write the artifact)
- Test: `python/tests/test_cards.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_evolution_map(card_data) -> dict[int, list[int]]` mapping a card id to the ids of every card it evolves from. Written to `<out-dir>/evolution_map.npy`.

- [ ] **Step 1: Write the failing test**

Append to `python/tests/test_cards.py`:

```python
def test_evolution_map_resolves_names_to_ids():
    from ptcg_mine.cards import build_evolution_map

    class _C:
        def __init__(self, cid, name, evolves_from):
            self.cardId, self.name, self.evolvesFrom = cid, name, evolves_from

    cards = [_C(1, "Charmander", None), _C(2, "Charmeleon", "Charmander"),
             _C(3, "Charmander", None)]     # reprint: same name, different id
    m = build_evolution_map(cards)
    assert sorted(m[2]) == [1, 3], "a reprint of the pre-evolution must also count"
    assert m[1] == []


def test_evolution_map_handles_unresolvable_names(caplog):
    """An unresolved name must be logged, not silently dropped -- a silent drop
    reads downstream as 'this card can never be evolved'."""
    from ptcg_mine.cards import build_evolution_map

    class _C:
        def __init__(self, cid, name, evolves_from):
            self.cardId, self.name, self.evolvesFrom = cid, name, evolves_from

    m = build_evolution_map([_C(1, "Charmeleon", "Charmander")])
    assert m[1] == []
    assert "Charmander" in caplog.text


def test_evolution_map_covers_the_real_engine():
    from ptcg_mine.cards import build_evolution_map, load_engine

    cards, _ = load_engine()
    m = build_evolution_map(cards)
    evolvers = [c for c in cards if getattr(c, "evolvesFrom", None)]
    assert evolvers, "no evolution cards in the engine tables"
    unresolved = [c.name for c in evolvers if not m.get(c.cardId)]
    assert not unresolved, f"unresolved pre-evolution names: {unresolved[:10]}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_cards.py -k evolution -v
```

Expected: `ImportError: cannot import name 'build_evolution_map'`.

- [ ] **Step 3: Implement**

Add to `python/ptcg_mine/cards.py`:

```python
import logging

logger = logging.getLogger(__name__)


def build_evolution_map(card_data: list) -> dict[int, list[int]]:
    """``{card_id: [pre_evolution_card_ids]}`` resolved from ``evolvesFrom``.

    ``CardData.evolvesFrom`` is a *name*, and names are not in the feature row,
    so the hand-playability flag needs this side table.  Several distinct card
    ids share a name (reprints), hence the list value -- any of them in play
    makes the evolution legal.

    An unresolvable name is logged rather than dropped: silently returning an
    empty list is indistinguishable downstream from "this card evolves from
    nothing", which would teach the model the evolution is never available.
    """
    by_name: dict[str, list[int]] = {}
    for c in card_data:
        by_name.setdefault(c.name, []).append(int(c.cardId))

    out: dict[int, list[int]] = {}
    for c in card_data:
        pre = getattr(c, "evolvesFrom", None)
        if not pre:
            out[int(c.cardId)] = []
            continue
        ids = by_name.get(pre)
        if ids is None:
            logger.warning(
                "evolution_map: %s (id %d) evolves from %r, which matches no "
                "card name in the engine tables", c.name, int(c.cardId), pre)
            out[int(c.cardId)] = []
        else:
            out[int(c.cardId)] = sorted(ids)
    return out
```

In `python/ptcg_mine/mine.py`, right after line 420
(`np.save(out_dir / "engine_card_features.npy", ...)`):

```python
    evolution_map = build_evolution_map(cards)
    np.save(out_dir / "evolution_map.npy", evolution_map)
```

and add `build_evolution_map` to the `from ptcg_mine.cards import (...)` block
at line 22.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_cards.py -v
```

Expected: all PASS. If `test_evolution_map_covers_the_real_engine` reports
unresolved names, record them in the test as an explicit allowlist with a
comment naming why each is unresolvable — do **not** weaken the assertion to
`len(unresolved) < N`.

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_mine/cards.py python/ptcg_mine/mine.py python/tests/test_cards.py
git commit -m "feat(mine): evolution_map.npy artifact resolving evolvesFrom names to ids"
```

---

### Task 6: Hand playability flags (`F_HAND` 2 → 7)

**Files:**
- Modify: `python/ptcg_il/featurizer.py` (`F_HAND`, `_build_hand_tokens`, the `featurize` call site at line 954)
- Test: `python/tests/test_featurizer.py` (extend)

**Interfaces:**
- Consumes: `evolution_map` from Task 5; `engine_card_features` already threaded through `featurize`.
- Produces: `_build_hand_tokens(state, your_index, engine_card_features=None, evolution_map=None)`; `featurize(..., evolution_map=None)` gains the keyword argument.

`hand_feat` columns: `[0]` `idx/H_MAX`, `[1]` `dup_count/COUNT_N`, `[2]`
`can_bench`, `[3]` `can_evolve`, `[4]` `can_attach_energy`, `[5]`
`can_play_supporter`, `[6]` `can_play_stadium`.

- [ ] **Step 1: Write the failing test**

Append to `python/tests/test_featurizer.py`:

```python
def test_hand_feat_width_is_7():
    from ptcg_il.featurizer import F_HAND
    assert F_HAND == 7


def test_can_bench_needs_a_basic_and_a_free_slot(basic_obs, engine_feats):
    """A Basic in hand is playable only while the bench has room."""
    from ptcg_il.featurizer import featurize
    obs = basic_obs()            # bench has room, hand[0] is a Basic Pokemon
    t = featurize(obs, vocab={}, engine_card_features=engine_feats)
    assert t["hand_feat"][0, 2] == 1.0

    me = obs["current"]["players"][obs["current"]["yourIndex"]]
    me["bench"] = [dict(me["bench"][0]) for _ in range(me["benchMax"])]
    t = featurize(obs, vocab={}, engine_card_features=engine_feats)
    assert t["hand_feat"][0, 2] == 0.0, "full bench must clear can_bench"


def test_can_attach_energy_respects_the_once_per_turn_flag(energy_obs, engine_feats):
    from ptcg_il.featurizer import featurize
    obs = energy_obs()           # hand[0] is a Basic Energy card
    obs["current"]["energyAttached"] = False
    assert featurize(obs, vocab={}, engine_card_features=engine_feats)["hand_feat"][0, 4] == 1.0
    obs["current"]["energyAttached"] = True
    assert featurize(obs, vocab={}, engine_card_features=engine_feats)["hand_feat"][0, 4] == 0.0


def test_can_evolve_does_not_fire_across_unrelated_lines(evolve_obs, engine_feats):
    """The tempting shortcut -- 'this is a Stage 1 and something Basic is in
    play' -- fires across unrelated evolution lines and would teach the model an
    illegal play is available."""
    from ptcg_il.featurizer import featurize
    obs, evo_map, matching_slot, wrong_slot = evolve_obs()
    t = featurize(obs, vocab={}, engine_card_features=engine_feats, evolution_map=evo_map)
    assert t["hand_feat"][matching_slot, 3] == 1.0
    assert t["hand_feat"][wrong_slot, 3] == 0.0


def test_can_evolve_blocked_for_a_pokemon_that_appeared_this_turn(evolve_obs, engine_feats):
    from ptcg_il.featurizer import featurize
    obs, evo_map, matching_slot, _ = evolve_obs()
    obs["current"]["players"][obs["current"]["yourIndex"]]["active"][0]["appearThisTurn"] = True
    t = featurize(obs, vocab={}, engine_card_features=engine_feats, evolution_map=evo_map)
    assert t["hand_feat"][matching_slot, 3] == 0.0


def test_pad_hand_slots_stay_all_zero(basic_obs, engine_feats):
    """All-zero is the PAD sentinel and masks read it."""
    from ptcg_il.featurizer import featurize
    obs = basic_obs()
    n = len(obs["current"]["players"][obs["current"]["yourIndex"]]["hand"])
    t = featurize(obs, vocab={}, engine_card_features=engine_feats)
    assert not t["hand_feat"][n:].any()
```

**Fixture convention — read this before writing them.** `python/tests/test_featurizer.py`
does **not** use pytest fixtures for observations; it uses module-level helpers
(around lines 61–138):

- `_load_episode() -> dict` — one real episode from the corpus
- `_get_active_step(...) -> (obs, action)` — pulls a real decision point out of it
- `_engine_card_features() -> dict[int, np.ndarray]` (cached in `_ECF_CACHE`)
- `_engine_attack_features() -> dict[int, np.ndarray]` (cached in `_EAF_CACHE`)
- `_feat_of(card_id) -> np.ndarray`

Write the new observation builders in the same style — plain functions returning
a **deep copy** of a real observation with the relevant field forced, e.g.:

```python
def _obs_with_hand(card_ids, **state_flags):
    """Real decision point with the actor's hand replaced by *card_ids*."""
    obs, _ = _get_active_step(_load_episode())
    obs = copy.deepcopy(obs)
    state = obs["current"]
    state.update(state_flags)
    state["players"][state["yourIndex"]]["hand"] = [{"id": c} for c in card_ids]
    return obs
```

Then convert the fixture parameters in the tests above to calls
(`engine_feats` → `_engine_card_features()`, `basic_obs()` → `_obs_with_hand([...])`).
Build from a real observation rather than a hand-written dict so the tests keep
exercising the actual key set the engine emits.

- [ ] **Step 2: Run test to verify it fails**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_featurizer.py -k hand -v
```

Expected: `assert 2 == 7` and IndexError on column 2.

- [ ] **Step 3: Implement**

In `python/ptcg_il/featurizer.py` set `F_HAND = 7` and add the cardType column
constants near `CARD_FEAT_BASIC_COL`:

```python
#: ``card_static_row[2:9]`` is the cardType one-hot (see ptcg_mine.cards).
CARD_FEAT_CARDTYPE_START = 2
CARD_TYPE_ITEM = 1
CARD_TYPE_SUPPORTER = 3
CARD_TYPE_STADIUM = 4
CARD_TYPE_BASIC_ENERGY = 5
CARD_TYPE_SPECIAL_ENERGY = 6
```

Replace `_build_hand_tokens`:

```python
def _build_hand_tokens(
    state: dict, your_index: int,
    engine_card_features: dict | None = None,
    evolution_map: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build hand_card_id[H_MAX] int64 and hand_feat[H_MAX, F_HAND] float32.

    Cards are packed to the front; remaining slots are PAD (all-zero).

    hand_feat = [idx/H_MAX, dup_count/COUNT_N, can_bench, can_evolve,
                 can_attach_energy, can_play_supporter, can_play_stadium].

    The five legality flags are deterministic and the engine already enumerates
    legal plays at a MAIN select -- their value is at *non-MAIN* decisions and
    for lookahead, e.g. seeing while choosing a discard that one candidate is a
    Supporter not yet played this turn.
    """
    hand_card_id = np.full(H_MAX, PAD_CARD, dtype=np.int64)
    hand_feat = np.zeros((H_MAX, F_HAND), dtype=np.float32)

    player = state["players"][your_index]
    hand = player.get("hand")
    if hand is None:
        return hand_card_id, hand_feat

    id_counts: dict[int, int] = {}
    for card in hand:
        cid = card["id"]
        id_counts[cid] = id_counts.get(cid, 0) + 1

    bench_has_room = len(player["bench"]) < int(player.get("benchMax", 5))
    energy_free = not state.get("energyAttached", False)
    supporter_free = not state.get("supporterPlayed", False)
    stadium_free = not state.get("stadiumPlayed", False)

    # Ids of my in-play Pokemon that did not arrive this turn -- only those can
    # be evolved.
    evolvable_ids: set[int] = set()
    in_play = list(player["active"] or []) + list(player["bench"] or [])
    for poke in in_play:
        if poke is None or poke.get("appearThisTurn", False):
            continue
        evolvable_ids.add(int(poke["id"]))

    for i, card in enumerate(hand):
        if i >= H_MAX:
            break
        cid = card["id"]
        hand_card_id[i] = _raw_card(cid)
        f = hand_feat[i]
        f[0] = _clip_norm(float(i), HAND_N)
        f[1] = _clip_norm(float(id_counts.get(cid, 1)), COUNT_N)

        crow = None
        if engine_card_features is not None and cid is not None:
            crow = engine_card_features.get(int(cid))
        if crow is None:
            continue

        is_basic = crow[CARD_FEAT_BASIC_COL] > 0.5
        ctype_slice = crow[CARD_FEAT_CARDTYPE_START:CARD_FEAT_CARDTYPE_START + 7]

        def _is(ct):
            return ctype_slice[ct] > 0.5

        f[2] = 1.0 if (is_basic and _is(0) and bench_has_room) else 0.0
        if evolution_map is not None:
            pre = evolution_map.get(int(cid)) or ()
            f[3] = 1.0 if any(p in evolvable_ids for p in pre) else 0.0
        f[4] = 1.0 if ((_is(CARD_TYPE_BASIC_ENERGY) or _is(CARD_TYPE_SPECIAL_ENERGY))
                       and energy_free) else 0.0
        f[5] = 1.0 if (_is(CARD_TYPE_SUPPORTER) and supporter_free) else 0.0
        f[6] = 1.0 if (_is(CARD_TYPE_STADIUM) and stadium_free) else 0.0

    return hand_card_id, hand_feat
```

Add `evolution_map: dict | None = None` to `featurize`'s signature (after
`engine_attack_features`), document it in the docstring, and update the call at
line 954:

```python
    hand_card_id, hand_feat = _build_hand_tokens(
        state, your_index, engine_card_features, evolution_map
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_featurizer.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_il/featurizer.py python/tests/test_featurizer.py
git commit -m "feat(il): hand playability flags (F_HAND 2->7)"
```

---

### Task 7: KO-pressure scalars (`F_GLOBAL` 93 → 97)

**Files:**
- Modify: `python/ptcg_il/featurizer.py` (`F_GLOBAL`, `_build_cls_features`, call site line 956)
- Test: `python/tests/test_featurizer.py` (extend)

**Interfaces:**
- Consumes: `ptcg_mine.damage.{effective_damage, attack_is_affordable}` (Task 3) — imported lazily inside the function, because `featurizer.py` is vendored into the bundle without `ptcg_mine`. **Inline the two functions into `featurizer.py` instead** (they are 20 lines total) rather than importing across the boundary.
- Produces: `cls_feat[93:97]`.

Columns: `[93]` my best affordable damage ÷ opp active current HP (clipped
`[0, 2]`), `[94]` `can_ko_opp`, `[95]` opp best damage (affordability ignored) ÷
my active current HP (clipped `[0, 2]`), `[96]` `opp_can_ko_me`.

The asymmetry is deliberate: on my turn I can only use an attack I can pay for
now; the opponent gets a full turn to attach Energy first.

- [ ] **Step 1: Write the failing test**

Append to `python/tests/test_featurizer.py`:

```python
def test_cls_feat_width_is_97():
    from ptcg_il.featurizer import F_GLOBAL
    assert F_GLOBAL == 97


def test_existing_cls_slices_did_not_move():
    """embed.py reads cls_feat[87:88] and [88:89]; the new dims append at 93."""
    from ptcg_il.featurizer import F_GLOBAL
    assert F_GLOBAL - 4 == 93


def test_ko_ratio_is_damage_over_hp(ko_obs, engine_feats):
    from ptcg_il.featurizer import featurize
    obs = ko_obs(my_damage=60, opp_hp=120)     # affordable, no weakness
    t = featurize(obs, vocab={}, engine_card_features=engine_feats)
    assert t["cls_feat"][93] == pytest.approx(0.5)
    assert t["cls_feat"][94] == 0.0


def test_can_ko_fires_when_damage_reaches_hp(ko_obs, engine_feats):
    from ptcg_il.featurizer import featurize
    t = featurize(ko_obs(my_damage=120, opp_hp=120), vocab={},
                  engine_card_features=engine_feats)
    assert t["cls_feat"][93] == pytest.approx(1.0)
    assert t["cls_feat"][94] == 1.0


def test_ko_ratio_is_clipped_at_2(ko_obs, engine_feats):
    from ptcg_il.featurizer import featurize
    t = featurize(ko_obs(my_damage=900, opp_hp=60), vocab={},
                  engine_card_features=engine_feats)
    assert t["cls_feat"][93] == pytest.approx(2.0)


def test_unaffordable_attack_does_not_count_for_me(ko_obs, engine_feats):
    """My side uses only attacks I can pay for right now."""
    from ptcg_il.featurizer import featurize
    obs = ko_obs(my_damage=200, opp_hp=120, my_energy=0)
    t = featurize(obs, vocab={}, engine_card_features=engine_feats)
    assert t["cls_feat"][93] == 0.0
    assert t["cls_feat"][94] == 0.0


def test_opponent_threat_ignores_affordability(ko_obs, engine_feats):
    """The opponent gets a whole turn to attach before attacking."""
    from ptcg_il.featurizer import featurize
    obs = ko_obs(opp_damage=200, my_hp=120, opp_energy=0)
    t = featurize(obs, vocab={}, engine_card_features=engine_feats)
    assert t["cls_feat"][96] == 1.0


def test_empty_board_leaves_the_block_zero(basic_obs, engine_feats):
    from ptcg_il.featurizer import featurize
    obs = basic_obs()
    obs["current"]["players"][1 - obs["current"]["yourIndex"]]["active"] = []
    t = featurize(obs, vocab={}, engine_card_features=engine_feats)
    assert t["cls_feat"][93] == 0.0 and t["cls_feat"][94] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_featurizer.py -k "cls_feat or ko" -v
```

Expected: `assert 93 == 97`.

- [ ] **Step 3: Implement**

Set `F_GLOBAL = 97`. Add the inlined damage helpers near `_clip_norm` (copy the
bodies from `ptcg_mine/damage.py`, with a comment explaining that they are
duplicated because this module ships without `ptcg_mine`, and that
`tests/test_damage.py` pins the constants):

```python
# Duplicated from ptcg_mine.damage rather than imported: this module is vendored
# into the Kaggle bundle, where ptcg_mine does not exist.  The constants are
# engine-measured; python/tests/test_damage.py pins them and a test in
# test_featurizer.py asserts the two copies agree.
WEAKNESS_MULT = 2.0       # keep in sync with ptcg_mine.damage
RESISTANCE_DELTA = -30.0  # keep in sync with ptcg_mine.damage
```

Add a test asserting the two copies agree:

```python
def test_damage_constants_match_ptcg_mine():
    import ptcg_il.featurizer as fz
    from ptcg_mine import damage
    assert fz.WEAKNESS_MULT == damage.WEAKNESS_MULT
    assert fz.RESISTANCE_DELTA == damage.RESISTANCE_DELTA
```

Add these helpers above `_build_cls_features`:

```python
# Slices into a card_static_row (see ptcg_mine.cards).  Named because an
# off-by-one here reads the wrong energy type and silently mis-scores weakness.
_CARD_ENERGYTYPE = slice(12, 24)
_CARD_WEAKNESS = slice(24, 36)
_CARD_RESISTANCE = slice(36, 48)
# Derived, not written as 83: this module cannot import K_EFFECT (it ships
# without ptcg_mine), and a second hand-maintained copy of the offset is exactly
# the drift this change exists to remove.  The three attack blocks are the tail
# of the row, so the start is fixed by F_CARD and F_ATK alone.
CARD_ATTACK_BLOCK_START = F_CARD - 3 * F_ATK   # == 83


def _onehot_index(vec) -> int | None:
    """Index of the set bit in a one-hot slice, or None if all-zero."""
    nz = np.flatnonzero(np.asarray(vec) > 0.5)
    return int(nz[0]) if nz.size else None


def _effective_damage(base, atk_type, weakness, resistance) -> float:
    """Base damage adjusted for weakness/resistance.  Mirrors ptcg_mine.damage."""
    dmg = float(base)
    if dmg <= 0.0:
        return 0.0
    if weakness is not None and atk_type is not None and weakness == atk_type:
        dmg *= WEAKNESS_MULT
    if resistance is not None and atk_type is not None and resistance == atk_type:
        dmg += RESISTANCE_DELTA
    return max(dmg, 0.0)


def _attack_is_affordable(cost_hist, attached_hist) -> bool:
    """Colorless (index 0) accepts any energy; colored costs need their colour."""
    cost = np.asarray(cost_hist, dtype=np.float64)
    have = np.asarray(attached_hist, dtype=np.float64)
    if cost[1:].sum() > 0 and np.any(have[1:] < cost[1:]):
        return False
    return have.sum() >= cost.sum()


def _best_damage(attacker_row, defender_row, attached_hist, require_affordable):
    """Best damage the attacker can deal to the defender, in raw HP units.

    *attacker_row* / *defender_row* are card_static_rows; *attached_hist* is the
    attacker's 12-wide attached-energy histogram in raw counts.  Reads the three
    embedded attack blocks rather than the attack table, so no attack-id lookup
    is needed.
    """
    if attacker_row is None or defender_row is None:
        return 0.0
    atk_type = _onehot_index(attacker_row[_CARD_ENERGYTYPE])
    weakness = _onehot_index(defender_row[_CARD_WEAKNESS])
    resistance = _onehot_index(defender_row[_CARD_RESISTANCE])

    best = 0.0
    for i in range(3):
        off = CARD_ATTACK_BLOCK_START + i * F_ATK
        block = attacker_row[off:off + F_ATK]
        base = float(block[0]) * ATKDMG_N
        if base <= 0.0:
            continue
        if require_affordable:
            cost = np.asarray(block[1:13], dtype=np.float64) * ATKCOST_N
            if not _attack_is_affordable(cost, attached_hist):
                continue
        best = max(best, _effective_damage(base, atk_type, weakness, resistance))
    return best


def _ko_pressure(poke_card_feat, poke_feat) -> tuple[float, float, float, float]:
    """(my_ratio, can_ko_opp, opp_ratio, opp_can_ko_me).

    Poke slot layout (A.1): 0 = my active, 6 = opp active.

    The asymmetry is deliberate: on my turn I can only use an attack I can pay
    for right now, whereas the opponent gets a full turn to attach Energy first,
    so their side ignores affordability.
    """
    mine, opp = poke_card_feat[0], poke_card_feat[6]
    if not np.asarray(mine).any() or not np.asarray(opp).any():
        return 0.0, 0.0, 0.0, 0.0

    my_energy = np.asarray(poke_feat[0][3:15], dtype=np.float64) * ENERGY_N
    my_hp = float(poke_feat[0][0]) * HP_N
    opp_hp = float(poke_feat[6][0]) * HP_N

    my_dmg = _best_damage(mine, opp, my_energy, require_affordable=True)
    opp_dmg = _best_damage(opp, mine, None, require_affordable=False)

    my_ratio = min(my_dmg / opp_hp, 2.0) if opp_hp > 0 else 0.0
    opp_ratio = min(opp_dmg / my_hp, 2.0) if my_hp > 0 else 0.0
    return (my_ratio, 1.0 if my_ratio >= 1.0 else 0.0,
            opp_ratio, 1.0 if opp_ratio >= 1.0 else 0.0)
```

Change `_build_cls_features`'s signature to
`(state, select, your_index, poke_card_feat=None, poke_feat=None)` and append
before its return:

```python
    # [93:97] KO pressure.  Zero when either feature table is absent -- an
    # all-zero block is the same "no information" signal a PAD row carries.
    if poke_card_feat is not None and poke_feat is not None:
        cls_feat[93:97] = _ko_pressure(poke_card_feat, poke_feat)
```

In `featurize`, `poke_card_feat` is only computed at line 1005 via `_cfeat`, but
`_build_cls_features` is called at line 956. Hoist the conversion: right after
line 953's `_build_poke_tokens` call, add

```python
    poke_card_feat = _ids_to_feat(poke_card_id, engine_card_features, F_CARD, None)
```

pass it into `_build_cls_features`, and reuse the same array in the result dict
instead of calling `_cfeat(poke_card_id)` a second time.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_featurizer.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_il/featurizer.py python/tests/test_featurizer.py
git commit -m "feat(il): KO-pressure scalars on the CLS token (F_GLOBAL 93->97)"
```

---

### Task 8: Option damage preview (`F_OPT` 6 → 8)

**Files:**
- Modify: `python/ptcg_il/featurizer.py` (`F_OPT`, `_build_option_tokens`, call site line 984)
- Test: `python/tests/test_featurizer.py` (extend)

**Interfaces:**
- Consumes: the damage helpers inlined in Task 7; `poke_feat` and the engine tables.
- Produces: `opt_scalar[:, 6]` = damage ÷ target's current HP (clipped `[0, 2]`), `opt_scalar[:, 7]` = `is_lethal`. Both 0.0 for every non-ATTACK option and for the STOP column.

- [ ] **Step 1: Write the failing test**

```python
def test_opt_scalar_width_is_8():
    from ptcg_il.featurizer import F_OPT
    assert F_OPT == 8


def test_attack_option_carries_a_damage_ratio(attack_opt_obs, engine_feats, engine_atk_feats):
    from ptcg_il.featurizer import featurize
    obs = attack_opt_obs(damage=60, target_hp=120)
    t = featurize(obs, vocab={}, engine_card_features=engine_feats,
                  engine_attack_features=engine_atk_feats)
    assert t["opt_scalar"][0, 6] == pytest.approx(0.5)
    assert t["opt_scalar"][0, 7] == 0.0


def test_lethal_attack_option_is_flagged(attack_opt_obs, engine_feats, engine_atk_feats):
    from ptcg_il.featurizer import featurize
    t = featurize(attack_opt_obs(damage=130, target_hp=120), vocab={},
                  engine_card_features=engine_feats,
                  engine_attack_features=engine_atk_feats)
    assert t["opt_scalar"][0, 7] == 1.0


def test_non_attack_options_have_zero_preview(basic_obs, engine_feats):
    """A PLAY option must not inherit a stale damage ratio."""
    from ptcg_il.featurizer import featurize
    t = featurize(basic_obs(), vocab={}, engine_card_features=engine_feats)
    mask = t["opt_mask"]
    non_attack = (t["opt_type"] != 13) & mask
    assert non_attack.any(), "fixture has no non-ATTACK options"
    assert not t["opt_scalar"][non_attack, 6].any()
    assert not t["opt_scalar"][non_attack, 7].any()


def test_snipe_attack_uses_opt_tgt_idx_not_the_active(snipe_opt_obs, engine_feats, engine_atk_feats):
    from ptcg_il.featurizer import featurize
    obs = snipe_opt_obs(damage=60, active_hp=200, bench_hp=60)
    t = featurize(obs, vocab={}, engine_card_features=engine_feats,
                  engine_attack_features=engine_atk_feats)
    assert t["opt_scalar"][0, 6] == pytest.approx(1.0)   # vs the benched target
    assert t["opt_scalar"][0, 7] == 1.0


def test_stop_column_preview_is_zero(multiselect_obs, engine_feats):
    from ptcg_il.featurizer import featurize
    t = featurize(multiselect_obs(), vocab={}, engine_card_features=engine_feats)
    stop = int(t["stop_column"])
    assert stop >= 0, "fixture is not multi-select"
    assert t["opt_scalar"][stop, 6] == 0.0
    assert t["opt_scalar"][stop, 7] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_featurizer.py -k opt_scalar -v
```

Expected: `assert 6 == 8`.

- [ ] **Step 3: Implement**

Set `F_OPT = 8`. Add `poke_feat`, `engine_card_features`, `engine_attack_features`
parameters to `_build_option_tokens` and, in the `otype == 13` branch, after
`opt_attack_idx[new_j] = ...`:

```python
            # Damage preview (A.7).  Target is the snipe slot when the option
            # names one, else the opponent's Active (poke slot 6).
            tgt_slot = (int(opt_tgt_idx[new_j]) - 1) if opt_tgt_idx[new_j] > 0 else 6
            ratio = _attack_damage_ratio(
                opt.get("attackId"), tgt_slot, poke_card_feat, poke_feat,
                engine_attack_features,
            )
            opt_scalar[new_j, 6] = ratio
            opt_scalar[new_j, 7] = 1.0 if ratio >= 1.0 else 0.0
```

Add the helper next to `_ko_pressure` (Task 7):

```python
def _attack_damage_ratio(attack_id, tgt_slot, poke_card_feat, poke_feat,
                         engine_attack_features) -> float:
    """Damage this attack would deal to *tgt_slot*, over that slot's current HP.

    Clipped to [0, 2].  Returns 0.0 when the attack or target is unresolvable,
    which is the same "no information" signal a PAD row carries.

    Uses the attack table (not the attacker's embedded blocks) because the
    option names a specific attackId, which need not be the attacker's best.
    """
    if engine_attack_features is None or attack_id is None:
        return 0.0
    arow = engine_attack_features.get(int(attack_id))
    if arow is None:
        return 0.0
    if not (0 <= tgt_slot < P_MAX):
        return 0.0

    defender = poke_card_feat[tgt_slot]
    attacker = poke_card_feat[0]            # my active is always the attacker
    if not np.asarray(defender).any() or not np.asarray(attacker).any():
        return 0.0

    tgt_hp = float(poke_feat[tgt_slot][0]) * HP_N
    if tgt_hp <= 0.0:
        return 0.0

    dmg = _effective_damage(
        float(arow[0]) * ATKDMG_N,
        _onehot_index(attacker[_CARD_ENERGYTYPE]),
        _onehot_index(defender[_CARD_WEAKNESS]),
        _onehot_index(defender[_CARD_RESISTANCE]),
    )
    return min(dmg / tgt_hp, 2.0)
```

**`opt_tgt_idx` is a state-token row index, not a poke slot.** `_build_tok_attrs`
places poke tokens at rows 1..12, so the poke slot is `row - 1`; poke slot 6 is
the opponent's Active (layout: 0 = my active, 1..5 = my bench, 6 = opp active,
7..11 = opp bench). The call site must therefore be:

```python
            tgt_slot = (int(opt_tgt_idx[new_j]) - 1) if opt_tgt_idx[new_j] > 0 else 6
```

An off-by-one here silently reads my own Pokémon's HP as the target's and every
ratio comes out plausible. `test_snipe_attack_uses_opt_tgt_idx_not_the_active`
is the guard — verify it fails if you write `int(opt_tgt_idx[new_j])` without
the `- 1`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_featurizer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_il/featurizer.py python/tests/test_featurizer.py
git commit -m "feat(il): option damage preview (F_OPT 6->8)"
```

---

### Task 9: float16 card features in shards

**Files:**
- Modify: `python/ptcg_il/shard_writer.py` (`_write_shard`, line 262-276)
- Test: `tests/test_shard_writer.py` (extend), `tests/test_dataset.py` (extend)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: shards whose ten card/attack float keys are `float16` on disk. The read path is unchanged — `ShardDataset.__getitem__` already ends every non-int, non-bool key with `.float()`.

- [ ] **Step 1: Write the failing test**

```python
FP16_KEYS = {
    "poke_card_feat", "hand_card_feat", "stadium_card_feat",
    "context_card_feat", "effect_card_feat", "discard_card_feat",
    "prize_card_feat", "opt_card_feat", "opt_attack_feat", "log_card_feat",
}


def test_card_feature_keys_are_stored_fp16():
    """Card features are 89.8% of shard bytes.  fp32 there is what makes the
    F_CARD expansion unaffordable."""
    import tempfile
    from ptcg_il.shard_writer import _write_shard

    ep = _make_synthetic_episode(1, decks={0: 1, 1: 30})
    buffer = _build_shard_buffer(ep, "train", ...)  # see existing writer tests
    with tempfile.TemporaryDirectory() as td:
        path = _write_shard("train", 0, buffer, Path(td))
        data = np.load(path)
    checked = 0
    for k in FP16_KEYS:
        if k not in data:
            continue
        assert data[k].dtype == np.float16, f"{k} is {data[k].dtype}, expected float16"
        checked += 1
    assert checked == len(FP16_KEYS), f"only {checked} of {len(FP16_KEYS)} keys present"


def test_non_card_floats_stay_fp32():
    import tempfile
    from ptcg_il.shard_writer import _write_shard

    ep = _make_synthetic_episode(1, decks={0: 1, 1: 30})
    buffer = _build_shard_buffer(ep, "train", ...)
    with tempfile.TemporaryDirectory() as td:
        data = np.load(_write_shard("train", 0, buffer, Path(td)))
    for k in ("cls_feat", "poke_feat", "opt_scalar", "value_target"):
        if k in data:
            assert data[k].dtype == np.float32, f"{k} must not be downcast"


def test_dataset_upcasts_fp16_to_fp32():
    """ShardDataset.__getitem__ already calls .float() on every float key."""
    import tempfile
    from pathlib import Path
    from ptcg_il.shard_writer import _write_shard
    from ptcg_il.train.dataset import ShardDataset

    ep = _make_synthetic_episode(1, decks={0: 1, 1: 30})
    buffer = _build_shard_buffer(ep, "train", ...)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _write_shard("train", 0, buffer, td)
        (td / "meta.parquet").write_bytes(  # ShardDataset requires this
            b"TODO: write minimal meta.parquet; see test_dataset.py's _build_synthetic_data"
        )
        ds = ShardDataset(td, split="train")
        sample = ds[0]
    assert sample["opt_card_feat"].dtype == torch.float32
```

**The `_build_shard_buffer` helper.** The existing `tests/test_shard_writer.py`
calls `_make_minimal_obs` + `featurize` to build a sample, then stacks them into
a buffer for `_write_shard`. Read the existing `_make_minimal_obs` (line 35) and
follow that pattern — a buffer is a `list[dict[str, np.ndarray]]` where each
dict is one `featurize()` return. The `...` in the param above means replacing
with the actual archetypes/vocab references those helpers need.


def test_fp16_round_trip_preserves_distinct_card_rows():
    """fp16 rounding two genuinely distinct cards onto one row would make
    qa._attachment_collision over-report.  Tightest case is damage/350.0:
    adjacent values differ by 2.9e-2 against fp16 precision of ~4.9e-4."""
    from ptcg_mine.cards import build_engine_card_features, load_engine
    cards, attacks = load_engine()
    feats = build_engine_card_features(cards, attacks)
    assert feats, "no cards examined"
    before = {v.tobytes() for v in feats.values()}
    after = {v.astype(np.float16).astype(np.float32).tobytes() for v in feats.values()}
    assert len(after) == len(before), (
        f"fp16 merged {len(before) - len(after)} distinct card rows"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_shard_writer.py -k fp16 -v
```

Expected: `float32 is not float16`.

- [ ] **Step 3: Implement**

In `python/ptcg_il/shard_writer.py`, above `_write_shard`:

```python
#: Keys stored as float16.  These are 89.8% of a shard's uncompressed bytes
#: (1937.9 MB of 2158.85 MB, measured on data/shards/test-00000.npz), and every
#: value in them is a one-hot, a small count, or a fixed-divisor ratio -- the
#: tightest case is damage/350.0, whose adjacent values differ by 2.9e-2 against
#: fp16 precision of ~4.9e-4 near 0.37.  Two orders of margin.
#:
#: ShardDataset.__getitem__ already calls .float() on every non-int, non-bool
#: key, so the read path needs no change.
_FP16_KEYS = frozenset({
    "poke_card_feat", "hand_card_feat", "stadium_card_feat",
    "context_card_feat", "effect_card_feat", "discard_card_feat",
    "prize_card_feat", "opt_card_feat", "opt_attack_feat", "log_card_feat",
})
```

and in `_write_shard`, replace the stacking loop:

```python
    for k in keys:
        arrays = [s[k] for s in buffer]
        stacked_k = np.stack(arrays, axis=0)
        if k in _FP16_KEYS:
            stacked_k = stacked_k.astype(np.float16)
        stacked[k] = stacked_k
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_shard_writer.py tests/test_dataset.py -v
```

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_il/shard_writer.py python/tests/test_shard_writer.py python/tests/test_dataset.py
git commit -m "perf(il): store shard card features as float16"
```

---

### Task 10: Stamp fingerprints, full rebuild, and the size/memory gates

The mechanical success criteria from spec §8.3 and §8.4. Nothing here is a code
feature — it is the verification that the previous nine tasks produce a corpus
that trains.

**Files:**
- Modify: `python/ptcg_mine/stamp.py:60-67` and `:89-96`
- Test: `tests/test_stamp.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_keywords_and_damage_are_in_the_mine_fingerprint():
    """Editing a regex must invalidate engine_card_features.npy.  Without this
    the stamp serves a stale artifact and the new columns are silently zero."""
    from ptcg_mine.stamp import STAGES
    code = STAGES["mine"].code
    assert "ptcg_mine/keywords.py" in code
    assert "ptcg_mine/damage.py" in code


def test_keywords_is_in_the_shards_fingerprint():
    """featurizer.py is already covered, but the keyword block reaches a shard
    through engine_card_features.npy, which featurizer.py does not build."""
    from ptcg_mine.stamp import STAGES
    assert "ptcg_mine/keywords.py" in STAGES["shards"].code
```

The two lists are `ptcg_mine/stamp.py`'s module-level `_MINE_CODE` tuple
(line 59) and the inline `code=(...)` tuple of `STAGES["shards"]` (line 88).
Both are reachable as `STAGES[<name>].code`.

- [ ] **Step 2: Run test to verify it fails**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_stamp.py -k fingerprint -v
```

- [ ] **Step 3: Add the source files to both fingerprints**

In `python/ptcg_mine/stamp.py`, add `"ptcg_mine/keywords.py"` and
`"ptcg_mine/damage.py"` to the mine list (after `"ptcg_mine/cards.py"`, line 65)
and `"ptcg_mine/keywords.py"` to the shards list (after
`"ptcg_il/featurizer.py"`, line 90).

- [ ] **Step 4: Run both full test suites**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q
cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q
```

Expected: all PASS. Report any failure rather than adjusting the test to match.

- [ ] **Step 5: Rebuild the artifacts and shards**

```bash
cd python
uv run python -m ptcg_mine.mine --skip-download --raw-dir raw --out-dir data --force
uv run python -m ptcg_il.cli build-shards --data-dir data --force
```

Confirm `data/evolution_map.npy` exists and
`np.load("data/engine_card_features.npy", allow_pickle=True).item()` yields
212-wide rows.

- [ ] **Step 6: Check the size gate (spec §8.3)**

```bash
cd python && uv run python -c "
import numpy as np, glob, os
f = sorted(glob.glob('data/shards/test-*.npz'))[0]
d = np.load(f)
tot = sum(d[k].nbytes for k in d.files)
print(f'{os.path.basename(f)}: {tot/1e6:.1f} MB uncompressed')
print('gate: <= 2442 MB (spec 7); today was 2158.9 MB')
"
```

Expected: ~2442 MB (+13%). If it materially exceeds that, the fp16 cast in
Task 9 is not covering every key — check `_FP16_KEYS` against the shard's
actual key list before proceeding.

- [ ] **Step 7: Check the peak-PSS gate (spec §8.4)**

Run a 60-step training run over the full corpus and sample PSS:

```bash
cd python
uv run python -m ptcg_il.cli train --data-dir data --out-dir /tmp/ckpt-smoke \
    --archetype-self 0 --total-steps 60 --no-wandb &
TRAIN_PID=$!
while kill -0 $TRAIN_PID 2>/dev/null; do
  awk '/^Pss:/ {s+=$2} END {print s/1024 " MB"}' /proc/$TRAIN_PID/smaps_rollup 2>/dev/null
  sleep 5
done
```

Expected peak: ~6.7 GB (spec §7 projection; today's baseline is 5.9 GB). A
result near 12 GB means fp16 is not reaching the mmap cache — delete
`data/shards/.mmap-cache/` and re-run, since it self-invalidates on size/mtime
but a stale cache would mask the change.

- [ ] **Step 8: Commit**

```bash
git add python/ptcg_mine/stamp.py python/tests/test_stamp.py
git commit -m "chore(mine): add keywords.py and damage.py to stage fingerprints"
```

- [ ] **Step 9: Record what the rebuild invalidated**

Every checkpoint under `python/checkpoints*` is now unloadable (`CardFeaturizer.mlp`
went `[94→256]` to `[212→256]`), `data/il_baselines.json` no longer matches any
checkpoint SHA, and `python/arena_ratings.json` is void. This is spec §10 and was
accepted deliberately. Re-training and re-recording the baseline is the next
piece of work, tracked separately — **it is not part of this plan.**

---

## Deferred to a follow-up

- Retraining the archetype-0 and archetype-2 specialists and comparing against a
  freshly recorded `data/il_baselines.json` (spec §8.5). This is the real
  success criterion, but it is hours of GPU time and belongs in its own cycle.
- Re-running stage 5 RL and the arena roster.
