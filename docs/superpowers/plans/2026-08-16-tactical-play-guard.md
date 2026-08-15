# Tactical Play Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ptcg_il/play_guard.py`, a second inference-time guard that stops the agent throwing away an attack, an attacker's energy, or damage counters — the three failures behind two measured live losses.

**Architecture:** Mirrors the existing `ptcg_il/deck_guard.py`. Hard rules edit `batch["opt_mask"]` in place (masks compose with `&`, and `select_multi`'s cloned picked-mask honours them for free); soft rules return adjusted logits. `DeckGuard.pick` stays the single owner of the final index. NumPy + stdlib only, vendored verbatim into the bundle.

**Tech Stack:** Python 3.11, uv, NumPy, pytest. PyTorch only at the call sites, never at `play_guard` module scope.

**Spec:** `docs/superpowers/specs/2026-08-16-tactical-play-guard-design.md`

## Global Constraints

- **Greedy inference only.** Do not modify `search_infer.py`, `MAIN_PY_TEMPLATE` (the MCTS template), or the Rust PUCT tree. MCTS is obsolete per the owner's decision.
- **No retraining.** Do not touch `featurizer.py`. Any edit there is fingerprinted by `ptcg_mine.stamp` and silently redefines every checkpoint and shard.
- **`play_guard.py` imports nothing from `ptcg_il`.** It is copied verbatim into the bundle like `deck_guard.py` and `ref_map.py`, so it must be self-contained: NumPy + stdlib, `import torch` only inside the one function that needs it.
- **Never read the `energies` key.** Resolve energy types via `energyCards[i].id → engine_card_features[id][12:24]` argmax. Spec §2.3: whether `energies` holds types or card ids is undetermined by available data (byte-identical across 2864 observations).
- **`np.rint`, never `int()`,** on any `× ATKCOST_N` round-trip.
- **Every hard rule carries a never-empty floor.** If a mask would clear every legal option, do not apply it.
- **Tests that count occurrences must fail on zero examined.** Repo convention — a loop that never runs must not pass.
- **Validate each new guard by mutation.** Break it deliberately, confirm the test goes red.
- **pytest invocation:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest ...`, and the two test directories run **separately** (`tests/` and `python/tests/` share basenames and collide on collection).
- **Do not run git commands unless the owner asks.** Commit steps below are written out but must be confirmed before running.

### Verified constants (do not re-derive; copy these)

```python
ATKCOST_N = 5.0
CARD_ATTACK_BLOCK_START = 85     # card row attack blocks: 85 + 46*i
F_ATK = 46                       # 85 + 3*46 = 223 = F_CARD
N_ENERGY = 12
CARD_ENERGY_TYPE_COL = 12        # row[12:24] energyType one-hot
CARD_EX_COL = 48                 # row[48]=ex, row[49]=megaEx  -> 2 prizes
CARD_ATTACK_COUNT_COL = 82       # 52 + K_EFFECT(29) + 1; = min(len(attacks),3)/3
O_MAX = 64
```

Engine enums (`cg/api.py`):

```python
# EnergyType
E_COLORLESS, E_RAINBOW, E_TEAM_ROCKET = 0, 10, 11
# OptionType
OT_YES, OT_NO, OT_CARD, OT_ATTACH = 1, 2, 3, 8
OT_ABILITY, OT_RETREAT, OT_ATTACK, OT_END = 10, 12, 13, 14
# SelectType
ST_MAIN = 0
# SelectContext
SC_MAIN, SC_DAMAGE_COUNTER, SC_DAMAGE_COUNTER_ANY, SC_DAMAGE = 0, 13, 14, 15
# AreaType
AREA_HAND, AREA_ACTIVE, AREA_BENCH = 2, 4, 5
```

### The column-index invariant (load-bearing — read before Task 2)

`featurize` can reorder options via `index_remap`, but **at inference it never does**. With `action=None` the `chosen` set is empty, so `indices_to_keep` fills in ascending order and `index_remap` is the identity. Therefore batch column *j* is raw `select["option"][j]`, for `j < n_opts`.

Two consequences the code must respect:

1. `n_opts = min(len(select["option"]), O_MAX - 1 if wants_stop else O_MAX)`. Options past that are **dropped**, so never index `select["option"]` by a column without bounds-checking.
2. When `stop_column >= 0` it equals `n_opts` — that column is the STOP pseudo-option, **not** a raw option. Skip it in every rule.

Task 2 asserts this invariant rather than assuming it.

---

## File Structure

| file | responsibility |
|---|---|
| `python/ptcg_il/play_guard.py` | **new.** Engine-data helpers, `TacticsConfig`, `TacticsStats`, `PlayGuard` (`apply_mask` for hard rules, `rerank` for soft rules). Self-contained. |
| `python/tests/test_play_guard.py` | **new.** Unit tests per rule + data-contract tests. |
| `python/tests/test_play_guard_replays.py` | **new.** Regression against the two real episodes. |
| `scripts/derive_low_hp_frac.py` | **new.** Corpus scan producing `low_hp_frac` (spec §4.1). |
| `scripts/derive_play_guard_margin.py` | **new.** Logit-gap scan producing `margin` (spec §4.2). |
| `python/ptcg_il/live_eval.py` | modify: `use_play_guard` flag, `_ensure_play_guard`, call sites. |
| `scripts/build_submission.py` | modify: copy `play_guard.py`; wire into `MAIN_PY_TEMPLATE_GREEDY` only. |
| `main.py` (repo root) | modify: mirror the greedy template. |

---

## Task 1: Engine-data helpers and their contracts

**Files:**
- Create: `python/ptcg_il/play_guard.py`
- Test: `python/tests/test_play_guard.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `attack_cost_hist(card_row, block: int) -> np.ndarray` — int64[12]
  - `attack_block_count(card_row) -> int`
  - `energy_type_of_card(card_row) -> int`
  - `attached_energy_types(pokemon: dict, engine_card_features) -> list[int]`
  - `is_two_prize(card_row) -> bool`
  - `unmet_cost(cost_hist, attached: list[int]) -> int`
  - module constants from Global Constraints above.

- [ ] **Step 1: Write the failing contract tests**

```python
# python/tests/test_play_guard.py
"""Tests for ptcg_il/play_guard.py — inference-time tactical guard.

R1a/R1c/R4 are hard mask rules; R2/R3 are gated near-tie re-ranks.  See
docs/superpowers/specs/2026-08-16-tactical-play-guard-design.md.
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def engine():
    """Real engine card + attack data. Skips if the vendored engine is absent."""
    import sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "pokemon-tcg-ai-battle" / "sample_submission"
                          / "sample_submission"))
    try:
        from cg.api import all_card_data, all_attack, CardType
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"vendored engine unavailable: {exc}")
    from ptcg_mine.cards import card_static_row
    attacks = {a.attackId: a for a in all_attack()}
    cards = all_card_data()
    rows = {c.cardId: card_static_row(c, attacks) for c in cards}
    return {"cards": {c.cardId: c for c in cards}, "attacks": attacks,
            "rows": rows, "CardType": CardType}


class TestEngineDataContracts:
    def test_cost_recovers_exactly_from_every_card_attack_block(self, engine):
        from ptcg_il.play_guard import attack_cost_hist, attack_block_count
        checked = 0
        for cid, card in engine["cards"].items():
            if int(card.cardType) != int(engine["CardType"].POKEMON):
                continue
            row = engine["rows"][cid]
            for i, aid in enumerate((getattr(card, "attacks", []) or [])[:3]):
                atk = engine["attacks"].get(int(aid))
                if atk is None:
                    continue
                want = [list(atk.energies).count(t) for t in range(12)]
                assert list(attack_cost_hist(row, i)) == want, (cid, aid)
                checked += 1
        assert checked >= 1000, f"examined only {checked} attack blocks"

    def test_attack_block_count_matches_the_card(self, engine):
        from ptcg_il.play_guard import attack_block_count
        checked = 0
        for cid, card in engine["cards"].items():
            if int(card.cardType) != int(engine["CardType"].POKEMON):
                continue
            want = min(len(getattr(card, "attacks", []) or []), 3)
            assert attack_block_count(engine["rows"][cid]) == want, cid
            checked += 1
        assert checked >= 500, f"examined only {checked} pokemon"

    def test_no_pokemon_exceeds_three_attack_blocks(self, engine):
        """The 3-block cap must never truncate. Fails loudly if a set breaks it."""
        over = [c.cardId for c in engine["cards"].values()
                if len(getattr(c, "attacks", []) or []) > 3]
        assert over == [], f"cards with >3 attacks would be truncated: {over}"

    def test_two_prize_flag_matches_ex_and_mega_ex(self, engine):
        from ptcg_il.play_guard import is_two_prize
        checked = 0
        for cid, card in engine["cards"].items():
            want = bool(card.ex) or bool(card.megaEx)
            assert is_two_prize(engine["rows"][cid]) == want, cid
            checked += 1
        assert checked >= 1000, f"examined only {checked} cards"

    def test_energy_type_of_special_energy_is_not_its_card_id(self, engine):
        """Boomerang(9)->COLORLESS, Legacy(12)->RAINBOW. Reading the card id as
        a type would give DRAGON and TEAM_ROCKET. This is why the guard never
        reads the `energies` key (spec 2.3)."""
        from ptcg_il.play_guard import energy_type_of_card
        assert energy_type_of_card(engine["rows"][9]) == 0    # COLORLESS
        assert energy_type_of_card(engine["rows"][12]) == 10  # RAINBOW
        assert energy_type_of_card(engine["rows"][2]) == 2    # Basic {R} = FIRE
        assert energy_type_of_card(engine["rows"][5]) == 5    # Basic {P} = PSYCHIC

    def test_guard_source_never_reads_the_energies_key(self):
        """Structural: a future edit reintroducing it fails here."""
        import pathlib, ptcg_il.play_guard as pg
        src = pathlib.Path(pg.__file__).read_text()
        assert '"energies"' not in src and "'energies'" not in src


class TestUnmetCost:
    def test_exact_colors_are_met(self):
        from ptcg_il.play_guard import unmet_cost
        cost = [0] * 12
        cost[2] = 1; cost[5] = 1          # Phantom Dive: Fire + Psychic
        assert unmet_cost(cost, [2, 5]) == 0

    def test_missing_color_counts_as_unmet(self):
        from ptcg_il.play_guard import unmet_cost
        cost = [0] * 12
        cost[2] = 1; cost[5] = 1
        assert unmet_cost(cost, [2]) == 1
        assert unmet_cost(cost, []) == 2

    def test_colorless_is_paid_by_any_surplus(self):
        from ptcg_il.play_guard import unmet_cost
        cost = [0] * 12
        cost[0] = 1                        # Jet Headbutt: one Colorless
        assert unmet_cost(cost, [7]) == 0  # a Darkness pays it
        assert unmet_cost(cost, []) == 1

    def test_colored_need_is_not_paid_by_the_wrong_color(self):
        from ptcg_il.play_guard import unmet_cost
        cost = [0] * 12
        cost[2] = 2
        assert unmet_cost(cost, [7, 7, 7]) == 2

    def test_rainbow_supplies_any_colour(self):
        from ptcg_il.play_guard import unmet_cost
        cost = [0] * 12
        cost[2] = 1; cost[5] = 1
        assert unmet_cost(cost, [10, 10]) == 0

    def test_team_rocket_supplies_psychic_or_darkness_only(self):
        from ptcg_il.play_guard import unmet_cost
        cost = [0] * 12
        cost[5] = 1
        assert unmet_cost(cost, [11]) == 0
        cost2 = [0] * 12
        cost2[2] = 1
        assert unmet_cost(cost2, [11]) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ptcg_il.play_guard'`

- [ ] **Step 3: Write the module**

```python
# python/ptcg_il/play_guard.py
"""Inference-time tactical guard (agent side only — no retraining).

Two measured live losses motivate this module.  In episode 93320914 the agent
held a lethal Phantom Dive against a 130/300 Archaludon ex and played Boss's
Orders instead, gusting the KO-able target away; it also spent five damage
counters on a Relicanth already at 0 HP.  In episode 93323813 it never once
assembled Phantom Dive's Fire+Psychic cost across 24 turns, attaching ~5 energy
and discarding 5 — four of them to retreats taken while an attack was on the
menu.

Five rules, split by whether the claim is provable:

- **R1a (hard).**  END and RETREAT are masked while a lethal attack is legal.
  Both alternatives forfeit the attack outright, so a KO cannot be worse.
- **R1c (hard).**  Gust cards are masked when the opponent's active is both
  KO-able and worth two prizes.  Moving a 2-prize KO off the active spot trades
  two prizes for one.
- **R2 (soft).**  RETREAT is demoted within a logit margin when the active is
  loaded, can attack now, and is not about to be knocked out.
- **R3 (soft).**  ATTACH is promoted within a logit margin toward a Pokémon
  whose attack cost is actually unmet and whose HP is not low.
- **R4 (hard).**  Damage counters are never placed on a target already at 0 HP.

Why the naive "if lethal is legal, attack" rule is *absent*: MAIN is
re-presented after every sub-action (10-30 times per turn in the replays) and
attacking ends the turn, so forcing the attack at the first opportunity
forfeits every remaining attachment, ability and Supporter.  R1a is the
provable subset of it.

Energy types are resolved through ``energyCards[i].id`` and the card table's
``energyType`` one-hot, *never* through the ``energies`` key: basic energy card
ids 1-8 coincide exactly with their EnergyType, so the two readings are
indistinguishable on observed data, and they differ for special energy
(Boomerang is card 9 / COLORLESS, Legacy is card 12 / RAINBOW).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ── Static layouts (ptcg_mine/cards.py, ptcg_il/featurizer.py) ────────────
# Repeated here rather than imported so this module stays self-contained and
# vendors into the submission bundle verbatim, like deck_guard.py/ref_map.py.
ATKCOST_N = 5.0
CARD_ATTACK_BLOCK_START = 85
F_ATK = 46
MAX_ATTACK_BLOCKS = 3
N_ENERGY = 12
CARD_ENERGY_TYPE_COL = 12       # row[12:24] energyType one-hot
CARD_EX_COL = 48
CARD_MEGA_EX_COL = 49
CARD_ATTACK_COUNT_COL = 82      # 52 + K_EFFECT(29) + 1; min(len(attacks),3)/3
O_MAX = 64

# ── Engine enums (cg/api.py) ──────────────────────────────────────────────
E_COLORLESS, E_RAINBOW, E_TEAM_ROCKET = 0, 10, 11
#: TEAM_ROCKET energy provides PSYCHIC or DARKNESS.
_TEAM_ROCKET_SUPPLIES = (5, 7)

OT_YES, OT_NO, OT_CARD, OT_ATTACH = 1, 2, 3, 8
OT_ABILITY, OT_RETREAT, OT_ATTACK, OT_END = 10, 12, 13, 14

ST_MAIN = 0
SC_MAIN = 0
SC_DAMAGE_COUNTER, SC_DAMAGE_COUNTER_ANY, SC_DAMAGE = 13, 14, 15
_COUNTER_CONTEXTS = (SC_DAMAGE_COUNTER, SC_DAMAGE_COUNTER_ANY, SC_DAMAGE)

AREA_HAND, AREA_ACTIVE, AREA_BENCH = 2, 4, 5

#: opt_scalar[:, 7] is the featurizer's KO flag (damage >= target current HP),
#: stored float32; compare with slack, never == 1.0.
_KO_EPS = 1e-6
_OPT_KO_COL = 7

#: Cards that switch in one of the opponent's Benched Pokemon, enumerated from
#: every card's ``skills[].text`` (the text is on `skills`, *not* `card.text`,
#: which is empty for trainers) matching "opponent" + "switch in".
GUST_PLAY_CARDS = frozenset({
    1088,   # Prime Catcher (ACE SPEC)
    1124,   # Pokemon Catcher (coin flip)
    1182,   # Boss's Orders
    1204,   # Lisia's Appeal (Basic only)
    1218,   # Team Rocket's Giovanni
})
#: Meowstic's Ability does nothing *but* gust, so masking it costs nothing.
GUST_ABILITY_CARDS = frozenset({221})
#: Hop's Dubwool (310) and Hariyama (674) gust as a "you may" rider on an
#: evolution that is valuable in its own right.  Deliberately NOT masked --
#: blocking the evolve costs more than the gust does.  Declined at the YES_NO.
GUST_ON_EVOLVE_CARDS = frozenset({310, 674})


def attack_block_count(card_row) -> int:
    """How many attack blocks this card row actually carries (0..3)."""
    return int(round(float(card_row[CARD_ATTACK_COUNT_COL]) * MAX_ATTACK_BLOCKS))


def attack_cost_hist(card_row, block: int) -> np.ndarray:
    """int64[12] energy-cost histogram of attack *block* on this card row.

    ``np.rint``, not ``int()``: the value is a float32 round-trip through
    ``/ATKCOST_N``, and truncation is the same class of bug as the KO flag's
    ``1.0 - 1e-6`` comparison in deck_guard.
    """
    off = CARD_ATTACK_BLOCK_START + block * F_ATK
    raw = np.asarray(card_row[off + 1: off + 1 + N_ENERGY], dtype=np.float64)
    return np.rint(raw * ATKCOST_N).astype(np.int64)


def energy_type_of_card(card_row) -> int:
    """EnergyType this card provides, from the row[12:24] one-hot."""
    one_hot = np.asarray(
        card_row[CARD_ENERGY_TYPE_COL:CARD_ENERGY_TYPE_COL + N_ENERGY])
    return int(np.argmax(one_hot))


def is_two_prize(card_row) -> bool:
    """True when knocking this Pokemon out awards two prizes (ex / Mega ex)."""
    return bool(float(card_row[CARD_EX_COL]) > 0.0
                or float(card_row[CARD_MEGA_EX_COL]) > 0.0)


def attached_energy_types(pokemon: dict, engine_card_features) -> list[int]:
    """EnergyTypes attached to *pokemon*, resolved via ``energyCards`` ids.

    Never reads the ``energies`` key -- see the module docstring.  A card the
    table does not know contributes nothing rather than an invented type; that
    under-counts, which makes R2/R3 fire less often, never more aggressively.
    """
    if engine_card_features is None:
        return []
    out: list[int] = []
    for card in (pokemon.get("energyCards") or []):
        row = engine_card_features.get(int(card.get("id", -1)))
        if row is None:
            continue
        out.append(energy_type_of_card(row))
    return out


def unmet_cost(cost_hist, attached: list[int]) -> int:
    """How many more energy *this* attack needs, given what is attached.

    Colored requirements are matched first; ``COLORLESS`` is then paid from
    whatever is left over.  ``RAINBOW`` supplies any type and ``TEAM_ROCKET``
    supplies PSYCHIC or DARKNESS, so both are held back and spent only against
    a requirement nothing else covers.
    """
    cost = [int(c) for c in cost_hist]
    pool: dict[int, int] = {}
    wild = 0
    flexible: list[int] = []
    for t in attached:
        t = int(t)
        if t == E_RAINBOW:
            wild += 1
        elif t == E_TEAM_ROCKET:
            flexible.append(t)
        else:
            pool[t] = pool.get(t, 0) + 1

    need = 0
    for t in range(1, N_ENERGY):
        if t in (E_RAINBOW, E_TEAM_ROCKET):
            continue
        want = cost[t]
        have = pool.get(t, 0)
        used = min(want, have)
        want -= used
        pool[t] = have - used
        if want and t in _TEAM_ROCKET_SUPPLIES:
            used = min(want, len(flexible))
            for _ in range(used):
                flexible.pop()
            want -= used
        used = min(want, wild)
        wild -= used
        want -= used
        need += want

    colorless = cost[E_COLORLESS]
    surplus = sum(pool.values()) + wild + len(flexible)
    need += max(0, colorless - surplus)
    return need
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -v`
Expected: PASS, all of `TestEngineDataContracts` and `TestUnmetCost`.

- [ ] **Step 5: Mutation check**

Change `np.rint` to `np.trunc` in `attack_cost_hist`, rerun, confirm `test_cost_recovers_exactly_from_every_card_attack_block` still passes (float32 happens to round up here), then change it to `raw * ATKCOST_N - 0.01` and confirm it goes **red**. Revert. Record in the commit message that `trunc` alone is not caught, which is why the constant is documented rather than relied upon.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_il/play_guard.py python/tests/test_play_guard.py
git commit -m "feat(play_guard): engine-data helpers for attack cost, energy type, prize value"
```

---

## Task 2: `PlayGuard` skeleton, the column invariant, and R4

**Files:**
- Modify: `python/ptcg_il/play_guard.py`
- Test: `python/tests/test_play_guard.py`

**Interfaces:**
- Consumes: Task 1's helpers.
- Produces:
  - `TacticsConfig` (frozen dataclass), `TacticsStats` (dataclass)
  - `PlayGuard(config, engine_card_features=None, stats=None)`
  - `PlayGuard.apply_mask(batch, obs_dict, *, sel_type: int, sel_ctx: int) -> None`
  - `option_pokemon(obs_dict, option: dict) -> dict | None`
  - `n_option_columns(batch, obs_dict) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# append to python/tests/test_play_guard.py

def _batch(n_opts, *, mask=None, ko=None, otype=None, card_id=None,
           stop_column=-1):
    """Batch-1 tensors shaped like the featurizer's output."""
    import torch
    O = 64
    m = np.zeros(O, dtype=bool); m[:n_opts] = True
    if mask is not None:
        m[:len(mask)] = mask
    sc = np.zeros((O, 14), dtype=np.float32)
    if ko is not None:
        sc[:len(ko), 7] = ko
    ot = np.zeros(O, dtype=np.int64)
    if otype is not None:
        ot[:len(otype)] = otype
    cid = np.zeros(O, dtype=np.int64)
    if card_id is not None:
        cid[:len(card_id)] = card_id
    return {
        "opt_mask": torch.from_numpy(m).unsqueeze(0),
        "opt_scalar": torch.from_numpy(sc).unsqueeze(0),
        "opt_type": torch.from_numpy(ot).unsqueeze(0),
        "opt_card_id": torch.from_numpy(cid).unsqueeze(0),
        "stop_column": torch.tensor([stop_column]),
    }


def _obs(my=None, opp=None, select=None):
    my = my or {}
    opp = opp or {}
    return {"current": {"yourIndex": 0, "players": [my, opp]},
            "select": select or {"option": [], "type": 0, "context": 0,
                                 "minCount": 1, "maxCount": 1}}


def _poke(cid=121, hp=320, maxhp=320, energy_ids=()):
    return {"id": cid, "hp": hp, "maxHp": maxhp,
            "energyCards": [{"id": e} for e in energy_ids]}


class TestColumnInvariant:
    def test_inference_remap_is_identity(self):
        """action=None => chosen empty => indices fill in order => identity.
        If this ever stops holding, every rule indexes the wrong option."""
        from ptcg_il.featurizer import _build_option_tensors
        select = {"option": [{"type": 14} for _ in range(80)],
                  "type": 0, "context": 0, "minCount": 1, "maxCount": 1}
        out = _build_option_tensors(select, None, {}, 0, {}, None, None)
        index_remap = out[8]
        assert index_remap, "examined zero options"
        assert all(k == v for k, v in index_remap.items()), index_remap

    def test_n_option_columns_excludes_the_stop_column(self):
        from ptcg_il.play_guard import n_option_columns
        obs = _obs(select={"option": [{"type": 14}, {"type": 13}],
                           "type": 0, "context": 0,
                           "minCount": 0, "maxCount": 1})
        b = _batch(2, stop_column=2)
        assert n_option_columns(b, obs) == 2


class TestR4DeadTarget:
    """R1c/R4 fix ep 93320914[140]-[144]: five counters into a 0-HP Relicanth
    while two 130-HP Duraludon were legal targets."""

    @staticmethod
    def _guard():
        from ptcg_il.play_guard import PlayGuard, TacticsConfig
        return PlayGuard(TacticsConfig(), engine_card_features={})

    def _counter_obs(self, hps):
        opp = {"active": [], "bench": [_poke(hp=h, maxhp=100) for h in hps]}
        opts = [{"type": 3, "area": 5, "index": i, "playerIndex": 1}
                for i in range(len(hps))]
        return _obs(opp=opp, select={"option": opts, "type": 1, "context": 14,
                                     "minCount": 1, "maxCount": 1})

    def test_dead_target_is_masked(self):
        obs = self._counter_obs([0, 130, 160])
        b = _batch(3, otype=[3, 3, 3])
        self._guard().apply_mask(b, obs, sel_type=1, sel_ctx=14)
        assert list(b["opt_mask"][0][:3].numpy()) == [False, True, True]

    def test_negative_hp_target_is_masked(self):
        obs = self._counter_obs([-40, 130])
        b = _batch(2, otype=[3, 3])
        self._guard().apply_mask(b, obs, sel_type=1, sel_ctx=14)
        assert list(b["opt_mask"][0][:2].numpy()) == [False, True]

    def test_live_targets_are_untouched(self):
        obs = self._counter_obs([10, 130])
        b = _batch(2, otype=[3, 3])
        self._guard().apply_mask(b, obs, sel_type=1, sel_ctx=14)
        assert list(b["opt_mask"][0][:2].numpy()) == [True, True]

    def test_floor_holds_when_every_target_is_dead(self):
        """A spread with more counters than live targets must still be legal."""
        obs = self._counter_obs([0, 0])
        b = _batch(2, otype=[3, 3])
        self._guard().apply_mask(b, obs, sel_type=1, sel_ctx=14)
        assert b["opt_mask"][0][:2].any(), "floor cleared the whole mask"

    def test_rule_is_inert_outside_counter_contexts(self):
        obs = self._counter_obs([0, 130])
        b = _batch(2, otype=[3, 3])
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert list(b["opt_mask"][0][:2].numpy()) == [True, True]

    def test_disabled_by_config(self):
        from ptcg_il.play_guard import PlayGuard, TacticsConfig
        g = PlayGuard(TacticsConfig(enable_dead_target=False),
                      engine_card_features={})
        obs = self._counter_obs([0, 130])
        b = _batch(2, otype=[3, 3])
        g.apply_mask(b, obs, sel_type=1, sel_ctx=14)
        assert list(b["opt_mask"][0][:2].numpy()) == [True, True]

    def test_stats_count_what_was_masked(self):
        g = self._guard()
        obs = self._counter_obs([0, 0, 130])
        b = _batch(3, otype=[3, 3, 3])
        g.apply_mask(b, obs, sel_type=1, sel_ctx=14)
        assert g.stats.r4_masked == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -k "ColumnInvariant or R4" -v`
Expected: FAIL — `ImportError: cannot import name 'PlayGuard'`

- [ ] **Step 3: Implement**

```python
# append to python/ptcg_il/play_guard.py

@dataclass(frozen=True)
class TacticsConfig:
    """Rule switches and thresholds.

    ``low_hp_frac`` and the two margins are **measured**, not chosen -- see
    scripts/derive_low_hp_frac.py and scripts/derive_play_guard_margin.py, and
    spec section 4.  They are left at sentinel ``-1.0`` until those scripts have
    run; a guard constructed with a sentinel raises rather than guessing.
    """

    enable_lethal_end: bool = True      # R1a
    enable_lethal_gust: bool = True     # R1c
    enable_retreat: bool = True         # R2
    enable_attach: bool = True          # R3
    enable_dead_target: bool = True     # R4
    low_hp_frac: float = -1.0
    retreat_margin: float = -1.0
    attach_margin: float = -1.0


@dataclass
class TacticsStats:
    decisions: int = 0
    r1a_fired: int = 0
    r1c_fired: int = 0
    r2_gated: int = 0
    r2_changed: int = 0
    r3_gated: int = 0
    r3_changed: int = 0
    r4_masked: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=50))


def n_option_columns(batch, obs_dict) -> int:
    """Number of batch columns that map to real options.

    Excludes the STOP pseudo-column (which sits at ``stop_column == n_opts``)
    and respects the featurizer's O_MAX truncation.
    """
    select = (obs_dict.get("select") or {})
    n_raw = len(select.get("option") or [])
    stop = int(batch["stop_column"][0]) if "stop_column" in batch else -1
    cap = (O_MAX - 1) if stop >= 0 else O_MAX
    n = min(n_raw, cap)
    if stop >= 0:
        n = min(n, stop)
    return n


def option_pokemon(obs_dict, option: dict):
    """The in-play Pokemon an option refers to, or None.

    Reads ``area``/``index``/``playerIndex`` for CARD-style options and
    ``inPlayArea``/``inPlayIndex`` for ATTACH-style ones.
    """
    cur = obs_dict.get("current") or {}
    players = cur.get("players") or []
    me = int(cur.get("yourIndex", 0))
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    if area is None or index is None:
        return None
    owner = int(option.get("playerIndex", me))
    if owner >= len(players):
        return None
    player = players[owner] or {}
    if int(area) == AREA_ACTIVE:
        active = player.get("active") or []
        return active[0] if active else None
    if int(area) == AREA_BENCH:
        bench = player.get("bench") or []
        return bench[int(index)] if 0 <= int(index) < len(bench) else None
    return None


class PlayGuard:
    """Batch-dict adapter, mirroring deck_guard.DeckGuard.

    ``apply_mask`` holds the hard rules and edits ``opt_mask`` in place, so both
    the single-select ``masked_fill`` and ``select_multi``'s cloned picked-mask
    honour them.  ``rerank`` (Tasks 6-7) holds the soft rules and returns
    adjusted logits.  The final index is chosen by ``DeckGuard.pick``, which
    stays the single owner.
    """

    def __init__(self, config: TacticsConfig | None = None,
                 engine_card_features=None,
                 stats: TacticsStats | None = None) -> None:
        self.config = config or TacticsConfig()
        self.engine_card_features = engine_card_features
        self.stats = stats or TacticsStats()

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _np(batch: dict, key: str) -> np.ndarray:
        return batch[key][0].detach().cpu().numpy()

    def _row(self, card_id):
        if self.engine_card_features is None:
            return None
        return self.engine_card_features.get(int(card_id))

    @staticmethod
    def _apply_kill(batch: dict, kill: np.ndarray) -> int:
        """Mask ``kill`` out, unless doing so would clear every legal option.

        Returns how many columns were actually masked.  The floor exists for
        the same reason deck_guard's does: an empty mask turns argmax into an
        arbitrary engine-legal pick, which is worse than the thing prevented.
        """
        import torch

        mask = batch["opt_mask"][0]
        legal = mask.detach().cpu().numpy().astype(bool)
        kill = kill & legal
        if not kill.any():
            return 0
        if not (legal & ~kill).any():
            return 0
        mask &= ~torch.from_numpy(kill).to(mask.device)
        return int(kill.sum())

    # ---- hard rules ----------------------------------------------------
    def apply_mask(self, batch: dict, obs_dict: dict, *, sel_type: int,
                   sel_ctx: int) -> None:
        self.stats.decisions += 1
        if int(sel_ctx) in _COUNTER_CONTEXTS:
            self._mask_dead_targets(batch, obs_dict)

    def _mask_dead_targets(self, batch: dict, obs_dict: dict) -> None:
        """R4 -- a target already at 0 HP absorbs counters for nothing."""
        if not self.config.enable_dead_target:
            return
        options = (obs_dict.get("select") or {}).get("option") or []
        n = n_option_columns(batch, obs_dict)
        if n <= 0:
            return
        kill = np.zeros(len(self._np(batch, "opt_mask")), dtype=bool)
        for j in range(n):
            poke = option_pokemon(obs_dict, options[j])
            if poke is not None and int(poke.get("hp", 1)) <= 0:
                kill[j] = True
        self.stats.r4_masked += self._apply_kill(batch, kill)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Mutation check**

Delete the `if not (legal & ~kill).any(): return 0` floor; confirm `test_floor_holds_when_every_target_is_dead` goes red. Restore.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_il/play_guard.py python/tests/test_play_guard.py
git commit -m "feat(play_guard): PlayGuard skeleton, column invariant, R4 dead-target mask"
```

---

## Task 3: R1a — never end or retreat away a lethal attack

**Files:**
- Modify: `python/ptcg_il/play_guard.py`
- Test: `python/tests/test_play_guard.py`

**Interfaces:**
- Consumes: `PlayGuard`, `n_option_columns`, `_apply_kill` from Task 2.
- Produces: `PlayGuard._mask_end_and_retreat(batch, obs_dict)`, dispatched from `apply_mask` when `sel_type == ST_MAIN`.

- [ ] **Step 1: Write the failing tests**

```python
# append to python/tests/test_play_guard.py

class TestR1aLethalEndRetreat:
    """END and RETREAT both forfeit the attack, so replacing them with a KO
    cannot be worse.  Deliberately narrow: forcing the attack over *any* option
    would forfeit the rest of the turn, since MAIN is re-presented after every
    sub-action and attacking ends the turn."""

    @staticmethod
    def _guard(**kw):
        from ptcg_il.play_guard import PlayGuard, TacticsConfig
        return PlayGuard(TacticsConfig(**kw), engine_card_features={})

    def _main(self, otypes, ko):
        opts = [{"type": t} for t in otypes]
        obs = _obs(select={"option": opts, "type": 0, "context": 0,
                           "minCount": 1, "maxCount": 1})
        return obs, _batch(len(otypes), otype=otypes, ko=ko)

    def test_end_and_retreat_masked_when_a_lethal_attack_is_legal(self):
        obs, b = self._main([7, 13, 12, 14], [0, 1.0, 0, 0])
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert list(b["opt_mask"][0][:4].numpy()) == [True, True, False, False]

    def test_untouched_when_the_attack_is_not_lethal(self):
        obs, b = self._main([7, 13, 12, 14], [0, 0.4, 0, 0])
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert list(b["opt_mask"][0][:4].numpy()) == [True, True, True, True]

    def test_untouched_when_no_attack_is_legal(self):
        obs, b = self._main([7, 12, 14], [0, 0, 0])
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert list(b["opt_mask"][0][:3].numpy()) == [True, True, True]

    def test_ko_flag_uses_epsilon_not_equality(self):
        """opt_scalar is float32; an exactly-lethal attack round-trips to
        0.99999994 and == 1.0 misses it (the KO-flag float32 trap)."""
        obs, b = self._main([13, 14], [np.float32(1.0) - np.float32(5e-8), 0])
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert not bool(b["opt_mask"][0][1])

    def test_illegal_attack_column_does_not_arm_the_rule(self):
        obs, b = self._main([13, 14], [1.0, 0])
        b["opt_mask"][0][0] = False           # attack not actually legal
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert bool(b["opt_mask"][0][1]), "END masked with no legal attack"

    def test_floor_holds_when_end_is_the_only_other_option(self):
        obs, b = self._main([13, 14], [1.0, 0])
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert b["opt_mask"][0][:2].any()

    def test_disabled_by_config(self):
        obs, b = self._main([13, 14], [1.0, 0])
        self._guard(enable_lethal_end=False).apply_mask(
            b, obs, sel_type=0, sel_ctx=0)
        assert list(b["opt_mask"][0][:2].numpy()) == [True, True]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -k R1a -v`
Expected: FAIL — END/RETREAT still `True`.

- [ ] **Step 3: Implement**

```python
# in play_guard.py, replace apply_mask's body with:

    def apply_mask(self, batch: dict, obs_dict: dict, *, sel_type: int,
                   sel_ctx: int) -> None:
        self.stats.decisions += 1
        if int(sel_ctx) in _COUNTER_CONTEXTS:
            self._mask_dead_targets(batch, obs_dict)
        if int(sel_type) == ST_MAIN:
            self._mask_end_and_retreat(batch, obs_dict)

# and add:

    def _lethal_attack_columns(self, batch: dict) -> np.ndarray:
        """bool[O] -- legal ATTACK options whose KO flag is set."""
        legal = self._np(batch, "opt_mask").astype(bool)
        otype = self._np(batch, "opt_type")
        ko = self._np(batch, "opt_scalar")[:, _OPT_KO_COL]
        return legal & (otype == OT_ATTACK) & (ko >= 1.0 - _KO_EPS)

    def _mask_end_and_retreat(self, batch: dict, obs_dict: dict) -> None:
        """R1a -- END and RETREAT forfeit an attack that is already paid for."""
        if not self.config.enable_lethal_end:
            return
        if not self._lethal_attack_columns(batch).any():
            return
        otype = self._np(batch, "opt_type")
        kill = (otype == OT_END) | (otype == OT_RETREAT)
        masked = self._apply_kill(batch, kill)
        if masked:
            self.stats.r1a_fired += 1
            self.stats.recent.append({"rule": "R1a", "masked": masked})
```

- [ ] **Step 4: Run to verify pass**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Mutation check**

Change `ko >= 1.0 - _KO_EPS` to `ko == 1.0`; confirm `test_ko_flag_uses_epsilon_not_equality` goes red. Restore. Then drop `legal &` from `_lethal_attack_columns`; confirm `test_illegal_attack_column_does_not_arm_the_rule` goes red. Restore.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_il/play_guard.py python/tests/test_play_guard.py
git commit -m "feat(play_guard): R1a mask END/RETREAT while a lethal attack is legal"
```

---

## Task 4: R1c — do not gust a KO-able two-prize target away

**Files:**
- Modify: `python/ptcg_il/play_guard.py`
- Test: `python/tests/test_play_guard.py`

**Interfaces:**
- Consumes: `_lethal_attack_columns`, `is_two_prize`, `GUST_PLAY_CARDS`, `GUST_ABILITY_CARDS`, `GUST_ON_EVOLVE_CARDS`.
- Produces: `PlayGuard._mask_gusts(batch, obs_dict)`, and `PlayGuard.decline_gust(obs_dict, sel_ctx) -> bool` for the on-evolve YES/NO.

- [ ] **Step 1: Write the failing tests**

```python
# append to python/tests/test_play_guard.py

class TestR1cGust:
    """ep 93320914[157]: Dragapult ex held Phantom Dive (200) against a
    130/300 Archaludon ex, played Boss's Orders instead, and at [158] gusted
    the KO-able 2-prize target off the active spot for a 1-prize Duraludon."""

    @staticmethod
    def _rows():
        # 121 Dragapult ex (2 prizes), 190 Archaludon ex (2 prizes),
        # 169 Duraludon (1 prize), 1182 Boss's Orders, 221 Meowstic.
        def row(ex=False):
            r = np.zeros(223, dtype=np.float32)
            if ex:
                r[48] = 1.0
            return r
        return {121: row(True), 190: row(True), 169: row(False),
                1182: row(), 221: row()}

    def _guard(self, **kw):
        from ptcg_il.play_guard import PlayGuard, TacticsConfig
        return PlayGuard(TacticsConfig(**kw), engine_card_features=self._rows())

    def _main(self, otypes, card_ids, ko, opp_active_id):
        opts = []
        for t, c in zip(otypes, card_ids):
            opts.append({"type": t} if t != 7 else {"type": 7, "index": 0})
        opp = {"active": [_poke(cid=opp_active_id, hp=130, maxhp=300)],
               "bench": []}
        obs = _obs(opp=opp, select={"option": opts, "type": 0, "context": 0,
                                    "minCount": 1, "maxCount": 1})
        return obs, _batch(len(otypes), otype=otypes, card_id=card_ids, ko=ko)

    def test_gust_play_masked_against_a_ko_able_two_prize_active(self):
        obs, b = self._main([7, 13, 14], [1182, 0, 0], [0, 1.0, 0], 190)
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert not bool(b["opt_mask"][0][0]), "Boss's Orders not masked"
        assert bool(b["opt_mask"][0][1]), "the attack must stay legal"

    def test_untouched_when_the_target_is_worth_one_prize(self):
        obs, b = self._main([7, 13, 14], [1182, 0, 0], [0, 1.0, 0], 169)
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert bool(b["opt_mask"][0][0])

    def test_untouched_when_the_attack_is_not_lethal(self):
        obs, b = self._main([7, 13, 14], [1182, 0, 0], [0, 0.5, 0], 190)
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert bool(b["opt_mask"][0][0])

    def test_non_gust_play_is_untouched(self):
        obs, b = self._main([7, 13, 14], [1152, 0, 0], [0, 1.0, 0], 190)
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert bool(b["opt_mask"][0][0]), "Poke Pad is not a gust"

    def test_meowstic_ability_is_masked(self):
        obs, b = self._main([10, 13, 14], [221, 0, 0], [0, 1.0, 0], 190)
        self._guard().apply_mask(b, obs, sel_type=0, sel_ctx=0)
        assert not bool(b["opt_mask"][0][0])

    def test_every_enumerated_gust_play_card_is_covered(self):
        from ptcg_il.play_guard import GUST_PLAY_CARDS
        assert GUST_PLAY_CARDS == frozenset({1088, 1124, 1182, 1204, 1218})
        checked = 0
        for cid in GUST_PLAY_CARDS:
            rows = self._rows(); rows[cid] = np.zeros(223, dtype=np.float32)
            from ptcg_il.play_guard import PlayGuard, TacticsConfig
            g = PlayGuard(TacticsConfig(), engine_card_features=rows)
            obs, b = self._main([7, 13, 14], [cid, 0, 0], [0, 1.0, 0], 190)
            g.apply_mask(b, obs, sel_type=0, sel_ctx=0)
            assert not bool(b["opt_mask"][0][0]), cid
            checked += 1
        assert checked == 5, f"examined {checked} gust cards"

    def test_on_evolve_gusts_are_not_masked(self):
        """Hop's Dubwool / Hariyama gust as a rider on a valuable evolution."""
        from ptcg_il.play_guard import GUST_ON_EVOLVE_CARDS, GUST_PLAY_CARDS
        assert GUST_ON_EVOLVE_CARDS == frozenset({310, 674})
        assert not (GUST_ON_EVOLVE_CARDS & GUST_PLAY_CARDS)

    def test_decline_gust_at_the_yes_no(self):
        from ptcg_il.play_guard import PlayGuard, TacticsConfig
        g = PlayGuard(TacticsConfig(), engine_card_features=self._rows())
        obs, b = self._main([13, 14], [0, 0], [1.0, 0], 190)
        assert g.decline_gust(obs, b, sel_ctx=0) is False   # not a YES_NO
```

- [ ] **Step 2: Run to verify failure**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -k R1c -v`
Expected: FAIL — gust options still legal.

- [ ] **Step 3: Implement**

```python
# in play_guard.py, extend apply_mask's MAIN branch:

        if int(sel_type) == ST_MAIN:
            self._mask_end_and_retreat(batch, obs_dict)
            self._mask_gusts(batch, obs_dict)

# and add:

    def _opponent_active(self, obs_dict):
        cur = obs_dict.get("current") or {}
        players = cur.get("players") or []
        me = int(cur.get("yourIndex", 0))
        if len(players) < 2:
            return None
        active = (players[1 - me] or {}).get("active") or []
        return active[0] if active else None

    def _mask_gusts(self, batch: dict, obs_dict: dict) -> None:
        """R1c -- moving a KO-able 2-prize active away trades 2 prizes for 1.

        Implemented at MAIN, not at the SWITCH select: once the gust card is
        played the switch is forced (minCount 1) and *every* option moves the
        active away, so there is nothing left to choose between.
        """
        if not self.config.enable_lethal_gust:
            return
        if not self._lethal_attack_columns(batch).any():
            return
        active = self._opponent_active(obs_dict)
        if active is None:
            return
        row = self._row(active.get("id", -1))
        if row is None or not is_two_prize(row):
            return

        otype = self._np(batch, "opt_type")
        card_id = self._np(batch, "opt_card_id")
        n = len(otype)
        kill = np.zeros(n, dtype=bool)
        for j in range(n):
            cid = int(card_id[j])
            if int(otype[j]) == OT_ABILITY and cid in GUST_ABILITY_CARDS:
                kill[j] = True
            elif cid in GUST_PLAY_CARDS and int(otype[j]) in (OT_CARD, 7):
                kill[j] = True
        masked = self._apply_kill(batch, kill)
        if masked:
            self.stats.r1c_fired += 1
            self.stats.recent.append({"rule": "R1c", "masked": masked,
                                      "target": int(active.get("id", -1))})

    def decline_gust(self, obs_dict: dict, batch: dict, *, sel_ctx: int) -> bool:
        """True when an optional on-evolve gust should be declined.

        Hop's Dubwool and Hariyama gust as a "you may" rider, so the evolve is
        never masked; the decision is deferred to this YES_NO instead.
        """
        if not self.config.enable_lethal_gust:
            return False
        select = obs_dict.get("select") or {}
        types = {int(o.get("type", -1)) for o in (select.get("option") or [])}
        if types != {OT_YES, OT_NO}:
            return False
        if not self._lethal_attack_columns(batch).any():
            return False
        active = self._opponent_active(obs_dict)
        if active is None:
            return False
        row = self._row(active.get("id", -1))
        return bool(row is not None and is_two_prize(row))
```

Note: `OT_PLAY` is 7; it is referenced literally above because `deck_guard`
already defines `_OT_PLAY = 7` and duplicating the name here would invite the
two to drift. Add `OT_PLAY = 7` to the enum block and use it instead.

- [ ] **Step 4: Run to verify pass**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Mutation check**

Remove the `is_two_prize(row)` condition; confirm `test_untouched_when_the_target_is_worth_one_prize` goes red. Restore.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_il/play_guard.py python/tests/test_play_guard.py
git commit -m "feat(play_guard): R1c mask gusts against a KO-able two-prize active"
```

---

## Task 5: Derive `low_hp_frac` from the corpus

**Files:**
- Create: `scripts/derive_low_hp_frac.py`
- Modify: `python/ptcg_il/play_guard.py` (set the measured default)

**Interfaces:**
- Consumes: `python/raw/**/*.json` (96,068 episodes across 42 day-directories).
- Produces: a printed report and the measured `TacticsConfig.low_hp_frac`.

- [ ] **Step 1: Write the derivation script**

```python
#!/usr/bin/env python3
"""Derive TacticsConfig.low_hp_frac (spec section 4.1).

Definition: the HP fraction at which P(my active is knocked out on the
opponent's next turn | hp/maxHp) crosses 0.5.  Below it, "don't invest energy
here" (R3) and "retreating is fine here" (R2) are both correct, because the
Pokemon is empirically about to die.

Usage:
    uv run python scripts/derive_low_hp_frac.py --raw-dir python/raw \\
        --bins 20 --jobs 16
"""
import argparse, json, pathlib, collections
from multiprocessing import Pool

import numpy as np


def _episode_transitions(path):
    """(hp_frac, was_ko_next_opponent_turn) per active-Pokemon turn boundary."""
    try:
        with open(path, "rb") as fh:
            ep = json.loads(fh.read())
    except Exception:
        return []
    steps = ep.get("steps") or []
    if not steps or not steps[0]:
        return []
    vis = (steps[0][0] or {}).get("visualize") or []
    out = []
    # Track, per player, the active's serial+hp at the end of each of their
    # turns, then check whether that serial is gone by their next turn.
    last = {}
    for e in vis:
        cur = e.get("current") or {}
        players = cur.get("players") or []
        turn = cur.get("turn")
        for pi, p in enumerate(players):
            active = (p or {}).get("active") or []
            if not active:
                continue
            a = active[0]
            maxhp = float(a.get("maxHp") or 0)
            if maxhp <= 0:
                continue
            frac = float(a.get("hp", 0)) / maxhp
            prev = last.get(pi)
            if prev is not None and prev[0] != turn:
                # prev turn's active: is that exact card still the active?
                survived = (a.get("serial") == prev[1])
                out.append((prev[2], 0 if survived else 1))
            last[pi] = (turn, a.get("serial"), frac)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="python/raw")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap episodes (0 = all); use for a smoke run")
    args = ap.parse_args()

    paths = sorted(pathlib.Path(args.raw_dir).rglob("*.json"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no episodes under {args.raw_dir}")

    tot = collections.Counter()
    kos = collections.Counter()
    n_obs = 0
    with Pool(args.jobs) as pool:
        for rows in pool.imap_unordered(_episode_transitions,
                                        [str(p) for p in paths], chunksize=32):
            for frac, ko in rows:
                b = min(int(frac * args.bins), args.bins - 1)
                tot[b] += 1
                kos[b] += ko
                n_obs += 1

    assert n_obs > 0, "examined zero turn transitions"
    print(f"episodes={len(paths)}  transitions={n_obs}")
    print(f"{'hp_frac':>12} {'n':>9} {'P(KO next turn)':>16}")
    crossing = None
    for b in range(args.bins):
        lo, hi = b / args.bins, (b + 1) / args.bins
        n = tot[b]
        if not n:
            continue
        p = kos[b] / n
        print(f"{lo:.2f}-{hi:.2f} {n:>9} {p:>16.4f}")
        if crossing is None and p < 0.5:
            crossing = hi
    print(f"\nlow_hp_frac (P(KO) drops below 0.5 at): {crossing}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run it on a slice**

Run: `uv run python scripts/derive_low_hp_frac.py --limit 500 --jobs 8`
Expected: a table with a nonzero `transitions` count and a printed crossing point. If `transitions=0` the assertion fires — fix the parser before scaling up.

- [ ] **Step 3: Run the full scan and record the number**

Run: `uv run python scripts/derive_low_hp_frac.py --jobs 16 2>&1 | tee /tmp/low_hp_frac.txt`
Expected: ~96k episodes, on the order of 10^6 transitions.

- [ ] **Step 4: Write the measured value into the config**

Replace the sentinel in `TacticsConfig`, documenting it the way `GuardConfig` documents `deck_low`:

```python
    #: Measured, not chosen.  P(my active is KO'd on the opponent's next turn)
    #: crosses 0.5 at this HP fraction, over <N> turn transitions from the
    #: <M>-episode corpus (scripts/derive_low_hp_frac.py).  Below it the
    #: Pokemon is empirically about to die, which is what makes both consumers
    #: correct: don't invest energy here (R3), retreating is fine here (R2).
    low_hp_frac: float = <MEASURED>
```

Substitute `<N>`, `<M>` and `<MEASURED>` from Step 3's output. Do not round to a "nicer" number.

- [ ] **Step 5: Add a regression test pinning the constant**

```python
# append to python/tests/test_play_guard.py

def test_low_hp_frac_is_a_measured_value_not_a_sentinel():
    from ptcg_il.play_guard import TacticsConfig
    v = TacticsConfig().low_hp_frac
    assert 0.0 < v < 1.0, f"low_hp_frac still a sentinel: {v}"
```

- [ ] **Step 6: Commit**

```bash
git add scripts/derive_low_hp_frac.py python/ptcg_il/play_guard.py \
        python/tests/test_play_guard.py
git commit -m "feat(play_guard): derive low_hp_frac from corpus turn transitions"
```

---

## Task 6: R2 — do not retreat a loaded attacker

**Files:**
- Modify: `python/ptcg_il/play_guard.py`
- Test: `python/tests/test_play_guard.py`

**Interfaces:**
- Consumes: `attached_energy_types`, `TacticsConfig.low_hp_frac`, `retreat_margin`.
- Produces: `PlayGuard.rerank(logits, batch, obs_dict, *, sel_type, sel_ctx, max_count) -> np.ndarray` (adjusted logits, float64[O]).

- [ ] **Step 1: Write the failing tests**

```python
# append to python/tests/test_play_guard.py

class TestR2Retreat:
    """ep 93323813 t7 and t21: options were `Attack153 | Retreat | End`, the
    agent chose Retreat, and the retreat cost discarded the only energy on the
    attacker.  At t21 that also put a 30 HP Budew in front of a 200-damage
    attacker."""

    @staticmethod
    def _guard(margin=0.5, low=0.3):
        from ptcg_il.play_guard import PlayGuard, TacticsConfig
        rows = {121: np.zeros(223, dtype=np.float32),
                2: np.zeros(223, dtype=np.float32)}
        rows[2][12 + 2] = 1.0          # Basic {R} Energy -> FIRE
        return PlayGuard(TacticsConfig(low_hp_frac=low, retreat_margin=margin,
                                       attach_margin=margin),
                         engine_card_features=rows)

    def _state(self, hp, maxhp=320, energy=(2,), with_attack=True):
        otypes = ([13] if with_attack else []) + [12, 14]
        opts = [{"type": t} for t in otypes]
        me = {"active": [_poke(cid=121, hp=hp, maxhp=maxhp, energy_ids=energy)],
              "bench": []}
        obs = _obs(my=me, select={"option": opts, "type": 0, "context": 0,
                                  "minCount": 1, "maxCount": 1})
        return obs, _batch(len(otypes), otype=otypes), otypes

    def test_retreat_demoted_when_loaded_healthy_and_able_to_attack(self):
        obs, b, otypes = self._state(hp=300)
        logits = np.zeros(64); logits[otypes.index(12)] = 1.0   # model wants retreat
        logits[otypes.index(13)] = 0.9
        out = self._guard().rerank(logits, b, obs, sel_type=0, sel_ctx=0,
                                   max_count=1)
        assert int(np.argmax(np.where(b["opt_mask"][0].numpy(), out, -np.inf))) \
            == otypes.index(13)

    def test_untouched_when_the_active_is_about_to_die(self):
        obs, b, otypes = self._state(hp=30)      # 30/320 = 0.09 < low_hp_frac
        logits = np.zeros(64); logits[otypes.index(12)] = 1.0
        logits[otypes.index(13)] = 0.9
        out = self._guard().rerank(logits, b, obs, sel_type=0, sel_ctx=0,
                                   max_count=1)
        assert int(np.argmax(np.where(b["opt_mask"][0].numpy(), out, -np.inf))) \
            == otypes.index(12)

    def test_untouched_when_no_attack_is_legal(self):
        """The engine only offers ATTACK when the cost is paid, so the option
        list is the affordability answer -- no cost math needed here."""
        obs, b, otypes = self._state(hp=300, with_attack=False)
        logits = np.zeros(64); logits[otypes.index(12)] = 1.0
        out = self._guard().rerank(logits, b, obs, sel_type=0, sel_ctx=0,
                                   max_count=1)
        assert int(np.argmax(np.where(b["opt_mask"][0].numpy(), out, -np.inf))) \
            == otypes.index(12)

    def test_untouched_when_nothing_is_attached(self):
        obs, b, otypes = self._state(hp=300, energy=())
        logits = np.zeros(64); logits[otypes.index(12)] = 1.0
        logits[otypes.index(13)] = 0.9
        out = self._guard().rerank(logits, b, obs, sel_type=0, sel_ctx=0,
                                   max_count=1)
        assert int(np.argmax(np.where(b["opt_mask"][0].numpy(), out, -np.inf))) \
            == otypes.index(12)

    def test_confident_retreat_outside_the_margin_survives(self):
        obs, b, otypes = self._state(hp=300)
        logits = np.zeros(64); logits[otypes.index(12)] = 5.0
        logits[otypes.index(13)] = 0.1
        out = self._guard(margin=0.25).rerank(logits, b, obs, sel_type=0,
                                              sel_ctx=0, max_count=1)
        assert int(np.argmax(np.where(b["opt_mask"][0].numpy(), out, -np.inf))) \
            == otypes.index(12)

    def test_stats_record_gate_and_change(self):
        g = self._guard()
        obs, b, otypes = self._state(hp=300)
        logits = np.zeros(64); logits[otypes.index(12)] = 1.0
        logits[otypes.index(13)] = 0.9
        g.rerank(logits, b, obs, sel_type=0, sel_ctx=0, max_count=1)
        assert g.stats.r2_gated == 1 and g.stats.r2_changed == 1

    def test_rerank_is_inert_for_multi_select(self):
        obs, b, otypes = self._state(hp=300)
        logits = np.zeros(64); logits[otypes.index(12)] = 1.0
        out = self._guard().rerank(logits, b, obs, sel_type=0, sel_ctx=0,
                                   max_count=3)
        assert np.allclose(out, logits)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -k R2 -v`
Expected: FAIL — `AttributeError: 'PlayGuard' object has no attribute 'rerank'`.

- [ ] **Step 3: Implement**

```python
# add to play_guard.py

    def _require_thresholds(self) -> None:
        c = self.config
        for name in ("low_hp_frac", "retreat_margin", "attach_margin"):
            if getattr(c, name) < 0.0:
                raise ValueError(
                    f"TacticsConfig.{name} is still the -1.0 sentinel. It is a "
                    f"measured value; run the derivation script rather than "
                    f"guessing (spec section 4).")

    def _my_active(self, obs_dict):
        cur = obs_dict.get("current") or {}
        players = cur.get("players") or []
        me = int(cur.get("yourIndex", 0))
        if me >= len(players):
            return None
        active = (players[me] or {}).get("active") or []
        return active[0] if active else None

    @staticmethod
    def _hp_frac(poke) -> float:
        maxhp = float(poke.get("maxHp") or 0.0)
        if maxhp <= 0.0:
            return 1.0
        return float(poke.get("hp", 0)) / maxhp

    def rerank(self, logits, batch: dict, obs_dict: dict, *, sel_type: int,
               sel_ctx: int, max_count: int):
        """Soft rules.  Returns adjusted logits; the index is DeckGuard's.

        Multi-select is left alone: these rules reason about a single chosen
        action, and select_multi advances its own state per pick.
        """
        out = np.asarray(
            logits[0].detach().cpu().numpy() if hasattr(logits, "detach")
            else logits, dtype=np.float64).copy()
        if int(sel_type) != ST_MAIN or int(max_count) != 1:
            return out
        self._require_thresholds()
        out = self._rerank_retreat(out, batch, obs_dict)
        return out

    def _bump_within_margin(self, scores, batch, prefer, demote, margin,
                            ) -> bool:
        """Lift the best `prefer` column above `demote` when both sit inside
        `margin` of the masked top.  Returns whether the top actually moved.
        """
        legal = self._np(batch, "opt_mask").astype(bool)
        masked = np.where(legal, scores, -np.inf)
        top = int(np.argmax(masked))
        if not demote[top]:
            return False
        cand = np.flatnonzero(legal & prefer
                              & (masked >= masked[top] - margin))
        if cand.size == 0:
            return False
        best = int(cand[np.argmax(masked[cand])])
        scores[best] = masked[top] + 1e-6
        return True

    def _rerank_retreat(self, scores, batch, obs_dict):
        """R2 -- retreating discards the attacker's energy for nothing when it
        could attack right now and is not about to be knocked out."""
        if not self.config.enable_retreat:
            return scores
        otype = self._np(batch, "opt_type")
        legal = self._np(batch, "opt_mask").astype(bool)
        if not (legal & (otype == OT_RETREAT)).any():
            return scores
        if not (legal & (otype == OT_ATTACK)).any():
            return scores
        active = self._my_active(obs_dict)
        if active is None:
            return scores
        if not attached_energy_types(active, self.engine_card_features):
            return scores
        if self._hp_frac(active) <= self.config.low_hp_frac:
            return scores

        self.stats.r2_gated += 1
        changed = self._bump_within_margin(
            scores, batch,
            prefer=(otype != OT_RETREAT) & (otype != OT_END),
            demote=(otype == OT_RETREAT),
            margin=self.config.retreat_margin)
        if changed:
            self.stats.r2_changed += 1
            self.stats.recent.append({"rule": "R2",
                                      "hp_frac": self._hp_frac(active)})
        return scores
```

- [ ] **Step 4: Run to verify pass**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Mutation check**

Remove the `_hp_frac(active) <= low_hp_frac` early return; confirm `test_untouched_when_the_active_is_about_to_die` goes red. Restore. Then remove the `max_count != 1` early return; confirm `test_rerank_is_inert_for_multi_select` goes red. Restore.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_il/play_guard.py python/tests/test_play_guard.py
git commit -m "feat(play_guard): R2 demote retreat when the attacker is loaded and healthy"
```

---

## Task 7: R3 — attach energy toward an unmet attack cost

**Files:**
- Modify: `python/ptcg_il/play_guard.py`
- Test: `python/tests/test_play_guard.py`

**Interfaces:**
- Consumes: `unmet_cost`, `attack_cost_hist`, `attack_block_count`, `attached_energy_types`, `option_pokemon`, `energy_type_of_card`.
- Produces: `PlayGuard._rerank_attach(scores, batch, obs_dict)`, called from `rerank`; `PlayGuard.attach_value(obs_dict, option) -> tuple[int, bool]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to python/tests/test_play_guard.py

class TestR3Attach:
    """ep 93323813: Phantom Dive (cost Fire+Psychic) was offered ZERO times in
    24 turns while Jet Headbutt (one Colorless, 70 base / 40 after resistance)
    was offered 16.  Energy in play never exceeded 2."""

    @staticmethod
    def _rows():
        rows = {}
        # Dragapult ex 121: attack 0 = Jet Headbutt [0], attack 1 = Phantom
        # Dive [2, 5].  Two attacks -> col 82 = 2/3.
        r = np.zeros(223, dtype=np.float32)
        r[82] = 2.0 / 3.0
        off0 = 85 + 0 * 46
        r[off0 + 1 + 0] = 1.0 / 5.0            # 1 Colorless
        off1 = 85 + 1 * 46
        r[off1 + 1 + 2] = 1.0 / 5.0            # 1 Fire
        r[off1 + 1 + 5] = 1.0 / 5.0            # 1 Psychic
        rows[121] = r
        # Munkidori 112: one attack costing 1 Darkness
        m = np.zeros(223, dtype=np.float32)
        m[82] = 1.0 / 3.0
        m[85 + 1 + 7] = 1.0 / 5.0
        rows[112] = m
        for cid, etype in ((2, 2), (5, 5), (7, 7)):
            e = np.zeros(223, dtype=np.float32)
            e[12 + etype] = 1.0
            rows[cid] = e
        return rows

    def _guard(self, margin=0.5, low=0.3):
        from ptcg_il.play_guard import PlayGuard, TacticsConfig
        return PlayGuard(TacticsConfig(low_hp_frac=low, retreat_margin=margin,
                                       attach_margin=margin),
                         engine_card_features=self._rows())

    def _state(self, hand_ids, active_energy=(), bench=(), active_hp=320):
        opts, me_hand = [], []
        for i, cid in enumerate(hand_ids):
            me_hand.append({"id": cid})
            opts.append({"type": 8, "area": 2, "index": i,
                         "inPlayArea": 4, "inPlayIndex": 0})
        for i, cid in enumerate(hand_ids):
            opts.append({"type": 8, "area": 2, "index": i,
                         "inPlayArea": 5, "inPlayIndex": 0})
        opts.append({"type": 14})
        me = {"hand": me_hand,
              "active": [_poke(cid=121, hp=active_hp, maxhp=320,
                               energy_ids=active_energy)],
              "bench": [_poke(cid=c, hp=110, maxhp=110) for c in bench]}
        obs = _obs(my=me, select={"option": opts, "type": 0, "context": 0,
                                  "minCount": 1, "maxCount": 1})
        otypes = [o["type"] for o in opts]
        return obs, _batch(len(opts), otype=otypes), opts

    def test_attach_that_reduces_unmet_cost_wins_a_near_tie(self):
        obs, b, opts = self._state([2], active_energy=(5,), bench=[112])
        logits = np.zeros(64)
        logits[1] = 1.0        # model prefers attaching to the bench
        logits[0] = 0.9        # active needs Fire to complete Phantom Dive
        out = self._guard().rerank(logits, b, obs, sel_type=0, sel_ctx=0,
                                   max_count=1)
        assert int(np.argmax(np.where(b["opt_mask"][0].numpy(), out,
                                      -np.inf))) == 0

    def test_attach_to_an_already_paid_attack_is_not_promoted(self):
        obs, b, opts = self._state([2], active_energy=(2, 5), bench=[112])
        logits = np.zeros(64)
        logits[1] = 1.0
        logits[0] = 0.9
        out = self._guard().rerank(logits, b, obs, sel_type=0, sel_ctx=0,
                                   max_count=1)
        assert int(np.argmax(np.where(b["opt_mask"][0].numpy(), out,
                                      -np.inf))) == 1

    def test_low_hp_target_is_excluded(self):
        obs, b, opts = self._state([2], active_energy=(5,), bench=[112],
                                   active_hp=30)
        logits = np.zeros(64)
        logits[1] = 1.0
        logits[0] = 0.9
        out = self._guard().rerank(logits, b, obs, sel_type=0, sel_ctx=0,
                                   max_count=1)
        assert int(np.argmax(np.where(b["opt_mask"][0].numpy(), out,
                                      -np.inf))) == 1

    def test_wrong_colour_does_not_promote(self):
        obs, b, opts = self._state([7], active_energy=(5,), bench=[112])
        logits = np.zeros(64)
        logits[1] = 1.0
        logits[0] = 0.9
        out = self._guard().rerank(logits, b, obs, sel_type=0, sel_ctx=0,
                                   max_count=1)
        assert int(np.argmax(np.where(b["opt_mask"][0].numpy(), out,
                                      -np.inf))) == 1

    def test_attach_value_reports_the_reduction(self):
        g = self._guard()
        obs, b, opts = self._state([2], active_energy=(5,), bench=[112])
        gain, healthy = g.attach_value(obs, opts[0])
        assert gain == 1 and healthy is True
        gain_wrong, _ = g.attach_value(obs, opts[1])
        assert gain_wrong <= 0 or gain_wrong == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -k R3 -v`
Expected: FAIL — `AttributeError: 'PlayGuard' object has no attribute 'attach_value'`.

- [ ] **Step 3: Implement**

```python
# add to play_guard.py

    def _cheapest_unmet(self, poke, attached) -> int:
        row = self._row(poke.get("id", -1))
        if row is None:
            return 0
        n = attack_block_count(row)
        if n <= 0:
            return 0
        return min(unmet_cost(attack_cost_hist(row, i), attached)
                   for i in range(n))

    def attach_value(self, obs_dict: dict, option: dict):
        """(unmet-cost reduction, target-is-healthy) for one ATTACH option."""
        cur = obs_dict.get("current") or {}
        players = cur.get("players") or []
        me = int(cur.get("yourIndex", 0))
        hand = ((players[me] or {}).get("hand") or []) if me < len(players) else []
        idx = option.get("index")
        if idx is None or not (0 <= int(idx) < len(hand)):
            return 0, False
        card_row = self._row(hand[int(idx)].get("id", -1))
        if card_row is None:
            return 0, False
        supplied = energy_type_of_card(card_row)

        target = option_pokemon(obs_dict, option)
        if target is None:
            return 0, False
        attached = attached_energy_types(target, self.engine_card_features)
        before = self._cheapest_unmet(target, attached)
        after = self._cheapest_unmet(target, attached + [supplied])
        return int(before - after), bool(
            self._hp_frac(target) > self.config.low_hp_frac)

    def _rerank_attach(self, scores, batch, obs_dict):
        """R3 -- prefer attaching where it actually completes an attack cost."""
        if not self.config.enable_attach:
            return scores
        otype = self._np(batch, "opt_type")
        legal = self._np(batch, "opt_mask").astype(bool)
        if not (legal & (otype == OT_ATTACH)).any():
            return scores
        options = (obs_dict.get("select") or {}).get("option") or []
        n = n_option_columns(batch, obs_dict)
        prefer = np.zeros(len(otype), dtype=bool)
        gains = np.zeros(len(otype), dtype=np.int64)
        for j in range(n):
            if int(otype[j]) != OT_ATTACH or not legal[j]:
                continue
            gain, healthy = self.attach_value(obs_dict, options[j])
            if gain > 0 and healthy:
                prefer[j] = True
                gains[j] = gain
        if not prefer.any():
            return scores

        self.stats.r3_gated += 1
        masked = np.where(legal, scores, -np.inf)
        top = int(np.argmax(masked))
        if prefer[top]:
            return scores
        cand = np.flatnonzero(
            prefer & (masked >= masked[top] - self.config.attach_margin))
        if cand.size == 0:
            return scores
        best = int(cand[np.lexsort((masked[cand], gains[cand]))[-1]])
        scores[best] = masked[top] + 1e-6
        self.stats.r3_changed += 1
        self.stats.recent.append({"rule": "R3", "gain": int(gains[best])})
        return scores

# and extend rerank():
        out = self._rerank_retreat(out, batch, obs_dict)
        out = self._rerank_attach(out, batch, obs_dict)
        return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Mutation check**

Change `after = self._cheapest_unmet(target, attached + [supplied])` to reuse `attached`; confirm `test_attach_that_reduces_unmet_cost_wins_a_near_tie` goes red. Restore. Then drop the `and healthy` condition; confirm `test_low_hp_target_is_excluded` goes red. Restore.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_il/play_guard.py python/tests/test_play_guard.py
git commit -m "feat(play_guard): R3 promote attachments that complete an attack cost"
```

---

## Task 8: Derive the R2/R3 margins against the deployed ensemble

**Files:**
- Create: `scripts/derive_play_guard_margin.py`
- Modify: `python/ptcg_il/play_guard.py` (set the measured defaults)

**Interfaces:**
- Consumes: `data/shards/val-repack-a16-*.npz`, the `ens-3-16` members from `data/il_baselines.json`, `PlayGuard`'s gate predicates from Tasks 6–7.
- Produces: measured `retreat_margin` and `attach_margin`.

- [ ] **Step 1: Read which checkpoints `ens-3-16` selected**

Run:

```bash
uv run --no-project python -c "
import json; b=json.load(open('data/il_baselines.json'))
r=b.get('ens-3-16', {})
print(json.dumps({k: r.get(k) for k in ('selection','members','ckpt_sha1','split')}, indent=2))"
```

If `selection`/`members` is absent or names checkpoints not on disk, **stop and ask the owner** (spec §8 open item 3) rather than substituting seeds.

- [ ] **Step 2: Write the derivation script**

```python
#!/usr/bin/env python3
"""Derive TacticsConfig.retreat_margin / attach_margin (spec section 4.2).

p25 of the masked top-2 logit gap over exactly the decisions R2 and R3 gate.

Fitted against the DEPLOYED 3-member ensemble, not a single seed:
EnsemblePolicy combines members as log(mean(softmax(masked logits))), which
compresses the logit scale, so a margin fitted on one member is the wrong width
for what actually ships.

Usage:
    cd python && uv run python ../scripts/derive_play_guard_margin.py \\
        --shards data/shards --split val-repack-a16 \\
        --ckpt checkpoints_a16_s7/ckpt-best.pt \\
        --ckpt checkpoints_a16_s8/ckpt-best.pt \\
        --ckpt checkpoints_a16_s9/ckpt-best.pt
"""
import argparse, glob, json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--split", default="val-repack-a16")
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    from ptcg_il.ensemble import EnsemblePolicy
    from ptcg_il.featurizer import load_engine_tables
    from ptcg_il.play_guard import PlayGuard, TacticsConfig, OT_RETREAT, \
        OT_ATTACK, OT_ATTACH

    tables = load_engine_tables(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_card = torch.zeros(max(tables["engine_card_features"]) + 1, 223)
    for cid, feat in tables["engine_card_features"].items():
        all_card[int(cid)] = torch.from_numpy(np.asarray(feat, dtype=np.float32))
    all_atk = torch.zeros(max(tables["engine_attack_features"]) + 1, 46)
    for aid, feat in tables["engine_attack_features"].items():
        all_atk[int(aid)] = torch.from_numpy(np.asarray(feat, dtype=np.float32))
    model = EnsemblePolicy.from_checkpoints(args.ckpt, all_card, all_atk,
                                            device=device)

    r2_gaps, r3_gaps = [], []
    paths = sorted(glob.glob(f"{args.shards}/{args.split}-*.npz"))
    if not paths:
        raise SystemExit(f"no shards matching {args.split}")
    for path in paths:
        z = np.load(path)
        n = len(z["sel_type"])
        for i in range(n):
            if int(z["sel_type"][i]) != 0 or int(z["maxCount"][i]) != 1:
                continue
            batch = {k: torch.from_numpy(np.asarray(z[k][i])).unsqueeze(0).to(device)
                     for k in z.files if k not in ("action_idx", "action_len")}
            with torch.no_grad():
                logits, _v, _h = model(batch)
            mask = z["opt_mask"][i].astype(bool)
            otype = z["opt_type"][i]
            lg = np.where(mask, logits[0].float().cpu().numpy(), -np.inf)
            order = np.sort(lg[np.isfinite(lg)])[::-1]
            if order.size < 2:
                continue
            gap = float(order[0] - order[1])
            if (mask & (otype == OT_RETREAT)).any() and \
               (mask & (otype == OT_ATTACK)).any():
                r2_gaps.append(gap)
            if (mask & (otype == OT_ATTACH)).any():
                r3_gaps.append(gap)

    assert r2_gaps, "examined zero R2-gated decisions"
    assert r3_gaps, "examined zero R3-gated decisions"
    for name, gaps in (("retreat_margin", r2_gaps), ("attach_margin", r3_gaps)):
        a = np.asarray(gaps)
        print(f"{name}: n={a.size} p25={np.percentile(a,25):.4f} "
              f"p50={np.percentile(a,50):.4f} p75={np.percentile(a,75):.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run the command in the script docstring, substituting the members from Step 1.
Expected: two lines with nonzero `n` and a p25 for each. If the two p25s are within ~20% of each other, use one shared value for both and say so in the config comment; otherwise keep them separate (spec §8 open item 4).

- [ ] **Step 4: Write the measured values into the config**

```python
    #: Measured, not chosen.  p25 of the masked top-2 logit gap over the
    #: <N> R2-gated decisions in val-repack-a16, scored with the deployed
    #: ens-3-16 ensemble (scripts/derive_play_guard_margin.py).  Fitted on the
    #: ensemble deliberately: log(mean(softmax(.))) compresses the logit scale,
    #: so a single-member fit would be the wrong width for what ships.
    retreat_margin: float = <MEASURED>
    #: As above, over the <M> R3-gated decisions.
    attach_margin: float = <MEASURED>
```

- [ ] **Step 5: Extend the sentinel test**

```python
# replace test_low_hp_frac_is_a_measured_value_not_a_sentinel with:

def test_thresholds_are_measured_values_not_sentinels():
    from ptcg_il.play_guard import TacticsConfig
    c = TacticsConfig()
    for name in ("low_hp_frac", "retreat_margin", "attach_margin"):
        v = getattr(c, name)
        assert v > 0.0, f"{name} still a sentinel: {v}"
    assert 0.0 < c.low_hp_frac < 1.0
```

- [ ] **Step 6: Commit**

```bash
git add scripts/derive_play_guard_margin.py python/ptcg_il/play_guard.py \
        python/tests/test_play_guard.py
git commit -m "feat(play_guard): derive R2/R3 margins against the deployed ens-3-16"
```

---

## Task 9: Wire into live_eval, the greedy bundle, and root main.py

**Files:**
- Modify: `python/ptcg_il/live_eval.py:131-214` and `:234-260`
- Modify: `scripts/build_submission.py` (`build_model_package`, `MAIN_PY_TEMPLATE_GREEDY`)
- Modify: `main.py` (repo root)
- Test: `python/tests/test_live_eval.py`, `tests/test_build_submission.py`

**Interfaces:**
- Consumes: `PlayGuard`, `TacticsConfig` from Tasks 2–8.
- Produces: `PolicyAgent(..., use_play_guard: bool = True)` and `PolicyAgent._ensure_play_guard()`.

- [ ] **Step 1: Write the failing wiring tests**

```python
# append to python/tests/test_live_eval.py

class TestPolicyAgentPlayGuard:
    def test_play_guard_is_on_by_default(self):
        from ptcg_il.live_eval import PolicyAgent
        import inspect
        sig = inspect.signature(PolicyAgent.__init__)
        assert sig.parameters["use_play_guard"].default is True

    def test_play_guard_is_dropped_from_the_pickle(self):
        """It holds the 1.1 MB card table; it must not ride into every worker,
        exactly like _guard."""
        from ptcg_il.live_eval import PolicyAgent
        src = inspect.getsource(PolicyAgent.__getstate__)
        assert "_play_guard" in src
```

```python
# append to tests/test_build_submission.py

def test_greedy_template_wires_the_play_guard(bs):
    imported = [l for l in bs.MAIN_PY_TEMPLATE_GREEDY.splitlines()
                if l.startswith("from model")]
    assert any("play_guard" in m for m in imported), imported


def test_mcts_template_does_not_wire_the_play_guard(bs):
    """MCTS is obsolete; the greedy path is the only wired one."""
    imported = [l for l in bs.MAIN_PY_TEMPLATE.splitlines()
                if l.startswith("from model")]
    assert not any("play_guard" in m for m in imported), imported


def test_play_guard_shipped_in_both_packages(bs, tmp_path):
    """Vendors verbatim like deck_guard.py / ref_map.py."""
    src = tmp_path / "src"
    (src / "python" / "ptcg_il" / "model").mkdir(parents=True)
    (src / "python" / "ptcg_il" / "play_guard.py").write_text("# MARKER\n")
    for build in ("greedy", "mcts"):
        bs.build_model_package(src, tmp_path / build, mcts=(build == "mcts"))
        out = tmp_path / build / "model" / "play_guard.py"
        assert out.exists(), f"play_guard.py missing from {build} build"
        assert "MARKER" in out.read_text()
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_live_eval.py -k PlayGuard -v
cd .. && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_build_submission.py -k play_guard -v
```
Expected: FAIL on all five.

- [ ] **Step 3: Wire `live_eval.py`**

In `__init__` (after line 156):

```python
        # Tactical guard, matching the submission bundle's greedy agent.
        self.use_play_guard = use_play_guard
        self._play_guard = None
```

Add `use_play_guard: bool = True` to the signature after `use_deck_guard`.

In `__getstate__` (after line 199):

```python
        state["_play_guard"] = None  # holds the card table; rebuilt on demand
```

Add after `_ensure_guard`:

```python
    def _ensure_play_guard(self):
        """The shared PlayGuard, built with this agent's engine card table."""
        if self._play_guard is None:
            from ptcg_il.play_guard import PlayGuard, TacticsConfig

            self._play_guard = PlayGuard(
                TacticsConfig(), self._tables()["engine_card_features"])
        return self._play_guard
```

In `__call__`, after the existing `guard.apply_mask(...)` at `:245-246`:

```python
        play = self._ensure_play_guard() if self.use_play_guard else None
        if play is not None:
            play.apply_mask(batch, obs_dict,
                            sel_type=int(sample["sel_type"]),
                            sel_ctx=int(sample["sel_ctx"]))
```

Inside the `max_count == 1` branch, between the forward pass and `guard.pick`:

```python
                if play is not None:
                    logits = torch.as_tensor(
                        play.rerank(logits, batch, obs_dict,
                                    sel_type=int(sample["sel_type"]),
                                    sel_ctx=int(sample["sel_ctx"]),
                                    max_count=max_count),
                        dtype=logits.dtype, device=logits.device).unsqueeze(0)
```

- [ ] **Step 4: Wire the bundle**

In `build_submission.build_model_package`, directly after the `deck_guard.py` block at `:832-840`, add the identical block for `play_guard.py` (copy verbatim, ships in both builds).

In `MAIN_PY_TEMPLATE_GREEDY`, after `from model.deck_guard import DeckGuard, GuardConfig`:

```python
from model.play_guard import PlayGuard, TacticsConfig
```

After `_guard = DeckGuard(GuardConfig(), _engine_card_features)`:

```python
# Tactical guard: R1a/R1c/R4 are hard masks (a lethal attack forfeited by END
# or RETREAT, a gust that moves a KO-able 2-prize target away, a damage counter
# on a corpse); R2/R3 are gated near-tie re-ranks. Thresholds are measured —
# see TacticsConfig.
_play = PlayGuard(TacticsConfig(), _engine_card_features)
```

In the template's `agent()`, after `_guard.apply_mask(...)`:

```python
    _play.apply_mask(batch, obs_dict, sel_type=_sel_type,
                     sel_ctx=int(feats.get("sel_ctx", 0)))
```

and inside `if feat_max_count == 1:` between the forward pass and `_guard.pick`:

```python
            logits = torch.as_tensor(
                _play.rerank(logits, batch, obs_dict, sel_type=_sel_type,
                             sel_ctx=int(feats.get("sel_ctx", 0)),
                             max_count=feat_max_count),
                dtype=logits.dtype, device=logits.device).unsqueeze(0)
```

- [ ] **Step 5: Mirror the same four edits into repo-root `main.py`**

Apply the identical import, `_play = PlayGuard(...)`, `apply_mask` and `rerank`
edits at `main.py:31`, `:192`, and inside `agent()`. The root file is a copy of
the greedy template; keeping them identical is what the build smoke test relies
on.

- [ ] **Step 6: Run the full suites and the bundle smoke test**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q
cd .. && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q
uv run python scripts/build_submission.py --data-dir data \
    --ckpt checkpoints_a16_s7/ckpt-best.pt --no-mcts \
    --out /tmp/submission-greedy-smoke.tar.gz
```
Expected: both suites pass; the build prints `Import verification: PASS`.

- [ ] **Step 7: Commit**

```bash
git add python/ptcg_il/live_eval.py scripts/build_submission.py main.py \
        python/tests/test_live_eval.py tests/test_build_submission.py
git commit -m "feat(play_guard): wire into live_eval, greedy bundle and root main.py"
```

---

## Task 10: Replay regression against the two real losses

**Files:**
- Create: `python/tests/test_play_guard_replays.py`
- Create: `python/tests/fixtures/replay_93320914_decisions.json`

**Interfaces:**
- Consumes: `PlayGuard.apply_mask`.
- Produces: nothing downstream — this is the proof the rules fix the observed failures rather than merely firing.

- [ ] **Step 1: Extract the two decisions into a committed fixture**

The raw episodes are 5.4 MB each and live in `~/Downloads`, so extract only the
two entries the test needs. `selected` at visualize entry *i* answers the
`select` at entry *i−1* — verified against three attack logs — so the fixture
records the `select`/`current` from `[157]` and `[139]`, plus the answer taken
from `[158]` and `[140]`.

```bash
uv run --no-project python - <<'PY'
import json, pathlib
src = json.load(open('/home/charles/Downloads/93320914.json'))
V = src['steps'][0][0]['visualize']
out = {}
for name, i in (("gust_157", 157), ("dead_target_139", 139)):
    e = V[i]
    out[name] = {"current": e["current"], "select": e["select"],
                 "chosen": V[i + 1].get("selected")}
p = pathlib.Path('python/tests/fixtures/replay_93320914_decisions.json')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out))
print(p, p.stat().st_size, "bytes")
PY
```

- [ ] **Step 2: Write the failing regression test**

```python
# python/tests/test_play_guard_replays.py
"""Regression against the two live losses that motivated play_guard.

These are the only tests that show the rules change the decisions that actually
lost games, rather than merely firing somewhere.
"""
import json
import pathlib

import numpy as np
import pytest

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / \
    "replay_93320914_decisions.json"


@pytest.fixture(scope="module")
def decisions():
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}")
    return json.loads(FIXTURE.read_text())


def _featurize_batch(entry, tables, vocab):
    import torch
    from ptcg_il.featurizer import featurize
    obs = {"current": entry["current"], "select": entry["select"],
           "logs": [], "step": 0}
    feats = featurize(obs, vocab, value_target=0.0, sample_weight=1.0, **tables)
    batch = {}
    for k, v in feats.items():
        if not isinstance(v, (np.ndarray, np.generic)):
            continue
        arr = np.asarray(v)
        t = torch.from_numpy(np.ascontiguousarray(arr) if arr.ndim else arr)
        batch[k] = t.unsqueeze(0)
    return obs, batch, feats


def test_r1c_refuses_the_gust_that_lost_episode_93320914(decisions, il_tables,
                                                         il_vocab):
    """[157]: Dragapult ex 320/320 with Phantom Dive legal, Archaludon ex at
    130/300 active. The agent played Boss's Orders and gusted the KO-able
    2-prize target away, taking 1 prize instead of 2 and leaving the only
    Pokemon scoring against it alive."""
    from ptcg_il.play_guard import PlayGuard, TacticsConfig

    entry = decisions["gust_157"]
    obs, batch, feats = _featurize_batch(entry, il_tables, il_vocab)
    chosen = entry["chosen"][0]
    assert bool(batch["opt_mask"][0][chosen]), "the losing option was not legal"

    g = PlayGuard(TacticsConfig(), il_tables["engine_card_features"])
    g.apply_mask(batch, obs, sel_type=int(feats["sel_type"]),
                 sel_ctx=int(feats["sel_ctx"]))

    assert not bool(batch["opt_mask"][0][chosen]), \
        "R1c did not mask the Boss's Orders that lost the game"
    otype = batch["opt_type"][0].numpy()
    mask = batch["opt_mask"][0].numpy()
    assert (mask & (otype == 13)).any(), "the lethal attack must remain legal"
    assert g.stats.r1c_fired == 1


def test_r4_refuses_the_zero_hp_relicanth(decisions, il_tables, il_vocab):
    """[139]-[144]: Relicanth reached 0 HP, then took five more counters while
    two 130 HP Duraludon were legal targets."""
    from ptcg_il.play_guard import PlayGuard, TacticsConfig

    entry = decisions["dead_target_139"]
    obs, batch, feats = _featurize_batch(entry, il_tables, il_vocab)
    chosen = entry["chosen"][0]

    g = PlayGuard(TacticsConfig(), il_tables["engine_card_features"])
    g.apply_mask(batch, obs, sel_type=int(feats["sel_type"]),
                 sel_ctx=int(feats["sel_ctx"]))

    assert not bool(batch["opt_mask"][0][chosen]), \
        "R4 did not mask the already-KO'd Relicanth"
    assert batch["opt_mask"][0].any(), "floor cleared the whole mask"
    assert g.stats.r4_masked >= 1
```

Add the two fixtures to `python/tests/conftest.py` if not already present:

```python
@pytest.fixture(scope="session")
def il_tables():
    from ptcg_il.featurizer import load_engine_tables
    return load_engine_tables("data")


@pytest.fixture(scope="session")
def il_vocab():
    import json
    from ptcg_il.featurizer import normalize_vocab
    with open("data/vocab.json") as fh:
        return normalize_vocab(json.load(fh))
```

- [ ] **Step 3: Run to verify it fails without the guard**

Temporarily construct with `TacticsConfig(enable_lethal_gust=False, enable_dead_target=False)`, run, confirm both tests fail. Restore.

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard_replays.py -v`

- [ ] **Step 4: Run to verify it passes with the guard**

Run: `cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_play_guard_replays.py -v`
Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_play_guard_replays.py \
        python/tests/fixtures/replay_93320914_decisions.json \
        python/tests/conftest.py
git commit -m "test(play_guard): regression against the two decisions that lost 93320914"
```

---

## Task 11: Live A/B and the ship decision

**Files:**
- No source changes. Produces a results document.
- Create: `docs/superpowers/specs/2026-08-16-play-guard-results.md`

- [ ] **Step 1: Run the paired evaluation**

```bash
cd python
uv run python -m ptcg_il.cli live-eval --data-dir data \
    --ckpt checkpoints_a16_s7/ckpt-best.pt --games 200 --seed 0 \
    --use-play-guard 2>&1 | tee /tmp/ab_on.txt
uv run python -m ptcg_il.cli live-eval --data-dir data \
    --ckpt checkpoints_a16_s7/ckpt-best.pt --games 200 --seed 0 \
    --no-use-play-guard 2>&1 | tee /tmp/ab_off.txt
```

If `live-eval` has no such flag, add it as a passthrough to `PolicyAgent(use_play_guard=...)` in the same commit — the flag is what makes the A/B possible, so it belongs to this task.

- [ ] **Step 2: Record the three measured dimensions**

Win rate, illegal-action count, and OOV rate, for each arm, plus the
`TacticsStats` counters (how often each rule fired and how often R2/R3 changed
the pick). A rule that never fires is not evidence of safety — it is evidence
the gate is mis-specified, and should be reported as such.

- [ ] **Step 3: Write the results document**

Mirror `docs/superpowers/specs/2026-08-12-prior-determinizer-results.md`:
setup, per-opponent table, the stats counters, and an explicit ship/no-ship
recommendation.

- [ ] **Step 4: Ask the owner for the ship decision**

Spec §8 open item 2: at 200 games/opponent the CIs will very likely overlap, as
they did for the determinizer, which shipped on "better or equal on every
measured dimension" rather than significance. Confirm that bar applies here
before shipping — do not assume it.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-08-16-play-guard-results.md
git commit -m "docs(play_guard): live A/B results and ship recommendation"
```

---

## Self-Review

**Spec coverage:**

| spec section | task |
|---|---|
| §2.1 module, self-contained | 1 |
| §2.2 no new artifact, `rint`, 3-block cap | 1 |
| §2.3 energy types via `energyCards` | 1 (incl. structural test) |
| §2.4 interface | 2, 6 |
| §2.5 precedence, never-empty floor | 2, 9 |
| §3.1 R1a | 3 |
| §3.2 R1c + `GUST_CARDS` | 4 |
| §3.3 R2 | 6 |
| §3.4 R3 | 7 |
| §3.5 R4 | 2 |
| §4.1 `low_hp_frac` | 5 |
| §4.2 `margin` on the ensemble | 8 |
| §5 call sites | 9 |
| §6 not changing (MCTS untouched) | 9 (`test_mcts_template_does_not_wire_the_play_guard`) |
| §7 unit / contract / mutation / replay / A-B | 1–4, 6, 7, 10, 11 |
| §8 open items 2, 3, 4 | 11 step 4, 8 step 1, 8 step 3 |

No gaps.

**Placeholder scan:** the only intentional blanks are `<MEASURED>`, `<N>`, `<M>`
in Tasks 5 and 8, which are *outputs of the step immediately above them* — the
whole point of those tasks is that the numbers must be measured, not chosen, and
`_require_thresholds` plus `test_thresholds_are_measured_values_not_sentinels`
make an unfilled value fail loudly rather than silently ship a guess.

**Type consistency:** `apply_mask(batch, obs_dict, *, sel_type, sel_ctx)` and
`rerank(logits, batch, obs_dict, *, sel_type, sel_ctx, max_count)` are used
identically in Tasks 2, 3, 4, 6, 7, 9 and 10. `TacticsConfig` field names
(`enable_lethal_end`, `enable_lethal_gust`, `enable_retreat`, `enable_attach`,
`enable_dead_target`, `low_hp_frac`, `retreat_margin`, `attach_margin`) are
consistent across Tasks 2–9. `OT_PLAY = 7` is added to the enum block in Task 4
Step 3 and used in place of the literal.
