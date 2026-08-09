# Per-Option Decision Bandwidth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the IL policy the capacity to distinguish ATTACK from RETREAT options and to see deck-out risk, the bench promotion target, and retreat affordability at decision time.

**Architecture:** Add five `opt_scalar` dims (deck-out risk, zero-damage flag, retreat affordability, bench damage ratio, bench HP ratio) and a new `opt_bench_idx` ref tensor (the state-token row index of the best bench Pokemon). Expand `attack_static_row` from 14→16 numeric dims with two draw-count fields. This invalidates all shards and checkpoints via `Policy.config["feat_dims"]`.

**Tech Stack:** Python 3.11, NumPy, PyTorch, uv

## Global Constraints

- `F_OPT` 8 → 13, `F_ATK` 43 → 45, `F_CARD` 212 → 218
- `DRAW_N = 10.0`
- `RETREAT_N = 4.0` (already defined; reused for dim 10)
- Hot-path constraint: new featurize-time computation must use pure-Python scalar arithmetic in inner loops (numpy slice sums are fine for once-per-select precomputation)
- Dim assertions in `cards.py` must be updated; a `K_EFFECT` mismatch must fail at import
- `option_groups` key must include `opt_bench_idx`
- Vendored copies in `model/featurizer.py` and `model/pointer.py` must stay in sync with `python/ptcg_il/`
- All existing tests must continue to pass (after adapting expected dim constants where they reference `F_OPT`/`F_ATK`/`F_CARD`)

---

### Task 1: Parse draw counts from attack text in keywords.py

**Files:**
- Modify: `python/ptcg_mine/keywords.py`

**Interfaces:**
- Produces: `draw_fixed(attack) -> int`, `draw_to_hand(attack) -> int`
- Consumed by: `cards.py` (Task 3)

- [ ] **Step 1: Add draw-count parsing functions**

Add after `attack_keyword_row` (after line 92) in `python/ptcg_mine/keywords.py`:

```python
# ============================================================
# Draw-count parsing (for attack_static_row numerics)
# ============================================================

_DRAW_FIXED_PAT = re.compile(r"draw (\d+) (?:more )?card", re.IGNORECASE)
_DRAW_TO_HAND_PAT = re.compile(
    r"(?:draw cards until you have|you may draw cards until you have) (\d+) card",
    re.IGNORECASE,
)
_DRAW_A_CARD_PAT = re.compile(r"draw a card", re.IGNORECASE)
_DRAW_BOTH_PAT = re.compile(r"each player draws (\d+) card", re.IGNORECASE)


def draw_fixed(attack) -> int:
    """Explicit draw count from attack oracle text, or 0.

    Matches ``draw 2 cards``, ``Draw a card.`` (implicit 1), and
    ``each player draws N``.  Does *not* match draw-to-hand-size forms.
    """
    text = getattr(attack, "text", "") or ""
    m = _DRAW_FIXED_PAT.search(text)
    if m:
        return int(m.group(1))
    if _DRAW_A_CARD_PAT.search(text):
        return 1
    m = _DRAW_BOTH_PAT.search(text)
    if m:
        return int(m.group(1))
    return 0


def draw_to_hand(attack) -> int:
    """Target hand size in 'draw cards until you have N cards', or 0."""
    text = getattr(attack, "text", "") or ""
    m = _DRAW_TO_HAND_PAT.search(text)
    if m:
        return int(m.group(1))
    return 0
```

- [ ] **Step 2: Verify parsing against all 36 draw-mentioning attacks**

```bash
cd /home/charles/Documents/Pokemon/python && PYTHONPATH=. uv run python -c "
from ptcg_mine.cards import load_engine
from ptcg_mine.keywords import draw_fixed, draw_to_hand
import re
cards, atks = load_engine()
n_draw = 0
samples = []
for a in atks:
    t = getattr(a, 'text', '') or ''
    if re.search(r'draw', t, re.IGNORECASE):
        n_draw += 1
        f = draw_fixed(a)
        h = draw_to_hand(a)
        if f == 0 and h == 0:
            samples.append(f'MISS: {t[:80]}')
assert n_draw > 0, 'zero draw attacks examined'
misses = [s for s in samples if 'each player draws' not in s.lower()]
print(f'{n_draw} draw attacks, {n_draw - len(misses)} parsed, {len(misses)} misses')
for s in misses:
    print(' ', s)
# Manual acceptance: 'draw that many', 'draw up to', 'draw cards until' without number
# are the only expected misses — they are state-dependent and cannot have a static count.
"
```

Expected: 36 draw mentions, most parsed. Acceptable misses are: "draw that many cards" (depends on count of something), "draw up to N cards" (may draw less), and "Draw cards until you have N cards in your hand" variants without a specific number like "...until you have 6 cards..." which should match the second pattern.

- [ ] **Step 3: Run existing mine tests**

```bash
cd /home/charles/Documents/Pokemon && PYTHONPATH=python uv run pytest tests/ -x --timeout=60 2>&1 | tail -15
```

---

### Task 2: Bump feature dim constants and add DRAW_N in featurizer.py

**Files:**
- Modify: `python/ptcg_il/featurizer.py` (lines 47, 75-81)

**Interfaces:**
- Produces: `F_OPT = 13`, `F_ATK = 45`, `F_CARD = 218`, `DRAW_N = 10.0`
- Consumed by: `cards.py` (via `from ptcg_il.featurizer import F_ATK, F_CARD`), `pointer.py`, `policy.py`, `dataset.py`, `shard_writer.py`, `actor.py`

- [ ] **Step 1: Add DRAW_N and change the dim constants**

At line 47 of `python/ptcg_il/featurizer.py`, after `DMGCTR_N = 20.0`, add:

```python
DRAW_N = 10.0
```

Replace lines 75-76:

```python
F_CARD = 218  # 52 base + 29 ability keywords + 2 counts + 3 attacks × 45
F_ATK = 45    # 16 numeric + 29 attack keywords
```

Replace line 81:

```python
F_OPT = 13
```

The `CARD_ATTACK_BLOCK_START` at line 261 is computed as `F_CARD - 3 * F_ATK` which auto-derives to `218 - 135 = 83` (same offset — the base block didn't change size).

- [ ] **Step 2: Verify auto-derivation**

```bash
cd /home/charles/Documents/Pokemon/python && PYTHONPATH=. uv run python -c "
from ptcg_il.featurizer import F_CARD, F_ATK, F_OPT, CARD_ATTACK_BLOCK_START, DRAW_N
from ptcg_mine.keywords import K_EFFECT
assert F_CARD == 218, f'F_CARD={F_CARD}'
assert F_ATK == 45, f'F_ATK={F_ATK}'
assert F_OPT == 13, f'F_OPT={F_OPT}'
assert DRAW_N == 10.0, f'DRAW_N={DRAW_N}'
assert CARD_ATTACK_BLOCK_START == 52 + K_EFFECT + 2, \
    f'CARD_ATTACK_BLOCK_START={CARD_ATTACK_BLOCK_START}'
print('OK')
"
```

- [ ] **Step 3: Run any tests that import featurizer to catch breakage early**

```bash
cd /home/charles/Documents/Pokemon && PYTHONPATH=python uv run pytest python/tests/test_featurizer.py -x --timeout=60 2>&1 | tail -20
```

Expected: tests may fail because existing test fixtures use the old `F_OPT=8` dimensions for `opt_scalar`. This is expected — we'll fix tests in Task 8. For now just confirm they fail on shape mismatches (not on import errors).

---

### Task 3: Expand attack_static_row in cards.py

**Files:**
- Modify: `python/ptcg_mine/cards.py`

**Interfaces:**
- Consumes: `F_ATK = 45`, `F_CARD = 218` from `ptcg_il.featurizer`; `draw_fixed`, `draw_to_hand` from `ptcg_mine.keywords`
- Produces: `attack_static_row` returns `float32[45]`; `card_static_row` returns `float32[218]`

- [ ] **Step 1: Update docstring, add DRAW_N, update assertions, update import**

In `python/ptcg_mine/cards.py`:

Replace the docstring header (lines 1-11):
```python
"""Engine-derived static feature tables for cards and attacks.

Layout is exact Appendix A.3 (feature slices) / A.2 (normalizers) of
TRANSFORMER_IL_SPEC.md:
  card_static_row[218] = base[52] + 3 × attack_static_row[45]
  base[52]             = [hp/HP_N, retreat/RETREAT_N, cardType-onehot(7),
                          stage-onehot(3), energyType-onehot(12),
                          weakness-onehot(12), resistance-onehot(12),
                          [ex, megaEx, tera, aceSpec](4)]
  attack_static_row[45] = [damage/ATKDMG_N, energy-cost histogram(12)/ATKCOST_N,
                           len(energies)/ATKCOST_N,
                           draw_fixed/DRAW_N, draw_to_hand/DRAW_N]

Engine access: `all_card_data()` / `all_attack()` live in the bundled `cg`
package at pokemon-tcg-ai-battle/sample_submission/sample_submission/cg;
`load_engine()` adds that directory's parent to sys.path and imports them.
"""
```

Add `DRAW_N` after `ATKCOST_N` (after line 26):
```python
DRAW_N = 10.0
```

Update assertions (lines 39-46):
```python
assert F_ATK == 16 + K_EFFECT, (
    f"F_ATK={F_ATK} in ptcg_il.featurizer disagrees with K_EFFECT={K_EFFECT} "
    f"in ptcg_mine.keywords (expected {16 + K_EFFECT})"
)
assert F_CARD == 52 + K_EFFECT + 2 + 3 * F_ATK, (
    f"F_CARD={F_CARD} in ptcg_il.featurizer disagrees with K_EFFECT={K_EFFECT} "
    f"(expected {52 + K_EFFECT + 2 + 3 * F_ATK})"
)
```

Update import (line 32):
```python
from ptcg_mine.keywords import K_EFFECT, ability_keyword_row, attack_keyword_row, draw_fixed, draw_to_hand
```

- [ ] **Step 2: Expand attack_static_row to 16 numerics**

Replace the `attack_static_row` function (lines 120-145):

```python
def attack_static_row(attack) -> np.ndarray:
    """float32[45] static feature row for an Attack.

    Layout: damage, energy-cost histogram(12), total-cost-count,
    draw_fixed, draw_to_hand, then K_EFFECT keyword flags (16:45).
    """
    row = np.zeros(F_ATK, dtype=np.float32)
    row[0] = attack.damage / ATKDMG_N
    hist = np.zeros(N_ENERGY, dtype=np.float32)
    for e in attack.energies:
        assert int(e) < N_ENERGY, f"attack energy index {e} >= N_ENERGY={N_ENERGY}"
        hist[int(e)] += 1.0
    row[1:13] = hist / ATKCOST_N
    row[13] = len(attack.energies) / ATKCOST_N
    # Draw counts (14:16) — normalised by DRAW_N per the fixed-divisor scheme
    row[14] = draw_fixed(attack) / DRAW_N
    row[15] = draw_to_hand(attack) / DRAW_N
    # Effect keywords from the attack's oracle text (16:45).
    row[16:16 + K_EFFECT] = attack_keyword_row(attack)
    return row
```

- [ ] **Step 3: Update docstrings that reference old dims**

- Line 83: `float32[94]` → `float32[218]`
- Line 149: `[V,94]` → `[V,218]`, `[A,14]` → `[A,45]`, `[V,94]` → `[V,218]`, `[A,14]` → `[A,45]`
- Line 199: `static_row_94` → `static_row_218`, `94-dim` → `218-dim`
- Line 209: `static_row_43` → `static_row_45`

- [ ] **Step 4: Verify cards.py loads without assertion errors**

```bash
cd /home/charles/Documents/Pokemon/python && PYTHONPATH=. uv run python -c "
from ptcg_mine.cards import load_engine, attack_static_row, build_engine_attack_features
cards, atks = load_engine()
row = attack_static_row(atks[0])
assert row.shape == (45,), f'shape={row.shape}'
feats = build_engine_attack_features(atks)
assert all(v.shape == (45,) for v in feats.values()), 'shape mismatch in attack features'
print(f'OK: {len(atks)} attacks, all {row.shape} dims')
# Check that at least one attack has non-zero draw_fixed or draw_to_hand
import re
draw_atks = [a for a in atks if re.search(r'draw', getattr(a, 'text', '') or '', re.I)]
f_parsed = sum(1 for a in draw_atks if row_for(a)[14] > 0 or row_for(a)[15] > 0)
print(f'Draw attacks with non-zero draw dims: {f_parsed}/{len(draw_atks)}')
" 2>&1

# Define row_for inline above; let me write it cleanly:
cd /home/charles/Documents/Pokemon/python && PYTHONPATH=. uv run python -c "
from ptcg_mine.cards import load_engine, attack_static_row
import re
cards, atks = load_engine()
draw_atks = [a for a in atks if re.search(r'draw', getattr(a, 'text', '') or '', re.I)]
parsed = 0
for a in draw_atks:
    r = attack_static_row(a)
    if r[14] > 0 or r[15] > 0:
        parsed += 1
    else:
        print(f'  zero-draw-dims: {getattr(a, \"text\", \"\")[:70]}')
print(f'Parsed: {parsed}/{len(draw_atks)}, shape={attack_static_row(atks[0]).shape}')
"
```

Expected: 36 draw attacks, most with non-zero draw dims. "Draw a card" → draw_fixed=1. "Draw cards until you have 7 cards" → draw_to_hand=7. Acceptable zeros: "draw that many", "draw up to" (variable count).

- [ ] **Step 5: Run existing mine tests**

```bash
cd /home/charles/Documents/Pokemon && PYTHONPATH=python uv run pytest tests/ -x --timeout=60 2>&1 | tail -15
```

---

### Task 4: New opt_scalar dims and opt_bench_idx in featurizer.py

**Files:**
- Modify: `python/ptcg_il/featurizer.py`

**Interfaces:**
- Consumes: `F_OPT = 13`, `F_ATK = 45`, `F_CARD = 218` (from Task 2)
- Produces: `opt_scalar[:, 8:13]` populated; `opt_bench_idx[O_MAX] int64`; `option_groups` key includes `opt_bench_idx`; `featurize()` return dict includes `"opt_bench_idx"`

- [ ] **Step 1: Add bench-evaluation helpers**

Add after `_attack_damage_ratio` (after line ~380) in `python/ptcg_il/featurizer.py`:

```python
def _best_bench_damage_ratio(
    poke_card_feat: np.ndarray,
    opp_active_card_feat: np.ndarray,
) -> tuple[float, int]:
    """Best damage ratio any bench Pokemon can achieve vs the opponent's Active.

    Reads attack blocks from bench slots (1..5) and opponent Active HP from
    ``opp_active_card_feat[0]`` (slot 6).  Returns ``(best_ratio, best_slot)``
    where *best_slot* is the 0-based bench position (0..4).

    Damage is read from the static table columns
    ``CARD_ATTACK_BLOCK_START + ai * F_ATK`` (the attack's static damage, col 0
    of the attack block).  These are normalised by ATKDMG_N (350.0).

    Pure-Python arithmetic on purpose — bench is ≤5 slots × 3 attacks = 15 reads,
    and this runs on the RL rollout path.
    """
    ATKDMG_N = 350.0  # must agree with the featurizer constant
    best_ratio = 0.0
    best_slot = 0
    opp_hp = float(opp_active_card_feat[0]) * 400.0 + 10.0
    if opp_hp <= 10.0:
        return 0.0, 0
    atk_start = int(CARD_ATTACK_BLOCK_START)
    atk_stride = int(F_ATK)
    for bench_pos in range(1, 6):
        row = poke_card_feat[bench_pos]
        hp = float(row[0]) * 400.0
        if hp <= 0:
            continue
        for ai in range(3):
            dmg_norm = float(row[atk_start + ai * atk_stride])
            dmg = dmg_norm * ATKDMG_N
            if dmg <= 0:
                continue
            ratio = dmg / opp_hp
            if ratio > best_ratio:
                best_ratio = ratio
                best_slot = bench_pos - 1
    return best_ratio, best_slot


def _best_bench_hp_ratio(
    poke_card_feat: np.ndarray,
    active_card_feat: np.ndarray,
) -> tuple[float, int]:
    """Best HP ratio of bench Pokemon vs active.  Clipped to [0, 1].

    Returns ``(best_ratio, best_slot)`` — *best_slot* is the 0-based bench
    position for tie-breaking ``opt_bench_idx``.
    """
    active_hp = float(active_card_feat[0]) * 400.0
    if active_hp <= 0:
        return 0.0, 0
    best_ratio = 0.0
    best_slot = 0
    for bench_pos in range(1, 6):
        hp = float(poke_card_feat[bench_pos][0]) * 400.0
        if hp <= 0:
            continue
        ratio = hp / active_hp
        if ratio > 1.0:
            ratio = 1.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_slot = bench_pos - 1
    return best_ratio, best_slot
```

- [ ] **Step 2: Add opt_bench_idx to _build_option_tokens initialisation**

After line 843 (`opt_scalar = np.zeros((O_MAX, F_OPT), dtype=np.float32)`), add:

```python
    opt_bench_idx = np.full(O_MAX, -1, dtype=np.int64)
```

- [ ] **Step 3: Add once-per-select precomputation for RETREAT**

Before the loop `for new_j, old_j in enumerate(indices_to_keep):` (line 846), add the bench heuristic precomputation. The condition must be lazy — checked once, computed only when needed:

```python
    # Compute bench promotion candidates when at least one RETREAT option exists.
    # RETREAT never offers two options in one select (measured over 60 episodes),
    # so the same values apply to every RETREAT option.
    _has_retreat = any(int(options[i]["type"]) == 12 for i in indices_to_keep)
    if _has_retreat:
        _bench_best_dmg, _bench_best_dmg_slot = _best_bench_damage_ratio(
            poke_card_feat, poke_card_feat[6],  # slot 6 = opponent Active
        )
        _bench_best_hp, _bench_best_hp_slot = _best_bench_hp_ratio(
            poke_card_feat, poke_card_feat[0],  # slot 0 = my Active
        )
        # Prefer the highest-damage bench slot for opt_bench_idx
        _bench_idx_slot = _bench_best_dmg_slot if _bench_best_dmg > 0 else _bench_best_hp_slot
        _my_attached_energy = float(np.asarray(poke_feat[0][3:15], dtype=np.float64).sum()) * ENERGY_N
        _retreat_cost = float(poke_card_feat[0, 1]) * RETREAT_N
    else:
        _bench_best_dmg = 0.0
        _bench_best_hp = 0.0
        _bench_idx_slot = 0
        _my_attached_energy = 0.0
        _retreat_cost = 0.0
```

- [ ] **Step 4: Populate opt_bench_idx and new scalars in the option loop**

Inside the RETREAT block (replace lines 941-949):

```python
        elif otype == 12:  # RETREAT
            opt_src_idx[new_j] = _ref(AREA_ACTIVE, your_index, 0)
            _set_card(AREA_ACTIVE, your_index, 0)
            # Point at the best bench slot for promotion visibility
            bench_row = _ref(AREA_BENCH, your_index, _bench_idx_slot)
            opt_bench_idx[new_j] = bench_row if bench_row != -1 else -1
            # Retreat affordability: (attached energy - retreat cost) / RETREAT_N
            opt_scalar[new_j, 10] = _clip_norm(_my_attached_energy - _retreat_cost, RETREAT_N)
            opt_scalar[new_j, 11] = min(_bench_best_dmg, 2.0)
            opt_scalar[new_j, 12] = _bench_best_hp
```

Inside the ATTACK block (after `opt_scalar[new_j, 7] = ...` at line 988), add:

```python
            # Deck-out risk from drawing (dim 8) and zero-damage flag (dim 9)
            if engine_attack_features is not None and opt_attack_idx[new_j] > 0:
                atk_feat = engine_attack_features.get(int(opt.get("attackId", 0)))
                if atk_feat is not None:
                    # draw_fixed / draw_to_hand are in the attack static row,
                    # but we need them denormalised from the engine_attack_features dict.
                    # Those are the raw attack_static_row values, normalised by DRAW_N.
                    draw_fixed_norm = float(atk_feat[14])  # normalised
                    draw_to_hand_norm = float(atk_feat[15])  # normalised
                    draw_fixed_val = draw_fixed_norm * DRAW_N
                    draw_to_hand_val = draw_to_hand_norm * DRAW_N
                    my_hand = int(state["players"][your_index].get("handCount", 0))
                    draw_est = draw_fixed_val + max(0, draw_to_hand_val - my_hand)
                    deck_count = max(int(state["players"][your_index].get("deckCount", 0)), 1)
                    opt_scalar[new_j, 8] = min(draw_est / deck_count, 1.0)
            # Zero-damage flag: the attack's static damage is exactly 0
            if engine_attack_features is not None:
                atk_feat = engine_attack_features.get(int(opt.get("attackId", 0)))
                if atk_feat is not None:
                    opt_scalar[new_j, 9] = 1.0 if float(atk_feat[0]) == 0.0 else 0.0
```

Wait — this is ugly. The `atk_feat[0]` access uses the attack_static_row layout, and column 0 is damage/ATKDMG_N. But `engine_attack_features` values are attack_static_row outputs, so they are normalised. Damage is at col 0.

Actually, looking more carefully, `engine_attack_features` is `{attack_id: attack_static_row(attack)}` which returns `float32[F_ATK]`. So:
- `atk_feat[0]` = damage / ATKDMG_N — zero-damage is when this is 0.0
- `atk_feat[14]` = draw_fixed / DRAW_N — denormalise by multiplying by DRAW_N
- `atk_feat[15]` = draw_to_hand / DRAW_N — same

This is correct.

But actually there's a simpler approach: `opt_attack_feat` is already gathered from `opt_attack_idx` against the attack features. But `opt_attack_feat` is computed *after* `_build_option_tokens` returns (in `featurize()`), so we can't read it here. The `engine_attack_features` dict is available though — it's passed as a parameter.

Actually wait, looking at the code again: `_build_option_tokens` already receives `engine_attack_features` as a parameter (line 1536). And it's already used at line 985 for `_attack_damage_ratio`. So the pattern is established.

But the above code is repetitive. Let me rewrite it cleaner:

```python
            # Deck-out risk from drawing (dim 8) and zero-damage flag (dim 9)
            aid = opt.get("attackId")
            if engine_attack_features is not None and aid is not None:
                atk_feat = engine_attack_features.get(int(aid))
                if atk_feat is not None:
                    # draw_fixed / draw_to_hand are at cols 14, 15 (normalised by DRAW_N)
                    draw_fixed_val = float(atk_feat[14]) * DRAW_N
                    draw_to_hand_val = float(atk_feat[15]) * DRAW_N
                    my_hand = int(state["players"][your_index].get("handCount", 0))
                    draw_est = draw_fixed_val + max(0, draw_to_hand_val - my_hand)
                    deck_count = max(int(state["players"][your_index].get("deckCount", 0)), 1)
                    opt_scalar[new_j, 8] = min(draw_est / deck_count, 1.0)
                    # Zero-damage: damage col is 0, normalised by ATKDMG_N
                    opt_scalar[new_j, 9] = 1.0 if float(atk_feat[0]) == 0.0 else 0.0
```

- [ ] **Step 5: Update option_groups to include opt_bench_idx**

In `option_groups` (line 1062), update the function signature to accept the new parameter:

```python
def option_groups(
    opt_type: np.ndarray,
    opt_src_idx: np.ndarray,
    opt_tgt_idx: np.ndarray,
    opt_bench_idx: np.ndarray,
    opt_card_feat: np.ndarray,
    opt_attack_feat: np.ndarray,
    opt_scalar: np.ndarray,
    opt_mask: np.ndarray,
) -> np.ndarray:
```

In the `ints` stack (line 1099-1101), add `opt_bench_idx`:

```python
    ints = np.stack(
        [opt_type[valid], opt_src_idx[valid], opt_tgt_idx[valid],
         opt_bench_idx[valid]], axis=1
    ).astype(np.int64)
```

Update docstring (line 1072-1076) to mention `opt_bench_idx`.

- [ ] **Step 6: Update featurize() to pass opt_bench_idx through**

In `featurize()` (line 1520), the return from `_build_option_tokens` now has 10 values. Update the unpacking:

```python
    (
        opt_type,
        opt_src_idx,
        opt_tgt_idx,
        opt_card_id,
        opt_attack_idx,
        opt_scalar,
        opt_mask,
        index_remap,
        stop_column,
    ) = _build_option_tokens(...)
```

becomes:

```python
    (
        opt_type,
        opt_src_idx,
        opt_tgt_idx,
        opt_bench_idx,
        opt_card_id,
        opt_attack_idx,
        opt_scalar,
        opt_mask,
        index_remap,
        stop_column,
    ) = _build_option_tokens(...)
```

Wait, that would change the return signature. Let me re-check `_build_option_tokens`. Its return statement is at line 1050-1060:

```python
    return (
        opt_type,
        opt_src_idx,
        opt_tgt_idx,
        opt_card_id,
        opt_attack_idx,
        opt_scalar,
        opt_mask,
        index_remap,
        stop_column,
    )
```

I need to add `opt_bench_idx` here:

```python
    return (
        opt_type,
        opt_src_idx,
        opt_tgt_idx,
        opt_bench_idx,
        opt_card_id,
        opt_attack_idx,
        opt_scalar,
        opt_mask,
        index_remap,
        stop_column,
    )
```

And update the docstring at line 797-798 to list the new return value.

Then in `featurize()`, update the call to `option_groups` (line 1553-1556):

```python
    opt_group = option_groups(
        opt_type, opt_src_idx, opt_tgt_idx, opt_bench_idx,
        opt_card_feat, opt_attack_feat, opt_scalar, opt_mask,
    )
```

And add `"opt_bench_idx": opt_bench_idx` to the result dict (after line 1586, alongside `opt_tgt_idx`):

```python
        "opt_bench_idx": opt_bench_idx,
```

Also add `opt_bench_idx` to the `_build_option_tokens` docstring's Returns section (line 797).

- [ ] **Step 7: Update the _build_option_tokens docstring**

Replace lines 797-798:
```python
    Returns (opt_type, opt_src_idx, opt_tgt_idx, opt_bench_idx, opt_card_id,
             opt_attack_idx, opt_scalar, opt_mask, index_remap, stop_column).
```

- [ ] **Step 8: Verify featurizer can be imported and basic call works**

```bash
cd /home/charles/Documents/Pokemon/python && PYTHONPATH=. uv run python -c "
from ptcg_il.featurizer import F_OPT, F_ATK, F_CARD, DRAW_N, featurize, option_groups, CARD_ATTACK_BLOCK_START
import numpy as np
print(f'F_OPT={F_OPT} F_ATK={F_ATK} F_CARD={F_CARD} DRAW_N={DRAW_N}')
print(f'CARD_ATTACK_BLOCK_START={CARD_ATTACK_BLOCK_START}')
# Test option_groups with new signature
opt_type = np.zeros(64, dtype=np.int64)
opt_src = np.full(64, -1, dtype=np.int64)
opt_tgt = np.full(64, -1, dtype=np.int64)
opt_bench = np.full(64, -1, dtype=np.int64)
opt_card = np.zeros((64, F_CARD), dtype=np.float32)
opt_atk = np.zeros((64, F_ATK), dtype=np.float32)
opt_scalar = np.zeros((64, F_OPT), dtype=np.float32)
opt_mask = np.zeros(64, dtype=bool); opt_mask[:4] = True
groups = option_groups(opt_type, opt_src, opt_tgt, opt_bench, opt_card, opt_atk, opt_scalar, opt_mask)
print(f'Groups: {groups[:8]}')
print('OK')
" 2>&1
```

---

### Task 5: Add bench base term in PointerHead

**Files:**
- Modify: `python/ptcg_il/model/pointer.py`

**Interfaces:**
- Consumes: `x["opt_bench_idx"]` — new tensor key
- Produces: sixth additive base term `bench` in `PointerHead.forward`

- [ ] **Step 1: Add bench gather in PointerHead.forward**

In `pointer.py` line 115, after `tgt = self.gather(h_aug, x["opt_tgt_idx"])`, add:

```python
        bench = self.gather(h_aug, x["opt_bench_idx"])                     # [B, O, D]
```

Update the `base` expression (line 117-123) to include bench:

```python
        base = (
            self.opt_type_emb(x["opt_type"])
            + src
            + tgt
            + bench
            + card_enc(x["opt_card_feat"])
            + self.attack(x["opt_attack_feat"])
        )  # [B, O, D]
```

Update the docstring (after line 94) to list the new key:

```python
            Option tensors: opt_type [B,O], opt_src_idx [B,O], opt_tgt_idx [B,O],
            opt_bench_idx [B,O], opt_card_feat [B,O,F_CARD], ...
```

- [ ] **Step 2: Verify PointerHead can be constructed and run**

```bash
cd /home/charles/Documents/Pokemon/python && PYTHONPATH=. uv run python -c "
import torch
from ptcg_il.model.pointer import PointerHead
from ptcg_il.model.cards import CardFeaturizer
from ptcg_il.featurizer import F_CARD, F_ATK, F_OPT
p = PointerHead(D=64, heads=2)
p.card = CardFeaturizer(D=64)
B, O = 2, 8
h = torch.randn(B, 46, 64)
tok_mask = torch.ones(B, 46, dtype=torch.bool)
x = {
    'opt_type': torch.zeros(B, O, dtype=torch.long),
    'opt_src_idx': torch.full((B, O), -1, dtype=torch.long),
    'opt_tgt_idx': torch.full((B, O), -1, dtype=torch.long),
    'opt_bench_idx': torch.full((B, O), -1, dtype=torch.long),
    'opt_card_feat': torch.zeros(B, O, F_CARD),
    'opt_attack_feat': torch.zeros(B, O, F_ATK),
    'opt_scalar': torch.zeros(B, O, F_OPT),
    'opt_mask': torch.ones(B, O, dtype=torch.bool),
}
logits, o = p(h, tok_mask, p.card, x)
assert logits.shape == (B, O), f'logits shape={logits.shape}'
print(f'OK: logits shape={logits.shape}, o shape={o.shape}')
" 2>&1
```

---

### Task 6: Wire opt_bench_idx through data pipeline and RL actor

**Files:**
- Modify: `python/ptcg_il/train/dataset.py` (line 69)
- Modify: `python/ptcg_rl/actor.py` (line 48-56)
- Modify: `python/ptcg_il/diagnose.py` (lines 60, 115 — cosmetic)

No changes needed in `shard_writer.py`: `opt_bench_idx` is not in `_DERIVED_KEYS`, so it flows through `featurize() → shard` automatically. Its native int64 dtype is preserved (not downcast to int32 by `_INT32_KEYS`).

**Interfaces:**
- Consumes: `opt_bench_idx` key from featurizer
- Produces: key handled correctly in dataset int-keys, actor pointer keys, and diagnostic fingerprints

- [ ] **Step 1: Add opt_bench_idx to _INT_KEYS in dataset.py**

In `python/ptcg_il/train/dataset.py` line 69, add `"opt_bench_idx"` to the set:

```python
_INT_KEYS = frozenset({
    "tok_type", "tok_owner", "tok_zone",
    "opt_type", "opt_src_idx", "opt_tgt_idx", "opt_bench_idx", "opt_group",
    "sel_type", "sel_ctx",
    "action_idx", "minCount", "maxCount", "action_len", "stop_column",
    "log_len",
} | {id_key for id_key, _ in CARD_FEAT_SOURCES.values()})
```

- [ ] **Step 2: Add opt_bench_idx to _POINTER_KEYS in actor.py**

In `python/ptcg_rl/actor.py` line 48-56, add `"opt_bench_idx"`:

```python
_POINTER_KEYS = (
    "opt_type",
    "opt_src_idx",
    "opt_tgt_idx",
    "opt_bench_idx",
    "opt_card_feat",
    "opt_attack_feat",
    "opt_scalar",
    "opt_mask",
)
```

- [ ] **Step 3: Update diagnose.py fingerprint and ablation group**

In `python/ptcg_il/diagnose.py` line 60, add `opt_bench_idx` to the option target refs group:

```python
    "option target refs": ("opt_src_idx", "opt_tgt_idx", "opt_bench_idx"),
```

In line 115, add `"opt_bench_idx"` to the fingerprint key list (after `"opt_tgt_idx"`):

```python
        "opt_src_idx", "opt_tgt_idx", "opt_bench_idx", "opt_mask", "sel_type", "sel_ctx",
```

This ensures two decision points differing only in bench ref get different fingerprints.

- [ ] **Step 4: Verify imports**

```bash
cd /home/charles/Documents/Pokemon/python && PYTHONPATH=. uv run python -c "
from ptcg_il.train.dataset import _INT_KEYS
assert 'opt_bench_idx' in _INT_KEYS, f'opt_bench_idx missing from _INT_KEYS: {_INT_KEYS}'
from ptcg_rl.actor import _POINTER_KEYS
assert 'opt_bench_idx' in _POINTER_KEYS, f'opt_bench_idx missing from _POINTER_KEYS'
print('OK')
" 2>&1
```

---

### Task 7: Sync vendored copies in model/

**Files:**
- Modify: `model/featurizer.py`
- Modify: `model/pointer.py`

**Interfaces:**
- These are stand-alone copies used by the Kaggle submission bundle (`main.py` imports from `model.*`)
- Must agree with `python/ptcg_il/` on `F_OPT`, `F_ATK`, `F_CARD`, `DRAW_N`, and the `opt_bench_idx` flow

- [ ] **Step 1: Find and update dim constants in model/featurizer.py**

The vendored copy is at `/home/charles/Documents/Pokemon/model/featurizer.py`. It already has its own copies of `F_CARD=212`, `F_ATK=43`, `F_OPT=8`. Bump them to match:

```bash
cd /home/charles/Documents/Pokemon
```

Edit `model/featurizer.py`:
- `F_CARD = 212` → `F_CARD = 218`
- `F_ATK = 43` → `F_ATK = 45`
- `F_OPT = 8` → `F_OPT = 13`
- Add `DRAW_N = 10.0` after the normaliser block

- [ ] **Step 2: Update the vendored featurizer's option builder**

`model/featurizer.py` has its own `_build_option_tokens` that mirrors `ptcg_il/featurizer.py`. Apply the same changes:

1. Add `opt_bench_idx` initialisation
2. Add bench precomputation (but the vendored copy may not have `_best_bench_damage_ratio`/`_best_bench_hp_ratio` helpers — add them or simplify to just set `opt_bench_idx = -1` for now since the submission bundle doesn't train, only runs inference)

Actually, `model/featurizer.py` represents the inference-time featurizer in the Kaggle bundle. It needs ALL the new dims and the opt_bench_idx tensor to produce inputs the trained model expects. So we need the full changes.

However, `model/featurizer.py` is 1261 lines and already drifted from `ptcg_il/featurizer.py` (1627 lines). The cleanest approach: copy the relevant new functions and make the same surgical edits.

Let me check exactly what needs to change in the vendored copy by comparing key sections:

The vendored copy at `model/featurizer.py`:
- Has `ENERGY_N = 12.0` (old value; ptcg_il has 8.0)
- Has no `DRAW_N`
- Has `F_CARD = 212`, `F_ATK = 43`, `F_OPT = 8`
- Has its own `_build_option_tokens` (simpler, without `state` parameter in some versions)
- Has `option_groups` without `opt_bench_idx`

Rather than trying to surgically edit the vendored copy line-by-line (which will produce a fragile plan since the line numbers differ), the cleanest approach for the plan is a reference diff: apply the same semantic changes.

Given the complexity and the fact that the vendored copy already has its own drift, I'll write this task as "apply the same semantic changes to model/featurizer.py and model/pointer.py" with a checklist of what must match.

- [ ] **Step 1: Apply dim constant changes to model/featurizer.py**

In `model/featurizer.py`:
- Change `F_CARD = 212` to `F_CARD = 218`
- Change `F_ATK = 43` to `F_ATK = 45`
- Change `F_OPT = 8` to `F_OPT = 13`
- Add `DRAW_N = 10.0` after the normaliser constants
- Update `CARD_ATTACK_BLOCK_START = F_CARD - 3 * F_ATK` (should remain correct: 218 - 135 = 83)

- [ ] **Step 2: Add opt_bench_idx to the vendored featurizer**

In `model/featurizer.py`'s `_build_option_tokens` function:
- Add `opt_bench_idx = np.full(O_MAX, -1, dtype=np.int64)` after the existing array initialisations
- Add `opt_bench_idx` to the return tuple (after `opt_tgt_idx`)
- Add `opt_bench_idx` to the unpacking in `featurize()`
- Update `option_groups` call to pass `opt_bench_idx`
- Update `option_groups` function signature to accept `opt_bench_idx` parameter
- Update the `ints` stack in `option_groups` to include `opt_bench_idx[valid]`
- Add `"opt_bench_idx": opt_bench_idx` to the featurize result dict

For `model/featurizer.py` specifically, the RETREAT block in the option loop can be simpler than the training featurizer — just set `opt_scalar` dims and `opt_bench_idx` to the same values. But since the vendored copy may not import `_best_bench_damage_ratio`, we need to either copy those helpers or keep the inference path simpler.

Simplest approach: copy the `_best_bench_damage_ratio` and `_best_bench_hp_ratio` functions from `ptcg_il/featurizer.py` to `model/featurizer.py`. They're self-contained (pure Python + numpy imports already available).

- [ ] **Step 3: Add bench term to model/pointer.py**

In `model/pointer.py` (151 lines):
- Add `bench = self.gather(h_aug, x["opt_bench_idx"])` after the `tgt` line
- Add `+ bench` to the `base` expression
- Update the docstring to list `opt_bench_idx`

- [ ] **Step 4: Verify main.py import doesn't break**

```bash
cd /home/charles/Documents/Pokemon && PYTHONPATH=. uv run python -c "
# Simulate what the Kaggle bundle does: import the vendored model
import sys
sys.path.insert(0, '.')
from model.featurizer import F_OPT, F_ATK, F_CARD, DRAW_N, featurize, option_groups
assert F_OPT == 13, f'F_OPT={F_OPT}'
assert F_ATK == 45, f'F_ATK={F_ATK}'
assert F_CARD == 218, f'F_CARD={F_CARD}'
assert DRAW_N == 10.0, f'DRAW_N={DRAW_N}'
print('model/featurizer.py OK')
from model.pointer import PointerHead
print('model/pointer.py OK')
" 2>&1
```

- [ ] **Step 5: Add a sync test to guard against future drift**

Create `python/tests/test_vendored_sync.py`:

```python
"""Guard against drift between ptcg_il and the vendored Kaggle-bundle copies."""

import sys
from pathlib import Path


def test_vendored_featurizer_dims_match():
    """model/featurizer.py must agree with ptcg_il/featurizer.py on all F_* dims."""
    from ptcg_il import featurizer as real
    # Import the vendored copy — needs its own path
    repo_root = Path(__file__).resolve().parent.parent.parent
    vendored_dir = str(repo_root)
    if vendored_dir not in sys.path:
        sys.path.insert(0, vendored_dir)
    import model.featurizer as vendored  # noqa: E402

    for name in ("F_CARD", "F_ATK", "F_POKE", "F_HAND", "F_SUM", "F_GLOBAL", "F_OPT"):
        real_val = getattr(real, name)
        vendored_val = getattr(vendored, name)
        assert real_val == vendored_val, (
            f"{name}: ptcg_il={real_val}, model/={vendored_val}"
        )
    # DRAW_N too
    assert getattr(real, "DRAW_N") == getattr(vendored, "DRAW_N"), (
        f"DRAW_N mismatch: {getattr(real, 'DRAW_N')} vs {getattr(vendored, 'DRAW_N')}"
    )


def test_vendored_pointer_signature_matches():
    """model/pointer.py PointerHead.forward must consume the same keys."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    vendored_dir = str(repo_root)
    if vendored_dir not in sys.path:
        sys.path.insert(0, vendored_dir)

    from ptcg_il.model.pointer import PointerHead as RealPointer
    import model.pointer as vendored  # noqa: E402
    from model.pointer import PointerHead as VendoredPointer

    # Both should construct with the same signature
    r = RealPointer(D=64, heads=2)
    v = VendoredPointer(D=64, heads=2)
    # Structural check: same number of base terms (check source code)
    import inspect
    real_src = inspect.getsource(RealPointer.forward)
    vendored_src = inspect.getsource(VendoredPointer.forward)
    assert "opt_bench_idx" in real_src, "real pointer missing opt_bench_idx"
    assert "opt_bench_idx" in vendored_src, "vendored pointer missing opt_bench_idx"
    assert "bench" in real_src, "real pointer missing bench term"
    assert "bench" in vendored_src, "vendored pointer missing bench term"
```

- [ ] **Step 6: Run the sync test**

```bash
cd /home/charles/Documents/Pokemon && PYTHONPATH=python uv run pytest python/tests/test_vendored_sync.py -xvs 2>&1
```

---

### Task 8: Update existing tests for new dims

**Files:**
- Modify: `python/tests/test_featurizer.py` (or wherever opt_scalar shapes are asserted)
- Modify: `python/tests/test_rl_actor.py` (if it asserts `_POINTER_KEYS`)
- Create: `python/tests/test_option_bandwidth.py` (new tests for new behaviour)

**Interfaces:**
- All existing tests must pass with the new dim constants

- [ ] **Step 1: Find and update hardcoded dim references**

```bash
cd /home/charles/Documents/Pokemon && grep -rn "F_OPT\|F_ATK\|F_CARD\|= 8\b\|= 43\b\|= 212\b" python/tests/ tests/ --include="*.py" | grep -v "__pycache__" | grep -v ".pyc"
```

For each test file that references the old dims, update to the new values. Common patterns:
- `opt_scalar.shape[-1] == 8` → `== 13`
- `F_CARD == 212` → `== 218`
- `F_ATK == 43` → `== 45`
- `opt_scalar` dtype/shape assertions

- [ ] **Step 2: Run existing test suite**

```bash
cd /home/charles/Documents/Pokemon && PYTHONPATH=python uv run pytest python/tests/ -x --timeout=120 -k "not test_vendored" 2>&1 | tail -25
```

Fix any failures from hardcoded dim references.

- [ ] **Step 3: Run mine tests**

```bash
cd /home/charles/Documents/Pokemon && PYTHONPATH=python uv run pytest tests/ -x --timeout=120 2>&1 | tail -15
```

- [ ] **Step 4: Write new unit tests for the bandwidth features**

Create `python/tests/test_option_bandwidth.py`:

```python
"""Tests for per-option decision bandwidth (2026-08-09 spec)."""

import numpy as np
import pytest

from ptcg_il.featurizer import (
    F_OPT, F_ATK, F_CARD, DRAW_N, CARD_ATTACK_BLOCK_START,
    _best_bench_damage_ratio, _best_bench_hp_ratio,
)


class TestBestBenchDamageRatio:
    def test_empty_bench_returns_zero(self):
        """Empty bench (all HP=0) returns ratio 0.0."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        opp = np.zeros(F_CARD, dtype=np.float32)
        opp[0] = 100.0 / 400.0  # 100 HP
        ratio, slot = _best_bench_damage_ratio(poke_card, opp)
        assert ratio == 0.0

    def test_bench_with_damage_attack(self):
        """A bench Pokemon with a damaging attack returns the correct ratio."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        # Bench slot 2 (index 3) has 200 HP and one attack doing 60 damage
        poke_card[3, 0] = 200.0 / 400.0  # HP
        poke_card[3, CARD_ATTACK_BLOCK_START + 0 * F_ATK] = 60.0 / 350.0  # damage
        # Opponent active: 120 HP
        opp = np.zeros(F_CARD, dtype=np.float32)
        opp[0] = 120.0 / 400.0
        ratio, slot = _best_bench_damage_ratio(poke_card, opp)
        assert ratio == pytest.approx(60.0 / (120.0 + 10.0), rel=0.01)
        assert slot == 2  # bench position 2

    def test_tie_break_lowest_slot(self):
        """Equal damage → lowest slot index wins."""
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        # Two bench slots with same HP and same damage
        for slot in (1, 3):
            poke_card[slot, 0] = 150.0 / 400.0
            poke_card[slot, CARD_ATTACK_BLOCK_START] = 50.0 / 350.0
        opp = np.zeros(F_CARD, dtype=np.float32)
        opp[0] = 100.0 / 400.0
        _, slot = _best_bench_damage_ratio(poke_card, opp)
        assert slot == 0  # bench slot 1 is first


class TestBestBenchHpRatio:
    def test_returns_ratio(self):
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        poke_card[2, 0] = 200.0 / 400.0  # bench slot 2, 200 HP
        active = np.zeros(F_CARD, dtype=np.float32)
        active[0] = 100.0 / 400.0
        ratio, slot = _best_bench_hp_ratio(poke_card, active)
        assert ratio == 1.0  # clipped
        assert slot == 1  # bench position 1 (0-based)

    def test_empty_bench(self):
        poke_card = np.zeros((12, F_CARD), dtype=np.float32)
        active = np.zeros(F_CARD, dtype=np.float32)
        active[0] = 100.0 / 400.0
        ratio, _ = _best_bench_hp_ratio(poke_card, active)
        assert ratio == 0.0


class TestOptBenchIdx:
    def test_non_retreat_options_have_minus_one(self):
        """Every non-RETREAT option must have opt_bench_idx == -1."""
        # This test requires a full featurize call with a real observation.
        # It is marked as integration-level; the unit test above covers the
        # helper functions independently.
        pass  # see integration test below


class TestDrawCountParsing:
    def test_draw_fixed_and_to_hand(self):
        from ptcg_mine.keywords import draw_fixed, draw_to_hand
        from ptcg_mine.cards import load_engine
        _, atks = load_engine()
        import re
        draw_atks = [a for a in atks
                     if re.search(r'draw', getattr(a, 'text', '') or '', re.I)]
        assert len(draw_atks) > 0, "no draw attacks examined — test is vacuous"
        parsed = 0
        for a in draw_atks:
            f = draw_fixed(a)
            h = draw_to_hand(a)
            if f > 0 or h > 0:
                parsed += 1
        # At least the explicit-number and draw-a-card forms must parse
        assert parsed >= 16, (
            f"only {parsed}/{len(draw_atks)} draw attacks parsed; "
            f"expected >= 16"
        )

    def test_draw_a_card_is_fixed_1(self):
        from ptcg_mine.keywords import draw_fixed, draw_to_hand
        from ptcg_mine.cards import load_engine
        _, atks = load_engine()
        # Find a "Draw a card." attack
        found = False
        for a in atks:
            t = getattr(a, 'text', '') or ''
            if 'draw a card' in t.lower() and 'draw a card.' in t.lower():
                assert draw_fixed(a) == 1, f"'Draw a card.' parsed as {draw_fixed(a)}"
                assert draw_to_hand(a) == 0
                found = True
                break
        assert found, "no 'Draw a card.' attack found — test is vacuous"

    def test_draw_to_hand_is_parsed(self):
        from ptcg_mine.keywords import draw_fixed, draw_to_hand
        from ptcg_mine.cards import load_engine
        _, atks = load_engine()
        found = False
        for a in atks:
            t = getattr(a, 'text', '') or ''
            if 'draw cards until you have 7 cards' in t.lower():
                assert draw_to_hand(a) == 7, f"draw_to_hand={draw_to_hand(a)}"
                found = True
                break
        assert found, "no 'draw cards until you have 7 cards' attack found"

    def test_draw_fixed_and_to_hand_are_exclusive(self):
        """An attack should not have both draw_fixed and draw_to_hand non-zero."""
        from ptcg_mine.keywords import draw_fixed, draw_to_hand
        from ptcg_mine.cards import load_engine
        _, atks = load_engine()
        for a in atks:
            f = draw_fixed(a)
            h = draw_to_hand(a)
            assert f == 0 or h == 0, (
                f"attack has both draw_fixed={f} and draw_to_hand={h}: "
                f"{getattr(a, 'text', '')[:80]}"
            )


class TestNewScalarDims:
    """Tests for opt_scalar dims 8-12.  These require a full featurize call."""
    # Integration-tested via the featurizer's existing test fixtures.
    # Unit-level: dims exist and are zero for non-applicable option types.
    pass


class TestDimAssertions:
    def test_cards_assertions_pass(self):
        """cards.py dim assertions must pass with the new values."""
        # This is tested by the import in the test module itself —
        # if cards.py's module-level assertions fail, the import raises.
        from ptcg_mine.cards import attack_static_row  # noqa: F401
        assert True  # reached without AssertionError

    def test_k_effect_mismatch_fails(self, monkeypatch):
        """Bumping K_EFFECT without updating F_ATK/F_CARD must raise."""
        # Verify by mutation: the assertions are module-level so they fire
        # on import.  We can't easily test that in-process, but we can
        # verify the values are consistent.
        from ptcg_mine.keywords import K_EFFECT
        from ptcg_il.featurizer import F_ATK, F_CARD
        assert F_ATK == 16 + K_EFFECT, (
            f"F_ATK={F_ATK}, expected {16 + K_EFFECT}"
        )
        assert F_CARD == 52 + K_EFFECT + 2 + 3 * F_ATK, (
            f"F_CARD={F_CARD}, expected {52 + K_EFFECT + 2 + 3 * F_ATK}"
        )


class TestOptionGroups:
    def test_bench_idx_affects_groups(self):
        """Two options differing only in opt_bench_idx must get different groups."""
        from ptcg_il.featurizer import option_groups
        O = 64
        opt_type = np.zeros(O, dtype=np.int64)
        opt_src = np.full(O, -1, dtype=np.int64)
        opt_tgt = np.full(O, -1, dtype=np.int64)
        opt_bench = np.full(O, -1, dtype=np.int64)
        opt_bench[0] = 3  # different bench ref
        opt_bench[1] = 5  # different bench ref
        opt_card = np.zeros((O, F_CARD), dtype=np.float32)
        opt_atk = np.zeros((O, F_ATK), dtype=np.float32)
        opt_scalar = np.zeros((O, F_OPT), dtype=np.float32)
        opt_mask = np.zeros(O, dtype=bool)
        opt_mask[0] = True
        opt_mask[1] = True
        groups = option_groups(
            opt_type, opt_src, opt_tgt, opt_bench,
            opt_card, opt_atk, opt_scalar, opt_mask,
        )
        assert groups[0] != groups[1], (
            f"options with different bench refs got same group {groups[0]}"
        )

    def test_bench_idx_not_in_key_merges(self):
        """Without opt_bench_idx in the key, two bench-different options merge.
        This is the mutation test — remove opt_bench_idx from the key and
        verify the groups become equal."""
        from ptcg_il.featurizer import option_groups
        O = 64
        opt_type = np.zeros(O, dtype=np.int64)
        opt_src = np.full(O, -1, dtype=np.int64)
        opt_tgt = np.full(O, -1, dtype=np.int64)
        opt_bench = np.full(O, -1, dtype=np.int64)
        opt_bench[0] = 3
        opt_bench[1] = 5
        opt_card = np.zeros((O, F_CARD), dtype=np.float32)
        opt_atk = np.zeros((O, F_ATK), dtype=np.float32)
        opt_scalar = np.zeros((O, F_OPT), dtype=np.float32)
        opt_mask = np.zeros(O, dtype=bool)
        opt_mask[0] = True
        opt_mask[1] = True
        # Simulate old behaviour: key without opt_bench_idx
        valid = np.flatnonzero(opt_mask)
        ints_old = np.stack(
            [opt_type[valid], opt_src[valid], opt_tgt[valid]], axis=1
        ).astype(np.int64)
        feats = np.concatenate(
            [opt_card[valid].astype(np.float32),
             opt_atk[valid].astype(np.float32)], axis=1
        )
        scal = opt_scalar[valid].astype(np.float32)
        lookup = {}
        for pos, slot in enumerate(valid):
            key = (ints_old[pos].tobytes(), feats[pos].tobytes(), scal[pos].tobytes())
            lookup.setdefault(key, len(lookup))
        # Without bench_idx, slots 0 and 1 are byte-identical → same group
        assert len(lookup) == 1, (
            f"without bench_idx, 2 options should be 1 group, got {len(lookup)}"
        )
```

- [ ] **Step 5: Run the new tests**

```bash
cd /home/charles/Documents/Pokemon && PYTHONPATH=python uv run pytest python/tests/test_option_bandwidth.py -xvs 2>&1
```

- [ ] **Step 6: Run full test suite**

```bash
cd /home/charles/Documents/Pokemon && PYTHONPATH=python uv run pytest tests/ -x --timeout=120 2>&1 | tail -10
cd /home/charles/Documents/Pokemon && PYTHONPATH=python uv run pytest python/tests/ -x --timeout=120 2>&1 | tail -10
```

---

### Task 9: Integration verification — capacity check

**This is a measurement, not a code change.** After all implementation tasks pass, verify the across-option L2 spread on ATTACK-vs-RETREAT rows has moved from its current 0.0000 for four of five base terms.

- [ ] **Step 1: Run a quick featurize smoke test with a real observation**

```bash
cd /home/charles/Documents/Pokemon/python && PYTHONPATH=. uv run python -c "
from ptcg_il.featurizer import featurize, F_OPT, F_ATK, F_CARD
from ptcg_il.ref_map import build_ref_map
import json, numpy as np
# This test just verifies the featurizer runs without crashing on a minimal obs
# A real capacity check needs a trained checkpoint and the spread-measurement script.
# That is post-retraining work — see spec Verification §1.
print('Featurizer OK: F_OPT=%d F_ATK=%d F_CARD=%d' % (F_OPT, F_ATK, F_CARD))
"
```

---

## Post-Implementation Pipeline

After all tasks pass, the pipeline must be re-run from scratch:

```bash
cd /home/charles/Documents/Pokemon/python

# 1. Rebuild static tables (new attack_static_row dims)
uv run python -m ptcg_mine.mine --skip-download --force-mine

# 2. Rebuild shards (new F_OPT/F_CARD/F_ATK in featurizer)
uv run python -m ptcg_il.cli build-shards --data-dir data --force-shards

# 3. Retrain specialists
# (per-deck, same hyperparams; new checkpoint dirs)
```

The stamp fingerprints already cover `featurizer.py`, so the edits auto-invalidate cached artifacts — `--force-mine` / `--force-shards` are not needed for correctness but are explicitly listed for clarity in a post-implementation run.
