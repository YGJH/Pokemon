# Option-Collision Groups + Shard-Internal Round-Trip Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop charging the policy for choosing between options it physically cannot tell apart, and replace the never-executed reference round-trip QA gate with one that actually runs.

**Architecture:** The featurizer emits a new per-option `opt_group` index — options in the same group are byte-identical model inputs. That one array is consumed in three places: the training CE marginalises over the target's group, offline eval reports a collision-adjusted top-1, and the QA collision audit reports the true (not overestimated) share. Separately, `check_reference_roundtrip` is rewritten to compare `opt_card_feat` against the state tensors *inside the shard*, deleting the observation-file path that never had a producer.

**Tech Stack:** Python 3.11 / uv, NumPy, PyTorch, pytest.

## Global Constraints

- All commands run from `python/` (`cd python` first). `ptcg_il`/`ptcg_mine` are not importable from the repo root.
- Run the two test directories **separately** — they share basenames and collection fails otherwise:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/` and `... uv run pytest tests/` (from repo root).
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is mandatory — ROS steals plugin autoload in this environment.
- Tests that count occurrences **must fail on zero examined**. A loop that never executes must not report a pass.
- Validate every new guard by **mutation**: break the thing deliberately, confirm the test goes red, revert.
- `opt_group` is `int64` and must **never** be added to `shard_writer._FP16_KEYS`.
- Do **not** add anything to `policy.FEATURE_DIM_KEYS` (`F_CARD, F_ATK, F_POKE, F_HAND, F_SUM, F_GLOBAL, F_OPT`). `opt_group` is an index array, not a feature width; adding it would invalidate every existing checkpoint's `feat_dims` pin for no reason.
- Editing `featurizer.py` invalidates the build-shards stamp automatically (`ptcg_mine/stamp.py` fingerprints the source). Shards get rebuilt; **existing checkpoints stay loadable** and stay valid.
- Group-marginal CE is **on by default**, with `--no-group-marginal-ce` as the escape.
- Baseline numbers this plan is measured against, from `data/shards/train-00000.npz` (50 000 samples, 387 443 valid options):
  - duplicate options under `(src, card_feat)` — the *old* audit's definition: **47.79 %**
  - duplicate options under the **full model input**: **10.00 %** of options, **15.58 %** of samples
  - samples whose label splits a duplicate group: **6 957 / 50 000 = 13.9 %**
  - fully-identical options by `opt_type`: `{3: 38 417, 15: 304, 9: 4}` — attachment types 4/5/6 contribute **zero**
  - every type-3 collision has `opt_src_idx == -1`, `sel_type == 1`, `opt_scalar` all-zero

---

### Task 1: `option_groups()` in the featurizer

Equivalence classes over the 64 option slots. Two slots share a group iff every tensor the pointer head reads for them is identical.

**Files:**
- Modify: `python/ptcg_il/featurizer.py` (new function after `_build_option_tokens`, ends line 944; new key in the `result` dict at line 1214-1260)
- Test: `python/tests/test_featurizer.py`

**Interfaces:**
- Produces: `option_groups(opt_type, opt_src_idx, opt_tgt_idx, opt_card_feat, opt_attack_feat, opt_scalar, opt_mask) -> np.ndarray[O_MAX] int64`. Valid slots get `0..G-1` in first-appearance order; masked-out slots get `-1`.
- Produces: `featurize(...)["opt_group"]` — `int64[O_MAX]`.

- [ ] **Step 1: Write the failing tests**

Add to `python/tests/test_featurizer.py`:

```python
from ptcg_il.featurizer import O_MAX, F_CARD, F_ATK, F_OPT, option_groups


def _opt_arrays(n_valid: int):
    """Blank option tensors with the first n_valid slots marked valid."""
    return {
        "opt_type": np.zeros(O_MAX, dtype=np.int64),
        "opt_src_idx": np.full(O_MAX, -1, dtype=np.int64),
        "opt_tgt_idx": np.full(O_MAX, -1, dtype=np.int64),
        "opt_card_feat": np.zeros((O_MAX, F_CARD), dtype=np.float32),
        "opt_attack_feat": np.zeros((O_MAX, F_ATK), dtype=np.float32),
        "opt_scalar": np.zeros((O_MAX, F_OPT), dtype=np.float32),
        "opt_mask": np.array([i < n_valid for i in range(O_MAX)]),
    }


def test_option_groups_merges_identical_options():
    a = _opt_arrays(3)
    a["opt_type"][:3] = 3
    a["opt_card_feat"][:3, 7] = 1.0          # all three are the same card
    g = option_groups(**a)
    assert g[0] == g[1] == g[2]
    assert (g[3:] == -1).all(), "masked slots must be -1"


def test_option_groups_separates_on_each_field():
    for field, setter in [
        ("opt_type", lambda a: a["opt_type"].__setitem__(1, 5)),
        ("opt_src_idx", lambda a: a["opt_src_idx"].__setitem__(1, 4)),
        ("opt_tgt_idx", lambda a: a["opt_tgt_idx"].__setitem__(1, 4)),
        ("opt_card_feat", lambda a: a["opt_card_feat"].__setitem__((1, 3), 1.0)),
        ("opt_attack_feat", lambda a: a["opt_attack_feat"].__setitem__((1, 2), 1.0)),
        ("opt_scalar", lambda a: a["opt_scalar"].__setitem__((1, 2), 0.25)),
    ]:
        a = _opt_arrays(2)
        setter(a)
        g = option_groups(**a)
        assert g[0] != g[1], f"{field} must split the group"


def test_option_groups_ignores_fp32_noise_below_fp16_resolution():
    """Shards store card features as fp16, so the model cannot see a smaller
    difference than fp16 resolution — grouping must not either."""
    a = _opt_arrays(2)
    a["opt_card_feat"][0, 0] = 1.0
    a["opt_card_feat"][1, 0] = 1.0 + 1e-8
    g = option_groups(**a)
    assert g[0] == g[1]


def test_option_groups_all_masked_returns_all_minus_one():
    g = option_groups(**_opt_arrays(0))
    assert (g == -1).all()


def test_featurize_emits_opt_group():
    ep = _load_episode()
    vocab = _build_test_vocab(ep)
    obs, action = _get_active_step(ep, 8, 0)   # the MAIN select used elsewhere in this file
    out = featurize(obs, vocab, action)
    assert out["opt_group"].dtype == np.int64
    assert out["opt_group"].shape == (O_MAX,)
    valid = out["opt_mask"]
    assert (out["opt_group"][~valid] == -1).all()
    assert (out["opt_group"][valid] >= 0).all()
    assert valid.sum() > 0, "fixture produced no options — the assertions above are vacuous"
```

- [ ] **Step 2: Run the tests, confirm they fail**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_featurizer.py -k option_group -v
```
Expected: `ImportError: cannot import name 'option_groups'`.

- [ ] **Step 3: Implement `option_groups`**

Insert in `python/ptcg_il/featurizer.py` immediately after `_build_option_tokens` (after line 944):

```python
def option_groups(
    opt_type: np.ndarray,
    opt_src_idx: np.ndarray,
    opt_tgt_idx: np.ndarray,
    opt_card_feat: np.ndarray,
    opt_attack_feat: np.ndarray,
    opt_scalar: np.ndarray,
    opt_mask: np.ndarray,
) -> np.ndarray:
    """Equivalence classes over option slots: same id == identical model input.

    The pointer head reads exactly ``opt_type``, the gathered rows named by
    ``opt_src_idx``/``opt_tgt_idx``, ``opt_card_feat``, ``opt_attack_feat`` and
    ``opt_scalar`` (``model/pointer.py:117-130``).  Two options agreeing on all
    of them produce the same logit by construction, so a label that names one
    of them and not the other asks for a distinction the network cannot make.
    Measured on ``train-00000``: 10.0% of options and 15.6% of samples contain
    such a pair, and 13.9% of samples have a label that splits one.

    Card and attack features are compared **at fp16**, because that is the
    precision shards store them at (``shard_writer._FP16_KEYS``) and therefore
    the precision the model actually reads.  Grouping at fp32 would call two
    options distinct that are byte-identical by the time they reach training.

    Returns ``int64[O_MAX]``: ``0..G-1`` in first-appearance order for valid
    slots, ``-1`` for masked slots (so a padded slot never matches anything).
    """
    groups = np.full(O_MAX, -1, dtype=np.int64)
    valid = np.flatnonzero(opt_mask)
    if valid.size == 0:
        return groups

    ints = np.stack(
        [opt_type[valid], opt_src_idx[valid], opt_tgt_idx[valid]], axis=1
    ).astype(np.int64)
    feats = np.concatenate(
        [
            opt_card_feat[valid].astype(np.float16),
            opt_attack_feat[valid].astype(np.float16),
        ],
        axis=1,
    )
    scal = opt_scalar[valid].astype(np.float32)

    lookup: dict[tuple[bytes, bytes, bytes], int] = {}
    for pos, slot in enumerate(valid):
        key = (ints[pos].tobytes(), feats[pos].tobytes(), scal[pos].tobytes())
        gid = lookup.setdefault(key, len(lookup))
        groups[slot] = gid
    return groups
```

- [ ] **Step 4: Wire it into `featurize()`**

In `python/ptcg_il/featurizer.py`, the `result` dict currently builds the option feature rows inline (lines 1239-1240). Hoist them so the groups can be computed from the same arrays, then add the key. Replace:

```python
        "opt_card_feat": _cfeat(opt_card_id),
        "opt_attack_feat": _afeat(opt_attack_idx),
```

with a lookup of two locals defined just above the `result = {` literal (line 1214):

```python
    opt_card_feat = _cfeat(opt_card_id)
    opt_attack_feat = _afeat(opt_attack_idx)
    opt_group = option_groups(
        opt_type, opt_src_idx, opt_tgt_idx,
        opt_card_feat, opt_attack_feat, opt_scalar, opt_mask,
    )
    result = {
        ...
        "opt_card_feat": opt_card_feat,
        "opt_attack_feat": opt_attack_feat,
        "opt_group": opt_group,
        ...
    }
```

`option_groups` must run **after** the STOP column is appended (it is — `_build_option_tokens` sets `opt_mask[n_opts] = True` for STOP at line 931 before returning). STOP carries `opt_type = STOP_OPT_TYPE = 17`, which no real option uses, so it always lands in a group of its own.

- [ ] **Step 5: Run the tests, confirm they pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_featurizer.py -v
```
Expected: PASS, including the pre-existing featurizer tests.

- [ ] **Step 6: Mutation check**

Temporarily drop `scal[pos].tobytes()` from the key tuple, re-run `test_option_groups_separates_on_each_field`, confirm the `opt_scalar` case goes red, then revert.

- [ ] **Step 7: Commit**

```bash
git add python/ptcg_il/featurizer.py python/tests/test_featurizer.py
git commit -m "feat(featurizer): emit opt_group equivalence classes per decision point"
```

---

### Task 2: Shard + dataset plumbing for `opt_group`

**Files:**
- Modify: `python/ptcg_il/train/dataset.py:49-56` (`_INT_KEYS`), `:369-403` (`__getitem__`)
- Test: `python/tests/test_dataset.py`, `python/tests/test_shard_writer.py`
- Also modify (synthetic sample builders — they hand-write the shard key set, so without this every downstream test silently exercises the legacy fallback): `python/tests/test_dataset.py:74` `_make_synthetic_sample`, `python/tests/test_train_loop.py:65` `_synthetic_shard_sample`, `python/tests/test_model_policy.py:17` `_make_synthetic_batch`. Each already builds `opt_type`/`opt_src_idx`/`opt_mask` in the same block; add beside them:

```python
        # numpy builders (test_dataset.py, test_train_loop.py)
        "opt_group": np.full(O_MAX, -1, dtype=np.int64),
        # torch builder (test_model_policy.py) — [B, O]
        "opt_group": torch.full((B, O), -1, dtype=torch.long),
```

and fill `0..n-1` into the slots each helper marks valid in `opt_mask`.

**Interfaces:**
- Consumes: `featurize(...)["opt_group"]` from Task 1.
- Produces: `batch["opt_group"]` — `int64[B, O_MAX]` out of `collate_fn`. Batches built from shards written before this change get an all-distinct fallback (`arange(O_MAX)` masked to `-1`), which makes every downstream group operation a no-op.

- [ ] **Step 1: Write the failing tests**

Add to `python/tests/test_dataset.py`:

```python
def test_dataset_yields_opt_group_as_int64():
    data_dir = _build_synthetic_data(n_train=8, n_val=4)
    ds = ShardDataset(data_dir, split="train")
    sample = ds[0]
    assert sample["opt_group"].dtype == torch.int64
    assert sample["opt_group"].shape == (O_MAX,)


def test_dataset_backfills_opt_group_for_legacy_shards():
    """A shard written before opt_group existed must load, with every option in
    its own group so group-marginal CE degenerates to plain CE."""
    data_dir = _build_synthetic_data(n_train=8, n_val=4, drop_keys=("opt_group",))
    ds = ShardDataset(data_dir, split="train")
    sample = ds[0]
    valid = sample["opt_mask"]
    g = sample["opt_group"]
    assert g[valid].unique().numel() == int(valid.sum()), "legacy fallback must be all-distinct"
    assert (g[~valid] == -1).all()
```

Add to `python/tests/test_shard_writer.py`:

```python
def test_opt_group_is_not_stored_as_fp16():
    from ptcg_il.shard_writer import _FP16_KEYS
    assert "opt_group" not in _FP16_KEYS, "opt_group is an index array, not a feature"
```

Extend `_build_synthetic_data` (`test_dataset.py:116`) with `drop_keys: tuple[str, ...] = ()`, popping those keys from the stacked dict just before `np.savez_compressed`.

- [ ] **Step 2: Run, confirm failure**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_dataset.py -k opt_group -v
```
Expected: FAIL — `opt_group` comes back as float32 (it falls through to the float branch), and the legacy test KeyErrors.

- [ ] **Step 3: Implement**

`python/ptcg_il/train/dataset.py`, add to `_INT_KEYS` (line 49):

```python
_INT_KEYS = frozenset({
    "tok_type", "tok_owner", "tok_zone",
    "opt_type", "opt_src_idx", "opt_tgt_idx", "opt_group", "sel_type", "sel_ctx",
    "action_idx", "minCount", "maxCount", "action_len", "stop_column",
    "log_len",
})
```

In `__getitem__`, next to the existing `stop_column` back-fill (line 393):

```python
        # Backward compat: shards written before opt_group existed.  Every
        # option becomes its own group, which makes group-marginal CE identical
        # to plain CE rather than silently merging unrelated options.
        if "opt_group" not in sample:
            g = torch.arange(sample["opt_mask"].shape[0], dtype=torch.long)
            sample["opt_group"] = torch.where(sample["opt_mask"], g, torch.full_like(g, -1))
```

- [ ] **Step 4: Run, confirm pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_dataset.py tests/test_shard_writer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_il/train/dataset.py python/tests/test_dataset.py python/tests/test_shard_writer.py
git commit -m "feat(dataset): carry opt_group into batches, with legacy-shard fallback"
```

---

### Task 3: Rebuild shards and confirm the measured collision share

**Files:**
- No source changes. Regenerates `python/data/shards/*.npz` and `python/data/.stamp-shards.json`.

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: shards carrying `opt_group`; the measured numbers Task 4's audit must reproduce.

- [ ] **Step 1: Rebuild**

The stamp already sees the `featurizer.py` edit, so no `--force` is needed; pass it anyway to be explicit.

```bash
cd python && uv run python -m ptcg_il.cli build-shards --data-dir data --force 2>&1 | tail -20
```
Expected: ~35 min, 9 shards rewritten. This does **not** invalidate existing checkpoints.

- [ ] **Step 2: Verify the new key and reproduce the headline number**

```bash
cd python && uv run python -c "
import numpy as np
d = np.load('data/shards/train-00000.npz')
g, m = d['opt_group'], d['opt_mask']
assert g.dtype == np.int64 and g.shape == m.shape
assert (g[~m] == -1).all()
dup = sum(int(((g[s][m[s]][:, None] == g[s][m[s]][None, :]).sum(1) > 1).sum()) for s in range(2000))
tot = int(m[:2000].sum())
print(f'dup {dup}/{tot} = {100*dup/tot:.2f}%')
"
```
Expected: within ~1 pt of **10.00 %**. A wildly different number means the group key does not match what the earlier measurement used — stop and diff before continuing.

- [ ] **Step 3: Commit**

Shards are not tracked in git; commit only if the stamp file is tracked.

```bash
git status --short python/data
```

---

### Task 4: Fix the collision audit to report the true share

`check_attachment_collision` keys on `(opt_src_idx, opt_card_feat)` over types 3-6 only, which is where the misleading 39.78 % came from. Re-key it on `opt_group` so the audit and the loss cannot disagree.

**Files:**
- Modify: `python/ptcg_il/qa.py:507-570` (`check_attachment_collision`)
- Test: `python/tests/test_qa.py`

**Interfaces:**
- Consumes: `opt_group` in shards (Task 3).
- Produces: `check_attachment_collision(...)` returns the existing keys `n_collision_options`, `collision_share`, `n_options_total` plus `n_indistinguishable_options`, `indistinguishable_share`, `collision_by_opt_type` (`dict[int, int]`). The legacy keys keep their legacy meaning so anything printing them does not silently change.

- [ ] **Step 1: Write the failing test**

Add to `python/tests/test_qa.py`:

```python
def test_attachment_collision_uses_opt_group(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    O = 64
    # Sample 0: slots 0 and 1 identical (group 0), slot 2 distinct (group 1).
    opt_group = np.full((1, O), -1, dtype=np.int64)
    opt_group[0, :3] = [0, 0, 1]
    opt_mask = np.zeros((1, O), dtype=bool)
    opt_mask[0, :3] = True
    opt_type = np.zeros((1, O), dtype=np.int64)
    opt_type[0, :3] = 3
    np.savez(shard_dir / "train-00000.npz",
             opt_group=opt_group, opt_mask=opt_mask, opt_type=opt_type)

    rep = check_attachment_collision(shard_dir, pd.DataFrame({"shard": ["train-00000.npz"]}))
    assert rep["n_options_total"] == 3
    assert rep["n_indistinguishable_options"] == 2
    assert rep["indistinguishable_share"] == pytest.approx(2 / 3)
    assert rep["collision_by_opt_type"] == {3: 2}


def test_attachment_collision_raises_on_zero_options(tmp_path):
    """An audit that examined nothing must not report a clean 0.0 share."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    with pytest.raises(ValueError, match="no options"):
        check_attachment_collision(shard_dir, pd.DataFrame({"shard": []}))
```

- [ ] **Step 2: Run, confirm failure**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_qa.py -k attachment_collision -v
```
Expected: `KeyError: 'n_indistinguishable_options'`, and no raise on the empty case.

- [ ] **Step 3: Implement**

In `python/ptcg_il/qa.py`, keep the existing `(src, card_feat)` loop exactly as-is for the legacy keys, and add the group-based pass alongside it. Inside the per-shard loop, after the existing `att_mask` work:

```python
        # True indistinguishability: two options are the same input to the
        # pointer only if *every* tensor it reads agrees.  The legacy
        # (src, card_feat) key above ignores opt_type/opt_tgt_idx/opt_scalar
        # and so overstates the share by ~4x (47.8% vs 10.0% on train-00000).
        if "opt_group" in data:
            groups = data["opt_group"]
            for s in range(groups.shape[0]):
                row_valid = opt_mask[s]
                if not row_valid.any():
                    continue
                gs = groups[s][row_valid]
                ts = opt_type[s][row_valid]
                counts = collections.Counter(gs.tolist())
                for g_id, t in zip(gs.tolist(), ts.tolist()):
                    n_options_seen += 1
                    if counts[g_id] > 1:
                        n_indistinguishable += 1
                        by_opt_type[int(t)] = by_opt_type.get(int(t), 0) + 1
```

Initialise `n_options_seen = 0`, `n_indistinguishable = 0`, `by_opt_type: dict[int, int] = {}` before the shard loop, `import collections` at module top, and after the loop:

```python
    if n_options_seen == 0:
        raise ValueError(
            "check_attachment_collision examined no options — shard_dir has no "
            "shards with opt_mask, so a 0.0 collision share would be a lie."
        )
    report["n_options_total"] = n_options_seen
    report["n_indistinguishable_options"] = n_indistinguishable
    report["indistinguishable_share"] = n_indistinguishable / n_options_seen
    report["collision_by_opt_type"] = by_opt_type
```

Note `np.load(sf, mmap_mode="r")` at line 532 is a no-op for `.npz` — every shard is fully decompressed into RAM (~528 MB each). Honour `max_samples` by breaking out of the shard loop, not only the row loop.

- [ ] **Step 4: Run, confirm pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_qa.py -k attachment_collision -v
```

- [ ] **Step 5: Run it on the real corpus and record the numbers**

```bash
cd python && uv run python -c "
import pandas as pd
from ptcg_il.qa import check_attachment_collision
meta = pd.read_parquet('data/meta.parquet')
r = check_attachment_collision('data/shards', meta, max_samples=20000)
print({k: r[k] for k in ('n_options_total','n_indistinguishable_options','indistinguishable_share','collision_by_opt_type')})
"
```
Expected: share ≈ 0.10, `collision_by_opt_type` dominated by key `3`. Paste the output into the commit message.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_il/qa.py python/tests/test_qa.py
git commit -m "fix(qa): report true option-indistinguishability share from opt_group"
```

---

### Task 5: Rewrite `check_reference_roundtrip` as a shard-internal check

The current gate needs `data/observations/`, which no code has ever written, and its `_build_simple_ref_from_obs` helper is unusable regardless: it reads `obs["yourIndex"]` (the real field is `obs["current"]["yourIndex"]`, `ref_map.py:58`), builds a flat hand→bench→active→discard counter while `opt_src_idx` is an A.1 **state-token row**, treats `active` as a dict when the engine ships a list (`ref_map.py:63-65`), and never covers the opponent's board. Comparing the shard's option tensors against the shard's own state tensors is both cheaper and a genuinely independent check — the option path and the state path are different code in `featurizer.py`.

**Files:**
- Modify: `python/ptcg_il/qa.py:190-343` (rewrite), `:346-387` (delete `_build_simple_ref_from_obs`), `:159-167` (caller)
- Test: `python/tests/test_qa.py:355-470` (delete the three obs-based tests, add new ones)

**Interfaces:**
- Consumes: shard keys `opt_src_idx`, `opt_card_feat`, `opt_mask`, `poke_card_feat`, `hand_card_feat`, `stadium_card_feat`.
- Produces: `check_reference_roundtrip(shard_dir: str | Path | None, max_samples: int | None = None) -> tuple[bool | None, dict]`. The `obs_dir` parameter is **gone**. `details` carries `n_compared`, `n_mismatch`, `n_src_out_of_range`, `n_skipped_pad`.

Row → state tensor mapping, from the A.1 layout in `ref_map.py:30-36` and the shard shapes (`poke_card_feat[S,12,212]`, `hand_card_feat[S,30,212]`, `stadium_card_feat[S,1,212]`):

| `opt_src_idx` | state tensor |
|---|---|
| `-1` | no source — skip |
| `1..12` | `poke_card_feat[s, row-1]` (1-6 my active+bench, 7-12 opp) |
| `13..42` | `hand_card_feat[s, row-13]` |
| `45` | `stadium_card_feat[s, 0]` |
| `0`, `43`, `44`, `>45` | CLS / summary tokens — never a legal source, count as out-of-range |

- [ ] **Step 1: Write the failing tests**

Replace `python/tests/test_qa.py:355-470` (the three `test_reference_roundtrip_*` tests and their fixtures) with:

```python
def _roundtrip_shard(tmp_path, *, src_row, card_row_value, state_slot_value):
    """One-sample shard whose only option points at a hand row."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(exist_ok=True)
    O, F = 64, 8
    opt_mask = np.zeros((1, O), dtype=bool); opt_mask[0, 0] = True
    opt_src = np.full((1, O), -1, dtype=np.int64); opt_src[0, 0] = src_row
    opt_card = np.zeros((1, O, F), dtype=np.float16); opt_card[0, 0, 0] = card_row_value
    hand = np.zeros((1, 30, F), dtype=np.float16); hand[0, src_row - 13, 0] = state_slot_value
    np.savez(shard_dir / "train-00000.npz",
             opt_mask=opt_mask, opt_src_idx=opt_src, opt_card_feat=opt_card,
             hand_card_feat=hand,
             poke_card_feat=np.zeros((1, 12, F), dtype=np.float16),
             stadium_card_feat=np.zeros((1, 1, F), dtype=np.float16))
    return shard_dir


def test_reference_roundtrip_passes_when_pointer_resolves(tmp_path):
    shard_dir = _roundtrip_shard(tmp_path, src_row=15, card_row_value=1.0, state_slot_value=1.0)
    passed, details = check_reference_roundtrip(shard_dir)
    assert passed is True
    assert details["n_compared"] == 1
    assert details["n_mismatch"] == 0


def test_reference_roundtrip_catches_off_by_one(tmp_path):
    """The pointer names hand row 15 but the card there is a different card."""
    shard_dir = _roundtrip_shard(tmp_path, src_row=15, card_row_value=1.0, state_slot_value=0.5)
    passed, details = check_reference_roundtrip(shard_dir)
    assert passed is False
    assert details["n_mismatch"] == 1


def test_reference_roundtrip_flags_impossible_source_rows(tmp_path):
    """Row 0 is CLS and row 43-44 are summary tokens — no option may point there."""
    shard_dir = _roundtrip_shard(tmp_path, src_row=15, card_row_value=1.0, state_slot_value=1.0)
    d = dict(np.load(shard_dir / "train-00000.npz"))
    d["opt_src_idx"][0, 0] = 43
    np.savez(shard_dir / "train-00000.npz", **d)
    passed, details = check_reference_roundtrip(shard_dir)
    assert passed is False
    assert details["n_src_out_of_range"] == 1


def test_reference_roundtrip_returns_none_when_nothing_compared(tmp_path):
    """Zero comparisons is 'skipped', never 'passed'."""
    shard_dir = _roundtrip_shard(tmp_path, src_row=15, card_row_value=0.0, state_slot_value=0.0)
    passed, details = check_reference_roundtrip(shard_dir)
    assert passed is None
    assert details["n_compared"] == 0
```

Also delete the now-dangling `obs_dir` fixtures in that block and drop `obs_dir` from the import line if it appears.

- [ ] **Step 2: Run, confirm failure**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_qa.py -k reference_roundtrip -v
```
Expected: `TypeError` — the old signature requires `obs_dir`.

- [ ] **Step 3: Implement**

Replace `python/ptcg_il/qa.py:190-387` (the function and the helper) with:

```python
def check_reference_roundtrip(
    shard_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> tuple[bool | None, dict[str, Any]]:
    """Verify ``opt_src_idx`` names the state row holding the option's own card.

    The featurizer writes an option's source as a state-token row (A.1) and its
    card features by dereferencing the same location.  Those are two separate
    code paths — ``ref_map.build_ref_map`` for the row, the state-token builders
    for the row's contents — so comparing them catches an off-by-one in either.

    This reads only the shard.  The previous implementation wanted raw
    observations in ``data/observations/``, which nothing has ever written, and
    resolved them with a helper whose index space did not match ``opt_src_idx``
    at all; it could not have passed or failed meaningfully.

    Options whose card row is PAD (all-zero) are skipped: ``card_id_at``
    deliberately leaves deck slots and face-down prizes at PAD, and RETREAT and
    ATTACK point at row 1 without ever setting a card.  Empty state slots are
    PAD for the same reason and are skipped too.

    Returns ``(passed, details)``; ``passed`` is None when nothing was compared.
    """
    details: dict[str, Any] = {
        "n_compared": 0, "n_mismatch": 0,
        "n_src_out_of_range": 0, "n_skipped_pad": 0,
    }
    if shard_dir is None:
        details["skipped"] = "no shard_dir"
        return None, details

    n_compared = n_mismatch = n_oob = n_pad = 0
    examples: list[dict] = []

    for sf in sorted(Path(shard_dir).glob("*.npz")):
        try:
            data = np.load(sf, allow_pickle=False)
        except OSError:
            continue
        needed = ("opt_src_idx", "opt_card_feat", "opt_mask",
                  "poke_card_feat", "hand_card_feat", "stadium_card_feat")
        if any(k not in data for k in needed):
            continue

        opt_src = data["opt_src_idx"]
        opt_card = data["opt_card_feat"]
        opt_mask = data["opt_mask"]
        poke = data["poke_card_feat"]
        hand = data["hand_card_feat"]
        stadium = data["stadium_card_feat"]

        for s in range(opt_src.shape[0]):
            for o in np.flatnonzero(opt_mask[s]):
                row = int(opt_src[s, o])
                if row < 0:
                    continue
                if 1 <= row <= 12:
                    state_row = poke[s, row - 1]
                elif 13 <= row <= 42:
                    state_row = hand[s, row - 13]
                elif row == 45:
                    state_row = stadium[s, 0]
                else:
                    n_oob += 1
                    continue

                card_row = opt_card[s, o]
                if not card_row.any() or not state_row.any():
                    n_pad += 1
                    continue

                n_compared += 1
                if not np.array_equal(card_row, state_row):
                    n_mismatch += 1
                    if len(examples) < 5:
                        examples.append(
                            {"shard": sf.name, "sample": int(s), "option": int(o), "src_row": row}
                        )

            if max_samples is not None and (s + 1) >= max_samples:
                break
        if max_samples is not None:
            break

    details.update(
        n_compared=n_compared, n_mismatch=n_mismatch,
        n_src_out_of_range=n_oob, n_skipped_pad=n_pad, examples=examples,
    )

    if n_oob > 0:
        logger.error(
            "REFERENCE ROUND-TRIP: %d option(s) point at a non-card state row "
            "(CLS/summary/out of range) — the pointer encoding is wrong.", n_oob,
        )
        return False, details

    if n_compared == 0:
        details["skipped"] = (
            "no comparable options — every option was PAD or sourceless, so the "
            "gate examined nothing and must not report a pass."
        )
        logger.info("Reference round-trip QA gate: %s", details["skipped"])
        return None, details

    if n_mismatch > 0:
        logger.error(
            "REFERENCE ROUND-TRIP FAIL: %d/%d options carry card features that do "
            "not match the state row opt_src_idx names.  Examples: %s",
            n_mismatch, n_compared, examples,
        )
        return False, details

    return True, details
```

Delete `_build_simple_ref_from_obs` entirely, and update the caller at `qa.py:159-167`:

```python
    # --- Reference round-trip (D.5.x) ---
    rt_pass, rt_details = check_reference_roundtrip(shard_dir, max_samples=max_samples)
    results["reference_roundtrip_pass"] = rt_pass
    results["reference_roundtrip_details"] = rt_details
```

- [ ] **Step 4: Run, confirm pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_qa.py -v
```

- [ ] **Step 5: Run the gate against the real shards — this is the point of the task**

```bash
cd python && uv run python -c "
from ptcg_il.qa import check_reference_roundtrip
p, d = check_reference_roundtrip('data/shards', max_samples=5000)
print(p, d)
"
```
Expected: `True` with `n_compared` in the tens of thousands.

**If it reports mismatches, stop and investigate — do not weaken the gate.** The two likely real bugs it can surface: a bench with more than 5 entries makes `build_ref_map` (`ref_map.py:69-70`, unbounded) write rows past 6 and alias onto the opponent's active at row 7; and `card_id_at` resolving a zone the state-token builder fills differently. Report the finding rather than adjusting the threshold.

- [ ] **Step 6: Mutation check**

Change `hand[s, row - 13]` to `hand[s, row - 12]`, re-run step 5, confirm a large `n_mismatch`, then revert.

- [ ] **Step 7: Commit**

```bash
git add python/ptcg_il/qa.py python/tests/test_qa.py
git commit -m "fix(qa): make the reference round-trip gate a shard-internal check that can actually run"
```

---

### Task 6: Group-marginal CE for single-select

**Files:**
- Modify: `python/ptcg_il/train/loop.py:77-122` (`masked_label_smoothed_ce`), new `target_group_mask` beside it, `:277-378` (`_compute_loss`)
- Test: `python/tests/test_train_loop.py`

**Interfaces:**
- Produces: `target_group_mask(opt_group: Tensor[B,O] int64, targets: Tensor[B] int64, mask: Tensor[B,O] bool) -> Tensor[B,O] bool`
- Produces: `masked_label_smoothed_ce(logits, targets, mask, label_smoothing=0.05, target_group=None)` — `target_group=None` keeps today's behaviour exactly, so every existing caller and test is unaffected.
- Produces: `_compute_loss(..., group_marginal: bool = True)`

- [ ] **Step 1: Write the failing tests**

Add to `python/tests/test_train_loop.py`:

```python
def test_group_marginal_ce_credits_the_whole_group():
    """Two identical options splitting the mass must cost log(2) less than one."""
    logits = torch.tensor([[0.0, 0.0, -20.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    targets = torch.tensor([0])
    plain = masked_label_smoothed_ce(logits, targets, mask, label_smoothing=0.0)
    group = masked_label_smoothed_ce(
        logits, targets, mask, label_smoothing=0.0,
        target_group=torch.tensor([[True, True, False]]),
    )
    assert torch.allclose(plain - group, torch.tensor([np.log(2.0)]), atol=1e-5)


def test_group_marginal_ce_matches_plain_ce_for_singleton_groups():
    logits = torch.randn(4, 6)
    mask = torch.ones(4, 6, dtype=torch.bool)
    targets = torch.tensor([0, 1, 2, 3])
    singleton = F.one_hot(targets, 6).bool()
    assert torch.allclose(
        masked_label_smoothed_ce(logits, targets, mask, target_group=singleton),
        masked_label_smoothed_ce(logits, targets, mask),
        atol=1e-6,
    )


def test_group_marginal_ce_gradient_is_finite_with_masked_options():
    logits = torch.randn(2, 8, requires_grad=True)
    mask = torch.tensor([[True] * 4 + [False] * 4, [True] * 3 + [False] * 5])
    targets = torch.tensor([0, 1])
    tg = torch.zeros(2, 8, dtype=torch.bool); tg[0, :2] = True; tg[1, 1] = True
    masked_label_smoothed_ce(logits, targets, mask, target_group=tg).sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_target_group_mask_excludes_padding_and_other_groups():
    opt_group = torch.tensor([[0, 0, 1, -1]])
    mask = torch.tensor([[True, True, True, False]])
    tg = target_group_mask(opt_group, torch.tensor([0]), mask)
    assert tg.tolist() == [[True, True, False, False]]
```

- [ ] **Step 2: Run, confirm failure**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_train_loop.py -k "group_marginal or target_group" -v
```
Expected: `TypeError: unexpected keyword argument 'target_group'`.

- [ ] **Step 3: Implement**

In `python/ptcg_il/train/loop.py`, change the signature at line 77 and the NLL at line 110:

```python
def masked_label_smoothed_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.05,
    target_group: torch.Tensor | None = None,
) -> torch.Tensor:
    """...existing docstring...

    target_group : bool Tensor[B, O] or None
        When given, the NLL is taken over the *summed* probability of the
        target's equivalence class instead of the target alone.  Options in a
        class are byte-identical inputs to the pointer head, so a plain CE asks
        the model to rank one above the others using information it does not
        have; on this corpus that is 13.9% of samples.  Each row must have at
        least one True (the target itself) or the logsumexp underflows to -inf.
        None reproduces the ungrouped behaviour exactly.
    """
    logits_inf = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(logits_inf, dim=-1)

    if target_group is None:
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    else:
        nll = -torch.logsumexp(
            log_probs.masked_fill(~target_group, float("-inf")), dim=-1
        )
    # ...smoothing block unchanged...
```

Add directly below it:

```python
def target_group_mask(
    opt_group: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """[B, O] bool marking options indistinguishable from each row's target.

    ``opt_group`` is -1 on padded slots, so those never join a group.  The
    target's own slot is always included because it is valid by construction.
    """
    g = opt_group.gather(1, targets.unsqueeze(1))          # [B, 1]
    return (opt_group == g) & mask & (opt_group >= 0)
```

In `_compute_loss` (line 277), add `group_marginal: bool = True` to the signature and build the mask for both single-select branches:

```python
        opt_group = batch.get("opt_group")
        use_groups = group_marginal and opt_group is not None

        if single.all() and bool(has_target.all()):
            tgt = batch["action_idx"][:, 0]
            ce = masked_label_smoothed_ce(
                logits, tgt, batch["opt_mask"],
                label_smoothing=label_smoothing,
                target_group=(
                    target_group_mask(opt_group, tgt, batch["opt_mask"])
                    if use_groups else None
                ),
            )
        else:
            ...
            if single_ok.any():
                s_idx = torch.where(single_ok)[0]
                tgt = batch["action_idx"][s_idx, 0]
                ce = ce.index_put(
                    (s_idx,),
                    masked_label_smoothed_ce(
                        logits[s_idx], tgt, batch["opt_mask"][s_idx],
                        label_smoothing=label_smoothing,
                        target_group=(
                            target_group_mask(opt_group[s_idx], tgt, batch["opt_mask"][s_idx])
                            if use_groups else None
                        ),
                    ),
                )
```

- [ ] **Step 4: Run, confirm pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_train_loop.py -v
```

- [ ] **Step 5: Mutation check**

Drop `& mask` from `target_group_mask`, confirm `test_target_group_mask_excludes_padding_and_other_groups` goes red, revert.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_il/train/loop.py python/tests/test_train_loop.py
git commit -m "feat(train): marginalise CE over indistinguishable option groups (single-select)"
```

---

### Task 7: Group-marginal CE for multi-select

**Files:**
- Modify: `python/ptcg_il/model/policy.py:351-450` (`multiselect_ce`), and its call in `python/ptcg_il/train/loop.py:358-361`
- Test: `python/tests/test_model_policy.py`

**Interfaces:**
- Consumes: `target_group_mask` from Task 6.
- Produces: `multiselect_ce(policy, x, label_smoothing=0.05, group_marginal: bool = True)`

The group must be intersected with `step_mask`, not `opt_mask`: once one member of a group has been picked, the remaining members are still interchangeable but the picked one is gone. The STOP column is `opt_type = 17`, unique, so it is always a singleton group and this step is a no-op for it.

- [ ] **Step 1: Write the failing test**

Add to `python/tests/test_model_policy.py`:

```python
def test_multiselect_ce_marginalises_over_duplicate_options():
    """A batch whose first two options are identical must cost less under
    group-marginal CE than under plain CE, and the gap must be positive."""
    policy = Policy(D=32, heads=4, layers=1, ff=64)
    batch = _make_synthetic_batch(B=2, max_count=3)
    plain = multiselect_ce(policy, batch, group_marginal=False)

    batch["opt_group"][:, 1] = batch["opt_group"][:, 0]   # options 0 and 1 identical
    grouped = multiselect_ce(policy, batch, group_marginal=True)

    assert torch.isfinite(grouped).all()
    assert (grouped <= plain + 1e-5).all()
    assert (grouped < plain).any(), "merging a group must reduce CE for at least one row"


def test_multiselect_ce_unchanged_when_every_group_is_a_singleton():
    policy = Policy(D=32, heads=4, layers=1, ff=64)
    batch = _make_synthetic_batch(B=2, max_count=3)   # all-distinct opt_group after Task 2
    assert torch.allclose(
        multiselect_ce(policy, batch, group_marginal=True),
        multiselect_ce(policy, batch, group_marginal=False),
        atol=1e-5,
    )
```

Both tests rely on `_make_synthetic_batch` emitting an all-distinct `opt_group` (Task 2). If it does not, the first test's `plain` and `grouped` are trivially equal and it fails — which is the correct signal, not a flake.

- [ ] **Step 2: Run, confirm failure**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_model_policy.py -k multiselect_ce -v
```
Expected: `TypeError: unexpected keyword argument 'group_marginal'`.

- [ ] **Step 3: Implement**

In `python/ptcg_il/model/policy.py`, add the parameter to `multiselect_ce` (line 351) and extend the lazy import at line 400:

```python
    from ptcg_il.train.loop import masked_label_smoothed_ce, target_group_mask

    opt_group = x.get("opt_group")
    use_groups = group_marginal and opt_group is not None
```

and inside the AR loop, replace the `ce = masked_label_smoothed_ce(...)` call at line 432:

```python
            tgt = target[valid_idx].clamp(min=0)
            # Intersect with step_mask, not opt_mask: a group member already
            # picked this step is no longer an alternative to the target.
            tg = (
                target_group_mask(opt_group[valid_idx], tgt, step_mask[valid_idx])
                if use_groups else None
            )
            ce = masked_label_smoothed_ce(
                logits[valid_idx], tgt, step_mask[valid_idx],
                label_smoothing=label_smoothing,
                target_group=tg,
            )
```

In `python/ptcg_il/train/loop.py:358-361`, forward the flag:

```python
                ce = ce.index_put(
                    (m_idx,),
                    multiselect_ce(
                        policy, sub,
                        label_smoothing=label_smoothing,
                        group_marginal=group_marginal,
                    ),
                )
```

- [ ] **Step 4: Run, confirm pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_model_policy.py tests/test_train_loop.py -v
```

- [ ] **Step 5: Mutation check**

Swap `step_mask[valid_idx]` for `x["opt_mask"][valid_idx]` in the `target_group_mask` call, then run a two-pick multi-select sample where both picks come from the same group and confirm the CE for the second pick drops below the correct value (the already-picked twin still counts). Revert.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_il/model/policy.py python/ptcg_il/train/loop.py python/tests/test_model_policy.py
git commit -m "feat(train): group-marginal CE in the multi-select AR loop"
```

---

### Task 8: `--no-group-marginal-ce` flag

**Files:**
- Modify: `python/ptcg_il/cli.py:203` area (argument), `:562` area (call site), `python/ptcg_il/train/loop.py:448-...` (`train` signature, `_compute_loss` call at `:748`)
- Test: `python/tests/test_train_loop.py`

**Interfaces:**
- Consumes: `_compute_loss(group_marginal=...)` from Task 6.
- Produces: `train(..., group_marginal: bool = True)`; CLI flag `--no-group-marginal-ce`.

- [ ] **Step 1: Write the failing test**

```python
def test_train_forwards_group_marginal_flag(monkeypatch):
    seen = {}
    import ptcg_il.train.loop as loop_mod
    real = loop_mod._compute_loss

    def spy(policy, batch, **kw):
        seen["group_marginal"] = kw.get("group_marginal")
        return real(policy, batch, **kw)

    monkeypatch.setattr(loop_mod, "_compute_loss", spy)
    data_dir = _build_tiny_data(num_samples=8)
    loop_mod.train(
        _tiny_policy(), data_dir=data_dir, save_dir=data_dir / "ckpt",
        batch_size=4, total_steps=2, val_every=10_000, run_val=False,
        group_marginal=False,
    )
    assert seen["group_marginal"] is False
```

Match the `train(...)` keyword names to the call at `cli.py:550-580` if any of the above differ.

- [ ] **Step 2: Run, confirm failure**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_train_loop.py -k group_marginal_flag -v
```

- [ ] **Step 3: Implement**

`python/ptcg_il/cli.py`, next to `--label-smoothing` (line 203):

```python
    train_hp.add_argument(
        "--no-group-marginal-ce", dest="group_marginal", action="store_false",
        default=True,
        help="Score each option label on its own instead of marginalising over "
             "byte-identical options (the pre-2026-08 behaviour).",
    )
```

and in the `train(...)` call at line 562, beside `label_smoothing=args.label_smoothing`:

```python
        group_marginal=args.group_marginal,
```

`python/ptcg_il/train/loop.py`: add `group_marginal: bool = True` to `train` (line 448, beside `label_smoothing` at 459), and forward it at the `_compute_loss` call (line 748):

```python
        loss, ce, value_mse_loss, belief_parts = _compute_loss(
            policy, batch_gpu,
            lambda_v=lambda_v,
            label_smoothing=label_smoothing,
            use_amp=use_amp,
            belief_weights=belief_weights,
            group_marginal=group_marginal,
        )
```

Add it to the run-config dict logged at line 705 (`"label_smoothing": label_smoothing,`) so W&B records which loss a run used:

```python
                "group_marginal_ce": group_marginal,
```

- [ ] **Step 4: Run, confirm pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_train_loop.py tests/test_cli_wandb.py -v
```

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_il/cli.py python/ptcg_il/train/loop.py python/tests/test_train_loop.py
git commit -m "feat(cli): --no-group-marginal-ce escape hatch, and log the choice to W&B"
```

---

### Task 9: Collision-adjusted eval metrics

Without these, the retrained model's top-1 is not comparable to the old one — group-marginal CE deliberately stops optimising the tie-break that plain top-1 still scores.

**Files:**
- Modify: `python/ptcg_il/train/eval.py:76-106` (accumulators), `:132-199` (single- and multi-select blocks), `:215-240` (metric assembly)
- Test: `python/tests/test_train_loop.py` (or wherever `offline_eval` is currently tested)

**Interfaces:**
- Consumes: `batch["opt_group"]`.
- Produces: new metrics `val/top1_micro_collision_adj`, `val/nontrivial_top1_collision_adj`, `val/collision_share`, `val/multiselect_exact_set_collision_adj`. Existing keys keep their existing meaning.

- [ ] **Step 1: Write the failing test**

```python
def test_offline_eval_reports_collision_adjusted_top1():
    data_dir = _build_tiny_data(num_samples=32)
    # Merge options 0 and 1 into one group in the val shard so at least one
    # target sits in a group of size 2 — otherwise collision_share is 0 and
    # every assertion below passes vacuously.
    for name in ("train-00000.npz", "val-00000.npz"):
        d = dict(np.load(data_dir / "shards" / name))
        d["opt_group"][:, 1] = d["opt_group"][:, 0]
        np.savez_compressed(data_dir / "shards" / name, **d)

    ds = ShardDataset(data_dir, split="val")
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    m = offline_eval(_tiny_policy(), loader, torch.device("cpu"), max_batches=2)

    assert "val/top1_micro_collision_adj" in m
    assert m["val/collision_share"] > 0.0, "fixture has no collisions; test is vacuous"
    assert m["val/top1_micro_collision_adj"] >= m["val/top1_micro"], (
        "crediting a group can only ever help"
    )
```

- [ ] **Step 2: Run, confirm failure**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_train_loop.py -k collision_adjusted -v
```

- [ ] **Step 3: Implement**

In `python/ptcg_il/train/eval.py`, add accumulators beside the ones at line 93:

```python
    collision_correct_top1 = 0
    collision_nontrivial_correct = 0
    collision_rows = 0
    collision_total_rows = 0
    multi_exact_collision = 0
```

Hoist the lookup to just after `batch_gpu` is assembled (line 116), **not** inside the single-select block — an all-multi-select batch would otherwise hit an unbound name in the multi block below:

```python
        opt_group = batch_gpu.get("opt_group")
```

In the single-select block, after `correct_arr` is built (line 165):

```python
            if opt_group is not None:
                g_single = opt_group[single_mask]                         # [N, O]
                tgt_g = g_single.gather(1, single_targets.unsqueeze(1))   # [N, 1]
                pred_g = g_single.gather(1, top1_pred.unsqueeze(1))       # [N, 1]
                adj = (pred_g == tgt_g).squeeze(1)
                collision_correct_top1 += int(adj.sum().item())
                collision_total_rows += int(single_mask.sum().item())
                # A row "has a collision" when the target's own group has >1 member.
                group_sizes = (g_single == tgt_g).sum(dim=-1)             # [N]
                collision_rows += int((group_sizes > 1).sum().item())
                if nt.any():
                    collision_nontrivial_correct += int(adj[nt].sum().item())
```

In the multi-select block, replace nothing; add after `if sorted(pred) == sorted(tgt)` (line 197):

```python
                if opt_group is not None:
                    g_row = opt_group[multi_mask][i]
                    to_g = lambda idx: int(g_row[idx].item()) if idx >= 0 else idx
                    if sorted(map(to_g, pred)) == sorted(map(to_g, tgt)):
                        multi_exact_collision += 1
```

In the metric assembly (after line 221):

```python
        if collision_total_rows > 0:
            metrics["val/top1_micro_collision_adj"] = (
                collision_correct_top1 / collision_total_rows
            )
            metrics["val/collision_share"] = collision_rows / collision_total_rows
        if nontrivial_samples > 0 and collision_total_rows > 0:
            metrics["val/nontrivial_top1_collision_adj"] = (
                collision_nontrivial_correct / nontrivial_samples
            )
        if total_multi_samples > 0:
            metrics["val/multiselect_exact_set_collision_adj"] = (
                multi_exact_collision / total_multi_samples
            )
```

- [ ] **Step 4: Run, confirm pass**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_train_loop.py -v
```

- [ ] **Step 5: Measure the gap on an existing checkpoint — before any retraining**

```bash
cd python && uv run python -m ptcg_il.cli train --eval-only --data-dir data \
    --out-dir checkpoints_a0 --resume checkpoints_a0/ckpt-best.pt --archetype-self 0 2>&1 | tail -20
```
Record `val/top1_micro`, `val/top1_micro_collision_adj`, `val/collision_share`. The gap is exactly how much of the current 0.627 micro top-1 was being lost to unanswerable ties. This is the number the whole plan exists to produce — put it in the commit message.

- [ ] **Step 6: Commit**

```bash
git add python/ptcg_il/train/eval.py python/tests/test_train_loop.py
git commit -m "feat(eval): collision-adjusted top-1 and collision share"
```

---

### Task 10: Retrain, re-gate, re-record baselines

**Files:**
- No source changes. Produces `python/checkpoints_a*/`, refreshed `python/data/il_baselines.json`.

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Full test suites, both directories**

```bash
cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q
cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/ -q
```
Expected: green. Do not proceed on a red suite.

- [ ] **Step 2: Short A/B on one archetype**

Derive the archetype id — never hardcode it (`archetypes.json` ids are cluster indices):

```bash
cd python && uv run python -m ptcg_il.cli archetypes --data-dir data --describe
```

Then, with `<N>` from that output, train 3 000 steps each way:

```bash
cd python && uv run python -m ptcg_il.cli train --data-dir data --out-dir /tmp/ab_group \
    --archetype-self <N> --total-steps 3000 --val-every 500 --no-wandb
cd python && uv run python -m ptcg_il.cli train --data-dir data --out-dir /tmp/ab_plain \
    --archetype-self <N> --total-steps 3000 --val-every 500 --no-group-marginal-ce --no-wandb
```

Compare `val/top1_micro_collision_adj` between the two. Group-marginal should be ≥ plain on the adjusted metric; raw `val/top1_micro` may be slightly lower, and that is expected, not a regression — it scores a tie-break the new loss deliberately stopped optimising. Report both numbers.

- [ ] **Step 3: Full retrain + re-record baselines**

```bash
cd /home/charles/Documents/Pokemon && ./scripts/run_pipeline.sh --skip-download --no-rl
```

`data/il_baselines.json` is SHA-1-pinned to the checkpoint that produced it, so stage 4c must re-record against the new checkpoints or the RL gate will refuse to run. Confirm afterwards:

```bash
cd python && uv run python -c "
import json; b = json.load(open('data/il_baselines.json'))
print({k: (v['ckpt_sha1'][:8], round(v['nontrivial_top1'], 4)) for k, v in b.items()})
"
```

- [ ] **Step 4: Commit**

```bash
git add python/data/il_baselines.json
git commit -m "chore: re-record IL baselines against group-marginal-CE checkpoints"
```

---

## Notes for the implementer

- **Do not** try to "fix" the type-3 collisions by feeding the option's `area`/`index` into the model. That is a separate, larger change (it needs `F_OPT` to grow, which *does* move `FEATURE_DIM_KEYS` and invalidates every checkpoint's `feat_dims` pin) and it is not obviously desirable: N identical cards in the same zone are genuinely interchangeable, so learning which index the expert happened to click is learning noise. The only collisions that represent real lost information are ones spanning *different* areas (deck vs discard), and the shards do not currently record `area`, so that has to be measured before it is worth building.
- CE values from runs after Task 6 are **not comparable** to earlier runs: a group of size k lowers the NLL by up to `log k`. `val/top1_micro_collision_adj` is the metric to track across the boundary.
- The 39.78 % figure in `data_fix_plan.md` came from `check_attachment_collision`'s `(src, card_feat)` key over types 3-6. Task 4 leaves that key computing the same thing under its old name so nothing that reads it changes meaning; the new `indistinguishable_share` is the honest number.
