# Ensemble Policy Design

**Date**: 2026-08-05
**Status**: approved

## Overview

Train N independently-seeded specialist models on the same deck archetype, then
average their predictions at inference time. Members share vocab/archetypes/deck
but differ in init and batch order (`--seed`), producing decorrelated policies
whose averaged probabilities outperform any single member.

The build order is: A (seed) → C (AR refactor) → B (EnsemblePolicy) → E
(measure on existing checkpoints to prove plumbing is neutral) → D (package),
with packaging gated on the measurement result.

---

## A. `--seed` in the IL trainer

### Motivation

Two members trained identically on the same data produce identical weights.
`--seed` changes `torch.manual_seed` before model construction and the
`ShardDataset` shuffle seed, giving each member different initial parameters and
batch order. With `--dropout 0.0`, init and batch order are the only
decorrelation sources; dropout > 0 would add more but changes a tuned config.

### Changes

**`python/ptcg_il/cli.py`** — `--seed INT` (default 42) added to the train
argument group, under "Training (C.10)". Passed through `cmd_train` → `train()`.

**`python/ptcg_il/train/loop.py`** — line 570: `seed=42` replaced with
`seed=args.seed` (or the explicit parameter). `torch.manual_seed(seed)` called
before model construction (currently absent — init is deterministic, so the
first seeded point is the shuffle).

**`python/ptcg_il/model/policy.py`** — `Policy.config` gains `"seed": seed`.

### Invariants

The train/val/test split uses shard filename prefixes (`dataset.py:266`), not
the RNG — varying the seed changes batch order and init, never which sample
lands in which split. Member val/test numbers stay comparable.

### No bit-exact reproduction

GPU bit-exact reproducibility also needs
`torch.use_deterministic_algorithms(True)` + cuDNN flags, which cost throughput
and lack kernels for some ops. `--seed 0` gives a distinct, near-reproducible
member, not a bit-identical one.

### Training script

`scripts/run_ensemble_train.sh <ARCH> <N>`:

```bash
#!/bin/bash
# Usage: ./scripts/run_ensemble_train.sh a1 3
set -euo pipefail
ARCH="${1:?}"
N="${2:?}"
cd python
for S in $(seq 0 $((N - 1))); do
  uv run python -m ptcg_il.cli train --data-dir data --out-dir "checkpoints_${ARCH}_s${S}" \
      --archetype-self "${ARCH#a}" --seed "$S" \
      --d-model 256 --layers 6 --heads 8 --ff 1024 \
      --dropout 0.0 --batch-size 256 --total-steps 20000
done
```

Sequential, one GPU. Archetype id is derived by stripping the `a` prefix from
the arch argument (e.g. `a1` → `--archetype-self 1`).

---

## B. EnsemblePolicy

### Module

New file: `python/ptcg_il/ensemble.py`.

`EnsemblePolicy` is an `nn.Module` over an `nn.ModuleList` of member `Policy`
instances.

```python
class EnsemblePolicy(nn.Module):
    members: nn.ModuleList   # list[Policy]

    @classmethod
    def from_checkpoints(cls, paths: list[str], all_card_feat: Tensor,
                         device: str = "cpu") -> EnsemblePolicy:
        ...
```

### `from_checkpoints` factory

For each path:
1. Load checkpoint via `torch.load`. Auto-detect format: if
   `ckpt["ema_state_dict"]["shadow"]` exists, use EMA shadow; else fall back to
   `ckpt["model_state_dict"]`. Mirrors `build_model_weights`
   (`build_submission.py:847-855`), so the same code path works for measurement
   (training checkpoints) and packaging (stripped weights).
2. Build `Policy` from the checkpoint's own `config` via `policy_from_config`.
3. Load weights via `load_policy_state`.
4. Assert every member shares `deck.vocab_sha1`, `deck.archetypes_sha1`, and
   `deck.decklist`. Without this assertion, you can ensemble models trained
   against different vocabularies and get a bundle that runs fine and loses
   almost everything — the exact silent failure documented for single
   checkpoints in CLAUDE.md.

Heterogeneous members (different D/layers/heads/ff) are allowed — no
architecture assertion. The encode path is per-member, so shapes never mix.

### `forward(x, history_h=None)` → `(logits, value, history_h)`

Per member:
1. `logits_i = member(x, history_h)` → masked fill `~opt_mask` with `-1e9`
2. `probs_i = softmax(logits_i, dim=-1)`
3. Stack → `mean(dim=0)` → `log()`

Log rather than raw probability so `argmax` is unchanged: call sites that do
`.masked_fill(...).argmax()` work identically.

Value: `mean([member.value_head(...) for member in members])`. Only MCTS reads
value; in greedy-only ensemble mode it is unused.

`history_h`: returned from the first member only. Ensemble aggregation of
cross-turn GRU states is deferred — the current live-eval path does not thread
`history_h` between turns.

### `select_multi(x, history_h=None)` → `Tensor[B, batch_max]`

Ensemble AR loop:

1. **Pre-encode**: each member runs `_encode(x, history_h)` → `(h_i, history_h_i)`.
   Heterogeneous members produce different-shaped `h_i`, so a single `h` cannot
   serve all members.
2. **Per-member msgru state**: `msgru_h_list: list[Tensor]`, one per member.
3. **Per step**:
   - Each member: `logits_i, o_i = pointer_i(h_i, ..., msgru_h=msgru_h_i)`
   - Apply STOP mask and picked mask to each `logits_i`
   - `probs_i = softmax(logits_i)` → `mean(dim=0)` → `argmax`
   - STOP detection and picked-mask update use the ensemble-chosen index (same
     logic as today)
   - Each member's msgru advances with its own `o_i` at the ensemble-chosen
     index: `pointer_i.msgru(o_i[idx, chosen], msgru_h_i[idx])`
4. Members condition on the joint decision — this is what makes it an ensemble
   rather than N independent agents.

### Other methods

`belief_logits` / `forward_with_belief`: delegate to the first member only.
Belief is only consumed by MCTS; the ensemble is greedy-only. If MCTS+ensemble
is needed later, proper ensemble belief aggregation can be added then.

### Module dispatch in `select_multi`

`model/policy.py` `select_multi` gains a one-line dispatch at the top:

```python
def select_multi(policy, x, history_h=None):
    if hasattr(policy, 'select_multi'):
        return policy.select_multi(x, history_h)
    # ... existing single-Policy path ...
```

`select_multi` is a module-level function, not a `Policy` method, so
`hasattr(Policy(), 'select_multi')` returns `False`. `EnsemblePolicy` defines
`select_multi` as an instance method, so `hasattr(ensemble, 'select_multi')`
returns `True` and the ensemble AR loop runs. No type check needed — the
module-level function cannot be found through instance attribute lookup.

---

## C. AR loop refactor

### `_select_multi_raw` generalization

New optional parameters:

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `pointers` | `list[PointerHead] \| None` | `None` | N pointer heads for ensemble |
| `h_list` | `list[Tensor] \| None` | `None` | N encoded states, paired with `pointers` |

When both are `None`: behavior is byte-identical to today's `_select_multi_raw`.
The single `pointer` and single `h` parameters are used.

When provided: `pointers[i]` pairs with `h_list[i]`. `msgru_h` becomes
`msgru_h_list: list[Tensor]`, one per member. The per-step loop:
1. Each member produces `(logits_i, o_i)` via `pointer_i(h_list[i], ..., msgru_h=msgru_h_i)`
2. STOP mask and picked mask applied per-member (same column indices)
3. `probs = mean([softmax(logits_i) for i in range(N)])` → `argmax`
4. Each member's msgru updated with that member's `o_i` at the ensemble-chosen index

### Regression test

In `python/tests/test_model_policy.py`, a new test:
- Build a single `Policy`, pass `pointers=[policy.pointer]` and
  `h_list=[h]` to `_select_multi_raw`.
- Assert output is element-wise identical to the existing single-pointer path
  (fixed seed fixture with a real multi-select and a STOP column).
- This guarantees the refactor does not change single-model behavior.

---

## D. Packaging

### `build_submission.py` changes

**`--ckpt`** becomes `action="append"`.

**Single `--ckpt`**: today's output, byte-identical — single `data/model.pt`,
single `main.py`, no ensemble code shipped.

**Multiple `--ckpt`** (2+): ensemble mode. Implicit `--no-mcts` (ensemble is
greedy-only). If `--no-mcts` is passed explicitly, it is redundant but not an
error.

Per-member artifact check: `check_artifact_pairing` runs for each checkpoint.
`vocab_sha1` mismatch warns. `archetypes_sha1` mismatch warns (not read
because ensemble → implicit `--no-mcts`). `deck.decklist` mismatch across
members is fatal — a specialist trained on the wrong deck is a silent
catastrophe.

Weights: `data/model_0.pt` … `data/model_{N-1}.pt`. Each is a stripped EMA
weights file written by `build_model_weights`.

**`ensemble.json`**: `{"members": ["model_0.pt", "model_1.pt", ...]}`. Minimal —
`from_checkpoints` reads config from each .pt file, so the manifest does not
duplicate architecture metadata.

**`MODEL_FILES`** gains `"ensemble.py"`. `REWRITE_RULES` gains:
- `(r"from ptcg_il\.ensemble import", r"from model.ensemble import")`
- Import-rewrite verification test (`test_no_packaged_module_still_imports_ptcg_il`)
  must pass with the new module.

**`--out`** auto-suffix: when multiple `--ckpt` and `--out` is the default
(`submission.tar.gz`), the name becomes `submission-greedy-ensN.tar.gz` where N
is the member count. An explicit `--out` is honoured verbatim.

### Greedy main.py template changes

Conditional block at import time:

```python
_ENSEMBLE_MANIFEST = os.path.join(DATA_DIR, "ensemble.json")
if os.path.exists(_ENSEMBLE_MANIFEST):
    from model.ensemble import EnsemblePolicy
    import json
    _manifest = json.loads(open(_ENSEMBLE_MANIFEST).read())
    _member_paths = [os.path.join(DATA_DIR, m) for m in _manifest["members"]]
    _model = EnsemblePolicy.from_checkpoints(_member_paths, _all_card_feat, device=_device)
else:
    _ckpt = torch.load(os.path.join(DATA_DIR, "model.pt"), ...)
    _model = Policy(...)
    _model.load_state_dict(_ckpt["model_state_dict"], strict=False)
```

After this block, `_model` satisfies `forward` + `select_multi` regardless of
which branch ran. The rest of `agent()` — `_to_batch`, `_legal`, the
single/multi-select dispatch — is unchanged.

### `build_submit.sh` changes

New `--ensemble` flag accepting a glob or explicit paths:

```bash
./scripts/build_submit.sh a1 --ensemble "python/checkpoints_a1_s*/ckpt-best.pt"
./scripts/build_submit.sh a1 --ensemble \
    python/checkpoints_a1_s0/ckpt-best.pt \
    python/checkpoints_a1_s1/ckpt-best.pt
```

The script expands the glob and passes each matched file as a separate
`--ckpt` to `build_submission.py`. When `--ensemble` is passed and no files
match, the script aborts.

Single-checkpoint usage is unchanged:
```bash
./scripts/build_submit.sh a1                    # auto-select best ELO, single model
./scripts/build_submit.sh a1 path/to/ckpt.pt   # explicit, single model
```

---

## E. Measurement

### Offline eval: `--eval-only` with multiple `--ckpt`

When `--eval-only` receives 2+ `--ckpt` values, `_cmd_eval_only` enters
ensemble eval mode:

1. Build each member `Policy` from its checkpoint. All members use the same
   `--data-dir` (they share artifacts).
2. Build `EnsemblePolicy` via `from_checkpoints`.
3. **On val** (`--eval-split val`): run `offline_eval` on each member and on
   the ensemble. Print a table with top1_macro, top1_micro, top1_nontrivial,
   value_corr, value_std for each member row + the ensemble row.
4. **On test** (`--eval-split test`): run only the ensemble and the best
   single member (by val top1_nontrivial from step 3). Print only these two
   rows. This preserves test as the held-out set for the final ensemble-vs-best
   comparison.
5. `--record-baseline` with ensemble: records a baseline SHA-pinned to all
   member checkpoints under the ensemble's combined identity.

No new CLI flag — multiple `--ckpt` directly triggers this path.

### Live eval: ensemble vs. best single member

`PolicyAgent` already accepts anything with `forward` + `select_multi`, so
`EnsemblePolicy` drops in directly:
```python
ensemble_agent = make_agent_from_policy(ensemble, vocab, fixed_deck, device="cpu")
```

In `_run_live_eval`, when ensemble mode is active, add both the ensemble and
the best single member (identified from offline val ranking) as opponents.
Run head-to-head games.

### Ship rule

The ensemble ships only if its live win-rate against the best single member
has a Wilson lower bound above 50% (`wilson_interval` at `live_eval.py:82`).
Offline top-1 does not decide this — the project already has a 78%-offline /
48%-live gap.

### Scaling loop

After 3 members, train a 4th and 5th member and stop when the live win-rate
curve flattens (Wilson lower bound vs. best single member shows diminishing
returns). Cost: N× forward passes per decision, on CPU during live eval. At
~1-2 ms per forward pass with the current model size (~40 MB stripped), 3
members add ~2-4 ms per decision — within submission time limits.

### Cost

CPU live-eval throughput drops linearly with N. Each decision runs N forward
passes. For the 256/6/8/1024 config, a single forward pass takes ~1-2 ms on a
modern CPU core; 3 members ≈ 3-6 ms per decision. Kaggle's time limit is
generous enough that this is not a concern, but it should be measured before
committing to larger N.

---

## Build order

```
A (--seed) → C (AR refactor + 1-member regression test) → B (EnsemblePolicy)
→ E (measure on existing checkpoints_a1, prove plumbing is neutral)
→ train the 3 seeds → measure → D (package, only if E says ensemble wins)
```

A, C, and B are infrastructure. E validates the infrastructure on existing
checkpoints before committing GPU hours to training new seeds. D only runs if
the live Wilson gate passes.

---

## Files touched

| File | Change |
|------|--------|
| `python/ptcg_il/cli.py` | `--seed` flag, ensemble eval in `_cmd_eval_only` |
| `python/ptcg_il/train/loop.py` | `seed` parameter, `torch.manual_seed` call |
| `python/ptcg_il/model/policy.py` | `Policy.config["seed"]`, `_select_multi_raw` generalization, `select_multi` dispatch |
| `python/ptcg_il/ensemble.py` | **New file** — `EnsemblePolicy` |
| `python/tests/test_model_policy.py` | Regression test for 1-member `_select_multi_raw` equivalence |
| `python/tests/test_ensemble.py` | **New file** — `EnsemblePolicy.from_checkpoints`, forward, select_multi |
| `scripts/build_submission.py` | `--ckpt action="append"`, multi-ckpt packaging, `ensemble.json`, `ensemble.py` in MODEL_FILES |
| `scripts/build_submit.sh` | `--ensemble` flag |
| `scripts/run_ensemble_train.sh` | **New file** — training loop |
