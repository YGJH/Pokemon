# Ensemble Policy — Implementation Changes

**Spec:** `docs/superpowers/specs/2026-08-05-ensemble-design.md`
**Plan:** `docs/superpowers/plans/2026-08-05-ensemble-implementation.md`
**Date:** 2026-08-05
**Test results:** 62 passed (25 build_submission single-path byte-identical + 22 model_policy + 15 ensemble)

Build order: A (--seed) → C (AR refactor) → B (EnsemblePolicy) → E (measurement) → D (packaging)

---

## Summary

| | Files modified | Files created |
|---|---|---|
| **A. --seed** | `cli.py`, `train/loop.py`, `model/policy.py` | — |
| **C. AR refactor** | `model/policy.py`, `tests/test_model_policy.py` | — |
| **B. EnsemblePolicy** | — | `ensemble.py`, `tests/test_ensemble.py` |
| **E. Measurement** | `cli.py`, `baselines.py` | — |
| **D. Packaging** | `build_submission.py`, `build_submit.sh` | `run_ensemble_train.sh` |
| **Docs** | — | `plans/2026-08-05-ensemble-implementation.md` |

---

## A. `--seed` in the IL trainer

### `python/ptcg_il/cli.py`

- Added `--seed INT` argument (default 42) to the Training (C.10) argument group.
- `torch.manual_seed(args.seed)` called before `_build_policy()` in `cmd_train`.
- `seed=args.seed` passed through `cmd_train` → `train()`.
- `policy.config["seed"] = args.seed` set after `Policy` construction.

### `python/ptcg_il/train/loop.py`

- Added `seed: int = 42` keyword-only parameter to `train()` signature.
- `torch.manual_seed(seed)` called after device setup, before `ShardDataset`.
- Hardcoded `seed=42` replaced with `seed=seed` in `ShardDataset`.

### `python/ptcg_il/model/policy.py`

- Added `"seed": 42` entry to `Policy.config` dict.

---

## C. AR loop refactor

### `python/ptcg_il/model/policy.py` — `_select_multi_raw`

New optional parameters:

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `pointers` | `list[PointerHead] \| None` | `None` | N pointer heads for ensemble |
| `h_list` | `list[Tensor] \| None` | `None` | N encoded states, paired with `pointers` |

- When both are `None`: byte-identical to original single-pointer behavior.
- When provided: `pointers[i]` pairs with `h_list[i]`. Each member computes logits independently. Probabilities averaged across members (`softmax → mean → argmax`). Each member's msgru updated with its own option representation at the ensemble-chosen index.

### `python/ptcg_il/model/policy.py` — `select_multi`

Added dispatch at top of function:

```python
if hasattr(policy, 'select_multi'):
    return policy.select_multi(x, history_h)
```

### `python/tests/test_model_policy.py`

Added `test_single_member_ensemble_equivalent_to_single_pointer` — runs `_select_multi_raw` with `pointers=[p.pointer], h_list=[h]` and asserts element-wise identical output to the single-pointer path.

---

## B. EnsemblePolicy

### `python/ptcg_il/ensemble.py` (new file)

**`EnsemblePolicy(nn.Module)`**

- `members: nn.ModuleList[Policy]`.

**`from_checkpoints(paths, all_card_feat, device)` classmethod**

For each checkpoint:
1. Load via `torch.load`, auto-detect EMA shadow vs raw state dict.
2. Build `Policy` from checkpoint's own `config` via `policy_from_config`.
3. Load weights via `load_policy_state`.
4. Assert shared `deck` across members (fatal on mismatch). Warn on `vocab_sha1`/`archetypes_sha1` mismatch.
5. Heterogeneous architectures allowed (shapes never mix).

**`forward(x, history_h=None)` → `(logits, value, history_h)`**

- Per member: `forward` → `masked_fill(~opt_mask, -1e9)` → `softmax`.
- Stack → `mean(dim=0)` → `log()` → re-mask padding with `-1e9`.
- Value: `mean` across members.
- `history_h`: first member only.

**`select_multi(x, history_h=None)` → `Tensor[B, batch_max]`**

- Pre-encodes each member, delegates to `_select_multi_raw` with `pointers`/`h_list`.
- Members condition on the same joint decision (shared `picked_mask`).

**`belief_logits` / `forward_with_belief`**

- Delegate to first member only (ensemble is greedy-only).

### `python/tests/test_ensemble.py` (new file, 15 tests)

| Class | Tests |
|---|---|
| `TestEnsemblePolicyConstruction` | different seeds → different weights, same seed → identical params, `from_checkpoints` loads files, EMA shadow preferred, deck mismatch raises, missing state dict raises, empty paths raises |
| `TestEnsemblePolicyForward` | output shapes, value = mean of members, padding masked to `-1e9`, 1-member argmax matches Policy |
| `TestEnsemblePolicySelectMulti` | output shape `[B, maxC]`, all distinct picks, no out-of-range, 1-member matches single Policy |

---

## E. Measurement

### `python/ptcg_il/cli.py`

- Added `--ckpt action="append"` to train parser `Paths` group.
- `cmd_train`: when `--eval-only` with 2+ `--ckpt` values, dispatches to `_cmd_eval_only_ensemble()`.

**`_cmd_eval_only_ensemble(ckpt_paths, artifacts, args)`** (new)

- Builds `EnsemblePolicy` from checkpoints.
- Runs `offline_eval` on each member and the ensemble.
- Prints comparison table (top1_macro, top1_micro, top1_nontrivial, value_corr, value_std).
- Identifies best single member by `val/top1_nontrivial`.
- On `--record-baseline`: calls `record_ensemble_baseline()`.
- On `--live-eval`: runs head-to-head via `_run_ensemble_live_eval()`.

**`_run_ensemble_live_eval(ensemble, best_idx, artifacts, args)`** (new)

- The existing ``PolicyAgent`` class (from ``live_eval.py``) already accepts anything with
  ``forward`` + ``select_multi``, so ensemble works without a wrapper.
- Head-to-head games via ``LiveEvaluator``.
- Ship rule: Wilson lower bound must be > 50%.

### `python/ptcg_il/baselines.py`

- `record_ensemble_baseline(data_dir, archetype_self, ckpt_paths, ensemble_metrics)` — keyed as `"ens-N"`, SHA-pinned to all member checkpoints.
- `_combined_sha1(paths)` — SHA-1 of concatenated per-file SHA-1 digests.

---

## D. Packaging

### `scripts/build_submission.py`

| Change | Detail |
|---|---|
| `--ckpt` | Changed to `action="append"`. Single → byte-identical build. Multiple → ensemble. |
| `MODEL_FILES` | Added `"ensemble.py"`. |
| `REWRITE_RULES` | Added `(r"from ptcg_il\.ensemble import", r"from model.ensemble import")`. |
| `build_model_package` | Falls back to `ptcg_il/` when a file isn't in `ptcg_il/model/`. |
| `_build_member_weights` | New helper — writes `model_{i}.pt` for one ensemble member. |
| `main()` | Ensemble mode: implicit `--no-mcts`, per-member deck verification, writes `model_0.pt` … `model_{N-1}.pt` + `ensemble.json`. `--out` auto-suffix: `submission-greedy-ensN.tar.gz`. Single mode: byte-identical to original. |
| `MAIN_PY_TEMPLATE_GREEDY` | Conditional block at import time: detects `ensemble.json` → loads `EnsemblePolicy.from_checkpoints()`, else loads single `Policy` from `model.pt`. |

### `scripts/build_submit.sh`

- Added `--ensemble` flag: collects trailing args as paths, expands globs, aborts if no files match.
- Passes each matched file as `--ckpt` to `build_submission.py`.

### `scripts/run_ensemble_train.sh` (new file)

```bash
./scripts/run_ensemble_train.sh <ARCH> <N> [--extra-flags ...]
```

- Sequential, one GPU. Strips `a` prefix from arch argument.
- Each member gets `--seed S` for `S` in `0..N-1`.
- Extra flags forwarded to each `train` invocation.
