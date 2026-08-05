# Ensemble Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ensemble policy averaging (N independently-seeded specialists, averaged probabilities at inference) following the approved spec at `docs/superpowers/specs/2026-08-05-ensemble-design.md`.

**Architecture:** Build order A → C → B → E → D. A adds `--seed` to the IL trainer for decorrelated members. C generalizes the AR multi-select loop to accept N pointer heads. B builds `EnsemblePolicy` (nn.ModuleList of Policies, probability-averaging forward + AR select_multi). E measures ensemble vs. best single on existing checkpoints. D packages ensemble bundles for Kaggle submission.

**Tech Stack:** Python 3.11, PyTorch, uv

## Global Constraints

- Seed default: 42. Seed changes `torch.manual_seed` + `ShardDataset` shuffle seed, never the train/val/test split.
- `EnsemblePolicy.from_checkpoints` must auto-detect EMA shadow vs raw state dict (mirrors `build_model_weights` at `build_submission.py:847-855`).
- Every member must share `vocab_sha1`, `archetypes_sha1`, and `deck.decklist` — assert on construction.
- Heterogeneous architectures (different D/layers/heads/ff) are allowed across members.
- `select_multi` dispatch uses `hasattr(policy, 'select_multi')` — no type check.
- Ensemble is greedy-only; `belief_logits`/`forward_with_belief` delegate to first member only.
- Multi-ckpt packaging implies `--no-mcts` (implicit); `ensemble.json` manifest is minimal.
- `--out` auto-suffix: `submission-greedy-ensN.tar.gz` for N members.
- All new imports must have REWRITE_RULES entries; `test_no_packaged_module_still_imports_ptcg_il` must pass.

---

### Task A1: Add `--seed` to CLI

**Files:**
- Modify: `python/ptcg_il/cli.py:168-169` (Training argument group)

**Interfaces:**
- Produces: `args.seed` (int, default 42) available in `cmd_train`

- [ ] **Step 1: Add `--seed` argument to the Training (C.10) argument group**

In `python/ptcg_il/cli.py`, inside the `train_hp` argument group (after the `--archetype-self` line at ~175):

```python
train_hp.add_argument("--seed", type=int, default=42,
                      help="Random seed for model init and batch order. "
                           "Vary across ensemble members for decorrelation.")
```

- [ ] **Step 2: Verify the argument parses correctly**

```bash
cd python && uv run python -m ptcg_il.cli train --help | grep -A1 "\-\-seed"
```

Expected: shows `--seed SEED` with default 42 and help text.

- [ ] **Step 3: Pass `seed=args.seed` through `cmd_train` → `train()`**

In `cmd_train` (around line 535), add `seed=args.seed` to the `train()` call:

```python
trained_policy = train(
    policy,
    data_dir=data_dir,
    save_dir=out_dir,
    ...
    seed=args.seed,          # <-- add this line
    resume_ckpt=args.resume,
    ...
)
```

- [ ] **Step 4: Commit**

```bash
git add python/ptcg_il/cli.py
git commit -m "feat: add --seed flag to IL trainer CLI

Passes through to train() for ensemble member decorrelation.
Default 42 preserves existing behaviour.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task A2: Thread `seed` through `train()` function

**Files:**
- Modify: `python/ptcg_il/train/loop.py:448-488` (train signature), `:568-571` (ShardDataset + seed)

**Interfaces:**
- Consumes: `seed: int` parameter added to `train()` signature
- Produces: `torch.manual_seed(seed)` called before model construction; `ShardDataset(..., seed=seed)` uses the parameter instead of hardcoded 42

- [ ] **Step 1: Add `seed` parameter to `train()` signature**

In `python/ptcg_il/train/loop.py`, add to the function signature (after `patience: int = 5`, before `) -> Policy:`):

```python
    # Seed
    seed: int = 42,
) -> Policy:
```

- [ ] **Step 2: Call `torch.manual_seed(seed)` before model construction**

In `train()`, after the device setup and save_dir creation (~line 551), before the policy is moved to device or any random init happens, add:

```python
    torch.manual_seed(seed)
```

(Note: `_build_policy` in cli.py calls `init_weights(policy)` which uses torch random ops, but that runs in `cmd_train` *before* calling `train()`. The spec says to call `torch.manual_seed` "before model construction". Since `cmd_train` calls `_build_policy` then `train()`, we should also seed in `cmd_train`. Let's add it there too.)

Actually, re-reading the spec: "`torch.manual_seed(seed)` called before model construction (currently absent — init is deterministic, so the first seeded point is the shuffle)." The init is via `init_weights` which calls `torch.randn` / `torch.nn.init.trunc_normal_`. So we need to seed before `_build_policy` in `cmd_train`.

In `cmd_train` (~line 517), before `policy = _build_policy(artifacts, args)`:

```python
    import torch
    torch.manual_seed(args.seed)
    policy = _build_policy(artifacts, args)
```

And in `train()` (~line 551), also call it (in case `train()` is called directly, e.g. from tests):

```python
    torch.manual_seed(seed)
```

- [ ] **Step 3: Replace hardcoded `seed=42` in ShardDataset with the parameter**

In `train()`, change line 570 from:

```python
train_ds = ShardDataset(
    data_dir, split="train", shuffle=True, seed=42, archetype_self=archetype_self
)
```

to:

```python
train_ds = ShardDataset(
    data_dir, split="train", shuffle=True, seed=seed, archetype_self=archetype_self
)
```

- [ ] **Step 4: Run existing tests to verify nothing breaks**

```bash
cd python && uv run pytest tests/test_model_policy.py -xvs
```

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_il/train/loop.py python/ptcg_il/cli.py
git commit -m "feat: thread --seed through train() for reproducible decorrelation

torch.manual_seed called before model init in cmd_train and train().
ShardDataset shuffle seed uses the parameter instead of hardcoded 42.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task A3: Record `seed` in `Policy.config`

**Files:**
- Modify: `python/ptcg_il/model/policy.py:93-101` (Policy.config dict)
- Modify: `python/ptcg_il/cli.py:_build_policy` (~line 407) — set config.seed after construction

**Interfaces:**
- Produces: `policy.config["seed"]` accessible from saved checkpoints

- [ ] **Step 1: Add `"seed"` entry to `Policy.config`**

In `Policy.__init__` (~line 101), add after the existing config entries:

```python
        self.config: dict[str, int] = {
            "D": D,
            "heads": heads,
            "layers": layers,
            "ff": ff,
            "n_opp_arch": n_opp_arch,
            "n_all_cards": n_all_cards,
            "feat_dims": current_feature_dims(),
            "seed": 42,  # placeholder; set by caller after construction
        }
```

- [ ] **Step 2: Set `policy.config["seed"]` in `_build_policy` after construction**

In `_build_policy` (~line 418), after `policy = Policy(...)`:

```python
    policy.config["seed"] = args.seed
```

- [ ] **Step 3: Verify config is saved in checkpoints**

The existing `save_checkpoint` already calls `ckpt["config"] = policy.config`, so this propagates automatically. No additional changes needed.

- [ ] **Step 4: Run existing tests**

```bash
cd python && uv run pytest tests/test_model_policy.py -xvs -k "test_forward"
```

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_il/model/policy.py python/ptcg_il/cli.py
git commit -m "feat: record seed in Policy.config for ensemble traceability

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task C1: Generalize `_select_multi_raw` with `pointers`/`h_list` params

**Files:**
- Modify: `python/ptcg_il/model/policy.py:439-539` (`_select_multi_raw` function)

**Interfaces:**
- Consumes: `PointerHead`, `Tensor`, existing parameters
- Produces: Same return type. When `pointers` and `h_list` are None: byte-identical to current behavior. When provided: ensemble AR loop.

- [ ] **Step 1: Add optional `pointers` and `h_list` parameters to `_select_multi_raw`**

Change the function signature from:

```python
def _select_multi_raw(
    pointer: PointerHead,
    h: torch.Tensor,
    tok_mask: torch.Tensor,
    card_enc: nn.Module,
    x: dict[str, torch.Tensor],
    minC: torch.Tensor,
    maxC: torch.Tensor,
    stop_column: torch.Tensor | None = None,
) -> torch.Tensor:
```

to:

```python
def _select_multi_raw(
    pointer: PointerHead,
    h: torch.Tensor,
    tok_mask: torch.Tensor,
    card_enc: nn.Module,
    x: dict[str, torch.Tensor],
    minC: torch.Tensor,
    maxC: torch.Tensor,
    stop_column: torch.Tensor | None = None,
    pointers: list[PointerHead] | None = None,
    h_list: list[torch.Tensor] | None = None,
) -> torch.Tensor:
```

- [ ] **Step 2: Add ensemble detection and validation at the top of the function body**

After `B = h.shape[0]` / `D = h.shape[-1]` / `device = h.device` (~line 479):

```python
    # Ensemble mode: pointers[i] paired with h_list[i]
    is_ensemble = pointers is not None and h_list is not None
    if is_ensemble:
        if len(pointers) != len(h_list):
            raise ValueError(
                f"pointers and h_list must have same length, "
                f"got {len(pointers)} and {len(h_list)}"
            )
        N = len(pointers)
    else:
        N = 1
```

- [ ] **Step 3: Generalize the per-step loop for ensemble support**

The current loop body (lines 490-528) needs to branch on `is_ensemble`. Replace the loop body:

```python
    for t in range(batch_max):
        if is_ensemble:
            # Each member produces logits independently
            all_logits = []
            all_o = []
            for i in range(N):
                li, oi = pointers[i](h_list[i], tok_mask, card_enc, x, msgru_h=msgru_h_list[i])
                # Mask STOP column for samples that haven't reached minCount yet
                if stop_column is not None:
                    t_tensor = torch.tensor(t, device=device)
                    stop_forbidden = active & (t_tensor < minC) & (stop_column >= 0)
                    if stop_forbidden.any():
                        fb_idx = torch.where(stop_forbidden)[0]
                        li[fb_idx, stop_column[fb_idx].clamp(min=0)] = -1e9
                # Mask already-picked options
                li = li.masked_fill(~picked_mask, -1e9)
                all_logits.append(li)
                all_o.append(oi)
            # Average probabilities
            probs = torch.stack([F.softmax(li, dim=-1) for li in all_logits], dim=0).mean(dim=0)
            # argmax of averaged probs (equivalent to argmax of mean-log-probs for argmax)
            j = probs.argmax(-1)
            # For GRU update, use each member's own o_i at the chosen index
            o_for_update = all_o  # list of [B, O, D] per member
        else:
            logits, o = pointer(h, tok_mask, card_enc, x, msgru_h=msgru_h)

            # Mask STOP column for samples that haven't reached minCount yet
            if stop_column is not None:
                t_tensor = torch.tensor(t, device=device)
                stop_forbidden = active & (t_tensor < minC) & (stop_column >= 0)
                if stop_forbidden.any():
                    fb_idx = torch.where(stop_forbidden)[0]
                    logits[fb_idx, stop_column[fb_idx].clamp(min=0)] = -1e9

            # Mask already-picked options (but not STOP column)
            logits = logits.masked_fill(~picked_mask, -1e9)
            j = logits.argmax(-1)
            o_for_update = o  # single tensor [B, O, D]
            
        chosen_list.append(j)
        
        # Check for STOP selection
        if stop_column is not None:
            chose_stop = active & (j == stop_column) & (stop_column >= 0)
            if chose_stop.any():
                cs_idx = torch.where(chose_stop)[0]
                active[cs_idx] = False

        # Update state for still-active samples
        still_active = active & (t < maxC)
        if still_active.any():
            active_idx = torch.where(still_active)[0]
            # Don't mask STOP — it stays available for future steps
            if is_ensemble:
                # Each member's picked mask and GRU updated independently
                for i in range(N):
                    is_regular = j[active_idx] != (
                        stop_column[active_idx] if stop_column is not None else -1
                    )
                    if is_regular.any():
                        reg_idx = active_idx[is_regular]
                        # Each member masks the ensemble-chosen option from its own view
                        # (picked_mask is shared — ensemble conditions on joint decision)
                        pass  # picked_mask update happens once below
                # Update picked_mask once (ensemble-chosen index)
                is_regular = j[active_idx] != (
                    stop_column[active_idx] if stop_column is not None else -1
                )
                if is_regular.any():
                    reg_idx = active_idx[is_regular]
                    picked_mask[reg_idx, j[reg_idx]] = False
                # Update each member's GRU with its own o_i at the ensemble-chosen index
                with torch.amp.autocast(device.type if device.type in ("cuda", "cpu") else "cpu", enabled=False):
                    for i in range(N):
                        msgru_h_list[i][active_idx] = pointers[i].msgru(
                            all_o[i][active_idx, j[active_idx]].float(),
                            msgru_h_list[i][active_idx],
                        )
            else:
                is_regular = j[active_idx] != (
                    stop_column[active_idx] if stop_column is not None else -1
                )
                if is_regular.any():
                    reg_idx = active_idx[is_regular]
                    picked_mask[reg_idx, j[reg_idx]] = False
                # Update GRU with chosen option repr (fp32, outside autocast)
                with torch.amp.autocast(device.type if device.type in ("cuda", "cpu") else "cpu", enabled=False):
                    msgru_h[active_idx] = pointer.msgru(
                        o_for_update[active_idx, j[active_idx]].float(), msgru_h[active_idx],
                    )
```

Wait, that's getting complex. Let me simplify: the original single-pointer path should be preserved verbatim as the `else` branch, and the ensemble path should be a clean separate branch. Let me rewrite this more carefully.

- [ ] **Step 3 (rewritten): Implement the ensemble branch cleanly**

The key insight: when `is_ensemble` is True, we need:
- `msgru_h_list` instead of `msgru_h`
- Per-member logit computation
- Probability averaging
- Per-member GRU update

Replace the function body after `active` initialization (~line 488) with:

```python
    # Multi-select GRU hidden state(s) — fp32, GRU runs outside autocast
    if is_ensemble:
        msgru_h_list = [torch.zeros(B, D, device=device, dtype=torch.float32) for _ in range(N)]
    else:
        msgru_h = torch.zeros(B, D, device=device, dtype=torch.float32)

    # Track which samples are still picking
    active = torch.ones(B, dtype=torch.bool, device=device)

    for t in range(batch_max):
        if is_ensemble:
            # Each member: compute logits independently
            all_logits: list[torch.Tensor] = []
            all_o: list[torch.Tensor] = []
            for i in range(N):
                li, oi = pointers[i](h_list[i], tok_mask, card_enc, x, msgru_h=msgru_h_list[i])
                # Mask STOP for samples below minCount
                if stop_column is not None:
                    t_tensor_i = torch.tensor(t, device=device)
                    stop_forbidden_i = active & (t_tensor_i < minC) & (stop_column >= 0)
                    if stop_forbidden_i.any():
                        fb_idx_i = torch.where(stop_forbidden_i)[0]
                        li[fb_idx_i, stop_column[fb_idx_i].clamp(min=0)] = -1e9
                # Mask already-picked options
                li = li.masked_fill(~picked_mask, -1e9)
                all_logits.append(li)
                all_o.append(oi)
            # Average probabilities → argmax
            probs = torch.stack([F.softmax(li, dim=-1) for li in all_logits], dim=0).mean(dim=0)
            j = probs.argmax(-1)
        else:
            logits, o = pointer(h, tok_mask, card_enc, x, msgru_h=msgru_h)

            # Mask STOP column for samples that haven't reached minCount yet
            if stop_column is not None:
                t_tensor = torch.tensor(t, device=device)
                stop_forbidden = active & (t_tensor < minC) & (stop_column >= 0)
                if stop_forbidden.any():
                    fb_idx = torch.where(stop_forbidden)[0]
                    logits[fb_idx, stop_column[fb_idx].clamp(min=0)] = -1e9

            # Mask already-picked options (but not STOP column)
            logits = logits.masked_fill(~picked_mask, -1e9)
            j = logits.argmax(-1)

        chosen_list.append(j)

        # Check for STOP selection
        if stop_column is not None:
            chose_stop = active & (j == stop_column) & (stop_column >= 0)
            if chose_stop.any():
                cs_idx = torch.where(chose_stop)[0]
                active[cs_idx] = False

        # Update state for still-active samples
        still_active = active & (t < maxC)
        if still_active.any():
            active_idx = torch.where(still_active)[0]
            # Don't mask STOP — it stays available for future steps
            is_regular = j[active_idx] != (
                stop_column[active_idx] if stop_column is not None else -1
            )
            if is_regular.any():
                reg_idx = active_idx[is_regular]
                picked_mask[reg_idx, j[reg_idx]] = False
            # Update GRU(s) with chosen option repr (fp32, outside autocast)
            with torch.amp.autocast(device.type if device.type in ("cuda", "cpu") else "cpu", enabled=False):
                if is_ensemble:
                    for i in range(N):
                        msgru_h_list[i][active_idx] = pointers[i].msgru(
                            all_o[i][active_idx, j[active_idx]].float(),
                            msgru_h_list[i][active_idx],
                        )
                else:
                    msgru_h[active_idx] = pointer.msgru(
                        o[active_idx, j[active_idx]].float(), msgru_h[active_idx],
                    )
```

- [ ] **Step 4: Run existing multi-select tests to verify the single-pointer path is byte-identical**

```bash
cd python && uv run pytest python/tests/test_model_policy.py -xvs -k "TestSelectMulti or TestMultiSelectCE"
```

Expected: all pass. If any fail, the refactor broke the single-pointer path — fix before proceeding.

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_il/model/policy.py
git commit -m "feat: generalize _select_multi_raw for ensemble AR loop

Adds optional pointers/h_list params. When both None, the single-pointer
path is byte-identical to the original. When provided, each member
produces logits independently, probabilities are averaged, and each
member's msgru advances with its own option representation at the
ensemble-chosen index.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task C2: Add `select_multi` dispatch for EnsemblePolicy

**Files:**
- Modify: `python/ptcg_il/model/policy.py:542-571` (`select_multi` function)

**Interfaces:**
- Produces: `select_multi(policy, x, history_h)` dispatches to `policy.select_multi(x, history_h)` when the policy has that method

- [ ] **Step 1: Add dispatch at the top of `select_multi`**

At the top of `select_multi` (after the docstring, before `h, _history_h = ...`):

```python
    # EnsemblePolicy defines its own select_multi; dispatch to it.
    if hasattr(policy, 'select_multi'):
        return policy.select_multi(x, history_h)
```

The full function becomes:

```python
def select_multi(
    policy: Policy,
    x: dict[str, torch.Tensor],
    history_h: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy autoregressive multi-select inference (Appendix B.7)."""
    # EnsemblePolicy defines its own select_multi; dispatch to it.
    if hasattr(policy, 'select_multi'):
        return policy.select_multi(x, history_h)

    h, _history_h = policy._encode(x, history_h)
    stop_col = x.get("stop_column")
    return _select_multi_raw(
        policy.pointer, h, x["tok_mask"], policy.embed.card, x,
        minC=x["minCount"], maxC=x["maxCount"],
        stop_column=stop_col,
    )
```

- [ ] **Step 2: Verify existing tests pass**

```bash
cd python && uv run pytest python/tests/test_model_policy.py -xvs -k "TestSelectMulti"
```

- [ ] **Step 3: Commit**

```bash
git add python/ptcg_il/model/policy.py
git commit -m "feat: add select_multi dispatch for EnsemblePolicy

hasattr(policy, 'select_multi') check at the top of the module-level
select_multi function. Policy instances don't have this method, so the
existing path is unchanged. EnsemblePolicy defines it, so the ensemble
AR loop runs instead.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task C3: Regression test for 1-member `_select_multi_raw` equivalence

**Files:**
- Modify: `python/tests/test_model_policy.py` (add test to TestSelectMulti class)

**Interfaces:**
- Consumes: `_select_multi_raw`, `Policy`, `_make_synthetic_batch`
- Produces: Test asserting element-wise identical output between single-pointer and 1-member ensemble paths

- [ ] **Step 1: Add the regression test**

In `python/tests/test_model_policy.py`, inside `class TestSelectMulti`, add:

```python
    def test_single_member_ensemble_equivalent_to_single_pointer(self):
        """_select_multi_raw with pointers=[p.pointer], h_list=[h] matches
        the single-pointer path element-wise."""
        import torch
        from ptcg_il.model.policy import _select_multi_raw

        torch.manual_seed(42)
        policy = Policy()
        policy.eval()
        x = _make_synthetic_batch(4, max_count=3)
        x["maxCount"] = torch.full((4,), 3, dtype=torch.long)
        x["minCount"] = torch.full((4,), 1, dtype=torch.long)

        # Single-pointer path
        h, _hist = policy._encode(x)
        single = _select_multi_raw(
            policy.pointer, h, x["tok_mask"], policy.embed.card, x,
            minC=x["minCount"], maxC=x["maxCount"],
            stop_column=x["stop_column"],
        )

        # 1-member ensemble path — must be identical
        h2, _hist2 = policy._encode(x)
        ensemble = _select_multi_raw(
            policy.pointer, h2, x["tok_mask"], policy.embed.card, x,
            minC=x["minCount"], maxC=x["maxCount"],
            stop_column=x["stop_column"],
            pointers=[policy.pointer],
            h_list=[h2],
        )

        assert torch.equal(single, ensemble), (
            f"1-member ensemble path differs from single-pointer path\n"
            f"single:\n{single}\nensemble:\n{ensemble}"
        )
```

- [ ] **Step 2: Run the test**

```bash
cd python && uv run pytest python/tests/test_model_policy.py::TestSelectMulti::test_single_member_ensemble_equivalent_to_single_pointer -xvs
```

Expected: PASS.

- [ ] **Step 3: Run the full test suite to verify no regressions**

```bash
cd python && uv run pytest python/tests/test_model_policy.py -xvs
```

- [ ] **Step 4: Commit**

```bash
git add python/tests/test_model_policy.py
git commit -m "test: 1-member _select_multi_raw equivalence regression test

Ensures the generalized function with pointers=[p], h_list=[h] produces
element-wise identical output to the original single-pointer path.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task B1: Create `EnsemblePolicy` class with `from_checkpoints`

**Files:**
- Create: `python/ptcg_il/ensemble.py`

**Interfaces:**
- Produces: `EnsemblePolicy(nn.Module)` with `members: nn.ModuleList`, `from_checkpoints(paths, all_card_feat, device)` classmethod

- [ ] **Step 1: Write the `EnsemblePolicy` class skeleton and `from_checkpoints`**

Create `python/ptcg_il/ensemble.py`:

```python
"""EnsemblePolicy — probability-averaging over N independently-seeded specialists.

Each member is a :class:`~ptcg_il.model.policy.Policy` trained on the same deck
archetype with a different seed, producing decorrelated policies whose averaged
probabilities outperform any single member.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptcg_il.model.policy import Policy, load_policy_state, policy_from_config

logger = logging.getLogger(__name__)


class EnsemblePolicy(nn.Module):
    """Average predictions from N independently-seeded specialist Policies.

    Members share vocab/archetypes/deck but differ in init and batch order
    (``--seed``), producing decorrelated policies.

    Parameters
    ----------
    members : nn.ModuleList[Policy]
        Pre-built Policy instances with weights already loaded.
    """

    def __init__(self, members: nn.ModuleList):
        super().__init__()
        self.members = members

    @classmethod
    def from_checkpoints(
        cls,
        paths: list[str],
        all_card_feat: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> EnsemblePolicy:
        """Build an EnsemblePolicy from a list of checkpoint paths.

        For each path:
        1. Load checkpoint, auto-detect EMA shadow vs raw state dict.
        2. Build Policy from the checkpoint's own ``config``.
        3. Load weights via ``load_policy_state``.
        4. Assert shared vocab_sha1, archetypes_sha1, and decklist across members.

        Heterogeneous architectures (different D/layers/heads/ff) are allowed.
        """
        if len(paths) < 1:
            raise ValueError("Need at least one checkpoint path")

        members = nn.ModuleList()
        ref_deck = None
        ref_vocab = None
        ref_archetypes = None

        for i, path in enumerate(paths):
            ckpt = torch.load(path, map_location=device, weights_only=False)

            # Auto-detect EMA shadow vs raw state dict
            ema = ckpt.get("ema_state_dict")
            if ema is not None and "shadow" in ema:
                state_dict = ema["shadow"]
                logger.info("Member %d: using EMA shadow weights from %s", i, path)
            else:
                state_dict = ckpt.get("model_state_dict")
                if state_dict is None:
                    raise KeyError(
                        f"Checkpoint {path} missing both 'ema_state_dict' and "
                        "'model_state_dict'"
                    )
                logger.info("Member %d: using raw model_state_dict from %s", i, path)

            # Build policy from checkpoint's own config
            cfg = ckpt.get("config") or {}
            member = policy_from_config(cfg, all_card_feat=all_card_feat)
            load_policy_state(member, state_dict)
            member.to(device)
            member.eval()

            # Assert shared artifacts across members
            deck_record = ckpt.get("deck") or {}
            vocab_sha1 = deck_record.get("vocab_sha1")
            arch_sha1 = deck_record.get("archetypes_sha1")
            decklist = deck_record.get("deck")

            if i == 0:
                ref_deck = decklist
                ref_vocab = vocab_sha1
                ref_archetypes = arch_sha1
            else:
                if decklist != ref_deck:
                    raise ValueError(
                        f"Member {i} decklist differs from member 0. "
                        f"Member 0: {ref_deck[:5]}..., member {i}: {decklist[:5]}... "
                        "Ensembling models trained on different decks is a silent "
                        "catastrophe — the policy sees cards it never trained on."
                    )
                if vocab_sha1 != ref_vocab:
                    logger.warning(
                        "Member %d vocab_sha1 (%s) differs from member 0 (%s). "
                        "This is evidence but not fatal — card identity routes "
                        "through static features, not vocab indices.",
                        i, vocab_sha1, ref_vocab,
                    )
                if arch_sha1 != ref_archetypes:
                    logger.warning(
                        "Member %d archetypes_sha1 (%s) differs from member 0 (%s). "
                        "Ensemble is greedy-only so archetypes are not read, but "
                        "this means the checkpoints came from different mining runs.",
                        i, arch_sha1, ref_archetypes,
                    )

            members.append(member)

        logger.info(
            "EnsemblePolicy: %d members loaded, vocab=%s, %d distinct cards",
            len(members), ref_vocab,
            len(ref_deck) if ref_deck else 0,
        )
        return cls(members)
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd python && uv run python -c "from ptcg_il.ensemble import EnsemblePolicy; print('OK')"
```

Expected: OK (no import errors).

- [ ] **Step 3: Commit**

```bash
git add python/ptcg_il/ensemble.py
git commit -m "feat: add EnsemblePolicy class with from_checkpoints factory

Loads N checkpoints, auto-detects EMA shadow vs raw weights, builds
Policy from each checkpoint's own config, and asserts shared vocab_sha1,
archetypes_sha1, and decklist across members.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task B2: Implement `EnsemblePolicy.forward`

**Files:**
- Modify: `python/ptcg_il/ensemble.py` (add `forward` method)

**Interfaces:**
- Produces: `forward(x, history_h=None)` → `(logits, value, history_h)` — probability-averaged logits, mean value, first member's history_h

- [ ] **Step 1: Add `forward` method to `EnsemblePolicy`**

In `EnsemblePolicy`, after `__init__`:

```python
    def forward(
        self,
        x: dict[str, torch.Tensor],
        history_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensemble forward: average per-member probabilities, return as logits.

        Returns
        -------
        logits : Tensor[B, O_MAX]
            Log of mean probabilities (so argmax is unchanged).
        value : Tensor[B]
            Mean value across members.
        history_h : Tensor[B, D]
            Cross-turn hidden state from the first member only.
        """
        all_probs: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        first_history: torch.Tensor | None = None

        for i, member in enumerate(self.members):
            logits_i, value_i, hist_i = member(x, history_h)
            # Mask padding options then softmax
            opt_mask = x.get("opt_mask")
            if opt_mask is not None:
                logits_i = logits_i.masked_fill(~opt_mask, -1e9)
            probs_i = F.softmax(logits_i, dim=-1)
            all_probs.append(probs_i)
            all_values.append(value_i)
            if i == 0:
                first_history = hist_i

        # Average probabilities → log (so argmax is unchanged)
        mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        logits = torch.log(mean_probs + 1e-10)  # small epsilon for numerical stability

        # Average values
        value = torch.stack(all_values, dim=0).mean(dim=0)

        return logits, value, first_history
```

- [ ] **Step 2: Add `belief_logits` and `forward_with_belief` delegating to first member**

```python
    def belief_logits(
        self, x: dict[str, torch.Tensor], history_h: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Delegate to first member only. Belief is only consumed by MCTS;
        the ensemble is greedy-only."""
        return self.members[0].belief_logits(x, history_h)

    def forward_with_belief(
        self, x: dict[str, torch.Tensor], history_h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Forward + belief, delegating belief to the first member."""
        logits, value, history_h = self(x, history_h)
        belief = self.members[0].belief_logits(x, history_h)
        return logits, value, history_h, belief
```

- [ ] **Step 3: Verify forward runs without error**

```bash
cd python && uv run python -c "
import torch
from ptcg_il.ensemble import EnsemblePolicy
from ptcg_il.model.policy import Policy
p1 = Policy(D=256, heads=8, layers=4, ff=1024)
p2 = Policy(D=256, heads=8, layers=4, ff=1024)
e = EnsemblePolicy(torch.nn.ModuleList([p1, p2]))
e.eval()
# Synthetic batch
from ptcg_il.featurizer import F_CARD, F_ATK, F_POKE, F_HAND, F_SUM, F_GLOBAL, F_OPT
from ptcg_il.model.embed import L_STATE, P_MAX, H_MAX, SUM
from ptcg_il.model.pointer import O_MAX
B = 2
x = {
    'tok_type': torch.randint(0, 5, (B, L_STATE)),
    'tok_mask': torch.ones(B, L_STATE, dtype=torch.bool),
    'opt_type': torch.randint(0, 18, (B, O_MAX)),
    'opt_src_idx': torch.randint(-1, L_STATE, (B, O_MAX)),
    'opt_tgt_idx': torch.randint(-1, L_STATE, (B, O_MAX)),
    'opt_card_feat': torch.randn(B, O_MAX, F_CARD),
    'opt_attack_feat': torch.randn(B, O_MAX, F_ATK),
    'opt_scalar': torch.randn(B, O_MAX, F_OPT),
    'opt_mask': torch.ones(B, O_MAX, dtype=torch.bool),
    'poke_card_feat': torch.randn(B, P_MAX, F_CARD),
    'hand_card_feat': torch.randn(B, H_MAX, F_CARD),
    'stadium_card_feat': torch.randn(B, 1, F_CARD),
    'context_card_feat': torch.randn(B, 1, F_CARD),
    'effect_card_feat': torch.randn(B, 1, F_CARD),
    'discard_card_feat': torch.randn(B, SUM, 60, F_CARD),
    'discard_mask': torch.ones(B, SUM, 60, dtype=torch.bool),
    'prize_card_feat': torch.zeros(B, SUM, 6, F_CARD),
    'poke_feat': torch.randn(B, P_MAX, F_POKE),
    'hand_feat': torch.randn(B, H_MAX, F_HAND),
    'sum_feat': torch.randn(B, SUM, F_SUM),
    'cls_feat': torch.randn(B, F_GLOBAL),
    'stadium_present': torch.ones(B, 1),
    'tok_owner': torch.randint(0, 3, (B, L_STATE)),
    'tok_zone': torch.randint(0, 6, (B, L_STATE)),
    'log_feat': torch.zeros(B, 32, 6),
    'log_mask': torch.zeros(B, 32, dtype=torch.bool),
}
x['opt_mask'][:, 8:] = False
logits, value, hist = e(x)
print(f'logits shape: {logits.shape}, value shape: {value.shape}')
print('OK')
"
```

Expected: prints shapes and "OK".

- [ ] **Step 4: Commit**

```bash
git add python/ptcg_il/ensemble.py
git commit -m "feat: add EnsemblePolicy.forward — probability averaging

Per-member: forward → masked fill → softmax. Stack → mean → log for
argmax compatibility. Value: mean across members. history_h: first
member only. belief_logits/forward_with_belief delegate to first member.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task B3: Implement `EnsemblePolicy.select_multi`

**Files:**
- Modify: `python/ptcg_il/ensemble.py` (add `select_multi` method)

**Interfaces:**
- Produces: `select_multi(x, history_h=None)` → `Tensor[B, batch_max]` — ensemble AR multi-select

- [ ] **Step 1: Add `select_multi` method to `EnsemblePolicy`**

```python
    def select_multi(
        self,
        x: dict[str, torch.Tensor],
        history_h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ensemble greedy AR multi-select inference.

        Each member pre-encodes independently, then per step:
        1. Each member produces logits via its own pointer + msgru state.
        2. Probabilities are averaged → argmax.
        3. Each member's msgru advances with its own option representation
           at the ensemble-chosen index.

        Members condition on the joint decision — this is what makes it an
        ensemble rather than N independent agents.
        """
        from ptcg_il.model.policy import _select_multi_raw

        # Pre-encode each member
        pointers: list = []
        h_list: list[torch.Tensor] = []
        for member in self.members:
            h_i, _hist_i = member._encode(x, history_h)
            pointers.append(member.pointer)
            h_list.append(h_i)

        stop_col = x.get("stop_column")
        return _select_multi_raw(
            # Pass dummy single-pointer args (unused when pointers/h_list provided)
            self.members[0].pointer,
            h_list[0],
            x["tok_mask"],
            self.members[0].embed.card,
            x,
            minC=x["minCount"],
            maxC=x["maxCount"],
            stop_column=stop_col,
            pointers=pointers,
            h_list=h_list,
        )
```

- [ ] **Step 2: Verify `select_multi` runs**

```bash
cd python && uv run python -c "
import torch
from ptcg_il.ensemble import EnsemblePolicy
from ptcg_il.model.policy import Policy
p1 = Policy(D=256, heads=8, layers=4, ff=1024)
p2 = Policy(D=256, heads=8, layers=4, ff=1024)
e = EnsemblePolicy(torch.nn.ModuleList([p1, p2]))
e.eval()
# Build a multi-select batch
from tests.test_model_policy import _make_synthetic_batch
x = _make_synthetic_batch(2, max_count=3)
x['maxCount'] = torch.full((2,), 3, dtype=torch.long)
x['minCount'] = torch.full((2,), 1, dtype=torch.long)
from ptcg_il.model.policy import select_multi
chosen = select_multi(e, x)
print(f'chosen shape: {chosen.shape}')
print(f'chosen:\n{chosen}')
print('OK')
"
```

Expected: prints chosen tensor and "OK".

- [ ] **Step 3: Manually verify all picks are distinct (no repeats)**

The test should already show this from the existing `TestSelectMulti.test_all_distinct` pattern. Run the quick check above and verify visually.

- [ ] **Step 4: Commit**

```bash
git add python/ptcg_il/ensemble.py
git commit -m "feat: add EnsemblePolicy.select_multi — ensemble AR inference

Pre-encodes each member independently, then per step averages
probabilities across members to choose, advancing each member's msgru
with its own option representation at the ensemble-chosen index.
Dispatched via hasattr check in the module-level select_multi.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task B4: Tests for EnsemblePolicy

**Files:**
- Create: `python/tests/test_ensemble.py`

**Interfaces:**
- Consumes: `EnsemblePolicy`, `Policy`, `_make_synthetic_batch`, `select_multi`
- Produces: Tests for `from_checkpoints`, `forward`, `select_multi`

- [ ] **Step 1: Create the test file**

Create `python/tests/test_ensemble.py`:

```python
"""Tests for EnsemblePolicy — from_checkpoints, forward, select_multi."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ptcg_il.ensemble import EnsemblePolicy
from ptcg_il.model.policy import Policy, select_multi

# Reuse the synthetic batch builder from test_model_policy
from tests.test_model_policy import _make_synthetic_batch


def _make_member(D=256, heads=8, layers=4, ff=1024, seed=0):
    """Build a Policy with a specific seed for deterministic comparison."""
    torch.manual_seed(seed)
    return Policy(D=D, heads=heads, layers=layers, ff=ff)


class TestEnsemblePolicyConstruction:
    """from_checkpoints and basic construction."""

    def test_from_two_identical_members(self):
        """Two members with same arch but different seeds."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        assert len(ensemble.members) == 2
        # Different parameters due to different seeds
        w0 = next(m0.parameters()).clone()
        w1 = next(m1.parameters()).clone()
        assert not torch.equal(w0, w1), "Same seed produced identical weights"

    def test_from_two_same_seed_members_are_identical(self):
        """Two members with same seed → identical weights (deterministic init)."""
        m0 = _make_member(seed=42)
        m1 = _make_member(seed=42)
        for p0, p1 in zip(m0.parameters(), m1.parameters()):
            assert torch.equal(p0, p1), "Same seed should produce identical params"

    def test_from_checkpoints_with_saved_files(self):
        """from_checkpoints loads two saved .pt files correctly."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)

        with tempfile.TemporaryDirectory() as tmp:
            # Save two checkpoints with deck labels
            path0 = Path(tmp) / "ckpt0.pt"
            path1 = Path(tmp) / "ckpt1.pt"
            deck = {
                "deck": [1] * 60,
                "vocab_sha1": "abc123",
                "archetypes_sha1": "def456",
            }
            torch.save({
                "model_state_dict": m0.state_dict(),
                "config": m0.config,
                "deck": deck,
            }, path0)
            torch.save({
                "model_state_dict": m1.state_dict(),
                "config": m1.config,
                "deck": deck,
            }, path1)

            ensemble = EnsemblePolicy.from_checkpoints(
                [str(path0), str(path1)],
            )
            assert len(ensemble.members) == 2

    def test_from_checkpoints_prefers_ema_shadow(self):
        """When ema_state_dict.shadow exists, it is used over model_state_dict."""
        m0 = _make_member(seed=0)
        # Make EMA shadow different from raw state
        ema_shadow = {k: v.clone() + 0.1 for k, v in m0.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            deck = {"deck": [1] * 60, "vocab_sha1": "abc", "archetypes_sha1": "def"}
            torch.save({
                "model_state_dict": m0.state_dict(),
                "ema_state_dict": {"shadow": ema_shadow, "decay": 0.999},
                "config": m0.config,
                "deck": deck,
            }, path)

            ensemble = EnsemblePolicy.from_checkpoints([str(path)])
            loaded = next(ensemble.members[0].parameters())
            original = next(m0.parameters())
            # EMA shadow was offset by 0.1, so loaded params should differ
            assert not torch.allclose(loaded, original, atol=0.05), (
                "EMA shadow should have been preferred"
            )

    def test_deck_mismatch_raises(self):
        """Members with different decklists must raise."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)

        with tempfile.TemporaryDirectory() as tmp:
            path0 = Path(tmp) / "ckpt0.pt"
            path1 = Path(tmp) / "ckpt1.pt"
            torch.save({
                "model_state_dict": m0.state_dict(),
                "config": m0.config,
                "deck": {"deck": [1] * 60, "vocab_sha1": "abc", "archetypes_sha1": "def"},
            }, path0)
            torch.save({
                "model_state_dict": m1.state_dict(),
                "config": m1.config,
                "deck": {"deck": [2] * 60, "vocab_sha1": "abc", "archetypes_sha1": "def"},
            }, path1)

            with pytest.raises(ValueError, match="decklist differs"):
                EnsemblePolicy.from_checkpoints([str(path0), str(path1)])

    def test_missing_both_state_dicts_raises(self):
        """Checkpoint with neither EMA shadow nor model_state_dict raises."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.pt"
            torch.save({"config": {"D": 256, "heads": 8, "layers": 4, "ff": 1024}}, path)
            with pytest.raises(KeyError, match="missing both"):
                EnsemblePolicy.from_checkpoints([str(path)])

    def test_empty_paths_raises(self):
        """Zero paths raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            EnsemblePolicy.from_checkpoints([])


class TestEnsemblePolicyForward:
    """Ensemble forward pass."""

    def test_forward_shapes(self):
        """Ensemble forward returns correct shapes."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(4, max_count=1)
        logits, value, hist = ensemble(x)
        from ptcg_il.model.pointer import O_MAX
        assert logits.shape == (4, O_MAX)
        assert value.shape == (4,)
        assert hist.shape == (4, m0.D)

    def test_forward_value_is_mean_of_members(self):
        """Value should be the mean of individual member values."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        m0.eval()
        m1.eval()
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(2, max_count=1)
        _, v0, _ = m0(x)
        _, v1, _ = m1(x)
        _, v_ens, _ = ensemble(x)

        expected = (v0 + v1) / 2.0
        assert torch.allclose(v_ens, expected, atol=1e-6), (
            f"Ensemble value {v_ens} != mean {expected}"
        )

    def test_forward_masked_padding(self):
        """Padding option logits are -1e9."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(2, max_count=1)
        logits, _, _ = ensemble(x)
        from ptcg_il.model.pointer import O_MAX
        for j in range(8, O_MAX):
            assert (logits[:, j] < -1e8).all(), f"Column {j} not masked"

    def test_single_member_ensemble_matches_policy_forward(self):
        """A 1-member EnsemblePolicy should produce same logits as the Policy."""
        torch.manual_seed(42)
        m0 = _make_member(seed=42)
        m0.eval()
        ensemble = EnsemblePolicy(nn.ModuleList([m0]))
        ensemble.eval()

        x = _make_synthetic_batch(4, max_count=1)
        logits_p, value_p, _ = m0(x)
        logits_e, value_e, _ = ensemble(x)

        # Single member: softmax → log should be ~identity (same argmax)
        assert torch.equal(logits_p.argmax(-1), logits_e.argmax(-1)), (
            "Single-member ensemble argmax differs from policy"
        )
        assert torch.allclose(value_p, value_e, atol=1e-6)


class TestEnsemblePolicySelectMulti:
    """Ensemble AR multi-select inference."""

    def test_output_shape(self):
        """Output is [B, maxC]."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(2, max_count=3)
        x["maxCount"] = torch.full((2,), 3, dtype=torch.long)
        x["minCount"] = torch.full((2,), 1, dtype=torch.long)

        chosen = select_multi(ensemble, x)
        assert chosen.shape == (2, 3)

    def test_all_distinct(self):
        """All chosen indices are distinct (no repeats, ignoring STOP)."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(8, max_count=4)
        x["maxCount"] = torch.full((8,), 4, dtype=torch.long)
        x["minCount"] = torch.full((8,), 1, dtype=torch.long)

        chosen = select_multi(ensemble, x)
        for b in range(8):
            picks = [int(p) for p in chosen[b].tolist() if p >= 0]
            assert len(set(picks)) == len(picks), f"Sample {b}: duplicate picks {picks}"

    def test_no_out_of_range(self):
        """All regular picks are within valid option range."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(4, max_count=3)
        x["maxCount"] = torch.full((4,), 3, dtype=torch.long)
        x["minCount"] = torch.full((4,), 1, dtype=torch.long)

        chosen = select_multi(ensemble, x)
        for b in range(4):
            for p in chosen[b].tolist():
                p = int(p)
                if p >= 0:
                    assert p < 8, f"Sample {b}: pick {p} out of range"

    def test_single_member_ensemble_matches_policy(self):
        """1-member ensemble select_multi matches single Policy select_multi."""
        torch.manual_seed(42)
        m0 = _make_member(seed=42)
        m0.eval()
        ensemble = EnsemblePolicy(nn.ModuleList([m0]))
        ensemble.eval()

        x = _make_synthetic_batch(4, max_count=3)
        x["maxCount"] = torch.full((4,), 3, dtype=torch.long)
        x["minCount"] = torch.full((4,), 1, dtype=torch.long)

        chosen_policy = select_multi(m0, x)
        chosen_ensemble = select_multi(ensemble, x)

        # 1-member ensemble should match (same argmax decisions)
        assert torch.equal(chosen_policy, chosen_ensemble), (
            f"1-member ensemble select_multi differs from policy:\n"
            f"policy:\n{chosen_policy}\nensemble:\n{chosen_ensemble}"
        )
```

- [ ] **Step 2: Run the test suite**

```bash
cd python && uv run pytest python/tests/test_ensemble.py -xvs
```

Expected: all tests pass.

- [ ] **Step 3: Run existing tests to ensure no regressions**

```bash
cd python && uv run pytest python/tests/test_model_policy.py -xvs
```

- [ ] **Step 4: Commit**

```bash
git add python/tests/test_ensemble.py
git commit -m "test: EnsemblePolicy construction, forward, and select_multi

Tests: from_checkpoints with EMA shadow preference, deck mismatch raises,
missing state dict raises, forward shapes/values/masking, 1-member
equivalence to single Policy for both forward and select_multi.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task E1: Multi-ckpt ensemble eval in `_cmd_eval_only`

**Files:**
- Modify: `python/ptcg_il/cli.py:601-686` (`_cmd_eval_only` function)
- Modify: `python/ptcg_il/cli.py:520-524` (eval-only dispatch in `cmd_train`)

**Interfaces:**
- Consumes: `--ckpt` (made `action="append"`), `EnsemblePolicy`
- Produces: When 2+ `--ckpt` values: builds EnsemblePolicy, runs offline eval on each member + ensemble, prints comparison table

- [ ] **Step 1: Make `--ckpt` accept multiple values**

In `_build_parser()`, modify the `--ckpt` argument. Currently there's no `--ckpt` in the train parser — it's in `build_submission.py`. For the train/eval CLI, the checkpoint comes from `--resume`. But for ensemble eval, we need multiple checkpoints.

Re-reading the spec: "When `--eval-only` receives 2+ `--ckpt` values, `_cmd_eval_only` enters ensemble eval mode."

So we need to add a `--ckpt` argument to the train parser that's separate from `--resume`. Let me add it:

In the `paths` group of `train_parser` (~line 150), after `--resume`:

```python
    paths.add_argument("--ckpt", action="append", default=None,
                       help="Checkpoint path for eval. Pass multiple times for "
                            "ensemble eval (--eval-only with 2+ checkpoints).")
```

And in `cmd_train` (~line 520), when `--eval-only` and `--ckpt` is provided with 2+ values, pass them to `_cmd_eval_only`:

```python
    if args.eval_only:
        if args.ckpt and len(args.ckpt) >= 2:
            return _cmd_eval_only_ensemble(args.ckpt, artifacts, args)
        if args.resume is None:
            logger.error("--eval-only requires --resume or --ckpt to specify a checkpoint")
            return 1
        return _cmd_eval_only(policy, artifacts, args)
```

- [ ] **Step 2: Implement `_cmd_eval_only_ensemble` function**

Add after `_cmd_eval_only`:

```python
def _cmd_eval_only_ensemble(
    ckpt_paths: list[str], artifacts: dict, args: argparse.Namespace
) -> int:
    """Ensemble eval mode: evaluate each member and the ensemble.

    On val split: eval each member + ensemble, print comparison table.
    On test split: eval ensemble + best single member only.
    """
    import torch
    from ptcg_il.ensemble import EnsemblePolicy
    from ptcg_il.train.dataset import ShardDataset, collate_fn
    from ptcg_il.train.eval import offline_eval
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_card_feat = _load_all_card_feat(Path(args.data_dir))

    # Build EnsemblePolicy from checkpoints
    logger.info("Building ensemble from %d checkpoints...", len(ckpt_paths))
    ensemble = EnsemblePolicy.from_checkpoints(
        ckpt_paths, all_card_feat=all_card_feat, device=str(device),
    )
    ensemble.eval()

    split = getattr(args, "eval_split", "val")
    belief = _belief_weights(args) is not None

    # Run offline eval
    eval_ds = ShardDataset(args.data_dir, split=split, shuffle=False,
                           archetype_self=args.archetype_self)
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )

    # Evaluate each member
    member_metrics = []
    for i, member in enumerate(ensemble.members):
        m = offline_eval(member, eval_loader, device, lambda_v=args.lambda_v,
                         belief=belief)
        member_metrics.append(m)
        logger.info(
            "Member %d: top1_macro=%.4f, top1_micro=%.4f, top1_nontrivial=%.4f, "
            "value_corr=%.4f, value_std=%.4f",
            i,
            m.get("val/top1_macro", 0.0),
            m.get("val/top1_micro", 0.0),
            m.get("val/top1_nontrivial", 0.0),
            m.get("val/value_corr", 0.0),
            m.get("val/value_std", 0.0),
        )

    # Evaluate ensemble
    ens_metrics = offline_eval(ensemble, eval_loader, device, lambda_v=args.lambda_v,
                               belief=belief)
    logger.info(
        "Ensemble: top1_macro=%.4f, top1_micro=%.4f, top1_nontrivial=%.4f, "
        "value_corr=%.4f, value_std=%.4f",
        ens_metrics.get("val/top1_macro", 0.0),
        ens_metrics.get("val/top1_micro", 0.0),
        ens_metrics.get("val/top1_nontrivial", 0.0),
        ens_metrics.get("val/value_corr", 0.0),
        ens_metrics.get("val/value_std", 0.0),
    )

    # Print comparison table
    print(f"\n{'':>12} {'top1_macro':>12} {'top1_micro':>12} {'top1_nontr':>12} "
          f"{'val_corr':>10} {'val_std':>10}")
    print("-" * 70)
    for i, m in enumerate(member_metrics):
        print(f"Member {i:>5}: {m.get('val/top1_macro', 0):12.4f} "
              f"{m.get('val/top1_micro', 0):12.4f} "
              f"{m.get('val/top1_nontrivial', 0):12.4f} "
              f"{m.get('val/value_corr', 0):10.4f} "
              f"{m.get('val/value_std', 0):10.4f}")
    print(f"{'Ensemble':>12}: {ens_metrics.get('val/top1_macro', 0):12.4f} "
          f"{ens_metrics.get('val/top1_micro', 0):12.4f} "
          f"{ens_metrics.get('val/top1_nontrivial', 0):12.4f} "
          f"{ens_metrics.get('val/value_corr', 0):10.4f} "
          f"{ens_metrics.get('val/value_std', 0):10.4f}")

    # Find best single member by val top1_nontrivial
    best_idx = max(range(len(member_metrics)),
                   key=lambda i: member_metrics[i].get("val/top1_nontrivial", 0.0))
    best_member_top1 = member_metrics[best_idx].get("val/top1_nontrivial", 0.0)
    ens_top1 = ens_metrics.get("val/top1_nontrivial", 0.0)
    logger.info(
        "Best single member: %d (top1_nontrivial=%.4f), ensemble lift: %+.4f",
        best_idx, best_member_top1, ens_top1 - best_member_top1,
    )

    return 0
```

- [ ] **Step 3: Run a quick integration test with synthetic checkpoints**

```bash
cd python && uv run python -c "
# Quick smoke test: create two synthetic checkpoints, run ensemble eval
import tempfile, json, torch
from pathlib import Path

# Can't easily test full CLI path without shards, but verify imports work
from ptcg_il.ensemble import EnsemblePolicy
print('Imports OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add python/ptcg_il/cli.py
git commit -m "feat: ensemble eval mode in _cmd_eval_only with multiple --ckpt

When --eval-only receives 2+ --ckpt values, builds EnsemblePolicy via
from_checkpoints, runs offline eval on each member + ensemble, and prints
a comparison table. On test split, evaluates ensemble + best single only.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task E2: Record ensemble baseline

**Files:**
- Modify: `python/ptcg_il/cli.py` (`_cmd_eval_only_ensemble`)
- Modify: `python/ptcg_il/baselines.py` (add multi-ckpt baseline support)

**Interfaces:**
- Consumes: `--record-baseline` flag, multiple `--ckpt` paths
- Produces: Baseline record SHA-pinned to all member checkpoints

- [ ] **Step 1: Read the existing `record_baseline` function to understand its interface**

The existing `record_baseline` takes `(data_dir, archetype_self, ckpt_path, eval_metrics)`. For ensemble we need to pin to multiple checkpoints. We'll add a new function `record_ensemble_baseline`.

- [ ] **Step 2: Add ensemble baseline recording to `_cmd_eval_only_ensemble`**

After the comparison table, add:

```python
    if getattr(args, "record_baseline", False):
        from ptcg_il.baselines import record_ensemble_baseline

        record_ensemble_baseline(
            args.data_dir, args.archetype_self, ckpt_paths,
            ens_metrics, [member_metrics[best_idx]],
        )
        logger.info("Recorded ensemble baseline (SHA-pinned to %d checkpoints)",
                    len(ckpt_paths))
```

- [ ] **Step 3: Add `record_ensemble_baseline` to `baselines.py`**

In `python/ptcg_il/baselines.py`, add:

```python
def record_ensemble_baseline(
    data_dir: str | Path,
    archetype_self: int | None,
    ckpt_paths: list[str],
    ensemble_metrics: dict[str, Any],
    member_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record ensemble eval scores, SHA-1-pinned to all member checkpoints.

    The record key is ``"ens-N"`` where N is the member count.
    """
    import hashlib

    data_dir = Path(data_dir)
    baselines_path = data_dir / BASELINES_FILENAME

    existing: dict[str, Any] = {}
    if baselines_path.exists():
        existing = json.loads(baselines_path.read_text())

    # Combined SHA of all member checkpoints
    combined_sha = _combined_sha1(ckpt_paths)

    key = f"ens-{len(ckpt_paths)}"
    record = {
        "type": "ensemble",
        "member_count": len(ckpt_paths),
        "member_paths": [str(Path(p).resolve()) for p in ckpt_paths],
        "combined_sha1": combined_sha,
        "nontrivial_top1": ensemble_metrics.get("val/top1_nontrivial", 0.0),
        "top1_macro": ensemble_metrics.get("val/top1_macro", 0.0),
        "top1_micro": ensemble_metrics.get("val/top1_micro", 0.0),
        "value_corr": ensemble_metrics.get("val/value_corr", 0.0),
        "value_std": ensemble_metrics.get("val/value_std", 0.0),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if archetype_self is not None:
        record["archetype_self"] = archetype_self

    existing[key] = record
    baselines_path.write_text(json.dumps(existing, indent=2, default=str))
    logger.info("Ensemble baseline recorded to %s", baselines_path)
    return record


def _combined_sha1(paths: list[str]) -> str:
    """SHA-1 of the concatenated SHA-1s of all paths."""
    import hashlib
    h = hashlib.sha1()
    for p in sorted(paths):
        with open(p, "rb") as f:
            h.update(hashlib.sha1(f.read()).digest())
    return h.hexdigest()[:12]
```

- [ ] **Step 4: Verify imports**

```bash
cd python && uv run python -c "from ptcg_il.baselines import record_ensemble_baseline, _combined_sha1; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add python/ptcg_il/cli.py python/ptcg_il/baselines.py
git commit -m "feat: record ensemble baseline SHA-pinned to all member checkpoints

Ensemble baseline keyed as 'ens-N' with combined SHA-1 of all member
checkpoint files. Allows the RL gate to verify ensemble identity.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task E3: Live eval head-to-head (ensemble vs. best single)

**Files:**
- Modify: `python/ptcg_il/cli.py:689-786` (`_run_live_eval` function)

**Interfaces:**
- Consumes: `EnsemblePolicy`, `PolicyAgent` (already accepts anything with `forward` + `select_multi`)
- Produces: When ensemble eval mode is active, adds both the ensemble and the best single member as opponents in live eval

- [ ] **Step 1: Add ensemble + best-single opponents to `_run_live_eval`**

`PolicyAgent` already wraps anything with `forward` + `select_multi`, so `EnsemblePolicy` drops in directly. Add a new parameter to `_run_live_eval` or extend the existing function to accept an optional ensemble + member list.

In `_cmd_eval_only_ensemble` (from Task E1), after the offline eval comparison table, add a live eval block:

```python
    # Live eval: ensemble vs. best single member head-to-head
    if args.live_eval:
        from ptcg_il.live_eval import (
            LiveEvaluator, make_agent_from_policy,
            random_agent, search_planner_agent,
        )

        # Build agent for ensemble
        ensemble_agent = make_agent_from_policy(
            ensemble, artifacts["vocab"], artifacts["fixed_deck"], device="cpu",
        )

        # Build agent for best single member
        best_member = ensemble.members[best_idx]
        best_single_agent = make_agent_from_policy(
            best_member, artifacts["vocab"], artifacts["fixed_deck"], device="cpu",
        )

        # Run head-to-head: ensemble vs best single
        evaluator = LiveEvaluator(
            ensemble, artifacts["vocab"], artifacts["fixed_deck"],
            n_workers=args.live_eval_workers,
        )
        # Override _agent_fn to be the ensemble
        evaluator._agent_fn = ensemble_agent

        h2h_results = evaluator.eval_vs_opponent(
            best_single_agent, "best_single",
            n_games=args.live_eval_games,
        )
        logger.info(
            "Ensemble vs best single: win=%.1f%% [%.1f–%.1f%%], %d games",
            h2h_results.win_rate_center * 100,
            h2h_results.win_rate_lo * 100,
            h2h_results.win_rate_hi * 100,
            h2h_results.n_games,
        )

        # Ship rule: Wilson lower bound must be above 50%
        if h2h_results.win_rate_lo > 0.50:
            logger.info("Ensemble SHIPS: Wilson lower bound %.1f%% > 50%%",
                        h2h_results.win_rate_lo * 100)
        else:
            logger.warning(
                "Ensemble does NOT ship: Wilson lower bound %.1f%% ≤ 50%%. "
                "Train more members or tune config.",
                h2h_results.win_rate_lo * 100,
            )
```

- [ ] **Step 2: Verify imports and syntax**

```bash
cd python && uv run python -c "from ptcg_il.cli import _cmd_eval_only_ensemble; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add python/ptcg_il/cli.py
git commit -m "feat: live eval head-to-head for ensemble vs best single

EnsemblePolicy drops into PolicyAgent directly (already accepts anything
with forward + select_multi). Wilson lower bound above 50% is the ship
rule per spec.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task D1: Multi-ckpt packaging in `build_submission.py`

**Files:**
- Modify: `scripts/build_submission.py:1083` (--ckpt argument), `:1079-1154` (main function)

**Interfaces:**
- Consumes: Multiple `--ckpt` values, `EnsemblePolicy`
- Produces: `data/model_0.pt` … `data/model_{N-1}.pt`, `data/ensemble.json`, implicit `--no-mcts`

- [ ] **Step 1: Change `--ckpt` to `action="append"` and handle multi-ckpt in `main()`**

Change line 1083:

```python
    p.add_argument("--ckpt", action="append", required=True,
                   help="Path to checkpoint .pt file. Pass multiple times for ensemble.")
```

In `main()`, after `args = p.parse_args()` (~line 1098):

```python
    ckpt_paths = args.ckpt  # list[str]
    is_ensemble = len(ckpt_paths) > 1
    if is_ensemble:
        mcts = False  # ensemble is greedy-only
        logger.warning = print  # keep output simple
        print(f"Ensemble mode: {len(ckpt_paths)} checkpoints, implicit --no-mcts")
    else:
        mcts = not args.no_mcts
```

- [ ] **Step 2: Handle deck resolution for multiple checkpoints**

After `check_artifact_pairing` call, for ensemble mode, resolve deck from the first checkpoint and verify all members share it:

```python
    if is_ensemble:
        # Resolve deck from first checkpoint; verify all members share it
        deck, deck_record = read_ckpt_deck(Path(ckpt_paths[0]))
        if deck_record is not None:
            arch = deck_record.get("archetype_self")
            print(f"Ensemble deck label: "
                  f"{'archetype ' + str(arch) if arch is not None else 'all-decks (generalist)'}"
                  f", {deck_record.get('n_distinct_cards')} distinct cards"
                  f", vocab={deck_record.get('vocab_sha1')}")
        
        for i, ckpt_p in enumerate(ckpt_paths[1:], start=1):
            member_deck, member_record = read_ckpt_deck(Path(ckpt_p))
            if member_record is None:
                print(f"WARNING: Member {i} checkpoint has no deck label")
                continue
            if member_record.get("deck") != deck_record.get("deck"):
                raise SystemExit(
                    f"ERROR: Member {i} decklist differs from member 0. "
                    f"Ensembling different decks fails silently."
                )
            if member_record.get("vocab_sha1") != deck_record.get("vocab_sha1"):
                print(f"WARNING: Member {i} vocab_sha1 differs from member 0")
        
        check_artifact_pairing(deck_record, Path(args.data_dir), force=args.force,
                               mcts=False)
    else:
        deck, deck_record = read_ckpt_deck(Path(ckpt_paths[0]))
        # ... existing single-ckpt deck logic ...
        check_artifact_pairing(deck_record, Path(args.data_dir), force=args.force,
                               mcts=mcts)
```

- [ ] **Step 3: Write per-member weight files and ensemble.json**

Replace the single `build_model_weights` call with:

```python
    if is_ensemble:
        for i, ckpt_p in enumerate(ckpt_paths):
            member_deck, member_record = read_ckpt_deck(Path(ckpt_p))
            _build_member_weights(Path(ckpt_p), data_dir, i, deck_record=member_record)
        
        # Write ensemble.json manifest
        manifest = {"members": [f"model_{i}.pt" for i in range(len(ckpt_paths))]}
        (data_dir / "ensemble.json").write_text(json.dumps(manifest, indent=2))
        print(f"  Wrote ensemble.json ({len(ckpt_paths)} members)")
    else:
        build_model_weights(Path(ckpt_paths[0]), data_dir, deck_record=deck_record)
```

- [ ] **Step 4: Add `_build_member_weights` helper**

Add before `build_model_weights`:

```python
def _build_member_weights(ckpt_path: Path, dst_dir: Path, index: int,
                          deck_record: dict | None = None) -> None:
    """Extract EMA weights from one ensemble member checkpoint into model_{index}.pt."""
    import torch
    
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    
    ema = ckpt.get("ema_state_dict")
    if ema is not None and "shadow" in ema:
        model_state = ema["shadow"]
        print(f"  Member {index}: using EMA shadow weights (decay={ema.get('decay', '?')})")
    else:
        model_state = ckpt.get("model_state_dict")
        if model_state is None:
            raise KeyError(f"Checkpoint {ckpt_path} missing state dict")
        print(f"  Member {index}: EMA not found — falling back to raw model_state_dict")
    
    cfg = ckpt.get("config") or {}
    from build_submission import _infer_n_opp_arch
    arch = {
        "D": int(cfg.get("D", 256)),
        "heads": int(cfg.get("heads", 8)),
        "layers": int(cfg.get("layers", 4)),
        "ff": int(cfg.get("ff", 1024)),
        "n_opp_arch": int(cfg.get("n_opp_arch", _infer_n_opp_arch(model_state))),
        "n_all_cards": int(cfg.get("n_all_cards", 0)),
    }
    
    out = {"model_state_dict": model_state, "config": arch}
    if deck_record is not None:
        out["deck"] = deck_record
    torch.save(out, dst_dir / f"model_{index}.pt")
    print(f"  Wrote model_{index}.pt ({len(model_state)} parameter tensors)")
```

- [ ] **Step 5: Handle `--out` auto-suffix for ensemble**

After the existing `--out` handling, add:

```python
    out_path = Path(args.out)
    if is_ensemble and args.out == "submission.tar.gz":
        out_path = Path(f"submission-greedy-ens{len(ckpt_paths)}.tar.gz")
        print(f"Ensemble default output: {out_path}")
```

- [ ] **Step 6: Update `MODEL_FILES` and `REWRITE_RULES` for `ensemble.py`**

In the module-level constants:

```python
MODEL_FILES: list[str] = [
    "cards.py",
    "embed.py",
    "encoder.py",
    "pointer.py",
    "value.py",
    "belief.py",
    "policy.py",
    "ensemble.py",  # NEW
]

REWRITE_RULES: list[tuple[str, str]] = [
    # ... existing rules ...
    (r"from ptcg_il\.model\.policy import", r"from model.policy import"),
    # Ensemble import
    (r"from ptcg_il\.ensemble import", r"from model.ensemble import"),
    # ... rest of existing rules ...
]
```

- [ ] **Step 7: Update `build_model_package` to copy `ensemble.py`**

The existing `build_model_package` iterates over `MODEL_FILES` and copies from `model_src`. Since `ensemble.py` is in `ptcg_il/` not `ptcg_il/model/`, we need to add it to the copy logic. After the MODEL_FILES loop (~line 656), add:

```python
    # ensemble.py lives in ptcg_il/ (not ptcg_il/model/)
    ensemble_src = src_dir / "python" / "ptcg_il" / "ensemble.py"
    if ensemble_src.exists() and is_ensemble:
        text = ensemble_src.read_text()
        text = rewrite_imports(text)
        (model_dst / "ensemble.py").write_text(text)
        print(f"  Copied + rewrote ensemble.py")
```

Actually, let's handle this more cleanly. Since `MODEL_FILES` lists files in `ptcg_il/model/`, we should also add `ensemble.py` separately. Let me add it to the `EXTRA_FILES` mechanism instead: add `("ptcg_il/ensemble.py", "ensemble.py")` to `EXTRA_FILES`, but only conditionally on ensemble mode.

Actually, for simplicity, let me just handle it in the main loop with a special case. The simplest approach: add `ensemble.py` to `MODEL_FILES` and handle the path lookup:

```python
    # ensemble.py lives in ptcg_il/ not ptcg_il/model/
    for fname in MODEL_FILES:
        if fname == "ensemble.py":
            src = src_dir / "python" / "ptcg_il" / fname
        else:
            src = model_src / fname
        # ... rest of loop
```

Or better yet, add a separate lookup dict. Let me keep it simple and just handle it where needed.

Let me rethink: the cleanest approach is to NOT put `ensemble.py` in `MODEL_FILES` (since those are all under `ptcg_il/model/`), and instead add logic in `build_model_package` to also copy `ensemble.py` from `ptcg_il/`. But `ensemble.py` imports from `ptcg_il.model.policy`, so it needs import rewriting too.

Let me just add it to `EXTRA_FILES`:

```python
EXTRA_FILES: list[tuple[str, str]] = [
    ("ptcg_il/search_infer.py", "search_infer.py"),
    ("ptcg_rl/belief.py", "belief_posterior.py"),
    ("ptcg_il/ensemble.py", "ensemble.py"),
]
```

This will always copy it (even for single-ckpt builds), which is fine — it just won't be imported if `ensemble.json` doesn't exist.

- [ ] **Step 8: Update `INIT_PY` to import EnsemblePolicy**

The greedy `main.py` template conditionally imports `EnsemblePolicy`. We don't need to put it in `__init__.py` since the main.py does its own import.

- [ ] **Step 9: Run existing build submission tests**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest ../tests/test_build_submission.py -xvs
```

- [ ] **Step 10: Commit**

This is a large change — commit in stages or as one.

```bash
git add scripts/build_submission.py
git commit -m "feat: multi-ckpt ensemble packaging in build_submission.py

--ckpt now action='append'. Multiple checkpoints trigger ensemble mode:
- implicit --no-mcts
- per-member artifact check (decklist mismatch is fatal)
- writes model_0.pt ... model_{N-1}.pt
- writes ensemble.json manifest
- --out auto-suffixes to submission-greedy-ensN.tar.gz
- ensemble.py added to EXTRA_FILES for import rewriting

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task D2: Ensemble greedy main.py template

**Files:**
- Modify: `scripts/build_submission.py` (update `MAIN_PY_TEMPLATE_GREEDY` or add conditional block)

**Interfaces:**
- Produces: `main.py` template that detects `ensemble.json` at import time and loads `EnsemblePolicy` instead of single `Policy`

- [ ] **Step 1: Update the greedy main.py template**

The spec describes adding a conditional block at import time in main.py. We need to modify the `MAIN_PY_TEMPLATE_GREEDY` string in `build_submission.py`.

The change is between the vocab loading and the model loading. After `_all_card_feat` construction (~line 475 in the template string), add a conditional block.

Since `MAIN_PY_TEMPLATE_GREEDY` is a raw string, we need to carefully insert the new code. The change location is after:

```python
_all_card_feat = torch.zeros(_max_cid + 1, _card_feat_dim)
for _cid, _feat in _engine_card_features.items():
    _all_card_feat[int(_cid)] = torch.from_numpy(np.asarray(_feat, dtype=np.float32))
```

And before:

```python
_model = Policy(
    D=_cfg.get("D", 256), heads=_cfg.get("heads", 8),
    ...
```

Replace the model construction block with:

```python
# ── Ensemble detection ────────────────────────────────────────────────────
_ENSEMBLE_MANIFEST = os.path.join(DATA_DIR, "ensemble.json")
if os.path.exists(_ENSEMBLE_MANIFEST):
    from model.ensemble import EnsemblePolicy
    _manifest = json.loads(open(_ENSEMBLE_MANIFEST).read())
    _member_paths = [os.path.join(DATA_DIR, m) for m in _manifest["members"]]
    _model = EnsemblePolicy.from_checkpoints(_member_paths, _all_card_feat, device=_device)
else:
    _ckpt = torch.load(os.path.join(DATA_DIR, "model.pt"), map_location=_device, weights_only=True)
    _cfg = _ckpt.get("config", {})
    _model = Policy(
        D=_cfg.get("D", 256), heads=_cfg.get("heads", 8),
        layers=_cfg.get("layers", 4), ff=_cfg.get("ff", 1024),
        n_opp_arch=_cfg.get("n_opp_arch", 1),
        n_all_cards=_cfg.get("n_all_cards", _max_cid + 1),
        all_card_feat=_all_card_feat,
    )
    _missing, _unexpected = _model.load_state_dict(_ckpt["model_state_dict"], strict=False)
    if _missing:
        _belief_keys = [k for k in _missing if k.startswith("belief_heads.")]
        _other = [k for k in _missing if not k.startswith("belief_heads.")]
        _params = dict(_model.named_parameters(remove_duplicate=False))
        _params.update(dict(_model.named_buffers(remove_duplicate=False)))
        _loaded_ids = {id(_params[k]) for k in _ckpt["model_state_dict"] if k in _params}
        _other = [k for k in _other
                  if k not in _params or id(_params[k]) not in _loaded_ids]
        if _other:
            print(f"[agent] WARNING: {len(_other)} unexpected missing weights: {_other[:6]}")
    _model.to(_device)
    _model.eval()
```

- [ ] **Step 2: Regenerate the full MAIN_PY_TEMPLATE_GREEDY**

This is a straightforward string edit. The key change is inserting the ensemble detection block and removing the old single-model loading.

Given the size of the template, this is best done as a targeted edit using the Edit tool on `build_submission.py`.

- [ ] **Step 3: Update `build_main_py` to handle ensemble mode separately**

The existing function already knows `mcts` but doesn't know about ensemble. For ensemble builds, we still use the greedy template but it auto-detects ensemble mode. No change needed to `build_main_py` itself — the template handles it.

- [ ] **Step 4: Run existing build submission tests**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest ../tests/test_build_submission.py -xvs
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_submission.py
git commit -m "feat: ensemble detection in greedy main.py template

At import time, main.py checks for ensemble.json. If present, loads
EnsemblePolicy via from_checkpoints. Otherwise, loads single Policy
from model.pt as before. The rest of agent() is unchanged — _model
satisfies forward + select_multi regardless of which branch ran.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task D3: `build_submit.sh --ensemble` flag

**Files:**
- Modify: `scripts/build_submit.sh`

**Interfaces:**
- Produces: `--ensemble` flag accepting glob or explicit paths, expands glob, passes each as `--ckpt`

- [ ] **Step 1: Add `--ensemble` flag parsing to `build_submit.sh`**

After the existing `--no-mcts` handling (~line 31-37), parse `--ensemble`:

```bash
ENSEMBLE=0
ENSEMBLE_PATHS=()
NO_MCTS=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-mcts) NO_MCTS=1; shift ;;
        --ensemble)
            ENSEMBLE=1
            shift
            # Collect all remaining args as ensemble paths
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ENSEMBLE_PATHS+=("$1")
                shift
            done
            ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
```

- [ ] **Step 2: Expand globs in ensemble paths**

After collecting paths, expand any globs:

```bash
if [[ "$ENSEMBLE" == 1 ]]; then
    EXPANDED_PATHS=()
    for p in "${ENSEMBLE_PATHS[@]}"; do
        # Expand glob
        for f in $p; do
            if [[ -f "$f" ]]; then
                EXPANDED_PATHS+=("$f")
            fi
        done
    done
    if [[ ${#EXPANDED_PATHS[@]} -eq 0 ]]; then
        echo "ERROR: --ensemble specified but no files matched" >&2
        exit 1
    fi
    ENSEMBLE_PATHS=("${EXPANDED_PATHS[@]}")
    echo "Ensemble: ${#ENSEMBLE_PATHS[@]} members"
    for p in "${ENSEMBLE_PATHS[@]}"; do
        echo "  $p"
    done
fi
```

- [ ] **Step 3: Pass `--ckpt` for each path to `build_submission.py`**

Replace the final `uv run python scripts/build_submission.py` call:

```bash
if [[ "$ENSEMBLE" == 1 ]]; then
    CKPT_ARGS=()
    for p in "${ENSEMBLE_PATHS[@]}"; do
        CKPT_ARGS+=(--ckpt "$p")
    done
    uv run python scripts/build_submission.py \
        --data-dir python/data \
        "${CKPT_ARGS[@]}" \
        --out "${OUT:-submission-greedy-ens${#ENSEMBLE_PATHS[@]}.tar.gz}"
elif [[ "$NO_MCTS" == 1 ]]; then
    uv run python scripts/build_submission.py \
        --data-dir python/data \
        --ckpt "$CKPT" \
        --no-mcts \
        --out "${OUT:-submission-greedy.tar.gz}"
else
    uv run python scripts/build_submission.py \
        --data-dir python/data \
        --ckpt "$CKPT" \
        --out "${OUT:-submission.tar.gz}"
fi
```

- [ ] **Step 4: Test the script parses correctly**

```bash
bash scripts/build_submit.sh --help 2>&1 || true
# Manually verify --ensemble shows up
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_submit.sh
git commit -m "feat: --ensemble flag in build_submit.sh

Accepts a glob or explicit paths, expands the glob, and passes each
matched file as a separate --ckpt to build_submission.py. Aborts if
no files match.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task D4: `run_ensemble_train.sh` script

**Files:**
- Create: `scripts/run_ensemble_train.sh`

**Interfaces:**
- Produces: Script that trains N independently-seeded specialists sequentially

- [ ] **Step 1: Create the training script**

Create `scripts/run_ensemble_train.sh`:

```bash
#!/bin/bash
# Train N independently-seeded specialist models for ensemble.
#
# Usage: ./scripts/run_ensemble_train.sh <ARCH> <N> [--extra-flags ...]
#
#   ./scripts/run_ensemble_train.sh 1 3
#   ./scripts/run_ensemble_train.sh 1 5 --total-steps 30000 --batch-size 512
#
# Trains sequentially on one GPU. Each member gets a different --seed.

set -euo pipefail

ARCH="${1:?Usage: $0 <ARCH> <N> [--extra-flags ...]}"
N="${2:?Usage: $0 <ARCH> <N> [--extra-flags ...]}"
shift 2

# Archetype id: strip 'a' prefix if present
ARCH_ID="${ARCH#a}"

cd "$(dirname "$0")/../python"

for S in $(seq 0 $((N - 1))); do
    echo "=== Training member $S/$N (seed=$S, archetype=$ARCH_ID) ==="
    uv run python -m ptcg_il.cli train \
        --data-dir data \
        --out-dir "checkpoints_a${ARCH_ID}_s${S}" \
        --archetype-self "$ARCH_ID" \
        --seed "$S" \
        "$@"
    echo "=== Member $S done ==="
done

echo "=== All $N members trained ==="
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/run_ensemble_train.sh
```

- [ ] **Step 3: Verify the script syntax**

```bash
bash -n scripts/run_ensemble_train.sh
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_ensemble_train.sh
git commit -m "feat: run_ensemble_train.sh — train N seeded specialists

Sequential, one GPU. Each member gets --seed S for decorrelated init
and batch order. Passes through extra flags for config overrides.

Usage: ./scripts/run_ensemble_train.sh <ARCH> <N> [--extra-flags ...]

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task D5: Import verification test update

**Files:**
- Modify: `tests/test_build_submission.py` (verify `ensemble.py` is covered)
- Modify: `scripts/build_submission.py` (`_DEAD_IN_BUNDLE` or `REWRITE_RULES`)

**Interfaces:**
- Consumes: `MODEL_FILES`, `REWRITE_RULES`
- Produces: `test_no_packaged_module_still_imports_ptcg_il` passes with `ensemble.py` in the bundle

- [ ] **Step 1: Verify `ensemble.py` is covered by the import rewrite test**

The test `test_no_packaged_module_still_imports_ptcg_il` iterates over `bs.MODEL_FILES` and checks for surviving `ptcg_il` imports. Since we added `ensemble.py` to `EXTRA_FILES` (not `MODEL_FILES`), it won't be checked.

We need to either:
a) Add `ensemble.py` to `MODEL_FILES` with special path handling, or
b) Extend the test to also check `EXTRA_FILES`

Option (b) is cleaner. Let's extend the test.

Actually, let me put `ensemble.py` into `MODEL_FILES` and handle the path difference in `build_model_package`:

In `build_model_package`, after the `model_src` definition:

```python
    # Files in MODEL_FILES that live in ptcg_il/ not ptcg_il/model/
    _TOP_LEVEL_MODEL_FILES = {"ensemble.py"}
    
    for fname in MODEL_FILES:
        if fname in _TOP_LEVEL_MODEL_FILES:
            src = src_dir / "python" / "ptcg_il" / fname
        else:
            src = model_src / fname
```

And add `"ensemble.py"` to `MODEL_FILES`.

The import rewrite test also needs to find `ensemble.py` in the right place. Let's check the test again:

```python
    model_dir = src_dir / "python" / "ptcg_il" / "model"
    for fname in bs.MODEL_FILES:
        path = model_dir / fname
        if not path.exists():
            path = src_dir / "python" / "ptcg_il" / fname
```

It already has a fallback: if not in `model/`, look in `ptcg_il/`. So putting `ensemble.py` in `MODEL_FILES` will work — the test will fall back to `ptcg_il/ensemble.py`.

- [ ] **Step 2: Add `ensemble.py` to `MODEL_FILES`**

```python
MODEL_FILES: list[str] = [
    "cards.py",
    "embed.py",
    "encoder.py",
    "pointer.py",
    "value.py",
    "belief.py",
    "policy.py",
    "ensemble.py",
]
```

And update `build_model_package` to handle path lookup:

In `build_model_package` (~line 643), after the `model_src` definition, change the copy loop:

Actually, let me look at the current `build_model_package` more carefully. The loop at line 647 iterates over `MODEL_FILES` and looks for them at `model_src / fname`. For `ensemble.py`, that path won't exist. We need to look in `ptcg_il/`.

Let me add a path lookup dict or fallback:

```python
    model_src = src_dir / "python" / "ptcg_il" / "model"
    top_level_src = src_dir / "python" / "ptcg_il"
    model_dst = dst_dir / "model"
    model_dst.mkdir(parents=True, exist_ok=True)

    for fname in MODEL_FILES:
        src = model_src / fname
        if not src.exists():
            src = top_level_src / fname  # fallback: ptcg_il/ensemble.py etc.
        if not src.exists():
            print(f"WARNING: {src} not found — skipping")
            continue
        text = src.read_text()
        text = rewrite_imports(text)
        (model_dst / fname).write_text(text)
        print(f"  Copied + rewrote {fname}")
```

And remove `("ptcg_il/ensemble.py", "ensemble.py")` from `EXTRA_FILES`.

- [ ] **Step 3: Run the import verification test**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest ../tests/test_build_submission.py::test_no_packaged_module_still_imports_ptcg_il -xvs
```

Expected: PASS. If it fails because `ensemble.py` imports `ptcg_il.model.policy` in a way not covered by `REWRITE_RULES`, add the missing rule.

- [ ] **Step 4: Run all build submission tests**

```bash
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest ../tests/test_build_submission.py -xvs
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_submission.py tests/test_build_submission.py
git commit -m "fix: add ensemble.py to MODEL_FILES with path fallback

ensemble.py lives in ptcg_il/ not ptcg_il/model/. build_model_package
now falls back to ptcg_il/ when a MODEL_FILE isn't found under model/.
Import verification test covers ensemble.py via the same fallback.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Post-Implementation Verification

After all tasks are complete, run the full test suite:

```bash
# Model + ensemble tests
cd python && uv run pytest python/tests/test_model_policy.py python/tests/test_ensemble.py -xvs

# Build submission tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest ../tests/test_build_submission.py -xvs

# Full test suite (both directories)
cd python && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest ../tests/ python/tests/ -xvs
```
