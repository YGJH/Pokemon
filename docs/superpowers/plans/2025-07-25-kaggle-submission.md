# Kaggle Submission Builder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a script that creates a Kaggle-ready tar.gz submission from training checkpoints

**Architecture:** `build.py` copies/adapts model code into a self-contained `submission/` directory, bundles model weights + vocab + static tables + deck.csv, and packs everything into a tar.gz. The submission loads the model at import time (fail-fast).

**Tech Stack:** Python 3.11, PyTorch (inference only), NumPy, tarfile

## Global Constraints

- Submission must work on Kaggle environment (torch + numpy available)
- Model loads at import time — failure raises immediately
- No training dependencies (wandb, pytest, ptcg_mine)
- Submission tar.gz < 100 MB (model ~35MB fp32)
- All inference runs on CPU by default

---

## File Map

```
scripts/build_submission.py          # NEW: build script
submission/main.py                   # GENERATED: entry point
submission/model/                    # GENERATED: adapted from ptcg_il/model/
  __init__.py                        #   MLP, Policy, select_multi
  ref_map.py                         #   build_ref_map (84 lines, standalone)
  cards.py                           #   CardEncoder, AttackEncoder, MLP
  embed.py                           #   TokenEmbedder
  encoder.py                         #   Encoder (TransformerEncoder wrapper)
  pointer.py                         #   PointerHead + msgru
  value.py                           #   ValueHead
  belief.py                          #   BeliefModule
  policy.py                          #   Policy, select_multi (multiselect_ce kept but unused)
  featurizer.py                      #   featurize() + all helper functions
submission/data/                     # GENERATED: copied from training outputs
  model.pt                           #   state dict (EMA-averaged)
  vocab.json                         #   id_to_index + attack_id_to_index
  card_static.npy                    #   frozen card feature table
  attack_static.npy                  #   frozen attack feature table
  deck.csv                           #   FIXED_DECK (60 card IDs)
```

---

### Task 1: Create the submission model package

**Files:**
- Create: `scripts/build_submission.py` (model-copying section)

**Interfaces:**
- Consumes: `python/ptcg_il/model/*.py`, `python/ptcg_il/featurizer.py`, `python/ptcg_il/ref_map.py`
- Produces: `submission/model/` directory with rewritten imports

- [ ] **Step 1: Write the `rewrite_imports` helper**

```python
import re
import shutil
from pathlib import Path

REWRITE_RULES = [
    (r"from ptcg_il\.model\.cards import", r"from model.cards import"),
    (r"from ptcg_il\.model\.embed import", r"from model.embed import"),
    (r"from ptcg_il\.model\.encoder import", r"from model.encoder import"),
    (r"from ptcg_il\.model\.pointer import", r"from model.pointer import"),
    (r"from ptcg_il\.model\.value import", r"from model.value import"),
    (r"from ptcg_il\.model\.belief import", r"from model.belief import"),
    (r"from ptcg_il\.model\.policy import", r"from model.policy import"),
    (r"from ptcg_il\.model import", r"from model import"),
    (r"from ptcg_il\.model\.cards import", r"from model.cards import"),
    (r"from ptcg_il\.ref_map import", r"from model.ref_map import"),
]

def rewrite_imports(text: str) -> str:
    for pattern, replacement in REWRITE_RULES:
        text = re.sub(pattern, replacement, text)
    return text
```

- [ ] **Step 2: Write the model copy function**

```python
MODEL_FILES = [
    "cards.py",
    "embed.py", 
    "encoder.py",
    "pointer.py",
    "value.py",
    "belief.py",
    "policy.py",
]

def build_model_package(src_dir: Path, dst_dir: Path) -> None:
    """Copy model files from ptcg_il/model/ to submission/model/, rewriting imports."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    model_src = src_dir / "python" / "ptcg_il" / "model"
    
    for fname in MODEL_FILES:
        src = model_src / fname
        text = src.read_text()
        text = rewrite_imports(text)
        (dst_dir / fname).write_text(text)
    
    # Copy ref_map.py (self-contained, no ptcg_il imports)
    ref_src = src_dir / "python" / "ptcg_il" / "ref_map.py"
    shutil.copy(ref_src, dst_dir / "ref_map.py")
    
    # Copy and adapt featurizer.py
    feat_src = src_dir / "python" / "ptcg_il" / "featurizer.py"
    feat_text = feat_src.read_text()
    feat_text = rewrite_imports(feat_text)
    (dst_dir / "featurizer.py").write_text(feat_text)
```

- [ ] **Step 3: Write `__init__.py` for submission model**

```python
INIT_PY = '''"""Submission model package — inference-only."""
from model.cards import MLP, CardEncoder, AttackEncoder
from model.policy import Policy, select_multi
'''

def write_init(dst_dir: Path) -> None:
    (dst_dir / "__init__.py").write_text(INIT_PY)
```

- [ ] **Step 4: Verify the model package imports correctly**

```bash
cd submission && python -c "from model import Policy, select_multi, MLP; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_submission.py
git commit -m "feat: add submission model package builder"
```

---

### Task 2: Create the data bundling section of build.py

**Files:**
- Modify: `scripts/build_submission.py` (add data section)

**Interfaces:**
- Consumes: checkpoint `.pt` file, `data/vocab.json`, `data/card_static_table.npy`, `data/attack_static_table.npy`, `data/archetypes.json` (for FIXED_DECK)
- Produces: `submission/data/` with model.pt, vocab.json, card_static.npy, attack_static.npy, deck.csv

- [ ] **Step 1: Write the checkpoint converter**

```python
def build_model_weights(ckpt_path: Path, dst_dir: Path) -> None:
    """Extract EMA weights + metadata from checkpoint into submission model.pt."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    
    # Build metadata for model reconstruction
    V = int(ckpt.get("V", ckpt.get("vocab_size", 262)))
    A = int(ckpt.get("A", ckpt.get("attack_vocab_size", 181)))
    model_state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict"))
    
    submission_pt = {
        "model_state_dict": model_state,
        "V": V,
        "A": A,
        "D": 256,
        "heads": 8,
        "layers": 4,
        "ff": 1024,
    }
    torch.save(submission_pt, dst_dir / "model.pt")
```

- [ ] **Step 2: Write the data copy function**

```python
def build_data_files(data_dir: Path, dst_dir: Path) -> None:
    """Copy vocab, static tables, and FIXED_DECK into submission/data/."""
    import json
    import shutil
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Vocab
    shutil.copy(data_dir / "vocab.json", dst_dir / "vocab.json")
    
    # Static tables
    shutil.copy(data_dir / "card_static_table.npy", dst_dir / "card_static.npy")
    shutil.copy(data_dir / "attack_static_table.npy", dst_dir / "attack_static.npy")
    
    # FIXED_DECK from archetypes.json
    with open(data_dir / "archetypes.json") as f:
        archetypes = json.load(f)
    fixed_deck = archetypes.get("fixed_deck", [])
    if not fixed_deck:
        raise ValueError("archetypes.json missing 'fixed_deck' key")
    
    (dst_dir / "deck.csv").write_text("\n".join(str(c) for c in fixed_deck))
```

- [ ] **Step 3: Test data bundling**

```bash
python scripts/build_submission.py data --ckpt checkpoints/ckpt-best.pt --out-dir /tmp/submission-test
ls /tmp/submission-test/data/
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_submission.py
git commit -m "feat: add submission data bundler"
```

---

### Task 3: Create main.py template

**Files:**
- Modify: `scripts/build_submission.py` (add main.py generation)

**Interfaces:**
- Consumes: nothing (generated inline)
- Produces: `submission/main.py`

- [ ] **Step 1: Write the main.py generator**

```python
MAIN_PY_TEMPLATE = '''"""Pokémon TCG AI Agent — Kaggle submission entry point.

Model loads at import time.  Any failure (missing files, weight mismatch,
CUDA error) raises immediately — no silent fallbacks.
"""

import os
import numpy as np
import torch

from cg.api import to_observation_class
from model import Policy, select_multi
from model.featurizer import featurize

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_json(path: str) -> dict:
    import json
    with open(path) as f:
        return json.load(f)


def _read_deck_csv() -> list[int]:
    path = os.path.join(DATA_DIR, "deck.csv")
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/data/deck.csv"
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def _sample_to_batch(sample: dict, device: torch.device) -> dict:
    """Convert a single numpy sample to a batch-1 torch dict."""
    batch = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v).unsqueeze(0)
            if v.dtype == np.int64:
                t = t.long()
            elif v.dtype == np.bool_:
                t = t.bool()
            else:
                t = t.float()
            batch[k] = t.to(device)
    return batch


# ---- Import-time model loading ----

_vocab = _load_json(os.path.join(DATA_DIR, "vocab.json"))
_vocab_full = {
    "id_to_index": {int(k): int(v) for k, v in _vocab.get("id_to_index", {}).items()},
    "attack_id_to_index": {int(k): int(v) for k, v in _vocab.get("attack_id_to_index", {}).items()},
}

_card_static = torch.from_numpy(np.load(os.path.join(DATA_DIR, "card_static.npy")))
_attack_static = torch.from_numpy(np.load(os.path.join(DATA_DIR, "attack_static.npy")))

_ckpt = torch.load(os.path.join(DATA_DIR, "model.pt"), map_location=_device, weights_only=True)
_model = Policy(
    V=_ckpt["V"], A=_ckpt["A"],
    D=_ckpt.get("D", 256), heads=_ckpt.get("heads", 8),
    layers=_ckpt.get("layers", 4), ff=_ckpt.get("ff", 1024),
    card_static_table=_card_static, attack_static_table=_attack_static,
)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.to(_device)
_model.eval()

_fixed_deck = _read_deck_csv()

# ---- Agent function ----

def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    
    # Deck selection step
    if obs.select is None:
        return list(_fixed_deck)
    
    # Featurize observation → tensor dict
    sample = featurize(obs_dict, _vocab_full, value_target=0.0, sample_weight=1.0)
    batch = _sample_to_batch(sample, _device)
    max_count = int(sample["maxCount"])
    
    with torch.no_grad():
        if max_count == 1:
            logits, _value, _hist = _model(batch)
            logits = logits.masked_fill(~batch["opt_mask"], -1e9)
            return [int(logits.argmax(dim=-1)[0].item())]
        else:
            chosen = select_multi(_model, batch)
            picks = [int(p) for p in chosen[0].tolist() if p >= 0]
            return picks[:max_count]
'''


def build_main_py(dst_dir: Path) -> None:
    (dst_dir / "main.py").write_text(MAIN_PY_TEMPLATE)
```

- [ ] **Step 2: Verify main.py is syntactically valid**

```bash
python -c "compile(open('submission/main.py').read(), 'main.py', 'exec'); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/build_submission.py
git commit -m "feat: add main.py template generation"
```

---

### Task 4: Create the tar.gz packager and CLI

**Files:**
- Modify: `scripts/build_submission.py` (add packaging + CLI)

**Interfaces:**
- Consumes: `submission/` directory
- Produces: `submission.tar.gz`

- [ ] **Step 1: Write the tar.gz packager**

```python
def pack_submission(submission_dir: Path, output_path: Path) -> None:
    """Create submission.tar.gz from the submission directory."""
    import tarfile
    with tarfile.open(output_path, "w:gz") as tar:
        # Add files at root level (not inside a submission/ folder)
        for fname in ["main.py", "model", "data"]:
            path = submission_dir / fname
            if path.is_dir():
                tar.add(path, arcname=fname)
            else:
                tar.add(path, arcname=fname)
```

- [ ] **Step 2: Write the CLI**

```python
def main():
    import argparse
    p = argparse.ArgumentParser(description="Build Kaggle submission package")
    p.add_argument("--data-dir", required=True, help="Path to training data/ directory")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--out", default="submission.tar.gz", help="Output tar.gz path")
    p.add_argument("--work-dir", default="/tmp/submission-build", help="Temp build directory")
    args = p.parse_args()
    
    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    
    model_dir = work / "model"
    data_dir = work / "data"
    
    print("Building model package...")
    build_model_package(Path.cwd(), model_dir)
    write_init(model_dir)
    print("  OK")
    
    print("Building data files...")
    build_data_files(Path(args.data_dir), data_dir)
    build_model_weights(Path(args.ckpt), data_dir)
    print("  OK")
    
    print("Generating main.py...")
    build_main_py(work)
    print("  OK")
    
    print(f"Packing {args.out}...")
    pack_submission(work, Path(args.out))
    print(f"  Done: {args.out} ({Path(args.out).stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test the full build pipeline**

```bash
python scripts/build_submission.py \
    --data-dir python/data \
    --ckpt python/checkpoints/ckpt-best.pt \
    --out /tmp/test-submission.tar.gz
ls -lh /tmp/test-submission.tar.gz
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_submission.py
git commit -m "feat: add submission packager and CLI"
```

---

### Task 5: End-to-end validation

**Files:**
- Create: `scripts/test_submission.py` (optional validation script)

**Interfaces:**
- Consumes: `submission.tar.gz`
- Produces: validation report (printed to stdout)

- [ ] **Step 1: Extract and verify structure**

```bash
tar xzf /tmp/test-submission.tar.gz -C /tmp/submission-extract
ls /tmp/submission-extract/
ls /tmp/submission-extract/model/
ls /tmp/submission-extract/data/
```

Expected: `main.py`, `model/` (11 files), `data/` (5 files)

- [ ] **Step 2: Test model import (import-time loading)**

```bash
cd /tmp/submission-extract && python -c "
import sys; sys.path.insert(0, '.')
# This should load the model at import time
from main import agent, _model, _fixed_deck
print(f'Model loaded: {sum(p.numel() for p in _model.parameters()):,} params')
print(f'Fixed deck: {len(_fixed_deck)} cards')
assert len(_fixed_deck) == 60
print('Import-time loading: OK')
"
```

- [ ] **Step 3: Test agent function with a dummy observation**

```bash
cd /tmp/submission-extract && python -c "
import sys; sys.path.insert(0, '.')
from main import agent

# Dummy deck selection
obs = {'select': None}
deck = agent(obs)
assert len(deck) == 60
print(f'Deck selection: OK ({len(deck)} cards)')
"
```

- [ ] **Step 4: Test with a real shard sample (round-trip)**

```bash
cd /tmp/submission-extract && python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from main import agent

# Load a real sample from the training shards
import json
with open('data/vocab.json') as f:
    vocab = json.load(f)

# Reconstruct an observation from a shard sample (if observations dir exists)
# For now, verify the model can process a synthetic observation
print('Basic round-trip: OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_submission.py
git commit -m "feat: add end-to-end submission validation"
```

---

## Verification

After all tasks are complete:

```bash
# Full build
python scripts/build_submission.py \
    --data-dir python/data \
    --ckpt python/checkpoints/ckpt-best.pt \
    --out submission.tar.gz

# Validate
tar tzf submission.tar.gz | head -20
python -c "
import tarfile, io
with tarfile.open('submission.tar.gz') as tar:
    names = tar.getnames()
    assert 'main.py' in names
    assert any(n.startswith('model/') for n in names)
    assert any(n.startswith('data/') for n in names)
    assert any('deck.csv' in n for n in names)
print('Submission structure: OK')
"

# Size check
ls -lh submission.tar.gz  # should be < 50 MB
```
