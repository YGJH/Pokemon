#!/usr/bin/env python3
"""Build a self-contained Kaggle submission package.

Usage:
    python scripts/build_submission.py [--src-dir .] [--dst-dir submission]
    python scripts/build_submission.py --ckpt checkpoints/ckpt-best.pt --data-dir data

Sections:
    1. Model package  — copy ptcg_il/model/*.py to submission/model/, rewriting imports
    2. Model weights   — extract EMA weights + metadata from training checkpoint
    3. Data files      — copy vocab, static tables, and FIXED_DECK into submission/data/
    4. (future) Agent  — generate submission agent.py with Policy inference wrapper
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# Section 1: Model package builder
# ============================================================

REWRITE_RULES: list[tuple[str, str]] = [
    # Submodule-specific imports (most specific first, before the blanket rule)
    (r"from ptcg_il\.model\.cards import", r"from model.cards import"),
    (r"from ptcg_il\.model\.embed import", r"from model.embed import"),
    (r"from ptcg_il\.model\.encoder import", r"from model.encoder import"),
    (r"from ptcg_il\.model\.pointer import", r"from model.pointer import"),
    (r"from ptcg_il\.model\.value import", r"from model.value import"),
    (r"from ptcg_il\.model\.belief import", r"from model.belief import"),
    (r"from ptcg_il\.model\.policy import", r"from model.policy import"),
    # Blanket "from ptcg_il.model import X" rule (must come after submodule rules
    # so it doesn't prematurely match the submodule prefixes)
    (r"from ptcg_il\.model import", r"from model import"),
    # ref_map (copied into model/ so modules import it as model.ref_map)
    (r"from ptcg_il\.ref_map import", r"from model.ref_map import"),
]

MODEL_FILES: list[str] = [
    "cards.py",
    "embed.py",
    "encoder.py",
    "pointer.py",
    "value.py",
    "belief.py",
    "policy.py",
]

# __init__.py for the submission model package.
# IMPORTANT: MLP must be defined *before* the submodule imports because
# cards.py, embed.py, and pointer.py all do ``from model import MLP`` at
# module level.  If we tried to import MLP from model.cards we would create
# a circular import (model → model.cards → model.MLP which doesn't exist yet).
INIT_PY = '''\
"""Submission model package — inference-only."""
import torch.nn as nn


def MLP(
    in_features: int,
    hidden_features: int,
    out_features: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Two-layer MLP with GELU: Linear(i,h) -> GELU -> Dropout -> Linear(h,o).

    Matches the B.1 definition from TRANSFORMER_IL_SPEC.md exactly.
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_features, out_features),
    )


from model.cards import CardEncoder, AttackEncoder  # noqa: E402
from model.policy import Policy, select_multi  # noqa: E402
'''

MAIN_PY_TEMPLATE = '''"""Pokémon TCG AI Agent — Kaggle submission entry point.

Model loads at import time.  Any failure (missing files, weight mismatch,
CUDA error) raises immediately — no silent fallbacks.
"""

import os
import numpy as np
import torch

try:
    from cg.api import to_observation_class
except ImportError:
    # Local testing fallback when cg is not available
    def to_observation_class(obs_dict: dict):
        """Minimal Observation stub for local testing."""
        select = obs_dict.get("select")
        return type("Observation", (), {
            "select": None if select is None else type("SelectData", (), select)(),
        })()
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
    """Convert a single numpy sample to a batch-1 torch dict.

    ``featurize()`` emits some fields as 0-d numpy *scalars* rather than arrays
    -- notably ``minCount`` / ``maxCount`` (``np.int64(...)``).  Only copying
    ``np.ndarray`` silently drops those, and ``select_multi`` then dies with
    ``KeyError: 'minCount'`` on every multi-select decision (~9% of decisions,
    i.e. a forfeited game).  Training never hits this because the shards store
    them as arrays and ``collate_fn`` batches them.
    """
    batch = {}
    for k, v in sample.items():
        # Add the batch dim explicitly rather than via unsqueeze: for a 0-d
        # input np.ascontiguousarray already promotes to shape (1,), so an
        # unsqueeze on top would yield (1, 1) and _select_multi_raw would fail
        # with "too many indices for tensor of dimension 1".
        if isinstance(v, np.ndarray):
            arr = np.ascontiguousarray(v)[None, ...]
        elif isinstance(v, (np.generic, int, float, bool)):
            arr = np.asarray(v).reshape(1)
        else:
            continue
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if arr.dtype == np.bool_:
            t = t.bool()
        elif np.issubdtype(arr.dtype, np.integer):
            t = t.long()
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
_model.load_state_dict(_ckpt["model_state_dict"], strict=False)
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


def rewrite_imports(text: str) -> str:
    """Apply all REWRITE_RULES to *text*, returning the rewritten source."""
    for pattern, replacement in REWRITE_RULES:
        text = re.sub(pattern, replacement, text)
    return text


def write_init(dst_dir: Path) -> None:
    """Write the model package __init__.py to *dst_dir*."""
    (dst_dir / "__init__.py").write_text(INIT_PY)
    print(f"  Wrote __init__.py")


def build_model_package(src_dir: Path, dst_dir: Path) -> None:
    """Copy model files from ptcg_il/model/ to submission/model/, rewriting imports.

    Parameters
    ----------
    src_dir : Path
        Project root (contains ``python/ptcg_il/``).
    dst_dir : Path
        Submission root (``submission/``).  Model files land in
        ``dst_dir / "model" /``.
    """
    model_src = src_dir / "python" / "ptcg_il" / "model"
    model_dst = dst_dir / "model"
    model_dst.mkdir(parents=True, exist_ok=True)

    for fname in MODEL_FILES:
        src = model_src / fname
        if not src.exists():
            print(f"WARNING: {src} not found — skipping")
            continue
        text = src.read_text()
        text = rewrite_imports(text)
        (model_dst / fname).write_text(text)
        print(f"  Copied + rewrote {fname}")

    # ref_map.py — self-contained, no ptcg_il imports, copy verbatim
    ref_src = src_dir / "python" / "ptcg_il" / "ref_map.py"
    if ref_src.exists():
        shutil.copy(ref_src, model_dst / "ref_map.py")
        print(f"  Copied ref_map.py (verbatim)")
    else:
        print(f"WARNING: {ref_src} not found — skipping")

    # featurizer.py — has ptcg_il.ref_map import, needs rewriting
    feat_src = src_dir / "python" / "ptcg_il" / "featurizer.py"
    if feat_src.exists():
        feat_text = feat_src.read_text()
        feat_text = rewrite_imports(feat_text)
        (model_dst / "featurizer.py").write_text(feat_text)
        print(f"  Copied + rewrote featurizer.py")
    else:
        print(f"WARNING: {feat_src} not found — skipping")

    write_init(model_dst)


def build_main_py(dst_dir: Path) -> None:
    """Generate submission/main.py from the MAIN_PY_TEMPLATE."""
    (dst_dir / "main.py").write_text(MAIN_PY_TEMPLATE)
    print(f"  Wrote main.py")


# ============================================================
# Section 2: Model weights — extract EMA weights from checkpoint
# ============================================================

def build_model_weights(ckpt_path: Path, dst_dir: Path,
                        deck_record: dict | None = None) -> None:
    """Extract EMA weights + metadata from checkpoint into submission model.pt.

    Infers V (card vocab size) and A (attack vocab size) from the embedding
    layer shapes in the EMA shadow dict.  The checkpoint contains numpy scalars
    so loading requires ``weights_only=False``.

    Parameters
    ----------
    ckpt_path : Path
        Path to a training checkpoint (e.g. ``checkpoints/ckpt-best.pt``).
    dst_dir : Path
        Output directory for the submission ``model.pt``.
    """
    import torch

    dst_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Extract EMA-averaged weights (preferred) or fall back to raw model state
    ema = ckpt.get("ema_state_dict")
    if ema is not None and "shadow" in ema:
        model_state = ema["shadow"]
        print(f"  Using EMA shadow weights (decay={ema.get('decay', '?')})")
    else:
        model_state = ckpt.get("model_state_dict")
        if model_state is None:
            raise KeyError("Checkpoint missing both 'ema_state_dict' and 'model_state_dict'")
        print(f"  EMA not found — falling back to raw model_state_dict")

    # Infer V from card embedding, A from attack embedding
    V = None
    A = None
    for key, tensor in model_state.items():
        if key == "embed.card.id_emb.weight" and V is None:
            V = int(tensor.shape[0])
        if key == "pointer.attack.id_emb.weight" and A is None:
            A = int(tensor.shape[0])

    if V is None:
        V = int(ckpt.get("V", ckpt.get("vocab_size", 262)))
        print(f"  V not found in weights — using {V} from metadata")
    if A is None:
        A = int(ckpt.get("A", ckpt.get("attack_vocab_size", 181)))
        print(f"  A not found in weights — using {A} from metadata")

    print(f"  V={V}  A={A}")

    submission_pt = {
        "model_state_dict": model_state,
        "V": V,
        "A": A,
        "D": 256,
        "heads": 8,
        "layers": 4,
        "ff": 1024,
    }
    # Keep the deck label with the weights so the shipped model.pt is
    # self-describing and a wrong pairing is auditable after the fact.
    if deck_record is not None:
        submission_pt["deck"] = deck_record
    torch.save(submission_pt, dst_dir / "model.pt")
    print(f"  Wrote model.pt ({len(model_state)} parameter tensors"
          f"{', deck-labelled' if deck_record is not None else ''})")


# ============================================================
# Section 3: Data files — copy static assets + FIXED_DECK
# ============================================================

def read_ckpt_deck(ckpt_path: Path) -> tuple[list[int] | None, dict | None]:
    """Read the deck card list stamped into a checkpoint by ``save_checkpoint``.

    Returns ``(deck, record)``, both ``None`` for checkpoints written before
    deck labelling existed.
    """
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    record = ckpt.get("deck")
    if not record or not record.get("deck"):
        return None, None
    return [int(c) for c in record["deck"]], record


def build_data_files(data_dir: Path, dst_dir: Path, deck: list[int] | None = None) -> None:
    """Copy vocab, static tables, and FIXED_DECK into submission/data/.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``vocab.json``, ``card_static_table.npy``,
        ``attack_static_table.npy``, and ``archetypes.json``.
    dst_dir : Path
        Output directory for the submission data files.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Vocab (copy verbatim)
    shutil.copy(data_dir / "vocab.json", dst_dir / "vocab.json")
    print(f"  Copied vocab.json")

    # Static tables (renamed for submission)
    shutil.copy(data_dir / "card_static_table.npy", dst_dir / "card_static.npy")
    print(f"  Copied card_static.npy")
    shutil.copy(data_dir / "attack_static_table.npy", dst_dir / "attack_static.npy")
    print(f"  Copied attack_static.npy")

    # deck.csv (one card ID per line) — MUST be the deck this checkpoint was
    # trained on, not archetypes.json's `fixed_deck`.
    #
    # Per-deck specialists are trained with `--archetype-self`, and the six
    # archetype decks are near-disjoint (pairwise multiset-Jaccard <= 0.17).
    # Shipping `fixed_deck` with a specialist is a *silent* catastrophe: the
    # policy sees a board of cards it never trained on, every one of which maps
    # to UNKNOWN_CARD, and nothing raises.  An arch-2 model shipped with
    # `fixed_deck` (arch 9) overlaps its training deck by only 6%.
    #
    # `save_checkpoint` stamps the deck under the "deck" key, so prefer that and
    # refuse to guess when it is absent.
    with open(data_dir / "archetypes.json") as f:
        archetypes = json.load(f)

    if deck is not None:
        source = "checkpoint deck label"
    else:
        raise ValueError(
            "Checkpoint carries no 'deck' label, so the correct deck cannot be "
            "determined. Retrain with ptcg_il.cli (which stamps the deck into "
            "every .pt), or pass --deck-csv explicitly. "
            "Refusing to fall back to archetypes.json's fixed_deck: pairing a "
            "specialist with the wrong deck fails silently."
        )

    if len(deck) != 60:
        raise ValueError(f"deck has {len(deck)} cards, expected 60")

    (dst_dir / "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n")
    print(f"  Wrote deck.csv ({len(deck)} cards, from {source})")

    # Sanity check against archetypes.json so a mismatch is visible in the log.
    reps = {a["id"]: a["representative"] for a in archetypes.get("archetypes", [])}
    matched = [aid for aid, rep in reps.items() if rep == deck]
    print(f"  Deck matches archetype id(s): {matched or '<none>'}"
          f"{'  (== fixed_deck)' if deck == archetypes.get('fixed_deck') else ''}")


# ============================================================
# Verification
# ============================================================

def verify_model_imports(dst_dir: Path) -> bool:
    """Run a quick smoke-test import of the submission model package.

    Returns True if the import succeeds, False otherwise.
    """
    result = subprocess.run(
        [sys.executable, "-c", "from model import Policy, select_multi, CardEncoder, AttackEncoder; print('OK')"],
        cwd=str(dst_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and "OK" in result.stdout:
        print("  Import verification: PASS")
        return True
    print(f"  Import verification: FAIL")
    print(f"  stdout: {result.stdout.strip()}")
    print(f"  stderr: {result.stderr.strip()}")
    return False


# ============================================================
# Packaging
# ============================================================

def pack_submission(submission_dir: Path, output_path: Path) -> None:
    """Create submission.tar.gz from the submission directory."""
    import tarfile

    def _filter(info: "tarfile.TarInfo") -> "tarfile.TarInfo | None":
        # Import verification runs before packing and leaves __pycache__ behind;
        # shipping stale .pyc files is pure bloat and can shadow the sources.
        parts = Path(info.name).parts
        if "__pycache__" in parts or info.name.endswith(".pyc"):
            return None
        return info

    with tarfile.open(output_path, "w:gz") as tar:
        # Add files at root level (not inside a submission/ folder)
        for fname in ["main.py", "model", "data"]:
            tar.add(submission_dir / fname, arcname=fname, filter=_filter)


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="Build Kaggle submission package")
    p.add_argument("--data-dir", required=True, help="Path to training data/ directory")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--out", default="submission.tar.gz", help="Output tar.gz path")
    p.add_argument("--work-dir", default="/tmp/submission-build", help="Temp build directory")
    p.add_argument("--deck-csv", default=None,
                   help="Override the deck with this csv (one card id per line). "
                        "By default the deck stamped into the checkpoint is used; "
                        "only pass this if the checkpoint predates deck labelling.")
    args = p.parse_args()

    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    model_dir = work / "model"
    data_dir = work / "data"

    print("Building model package...")
    build_model_package(Path.cwd(), work)
    write_init(work / "model")
    print("  OK")

    # Resolve the deck BEFORE anything else — an unlabelled checkpoint or an
    # explicit override that disagrees with the label should abort the build
    # rather than produce a plausible-looking, silently-wrong bundle.
    deck, deck_record = read_ckpt_deck(Path(args.ckpt))
    if deck_record is not None:
        arch = deck_record.get("archetype_self")
        print(f"Checkpoint deck label: "
              f"{'archetype ' + str(arch) if arch is not None else 'all-decks (generalist)'}"
              f", {deck_record.get('n_distinct_cards')} distinct cards"
              f", vocab={deck_record.get('vocab_sha1')}")
    if args.deck_csv:
        override = [int(x) for x in Path(args.deck_csv).read_text().split()]
        if deck is not None and override != deck:
            print("  WARNING: --deck-csv disagrees with the checkpoint's own deck "
                  "label; using --deck-csv as instructed.")
        deck = override

    print("Building data files...")
    build_data_files(Path(args.data_dir), data_dir, deck=deck)
    build_model_weights(Path(args.ckpt), data_dir, deck_record=deck_record)
    print("  OK")

    print("Generating main.py...")
    build_main_py(work)
    print("  OK")

    print("Verifying packaged model imports...")
    if not verify_model_imports(work):
        raise SystemExit("Packaged model failed import verification — not packing.")
    print("  OK")

    print(f"Packing {args.out}...")
    pack_submission(work, Path(args.out))
    print(f"  Done: {args.out} ({Path(args.out).stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
