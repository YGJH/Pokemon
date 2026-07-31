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
import hashlib
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

# Additional Python files to bundle (from ptcg_il/ or ptcg_rl/)
EXTRA_FILES: list[tuple[str, str]] = [
    # (source_rel, dest_name_in_model)
    ("ptcg_il/search_infer.py", "search_infer.py"),
    ("ptcg_rl/belief.py", "belief_posterior.py"),
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


from model.cards import CardFeaturizer, AttackFeaturizer  # noqa: E402
from model.policy import Policy, select_multi  # noqa: E402
'''

MAIN_PY_TEMPLATE = r'''"""Pokémon TCG AI Agent — Kaggle submission entry point.

Uses belief model to predict opponent's deck, then PUCT MCTS (via bundled
libptcg_search.so) to search for the best action.  Falls back to greedy
policy if MCTS is unavailable.

Model loads at import time.  Any failure raises immediately.
"""

import json
import os
import numpy as np
import torch

try:
    from cg.api import to_observation_class
except ImportError:
    def to_observation_class(obs_dict: dict):
        select = obs_dict.get("select")
        return type("Observation", (), {
            "select": None if select is None else type("SelectData", (), select)(),
        })()

from model import Policy, select_multi
from model.featurizer import featurize
from model.search_infer import (
    mcts_search,
    predict_opponent_deck,
    extract_opp_visible_cards,
)

DATA_DIR = "/kaggle_simulations/agent/data" if os.path.exists("/kaggle_simulations/agent/") else "data"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── MCTS config (inference-time) ─────────────────────────────────────────

_MCTS_ITERATIONS = int(os.environ.get("MCTS_ITERATIONS", "64"))
_MCTS_C_PUCT = float(os.environ.get("MCTS_C_PUCT", "2.0"))
_MCTS_SEED = int(os.environ.get("MCTS_SEED", "0"))


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_deck_csv() -> list[int]:
    path = os.path.join(DATA_DIR, "deck.csv")
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/data/deck.csv"
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


# ── Import-time model loading ─────────────────────────────────────────────

_vocab_raw = _load_json(os.path.join(DATA_DIR, "vocab.json"))


def _index_to_id_map(raw):
    """Normalize ``index_to_id`` to ``{index: engine_card_id}``.

    ``vocab.json`` stores it as the list ``["PAD", "UNKNOWN", id2, ...]``, so
    ``.items()`` on it raises — and it would raise here at agent *import* time,
    which on Kaggle reads as a submission that simply does not start.  A dict
    form (int- or str-keyed) is accepted too; the PAD/UNKNOWN placeholders are
    dropped either way since they map to no engine card.
    """
    if not raw:
        return {}
    pairs = raw.items() if hasattr(raw, "items") else enumerate(raw)
    out = {}
    for k, v in pairs:
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            continue  # "PAD" / "UNKNOWN"
    return out


_vocab = {
    "id_to_index": {int(k): int(v) for k, v in _vocab_raw.get("id_to_index", {}).items()},
    "attack_id_to_index": {int(k): int(v) for k, v in _vocab_raw.get("attack_id_to_index", {}).items()},
    "index_to_id": _index_to_id_map(_vocab_raw.get("index_to_id")),
}

# Engine card/attack feature maps (built at packaging time from the bundled engine)
_engine_card_features = np.load(
    os.path.join(DATA_DIR, "engine_card_features.npy"), allow_pickle=True).item()
_engine_attack_features = np.load(
    os.path.join(DATA_DIR, "engine_attack_features.npy"), allow_pickle=True).item()

# All-card feature matrix for belief heads (sorted by card id)
_max_cid = max(_engine_card_features.keys()) if _engine_card_features else 0
_all_card_feat = torch.zeros(_max_cid + 1, 94)
for _cid, _feat in _engine_card_features.items():
    _all_card_feat[int(_cid)] = torch.from_numpy(np.asarray(_feat, dtype=np.float32))

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
    if _other:
        print(f"[agent] WARNING: {len(_other)} unexpected missing weights: {_other[:6]}")
_model.to(_device)
_model.eval()

_fixed_deck = _read_deck_csv()

# Load archetypes for belief posterior
_archetypes = _load_json(os.path.join(DATA_DIR, "archetypes.json"))

# Track visible opponent cards across the game (for belief posterior)
_opp_visible_cards: list[int] = []

# ── Agent function ───────────────────────────────────────────────────────


# Provide engine features to search_infer (it calls featurize internally)
import model.search_infer as _si
_si._engine_card_features = _engine_card_features
_si._engine_attack_features = _engine_attack_features


def agent(obs_dict: dict) -> list[int]:
    global _opp_visible_cards

    obs = to_observation_class(obs_dict)

    # Deck selection step
    if obs.select is None:
        _opp_visible_cards = []  # reset for new game
        return list(_fixed_deck)

    # Update visible opponent cards
    try:
        _opp_visible_cards = extract_opp_visible_cards(obs_dict)
    except Exception:
        pass

    # Predict opponent deck using belief model
    opp_deck = predict_opponent_deck(
        obs_dict, _model, _vocab, _archetypes, _device,
        observed_card_ids=_opp_visible_cards,
    )

    # Run MCTS via bundled libptcg_search.so
    result = mcts_search(
        obs_dict,
        fixed_deck=_fixed_deck,
        opp_deck=opp_deck,
        policy=_model,
        vocab=_vocab,
        device=_device,
        libcg_path=_find_libcg(),
        iterations=_MCTS_ITERATIONS,
        c_puct=_MCTS_C_PUCT,
        seed=_MCTS_SEED,
    )

    return result.get("indices", [])


def _libcg_name() -> str:
    """Platform-specific engine library name, matching cg/sim.py's own choice."""
    import platform
    os_name = platform.system()
    if os_name == "Windows":
        return "cg.dll"
    if os_name == "Darwin":
        return "libcg.dylib"
    if platform.machine() in ("arm64", "aarch64"):
        return "libcg-arm64.so"
    return "libcg.so"


def _find_libcg() -> str:
    """Locate the engine shared library.

    The Rust PUCT tree dlopens this itself, and ``puct_init`` returns NULL if
    the path is wrong — which ``mcts_search`` used to swallow, turning every
    decision into a greedy forward pass with no search at all.  So this has to
    actually find the file, not return a hopeful bare name.

    The most reliable source is the ``cg`` package itself: ``cg/sim.py``
    resolves the library next to its own ``__file__``, so if ``cg`` is
    importable at all, that directory holds the engine.
    """
    name = _libcg_name()
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        f"/kaggle_simulations/agent/{name}",
        name,
        os.path.join(here, name),
        os.path.join(here, "cg", name),
        os.path.join("cg", name),
    ]
    # The cg package knows where its own engine is — ask it first among dirs.
    try:
        import cg.sim as _cgsim
        candidates.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(_cgsim.__file__)), name))
    except Exception:
        pass
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    # Nothing found: return the bare name so ctypes can still try the loader
    # path, but say so — a silent miss here costs the entire search.
    print(f"[agent] WARNING: {name} not found; MCTS will fall back to greedy. "
          f"Looked in: {candidates}")
    return name
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

    Also bundles ``search_infer.py``, ``belief_posterior.py``, and
    ``libptcg_search.so`` for MCTS inference.

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

    # Extra files: search_infer.py, belief_posterior.py
    python_src = src_dir / "python"
    for rel_path, dest_name in EXTRA_FILES:
        src = python_src / rel_path
        if not src.exists():
            print(f"WARNING: {src} not found — skipping")
            continue
        text = src.read_text()
        # Rewrite belief_posterior imports (from ptcg_rl -> relative)
        if "ptcg_il" in text or "ptcg_rl" in text:
            text = text.replace("from ptcg_il.featurizer import", "from model.featurizer import")
            text = text.replace("from ptcg_il.model.policy import", "from model.policy import")
            text = text.replace("from ptcg_rl.belief import", "from model.belief_posterior import")
        (model_dst / dest_name).write_text(text)
        print(f"  Copied + rewrote {dest_name}")

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

def _infer_n_opp_arch(model_state: dict) -> int:
    """Recover n_opp_arch from the belief arch head's output width.

    Checkpoints written before `Policy.config` existed carry no record, and
    guessing 1 produces a bundle that raises on load.
    """
    for key, tensor in model_state.items():
        if key.endswith("belief_heads.arch_head.2.bias"):
            return int(tensor.shape[0])
    return 1


def check_artifact_pairing(deck_record: dict | None, data_dir: Path,
                           force: bool = False) -> None:
    """Abort if the checkpoint was trained against different artifacts.

    ``save_checkpoint`` pins ``vocab_sha1``/``archetypes_sha1`` precisely so
    this can be checked, but the builder only ever *printed* them.  The failure
    it guards is silent by construction: the model's card embedding is indexed
    by vocab position, so a vocab of the same length with a different id→index
    assignment produces a bundle that loads, imports, plays every game to the
    end, and loses nearly all of them.  Nothing raises at any point.

    A stale checkpoint from an earlier mining run is the normal way to hit
    this — archetype ids and vocab indices are both reassigned whenever mining
    re-runs.
    """
    if deck_record is None:
        return  # build_data_files already refuses an unlabelled checkpoint

    mismatches = []
    for key, fname in (("vocab_sha1", "vocab.json"),
                       ("archetypes_sha1", "archetypes.json")):
        pinned = deck_record.get(key)
        if not pinned:
            continue
        path = data_dir / fname
        if not path.exists():
            continue
        # ptcg_il.deck._sha1 truncates to 12 chars; compare on the pinned
        # length so a full digest and a truncated pin still agree.
        actual = hashlib.sha1(path.read_bytes()).hexdigest()
        if actual[:len(pinned)] != pinned:
            mismatches.append(f"    {fname}: checkpoint pins {pinned}, "
                              f"{data_dir}/{fname} is {actual[:len(pinned)]}")

    if not mismatches:
        return
    msg = ("Checkpoint was trained against different artifacts than the ones "
           "being packaged:\n" + "\n".join(mismatches) +
           "\n  The shipped vocab decides which embedding row each card id "
           "reads, so this bundle would play with scrambled card identities "
           "and lose almost every game — without erroring.\n"
           "  Point --ckpt at a checkpoint trained on this data/, or rebuild "
           "the corpus and retrain. Use --force to package anyway.")
    if force:
        print(f"WARNING: {msg}")
        return
    raise SystemExit(f"ERROR: {msg}")


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

    # Architecture comes from the checkpoint's own config record.
    cfg = ckpt.get("config") or {}
    arch = {
        "D": int(cfg.get("D", 256)),
        "heads": int(cfg.get("heads", 8)),
        "layers": int(cfg.get("layers", 4)),
        "ff": int(cfg.get("ff", 1024)),
        "n_opp_arch": int(cfg.get("n_opp_arch", _infer_n_opp_arch(model_state))),
        "n_all_cards": int(cfg.get("n_all_cards", 0)),
    }
    if cfg:
        print(f"  Architecture from checkpoint config: {arch}")
    else:
        print(f"  Checkpoint has no config record — assuming defaults {arch}")

    submission_pt = {"model_state_dict": model_state, "config": arch}
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


def _build_engine_features_from_engine(src_dir: Path, dst_dir: Path) -> None:
    """Build engine_card_features.npy and engine_attack_features.npy from the
    bundled engine's card/attack data.

    This ensures the feature tables match the exact engine version used in
    competition — cards added after training still get correct features.
    """
    import numpy as np
    sys.path.insert(0, str(src_dir / "python" / "pokemon-tcg-ai-battle"
                            / "sample_submission" / "sample_submission"))
    # ptcg_mine lives under python/, and this script is run from the repo root
    # (see scripts/build_submit.sh), so python/ has to be on the path too.
    sys.path.insert(0, str(src_dir / "python"))
    from cg.api import all_attack, all_card_data
    from ptcg_mine.cards import card_static_row, attack_static_row

    cards, attacks = all_card_data(), all_attack()
    # card.attacks holds attack *ids*, so the row builder needs the lookup to
    # fill the 52:94 attack half; without it those 42 dims would be all zeros.
    attacks_by_id = {a.attackId: a for a in attacks}
    card_feats = {c.cardId: card_static_row(c, attacks_by_id) for c in cards}
    attack_feats = {a.attackId: attack_static_row(a) for a in attacks}
    np.save(dst_dir / "engine_card_features.npy", card_feats)
    np.save(dst_dir / "engine_attack_features.npy", attack_feats)
    print(f"  Built engine_card_features.npy ({len(card_feats)} cards)")
    print(f"  Built engine_attack_features.npy ({len(attack_feats)} attacks)")


def build_data_files(data_dir: Path, dst_dir: Path, deck: list[int] | None = None,
                     src_dir: Path | None = None) -> None:
    """Copy vocab, archetypes, deck, and build engine features into submission/data/.

    Engine card/attack features are built from the bundled engine at packaging
    time, so they always match the competition engine — even if new cards were
    added after training.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``vocab.json`` and ``archetypes.json``.
    dst_dir : Path
        Output directory for the submission data files.
    src_dir : Path or None
        Project root.  Defaults to cwd.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Vocab (still needed for index_to_id reverse mapping in featurizer)
    shutil.copy(data_dir / "vocab.json", dst_dir / "vocab.json")
    print(f"  Copied vocab.json")

    # Engine card/attack features — built fresh from the bundled engine
    if src_dir is None:
        src_dir = Path.cwd()
    _build_engine_features_from_engine(src_dir, dst_dir)

    # Archetypes — needed by belief posterior for opponent deck prediction
    arch_path = data_dir / "archetypes.json"
    if arch_path.exists():
        shutil.copy(arch_path, dst_dir / "archetypes.json")
        print(f"  Copied archetypes.json")

    # Rust MCTS library
    if src_dir is None:
        src_dir = Path.cwd()
    so_candidates = [
        src_dir / "python" / "ptcg_search" / "target" / "release" / "libptcg_search.so",
        src_dir / "ptcg_search" / "target" / "release" / "libptcg_search.so",
    ]
    bundled = False
    for so_path in so_candidates:
        if so_path.exists():
            shutil.copy(so_path, dst_dir / "libptcg_search.so")
            size_kb = so_path.stat().st_size / 1024
            print(f"  Copied libptcg_search.so ({size_kb:.0f} KB)")
            bundled = True
            break
    if not bundled:
        print(f"  WARNING: libptcg_search.so not found — MCTS will fall back to greedy policy")

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
    # Import main.py, not just the model package: main.py is where Policy is
    # actually constructed and model.pt loaded, and that is where the bundle
    # breaks.  Verifying only `from model import ...` passed a submission whose
    # agent raised at import (wrong n_opp_arch => belief-head shape mismatch,
    # fatal even under strict=False) and therefore forfeited every game.
    result = subprocess.run(
        [sys.executable, "-c",
         "import main; assert callable(main.agent); print('OK')"],
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
    p.add_argument("--force", action="store_true",
                   help="Package even when the checkpoint's pinned vocab/archetypes "
                        "SHAs disagree with --data-dir (produces a silently broken bundle)")
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

    check_artifact_pairing(deck_record, Path(args.data_dir), force=args.force)

    print("Building data files...")
    build_data_files(Path(args.data_dir), data_dir, deck=deck, src_dir=Path.cwd())
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
