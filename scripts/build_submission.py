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
    (r"from ptcg_il\.model\.policy import", r"from model.policy import"),
    # ensemble.py imports (lives in ptcg_il/, bundled as model/ensemble.py)
    (r"from ptcg_il\.ensemble import", r"from model.ensemble import"),
    # Blanket "from ptcg_il.model import X" rule (must come after submodule rules
    # so it doesn't prematurely match the submodule prefixes)
    (r"from ptcg_il\.model import", r"from model import"),
    # ref_map (copied into model/ so modules import it as model.ref_map)
    (r"from ptcg_il\.ref_map import", r"from model.ref_map import"),
    # featurizer (copied into model/ so modules import it as model.featurizer).
    # model/{cards,embed,pointer}.py import the feature dims from it.
    (r"from ptcg_il\.featurizer import", r"from model.featurizer import"),
    # The module-object form of the same import.  `policy.current_feature_dims`
    # reads the widths off the module rather than naming each one, and it runs
    # at `Policy.__init__` — so missing this rule is not a latent bug, it is
    # every packaged agent failing to construct.
    (r"from ptcg_il import featurizer", r"from model import featurizer"),
    # deck_prior (copied into model/ so modules import it as model.deck_prior).
    # Both forms are listed deliberately: the module-object form has been
    # missed before, and a missed rule fails only at agent runtime on Kaggle.
    (r"from ptcg_il\.deck_prior import", r"from model.deck_prior import"),
    (r"from ptcg_il import deck_prior", r"from model import deck_prior"),
]

MODEL_FILES: list[str] = [
    "cards.py",
    "embed.py",
    "encoder.py",
    "pointer.py",
    "value.py",
    "policy.py",
    "ensemble.py",
]

# Additional Python files to bundle (from ptcg_il/)
EXTRA_FILES: list[tuple[str, str]] = [
    # (source_rel, dest_name_in_model)
    ("ptcg_il/search_infer.py", "search_infer.py"),
    ("ptcg_il/deck_prior.py", "deck_prior.py"),
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

Uses the mined archetype prior to predict the opponent's deck, then PUCT MCTS
(via bundled libptcg_search.so) to search for the best action.  Falls back to
greedy policy if MCTS is unavailable.

The opponent model is a Bayesian posterior over every archetype in
archetypes.json, seeded by mined frequency and sharpened by elimination.  No
learned belief head is involved: the packaged weights still carry them, they
are simply never called.

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
from model.search_infer import mcts_search, decision_time_budget
from model.deck_prior import OpponentDeckPredictor

DATA_DIR = "/kaggle_simulations/agent/data" if os.path.exists("/kaggle_simulations/agent/") else "data"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The Kaggle runner execs this file into a namespace with no ``__file__``, so
# anything reaching for it raises NameError at *decision* time — invisible to a
# build-time `import main` smoke test, which does bind ``__file__``.  Resolve
# the agent directory once, here, and never touch ``__file__`` again.
_AGENT_DIR = (
    "/kaggle_simulations/agent" if os.path.exists("/kaggle_simulations/agent/")
    else os.path.dirname(os.path.abspath(globals()["__file__"]))
    if "__file__" in globals() else os.getcwd()
)

# ── MCTS config (inference-time) ─────────────────────────────────────────

# The iteration count is now a *ceiling*, not the operating point: the search
# stops on a wall-clock deadline derived from `remainingOverageTime` (see
# `search_infer.decision_time_budget`).  It is set high enough that time is
# what binds on fast hardware, and harmless on slow hardware, where the
# deadline fires first.  A count low enough to bind (it was 16) pins the agent
# to whichever CPU it was measured on.
_MCTS_ITERATIONS = int("512")
_MCTS_C_PUCT = float("2.0")
_MCTS_SEED = int("0")

# Decisions played this game, for the time budget.  Reset at deck selection,
# which is the only signal a new game has started.
_decisions_made = 0


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
# Feeds hand_feat[3] (`can_evolve`), which is constant 0 without it.  Optional
# so a bundle built before it was packaged still runs.
_evolution_map_path = os.path.join(DATA_DIR, "evolution_map.npy")
_evolution_map = (np.load(_evolution_map_path, allow_pickle=True).item()
                  if os.path.exists(_evolution_map_path) else None)

# Detect ensemble: an ensemble.json manifest means multiple checkpoints were
# packaged as model_0.pt, model_1.pt, ...  Load the config from member 0
# (or model.pt for single-model) to resolve feature dimensions.
_ENSEMBLE_MANIFEST = os.path.join(DATA_DIR, "ensemble.json")
if os.path.exists(_ENSEMBLE_MANIFEST):
    _manifest = json.loads(open(_ENSEMBLE_MANIFEST).read())
    _member0 = torch.load(os.path.join(DATA_DIR, _manifest["members"][0]),
                          map_location=_device, weights_only=True)
    _cfg = _member0.get("config", {})
else:
    _ckpt = torch.load(os.path.join(DATA_DIR, "model.pt"), map_location=_device, weights_only=True)
    _cfg = _ckpt.get("config", {})

# All-card feature matrix for belief heads (sorted by card id).
#
# The width is F_CARD, which moved 94 -> 212 when the keyword features landed.
# Take it from the checkpoint, like every other dim below: the tables are
# rebuilt from the live engine at packaging time and the model was trained at
# whatever F_CARD was then, so a literal here is the one value that cannot
# track either.  Checkpoints older than the `feat_dims` pin carry no width, so
# fall back to the table's own rows.
_max_cid = max(_engine_card_features.keys()) if _engine_card_features else 0
_table_dim = int(np.asarray(next(iter(_engine_card_features.values()))).shape[-1]
                 ) if _engine_card_features else 0
_card_feat_dim = int(_cfg.get("feat_dims", {}).get("F_CARD", _table_dim))
if _engine_card_features and _card_feat_dim != _table_dim:
    raise SystemExit(
        f"card feature width mismatch: model.pt was trained at F_CARD="
        f"{_card_feat_dim}, engine_card_features.npy has {_table_dim}-wide rows. "
        "Rebuild the submission against the checkpoint's own featurizer.")
_all_card_feat = torch.zeros(_max_cid + 1, _card_feat_dim)
for _cid, _feat in _engine_card_features.items():
    _all_card_feat[int(_cid)] = torch.from_numpy(np.asarray(_feat, dtype=np.float32))

# Same table for attacks, and it is not optional.  `featurize` emits attack
# *ids* (`opt_attack_idx`) and `Policy._gather_card_feats` turns them back into
# features from this table on every forward, so a policy without it constructs,
# loads every weight and then raises on the first real decision.
_max_aid = max(_engine_attack_features.keys()) if _engine_attack_features else 0
_attack_row_dim = int(np.asarray(next(iter(_engine_attack_features.values()))).shape[-1]
                     ) if _engine_attack_features else 0
_attack_feat_dim = int(_cfg.get("feat_dims", {}).get("F_ATK", _attack_row_dim))
if _engine_attack_features and _attack_feat_dim != _attack_row_dim:
    raise SystemExit(
        f"attack feature width mismatch: the checkpoint was trained at F_ATK="
        f"{_attack_feat_dim}, engine_attack_features.npy has {_attack_row_dim}-wide "
        "rows. Rebuild the submission against the checkpoint's own featurizer.")
_all_attack_feat = torch.zeros(_max_aid + 1, _attack_feat_dim)
for _aid, _feat in _engine_attack_features.items():
    _all_attack_feat[int(_aid)] = torch.from_numpy(np.asarray(_feat, dtype=np.float32))

# ── Model construction (single or ensemble) ──────────────────────────────
if os.path.exists(_ENSEMBLE_MANIFEST):
    from model.ensemble import EnsemblePolicy
    _member_paths = [os.path.join(DATA_DIR, m) for m in _manifest["members"]]
    _model = EnsemblePolicy.from_checkpoints(
        _member_paths, _all_card_feat, _all_attack_feat, device=_device)
else:
    _model = Policy(
        D=_cfg.get("D", 256), heads=_cfg.get("heads", 8),
        layers=_cfg.get("layers", 4), ff=_cfg.get("ff", 1024),
        n_all_cards=_cfg.get("n_all_cards", _max_cid + 1),
        all_card_feat=_all_card_feat,
        all_attack_feat=_all_attack_feat,
    )
    _missing, _unexpected = _model.load_state_dict(_ckpt["model_state_dict"], strict=False)
    if _missing:
        _belief_keys = [k for k in _missing if k.startswith("belief_heads.")]
        _other = [k for k in _missing if not k.startswith("belief_heads.")]
        # Policy ties one CardEncoder into three places (`policy.py`:
        # `self.pointer.card = self.embed.card`, same for the belief heads), and
        # EMA's shadow de-duplicates shared parameters, so those aliases are absent
        # from the packaged state dict while the tensors they name are loaded
        # through `embed.card.*`.  Reporting them as missing cried wolf on every
        # single run, which is how a real gap would have gone unnoticed.  Compare
        # object identity rather than guessing at name prefixes.
        # remove_duplicate=False is the whole point: the default de-duplicates
        # shared parameters, so the alias names are absent and would be misread as
        # genuinely missing — the exact false alarm this is here to stop.
        _params = dict(_model.named_parameters(remove_duplicate=False))
        _params.update(dict(_model.named_buffers(remove_duplicate=False)))
        _loaded_ids = {id(_params[k]) for k in _ckpt["model_state_dict"] if k in _params}
        _other = [k for k in _other
                  if k not in _params or id(_params[k]) not in _loaded_ids]
        if _other:
            print(f"[agent] WARNING: {len(_other)} unexpected missing weights: {_other[:6]}")
    _model.to(_device)
    _model.eval()

_fixed_deck = _read_deck_csv()

# Load archetypes and build the opponent model once.  This is the entire
# opponent model now, so archetypes.json is load-bearing: a bundle whose
# archetypes.json does not match the checkpoint's archetypes_sha1 is refused
# at packaging time, not here.
_archetypes = _load_json(os.path.join(DATA_DIR, "archetypes.json"))
_opp_predictor = OpponentDeckPredictor(_archetypes)

# ── Agent function ───────────────────────────────────────────────────────


# Provide engine features to search_infer (it calls featurize internally)
import model.search_infer as _si
_si._engine_card_features = _engine_card_features
_si._engine_attack_features = _engine_attack_features
_si._evolution_map = _evolution_map


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

    The bundle carries its own copy in ``data/``, next to
    ``libptcg_search.so``, and that is the only source that exists on Kaggle:
    the agent process there has no importable ``cg`` package, so the branch
    below never fires and every other candidate is a local-development path.
    Looking only at those is what shipped a submission that played greedy for
    every decision of every game.

    The ``cg`` package, when there is one, still knows where its own engine is:
    ``cg/sim.py`` resolves the library next to its own ``__file__``.
    """
    name = _libcg_name()
    here = _AGENT_DIR
    candidates = [
        # Bundled by build_submission.build_data_files — mirrors the search
        # library's own lookup order in search_infer._load_search_lib.
        f"/kaggle_simulations/agent/data/{name}",
        os.path.join(here, "data", name),
        os.path.join("data", name),
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


# `agent` must be the LAST callable bound at module level: the Kaggle runner
# takes `[v for v in env.values() if callable(v)][-1]`, by position and not by
# name, so anything defined below this point is what it calls instead.  That
# shipped once — `_find_libcg` above used to live here, and every episode died
# at step 0 with "Player 1's deck does not have 60 cards" because a path string
# is not a 60-card deck.  Re-binding `agent = agent` at the end does not fix it;
# a dict keeps a re-bound key in its original position.
def agent(obs_dict: dict) -> list[int]:
    global _decisions_made
    obs = to_observation_class(obs_dict)

    # Deck selection step — a new game starts here.
    if obs.select is None:
        _opp_predictor.reset()
        _decisions_made = 0
        return list(_fixed_deck)

    # Fold this observation into the opponent model, then read out one
    # decklist for the determinizer.  `template()` never abstains: the mirror
    # heuristic it would otherwise fall back to shares 19.4 of their 60 cards,
    # against 44.6 for an argmax with no evidence at all.
    _opp_predictor.observe(obs_dict)
    opp_deck = _opp_predictor.template()

    # How long this move may take.  Read from the observation every decision:
    # the bank is shared across the whole game and a short game leaves more for
    # each remaining move.  Absent on a non-Kaggle caller, which means no
    # deadline and the iteration ceiling applies instead.
    _budget = decision_time_budget(
        obs_dict.get("remainingOverageTime"), _decisions_made)
    _decisions_made += 1

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
        time_budget_s=_budget,
    )

    return result.get("indices", [])
'''


MAIN_PY_TEMPLATE_GREEDY = r'''"""Pokémon TCG AI Agent — Kaggle submission entry point (no-MCTS build).

One policy forward pass per decision: featurize -> encode -> pointer logits ->
argmax over the legal options (autoregressive greedy for multi-select).  There
is no tree search, no engine rollout, and no ``libptcg_search.so`` — nothing
here dlopens anything, so the only way this agent fails is a genuine model or
data problem, not a missing native library.

The belief heads are still present in the packaged weights; they are simply
never called, because their only consumer was the MCTS determinizer.

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
from model.deck_guard import DeckGuard, GuardConfig

DATA_DIR = "/kaggle_simulations/agent/data" if os.path.exists("/kaggle_simulations/agent/") else "data"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Feeds hand_feat[3] (`can_evolve`), which is constant 0 without it.  Optional
# so a bundle built before it was packaged still runs.
_evolution_map_path = os.path.join(DATA_DIR, "evolution_map.npy")
_evolution_map = (np.load(_evolution_map_path, allow_pickle=True).item()
                  if os.path.exists(_evolution_map_path) else None)

# All-card feature matrix.  The belief heads own it, and Policy builds them
# unconditionally, so it is still required to construct the module and load the
# packaged state dict — even though this build never runs a belief head.
#
# The width is F_CARD, which moved 94 -> 212 when the keyword features landed.
# Take it from the checkpoint, like every other dim below: the tables are
# rebuilt from the live engine at packaging time and the model was trained at
# whatever F_CARD was then, so a literal here is the one value that cannot
# track either.  Checkpoints older than the `feat_dims` pin carry no width, so
# fall back to the table's own rows.
_max_cid = max(_engine_card_features.keys()) if _engine_card_features else 0
_table_dim = int(np.asarray(next(iter(_engine_card_features.values()))).shape[-1]
                 ) if _engine_card_features else 0

# Read F_CARD from the first available source: ensemble member 0, or model.pt.
_ENSEMBLE_MANIFEST = os.path.join(DATA_DIR, "ensemble.json")
if os.path.exists(_ENSEMBLE_MANIFEST):
    _manifest = json.loads(open(_ENSEMBLE_MANIFEST).read())
    _member0 = torch.load(os.path.join(DATA_DIR, _manifest["members"][0]),
                          map_location=_device, weights_only=True)
    _cfg = _member0.get("config", {})
else:
    _ckpt = torch.load(os.path.join(DATA_DIR, "model.pt"), map_location=_device, weights_only=True)
    _cfg = _ckpt.get("config", {})

_card_feat_dim = int(_cfg.get("feat_dims", {}).get("F_CARD", _table_dim))
if _engine_card_features and _card_feat_dim != _table_dim:
    raise SystemExit(
        f"card feature width mismatch: model.pt was trained at F_CARD="
        f"{_card_feat_dim}, engine_card_features.npy has {_table_dim}-wide rows. "
        "Rebuild the submission against the checkpoint's own featurizer.")
_all_card_feat = torch.zeros(_max_cid + 1, _card_feat_dim)
for _cid, _feat in _engine_card_features.items():
    _all_card_feat[int(_cid)] = torch.from_numpy(np.asarray(_feat, dtype=np.float32))

# Same table for attacks, and it is not optional.  `featurize` emits attack
# *ids* (`opt_attack_idx`) and `Policy._gather_card_feats` turns them back into
# features from this table on every forward, so a policy without it constructs,
# loads every weight and then raises on the first real decision — which `agent`
# catches, warns about, and answers with the first legal option for the rest of
# the game.
_max_aid = max(_engine_attack_features.keys()) if _engine_attack_features else 0
_attack_row_dim = int(np.asarray(next(iter(_engine_attack_features.values()))).shape[-1]
                     ) if _engine_attack_features else 0
_attack_feat_dim = int(_cfg.get("feat_dims", {}).get("F_ATK", _attack_row_dim))
if _engine_attack_features and _attack_feat_dim != _attack_row_dim:
    raise SystemExit(
        f"attack feature width mismatch: the checkpoint was trained at F_ATK="
        f"{_attack_feat_dim}, engine_attack_features.npy has {_attack_row_dim}-wide "
        "rows. Rebuild the submission against the checkpoint's own featurizer.")
_all_attack_feat = torch.zeros(_max_aid + 1, _attack_feat_dim)
for _aid, _feat in _engine_attack_features.items():
    _all_attack_feat[int(_aid)] = torch.from_numpy(np.asarray(_feat, dtype=np.float32))

# ── Ensemble detection ────────────────────────────────────────────────────
if os.path.exists(_ENSEMBLE_MANIFEST):
    from model.ensemble import EnsemblePolicy
    _manifest = json.loads(open(_ENSEMBLE_MANIFEST).read())
    _member_paths = [os.path.join(DATA_DIR, m) for m in _manifest["members"]]
    _model = EnsemblePolicy.from_checkpoints(
        _member_paths, _all_card_feat, _all_attack_feat, device=_device)
else:
    _model = Policy(
        D=_cfg.get("D", 256), heads=_cfg.get("heads", 8),
        layers=_cfg.get("layers", 4), ff=_cfg.get("ff", 1024),
        n_all_cards=_cfg.get("n_all_cards", _max_cid + 1),
        all_card_feat=_all_card_feat,
        all_attack_feat=_all_attack_feat,
    )
    _missing, _unexpected = _model.load_state_dict(_ckpt["model_state_dict"], strict=False)
    if _missing:
        _belief_keys = [k for k in _missing if k.startswith("belief_heads.")]
        _other = [k for k in _missing if not k.startswith("belief_heads.")]
        # Policy ties one CardEncoder into three places (`policy.py`:
        # `self.pointer.card = self.embed.card`, same for the belief heads), and
        # EMA's shadow de-duplicates shared parameters, so those aliases are absent
        # from the packaged state dict while the tensors they name are loaded
        # through `embed.card.*`.  Reporting them as missing cried wolf on every
        # single run, which is how a real gap would have gone unnoticed.  Compare
        # object identity rather than guessing at name prefixes.
        # remove_duplicate=False is the whole point: the default de-duplicates
        # shared parameters, so the alias names are absent and would be misread as
        # genuinely missing — the exact false alarm this is here to stop.
        _params = dict(_model.named_parameters(remove_duplicate=False))
        _params.update(dict(_model.named_buffers(remove_duplicate=False)))
        _loaded_ids = {id(_params[k]) for k in _ckpt["model_state_dict"] if k in _params}
        _other = [k for k in _other
                  if k not in _params or id(_params[k]) not in _loaded_ids]
        if _other:
            print(f"[agent] WARNING: {len(_other)} unexpected missing weights: {_other[:6]}")
    _model.to(_device)
    _model.eval()

_fixed_deck = _read_deck_csv()

# Anti-deck-out guard: mechanism A hard-masks options that provably empty the
# deck before the next-turn draw (prize-winning KOs exempt); mechanism B
# re-ranks near-ties by net deck delta when the deck is low and the hand is
# large.  Thresholds come from GuardConfig's corpus-measured defaults.
_guard = DeckGuard(GuardConfig(), _engine_card_features)


# ── Inference helpers ────────────────────────────────────────────────────

def _to_batch(feats: dict) -> dict:
    """Convert single-sample featurizer output to batch-1 torch tensors."""
    batch = {}
    for k, v in feats.items():
        # np.int64(3) is an np.generic *scalar*, not an ndarray, so an
        # isinstance(v, np.ndarray) filter silently drops every 0-d key the
        # featurizer emits: minCount, maxCount, stop_column, sel_type, sel_ctx,
        # action_len, log_len, value_target, sample_weight.  `select_multi`
        # reads minCount/maxCount, so multi-select decisions would die with
        # KeyError: 'minCount' while single-select ones — which never touch
        # those keys — went through fine.
        if not isinstance(v, (np.ndarray, np.generic)):
            continue
        # np.asarray keeps a scalar 0-d so unsqueeze(0) yields [B]; going via
        # ascontiguousarray would promote it to 1-d and give [B, 1], which
        # broadcasts wrongly inside _select_multi_raw instead of failing.
        arr = np.asarray(v)
        if arr.ndim:
            arr = np.ascontiguousarray(arr)
        t = torch.from_numpy(arr).unsqueeze(0)
        if v.dtype == np.bool_:
            t = t.bool()
        elif np.issubdtype(v.dtype, np.integer):
            t = t.long()
        else:
            t = t.float()
        batch[k] = t.to(device=_device)
    return batch


def _legal(indices: list[int], n_options: int, min_count: int, max_count: int) -> list[int]:
    """Force the engine's Select contract: distinct, in range, minCount<=k<=maxCount.

    Engine error codes 4 (count out of range), 5 (index OOB) and 6 (duplicate)
    all forfeit the game, so the return value is clamped here rather than
    trusted.  ``maxCount == 0`` legitimately means "select nothing", so this
    must not floor the count at 1.
    """
    seen, out = set(), []
    for i in indices:
        i = int(i)
        if 0 <= i < n_options and i not in seen:
            seen.add(i)
            out.append(i)
    for i in range(n_options):          # top up toward minCount
        if len(out) >= min_count:
            break
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:max_count]


# ── Agent function ───────────────────────────────────────────────────────

def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)

    # Deck selection step
    if obs.select is None:
        return list(_fixed_deck)

    select = obs_dict.get("select") or {}
    n_options = len(select.get("option") or [])
    min_count = int(select.get("minCount", 1) or 0)
    max_count = int(select.get("maxCount", 1) or 0)

    # Raw state for the deck guard — straight from the observation, no
    # featurizer involvement: my deck, my hand size, my prizes remaining.
    _cur = obs_dict.get("current") or {}
    _players = _cur.get("players") or [{}]
    _me = _players[int(_cur.get("yourIndex", 0))]
    _deck = int(_me.get("deckCount", 0) or 0)
    _hand = int(_me.get("handCount", 0) or 0)
    _prizes = len(_me.get("prize") or [])

    # Deliberately unguarded.  A featurizer or shape failure here is not a bad
    # position, it is a bundle that cannot play at all: a static table that was
    # never attached, a renamed key, a checkpoint whose widths moved.  The
    # handler that used to catch it printed one line and answered every
    # subsequent decision with the first `minCount` legal options, which is
    # indistinguishable downstream from a policy that simply plays badly — so a
    # dead agent survived whole experiments and its win rate was read as a
    # result.  Failing the run is the cheaper outcome by a wide margin.
    feats = featurize(
        obs_dict, _vocab,
        engine_card_features=_engine_card_features,
        engine_attack_features=_engine_attack_features,
        evolution_map=_evolution_map,
    )
    batch = _to_batch(feats)
    feat_max_count = int(feats.get("maxCount", max_count))
    _sel_type = int(feats.get("sel_type", select.get("type", 0)) or 0)
    # Mechanism A: guaranteed deck-out options become mask-illegal here, so
    # both the single-select masked_fill below and select_multi's cloned
    # picked-mask honor them.  In-place, zero inference overhead.
    _guard.apply_mask(batch, sel_type=_sel_type,
                      deck=_deck, hand=_hand, prizes=_prizes)

    with torch.no_grad():
        if feat_max_count == 1:
            logits, _value, _hist = _model(batch)
            # Mechanism B is inside pick(): near-tie re-ranking by deck delta
            # under (deck low, hand large); plain masked argmax otherwise.
            indices = [_guard.pick(logits, batch, sel_type=_sel_type,
                                   max_count=feat_max_count,
                                   deck=_deck, hand=_hand, prizes=_prizes)]
        else:
            chosen = select_multi(_model, batch)
            # -1 pads beyond maxCount, -2 marks the STOP pick.
            indices = [int(p) for p in chosen[0].tolist() if p >= 0]

    # Still clamped: this guards the model's *output* (a duplicate, an
    # out-of-range index, more picks than maxCount), which is a legal answer to
    # give badly, not a broken bundle.
    return _legal(indices, n_options, min_count, max_count)
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


def build_model_package(src_dir: Path, dst_dir: Path, mcts: bool = True) -> None:
    """Copy model files from ptcg_il/model/ to submission/model/, rewriting imports.

    Also bundles ``search_infer.py``, ``deck_prior.py``, and
    ``libptcg_search.so`` for MCTS inference.

    Parameters
    ----------
    src_dir : Path
        Project root (contains ``python/ptcg_il/``).
    dst_dir : Path
        Submission root (``submission/``).  Model files land in
        ``dst_dir / "model" /``.
    mcts : bool
        When False, ``search_infer.py`` and ``deck_prior.py`` are left
        out: the greedy ``main.py`` imports neither, and shipping the MCTS
        module in a bundle with no ``libptcg_search.so`` only invites a
        fallback path that reads as working search.
    """
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

    # Extra files: search_infer.py, deck_prior.py
    python_src = src_dir / "python"
    for rel_path, dest_name in (EXTRA_FILES if mcts else []):
        src = python_src / rel_path
        if not src.exists():
            print(f"WARNING: {src} not found — skipping")
            continue
        text = src.read_text()
        text = rewrite_imports(text)
        (model_dst / dest_name).write_text(text)
        print(f"  Copied + rewrote {dest_name}")

    # ref_map.py — self-contained, no ptcg_il imports, copy verbatim
    ref_src = src_dir / "python" / "ptcg_il" / "ref_map.py"
    if ref_src.exists():
        shutil.copy(ref_src, model_dst / "ref_map.py")
        print(f"  Copied ref_map.py (verbatim)")
    else:
        print(f"WARNING: {ref_src} not found — skipping")

    # deck_guard.py — self-contained (numpy + stdlib only), copy verbatim.
    # Ships in BOTH builds (not an EXTRA_FILES member): it is part of the
    # greedy agent's inference path, not search-only machinery.
    guard_src = src_dir / "python" / "ptcg_il" / "deck_guard.py"
    if guard_src.exists():
        shutil.copy(guard_src, model_dst / "deck_guard.py")
        print(f"  Copied deck_guard.py (verbatim)")
    else:
        print(f"WARNING: {guard_src} not found — skipping")

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


def build_main_py(dst_dir: Path, mcts: bool = True) -> None:
    """Generate submission/main.py from the appropriate template."""
    (dst_dir / "main.py").write_text(
        MAIN_PY_TEMPLATE if mcts else MAIN_PY_TEMPLATE_GREEDY)
    print(f"  Wrote main.py ({'MCTS' if mcts else 'greedy, no search'})")


# ============================================================
# Section 2: Model weights — extract EMA weights from checkpoint
# ============================================================



# What each pinned artifact actually controls at inference.  These are not
# interchangeable, and the guard used to describe both as one fatal error.
#
# vocab.json — governs which engine ids count as in-vocab, and nothing else.
#   `CardFeaturizer` is a pure MLP over 94 static features with *no learned id
#   embeddings* (`model/cards.py`), and `featurize` passes ``index_to_id=None``
#   so card ids stay raw engine ids resolved against `engine_card_features.npy`
#   — which this builder regenerates from the bundled engine anyway.  A vocab
#   index therefore never selects an embedding row.  The old message claimed it
#   did and promised "scrambled card identities"; that failure mode belongs to
#   an architecture this repo does not have.
#
# archetypes.json — genuinely load-bearing, and only for MCTS builds.
#   `BeliefHead.arch_head` is `Linear(D, n_arch)` whose output position *k*
#   means "cluster k of the archetypes.json seen during training".  Cluster ids
#   are reassigned on every mining run, so a mismatch silently points the
#   opponent-deck posterior at the wrong decklists and feeds the MCTS
#   determinizer bad decks.  `--no-mcts` bundles neither ship this file nor
#   call the belief heads, so there it is inert.
_PIN_CONSEQUENCE: dict[str, str] = {
    "vocab.json":
        "decides only which ids are in-vocab; card representation is pure "
        "static features (no learned id embeddings), so this alone is mostly "
        "cosmetic — but it does prove checkpoint and data/ came from "
        "different mining runs",
    "archetypes.json":
        "sets what each belief arch-head output position MEANS. Cluster ids "
        "are reassigned every mining run, so the opponent-deck posterior "
        "would name the wrong decks and mislead the MCTS determinizer. "
        "Inert in a --no-mcts build, which ships no archetypes.json",
}

# Which build types actually dereference each artifact.  Severity is derived
# from this rather than asserted, so the guard cannot drift from what the
# bundle really reads.
#
# vocab.json is in *neither* set: `featurize`'s own signature documents the
# parameter as "retained for API compatibility ... card and attack identity no
# longer routes through it", and `_raw_card`/`_raw_attack` pass engine ids
# straight through to `engine_card_features.npy` — which this builder
# regenerates from the bundled engine at packaging time.  A vocab mismatch is
# therefore always evidence, never a defect, and always warns.
_PIN_CONSUMED_BY: dict[str, frozenset[str]] = {
    "vocab.json": frozenset(),
    "archetypes.json": frozenset({"mcts"}),
}


def check_artifact_pairing(deck_record: dict | None, data_dir: Path,
                           force: bool = False, mcts: bool = True) -> None:
    """Warn — or abort — when the checkpoint was trained against other artifacts.

    ``save_checkpoint`` pins ``vocab_sha1``/``archetypes_sha1`` precisely so
    this can be checked, but the builder only ever *printed* them.  A stale
    checkpoint from an earlier mining run is the normal way to hit this.

    A mismatch aborts only when the packaged bundle actually *reads* the
    artifact in question (``_PIN_CONSUMED_BY``).  A ``--no-mcts`` bundle ships
    no ``archetypes.json`` and never calls the belief heads, and no build type
    routes card identity through the vocab, so refusing to package those was
    blocking correct bundles over pairings nothing dereferences.  Everything
    still *reports*, because a mismatch remains real evidence that checkpoint
    and ``data/`` came from different mining runs.
    """
    if deck_record is None:
        return  # build_data_files already refuses an unlabelled checkpoint

    build = "mcts" if mcts else "greedy"
    mismatches, fatal = [], []
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
            consumed = build in _PIN_CONSUMED_BY[fname]
            mismatches.append(
                f"    {fname}: checkpoint pins {pinned}, {data_dir}/{fname} "
                f"is {actual[:len(pinned)]}\n"
                f"      -> {_PIN_CONSEQUENCE[fname]}\n"
                f"      -> this {build} build "
                + ("READS this file — fatal." if consumed
                   else "does not read this file."))
            if consumed:
                fatal.append(fname)

    if not mismatches:
        return

    msg = ("Checkpoint was trained against different artifacts than the ones "
           "being packaged:\n" + "\n".join(mismatches))
    if fatal and not force:
        raise SystemExit(
            f"ERROR: {msg}\n"
            f"  {', '.join(fatal)} is consumed by this build, so the bundle "
            f"would be wrong. Point --ckpt at a checkpoint trained on this "
            f"data/, or rebuild the corpus and retrain. "
            f"Use --force to package anyway.")
    print(f"WARNING: {msg}\n"
          f"  Nothing this {build} build reads is affected — packaging anyway.")


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
    arch = {
        "D": int(cfg.get("D", 256)),
        "heads": int(cfg.get("heads", 8)),
        "layers": int(cfg.get("layers", 4)),
        "ff": int(cfg.get("ff", 1024)),
        "n_all_cards": int(cfg.get("n_all_cards", 0)),
    }

    out = {"model_state_dict": model_state, "config": arch}
    if deck_record is not None:
        out["deck"] = deck_record
    torch.save(out, dst_dir / f"model_{index}.pt")
    print(f"  Wrote model_{index}.pt ({len(model_state)} parameter tensors)")


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


# The vendored engine, relative to the repo root.  Both the feature builder
# (which imports `cg.api` from it) and the bundler (which copies the shared
# library out of it) read this one place, so a moved checkout breaks loudly in
# both rather than silently in one.
ENGINE_DIR_PARTS = ("python", "pokemon-tcg-ai-battle", "sample_submission",
                    "sample_submission", "cg")


def _build_engine_features_from_engine(src_dir: Path, dst_dir: Path) -> None:
    """Build engine_card_features.npy and engine_attack_features.npy from the
    bundled engine's card/attack data.

    This ensures the feature tables match the exact engine version used in
    competition — cards added after training still get correct features.
    """
    import numpy as np
    sys.path.insert(0, str(src_dir.joinpath(*ENGINE_DIR_PARTS).parent))
    # ptcg_mine lives under python/, and this script is run from the repo root
    # (see scripts/build_submit.sh), so python/ has to be on the path too.
    sys.path.insert(0, str(src_dir / "python"))
    from cg.api import all_attack, all_card_data
    from ptcg_mine.cards import build_evolution_map, card_static_row, attack_static_row

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
    # Built here rather than copied from data/ for the same reason as the two
    # above: it must describe the *competition* engine's evolution lines, not
    # whatever the corpus was mined against.  `hand_feat[3]` (`can_evolve`) is
    # constant 0 without it.
    evo = build_evolution_map(cards)
    np.save(dst_dir / "evolution_map.npy", evo)
    print(f"  Built evolution_map.npy ({len(evo)} evolution lines)")


def build_data_files(data_dir: Path, dst_dir: Path, deck: list[int] | None = None,
                     src_dir: Path | None = None, mcts: bool = True) -> None:
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
    mcts : bool
        When False, ``archetypes.json`` and ``libptcg_search.so`` are not
        bundled — the greedy agent reads neither.  The archetype sanity check
        below still runs, against ``data_dir``'s copy rather than the bundle's.
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
    if mcts and arch_path.exists():
        shutil.copy(arch_path, dst_dir / "archetypes.json")
        print(f"  Copied archetypes.json")

    # Rust MCTS library
    if mcts:
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

        # The engine itself.  libptcg_search.so dlopens libcg.so by the path
        # `main.py`'s `_find_libcg()` hands `puct_init`, and the Kaggle agent
        # process has no importable `cg` package to borrow one from — so an
        # engine that is not *in the bundle* means `puct_init` returns null and
        # every decision degrades to a greedy forward pass, reported only as a
        # line in the competition's stdout.  Bundling it locally is invisible:
        # the repo path this copies from exists here.
        #
        # Always `libcg.so`: the target is Kaggle's Linux x86-64 runner, so the
        # name is fixed regardless of what this build happens to run on.
        engine_candidates = [
            src_dir.joinpath(*ENGINE_DIR_PARTS) / "libcg.so",
            src_dir / "cg" / "libcg.so",
        ]
        for eng_path in engine_candidates:
            if eng_path.exists():
                shutil.copy(eng_path, dst_dir / "libcg.so")
                size_kb = eng_path.stat().st_size / 1024
                print(f"  Copied libcg.so ({size_kb:.0f} KB)")
                break
        else:
            # Fatal, unlike libptcg_search.so above: that one is missing
            # whenever cargo is, but the engine is vendored in the repo, so its
            # absence means the build is wrong. Warning and continuing is
            # exactly what shipped a greedy submission labelled as MCTS.
            raise FileNotFoundError(
                "libcg.so not found — an MCTS bundle cannot run search without "
                "the engine, and the agent would silently play greedy. Looked "
                f"in: {[str(p) for p in engine_candidates]}")
    else:
        print(f"  Skipped archetypes.json + libptcg_search.so + libcg.so (no-MCTS build)")

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
    # agent raised at import (architecture mismatch, fatal even under
    # strict=False) and therefore forfeited every game.
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
    p.add_argument("--ckpt", action="append", required=True,
                   help="Path to checkpoint .pt file. Pass multiple times for ensemble.")
    p.add_argument("--out", default="submission.tar.gz", help="Output tar.gz path")
    p.add_argument("--force", action="store_true",
                   help="Package even when the checkpoint's pinned vocab/archetypes "
                        "SHAs disagree with --data-dir (produces a silently broken bundle)")
    p.add_argument("--work-dir", default="/tmp/submission-build", help="Temp build directory")
    p.add_argument("--deck-csv", default=None,
                   help="Override the deck with this csv (one card id per line). "
                        "By default the deck stamped into the checkpoint is used; "
                        "only pass this if the checkpoint predates deck labelling.")
    p.add_argument("--no-mcts", action="store_true",
                   help="Build a pure-policy bundle: one greedy forward pass per "
                        "decision, no tree search. Drops search_infer.py, "
                        "deck_prior.py, archetypes.json and libptcg_search.so.")
    args = p.parse_args()
    ckpt_paths = args.ckpt  # list[str] (action="append")
    is_ensemble = len(ckpt_paths) > 1
    mcts = not args.no_mcts
    if is_ensemble:
        print(f"Ensemble mode: {len(ckpt_paths)} checkpoints"
              f"{' (MCTS)' if mcts else ' (greedy)'}")

    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    model_dir = work / "model"
    data_dir = work / "data"

    print(f"Building model package ({'MCTS' if mcts else 'greedy, no search'})...")
    build_model_package(Path.cwd(), work, mcts=mcts)
    write_init(work / "model")
    print("  OK")

    if is_ensemble:
        # ── Ensemble: resolve deck from first member, verify all members share it ──
        deck, deck_record = read_ckpt_deck(Path(ckpt_paths[0]))
        if deck_record is not None:
            arch = deck_record.get("archetype_self")
            print(f"Ensemble deck label (member 0): "
                  f"{'archetype ' + str(arch) if arch is not None else 'all-decks (generalist)'}"
                  f", {deck_record.get('n_distinct_cards')} distinct cards"
                  f", vocab={deck_record.get('vocab_sha1')}")

        for i, ckpt_p in enumerate(ckpt_paths[1:], start=1):
            member_deck, member_record = read_ckpt_deck(Path(ckpt_p))
            if member_record is None:
                print(f"WARNING: Member {i} checkpoint has no deck label — "
                      f"cannot verify deck match")
                continue
            if (deck_record is not None
                    and member_record.get("deck") != deck_record.get("deck")):
                raise SystemExit(
                    f"ERROR: Member {i} decklist differs from member 0. "
                    f"Ensembling models trained on different decks fails silently."
                )
            if (deck_record is not None
                    and member_record.get("vocab_sha1") != deck_record.get("vocab_sha1")):
                print(f"WARNING: Member {i} vocab_sha1 differs from member 0")

        if args.deck_csv:
            override = [int(x) for x in Path(args.deck_csv).read_text().split()]
            if deck is not None and override != deck:
                print("  WARNING: --deck-csv disagrees with member 0's deck "
                      "label; using --deck-csv as instructed.")
            deck = override

        check_artifact_pairing(deck_record, Path(args.data_dir), force=args.force,
                               mcts=mcts)

        print("Building data files...")
        build_data_files(Path(args.data_dir), data_dir, deck=deck, src_dir=Path.cwd(),
                         mcts=mcts)

        # Write per-member weight files
        for i, ckpt_p in enumerate(ckpt_paths):
            member_deck, member_record = read_ckpt_deck(Path(ckpt_p))
            _build_member_weights(Path(ckpt_p), data_dir, i, deck_record=member_record)

        # Write ensemble.json manifest
        manifest = {"members": [f"model_{i}.pt" for i in range(len(ckpt_paths))]}
        (data_dir / "ensemble.json").write_text(json.dumps(manifest, indent=2))
        print(f"  Wrote ensemble.json ({len(ckpt_paths)} members)")
        print("  OK")

        # Auto-suffix --out for ensemble
        out_path = Path(args.out)
        if args.out == "submission.tar.gz":
            prefix = "submission" if mcts else "submission-greedy"
            out_path = Path(f"{prefix}-ens{len(ckpt_paths)}.tar.gz")
            print(f"Ensemble default output: {out_path}")
    else:
        # ── Single checkpoint: original flow, byte-identical ──
        ckpt_path = ckpt_paths[0]
        deck, deck_record = read_ckpt_deck(Path(ckpt_path))
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

        check_artifact_pairing(deck_record, Path(args.data_dir), force=args.force,
                               mcts=mcts)

        print("Building data files...")
        build_data_files(Path(args.data_dir), data_dir, deck=deck, src_dir=Path.cwd(),
                         mcts=mcts)
        build_model_weights(Path(ckpt_path), data_dir, deck_record=deck_record)
        print("  OK")
        out_path = Path(args.out)

    print("Generating main.py...")
    build_main_py(work, mcts=mcts)
    print("  OK")

    print("Verifying packaged model imports...")
    if not verify_model_imports(work):
        raise SystemExit("Packaged model failed import verification — not packing.")
    print("  OK")

    print(f"Packing {out_path}...")
    pack_submission(work, out_path)
    print(f"  Done: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
