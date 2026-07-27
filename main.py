"""Pokémon TCG AI Agent — Kaggle submission entry point.

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

DATA_DIR = "/kaggle_simulations/agent/data" if os.path.exists("/kaggle_simulations/agent/") else "data"
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
# n_opp_arch sizes the belief arch head.  Omitting it built the head at [1, D]
# against a [n_opp_arch, D] checkpoint, and a *shape* mismatch is fatal even
# under strict=False — so the agent raised at import and forfeited every game.
_model = Policy(
    V=_ckpt["V"], A=_ckpt["A"],
    D=_ckpt.get("D", 256), heads=_ckpt.get("heads", 8),
    layers=_ckpt.get("layers", 4), ff=_ckpt.get("ff", 1024),
    n_opp_arch=_ckpt.get("n_opp_arch", 1),
    card_static_table=_card_static, attack_static_table=_attack_static,
)
_missing, _unexpected = _model.load_state_dict(_ckpt["model_state_dict"], strict=False)
# Loud about what strict=False swallowed: silently dropped tensors mean an
# agent that plays with partly random weights and simply loses.
if _missing:
    print(f"[agent] WARNING: {len(_missing)} tensors missing from checkpoint: {_missing[:6]}")
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
