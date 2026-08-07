"""Pokémon TCG AI Agent — Kaggle submission entry point (no-MCTS build).

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

_ckpt = torch.load(os.path.join(DATA_DIR, "model.pt"), map_location=_device, weights_only=True)
_cfg = _ckpt.get("config", {})

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
_card_feat_dim = int(_cfg.get("feat_dims", {}).get("F_CARD", _table_dim))
if _engine_card_features and _card_feat_dim != _table_dim:
    raise SystemExit(
        f"card feature width mismatch: model.pt was trained at F_CARD="
        f"{_card_feat_dim}, engine_card_features.npy has {_table_dim}-wide rows. "
        "Rebuild the submission against the checkpoint's own featurizer.")
_all_card_feat = torch.zeros(_max_cid + 1, _card_feat_dim)
for _cid, _feat in _engine_card_features.items():
    _all_card_feat[int(_cid)] = torch.from_numpy(np.asarray(_feat, dtype=np.float32))

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

    try:
        feats = featurize(
            obs_dict, _vocab,
            engine_card_features=_engine_card_features,
            engine_attack_features=_engine_attack_features,
        )
        batch = _to_batch(feats)
        feat_max_count = int(feats.get("maxCount", max_count))

        with torch.no_grad():
            if feat_max_count == 1:
                logits, _value, _hist = _model(batch)
                logits = logits.masked_fill(~batch["opt_mask"], -1e9)
                indices = [int(logits.argmax(dim=-1)[0].item())]
            else:
                chosen = select_multi(_model, batch)
                # -1 pads beyond maxCount, -2 marks the STOP pick.
                indices = [int(p) for p in chosen[0].tolist() if p >= 0]
    except Exception as e:
        # A featurizer or shape failure must not forfeit the game outright;
        # the first minCount legal options at least keep play going.
        print(f"[agent] WARNING: greedy inference failed ({type(e).__name__}: {e})"
              f" — falling back to the first {min_count} option(s)")
        indices = []

    return _legal(indices, n_options, min_count, max_count)
