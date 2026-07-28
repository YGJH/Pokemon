"""Inference-only PUCT MCTS search for Kaggle submission agent.

Bundles with ``libptcg_search.so`` and ``cg.api`` (provided by Kaggle).
Lightweight — no torch, no training deps — just numpy + ctypes.
"""

from __future__ import annotations

import ctypes
import json
import os
from typing import Any

import numpy as np

# ── Rust library loading ──────────────────────────────────────────────────


def _load_search_lib() -> ctypes.CDLL:
    """Find and load ``libptcg_search.so`` from the submission data dir."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "libptcg_search.so"),
        os.path.join("/kaggle_simulations/agent/data", "libptcg_search.so"),
        os.path.join("data", "libptcg_search.so"),
    ]
    for p in candidates:
        if os.path.exists(p):
            lib = ctypes.CDLL(p)
            # Register argtypes
            lib.puct_init.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_int, ctypes.c_double,
                ctypes.c_int, ctypes.c_int,
            ]
            lib.puct_init.restype = ctypes.c_int64

            lib.puct_select.argtypes = [ctypes.c_int64]
            lib.puct_select.restype = ctypes.c_void_p

            lib.puct_expand.argtypes = [
                ctypes.c_int64, ctypes.c_char_p, ctypes.c_double,
            ]
            lib.puct_expand.restype = ctypes.c_int

            lib.puct_result.argtypes = [ctypes.c_int64]
            lib.puct_result.restype = ctypes.c_void_p

            lib.puct_free.argtypes = [ctypes.c_int64]
            lib.puct_free.restype = None

            lib.puct_free_result.argtypes = [ctypes.c_void_p]
            lib.puct_free_result.restype = None

            lib.search_plan.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            lib.search_plan.restype = ctypes.c_void_p
            lib.search_plan_free.argtypes = [ctypes.c_void_p]
            lib.search_plan_free.restype = None

            return lib

    raise FileNotFoundError(
        "libptcg_search.so not found in submission data/. "
        "Build with: cd python/ptcg_search && cargo build --release"
    )


# ── Helpers ───────────────────────────────────────────────────────────────


def _read_and_free(lib: ctypes.CDLL, ptr) -> str:
    if ptr is None or ptr == 0:
        return ""
    try:
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
    finally:
        lib.puct_free_result(ptr)


# ── Belief-based opponent deck prediction ─────────────────────────────────


def predict_opponent_deck(
    obs_dict: dict,
    policy: Any,       # model.Policy (with belief heads)
    vocab: dict,       # normalized vocab
    archetypes: dict,  # parsed archetypes.json
    device: Any,       # torch device
    observed_card_ids: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Predict opponent's 60-card decklist using belief model.

    Fallback chain:
    1. Learned arch head (confident → representative decklist)
    2. ArchetypePosterior (elimination via observed cards)
    3. Card distribution head (bag-of-cards sampling)
    4. Empty list (Rust mirror fallback)

    Uses the same ``OpponentDeckOracle`` logic but inline for bundling.
    """
    try:
        from model.featurizer import featurize

        feats = featurize(obs_dict, vocab)
        batch = _dict_to_batch(feats, device)

        with _no_grad():
            belief_logits = policy.belief_logits(batch)

        arch_logits = belief_logits.get("arch")
        reps = _representatives(archetypes)

        # ── Tier 1: learned arch head ────────────────────────────────
        if arch_logits is not None and arch_logits.shape[-1] > 1 and reps:
            arch = arch_logits[0].cpu().numpy()
            arch = arch - arch.max()
            p = np.exp(arch[:len(reps)])
            p = p / p.sum()
            best = int(p.argmax())
            if p[best] >= 0.5 and len(reps[best]) == 60:
                return list(reps[best])

        # ── Tier 2: Bayesian posterior ────────────────────────────────
        if observed_card_ids and reps:
            try:
                from model.belief_posterior import ArchetypePosterior
                posterior = ArchetypePosterior(archetypes)
                probs = posterior.posterior(observed_card_ids)
                best = int(np.argmax(probs))
                if probs[best] >= 0.80 and best < len(reps):
                    rep = reps[best]
                    if len(rep) == 60:
                        return list(rep)
            except Exception:
                pass

        # ── Tier 3: card distribution ─────────────────────────────────
        deck_logits = belief_logits.get("deck")
        if deck_logits is not None:
            probs = deck_logits[0].float().cpu().numpy()
            probs[:2] = 0.0  # mask PAD, UNKNOWN
            total = probs.sum()
            if total > 0:
                probs = probs / total
                index_to_id = _index_to_id(vocab)
                return _deck_from_distribution(probs, index_to_id, rng=rng)

    except Exception:
        pass

    return []  # Rust mirror fallback


def extract_opp_visible_cards(obs_dict: dict) -> list[int]:
    """Extract visible opponent card ids from observation."""
    current = obs_dict.get("current", {})
    your_idx = current.get("yourIndex", 0)
    opp_idx = 1 - your_idx
    players = current.get("players", [])
    if opp_idx >= len(players):
        return []
    opp = players[opp_idx]
    ids = []

    for poke in (opp.get("active") or []):
        if isinstance(poke, dict) and "id" in poke:
            ids.append(poke["id"])
            for pre in (poke.get("preEvolution") or []):
                if isinstance(pre, dict) and "id" in pre:
                    ids.append(pre["id"])

    for poke in (opp.get("bench") or []):
        if isinstance(poke, dict) and "id" in poke:
            ids.append(poke["id"])
            for pre in (poke.get("preEvolution") or []):
                if isinstance(pre, dict) and "id" in pre:
                    ids.append(pre["id"])

    for card in (opp.get("discard") or []):
        if isinstance(card, dict) and "id" in card:
            ids.append(card["id"])

    for card in (opp.get("prize") or []):
        if isinstance(card, dict) and "id" in card:
            ids.append(card["id"])

    for card in (current.get("stadium") or []):
        if isinstance(card, dict) and "id" in card:
            ids.append(card["id"])

    return ids


# ── MCTS inference ────────────────────────────────────────────────────────


def mcts_search(
    obs_dict: dict,
    fixed_deck: list[int],
    opp_deck: list[int],
    policy: Any,
    vocab: dict,
    device: Any,
    libcg_path: str = "libcg.so",
    iterations: int = 64,
    c_puct: float = 2.0,
    seed: int = 0,
) -> dict:
    """Run PUCT MCTS via Rust libptcg_search.so for one decision point.

    Returns ``{"indices": [...], "visit_counts": ..., "root_value": ...}``.
    Falls back to greedy policy if the Rust library is unavailable.
    """
    try:
        lib = _load_search_lib()

        obs_json = json.dumps(obs_dict)
        fixed_json = json.dumps(fixed_deck)
        opp_json = json.dumps(opp_deck if opp_deck else [])

        # Single-tree PUCT via the C-ABI
        handle = lib.puct_init(
            obs_json.encode("utf-8"),
            libcg_path.encode("utf-8"),
            fixed_json.encode("utf-8"),
            opp_json.encode("utf-8"),
            iterations,
            c_puct,
            seed,
            1,  # host_initialized: Python always calls GameInitialize first
        )

        if handle == 0:
            raise RuntimeError("puct_init returned null")

        # MCTS loop
        while True:
            raw = _read_and_free(lib, lib.puct_select(handle))
            if not raw:
                break
            leaf = json.loads(raw)
            if leaf.get("error") or leaf.get("tree_done"):
                break

            # Evaluate leaf with policy network
            priors, value = _policy_evaluate_leaf(
                leaf["obs_json"], leaf["n_options"],
                leaf["is_terminal"], policy, vocab, device,
            )

            lib.puct_expand(
                handle,
                json.dumps(priors).encode("utf-8"),
                float(value),
            )

        raw = _read_and_free(lib, lib.puct_result(handle))
        lib.puct_free(handle)

        if raw:
            return json.loads(raw)
    except Exception:
        pass

    # Fallback: greedy policy
    return _greedy_action(obs_dict, policy, vocab, device)


def _policy_evaluate_leaf(
    obs_json: str,
    n_options: int,
    is_terminal: bool,
    policy: Any,
    vocab: dict,
    device: Any,
) -> tuple[list[float], float]:
    """Featurize a leaf observation and run one policy forward pass.

    Returns (priors, value).
    """
    from model.featurizer import featurize

    if is_terminal or n_options == 0:
        return ([], 0.0)

    obs_dict = json.loads(obs_json)
    feats = featurize(obs_dict, vocab)
    batch = _dict_to_batch(feats, device)

    with _no_grad():
        h = policy._encode(batch)
        logits = policy.pointer(h, batch["opt_mask"])
        value = policy.value(h[:, 0, :])

    mask = batch["opt_mask"][0].cpu().numpy()
    l = logits[0].float().cpu().numpy()
    l = np.where(mask, l, -np.inf)
    l = l - l.max()
    probs = np.exp(l)
    probs = probs / probs.sum()

    n_legal = int(mask.sum())
    priors = probs[:n_legal].tolist() if n_legal > 0 else []
    val = float(value[0, 0].cpu().numpy())

    return priors, val


def _greedy_action(
    obs_dict: dict, policy: Any, vocab: dict, device: Any
) -> dict:
    """Fallback: greedy single/multi-select without MCTS."""
    from model.featurizer import featurize
    from model.policy import select_multi

    feats = featurize(obs_dict, vocab)
    batch = _dict_to_batch(feats, device)
    max_count = int(feats.get("maxCount", 1))

    with _no_grad():
        if max_count == 1:
            logits, _value, _hist = policy(batch)
            logits = logits.masked_fill(~batch["opt_mask"], -1e9)
            indices = [int(logits.argmax(dim=-1)[0].item())]
        else:
            chosen = select_multi(policy, batch)
            indices = [int(p) for p in chosen[0].tolist() if p >= 0]
            indices = indices[:max_count]

    return {"indices": indices, "visit_counts": [], "root_value": None}


# ── Internal helpers ──────────────────────────────────────────────────────


class _no_grad:
    """``torch.no_grad`` without importing torch at module level."""

    def __enter__(self):
        import torch
        self._ctx = torch.no_grad()
        return self._ctx.__enter__()

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def _dict_to_batch(feats: dict, device: Any) -> dict:
    """Convert single-sample featurizer output to batch-1 torch tensors."""
    import torch

    batch = {}
    for k, v in feats.items():
        if not isinstance(v, np.ndarray):
            continue
        t = torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0)
        if v.dtype == np.bool_:
            t = t.bool()
        elif np.issubdtype(v.dtype, np.integer):
            t = t.long()
        else:
            t = t.float()
        batch[k] = t.to(device)
    return batch


def _representatives(archetypes: dict) -> list[list[int]]:
    """Representative decklists for each opp archetype, in opp_ids order."""
    by_id = {int(a["id"]): a for a in archetypes.get("archetypes", [])}
    out = []
    for gid in archetypes.get("opp_ids", []):
        rep = by_id.get(int(gid), {}).get("representative") or []
        out.append([int(c) for c in rep])
    return out


def _index_to_id(vocab: dict) -> list[int]:
    """vocab index → engine card id."""
    index_to_id = vocab.get("index_to_id")
    if not index_to_id:
        size = int(vocab.get("size", 0))
        index_to_id = [-1] * size
        for cid, idx in vocab.get("id_to_index", {}).items():
            if 0 <= int(idx) < size:
                index_to_id[int(idx)] = int(cid)
    return [int(c) if int(c) >= 0 else -1 for c in index_to_id]


def _deck_from_distribution(
    probs: np.ndarray,
    index_to_id: list[int],
    deck_size: int = 60,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Sample deck_size cards from a vocab distribution."""
    if rng is not None:
        counts = rng.multinomial(deck_size, probs)
    else:
        exact = probs * deck_size
        counts = np.floor(exact).astype(np.int64)
        short = deck_size - int(counts.sum())
        if short > 0:
            order = np.argsort(-(exact - counts))
            counts[order[:short]] += 1

    deck = []
    for idx in np.nonzero(counts)[0]:
        if idx >= len(index_to_id):
            continue
        cid = index_to_id[idx]
        if cid < 0:
            continue
        deck.extend([cid] * int(counts[idx]))
    return deck
