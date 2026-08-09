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

# Set by the agent at import time to provide engine card/attack features
# for the pure-feature featurizer.  `_evolution_map` feeds hand_feat[3]
# (`can_evolve`), which is constant 0 without it.
_engine_card_features: dict | None = None
_engine_attack_features: dict | None = None
_evolution_map: dict | None = None

# ── Rust library loading ──────────────────────────────────────────────────


_LIB: ctypes.CDLL | None = None
_BASICS_REGISTERED = False


def _register_basic_pokemon(lib: ctypes.CDLL) -> None:
    """Tell the Rust determinizer which card ids are Basic Pokémon.

    Only a Basic can legally be the opponent's face-down active.  Without this
    the guess is drawn from the whole predicted deck — where only ~16% of slots
    are Basic — and ``SearchBegin`` refuses the root with error 2, costing that
    decision its search and dropping the agent to a greedy forward pass.

    Deferred rather than done at load time: the agent assigns
    ``_engine_card_features`` at import, and this runs on the first decision, so
    it cannot race that assignment.  Best-effort — a failure here only returns
    the determinizer to its previous behaviour.
    """
    global _BASICS_REGISTERED
    if _BASICS_REGISTERED or not _engine_card_features:
        return
    try:
        from model.featurizer import CARD_FEAT_BASIC_COL
    except ImportError:  # running from the repo, not the bundle
        from ptcg_il.featurizer import CARD_FEAT_BASIC_COL
    try:
        ids = sorted(
            int(cid) for cid, row in _engine_card_features.items()
            if len(row) > CARD_FEAT_BASIC_COL and row[CARD_FEAT_BASIC_COL] > 0.5
        )
        if not ids:
            return
        n = lib.puct_set_basic_pokemon(json.dumps(ids).encode("utf-8"))
        _BASICS_REGISTERED = n >= 0
        # Printed on purpose: whether this ran is the first question asked when
        # "SearchBegin error code 2" shows up, and a bundle carrying an old
        # search_infer.py against a new .so is silent about it otherwise.
        print(f"[agent] MCTS determinizer: {n} Basic Pokémon ids registered")
    except AttributeError:
        print("[agent] WARNING: libptcg_search.so has no puct_set_basic_pokemon "
              "— rebuild it (cargo build --release) or the determinizer will "
              "guess the opponent's face-down active from the whole deck")
    except Exception as exc:  # noqa: BLE001 — never fail a decision over this
        print(f"[agent] WARNING: could not register Basic-Pokémon set ({exc})")


def _load_search_lib() -> ctypes.CDLL:
    """Find and load ``libptcg_search.so`` from the submission data dir.

    Cached: this used to re-resolve and re-register argtypes on every decision.
    """
    global _LIB
    if _LIB is not None:
        return _LIB
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

            # Optional: an older .so without it still works, just with the
            # whole-template active guess.
            try:
                lib.puct_set_basic_pokemon.argtypes = [ctypes.c_char_p]
                lib.puct_set_basic_pokemon.restype = ctypes.c_int
            except AttributeError:
                pass

            _LIB = lib
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

        feats = featurize(obs_dict, vocab,
                          engine_card_features=_engine_card_features,
                          engine_attack_features=_engine_attack_features,
                          evolution_map=_evolution_map)
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
        # `deck` is [n_all_cards] and indexed by **engine card id**, not by
        # vocab index: `belief_labels.deck_counts_dense` writes `out[cid]` for
        # a raw engine id, and `cli` sizes the head from the engine feature
        # matrix.  Routing it through `_index_to_id(vocab)` (≈311 entries
        # against a 1268-wide head) both relabelled every card and silently
        # dropped every id past the end of the vocab, so this tier returned a
        # short deck of the wrong cards — inside a bare `except Exception`,
        # with the empty-list mirror fallback right below it to absorb the
        # damage.  Positions are already engine ids; there is no hop to make.
        deck_logits = belief_logits.get("deck")
        if deck_logits is not None:
            probs = deck_logits[0].float().cpu().numpy()
            probs[0] = 0.0  # id 0 is "no card"; 1 is a real card, keep it
            total = probs.sum()
            if total > 0:
                probs = probs / total
                return _deck_from_distribution(probs, None, rng=rng)

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


_FALLBACK_WARNED: set[str] = set()
_FALLBACK_COUNTS: dict[str, int] = {}


def _warn_fallback(reason: str) -> None:
    """Report a degradation to greedy, with how often it has happened.

    ``mcts_search`` used to swallow every failure, so a run with no search at
    all was indistinguishable from a working one: the agent still returned
    legal moves, just from a single forward pass.

    Printing strictly once per reason overcorrected — a single transient
    rejection (one refused search root out of hundreds) produced a line that
    reads as though search never runs at all.  Re-reporting on a widening
    interval distinguishes "happened twice" from "happens every turn" without
    one line per decision.
    """
    n = _FALLBACK_COUNTS.get(reason, 0) + 1
    _FALLBACK_COUNTS[reason] = n
    _FALLBACK_WARNED.add(reason)
    if n == 1 or n in (10, 100) or n % 500 == 0:
        suffix = "" if n == 1 else f" — {n} times so far"
        print(f"[agent] WARNING: MCTS fell back to greedy ({reason}){suffix}")


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
    # The Rust tree applies one option index per node (`search_step(id, &[a])`),
    # so root visit counts rank the *first* pick only.  That is a complete
    # action for single-select, but a multi-select decision needs k indices in
    # one reply and the tree cannot express that subset — taking its top-k
    # would report a choice the search never actually evaluated (and may not
    # even be a legal co-selection).  Those go to the policy's autoregressive
    # `select_multi`, which is built for it.
    _sel = obs_dict.get("select") or {}
    try:
        _max_count = int(_sel.get("maxCount", 1) or 1)
    except (TypeError, ValueError):
        _max_count = 1
    if _max_count > 1:
        return _greedy_action(obs_dict, policy, vocab, device)

    try:
        lib = _load_search_lib()
        _register_basic_pokemon(lib)

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

            # Evaluate leaf with policy network.  The Rust side names this
            # field `leaf_obs_json` (see ptcg_search/src/lib.rs puct_select);
            # reading `obs_json` raised KeyError on the very first iteration,
            # which the old blanket except turned into a silent greedy move.
            priors, value = _policy_evaluate_leaf(
                leaf["leaf_obs_json"], leaf["n_options"],
                leaf["is_terminal"], policy, vocab, device,
                player_role=int(leaf.get("player_role", 0)),
            )

            lib.puct_expand(
                handle,
                json.dumps(priors).encode("utf-8"),
                float(value),
            )

        raw = _read_and_free(lib, lib.puct_result(handle))
        lib.puct_free(handle)

        if raw:
            res = json.loads(raw)
            # puct_result reports {"visit_counts": [[option_index, visits], ...]}
            # — it has no "indices" key, so returning it verbatim gave the agent
            # an empty action and the engine rejected it with IndexError.  The
            # move is the most-visited root child, which is what PUCT's visit
            # distribution is for.
            counts = res.get("visit_counts") or []
            if counts:
                best = max(counts, key=lambda pair: (pair[1], -pair[0]))[0]
                res["indices"] = [int(best)]
                return res
            _warn_fallback("puct_result returned no visit counts")
        else:
            _warn_fallback("puct_result returned no data")
    except Exception as exc:
        _warn_fallback(f"{type(exc).__name__}: {exc}")

    # Fallback: greedy policy
    return _greedy_action(obs_dict, policy, vocab, device)


def _policy_evaluate_leaf(
    obs_json: str,
    n_options: int,
    is_terminal: bool,
    policy: Any,
    vocab: dict,
    device: Any,
    player_role: int = 0,
) -> tuple[list[float], float]:
    """Featurize a leaf observation and run one policy forward pass.

    Returns ``(priors, value)`` with **value in the perspective of the player to
    move at this leaf** — i.e. exactly what the network natively predicts
    (``value_target = +1`` when the player whose observation this is won, see
    ``shard_writer``).

    The tree re-orients it: ``expand_leaf`` negates when ``player_role == 1``.
    That is the only place it can correctly happen — the network's input is
    egocentric, so "this is an opponent node" is not expressible to it.

    Priors need no re-orientation: they score the options belonging to whoever
    moves at that leaf, which is exactly the node's action set.
    """
    from model.featurizer import featurize

    if is_terminal or n_options == 0:
        return ([], 0.0)

    obs_dict = json.loads(obs_json)
    feats = featurize(obs_dict, vocab,
                      engine_card_features=_engine_card_features,
                      engine_attack_features=_engine_attack_features,
                      evolution_map=_evolution_map)
    batch = _dict_to_batch(feats, device)

    # Mirror Policy.forward: _encode returns (h, history_h), and PointerHead
    # needs (h, tok_mask, card_enc, x) and returns (logits, o).  The old call
    # `policy.pointer(h, batch["opt_mask"])` predated the pure-feature pointer
    # and raised TypeError on every leaf.
    with _no_grad():
        h, _history = policy._encode(batch)
        logits, _o = policy.pointer(h, batch["tok_mask"], policy.embed.card, batch)
        value = policy.value(h[:, 0])          # [B]

    mask = batch["opt_mask"][0].cpu().numpy()
    l = logits[0].float().cpu().numpy()
    l = np.where(mask, l, -np.inf)
    l = l - l.max()
    probs = np.exp(l)
    total = probs.sum()
    probs = probs / total if total > 0 else np.full_like(probs, 1.0 / max(len(probs), 1))

    # puct_expand expects exactly n_options priors — the mask can additionally
    # cover a STOP column on multi-select, which the tree knows nothing about.
    priors = probs[:n_options].tolist() if n_options > 0 else []
    val = float(value.reshape(-1)[0].cpu().numpy())
    # No flip here: `expand_leaf` in ptcg_search re-orients the value using the
    # node's player_role.  Negating here as well would double-negate and
    # restore the original inverted backup.
    return priors, val


def _greedy_action(
    obs_dict: dict, policy: Any, vocab: dict, device: Any
) -> dict:
    """Fallback: greedy single/multi-select without MCTS."""
    from model.featurizer import featurize
    from model.policy import decode_single_select, select_multi

    feats = featurize(obs_dict, vocab,
                      engine_card_features=_engine_card_features,
                      engine_attack_features=_engine_attack_features,
                      evolution_map=_evolution_map)
    batch = _dict_to_batch(feats, device)
    max_count = int(feats.get("maxCount", 1))

    with _no_grad():
        if max_count == 1:
            logits, _value, _hist = policy(batch)
            # STOP means decline; its column sits past the real options, so
            # returning it would be engine error 5 rather than "take nothing".
            indices = decode_single_select(
                logits, batch["opt_mask"], batch.get("stop_column"),
            )
        else:
            chosen = select_multi(policy, batch)
            # Truncate at the first STOP rather than filtering it out — picks
            # after it come from a stale mask and a stale msgru.
            indices = []
            for p in chosen[0].tolist():
                if p == -2:
                    break
                if p >= 0:
                    indices.append(int(p))
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
        # np.int64(3) is an np.generic *scalar*, not an ndarray, so an
        # isinstance(v, np.ndarray) filter silently drops every 0-d key the
        # featurizer emits: minCount, maxCount, stop_column, sel_type, sel_ctx,
        # action_len, log_len, value_target, sample_weight.  `select_multi`
        # reads minCount/maxCount, so multi-select decisions died with
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
    index_to_id: list[int] | None,
    deck_size: int = 60,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Sample *deck_size* cards from a card distribution.

    *index_to_id* maps position → engine card id.  Pass ``None`` when *probs*
    is already indexed by engine card id — the same convention
    ``featurizer._ids_to_feat`` uses, and what the belief ``deck`` head emits.
    """
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
        if index_to_id is None:
            cid = int(idx)
        elif idx >= len(index_to_id):
            continue
        else:
            cid = index_to_id[idx]
        if cid < 0:
            continue
        deck.extend([cid] * int(counts[idx]))
    return deck
