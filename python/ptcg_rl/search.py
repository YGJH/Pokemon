"""PUCT MCTS search — Python-driven, Rust-accelerated (RL_SPEC §8, Phase 3a+).

The Rust ``libptcg_search.so`` provides per-tree C-ABI functions
(``puct_init`` / ``puct_select`` / ``puct_expand`` / ``puct_result`` /
``puct_free``).  This module wraps them and adds the Python-side loop:
featurize → batch GPU forward → apply priors/values.

**Design C (RL_SPEC §8.3):** MCTS is training-only, for offline distillation
targets.  Acting uses ``π_θ`` with greedy sampling; search runs on a small
fraction (ρ=0.05) of rollout decision points.

Phase 3a provides single-tree PUCT.  Phase 3b adds batched multi-tree search.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Replay Buffer (AlphaZero-style, Phase 3+) ────────────────────────────


class ReplayBuffer:
    """Fixed-capacity replay buffer for MCTS self-play data.

    Stores (features, action_idx, action_len, mcts_pi, mcts_value,
    game_outcome, opp_archetype) tuples.  Uniform random sampling for
    training.

    This decouples data collection (self-play + MCTS) from training,
    allowing the network to be trained on a diverse mix of old and new
    data — the key stabiliser in AlphaZero-style pipelines.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._buffer: list[dict] = []
        self._pos = 0  # overwrite position for circular buffer

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.capacity

    def push(
        self,
        features: dict[str, np.ndarray],
        action_idx: np.ndarray,
        action_len: int,
        mcts_pi: np.ndarray | None,
        mcts_value: float | None,
        game_outcome: float,
        opp_archetype: int | None = None,
    ) -> None:
        """Store one decision point.

        Parameters
        ----------
        features : dict
            Featurized tensors for this state.
        action_idx : np.ndarray
            AR action targets including STOP, -1 padded.
        action_len : int
            Number of valid entries in action_idx.
        mcts_pi : np.ndarray or None
            MCTS visit distribution π̃(o) over legal options (None if no
            MCTS was run on this state).
        mcts_value : float or None
            MCTS root value Ṽ (None same as above).
        game_outcome : float
            Terminal reward from our perspective (+1 win, -1 loss, 0 draw).
        opp_archetype : int or None
            Which 𝒟_opp archetype the opponent was using (known in self-play).
        """
        entry = {
            "features": features,
            "action_idx": action_idx,
            "action_len": action_len,
            "mcts_pi": mcts_pi,
            "mcts_value": mcts_value,
            "game_outcome": game_outcome,
            "opp_archetype": opp_archetype,
        }
        if len(self._buffer) < self.capacity:
            self._buffer.append(entry)
        else:
            self._buffer[self._pos] = entry
        self._pos = (self._pos + 1) % self.capacity

    def push_game(
        self,
        decisions: list,
        game_outcome: float,
        opp_archetype: int | None = None,
    ) -> int:
        """Push all decisions from one game.  Returns number pushed."""
        n = 0
        for dec in decisions:
            mcts_pi = getattr(dec, "mcts_pi", None)
            mcts_value = getattr(dec, "mcts_value", None)
            self.push(
                features=dec.features,
                action_idx=dec.action_idx,
                action_len=dec.action_len,
                mcts_pi=mcts_pi,
                mcts_value=mcts_value,
                game_outcome=game_outcome,
                opp_archetype=opp_archetype,
            )
            n += 1
        return n

    def sample(self, batch_size: int, rng: np.random.Generator | None = None
               ) -> list[dict]:
        """Uniform random sample of *batch_size* entries."""
        if not self._buffer:
            return []
        rng = rng or np.random.default_rng()
        n = min(batch_size, len(self._buffer))
        indices = rng.choice(len(self._buffer), size=n, replace=False)
        return [self._buffer[int(i)] for i in indices]

    def sample_with_mcts(self, batch_size: int,
                         rng: np.random.Generator | None = None
                         ) -> list[dict]:
        """Sample entries that HAVE MCTS targets (mcts_pi is not None)."""
        mcts_entries = [e for e in self._buffer if e["mcts_pi"] is not None]
        if not mcts_entries:
            return []
        rng = rng or np.random.default_rng()
        n = min(batch_size, len(mcts_entries))
        indices = rng.choice(len(mcts_entries), size=n, replace=False)
        return [mcts_entries[int(i)] for i in indices]

    def clear(self) -> None:
        self._buffer.clear()
        self._pos = 0


class _NullContext:
    """No-op context manager for when bf16 is disabled / not on CUDA."""

    def __enter__(self):
        return None

    def __exit__(self, *exc) -> bool:
        return False

# ── Rust library discovery ──────────────────────────────────────────────────


def _find_libptcg_search() -> str:
    """Return the path to ``libptcg_search.so``, searching common locations."""
    base = Path(__file__).resolve().parent.parent  # python/
    candidates = [
        base / "ptcg_search/target/release/libptcg_search.so",
        base / "ptcg_search/target/debug/libptcg_search.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "libptcg_search.so not found.  Build it with: "
        "cd python/ptcg_search && cargo build --release"
    )


def _load_lib():
    """Load the Rust library once and cache it."""
    path = _find_libptcg_search()
    lib = ctypes.CDLL(path)

    # ── puct_init ────────────────────────────────────────────────────────
    lib.puct_init.argtypes = [
        ctypes.c_char_p,  # obs_json
        ctypes.c_char_p,  # lib_path (to libcg.so)
        ctypes.c_char_p,  # fixed_deck_json
        ctypes.c_char_p,  # opp_deck_json
        ctypes.c_int,     # iterations
        ctypes.c_double,  # c_puct
        ctypes.c_int,     # seed
        ctypes.c_int,     # host_initialized
    ]
    lib.puct_init.restype = ctypes.c_int64

    # ── puct_select ──────────────────────────────────────────────────────
    lib.puct_select.argtypes = [ctypes.c_int64]
    lib.puct_select.restype = ctypes.c_void_p

    # ── puct_expand ──────────────────────────────────────────────────────
    lib.puct_expand.argtypes = [
        ctypes.c_int64,
        ctypes.c_char_p,  # priors_json
        ctypes.c_double,  # value
    ]
    lib.puct_expand.restype = ctypes.c_int

    # ── puct_result ──────────────────────────────────────────────────────
    lib.puct_result.argtypes = [ctypes.c_int64]
    lib.puct_result.restype = ctypes.c_void_p

    # ── puct_free / puct_free_result ─────────────────────────────────────
    lib.puct_free.argtypes = [ctypes.c_int64]
    lib.puct_free.restype = None

    lib.puct_free_result.argtypes = [ctypes.c_void_p]
    lib.puct_free_result.restype = None

    return lib


def _read_and_free(lib, ptr) -> str:
    """Read a C string from a Rust-allocated pointer and free it."""
    if ptr is None or ptr == 0:
        return ""
    try:
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
    finally:
        lib.puct_free_result(ptr)


# ── PUCT config ─────────────────────────────────────────────────────────────


class PuctConfig:
    """Hyperparameters for PUCT search (RL_SPEC §9.4)."""

    def __init__(
        self,
        iterations: int = 128,
        c_puct: float = 2.0,
        seed: int = 42,
    ):
        self.iterations = iterations
        self.c_puct = c_puct
        self.seed = seed


# ── PUCT Searcher (single-tree, Phase 3a) ───────────────────────────────────


class PuctSearcher:
    """One PUCT search tree, wrapping the Rust C-ABI.

    Usage::

        searcher = PuctSearcher(
            obs_dict, fixed_deck, opp_deck_template, libcg_path,
            our_player_index, config,
        )
        while not searcher.done:
            leaf = searcher.select()
            priors, value = your_nn_evaluator(leaf.obs_json)
            searcher.expand(priors, value)
        result = searcher.result()
        searcher.close()
    """

    def __init__(
        self,
        obs_dict: dict,
        fixed_deck: list[int],
        opp_deck_template: list[int] | None = None,
        libcg_path: str | None = None,
        our_player_index: int | None = None,
        config: PuctConfig | None = None,
    ):
        self._lib = _load_lib()
        self._config = config or PuctConfig()
        self._handle: int = 0
        self._done: bool = False

        obs_json = json.dumps(obs_dict)
        fixed_json = json.dumps(fixed_deck)
        opp_json = json.dumps(opp_deck_template if opp_deck_template else [])

        lib_path = libcg_path or _find_libcg()
        if our_player_index is None:
            our_player_index = (
                obs_dict.get("current", {}).get("yourIndex", 0)
            )

        self._handle = self._lib.puct_init(
            obs_json.encode("utf-8"),
            lib_path.encode("utf-8"),
            fixed_json.encode("utf-8"),
            opp_json.encode("utf-8"),
            self._config.iterations,
            self._config.c_puct,
            self._config.seed,
            1,  # host_initialized: Python always initialises GameInitialize
        )

        if self._handle == 0:
            raise RuntimeError("puct_init returned null handle")

    @property
    def done(self) -> bool:
        return self._done

    def select(self) -> LeafRequest | None:
        """Walk tree to a leaf.  Returns None when the search is complete."""
        raw = _read_and_free(self._lib, self._lib.puct_select(self._handle))
        if not raw:
            raise RuntimeError("puct_select returned empty")
        result = json.loads(raw)
        err = result.get("error")
        if err:
            raise RuntimeError(f"puct_select: {err}")
        if result["tree_done"]:
            self._done = True
            return None
        return LeafRequest(
            obs_json=result["leaf_obs_json"],
            player_role=result["player_role"],
            is_terminal=result["is_terminal"],
            n_options=result["n_options"],
            iter_count=result["iter_count"],
        )

    def expand(self, priors: list[float], value: float) -> None:
        """Apply NN priors and value, backpropagate."""
        code = self._lib.puct_expand(
            self._handle,
            json.dumps(priors).encode("utf-8"),
            float(value),
        )
        if code != 0:
            raise RuntimeError(f"puct_expand returned {code}")

    def result(self) -> PuctResult:
        """Get final visit counts and root value."""
        raw = _read_and_free(self._lib, self._lib.puct_result(self._handle))
        if not raw:
            raise RuntimeError("puct_result returned empty")
        data = json.loads(raw)
        err = data.get("error")
        if err:
            raise RuntimeError(f"puct_result: {err}")
        return PuctResult(
            visit_counts=data["visit_counts"],
            root_value=data["root_value"],
            iterations=data["iterations"],
            nodes_created=data["nodes_created"],
        )

    def close(self) -> None:
        if self._handle != 0:
            self._lib.puct_free(self._handle)
            self._handle = 0

    def __enter__(self) -> PuctSearcher:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        self.close()


# ── Data classes ────────────────────────────────────────────────────────────


class LeafRequest:
    """A leaf node that needs NN evaluation."""

    __slots__ = ("obs_json", "player_role", "is_terminal", "n_options", "iter_count")

    def __init__(
        self,
        obs_json: str,
        player_role: int,
        is_terminal: bool,
        n_options: int,
        iter_count: int,
    ):
        self.obs_json = obs_json
        self.player_role = player_role  # 0 = us, 1 = opponent
        self.is_terminal = is_terminal
        self.n_options = n_options
        self.iter_count = iter_count


class PuctResult:
    """Search result for one decision point."""

    __slots__ = ("visit_counts", "root_value", "iterations", "nodes_created")

    def __init__(
        self,
        visit_counts: list[tuple[int, int]],
        root_value: float | None,
        iterations: int,
        nodes_created: int,
    ):
        self.visit_counts = visit_counts  # [(option_idx, visits), ...]
        self.root_value = root_value  # mean playout outcome in [-1, 1]
        self.iterations = iterations
        self.nodes_created = nodes_created


class StoredState:
    """One decision point tagged for MCTS distillation.

    Captures everything needed to reconstruct a determinized search root:
    the original observation, the featurized tensors (for L_search later),
    and opponent-visible cards (for belief posterior).
    """

    __slots__ = (
        "obs_dict", "features", "action_idx", "action_len",
        "opp_visible_card_ids", "fixed_deck",
    )

    def __init__(
        self,
        obs_dict: dict,
        features: dict[str, np.ndarray],
        action_idx: np.ndarray,
        action_len: int,
        opp_visible_card_ids: list[int] | None = None,
        fixed_deck: list[int] | None = None,
    ):
        self.obs_dict = obs_dict
        self.features = features
        self.action_idx = action_idx
        self.action_len = action_len
        self.opp_visible_card_ids = opp_visible_card_ids or []
        self.fixed_deck = fixed_deck or []


class SearchTarget:
    """Distillation target for one decision point (RL_SPEC §8.3).

    ``visit_distribution`` is π̃(o) — the aggregated visit-count distribution
    over K determinizations, normalised to sum to 1 (over LEGAL options).
    ``root_value`` is Ṽ — the mean root value across determinizations.
    """

    __slots__ = ("visit_distribution", "root_value", "n_options")

    def __init__(
        self,
        visit_distribution: np.ndarray,
        root_value: float,
        n_options: int,
    ):
        self.visit_distribution = visit_distribution
        self.root_value = root_value
        self.n_options = n_options


# ── Helpers ─────────────────────────────────────────────────────────────────


def _find_libcg() -> str:
    """Find libcg.so on the filesystem."""
    base = Path(__file__).resolve().parent.parent.parent  # repo root
    candidates = [
        base / "python/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/libcg.so",
        base / "pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/libcg.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"libcg.so not found, searched: {candidates}")


def featurize_leaf(
    obs_json: str, policy: Any, vocab: dict, device: Any,
    *,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
) -> tuple[list[float], float]:
    """Featurize a leaf observation and run one forward pass.

    Returns (priors, value) where priors is a list of floats (one per legal
    option) and value is a float in [-1, 1].

    Parameters
    ----------
    obs_json : str
        The observation JSON string from the search engine (includes
        ``select`` and ``current`` keys).
    policy : Policy
        The trained policy model.
    vocab : dict
        Normalized vocab (id→index mapping).
    device : torch.device
    """
    import torch

    from ptcg_il.featurizer import featurize

    obs_dict = json.loads(obs_json)
    feats = featurize(obs_dict, vocab,
                      engine_card_features=engine_card_features,
                      engine_attack_features=engine_attack_features)

    # Build batch of size 1
    batch = {}
    for k, v in feats.items():
        if not isinstance(v, np.ndarray):
            continue
        t = torch.from_numpy(v).unsqueeze(0)
        if v.dtype == np.int64:
            t = t.long()
        elif v.dtype == np.bool_:
            t = t.bool()
        else:
            t = t.float()
        batch[k] = t.to(device)

    with torch.no_grad():
        h, _history_h = policy._encode(batch)
        logits, _ = policy.pointer(h, batch["tok_mask"],
                                    policy.embed.card, batch)
        value = policy.value(h[:, 0, :])

    # Mask and compute softmax for priors
    opt_mask = batch["opt_mask"][0].cpu().numpy()  # [O_MAX], bool
    logits_1d = logits[0].float().cpu().numpy()
    logits_1d = np.where(opt_mask, logits_1d, -np.inf)
    logits_1d = logits_1d - logits_1d.max()
    probs = np.exp(logits_1d)
    probs = probs / probs.sum()

    # Only return priors for actual legal options (up to n_options).
    # The Rust side expects one prior per option index.
    n_legal = int(opt_mask.sum())
    priors = probs[:n_legal].tolist() if n_legal > 0 else []
    val = float(value[0, 0].cpu().numpy())

    return priors, val


# ── Batch MCTS Forest (Phase 3b) ─────────────────────────────────────────


class MctsForest:
    """Batched multi-tree PUCT MCTS via Rust PuctForest.

    Wraps the ``puct_forest_*`` C-ABI.  The Python side drives the loop::

        forest = shared_forest(n_engines=4)   # not MctsForest(...)
        for state in tagged_states:
            forest.add_root(state.obs_dict, state.fixed_deck, ...)
        while forest.active_count > 0:
            leaves = forest.select_batch(batch_size=512)
            priors_vals = batch_forward(leaves, policy, vocab, device)
            forest.expand_batch(priors_vals)
        results = forest.results()
        forest.reset()

    Construct one per process, through :func:`shared_forest`, and
    ``reset()`` between batches.  Constructing one per batch and calling
    ``close()`` leaks: see :meth:`reset`.
    """

    def __init__(
        self,
        n_engines: int = 4,
        libcg_path: str | None = None,
    ):
        self._lib = _load_lib()
        self._n_engines = n_engines
        self._handle: int = 0

        lib_path = libcg_path or _find_libcg()

        # ── Register forest C-ABI ─────────────────────────────────────
        self._lib.puct_forest_create.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ]
        self._lib.puct_forest_create.restype = ctypes.c_int64

        self._lib.puct_forest_add_root.argtypes = [
            ctypes.c_int64, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_int,
        ]
        self._lib.puct_forest_add_root.restype = ctypes.c_int64

        self._lib.puct_forest_select_batch.argtypes = [
            ctypes.c_int64, ctypes.c_int,
        ]
        self._lib.puct_forest_select_batch.restype = ctypes.c_void_p

        self._lib.puct_forest_expand_batch.argtypes = [
            ctypes.c_int64, ctypes.c_char_p,
        ]
        self._lib.puct_forest_expand_batch.restype = ctypes.c_int

        self._lib.puct_forest_results.argtypes = [ctypes.c_int64]
        self._lib.puct_forest_results.restype = ctypes.c_void_p

        self._lib.puct_forest_set_basic_pokemon.argtypes = [
            ctypes.c_int64, ctypes.c_char_p,
        ]
        self._lib.puct_forest_set_basic_pokemon.restype = ctypes.c_int

        self._lib.puct_forest_reset.argtypes = [ctypes.c_int64]
        self._lib.puct_forest_reset.restype = ctypes.c_int

        self._lib.puct_forest_free.argtypes = [ctypes.c_int64]
        self._lib.puct_forest_free.restype = None

        self._handle = self._lib.puct_forest_create(
            n_engines,
            lib_path.encode("utf-8"),
            1,  # host_initialized
        )
        if self._handle == 0:
            raise RuntimeError("puct_forest_create returned null handle")

        self._tree_count = 0

    def set_basic_pokemon(self, card_ids) -> int:
        """Tell the determinizer which engine card ids are Basic Pokémon.

        Only a Basic can legally sit face-down as the opponent's active.
        Without this the guess is drawn from the whole deck template, and
        since only ~16% of a template's slots are Basic, ``SearchBegin``
        refuses most such roots with error 2.  Optional — a forest that is
        never told keeps the old behaviour.  Returns the number of ids stored.
        """
        ids = sorted({int(c) for c in card_ids})
        n = self._lib.puct_forest_set_basic_pokemon(
            self._handle, json.dumps(ids).encode("utf-8"),
        )
        if n < 0:
            raise RuntimeError("puct_forest_set_basic_pokemon failed")
        return int(n)

    def add_root(
        self,
        obs_dict: dict,
        fixed_deck: list[int],
        opp_deck_template: list[int] | None = None,
        iterations: int = 128,
        c_puct: float = 2.0,
        seed: int = 42,
    ) -> int:
        """Add a search root to the forest.

        Returns the tree_id, or ``-1`` if the engine refused the root.
        Refusal is routine and per-state — ``SearchBegin`` rejects some
        positions outright (error code 2) — so it is reported through the
        return value, not an exception: every caller adds roots in a loop
        over decision points and must keep the surviving ones rather than
        lose a whole self-play batch to one bad state.  The Rust side prints
        the reason to stderr.
        """
        obs_json = json.dumps(obs_dict)
        fixed_json = json.dumps(fixed_deck)
        opp_json = json.dumps(opp_deck_template if opp_deck_template else [])

        tree_id = self._lib.puct_forest_add_root(
            self._handle,
            obs_json.encode("utf-8"),
            fixed_json.encode("utf-8"),
            opp_json.encode("utf-8"),
            iterations,
            c_puct,
            seed,
        )
        if tree_id < 0:
            return -1
        self._tree_count = max(self._tree_count, tree_id + 1)
        return tree_id

    @property
    def active_count(self) -> int:
        """Estimate — will be refined when Rust provides the API."""
        return 1 if self._tree_count > 0 else 0  # placeholder

    def select_batch(self, batch_size: int = 512) -> list[dict]:
        """Select up to batch_size leaves.  Returns list of leaf dicts."""
        raw = _read_and_free(
            self._lib, self._lib.puct_forest_select_batch(self._handle, batch_size)
        )
        if not raw:
            return []
        return json.loads(raw)

    def expand_batch(self, expansions: list[dict]) -> int:
        """Expand leaves with NN priors and values.  Returns success count."""
        return self._lib.puct_forest_expand_batch(
            self._handle,
            json.dumps(expansions).encode("utf-8"),
        )

    def results(self) -> list[dict]:
        """Get results for all trees."""
        raw = _read_and_free(
            self._lib, self._lib.puct_forest_results(self._handle)
        )
        if not raw:
            return []
        return json.loads(raw)

    def reset(self) -> int:
        """Drop every tree, keeping the engines.  Returns trees dropped.

        Use this, not ``close()``, between search batches.  ``close()``
        frees the engine pool, and a freed engine's memory never comes
        back: libcg exports no ``AgentEnd``, and ``SearchEnd`` only
        returns the arena to that same agent for reuse (see
        ``cg/api.py``'s ``search_end`` docstring).  A forest per search
        batch therefore strands ``n_engines`` arenas per batch — the
        leak that took self-play out at ~2000 games.
        """
        if self._handle == 0:
            return 0
        n = self._lib.puct_forest_reset(self._handle)
        if n < 0:
            raise RuntimeError("puct_forest_reset failed")
        self._tree_count = 0
        return int(n)

    def close(self) -> None:
        if self._handle != 0:
            self._lib.puct_forest_free(self._handle)
            self._handle = 0

    def __enter__(self) -> "MctsForest":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


# ── Process-wide forest ──────────────────────────────────────────────────

_SHARED_FOREST: "MctsForest | None" = None
_SHARED_FOREST_ENGINES: int = 0


def shared_forest(n_engines: int = 4, libcg_path: str | None = None) -> "MctsForest":
    """The one `MctsForest` for this process, created on first call.

    Every later call resets it — trees dropped, engines kept — and hands
    back the same object.  The engines have to be process-lived: libcg's
    ``AgentStart`` has no counterpart in the ABI, so each pool that gets
    freed strands everything its agents allocated, permanently.

    *n_engines* is honoured on the first call only.  A later call asking
    for a different count logs a warning and keeps the existing pool
    rather than growing agents the process can never reclaim.
    """
    global _SHARED_FOREST, _SHARED_FOREST_ENGINES
    if _SHARED_FOREST is None:
        _SHARED_FOREST = MctsForest(n_engines=n_engines, libcg_path=libcg_path)
        _SHARED_FOREST_ENGINES = n_engines
        logger.info("MCTS forest created: %d engines (process-wide)", n_engines)
    else:
        if n_engines != _SHARED_FOREST_ENGINES:
            logger.warning(
                "shared_forest asked for %d engines but the process pool has "
                "%d — keeping it (libcg agents cannot be freed, so resizing "
                "leaks the old pool)",
                n_engines, _SHARED_FOREST_ENGINES,
            )
        _SHARED_FOREST.reset()
    return _SHARED_FOREST


def close_shared_forest() -> None:
    """Free the process-wide forest.  For tests and shutdown only."""
    global _SHARED_FOREST, _SHARED_FOREST_ENGINES
    if _SHARED_FOREST is not None:
        _SHARED_FOREST.close()
        _SHARED_FOREST = None
        _SHARED_FOREST_ENGINES = 0

    def __del__(self) -> None:
        self.close()


#: Sub-phases of :func:`batch_evaluate_leaves`, in seconds.  Module-level for
#: the same reason as ``_MCTS_PERF``: the call sits inside the per-game search
#: loop and threading an accumulator down would touch every frame between.
LEAF_PERF: dict[str, float] = {
    "t_featurize": 0.0,   # CPU: json.loads + featurize, per leaf, serial
    "t_forward": 0.0,     # GPU: one batched forward
    "n_leaves": 0.0,
}


def reset_leaf_perf() -> dict[str, float]:
    snap = dict(LEAF_PERF)
    for k in LEAF_PERF:
        LEAF_PERF[k] = 0.0
    return snap


def batch_evaluate_leaves(
    leaves: list[dict],
    policy: Any,
    vocab: dict,
    device: Any,
    bf16: bool = True,
    *,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
) -> list[dict]:
    """Batch-featurize and GPU-forward a list of leaf dicts.

    Each leaf is ``{tree_id, obs_json, player_role, is_terminal, n_options}``.
    Returns ``[{tree_id, priors: [...], value: float}, ...]``.

    Uses bf16 autocast when ``bf16=True`` and device is CUDA (matches
    rollout inference precision per §7).

    **Pass the engine feature tables.**  Card identity reaches this model only
    as static features looked up by engine card id (``featurizer._raw_card``),
    and ``_ids_to_feat`` returns an all-zero block when ``engine_features`` is
    ``None``.  Omitting them therefore evaluates every leaf on a board where no
    card has any identity — only HP, energy and slot scalars survive — so the
    priors and values driving the whole search are computed blind.  Nothing
    raises: zeros are a valid feature row, indistinguishable from an empty
    slot.  Measured on a real observation, dropping them took
    ``poke_card_feat`` from 2 non-zero rows to 0, ``hand_card_feat`` from 7 to
    0, and ``opt_card_feat``'s abs-sum from 14.0 to 0.0.  The rollout actor has
    always passed them; only search did not, so ``mcts_pi`` was distilled from
    a blind search while acting stayed sighted.
    """
    import time as _time

    import torch

    from ptcg_il.featurizer import featurize

    if not leaves:
        return []

    # Featurize all leaves.  Timed separately from the forward because the two
    # are on different processors and have opposite fixes: this loop is
    # single-threaded CPU (a json.loads plus a featurize per leaf), while the
    # forward below is one batched GPU call.  Reporting them as one "eval"
    # number pointed at the GPU when the cost was mostly here.
    _t0 = _time.perf_counter()
    feat_list = []
    for leaf in leaves:
        if leaf.get("is_terminal") or leaf.get("n_options", 0) == 0:
            feat_list.append(None)
            continue
        obs_dict = json.loads(leaf["obs_json"])
        feats = featurize(obs_dict, vocab,
                          engine_card_features=engine_card_features,
                          engine_attack_features=engine_attack_features)
        feat_list.append(feats)
    LEAF_PERF["t_featurize"] += _time.perf_counter() - _t0
    LEAF_PERF["n_leaves"] += len(leaves)

    # Build batch for non-terminal leaves
    valid_indices = [i for i, f in enumerate(feat_list) if f is not None]
    if not valid_indices:
        expansions = []
        for i, leaf in enumerate(leaves):
            expansions.append({
                "tree_id": leaf["tree_id"],
                "priors": [],
                "value": 0.0,
            })
        return expansions

    # Stack valid feats into a batch
    batch: dict[str, list] = {}
    for idx in valid_indices:
        for k, v in feat_list[idx].items():
            if not isinstance(v, np.ndarray):
                continue
            batch.setdefault(k, []).append(v)

    # Convert to tensors
    tensor_batch = {}
    for k, vs in batch.items():
        stacked = np.stack(vs)
        t = torch.from_numpy(stacked)
        if stacked.dtype == np.int64:
            t = t.long()
        elif stacked.dtype == np.bool_:
            t = t.bool()
        else:
            t = t.float()
        tensor_batch[k] = t.to(device)

    # GPU forward with bf16 autocast (matches rollout §7 precision)
    use_bf16 = bf16 and device.type == "cuda"
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else _NullContext()
    _t_fwd = _time.perf_counter()
    with torch.no_grad(), autocast_ctx:
        # Go through Policy.forward rather than driving _encode/pointer/value
        # by hand: the pointer head takes (h, tok_mask, card_encoder, x) and
        # applies the option mask itself.
        logits, values, _history_h = policy(tensor_batch)

    LEAF_PERF["t_forward"] += _time.perf_counter() - _t_fwd
    # Always read back in fp32 for prior/softmax stability
    logits_np = logits.float().cpu().numpy()
    values_np = values.float().cpu().numpy()
    opt_mask_np = tensor_batch["opt_mask"].cpu().numpy()

    # Build expansions
    expansions = []
    valid_pos = 0
    for i, leaf in enumerate(leaves):
        if i not in valid_indices:
            expansions.append({"tree_id": leaf["tree_id"], "priors": [], "value": 0.0})
            continue

        n_options = leaf.get("n_options", 0)
        if n_options == 0 or leaf.get("is_terminal"):
            expansions.append({"tree_id": leaf["tree_id"], "priors": [], "value": 0.0})
            valid_pos += 1
            continue

        mask = opt_mask_np[valid_pos]
        l = logits_np[valid_pos]

        # The STOP column is a label for autoregressive multi-select, not an
        # engine option — the tree has no child for it.  Leaving it in the
        # softmax scaled every real option's prior by 1 - P(stop), and where
        # the featurizer had to drop an option to make room for it (n_total ==
        # O_MAX) P(stop) landed on a *different* action's slot.

        stop_col = int(feat_list[i].get("stop_column", -1))
        if stop_col >= 0:
            mask = mask.copy()
            mask[stop_col] = False

        # Mask and softmax
        l = np.where(mask, l, -np.inf)
        l = l - l.max()
        probs = np.exp(l)
        probs = probs / probs.sum()

        # One prior per tree child, no more and no less.  The valid columns are
        # packed from index 0, so the head of `probs` is exactly them.  The cap
        # at n_options matters when the featurizer had to truncate a >O_MAX
        # option list: it returns fewer priors than the engine has options, and
        # the tree is built to match (expand_leaf takes the min), so options
        # past O_MAX are simply not searched — the policy cannot express them
        # either.
        n_legal = min(int(mask.sum()), n_options)
        priors = probs[:n_legal].tolist() if n_legal > 0 else []

        # Policy.forward returns value as [B], not [B, 1]
        value = float(values_np[valid_pos])
        valid_pos += 1

        expansions.append({
            "tree_id": leaf["tree_id"],
            "priors": priors,
            "value": value,
        })

    return expansions


# ── Multi-deck self-play with MCTS distillation (Phase 3+) ───────────────


def load_opp_archetype_decks(
    data_dir: str | Path,
    all_archetypes: bool = False,
) -> list[dict]:
    """Load opponent archetype decklists and metadata.

    With ``all_archetypes=False`` (default): loads only ``opp_ids`` (6 𝒟_opp).
    With ``all_archetypes=True``: loads every archetype in the file (179).

    Returns a list of dicts with keys: ``id``, ``deck`` (60 card ids),
    ``frequency`` (for weighted sampling), ``count``, ``name``.
    """
    import json as _json

    data_dir = Path(data_dir)
    with open(data_dir / "archetypes.json") as f:
        arch_data = _json.load(f)

    by_id = {int(a["id"]): a for a in arch_data.get("archetypes", [])}

    if all_archetypes:
        ids = sorted(by_id.keys())
    else:
        ids = [int(i) for i in arch_data.get("opp_ids", [])]

    total_games = sum(
        by_id[aid].get("count", 0) for aid in ids if aid in by_id
    )

    decks = []
    for aid in ids:
        info = by_id.get(aid, {})
        rep = info.get("representative") or []
        count = info.get("count", 0)
        freq = count / max(total_games, 1)
        if total_games <= 0:
            freq = 1.0 / max(len(ids), 1)
        decks.append({
            "id": aid,
            "deck": [int(c) for c in rep],
            "frequency": freq,
            "count": count,
            "name": info.get("name", f"archetype_{aid}"),
        })
    return decks


def sample_opp_archetype(
    archetypes: list[dict],
    rng: np.random.Generator | None = None,
) -> dict:
    """Sample an opponent archetype weighted by frequency."""
    if not archetypes:
        raise ValueError("no opponent archetypes to sample")
    rng = rng or np.random.default_rng()
    weights = np.array([a["frequency"] for a in archetypes], dtype=np.float64)
    weights = weights / weights.sum()
    idx = int(rng.choice(len(archetypes), p=weights))
    return archetypes[idx]


def mcts_distill_game(
    policy: Any,
    vocab: dict,
    archetypes: list[dict],
    fixed_deck: list[int],
    opp_archetype: dict,
    config: Any,  # RLConfig with MCTS params
    device: Any,
    seed: int = 0,
    *,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
) -> list[dict]:
    """Run one self-play game with MCTS distillation targets.

    Plays our fixed_deck against opp_archetype["deck"] using the policy
    network for both sides.  On ρ of decision points, runs batched MCTS
    to produce distillation targets (π̃, Ṽ).

    Because this is self-play, the opponent's deck is KNOWN — MCTS
    determinization uses the actual decklist, eliminating strategy fusion.

    Returns a list of dicts, one per decision point::

        {"features": ..., "action_idx": ..., "action_len": ...,
         "mcts_pi": np.ndarray | None, "mcts_value": float | None,
         "game_outcome": float, "opp_archetype": int}
    """
    import torch

    from ptcg_rl.rollout import PolicyActor
    from ptcg_rl.vec_env import RolloutPool

    policy.eval()  # self-play runs in eval mode (no dropout, etc.)

    opp_decklist = opp_archetype["deck"]
    opp_id = opp_archetype["id"]
    rng = np.random.default_rng(seed)

    decisions_with_targets = []

    actor = PolicyActor(policy, vocab, device=str(device), bf16=True, seed=seed)

    with RolloutPool(
        fixed_deck, opp_decklist,
        n_workers=getattr(config, "n_workers", 4),
        forward_batch=getattr(config, "forward_batch", 256),
        seed=seed, our_player=0,
    ) as pool:
        # Collect one game worth of decisions
        trajectories = pool.collect(actor, n_decisions=300)  # ~2 games

        for traj in trajectories:
            outcome = float(traj.reward)  # +1 win, -1 loss, 0 draw
            decisions = traj.decisions

            if not decisions:
                continue

            # ── Tag ρ fraction for MCTS ────────────────────────────
            n = len(decisions)
            n_tag = max(1, int(n * getattr(config, "mcts_rho", 0.05)))
            tagged_idx = set(
                rng.choice(n, size=min(n_tag, n), replace=False).tolist()
            )

            mcts_targets: dict[int, tuple] = {}  # idx -> (pi, value)

            if tagged_idx and getattr(config, "mcts_enabled", False):
                # Run MCTS on tagged states
                # Use the KNOWN opponent deck for perfect determinization
                mcts_targets = _run_mcts_on_decisions(
                    decisions, tagged_idx, policy, vocab,
                    fixed_deck, opp_decklist, config, device, seed,
                    engine_card_features=engine_card_features,
                    engine_attack_features=engine_attack_features,
                )

            # ── Build enriched decision dicts ──────────────────────
            for i, dec in enumerate(decisions):
                mcts_pi, mcts_value = mcts_targets.get(i, (None, None))
                decisions_with_targets.append({
                    "features": dec.features,
                    "action_idx": dec.action_idx,
                    "action_len": dec.action_len,
                    "logp": dec.logp,
                    "value": dec.value,
                    "mcts_pi": mcts_pi,
                    "mcts_value": mcts_value,
                    "game_outcome": outcome,
                    "opp_archetype": opp_id,
                })

    return decisions_with_targets


def _run_mcts_on_decisions(
    decisions: list,
    tagged_idx: set,
    policy: Any,
    vocab: dict,
    fixed_deck: list[int],
    opp_deck: list[int],
    config: Any,
    device: Any,
    seed: int,
    *,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
) -> dict[int, tuple]:
    """Run batched MCTS on tagged decisions, using the KNOWN opponent deck.

    Returns ``{decision_index: (mcts_pi_array, mcts_value_float)}``.
    """
    import json as _json

    # Process-wide pool, not a fresh one per call — see `shared_forest`.
    forest = shared_forest(
        n_engines=getattr(config, "mcts_n_engines", 4),
        libcg_path=None,
    )

    # tree_id -> (decision_idx, determinization_k)
    tree_map: dict[int, tuple[int, int]] = {}
    K = getattr(config, "mcts_k_determinizations", 8)

    try:
        for i in tagged_idx:
            dec = decisions[i]
            obs_json = getattr(dec, "obs_json", None)
            if not obs_json:
                continue
            obs_dict = _json.loads(obs_json)

            # K determinizations, all using the KNOWN opponent deck
            for k in range(K):
                tree_id = forest.add_root(
                    obs_dict,
                    fixed_deck,
                    opp_deck_template=opp_deck,  # KNOWN, not guessed!
                    iterations=getattr(config, "mcts_iterations", 128),
                    c_puct=getattr(config, "mcts_c_puct", 2.0),
                    seed=seed + i * 1000 + k,
                )
                if tree_id >= 0:
                    tree_map[tree_id] = (i, k)

        if not tree_map:
            return {}

        # MCTS batch loop
        leaf_batch = getattr(config, "mcts_leaf_batch", 512)
        while True:
            leaves = forest.select_batch(leaf_batch)
            if not leaves:
                break
            expansions = batch_evaluate_leaves(
                leaves, policy, vocab, device,
                engine_card_features=engine_card_features,
                engine_attack_features=engine_attack_features,
            )
            forest.expand_batch(expansions)

        results = {r["tree_id"]: r for r in forest.results()}
    finally:
        forest.reset()

    # Aggregate K determinizations per decision
    targets: dict[int, tuple] = {}
    state_results: dict[int, list[dict]] = {}
    for tree_id, result in results.items():
        di, k = tree_map[tree_id]
        state_results.setdefault(di, []).append(result)

    for di, k_results in state_results.items():
        agg_visits: dict[int, float] = {}
        agg_value = 0.0
        n_value = 0
        n_opts = 0

        for r in k_results:
            for opt, visits in r.get("visit_counts", []):
                agg_visits[opt] = agg_visits.get(opt, 0.0) + visits
                n_opts = max(n_opts, opt + 1)
            rv = r.get("root_value")
            if rv is not None:
                agg_value += rv
                n_value += 1

        if n_opts > 0 and agg_visits:
            total = sum(agg_visits.values())
            pi = np.zeros(n_opts, dtype=np.float32)
            for opt, v in agg_visits.items():
                if opt < n_opts and total > 0:
                    pi[opt] = float(v) / total
        else:
            pi = np.array([], dtype=np.float32)

        root_v = (agg_value / max(n_value, 1)) if n_value > 0 else None
        targets[di] = (pi, root_v)

    return targets


def replay_buffer_train_step(
    policy: Any,
    replay_buffer: ReplayBuffer,
    optimizer: Any,
    config: Any,
    device: Any,
    batch_size: int = 256,
) -> dict[str, float]:
    """One training step: sample from replay buffer, compute loss, update.

    Loss:
      L = CE(π_θ(s), π̃_mcts) + c_v * MSE(V_θ(s), outcome)

    When a sample has no MCTS target (mcts_pi is None), only the value
    loss contributes.  The KL anchor to π_IL is omitted here (the buffer
    stores the frozen IL's states), but can be added.
    """
    import torch
    import torch.nn.functional as F

    from ptcg_il.train.dataset import collate_fn

    rng = np.random.default_rng()
    samples = replay_buffer.sample(batch_size, rng)
    if not samples:
        return {"loss": 0.0}

    # Collate features into a batch
    feat_list = [s["features"] for s in samples]
    batch = _collate_feat_list(feat_list, device)

    # Forward
    h, _hh = policy._encode(batch)
    logits, _ = policy.pointer(h, batch["tok_mask"],
                                policy.embed.card, batch)
    values = policy.value(h[:, 0])  # [B]

    total_loss = torch.zeros((), device=device)
    n_pi = 0
    n_v = 0
    total_pi_loss = torch.zeros((), device=device)
    total_v_loss = torch.zeros((), device=device)

    for i, sample in enumerate(samples):
        outcome = torch.tensor(sample["game_outcome"], device=device)

        # Value loss: MSE to game outcome
        v_loss = F.mse_loss(values[i, 0], outcome)
        total_v_loss = total_v_loss + v_loss
        n_v += 1

        # Policy loss: CE to MCTS visit distribution (if available)
        mcts_pi = sample.get("mcts_pi")
        if mcts_pi is not None and len(mcts_pi) > 0:
            pi_tilde = torch.as_tensor(
                np.asarray(mcts_pi, dtype=np.float32)
            ).to(device)
            mask = batch["opt_mask"][i].bool()
            masked_logits = logits[i].float().masked_fill(~mask, float("-inf"))
            logp = F.log_softmax(masked_logits, dim=-1)
            n_legal = min(len(pi_tilde), int(mask.sum()))
            if n_legal > 0:
                pi_loss = -(pi_tilde[:n_legal] * logp[:n_legal]).sum()
                total_pi_loss = total_pi_loss + pi_loss
                n_pi += 1

    c_v = getattr(config, "c_value", 0.5)
    loss_pi = total_pi_loss / max(n_pi, 1)
    loss_v = total_v_loss / max(n_v, 1)
    loss = loss_pi + c_v * loss_v

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "pi_ce": float(loss_pi.detach()) if n_pi > 0 else 0.0,
        "v_mse": float(loss_v.detach()),
        "n_pi": n_pi,
        "n_v": n_v,
    }


def _collate_feat_list(
    feat_list: list[dict[str, np.ndarray]], device: Any
) -> dict[str, "torch.Tensor"]:
    """Stack a list of feature dicts into a batched tensor dict."""
    import torch

    keys = feat_list[0].keys()
    batch: dict[str, torch.Tensor] = {}
    for k in keys:
        arrs = []
        for f in feat_list:
            v = f.get(k)
            if isinstance(v, np.ndarray):
                arrs.append(v)
            else:
                arrs.append(np.array(v))
        stacked = np.stack(arrs)
        t = torch.from_numpy(stacked)
        if stacked.dtype == np.int64:
            t = t.long()
        elif stacked.dtype == np.bool_:
            t = t.bool()
        else:
            t = t.float()
        batch[k] = t.to(device)
    return batch
