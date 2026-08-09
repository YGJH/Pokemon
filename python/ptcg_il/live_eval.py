"""Live-engine eval harness — win-rate vs baselines (Section 5, C.8).

Wraps an EMA Policy as ``agent(obs_dict)`` per Section 6 reference forward
pass, then runs games via ``cg.game.battle_start/battle_select`` in a process
pool.  Reports win-rate with Wilson confidence intervals and tracks the OOV
rate at inference.

Usage::

    evaluator = LiveEvaluator(policy, vocab, fixed_deck)
    results = evaluator.eval_vs_opponents(
        n_games=500,
        opponents={"random": random_agent, "search": search_planner},
    )
"""

from __future__ import annotations

import logging
from rich.logging import RichHandler
import math
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

# ============================================================
# Engine path resolution
# ============================================================

_ENGINE_DIR = (
    Path(__file__).resolve().parent.parent
    / "pokemon-tcg-ai-battle"
    / "sample_submission"
    / "sample_submission"
)


def _your_index(obs: dict) -> int:
    """Index of the player who must act in ``obs``.

    The field lives on ``obs["current"]`` -- ``current`` *is* the ``State``
    dataclass (cg.api), and ``yourIndex`` is one of its fields.  It is **never**
    present at the top level of the observation dict.

    Reading ``obs.get("yourIndex", 0)`` therefore silently returned 0 for every
    decision, so ``_run_one_game`` handed every turn to ``agent0`` and the
    opponent never acted -- measured: 0/103 decisions had a top-level
    ``yourIndex`` while ``current.yourIndex`` alternated 55/48.  Every win rate
    produced that way describes one agent playing itself.
    """
    cur = obs.get("current")
    if isinstance(cur, dict) and cur.get("yourIndex") is not None:
        return int(cur["yourIndex"])
    # Fall back to the top level in case a future engine build moves it there.
    top = obs.get("yourIndex")
    return int(top) if top is not None else 0


def _add_engine_path() -> str:
    """Ensure the bundled ``cg`` package is importable."""
    engine_str = str(_ENGINE_DIR)
    if engine_str not in sys.path:
        sys.path.insert(0, engine_str)
    return engine_str


# ============================================================
# Wilson confidence interval
# ============================================================


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(center, lo, hi)``.  Center is the Wilson-adjusted point estimate.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return center, lo, hi


# ============================================================
# Agent wrapper
# ============================================================


class PolicyAgent:
    """Greedy-policy ``agent(obs_dict)`` callable (Section 6).

    Deliberately a class with ``__call__`` rather than a closure.  ``eval_vs_opponent``
    dispatches each game to a ``ProcessPoolExecutor``, and **every job argument is
    pickled**.  A local function is not picklable, so returning a closure here made
    every live-eval game die with::

        Can't pickle local object 'make_agent_from_policy.<locals>.agent'

    Behaviour:
    1. If ``obs["select"] is None``, return ``FIXED_DECK`` (the deck-submission step).
    2. Otherwise featurize, run the forward pass, and return the chosen option index
       (or indices, for multi-select).

    Parameters
    ----------
    policy : Policy (nn.Module)
        EMA-averaged policy.  Moved to ``device`` and put in eval mode here.
    vocab : dict
        Vocab dict with ``id_to_index`` and ``attack_id_to_index``.
    fixed_deck : list[int]
        The FIXED_DECK (60 card ids).
    device : str
        Torch device string (default ``"cpu"`` for live eval — each game runs in its
        own process, so GPU contention is avoided).
    """

    def __init__(
        self,
        policy: Any,
        vocab: dict,
        fixed_deck: list[int],
        *,
        device: str = "cpu",
        data_dir: Any = None,
    ) -> None:
        from ptcg_il.featurizer import normalize_vocab

        # normalize_vocab is required: vocab.json keys are strings, engine card ids
        # are ints, so the raw dict silently maps every card to UNKNOWN_CARD.
        self.vocab_full = normalize_vocab(vocab)
        self.fixed_deck = list(fixed_deck)
        self.device = device
        # Path, not the loaded dicts: every job argument is pickled to a worker
        # process (see the class docstring), and the card table alone is 1.1 MB.
        # The tables are loaded on first use and dropped again by __getstate__.
        self.data_dir = str(data_dir) if data_dir is not None else None
        self._engine_tables: dict | None = None

        policy.eval()
        policy.to(device)
        self.policy = policy

    def _tables(self) -> dict:
        """The static tables ``featurize`` needs, loaded once per process.

        Without these ``featurize`` returns **zeros** for every ``*_card_feat``
        and ``opt_attack_feat`` tensor -- so the agent played with no card
        identity, no attack identity, no KO-pressure block and no hand legality
        flags, against a model trained with all of them.  It does not raise and
        it does not look wrong from outside; it just quietly deletes most of the
        observation.  ``data_dir=None`` keeps the old behaviour for callers that
        genuinely have no artifacts, and says so once.
        """
        if self._engine_tables is None:
            if self.data_dir is None:
                if not getattr(type(self), "_warned_no_tables", False):
                    logger.warning(
                        "PolicyAgent built without data_dir: engine card/attack "
                        "features are unavailable, so every card feature will be "
                        "zero and the policy is running on a fraction of its "
                        "training inputs. Pass data_dir to fix this."
                    )
                    type(self)._warned_no_tables = True
                self._engine_tables = {
                    "engine_card_features": None,
                    "engine_attack_features": None,
                    "evolution_map": None,
                }
            else:
                from ptcg_il.featurizer import load_engine_tables

                self._engine_tables = load_engine_tables(self.data_dir)
        return self._engine_tables

    def __getstate__(self) -> dict:
        """Drop the loaded tables before pickling; the worker reloads them."""
        state = self.__dict__.copy()
        state["_engine_tables"] = None
        return state

    def __call__(self, obs_dict: dict) -> list[int]:
        import torch
        from ptcg_il.featurizer import featurize
        from ptcg_il.model.policy import decode_single_select, select_multi

        select = obs_dict.get("select")
        if select is None:
            return list(self.fixed_deck)

        # Featurize (action=None handled natively for inference mode)
        sample = featurize(
            obs_dict, self.vocab_full, value_target=0.0, sample_weight=1.0,
            **self._tables())

        # Build batch of size 1
        batch = _sample_to_batch(sample, self.device)
        max_count = int(sample["maxCount"])

        with torch.no_grad():
            if max_count == 1:
                logits, _value, _hist = self.policy(batch)
                # A minCount==0 select carries a STOP column; picking it means
                # declining, and returning its index would be engine error 5.
                return decode_single_select(
                    logits, batch["opt_mask"], batch.get("stop_column"),
                )
            else:
                chosen = select_multi(self.policy, batch)  # [1, batch_max]
                # Truncate at the first STOP (-2) rather than filtering it out:
                # _select_multi_raw keeps emitting picks for a sample that has
                # already stopped, with a stale picked_mask and a stale msgru,
                # so anything after the STOP is not a decision the model made.
                picks: list[int] = []
                for p in chosen[0].tolist():
                    if p == -2:
                        break
                    if p >= 0:
                        picks.append(int(p))
                return picks[:max_count]

    def __setstate__(self, state: dict) -> None:
        """Restore in a worker process: re-assert eval mode and device.

        ``nn.Module.__reduce__`` preserves parameters but a worker must not inherit
        the parent's autograd/training state assumptions.
        """
        self.__dict__.update(state)
        self.policy.eval()
        self.policy.to(self.device)


def make_agent_from_policy(
    policy: Any,
    vocab: dict,
    fixed_deck: list[int],
    *,
    device: str = "cpu",
    data_dir: Any = None,
) -> Callable[[dict], list[int]]:
    """Wrap an EMA Policy as a picklable ``agent(obs_dict)`` callable.

    Thin factory kept for API compatibility; see :class:`PolicyAgent`.  Pass
    *data_dir* (the mining output directory) so the agent can load the engine
    feature tables — without it every card feature it sees is zero.
    """
    return PolicyAgent(policy, vocab, fixed_deck, device=device, data_dir=data_dir)


def _sample_to_batch(sample: dict[str, np.ndarray], device: str) -> dict:
    """Convert a single numpy sample to a batch-1 torch dict."""
    import torch
    batch: dict[str, torch.Tensor] = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v).unsqueeze(0)  # add batch dim
            if v.dtype == np.int64:
                t = t.long()
            elif v.dtype == np.bool_:
                t = t.bool()
            else:
                t = t.float()
            batch[k] = t.to(device)
    return batch


# ============================================================
# Opponent agents
# ============================================================


def random_agent(obs_dict: dict) -> list[int]:
    """Return a uniformly random valid option from the current select."""
    select = obs_dict.get("select")
    if select is None:
        # Must return a 60-card deck; return a dummy one
        return list(range(60))
    options = select.get("option", [])
    if not options:
        return []
    # Pick random index
    import random
    idx = random.randint(0, len(options) - 1)
    return [idx]


def search_planner_agent(obs_dict: dict) -> list[int]:
    """Use the built-in ``search_begin/step`` planner backed by Rust MCTS.

    This is a strong baseline — the honest, non-trivial bar (Section 5).
    Loads ``libptcg_search.so`` (compiled from ``ptcg_search/``) and runs
    UCT/MCTS over the engine's determinized forward model via ``search_begin``
    and ``search_step``.  Falls back to a heuristic if the Rust library is
    unavailable.
    """
    select = obs_dict.get("select")
    if select is None:
        return _search_planner_deck()

    # Try the Rust MCTS library first
    result = _call_rust_search_planner(obs_dict)
    if result is not None:
        return result

    # Fallback: use Python-only search via the engine's cg API directly.
    # This is less efficient but works without the compiled Rust library.
    return _search_planner_fallback_python(obs_dict)


class SearchPlannerAgent:
    """:func:`search_planner_agent` with a belief-driven opponent model.

    The plain function leaves the Rust determinizer to assume the opponent
    mirrors our own deck.  Given an
    :class:`~ptcg_il.belief_infer.OpponentDeckOracle`, this variant predicts
    their decklist from the current observation instead, so the worlds MCTS
    searches are drawn from a deck the opponent might actually be playing.

    A class rather than a closure because live eval hands agents to a process
    pool: ``make_agent_from_policy``'s inner function could not be pickled, and
    a bound ``__call__`` can.
    """

    def __init__(self, oracle: Any = None, iterations: int = 200, seed: int = 42):
        self.oracle = oracle
        self.iterations = iterations
        self.seed = seed

    def __call__(self, obs_dict: dict) -> list[int]:
        select = obs_dict.get("select")
        if select is None:
            return _search_planner_deck()

        opp_deck = None
        if self.oracle is not None:
            opp_deck = self.oracle.predict(obs_dict) or None

        result = _call_rust_search_planner(
            obs_dict, opp_deck=opp_deck,
            iterations=self.iterations, seed=self.seed,
        )
        if result is not None:
            return result
        return _search_planner_fallback_python(obs_dict)


# ── Rust-backed search planner ──────────────────────────────────────────────

_rust_search_lib = None
_rust_search_error = None


def _load_rust_search_lib():
    """Lazily load ``libptcg_search.so``.  Returns the ctypes CDLL or None."""
    global _rust_search_lib, _rust_search_error
    if _rust_search_lib is not None:
        return _rust_search_lib
    if _rust_search_error is not None:
        return None

    import ctypes
    import os
    import sys

    # Search paths for the compiled .so
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "ptcg_search",
                     "target", "release", "libptcg_search.so"),
        os.path.join(os.path.dirname(__file__), "..", "ptcg_search",
                     "target", "debug", "libptcg_search.so"),
    ]

    lib = None
    for cand in candidates:
        if os.path.exists(os.path.normpath(cand)):
            try:
                lib = ctypes.CDLL(os.path.normpath(cand))
                break
            except OSError as e:
                _rust_search_error = str(e)
                continue

    if lib is None:
        _rust_search_error = _rust_search_error or "libptcg_search.so not found"
        logger.debug("Rust search planner unavailable: %s", _rust_search_error)
        return None

    # Define function signatures
    lib.search_plan.argtypes = [
        ctypes.c_char_p,  # obs_json
        ctypes.c_char_p,  # lib_path (to libcg.so)
        ctypes.c_char_p,  # fixed_deck_json
        ctypes.c_char_p,  # opp_deck_json
        ctypes.c_int,     # iterations
        ctypes.c_int,     # seed
        ctypes.c_int,     # host_initialized
    ]
    # c_void_p, NOT c_char_p.  With restype=c_char_p, ctypes eagerly copies the
    # returned char* into a Python bytes object and throws the original pointer
    # away.  Handing that bytes object back to search_plan_free() gives
    # CString::from_raw a pointer into Python's own heap, and glibc aborts the
    # process with "munmap_chunk(): invalid pointer".  c_void_p keeps the address
    # as a plain int so we can both read it and hand the real pointer back.
    lib.search_plan.restype = ctypes.c_void_p

    lib.search_plan_free.argtypes = [ctypes.c_void_p]
    lib.search_plan_free.restype = None

    _rust_search_lib = lib
    logger.info("Loaded Rust search planner from %s", cand if lib else "?")
    return lib


def _call_rust_search_planner(
    obs_dict: dict,
    opp_deck: list[int] | None = None,
    iterations: int = 200,
    seed: int = 42,
) -> list[int] | None:
    """Try to get an action from the Rust MCTS planner.  Returns None on failure.

    *opp_deck* is a 60-card opponent decklist for determinization; ``None`` (or
    an empty list) leaves the Rust side on its mirror-deck fallback.
    """
    plan = call_rust_search_plan(obs_dict, opp_deck, iterations, seed)
    if plan is None:
        return None
    return [int(i) for i in plan.get("indices", [])]


def call_rust_search_plan(
    obs_dict: dict,
    opp_deck: list[int] | None = None,
    iterations: int = 200,
    seed: int = 42,
) -> dict | None:
    """The full planner result, or ``None`` on any failure.

    ``{"indices": [...], "visit_counts": [[option, n], ...],
       "root_value": float | None, "iterations": int, "nodes_created": int}``

    :func:`_call_rust_search_planner` wants only ``indices``; RL_SPEC §8.3's
    search distillation wants ``visit_counts`` and ``root_value`` as well, so the
    FFI call lives here and the acting path is a thin projection of it.  Keeping
    one call site matters because the result string must be freed exactly once.

    ``root_value`` is ``None`` when the searcher built no tree (multi-select
    decisions), which is **not** the same as a value of ``0.0``.
    """
    lib = _load_rust_search_lib()
    if lib is None:
        return None

    import ctypes
    import json

    # Find libcg.so path
    cg_lib_path = _find_libcg_path()
    if cg_lib_path is None:
        logger.debug("Cannot find libcg.so — Rust planner requires it")
        return None

    try:
        obs_json = json.dumps(obs_dict, default=str)
        fixed_deck_json = json.dumps(_get_fixed_deck_for_search())
        opp_deck_json = json.dumps(opp_deck or [])  # empty → mirror fallback in Rust

        raw = lib.search_plan(
            obs_json.encode("utf-8"),
            cg_lib_path.encode("utf-8"),
            fixed_deck_json.encode("utf-8"),
            opp_deck_json.encode("utf-8"),
            iterations,
            seed,
            # host_initialized=1: importing cg.sim already ran GameInitialize() in
            # this process, and dlopen hands Rust back that same object.  A second
            # GameInitialize() throws the C++ std::runtime_error
            # "buffer full. capacity:7" across the FFI boundary, which Rust cannot
            # catch -- the worker dies with SIGABRT and every game in it is lost.
            1,
        )

        # `raw` is now an integer address (or None/0 for NULL) — see restype above.
        if not raw:
            return None

        result_str = ctypes.cast(raw, ctypes.c_char_p).value.decode("utf-8")
        lib.search_plan_free(raw)

        result = json.loads(result_str)
        if result.get("error"):
            logger.debug("Rust search planner error: %s", result["error"])
            return None

        return result

    except Exception as e:
        logger.debug("Rust search planner call failed: %s", e)
        return None


# ── Fallback: Python-only search via cg API ──────────────────────────────────

def _search_planner_fallback_python(obs_dict: dict) -> list[int]:
    """Python-only search using the engine's ``search_begin`` / ``search_step``
    API via ctypes.  Used when the Rust library is unavailable."""
    options = obs_dict.get("select", {}).get("option", [])
    if not options:
        return []

    # Single-step lookahead over all legal options using the engine's search API.
    # For each option, step forward once and evaluate the resulting state.
    try:
        return _python_search_one_step(obs_dict)
    except Exception as e:
        logger.warning("Python search planner error: %s — using first option", e)
        return [0]


def _python_search_one_step(obs_dict: dict) -> list[int]:
    """Try each legal option via search_step; pick the one leading to the best
    board (heuristic: opponent's prize cards remaining, or our HP advantage)."""
    import ctypes
    import json
    import os
    import random
    import sys
    from pathlib import Path

    # Ensure cg is importable
    engine_dir = str(
        Path(__file__).resolve().parent.parent
        / "pokemon-tcg-ai-battle"
        / "sample_submission"
        / "sample_submission"
    )
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)

    from cg.sim import lib
    from cg.api import search_begin, search_step, search_release, search_end
    from cg.api import to_observation_class, Observation

    obs = to_observation_class(obs_dict)
    select = obs.select
    if select is None:
        return list(range(60))

    options = select.option
    your_index = obs.current.yourIndex
    deck_size = obs.current.players[your_index].deckCount

    # Build guesses for hidden info (simplified fallback)
    fixed = _get_fixed_deck_for_search()
    your_deck = random.sample(fixed, min(deck_size, len(fixed))) if deck_size > 0 else []
    your_prize = random.sample(fixed, 6)
    opp_hand = random.sample(fixed, obs.current.players[1 - your_index].handCount)
    opp_deck = random.sample(fixed, obs.current.players[1 - your_index].deckCount)
    opp_prize = random.sample(fixed, 6)
    opp_active = []

    try:
        state = search_begin(obs, your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active)
    except (ValueError, RuntimeError) as e:
        logger.debug("search_begin failed in fallback: %s", e)
        return [0]

    root_id = state.searchId

    best_option = 0
    best_score = -1e9

    for i in range(min(len(options), 32)):  # cap at 32 options for speed
        try:
            child = search_step(root_id, [i])
        except (ValueError, RuntimeError):
            continue

        # Score: prefer states where opponent has fewer prizes (we're winning)
        child_obs = child.observation
        score = _score_state(child_obs, your_index)

        if score > best_score:
            best_score = score
            best_option = i

        try:
            search_release(child.searchId)
        except Exception:
            pass

    try:
        search_release(root_id)
    except Exception:
        pass

    return [best_option]


def _score_state(obs: dict, your_index: int) -> float:
    """Heuristic board evaluation for fallback search."""
    current = obs.get("current", {})
    players = current.get("players", [])
    if len(players) < 2:
        return 0.0

    me = players[your_index]
    opp = players[1 - your_index]

    # Prize advantage (fewer remaining = winning)
    my_prizes = len(me.get("prize", []))
    opp_prizes = len(opp.get("prize", []))

    # HP advantage on active Pokemon
    my_hp = _active_hp(me)
    opp_hp = _active_hp(opp)

    # Bench presence
    my_bench = len(me.get("bench", []))
    opp_bench = len(opp.get("bench", []))

    score = (
        (6.0 - my_prizes) * 10.0        # we want fewer prizes
        + (opp_prizes) * 10.0            # opponent having many prizes is good
        + (my_hp - opp_hp) * 0.1         # HP advantage
        + (my_bench - opp_bench) * 2.0   # bench advantage
    )
    return score


def _active_hp(player: dict) -> float:
    active = player.get("active", [])
    if active and active[0] is not None and isinstance(active[0], dict):
        return float(active[0].get("hp", 0))
    return 0.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_libcg_path() -> str | None:
    """Locate libcg.so on the filesystem."""
    import os
    from pathlib import Path

    candidates = [
        Path(__file__).resolve().parent.parent
        / "pokemon-tcg-ai-battle"
        / "sample_submission"
        / "sample_submission"
        / "cg"
        / "libcg.so",
        Path("cg") / "libcg.so",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return None


def _get_fixed_deck_for_search() -> list[int]:
    """Try to load FIXED_DECK, or return a dummy."""
    import json
    import os
    from pathlib import Path

    # "data" alone is cwd-relative, so invoking the CLI from anywhere but python/
    # silently fell through to the dummy deck below -- an *illegal* 60-card list
    # that makes every search_planner number meaningless without failing loudly.
    # Try the package-relative location too, so cwd stops mattering.
    candidates = []
    env_dir = os.environ.get("PTCG_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path("data"))
    candidates.append(Path(__file__).resolve().parent.parent / "data")

    for data_dir in candidates:
        arch_path = data_dir / "archetypes.json"
        if arch_path.exists():
            with open(arch_path) as f:
                arch = json.load(f)
            fd = arch.get("FIXED_DECK") or arch.get("fixed_deck")
            if fd:
                return fd

    logger.warning(
        "No FIXED_DECK found for search planner (looked in %s) — using dummy "
        "range(60), which is NOT a legal deck; search_planner results will be "
        "meaningless",
        ", ".join(str(c) for c in candidates),
    )
    return list(range(1, 61))


def _search_planner_deck() -> list[int]:
    """Return FIXED_DECK for deck-selection step."""
    return _get_fixed_deck_for_search()


# ============================================================
# Single-game runner (worker process)
# ============================================================


def _count_oov_opponent_cards(
    obs: dict, your_index: int, id_to_index: dict[int, int] | None
) -> tuple[int, int]:
    """Count opponent card IDs in *obs* that fall outside *id_to_index*.

    Returns ``(oov_count, total_count)`` over all visible opponent cards
    (active, bench, discard, stadium if owned by opponent).
    """
    if id_to_index is None:
        return 0, 0

    opp_index = 1 - your_index
    players = obs.get("players", [])
    if not players or opp_index >= len(players):
        return 0, 0

    oov = 0
    total = 0

    opp_state = players[opp_index]
    # Active pokemon — active is a list[Pokemon|None] (size 0 or 1)
    active_list = opp_state.get("active", [])
    for pokemon in active_list:
        if pokemon is not None and isinstance(pokemon, dict):
            cid = pokemon.get("id")
            if cid is not None:
                total += 1
                if int(cid) not in id_to_index:
                    oov += 1

    # Bench — list of Pokemon dicts
    for pokemon in opp_state.get("bench", []) or []:
        if isinstance(pokemon, dict):
            cid = pokemon.get("id")
            if cid is not None:
                total += 1
                if int(cid) not in id_to_index:
                    oov += 1

    # Discard — list of Card dicts (cg.api.Card: id, serial, playerIndex)
    for card in opp_state.get("discard", []) or []:
        if isinstance(card, dict):
            cid = card.get("id")
            if cid is not None:
                total += 1
                if int(cid) not in id_to_index:
                    oov += 1

    return oov, total


def _run_one_game(args: tuple) -> dict[str, Any]:
    """Run a single game between two agents.  Designed for use in a process pool.

    Parameters
    ----------
    args : tuple
        ``(deck0, deck1, agent0_fn, agent1_fn, seed, vocab_id_to_index)``.
        agent0_fn and agent1_fn must be importable callables (passed by name
        for pickling, or use a shared policy path for agent loading).
        vocab_id_to_index is a dict ``{card_id: index}`` used for OOV tracking;
        may be ``None`` to skip OOV counting.

    Returns
    -------
    dict
        ``{"winner": 0|1|-1, "steps": int, "illegal_actions": int,
           "oov_count": int, "total_opp_card_count": int}``.
        winner = -1 means error/abort.
    """
    _add_engine_path()
    from cg.game import battle_finish, battle_select, battle_start

    deck0, deck1, agent0, agent1, seed, *extra = args
    vocab_id_to_index = extra[0] if extra else None

    # Set per-process random seed for reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)

    illegal_actions = 0
    oov_count = 0
    total_opp_card_count = 0

    try:
        obs, start_data = battle_start(deck0, deck1)
        if obs is None:
            return {"winner": -1, "steps": 0, "illegal_actions": 0,
                    "oov_count": 0, "total_opp_card_count": 0,
                    "error": "battle_start returned None"}

        # Count OOV opponent cards in the initial observation
        initial_yi = _your_index(obs)
        oov, total = _count_oov_opponent_cards(obs, initial_yi, vocab_id_to_index)
        oov_count += oov
        total_opp_card_count += total

        steps = 0
        max_steps = 5000  # safety limit

        winner = -1  # default: unknown

        while True:
            select = obs.get("select")
            # Check for game end: current.result != -1 means the game is over.
            # result=0 → player 0 wins, result=1 → player 1 wins, result=2 → draw.
            current = obs.get("current", {})
            result = current.get("result", -1)
            if result != -1:
                if result in (0, 1):
                    winner = int(result)
                else:
                    winner = -1  # draw (result == 2) or unknown
                break

            # yourIndex lives on obs["current"], not at the top level — see
            # _your_index().  Reading it from the top level made agent0 play both
            # sides of every game.
            current_player = _your_index(obs)

            agent = agent0 if current_player == 0 else agent1

            try:
                action = agent(obs)
            except Exception:
                # Agent crashed — treat as illegal, pick first valid option
                options = select.get("option", []) if select is not None else []
                action = [0] if options else []
                illegal_actions += 1

            # Validate action is a list of ints
            if not isinstance(action, list) or not all(isinstance(x, int) for x in action):
                options = select.get("option", []) if select is not None else []
                action = [0] if options else []
                illegal_actions += 1

            try:
                obs = battle_select(action)
            except (IndexError, ValueError):
                # Illegal action — engine rejects.  Try first valid option.
                illegal_actions += 1
                if select is not None:
                    options = select.get("option", [])
                    if options:
                        try:
                            obs = battle_select([0])
                        except (IndexError, ValueError):
                            break
                    else:
                        break
                else:
                    break

            steps += 1

            # Count OOV opponent cards in the new observation
            new_yi = _your_index(obs)
            oov, total = _count_oov_opponent_cards(obs, new_yi, vocab_id_to_index)
            oov_count += oov
            total_opp_card_count += total

            if steps >= max_steps:
                break

    except Exception as e:
        logger.warning("Game error: %s", e)
        return {"winner": -1, "steps": 0, "illegal_actions": 0,
                "oov_count": 0, "total_opp_card_count": 0, "error": str(e)}
    finally:
        try:
            battle_finish()
        except Exception:
            pass

    return {
        "winner": winner,
        "steps": steps,
        "illegal_actions": illegal_actions,
        "oov_count": oov_count,
        "total_opp_card_count": total_opp_card_count,
    }


# ============================================================
# Live Evaluator
# ============================================================


@dataclass
class LiveResults:
    """Results from a live-eval run against one opponent."""
    opponent_name: str
    n_games: int
    wins: int
    losses: int
    draws: int
    errors: int
    win_rate_center: float
    win_rate_lo: float
    win_rate_hi: float
    mean_game_length: float
    total_illegal_actions: int
    total_oov_count: int
    total_opp_card_count: int

    @property
    def oov_rate(self) -> float:
        """Fraction of opponent card occurrences that fell outside the vocab."""
        if self.total_opp_card_count == 0:
            return 0.0
        return self.total_oov_count / self.total_opp_card_count


class LiveEvaluator:
    """Run games between a policy agent and baseline opponents.

    Parameters
    ----------
    policy : nn.Module
        EMA policy (should already be on CPU and in eval mode).
    vocab : dict
        Vocab dict with ``id_to_index`` and ``attack_id_to_index``.
    fixed_deck : list[int]
        60-card FIXED_DECK.
    n_workers : int
        Number of concurrent game processes.  Default: ``os.cpu_count()``.
    """

    def __init__(
        self,
        policy: Any,
        vocab: dict,
        fixed_deck: list[int],
        n_workers: int | None = None,
        data_dir: Any = None,
    ):
        self.policy = policy
        self.vocab = vocab
        self.fixed_deck = fixed_deck
        self.n_workers = n_workers or (os.cpu_count() or 4)
        self.data_dir = data_dir

        # Extract id_to_index for OOV tracking in worker processes
        raw = vocab.get("id_to_index", {})
        self._vocab_id_to_index = {int(k): int(v) for k, v in raw.items()}

        # Build the agent wrapper (CPU only — each process gets its own copy).
        # data_dir carries the engine feature tables; see PolicyAgent._tables.
        self._agent_fn = make_agent_from_policy(
            policy, vocab, fixed_deck, device="cpu", data_dir=data_dir)

    def eval_vs_opponent(
        self,
        opponent_agent: Callable[[dict], list[int]],
        opponent_name: str,
        n_games: int = 500,
        seed: int = 42,
    ) -> LiveResults:
        """Run *n_games* vs a single opponent and return results.

        Half the games use our agent as player 0, half as player 1, to
        control for first-player advantage.
        """
        decks = [self.fixed_deck, self.fixed_deck]  # both use FIXED_DECK for now

        # Generate game seeds
        rng = np.random.default_rng(seed)
        game_seeds = rng.integers(0, 2**31 - 1, size=n_games).tolist()

        wins = 0
        losses = 0
        draws = 0
        errors = 0
        total_steps = 0
        total_illegal = 0
        total_oov = 0
        total_opp_cards = 0
        games_completed = 0

        # Run games in process pool
        jobs = []
        for i in range(n_games):
            # Alternate our agent as player 0 vs player 1
            if i % 2 == 0:
                agent0 = self._agent_fn
                agent1 = opponent_agent
                our_player = 0
            else:
                agent0 = opponent_agent
                agent1 = self._agent_fn
                our_player = 1

            jobs.append((decks[0], decks[1], agent0, agent1, game_seeds[i], our_player, i, self._vocab_id_to_index))

        # MUST be "spawn", not the Linux default "fork".
        #
        # By this point the parent has already run torch forward passes (offline
        # eval, building the frozen opponent), so torch's OpenMP thread pool is
        # initialised.  fork() copies the memory but not the threads, leaving the
        # child holding locks whose owners no longer exist -- the first torch op in
        # the worker then blocks on a futex forever, at 0% CPU, and the pool's own
        # shutdown blocks behind it.  Measured: fork deadlocks, spawn runs 4 jobs
        # in 0.6 s.
        #
        # spawn requires every job argument to be picklable, which is why the agents
        # are classes rather than closures (see PolicyAgent).
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=ctx,
            initializer=_worker_init,
        ) as executor:
            futures = {
                executor.submit(_run_one_game_worker, job): job[6]
                for job in jobs
            }
            for future in as_completed(futures):
                game_idx = futures[future]
                try:
                    result = future.result(timeout=120)
                except Exception as e:
                    logger.warning("Game %d failed: %s", game_idx, e)
                    errors += 1
                    continue

                winner = result.get("winner", -1)
                if winner == -1 and "error" in result:
                    errors += 1
                    continue

                # Map winner to our agent's perspective
                our_player_val = jobs[game_idx][5] if game_idx < len(jobs) else -1
                if our_player_val == -1:
                    continue

                if winner == -1:
                    draws += 1
                elif winner == our_player_val:
                    wins += 1
                elif winner != our_player_val and winner in (0, 1):
                    losses += 1
                else:
                    draws += 1

                total_steps += result.get("steps", 0)
                total_illegal += result.get("illegal_actions", 0)
                total_oov += result.get("oov_count", 0)
                total_opp_cards += result.get("total_opp_card_count", 0)
                games_completed += 1

        # Wilson interval
        center, lo, hi = wilson_interval(wins, games_completed)

        return LiveResults(
            opponent_name=opponent_name,
            n_games=games_completed,
            wins=wins,
            losses=losses,
            draws=draws,
            errors=errors,
            win_rate_center=center,
            win_rate_lo=lo,
            win_rate_hi=hi,
            mean_game_length=total_steps / max(games_completed, 1),
            total_illegal_actions=total_illegal,
            total_oov_count=total_oov,
            total_opp_card_count=total_opp_cards,
        )

    def eval_vs_opponents(
        self,
        opponents: dict[str, Callable[[dict], list[int]]],
        n_games: int = 500,
        seed: int = 42,
    ) -> dict[str, LiveResults]:
        """Run *n_games* against each opponent and return named results.

        Parameters
        ----------
        opponents : dict
            ``{"random": random_agent, "search": search_planner_agent, ...}``.
        n_games : int
            Games per opponent.
        seed : int
            Base seed (incremented per opponent for different game seeds).

        Returns
        -------
        dict
            ``{name: LiveResults}``.
        """
        all_results: dict[str, LiveResults] = {}
        for i, (name, opp) in enumerate(opponents.items()):
            logger.info("Live eval vs %s: %d games...", name, n_games)
            start = time.perf_counter()
            results = self.eval_vs_opponent(opp, name, n_games=n_games, seed=seed + i * 1000)
            elapsed = time.perf_counter() - start
            # Always report completed/requested and the failure count.  A win rate
            # computed from a handful of surviving games is indistinguishable from a
            # valid one unless the dead games are visible: a pickling bug once killed
            # *every* game and the summary still printed a confident-looking "39.7%".
            # Escalate to WARNING when anything died so it cannot be skimmed past.
            log = logger.warning if results.errors else logger.info
            log(
                "  %s: %.1f%% [%.1f–%.1f%%], %d/%d games in %.1fs (%d failed)",
                name,
                results.win_rate_center * 100,
                results.win_rate_lo * 100,
                results.win_rate_hi * 100,
                results.n_games,
                n_games,
                elapsed,
                results.errors,
            )
            all_results[name] = results
        return all_results

    def to_wandb_metrics(
        self, results: dict[str, LiveResults]
    ) -> dict[str, float]:
        """Convert LiveResults to a flat W&B-ready metrics dict (C.9)."""
        metrics: dict[str, float] = {}
        for name, r in results.items():
            metrics[f"live/winrate@{name}"] = r.win_rate_center
            metrics[f"live/winrate_ci_lo@{name}"] = r.win_rate_lo
            metrics[f"live/winrate_ci_hi@{name}"] = r.win_rate_hi
            metrics[f"live/game_len_mean@{name}"] = r.mean_game_length
            metrics[f"live/illegal_action_rate@{name}"] = (
                r.total_illegal_actions / max(r.n_games, 1)
            )
            metrics[f"live/oov_rate@{name}"] = r.oov_rate
        return metrics


# ============================================================
# Worker entry point (for pickle-based process pool)
# ============================================================


def _worker_init() -> None:
    """Per-worker setup, run once when a spawned process starts.

    Pins torch to a single thread.  Workers are already the unit of parallelism, so
    letting each of N processes start its own OpenMP pool oversubscribes the machine
    badly (16 workers x 8 threads on a 16-core box) and makes every worker slower
    than if it were alone.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # pragma: no cover - torch always present in practice
        pass


def _run_one_game_worker(args: tuple) -> dict[str, Any]:
    """Unpack args and run a single game.  This top-level function is required
    for ProcessPoolExecutor pickle-based dispatch."""
    (
        deck0, deck1,
        agent0, agent1,
        seed, our_player, game_idx,
        vocab_id_to_index,
    ) = args
    return _run_one_game((deck0, deck1, agent0, agent1, seed, vocab_id_to_index))
