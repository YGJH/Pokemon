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
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

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
    ) -> None:
        from ptcg_il.featurizer import normalize_vocab

        # normalize_vocab is required: vocab.json keys are strings, engine card ids
        # are ints, so the raw dict silently maps every card to UNKNOWN_CARD.
        self.vocab_full = normalize_vocab(vocab)
        self.fixed_deck = list(fixed_deck)
        self.device = device

        policy.eval()
        policy.to(device)
        self.policy = policy

    def __call__(self, obs_dict: dict) -> list[int]:
        import torch
        from ptcg_il.featurizer import featurize
        from ptcg_il.model.policy import select_multi

        select = obs_dict.get("select")
        if select is None:
            return list(self.fixed_deck)

        # Featurize (action=None handled natively for inference mode)
        sample = featurize(
            obs_dict, self.vocab_full, value_target=0.0, sample_weight=1.0)

        # Build batch of size 1
        batch = _sample_to_batch(sample, self.device)
        max_count = int(sample["maxCount"])

        with torch.no_grad():
            if max_count == 1:
                logits, _value, _hist = self.policy(batch)
                logits = logits.masked_fill(~batch["opt_mask"], -1e9)
                chosen = int(logits.argmax(dim=-1)[0].item())
                return [chosen]
            else:
                chosen = select_multi(self.policy, batch)  # [1, batch_max]
                picks = chosen[0].tolist()
                # Filter STOP (-2) and padding (-1); keep only regular picks
                picks = [int(p) for p in picks if p >= 0]
                picks = picks[:max_count]
                return picks

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
) -> Callable[[dict], list[int]]:
    """Wrap an EMA Policy as a picklable ``agent(obs_dict)`` callable.

    Thin factory kept for API compatibility; see :class:`PolicyAgent`.
    """
    return PolicyAgent(policy, vocab, fixed_deck, device=device)


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
    ]
    lib.search_plan.restype = ctypes.c_char_p

    lib.search_plan_free.argtypes = [ctypes.c_char_p]
    lib.search_plan_free.restype = None

    _rust_search_lib = lib
    logger.info("Loaded Rust search planner from %s", cand if lib else "?")
    return lib


def _call_rust_search_planner(obs_dict: dict) -> list[int] | None:
    """Try to get an action from the Rust MCTS planner.  Returns None on failure."""
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
        opp_deck_json = json.dumps([])  # empty → mirror fallback in Rust

        raw = lib.search_plan(
            obs_json.encode("utf-8"),
            cg_lib_path.encode("utf-8"),
            fixed_deck_json.encode("utf-8"),
            opp_deck_json.encode("utf-8"),
            200,  # iterations
            42,   # seed
        )

        if raw is None:
            return None

        result_str = ctypes.c_char_p(raw).value.decode("utf-8")
        lib.search_plan_free(raw)

        result = json.loads(result_str)
        if result.get("error"):
            logger.debug("Rust search planner error: %s", result["error"])
            return None

        indices = result.get("indices", [])
        return [int(i) for i in indices]

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

    data_dir = Path(os.environ.get("PTCG_DATA_DIR", "data"))
    arch_path = data_dir / "archetypes.json"
    if arch_path.exists():
        with open(arch_path) as f:
            arch = json.load(f)
        fd = arch.get("FIXED_DECK") or arch.get("fixed_deck")
        if fd:
            return fd

    logger.warning("No FIXED_DECK found for search planner — using dummy range(60)")
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
        initial_yi = obs.get("yourIndex", 0)
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

            # Read yourIndex reliably from the observation dict — the engine
            # always includes this field per the State dataclass (cg.api).
            current_player = obs.get("yourIndex", 0)

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
            new_yi = obs.get("yourIndex", 0)
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
    ):
        self.policy = policy
        self.vocab = vocab
        self.fixed_deck = fixed_deck
        self.n_workers = n_workers or (os.cpu_count() or 4)

        # Extract id_to_index for OOV tracking in worker processes
        raw = vocab.get("id_to_index", {})
        self._vocab_id_to_index = {int(k): int(v) for k, v in raw.items()}

        # Build the agent wrapper (CPU only — each process gets its own copy)
        self._agent_fn = make_agent_from_policy(policy, vocab, fixed_deck, device="cpu")

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

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
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
            logger.info(
                "  %s: %.1f%% [%.1f–%.1f%%], %d games in %.1fs",
                name,
                results.win_rate_center * 100,
                results.win_rate_lo * 100,
                results.win_rate_hi * 100,
                results.n_games,
                elapsed,
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
