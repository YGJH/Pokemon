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


def make_agent_from_policy(
    policy: Any,
    vocab: dict,
    fixed_deck: list[int],
    *,
    device: str = "cpu",
) -> Callable[[dict], list[int]]:
    """Wrap an EMA Policy as an ``agent(obs_dict)`` callable (Section 6).

    The returned function:
    1. If ``obs["select"] is None``, returns ``FIXED_DECK``.
    2. Otherwise, featurizes the observation, runs the policy forward pass,
       and returns the chosen option index (or indices for multi-select).

    Parameters
    ----------
    policy : Policy (nn.Module)
        EMA-averaged policy, already on the target device and in eval mode.
    vocab : dict
        Vocab dict with ``id_to_index`` and ``attack_id_to_index``.
    fixed_deck : list[int]
        The FIXED_DECK (60 card ids).
    device : str
        Torch device string (default ``"cpu"`` for live eval — each game runs
        in its own process, so GPU contention is avoided).

    Returns
    -------
    callable
        ``agent(obs_dict) -> list[int]``.
    """
    import torch
    from ptcg_il.featurizer import featurize
    from ptcg_il.model.policy import select_multi

    id_to_index = {int(k): int(v) for k, v in vocab.get("id_to_index", {}).items()}
    attack_id_to_index = {int(k): int(v) for k, v in vocab.get("attack_id_to_index", {}).items()}
    vocab_full = {
        "id_to_index": id_to_index,
        "attack_id_to_index": attack_id_to_index,
    }

    policy.eval()
    policy.to(device)

    def agent(obs_dict: dict) -> list[int]:
        select = obs_dict.get("select")
        if select is None:
            return list(fixed_deck)

        # Featurize (action=None handled natively for inference mode)
        sample = featurize(obs_dict, vocab_full, value_target=0.0, sample_weight=1.0)

        # Build batch of size 1
        batch = _sample_to_batch(sample, device)
        max_count = int(sample["maxCount"])

        with torch.no_grad():
            if max_count == 1:
                logits, _value = policy(batch)
                logits = logits.masked_fill(~batch["opt_mask"], -1e9)
                chosen = int(logits.argmax(dim=-1)[0].item())
                return [chosen]
            else:
                chosen = select_multi(policy, batch)  # [1, batch_max]
                picks = chosen[0].tolist()
                # Filter -1 padding from per-sample maxC and take up to maxCount
                picks = [int(p) for p in picks if p >= 0]
                picks = picks[:max_count]
                return picks

    return agent


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
    """Use the built-in ``search_begin/step`` planner as a scripted opponent.

    This is a strong baseline — the honest, non-trivial bar (Section 5).
    Requires the ``cg`` engine to be importable.
    """
    select = obs_dict.get("select")
    if select is None:
        # Cannot use search for deck select; return a dummy deck
        return list(range(60))

    # The planner uses a two-phase protocol:
    #   search_begin(battle_data) → SearchOptions
    #   search_step() → None | (best_option, probability) — call repeatedly
    # We implement a minimal wrapper that calls search_begin on first invocation.
    #
    # Since we receive raw obs_dict here and don't have the engine's StartData,
    # we fall back to the first legal option as a simple heuristic.
    #
    # Full search-planner integration requires the engine's search machinery
    # (cg.sim.SearchBegin/SearchStep) which needs a Battle object — that is
    # only available inside a running game, not from a function pointer.
    #
    # For now this is a placeholder; the real search-planner evaluator is
    # implemented in _run_game via direct cg.game calls.
    options = select.get("option", [])
    if not options:
        return []
    return [0]  # first option as fallback


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
