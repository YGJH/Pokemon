"""Parallel rollout over the real engine (RL_SPEC §6).

**Division of labour.**  Worker processes drive ``libcg`` and nothing else.  They
send each observation to the parent and block for an action.  The parent
featurizes, batches, runs one forward pass, and replies.

That split follows from the measurements in §2.2 rather than from taste:

* Putting the *policy* in the workers means CPU inference, ~100× slower than the
  GPU per decision, and 6–8 hours of rollout for an R2-sized run.
* Putting the *featurizer* in the workers means shipping ~50 KB of feature
  arrays per decision instead of a ~4 KB observation, and a second featurizer
  call site whose output must match the first bit-for-bit — the exact parity
  hazard §6.4 calls "high, silent".

With engine-only workers the parent pays 60.5 µs featurize + 16 µs inference per
decision, or ~13k dec/s — comfortably more than an R2 iteration needs, and it
leaves exactly one featurizer in the system.

**No Rust here.**  §6.5 downgraded the Rust `VecEnv` port from prerequisite to
optional after R0d made the NumPy featurizer 3.57× faster: six workers already
saturate the GPU, and the machine has sixteen cores.

The engine is injectable so unit tests need no ``libcg.so``, following the
pattern ``ptcg_mine/download.py`` uses for the Kaggle API.
"""

from __future__ import annotations

import json
import logging
from rich.logging import RichHandler
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_time=False)])
logger = logging.getLogger(__name__)

# Worker → parent
MSG_ACT = "act"
MSG_DONE = "done"
MSG_ERROR = "error"
# Parent → worker
MSG_PICKS = "picks"
MSG_STOP = "stop"

# A game that exceeds this many decisions is treated as a timeout loss.  Real
# games run ~150 decisions; this is a hang detector, not a budget.
MAX_DECISIONS = 2_000


@dataclass
class Decision:
    """One recorded decision point, with everything PPO needs to replay it."""

    features: dict[str, np.ndarray]
    """The featurizer output — the state the policy actually saw."""
    picks: list[int]
    """Engine-facing option indices, STOP excluded."""
    action_idx: np.ndarray
    """AR targets including the STOP step, ``-1``-padded — what
    ``recompute_logp`` teacher-forces."""
    action_len: int
    logp: float
    """Joint log-probability under the behaviour policy.  Kept for diagnostics
    only: §7 requires the PPO update to *recompute* this in its own precision
    rather than trust a stored value."""
    value: float
    turn: int
    """Decision index within the game.  R1 checks that value accuracy rises with
    it; a critic that is no better late than early has learned nothing."""
    opp_id: int = -1
    """Archetype id of the opponent deck in *this* battle.

    A pool used to hold one opponent, so the archetype was a property of the
    whole call; with mixed-opponent pools it varies per battle, and both the
    seat-1 pilot and the MCTS ``opp_deck_template`` are chosen from it.  ``-1``
    means the environment did not report one."""
    opp_visible_card_ids: list[int] | None = None
    """Engine card ids of opponent cards visible at this decision point
    (active, bench, discard, face-up prizes).  Feeds the Bayesian
    ArchetypePosterior for MCTS determinization (Phase 3c)."""
    obs_json: str | None = None
    """Raw observation JSON from the engine at this decision point.
    Preserved for MCTS distillation (Phase 3d): the determinizer needs
    ``search_begin_input``, ``current``, and ``select`` fields to
    construct search roots."""
    your_index: int = 0
    """Engine seat of the player who was to move here (``current.yourIndex``).

    The observation is egocentric — the featurizer indexes every zone as
    ``[your_index, 1 - your_index]`` — so the seat is *not* recoverable from
    ``features``, while the engine reward is reported from one fixed seat's
    perspective.  Without this field a decision cannot be paired with the
    outcome of the player who actually made it, and the value target for the
    opposite seat is sign-inverted.  ``puct.rs`` expand_leaf assumes the
    critic speaks in the perspective of the player to move; this is what lets
    training honour that contract."""


def _extract_opp_visible_card_ids(obs: dict) -> list[int]:
    """Return engine card ids of visible opponent cards from an observation.

    Covers active, bench, discard, and face-up prize cards.  Face-down
    prizes and deck cards are excluded (they are hidden).
    """
    current = obs.get("current", {})
    your_idx = current.get("yourIndex", 0)
    opp_idx = 1 - your_idx
    players = current.get("players", [])
    if opp_idx >= len(players):
        return []
    opp = players[opp_idx]

    ids: list[int] = []

    # Active Pokémon
    for poke in (opp.get("active") or []):
        if isinstance(poke, dict) and "id" in poke:
            ids.append(poke["id"])
            # Pre-evolutions
            for pre in (poke.get("preEvolution") or []):
                if isinstance(pre, dict) and "id" in pre:
                    ids.append(pre["id"])

    # Bench Pokémon
    for poke in (opp.get("bench") or []):
        if isinstance(poke, dict) and "id" in poke:
            ids.append(poke["id"])
            for pre in (poke.get("preEvolution") or []):
                if isinstance(pre, dict) and "id" in pre:
                    ids.append(pre["id"])

    # Discard pile
    for card in (opp.get("discard") or []):
        if isinstance(card, dict) and "id" in card:
            ids.append(card["id"])

    # Face-up prize cards (only non-null entries)
    for card in (opp.get("prize") or []):
        if isinstance(card, dict) and "id" in card:
            ids.append(card["id"])

    # Stadium (if any)
    stadium = current.get("stadium") or []
    for card in stadium:
        if isinstance(card, dict) and "id" in card:
            ids.append(card["id"])

    return ids


@dataclass
class Trajectory:
    """One game from our player's point of view."""

    decisions: list[Decision] = field(default_factory=list)
    reward: float = 0.0
    """Terminal reward: +1 win, −1 loss, 0 no winner.  All other steps are 0."""
    timeout: bool = False
    """Budget exceeded.  Recorded as a **loss**, never discarded — dropping these
    games would silently select for slow policies (§4.2)."""
    error: str | None = None
    battle_idx: int = -1
    """Rust battle index — used to match drain results to pending trajectories."""
    opp_id: int = -1
    """Archetype id of the opponent deck this game was played against."""

    def __len__(self) -> int:
        return len(self.decisions)


# ── Worker ──────────────────────────────────────────────────────────────────


def _default_engine():
    """Import the real engine.  Called inside the worker, never at module import."""
    from ptcg_il.live_eval import _add_engine_path

    _add_engine_path()
    from cg.game import battle_finish, battle_select, battle_start

    return battle_start, battle_select, battle_finish


def env_worker(
    conn,
    deck0: Sequence[int],
    deck1: Sequence[int],
    our_player: int,
    seed: int,
    engine_factory: Callable[[], tuple] = _default_engine,
    max_decisions: int = MAX_DECISIONS,
) -> None:
    """Play games forever, asking the parent for every action.

    Both sides are driven by the parent's policy (mirror self-play), but only
    ``our_player``'s decisions are tagged for recording — the parent decides
    what to keep.  Sending the opponent's decisions too is what makes the
    opponent an actual policy rather than a random agent.
    """
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))

    try:
        battle_start, battle_select, battle_finish = engine_factory()
    except Exception as e:  # pragma: no cover - requires a broken install
        conn.send((MSG_ERROR, f"engine import failed: {e}"))
        return

    from ptcg_il.live_eval import _your_index

    while True:
        traj_meta: dict[str, Any] = {"n_decisions": 0, "timeout": False, "error": None}
        winner = -1
        try:
            obs, _start = battle_start(list(deck0), list(deck1))
            if obs is None:
                conn.send((MSG_DONE, {"winner": -1, "error": "battle_start returned None",
                                      **traj_meta}))
                if not _await_continue(conn):
                    return
                continue

            while True:
                current = obs.get("current", {}) or {}
                result = current.get("result", -1)
                if result != -1:
                    winner = int(result) if result in (0, 1) else -1
                    break

                if traj_meta["n_decisions"] >= max_decisions:
                    traj_meta["timeout"] = True
                    break

                actor = _your_index(obs)
                conn.send((MSG_ACT, {"obs": obs, "actor": actor,
                                     "record": actor == our_player}))
                tag, payload = conn.recv()
                if tag == MSG_STOP:
                    # Return, do NOT finish here: the `finally` below already
                    # does. Calling battle_finish twice frees the same battle
                    # pointer twice, which glibc reports as "double free or
                    # corruption" — a native abort that no Python except can
                    # catch and that can take the worker down mid-shutdown.
                    return

                picks = list(payload)
                traj_meta["n_decisions"] += 1
                try:
                    obs = battle_select(picks)
                except (IndexError, ValueError) as e:
                    # An illegal pick means the masking is wrong, which is a bug
                    # worth surfacing rather than papering over with option 0 as
                    # live_eval does — that would train the policy on an action
                    # it did not choose.
                    traj_meta["error"] = f"engine rejected {picks}: {e}"
                    break

        except Exception as e:
            traj_meta["error"] = str(e)
        finally:
            _safe_finish(battle_finish)

        conn.send((MSG_DONE, {"winner": winner, **traj_meta}))
        if not _await_continue(conn):
            return


def _await_continue(conn) -> bool:
    """Block for the parent's go-ahead. ``False`` means shut down."""
    tag, _ = conn.recv()
    return tag != MSG_STOP


def _safe_finish(battle_finish) -> None:
    try:
        battle_finish()
    except Exception:
        pass


# ── Parent ──────────────────────────────────────────────────────────────────


class RolloutPool:
    """A pool of engine-driving workers served by one batched policy.

    Parameters
    ----------
    deck_self, deck_opp : sequence of int
        The two 60-card decklists.
    n_workers : int
        Worker processes.  Six saturate the GPU for base rollout (§6.5); more
        only helps once MCTS is in the loop, which R2 does not have.
    forward_batch : int
        Maximum decisions per forward pass.  Throughput saturates at 256–512
        and larger batches only add latency (§2.3).
    """

    def __init__(
        self,
        deck_self: Sequence[int],
        deck_opp: Sequence[int],
        *,
        n_workers: int = 6,
        forward_batch: int = 512,
        seed: int = 0,
        our_player: int = 0,
        engine_factory: Callable[[], tuple] | None = None,
        max_decisions: int = MAX_DECISIONS,
    ):
        self.deck_self = list(deck_self)
        self.deck_opp = list(deck_opp)
        self.n_workers = n_workers
        self.forward_batch = forward_batch
        self.our_player = our_player
        self.seed = seed
        self.max_decisions = max_decisions
        self._engine_factory = engine_factory or _default_engine
        self._procs: list[mp.Process] = []
        self._conns: list[Any] = []
        # Trajectory under construction, per worker.
        self._open: list[Trajectory] = []
        # Workers whose pipe has closed. Retired rather than retried — see _sweep.
        self._dead: set[int] = set()

    # -- lifecycle --

    def start(self) -> None:
        if self._procs:
            raise RuntimeError("pool already started")
        # 'spawn' rather than 'fork': the parent holds CUDA context and a forked
        # child inheriting it is undefined behaviour in every CUDA version.
        ctx = mp.get_context("spawn")
        for i in range(self.n_workers):
            parent_conn, child_conn = ctx.Pipe()
            deck0, deck1 = self.deck_self, self.deck_opp
            our = self.our_player
            p = ctx.Process(
                target=env_worker,
                args=(child_conn, deck0, deck1, our, self.seed + i,
                      self._engine_factory, self.max_decisions),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self._procs.append(p)
            self._conns.append(parent_conn)
        self._open = [Trajectory() for _ in range(self.n_workers)]

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send((MSG_STOP, None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._procs.clear()
        self._conns.clear()
        self._dead.clear()

    def __enter__(self) -> "RolloutPool":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- collection --

    def collect(self, act_fn: Callable[[list[dict]], list[dict]], n_decisions: int
                ) -> list[Trajectory]:
        """Gather at least *n_decisions* recorded decision points.

        *act_fn* takes a list of pending requests (each ``{"obs", "actor",
        "record"}``) and returns, per request, a dict with at least ``picks``;
        recorded requests additionally carry ``features``, ``action_idx``,
        ``action_len``, ``logp`` and ``value``.  Batching it this way is what
        lets one GPU forward pass serve every worker that is currently waiting.

        Finished games are returned whole.  Trajectories are never truncated
        mid-game: GAE needs the terminal reward, and a partial game has none.
        """
        if not self._procs:
            raise RuntimeError("pool not started; use `with RolloutPool(...) as pool`")

        import time as _perf_time
        done: list[Trajectory] = []
        collected = 0

        self._perf_t_sweep = 0.0
        self._perf_t_act = 0.0
        self._perf_n_act = 0

        while collected < n_decisions:
            _ts = _perf_time.perf_counter()
            pending_idx, requests, finished = self._sweep()
            self._perf_t_sweep += _perf_time.perf_counter() - _ts
            for traj in finished:
                done.append(traj)
                collected += len(traj)

            if not pending_idx:
                if not requests and not finished and not self._any_alive():
                    break
                continue

            _ta = _perf_time.perf_counter()
            replies = act_fn(requests)
            self._perf_t_act += _perf_time.perf_counter() - _ta
            self._perf_n_act += 1
            if len(replies) != len(requests):
                raise RuntimeError(
                    f"act_fn returned {len(replies)} replies for {len(requests)} "
                    f"requests; every waiting worker must get exactly one action"
                )

            for w, req, rep in zip(pending_idx, requests, replies):
                self._conns[w].send((MSG_PICKS, rep["picks"]))
                if req["record"]:
                    self._open[w].decisions.append(
                        Decision(
                            features=rep["features"],
                            picks=list(rep["picks"]),
                            action_idx=rep["action_idx"],
                            action_len=int(rep["action_len"]),
                            logp=float(rep["logp"]),
                            value=float(rep["value"]),
                            turn=len(self._open[w].decisions),
                            opp_visible_card_ids=_extract_opp_visible_card_ids(
                                req["obs"]
                            ) if req.get("obs") else None,
                            obs_json=json.dumps(req["obs"]) if req.get("obs") else None,
                        )
                    )

        return done

    def _sweep(self) -> tuple[list[int], list[dict], list[Trajectory]]:
        """One pass: block for ready workers, then read exactly one message each.

        Every worker strictly alternates send-then-receive, so a connection that
        is ready has exactly one message waiting.  That invariant is what lets
        this be a single non-looping read per worker — and it is why the worker
        must never send twice in a row.

        Returns ``(indices awaiting an action, their requests, finished games)``.
        """
        import multiprocessing.connection as mpc

        # A dead worker's pipe stays permanently readable at EOF, so `wait` keeps
        # returning it and `recv` keeps raising. `collect` still terminates —
        # `_any_alive` covers the all-dead case — but every sweep would waste a
        # `forward_batch` slot on each corpse, and with enough of them the live
        # workers get starved out of the batch entirely. Retire them instead.
        conns = [
            c for w, (c, p) in enumerate(zip(self._conns, self._procs))
            if w not in self._dead and (p.is_alive() or c.poll())
        ]
        if not conns:
            return [], [], []

        ready = mpc.wait(conns)
        by_conn = {id(c): w for w, c in enumerate(self._conns)}

        idx: list[int] = []
        reqs: list[dict] = []
        finished: list[Trajectory] = []

        # Workers beyond the batch limit stay queued and are read next sweep;
        # nothing is dropped, only deferred.
        for conn in ready[: self.forward_batch]:
            w = by_conn[id(conn)]
            try:
                tag, payload = conn.recv()
            except (EOFError, OSError):
                logger.warning("rollout worker %d closed its pipe; retiring it", w)
                self._dead.add(w)
                continue
            if tag == MSG_ACT:
                idx.append(w)
                reqs.append(payload)
            elif tag == MSG_DONE:
                traj = self._close_game(w, payload)
                if traj is not None:
                    finished.append(traj)
            elif tag == MSG_ERROR:
                logger.error("rollout worker %d: %s", w, payload)

        return idx, reqs, finished

    def _any_alive(self) -> bool:
        return any(
            p.is_alive() for w, p in enumerate(self._procs) if w not in self._dead
        )

    def _close_game(self, w: int, payload: dict) -> Trajectory | None:
        traj = self._open[w]
        traj.timeout = bool(payload.get("timeout"))
        traj.error = payload.get("error")

        winner = payload.get("winner", -1)
        if traj.timeout:
            # A timeout is a loss.  Discarding the game instead would make slow
            # policies invisible to the objective and therefore selected for.
            traj.reward = -1.0
        elif winner == self.our_player:
            traj.reward = 1.0
        elif winner in (0, 1):
            traj.reward = -1.0
        else:
            traj.reward = 0.0

        self._open[w] = Trajectory()
        # Tell the worker to start the next game.  This is the reply half of the
        # worker's strict send/recv alternation, not an action.
        try:
            self._conns[w].send((MSG_PICKS, []))
        except (BrokenPipeError, OSError):
            pass
        return traj if traj.decisions else None
