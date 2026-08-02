"""Rust-backed parallel game environment (via C bridge, fixed ABI).

N battles managed by Rust with direct libcg FFI.  Python only
featurizes observations and runs GPU forwards.
"""

from __future__ import annotations

import ctypes
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from ptcg_rl.vec_env import Decision, Trajectory

logger = logging.getLogger(__name__)


def _find_libptcg_search() -> str:
    base = Path(__file__).resolve().parent.parent
    for target in ["release", "debug"]:
        p = base / f"ptcg_search/target/{target}/libptcg_search.so"
        if p.exists():
            return str(p)
    raise FileNotFoundError("libptcg_search.so not found; run: cargo build --release")


def _find_libcg() -> str:
    base = Path(__file__).resolve().parent.parent.parent
    candidates = [
        base / "python/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/libcg.so",
        base / "pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/libcg.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("libcg.so not found")


def _load_search_lib() -> ctypes.CDLL:
    path = _find_libptcg_search()
    lib = ctypes.CDLL(path)

    lib.vec_env_create.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.vec_env_create.restype = ctypes.c_int64

    lib.vec_env_create_multi.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.vec_env_create_multi.restype = ctypes.c_int64

    lib.vec_env_poll.argtypes = [ctypes.c_int64]
    lib.vec_env_poll.restype = ctypes.c_void_p

    lib.vec_env_reply.argtypes = [ctypes.c_int64, ctypes.c_char_p]
    lib.vec_env_reply.restype = ctypes.c_int

    lib.vec_env_drain.argtypes = [ctypes.c_int64]
    lib.vec_env_drain.restype = ctypes.c_void_p

    lib.vec_env_free.argtypes = [ctypes.c_int64]
    lib.vec_env_free.restype = None

    lib.puct_free_result.argtypes = [ctypes.c_void_p]
    lib.puct_free_result.restype = None

    return lib


def _read_and_free(lib, ptr) -> str:
    if ptr is None or ptr == 0:
        return ""
    try:
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
    finally:
        lib.puct_free_result(ptr)


class RustVecEnv:
    """N parallel battles in Rust via C bridge — no GIL, no IPC."""

    # Class-level defaults so the counters exist even for a subclass that does
    # not call ``__init__`` (the tests' FFI-free fake does exactly that).  They
    # are floats/ints, so ``self._perf_t_poll += x`` rebinds on the instance
    # and no state is shared between environments.
    _perf_n_act = 0
    _perf_t_act = 0.0
    _perf_t_poll = 0.0
    _perf_t_reply = 0.0
    _perf_t_drain = 0.0
    _perf_t_record = 0.0

    def __init__(
        self,
        deck_self: list[int],
        deck_opp: list[int] | None = None,
        n_envs: int = 12,
        our_player: int = 0,
        seed: int = 0,
        libcg_path: str | None = None,
        opp_decks: list[tuple[int, list[int]]] | None = None,
    ):
        """*opp_decks* holds ``(archetype_id, 60-card deck)`` pairs.

        Battle ``i`` is dealt entry ``i % len`` and keeps it across restarts,
        so one pool can span many archetypes.  That is what lets the inference
        batch be wide *and* the battles all be wanted: with one pool per
        opponent, a 201-deck sample gave mostly one-battle pools, and widening
        them ran battles that were destroyed unplayed.

        *deck_opp* is the single-opponent form and is still accepted; exactly
        one of the two must be given.
        """
        if (deck_opp is None) == (opp_decks is None):
            raise ValueError("pass exactly one of deck_opp or opp_decks")

        self._lib = _load_search_lib()
        lib_path = libcg_path or _find_libcg()
        _init_engine(lib_path)

        if opp_decks is not None:
            if not opp_decks:
                raise ValueError("opp_decks is empty")
            pairs = [[int(oid), [int(c) for c in deck]]
                     for oid, deck in opp_decks]
            self.opp_ids = [p[0] for p in pairs]
            self._handle = self._lib.vec_env_create_multi(
                n_envs,
                lib_path.encode("utf-8"),
                json.dumps(deck_self).encode("utf-8"),
                json.dumps(pairs).encode("utf-8"),
                our_player,
                seed,
                1,  # host_initialized
            )
            if self._handle == 0:
                raise RuntimeError(
                    "vec_env_create_multi returned null handle — check every "
                    "deck is exactly 60 cards"
                )
        else:
            self.opp_ids = [0]
            self._handle = self._lib.vec_env_create(
                n_envs,
                lib_path.encode("utf-8"),
                json.dumps(deck_self).encode("utf-8"),
                json.dumps(deck_opp).encode("utf-8"),
                our_player,
                seed,
                1,  # host_initialized
            )
            if self._handle == 0:
                raise RuntimeError("vec_env_create returned null handle")
        self.n_envs = n_envs
        self.our_player = our_player
        self.total_decisions = 0

        self._pending: dict[int, Trajectory] = {}

        # Perf counters.  ``mcts_train`` reads ``_perf_t_sweep``/``_perf_t_act``
        # behind a ``hasattr`` guard; only the *old* multiprocessing RolloutPool
        # ever defined them, so for every RustVecEnv run the guard was false and
        # the engine side of the wall clock was reported as zero and swept into
        # "other".  ``_perf_t_sweep`` is the Rust/FFI/JSON total — poll + reply
        # + drain — and is the sum of the three below.
        self._perf_n_act = 0
        self._perf_t_act = 0.0      # time inside act_fn (featurize + GPU)
        self._perf_t_poll = 0.0     # vec_env_poll FFI + JSON decode
        self._perf_t_reply = 0.0    # JSON encode + vec_env_reply FFI
        self._perf_t_drain = 0.0    # vec_env_drain FFI + JSON decode
        self._perf_t_record = 0.0   # Decision assembly, incl. obs re-serialisation

    @property
    def _perf_t_sweep(self) -> float:
        """Everything spent on the Rust side: FFI calls plus JSON at the border.

        Reported as one number because that is the actionable unit — it is what
        moving the featurizer or the env into Rust would attack.
        """
        return self._perf_t_poll + self._perf_t_reply + self._perf_t_drain

    def poll(self) -> list[dict]:
        _t = time.perf_counter()
        raw = _read_and_free(self._lib, self._lib.vec_env_poll(self._handle))
        out = json.loads(raw) if raw else []
        self._perf_t_poll += time.perf_counter() - _t
        return out

    def reply(self, picks_list: list[list[int]]) -> int:
        _t = time.perf_counter()
        rc = self._lib.vec_env_reply(
            self._handle,
            json.dumps(picks_list).encode("utf-8"),
        )
        self._perf_t_reply += time.perf_counter() - _t
        return rc

    def drain(self) -> list[Trajectory]:
        _t = time.perf_counter()
        raw = _read_and_free(self._lib, self._lib.vec_env_drain(self._handle))
        results = json.loads(raw) if raw else []
        trajs = []
        for r in results:
            t = Trajectory()
            t.battle_idx = int(r.get("battle_idx", -1))
            t.opp_id = int(r.get("opp_id", -1))
            t.reward = float(r.get("reward", 0))
            n = int(r.get("n_decisions", 0))
            self.total_decisions += n
            err = r.get("error")
            if err:
                t.error = str(err)
            trajs.append(t)
        self._perf_t_drain += time.perf_counter() - _t
        return trajs

    def collect_batch(
        self, act_fn, n_decisions: int, act_fn_opp=None,
    ) -> list[Trajectory]:
        """Collect at least *n_decisions* decision points.

        With *act_fn_opp* given, the polled batch is split by seat:
        ``self.our_player``'s observations go to *act_fn* and every other seat's
        go to *act_fn_opp*, and **only** *act_fn*'s replies are recorded as
        decisions.  That is the point of the split — the opponent's ``logp`` and
        ``value`` describe a different policy and must never reach θ's replay
        buffer.

        Leaving it ``None`` keeps the historical behaviour exactly: one actor
        answers both seats and both seats are recorded.  Self-play then has θ
        piloting the opponent's sampled archetype, which it has never seen, so
        the win rate measures the opponent's incompetence rather than θ's edge.

        Because only one seat is recorded, *n_decisions* buys roughly half as
        many decisions per game as it does without an opponent actor; callers
        size the budget accordingly.
        """
        done: list[Trajectory] = []
        collected = 0
        active_battles = set()

        t_last_progress = time.perf_counter()
        while collected < n_decisions:
            if time.perf_counter() - t_last_progress > 30:
                logger.warning("collect_batch: no progress in 30s, breaking")
                break
            # ── Poll: get all pending observations ───────────────────
            pending = self.poll()
            if pending:
                t_last_progress = time.perf_counter()

            # ── Drain: attach finished games to their pending trajectory
            for t in self.drain():
                bi = t.battle_idx
                if bi < 0:
                    continue
                pt = self._pending.get(bi)
                if pt is not None and len(pt.decisions) > 0:
                    pt.reward = t.reward
                    pt.error = t.error
                    # The pending Trajectory is built here, not by drain(), so
                    # every field drain() knows has to be copied across.  Losing
                    # opp_id is silent and expensive: it defaults to -1, the
                    # caller's `deck_by_id.get(-1, [])` then hands MCTS an
                    # *empty* opponent deck, and every per-archetype statistic
                    # collapses onto one bogus id.
                    if t.opp_id >= 0:
                        pt.opp_id = t.opp_id
                    done.append(pt)
                    collected += len(pt.decisions)
                    del self._pending[bi]
                    active_battles.discard(bi)
                    t_last_progress = time.perf_counter()
                # else: battle finished with no decisions (e.g. immediate error) —
                # nothing to recover, just let it be restarted by Rust.

            if not pending:
                # No battles need decisions — drain any stragglers
                for t in self.drain():
                    bi = t.battle_idx
                    if bi < 0:
                        continue
                    pt = self._pending.get(bi)
                    if pt is not None and len(pt.decisions) > 0:
                        pt.reward = t.reward
                        if t.opp_id >= 0:
                            pt.opp_id = t.opp_id
                        done.append(pt)
                        collected += len(pt.decisions)
                        del self._pending[bi]
                if not pending:
                    time.sleep(0.0001)
                    continue

            # ── Batch GPU forward ───────────────────────────────────
            # Rust hands `search_begin_input` back as its own `sbi` field
            # rather than leaving it in the observation, but Decision.obs_json
            # is contracted to be a *complete* obs — MCTS root construction
            # reads search_begin_input straight off it, and libcg segfaults on
            # an empty one.  Merge it back before anything sees the obs.
            _t_dec = time.perf_counter()
            obs_dicts = []
            for p in pending:
                obs = json.loads(p["obs_json"])
                obs["search_begin_input"] = p.get("sbi", "")
                obs_dicts.append(obs)
            self._perf_t_poll += time.perf_counter() - _t_dec
            requests = [
                {"obs": obs, "actor": "self", "record": True,
                 "opp_id": int(p.get("opp_id", -1))}
                for obs, p in zip(obs_dicts, pending)
            ]
            _t_act = time.perf_counter()
            if act_fn_opp is None:
                replies = act_fn(requests)
                record = [True] * len(requests)
            else:
                replies, record = self._route_by_seat(
                    requests, obs_dicts, act_fn, act_fn_opp,
                )
            self._perf_t_act += time.perf_counter() - _t_act
            self._perf_n_act += 1

            # ── Record decisions + apply picks ──────────────────────
            _t_rec = time.perf_counter()
            picks_list = []
            for i, rep in enumerate(replies):
                picks_list.append(rep["picks"])
                bi = pending[i]["battle_idx"]
                active_battles.add(bi)
                if bi not in self._pending:
                    self._pending[bi] = Trajectory()
                traj = self._pending[bi]
                if traj.opp_id < 0:
                    traj.opp_id = int(pending[i].get("opp_id", -1))
                if record[i] and rep.get("features") is not None:
                    traj.decisions.append(Decision(
                        features=rep["features"],
                        picks=list(rep["picks"]),
                        action_idx=rep["action_idx"],
                        action_len=int(rep["action_len"]),
                        logp=float(rep["logp"]),
                        value=float(rep["value"]),
                        turn=len(traj.decisions),
                        opp_visible_card_ids=None,
                        opp_id=int(pending[i].get("opp_id", -1)),
                        obs_json=json.dumps(obs_dicts[i]),
                        your_index=int(
                            obs_dicts[i].get("current", {}).get("yourIndex", 0)
                        ),
                    ))
            self._perf_t_record += time.perf_counter() - _t_rec
            self.reply(picks_list)

        return done

    def _route_by_seat(
        self, requests: list[dict], obs_dicts: list[dict], act_fn, act_fn_opp,
    ) -> tuple[list[dict], list[bool]]:
        """Split *requests* by seat, act, and reassemble **in poll order**.

        Returns ``(replies, record)`` where ``record[i]`` is True exactly when
        reply *i* came from *act_fn* and is therefore θ's own decision.

        The reassembly is not cosmetic: ``vec_env_reply`` consumes the picks
        list positionally against the batch ``vec_env_poll`` handed out, so
        returning the two groups concatenated would apply each seat's picks to
        the other seat's battles — legal moves for the wrong game, silently.
        """
        seats = [
            int(o.get("current", {}).get("yourIndex", self.our_player))
            for o in obs_dicts
        ]
        ours = [i for i, s in enumerate(seats) if s == self.our_player]
        theirs = [i for i, s in enumerate(seats) if s != self.our_player]

        replies: list[dict | None] = [None] * len(requests)
        record = [False] * len(requests)
        if ours:
            for k, rep in enumerate(act_fn([requests[i] for i in ours])):
                replies[ours[k]] = rep
                record[ours[k]] = True
        if theirs:
            for k, rep in enumerate(act_fn_opp([requests[i] for i in theirs])):
                replies[theirs[k]] = rep

        missing = [i for i, r in enumerate(replies) if r is None]
        if missing:
            # An actor that returns fewer replies than requests would otherwise
            # shift every later pick by one and desynchronise the whole batch.
            raise RuntimeError(
                f"seat routing produced {len(missing)} unanswered request(s) "
                f"out of {len(requests)}: an act_fn returned a short list",
            )
        return replies, record

    def close(self) -> None:
        if self._handle != 0:
            self._lib.vec_env_free(self._handle)
            self._handle = 0

    def __enter__(self) -> "RustVecEnv":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        self.close()


_engine_initialized = False


def _engine_already_initialized() -> bool:
    """True when something else in this process has run ``GameInitialize``.

    ``cg/api.py`` does ``from .sim import lib``, and importing ``cg.sim``
    calls ``GameInitialize()`` at import time — so anything that reaches for
    the bundled engine package (``ptcg_mine.cards.load_engine``, ``ptcg_il``'s
    QA gates, ``live_eval``) has already spent the one call this process gets.

    The module-level flag below cannot see that: it only records calls made
    *here*.  Importing ``cg`` first and then constructing a ``RustVecEnv``
    therefore called ``GameInitialize`` twice, which is not an error anyone
    can catch — libcg registers into a fixed-capacity global table and throws
    a C++ ``std::runtime_error("buffer full. capacity:7")``, taking the
    process down with SIGABRT.

    ``sys.modules`` is the honest signal, because the import *is* the call.
    """
    return "cg.sim" in sys.modules


def _init_engine(lib_path: str) -> None:
    global _engine_initialized
    if _engine_initialized:
        return
    if _engine_already_initialized():
        # Claim it anyway: the engine is initialized, which is all any caller
        # here needs, and a later call must still be suppressed.
        logger.debug(
            "GameInitialize already run by cg.sim — not calling it again",
        )
        _engine_initialized = True
        return
    ctypes.CDLL(lib_path).GameInitialize()
    _engine_initialized = True
