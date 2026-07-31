"""Rust-backed parallel game environment (via C bridge, fixed ABI).

N battles managed by Rust with direct libcg FFI.  Python only
featurizes observations and runs GPU forwards.
"""

from __future__ import annotations

import ctypes
import json
import logging
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

    def __init__(
        self,
        deck_self: list[int],
        deck_opp: list[int],
        n_envs: int = 12,
        our_player: int = 0,
        seed: int = 0,
        libcg_path: str | None = None,
    ):
        self._lib = _load_search_lib()
        lib_path = libcg_path or _find_libcg()
        _init_engine(lib_path)

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

    def poll(self) -> list[dict]:
        raw = _read_and_free(self._lib, self._lib.vec_env_poll(self._handle))
        return json.loads(raw) if raw else []

    def reply(self, picks_list: list[list[int]]) -> int:
        return self._lib.vec_env_reply(
            self._handle,
            json.dumps(picks_list).encode("utf-8"),
        )

    def drain(self) -> list[Trajectory]:
        raw = _read_and_free(self._lib, self._lib.vec_env_drain(self._handle))
        results = json.loads(raw) if raw else []
        trajs = []
        for r in results:
            t = Trajectory()
            t.battle_idx = int(r.get("battle_idx", -1))
            t.reward = float(r.get("reward", 0))
            n = int(r.get("n_decisions", 0))
            self.total_decisions += n
            err = r.get("error")
            if err:
                t.error = str(err)
            trajs.append(t)
        return trajs

    def collect_batch(self, act_fn, n_decisions: int) -> list[Trajectory]:
        """Collect at least *n_decisions* decision points."""
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
            obs_dicts = []
            for p in pending:
                obs = json.loads(p["obs_json"])
                obs["search_begin_input"] = p.get("sbi", "")
                obs_dicts.append(obs)
            requests = [
                {"obs": obs, "actor": "self", "record": True}
                for obs in obs_dicts
            ]
            replies = act_fn(requests)

            # ── Record decisions + apply picks ──────────────────────
            picks_list = []
            for i, rep in enumerate(replies):
                picks_list.append(rep["picks"])
                bi = pending[i]["battle_idx"]
                active_battles.add(bi)
                if bi not in self._pending:
                    self._pending[bi] = Trajectory()
                traj = self._pending[bi]
                if rep.get("features") is not None:
                    traj.decisions.append(Decision(
                        features=rep["features"],
                        picks=list(rep["picks"]),
                        action_idx=rep["action_idx"],
                        action_len=int(rep["action_len"]),
                        logp=float(rep["logp"]),
                        value=float(rep["value"]),
                        turn=len(traj.decisions),
                        opp_visible_card_ids=None,
                        obs_json=json.dumps(obs_dicts[i]),
                    ))
            self.reply(picks_list)

        return done

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


def _init_engine(lib_path: str) -> None:
    global _engine_initialized
    if _engine_initialized:
        return
    ctypes.CDLL(lib_path).GameInitialize()
    _engine_initialized = True
