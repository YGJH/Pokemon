"""Regression tests for the two silent breaks that made MCTS distillation
a no-op: decisions carried no ``obs_json``, and the CLI's MCTS flags never
reached the ``config.mcts_*`` names the search paths read.

Both failed by *omission* — a ``dict.get`` and a ``getattr`` default — so
neither raised and the only visible symptom was ``0 MCTS, 0%``.
"""

from __future__ import annotations

import json

import pytest

from ptcg_rl.mcts_train import parse_args
from ptcg_rl.rust_vec_env import RustVecEnv


# ── obs_json must survive poll() → Decision ──────────────────────────────

class _FakePolledEnv(RustVecEnv):
    """RustVecEnv with the FFI replaced — exercises collect_batch's
    bookkeeping without loading libcg or libptcg_search."""

    def __init__(self, polls):
        self._polls = list(polls)
        self._pending = {}
        self.n_envs = 1
        self.our_player = 0
        self.total_decisions = 0
        self.replies = []
        self._drained = False

    def poll(self):
        return self._polls.pop(0) if self._polls else []

    def drain(self):
        """Finish the battle only once every queued poll is consumed."""
        from ptcg_rl.vec_env import Trajectory
        if self._polls or self._drained:
            return []
        self._drained = True
        t = Trajectory()
        t.battle_idx = 0
        t.reward = 1.0
        return [t]

    def reply(self, picks_list):
        self.replies.append(picks_list)
        return 0

    def close(self):  # no handle to free
        pass


def _obs(turn: int) -> str:
    return json.dumps({"turn": turn, "current": {"yourIndex": 0}})


def _act_fn(requests):
    return [
        {"picks": [0], "features": {"dummy": 1}, "action_idx": [0],
         "action_len": 1, "logp": -0.5, "value": 0.1}
        for _ in requests
    ]


def test_collect_batch_records_obs_json():
    """Decision.obs_json must carry the raw observation from poll().

    MCTS distillation skips any decision whose obs_json is falsy
    (``mcts_train._run_mcts_on_decisions_pooled``), so a None here silently
    zeroes the MCTS target count.  Mutation check: reading a key that poll()
    does not emit — as ``pending[i].get("obs_json_raw")`` did — turns this
    red.
    """
    polls = [
        [{"battle_idx": 0, "obs_json": _obs(1), "sbi": "", "select_player": 0}],
        [{"battle_idx": 0, "obs_json": _obs(2), "sbi": "", "select_player": 0}],
        [],  # engine has nothing pending → battle drains
    ]
    env = _FakePolledEnv(polls)
    done = env.collect_batch(_act_fn, n_decisions=1)

    assert len(done) == 1
    decisions = done[0].decisions
    assert len(decisions) == 2, "fixture must produce decisions to examine"
    for dec, expected_turn in zip(decisions, (1, 2)):
        assert dec.obs_json, "Decision.obs_json is empty — MCTS would skip it"
        assert json.loads(dec.obs_json)["turn"] == expected_turn


# ── CLI flags must land on the mcts_* attribute names ────────────────────

_BASE_ARGV = ["--data-dir", "d", "--il-ckpt", "c"]

# (CLI flag, value, the attribute the MCTS code paths read)
_FLAG_TO_ATTR = [
    ("--rho", "1.0", "mcts_rho", 1.0),
    ("--iterations", "128", "mcts_iterations", 128),
    ("--c-puct", "3.5", "mcts_c_puct", 3.5),
    ("--k-determinizations", "6", "mcts_k_determinizations", 6),
    ("--leaf-batch", "4096", "mcts_leaf_batch", 4096),
    ("--n-engines", "20", "mcts_n_engines", 20),
]


@pytest.mark.parametrize("flag,value,attr,expected", _FLAG_TO_ATTR)
def test_mcts_flag_reaches_config_attribute(flag, value, attr, expected):
    """Every MCTS knob is read as ``getattr(config, "mcts_*", <default>)``,
    so an unmapped flag is not an error — it is silently the default."""
    args = parse_args(_BASE_ARGV + [flag, value])
    assert getattr(args, attr) == expected


def test_mcts_distill_and_enabled_are_unified():
    """mcts_train gates on ``mcts_distill``; ptcg_rl.search gates on
    ``mcts_enabled``.  Either flag must switch on both paths."""
    for flag in ("--mcts-distill", "--mcts-enabled"):
        args = parse_args(_BASE_ARGV + [flag])
        assert args.mcts_distill is True, flag
        assert args.mcts_enabled is True, flag

    off = parse_args(_BASE_ARGV)
    assert off.mcts_distill is False
    assert off.mcts_enabled is False


def test_pipeline_flag_set_is_fully_mapped():
    """The exact flag set scripts/run_pipeline.sh passes must survive
    normalization — no knob may fall back to a hardcoded default."""
    argv = _BASE_ARGV + [
        "--iterations", "64", "--c-puct", "2.0", "--k-determinizations", "4",
        "--leaf-batch", "4096", "--rho", "1.0", "--mcts-distill",
        "--n-engines", "20",
    ]
    args = parse_args(argv)
    assert (args.mcts_iterations, args.mcts_c_puct) == (64, 2.0)
    assert (args.mcts_k_determinizations, args.mcts_leaf_batch) == (4, 4096)
    assert (args.mcts_rho, args.mcts_n_engines) == (1.0, 20)
    assert args.mcts_distill is True
