"""``ptcg_rl.rollout`` and ``ptcg_rl.vec_env`` — GAE and the worker protocol.

The engine is injected (as ``ptcg_mine/download.py`` injects the Kaggle API), so
these run without ``libcg.so``.  The fake engine below is deliberately a real
state machine rather than a mock: the properties worth testing — that a timeout
is scored as a loss, that a finished game yields exactly its decisions, that
``our_player``'s rows and only those are recorded — are all about the *sequence*
of interactions, and a mock that returns canned values cannot exercise them.
"""

from __future__ import annotations

import numpy as np
import pytest

from ptcg_rl.rollout import compute_gae, normalize_advantages
from ptcg_rl.vec_env import RolloutPool, Trajectory

torch = pytest.importorskip("torch")


# ── A fake engine ───────────────────────────────────────────────────────────
# Module-level so the 'spawn' start method can pickle it by name.

FAKE_GAME_LENGTH = 8


class _FakeBattle:
    """A game of exactly ``FAKE_GAME_LENGTH`` alternating decisions, player 0 wins."""

    def __init__(self) -> None:
        self.step = 0

    def obs(self) -> dict:
        finished = self.step >= FAKE_GAME_LENGTH
        return {
            "current": {"yourIndex": self.step % 2, "result": 0 if finished else -1},
            "select": None if finished else {
                "option": [{"type": 14}, {"type": 14}],
                "minCount": 1,
                "maxCount": 1,
            },
        }


def fake_engine_factory():
    """``(battle_start, battle_select, battle_finish)`` over :class:`_FakeBattle`."""
    state: dict[str, _FakeBattle | None] = {"battle": None}

    def battle_start(deck0, deck1):
        state["battle"] = _FakeBattle()
        return state["battle"].obs(), {"battlePtr": 1}

    def battle_select(picks):
        b = state["battle"]
        assert b is not None, "battle_select before battle_start"
        b.step += 1
        return b.obs()

    def battle_finish():
        state["battle"] = None

    return battle_start, battle_select, battle_finish


def hang_engine_factory():
    """A game that never ends — for the timeout path."""
    def battle_start(deck0, deck1):
        return {"current": {"yourIndex": 0, "result": -1},
                "select": {"option": [{"type": 14}], "minCount": 1, "maxCount": 1}}, {}

    def battle_select(picks):
        return {"current": {"yourIndex": 0, "result": -1},
                "select": {"option": [{"type": 14}], "minCount": 1, "maxCount": 1}}

    def battle_finish():
        pass

    return battle_start, battle_select, battle_finish


def crash_engine_factory():
    """A worker that dies partway through its first game."""
    state = {"n": 0}

    def battle_start(deck0, deck1):
        return {"current": {"yourIndex": 0, "result": -1},
                "select": {"option": [{"type": 14}], "minCount": 1, "maxCount": 1}}, {}

    def battle_select(picks):
        state["n"] += 1
        if state["n"] >= 2:
            import os
            os._exit(1)          # hard death: no traceback, no DONE message
        return {"current": {"yourIndex": 0, "result": -1},
                "select": {"option": [{"type": 14}], "minCount": 1, "maxCount": 1}}

    def battle_finish():
        pass

    return battle_start, battle_select, battle_finish


def constant_act_fn(requests: list[dict]) -> list[dict]:
    """Reply to every request with option 0 and a fixed recorded payload."""
    return [
        {
            "picks": [0],
            "features": {"stub": np.zeros(1, dtype=np.float32)},
            "action_idx": np.array([0, -1], dtype=np.int64),
            "action_len": 1,
            "logp": -0.5,
            "value": 0.25,
        }
        for _ in requests
    ]


# ── GAE ─────────────────────────────────────────────────────────────────────


class TestGAE:
    def test_matches_hand_computation(self):
        """γ=1, λ=0.95, V=[0.1,0.2,0.3], r=+1 — worked through by hand.

        δ₂ = 1 − 0.3 = 0.7                       A₂ = 0.7
        δ₁ = 0.3 − 0.2 = 0.1                     A₁ = 0.1 + 0.95·0.7   = 0.765
        δ₀ = 0.2 − 0.1 = 0.1                     A₀ = 0.1 + 0.95·0.765 = 0.82675
        """
        adv, vtarget = compute_gae([0.1, 0.2, 0.3], 1.0, gamma=1.0, lam=0.95)
        assert np.allclose(adv, [0.82675, 0.765, 0.7], atol=1e-6)
        assert np.allclose(vtarget, adv + np.array([0.1, 0.2, 0.3]), atol=1e-6)

    def test_does_not_bootstrap_past_the_terminal_state(self):
        """The value after the last decision is 0, not V(s_T).

        Bootstrapping off a terminal state leaks value backwards from a state
        that has none, and it does so quietly — returns simply come out too
        large.
        """
        adv, _ = compute_gae([0.0, 0.0, 0.9], 1.0, gamma=1.0, lam=1.0)
        # With V=0 for the first two steps, every advantage is the terminal
        # return propagated back: δ₂ = 1 − 0.9 = 0.1, δ₁ = 0.9, δ₀ = 0.
        assert np.isclose(adv[2], 0.1, atol=1e-6)
        assert np.isclose(adv[1], 0.9 + 0.1, atol=1e-6)

    def test_dead_critic_degenerates_to_the_raw_return(self):
        """With V ≡ 0 every advantage equals the terminal reward.

        This is the R1 motivation made concrete: a collapsed critic turns PPO
        into REINFORCE with no variance reduction over the whole episode.
        """
        adv, _ = compute_gae([0.0] * 5, -1.0, gamma=1.0, lam=1.0)
        assert np.allclose(adv, [-1.0] * 5, atol=1e-6)

    def test_loss_gives_negative_advantages(self):
        adv, _ = compute_gae([0.0, 0.0], -1.0)
        assert (adv < 0).all()

    def test_empty_trajectory(self):
        adv, vtarget = compute_gae([], 1.0)
        assert adv.shape == (0,) and vtarget.shape == (0,)

    def test_output_is_fp32(self):
        """§7: GAE accumulates in fp32 no matter the rollout precision."""
        adv, vtarget = compute_gae([0.1] * 4, 1.0)
        assert adv.dtype == np.float32 and vtarget.dtype == np.float32


class TestAdvantageNormalisation:
    def test_zero_mean_unit_variance(self):
        adv = normalize_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert abs(float(adv.mean())) < 1e-6
        assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-5

    def test_constant_advantages_do_not_produce_nan(self):
        """All-equal advantages have zero variance; the epsilon must save us."""
        adv = normalize_advantages(torch.full((8,), 0.3))
        assert torch.isfinite(adv).all()

    def test_single_element_is_zeroed(self):
        adv = normalize_advantages(torch.tensor([5.0]))
        assert torch.isfinite(adv).all()
        assert float(adv[0]) == 0.0


# ── The worker protocol ─────────────────────────────────────────────────────


class TestRolloutPool:
    def test_collects_complete_games(self):
        """A finished game yields exactly its own player's decisions."""
        with RolloutPool(
            list(range(60)), list(range(60)),
            n_workers=2, forward_batch=8, our_player=0,
            engine_factory=fake_engine_factory,
        ) as pool:
            trajs = pool.collect(constant_act_fn, n_decisions=8)

        assert trajs, "collected no trajectories"
        # Player 0 acts on even steps, so half of the 8 decisions are ours.
        for t in trajs:
            assert len(t) == FAKE_GAME_LENGTH // 2, (
                f"expected {FAKE_GAME_LENGTH // 2} recorded decisions, got {len(t)}"
            )
            assert t.error is None
            assert not t.timeout

    def test_winner_maps_to_reward(self):
        """The fake engine always has player 0 win."""
        with RolloutPool(
            list(range(60)), list(range(60)),
            n_workers=1, forward_batch=4, our_player=0,
            engine_factory=fake_engine_factory,
        ) as pool:
            as_p0 = pool.collect(constant_act_fn, n_decisions=4)

        with RolloutPool(
            list(range(60)), list(range(60)),
            n_workers=1, forward_batch=4, our_player=1,
            engine_factory=fake_engine_factory,
        ) as pool:
            as_p1 = pool.collect(constant_act_fn, n_decisions=4)

        assert as_p0 and as_p1
        assert all(t.reward == 1.0 for t in as_p0), "player 0 should win"
        assert all(t.reward == -1.0 for t in as_p1), "player 1 should lose"

    def test_turn_index_counts_our_decisions(self):
        with RolloutPool(
            list(range(60)), list(range(60)),
            n_workers=1, forward_batch=4, our_player=0,
            engine_factory=fake_engine_factory,
        ) as pool:
            trajs = pool.collect(constant_act_fn, n_decisions=4)
        assert [d.turn for d in trajs[0].decisions] == list(range(len(trajs[0])))

    def test_timeout_is_recorded_as_a_loss(self):
        """§4.2: a worker that exceeds budget records a loss, never a discard.

        Discarding would make slow policies invisible to the objective and
        therefore actively selected for.
        """
        with RolloutPool(
            list(range(60)), list(range(60)),
            n_workers=1, forward_batch=4, our_player=0,
            engine_factory=hang_engine_factory,
            max_decisions=6,
        ) as pool:
            trajs = pool.collect(constant_act_fn, n_decisions=1)

        assert trajs, "a timed-out game produced no trajectory — it was discarded"
        assert trajs[0].timeout is True
        assert trajs[0].reward == -1.0, "timeout must score as a loss"

    def test_rejects_a_wrong_sized_reply(self):
        """Every waiting worker must get exactly one action, or it deadlocks."""
        def short_act_fn(requests):
            return constant_act_fn(requests)[:-1] if len(requests) > 1 else []

        with RolloutPool(
            list(range(60)), list(range(60)),
            n_workers=2, forward_batch=8, our_player=0,
            engine_factory=fake_engine_factory,
        ) as pool:
            with pytest.raises(RuntimeError, match="exactly one action"):
                pool.collect(short_act_fn, n_decisions=8)

    def test_a_dead_worker_does_not_hang_the_pool(self):
        """``collect`` must return even when every worker dies mid-game.

        A crashed worker leaves its pipe permanently readable at EOF, so
        ``mpc.wait`` keeps returning it and ``recv`` keeps raising. Termination
        comes from ``_any_alive``; the ``_dead`` set is what stops each corpse
        from consuming a ``forward_batch`` slot on every sweep and eventually
        starving the live workers out of the batch.

        Note this test does *not* discriminate the ``_dead`` set on its own —
        mutating it away still passes here, because with all workers dead
        ``_any_alive`` ends the loop regardless. It pins the termination
        guarantee, which is the part whose failure mode is a silent hang.
        """
        with RolloutPool(
            list(range(60)), list(range(60)),
            n_workers=2, forward_batch=4, our_player=0,
            engine_factory=crash_engine_factory,
        ) as pool:
            # Asks for more decisions than the dying workers can ever produce;
            # collect must still return rather than loop.
            trajs = pool.collect(constant_act_fn, n_decisions=10_000)

        assert isinstance(trajs, list)

    def test_collect_before_start_is_an_error(self):
        pool = RolloutPool(list(range(60)), list(range(60)),
                           engine_factory=fake_engine_factory)
        with pytest.raises(RuntimeError, match="not started"):
            pool.collect(constant_act_fn, n_decisions=1)


class TestDuelActor:
    """The gate must pit θ against π_IL, not against itself."""

    class _Tagged:
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, requests):
            return [{"picks": [0], "tag": self.tag} for _ in requests]

    def test_each_seat_gets_its_own_policy(self):
        from ptcg_rl.rollout import DuelActor

        duel = DuelActor(self._Tagged("theta"), self._Tagged("frozen"), our_player=0)
        replies = duel([{"actor": 0}, {"actor": 1}, {"actor": 1}, {"actor": 0}])
        assert [r["tag"] for r in replies] == ["theta", "frozen", "frozen", "theta"], (
            "seats were routed to the wrong policy, or replies came back reordered"
        )

    def test_reply_order_matches_request_order(self):
        """The pool zips replies against requests positionally.

        A reordered reply hands one worker another worker's action — which is
        legal-looking and produces a game nobody chose.
        """
        from ptcg_rl.rollout import DuelActor

        duel = DuelActor(self._Tagged("a"), self._Tagged("b"), our_player=1)
        reqs = [{"actor": i % 2, "n": i} for i in range(7)]
        replies = duel(reqs)
        assert len(replies) == len(reqs)
        for req, rep in zip(reqs, replies):
            assert rep["tag"] == ("a" if req["actor"] == 1 else "b")

    def test_all_requests_from_one_seat(self):
        from ptcg_rl.rollout import DuelActor

        duel = DuelActor(self._Tagged("theta"), self._Tagged("frozen"), our_player=0)
        assert [r["tag"] for r in duel([{"actor": 0}] * 3)] == ["theta"] * 3
        assert [r["tag"] for r in duel([{"actor": 1}] * 3)] == ["frozen"] * 3


class TestBuildBatch:
    def test_flattens_and_applies_gae_per_game(self):
        from ptcg_rl.rollout import build_batch
        from ptcg_rl.vec_env import Decision

        def _traj(values, reward):
            t = Trajectory(reward=reward)
            for i, v in enumerate(values):
                t.decisions.append(Decision(
                    features={"stub": np.zeros(1, dtype=np.float32)},
                    picks=[0], action_idx=np.array([0, -1], dtype=np.int64),
                    action_len=1, logp=-0.5, value=v, turn=i,
                ))
            return t

        batch = build_batch([_traj([0.1, 0.2], 1.0), _traj([0.0], -1.0)])
        assert len(batch) == 3
        assert batch.advantage.dtype == np.float32
        # The lost game's single advantage is its own terminal delta, not
        # contaminated by the won game that preceded it in the list.
        assert batch.advantage[2] < 0 < batch.advantage[0]

    def test_empty_input_raises(self):
        from ptcg_rl.rollout import build_batch
        with pytest.raises(ValueError, match="no decision points"):
            build_batch([])
