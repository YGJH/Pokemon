"""Trajectory assembly, GAE, and the batched action callback (RL_SPEC §4, §9.1).

Two responsibilities:

1. :func:`compute_gae` turns finished games into advantages and value targets.
   It runs in **fp32 regardless of the rollout precision** — §7 requires it,
   because a long backwards accumulation in bf16 loses precision quickly and
   the result feeds directly into the PPO ratio.
2. :class:`PolicyActor` is the callback :meth:`RolloutPool.collect` calls: it
   featurizes a batch of pending observations, runs one forward pass, samples
   with :mod:`ptcg_rl.actor`'s masking, and hands back picks.

The reward structure is minimal by design (§4.1): zero everywhere except the
terminal step, ``γ = 1.0``.  There is no shaping.  Any shaping that is not
potential-based changes the optimal policy, and the one potential-based option
the spec allows (prize differential) is not worth the risk before R2 has shown
that PPO works here at all.
"""

from __future__ import annotations

import logging
import os
from rich.logging import RichHandler
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ptcg_il.featurizer import featurize
from ptcg_rl.actor import sample_action
from ptcg_rl.vec_env import Trajectory
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_time=False)])
logger = logging.getLogger(__name__)


@dataclass
class RolloutBatch:
    """Flattened decision points ready for the PPO update."""

    features: list[dict[str, np.ndarray]]
    action_idx: np.ndarray      # [N, O_MAX] int64
    action_len: np.ndarray      # [N] int64
    logp_old: np.ndarray        # [N] fp32 — diagnostics only, see §7
    value_old: np.ndarray       # [N] fp32
    advantage: np.ndarray       # [N] fp32
    value_target: np.ndarray    # [N] fp32
    turn: np.ndarray            # [N] int64

    def __len__(self) -> int:
        return len(self.features)


def compute_gae(
    values: Sequence[float],
    reward: float,
    *,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation for one finished game.

    The game is a single episode with reward only at the end, so
    ``δ_t = γ·V(s_{t+1}) − V(s_t)`` everywhere except the last step, where
    ``δ_T = r_T − V(s_T)``.  The bootstrap past the terminal state is **zero**,
    not ``V(s_T)``: the game is over, and bootstrapping off a terminal state is
    a classic way to leak value backwards from nothing.

    Returns ``(advantages, value_targets)`` in fp32, where
    ``value_target = advantage + value`` per §9.2.

    Note what this degenerates to if the critic is dead: with ``V ≡ 0`` every
    advantage becomes the raw terminal return, i.e. REINFORCE with no variance
    reduction over ~150 decisions. That is why R1 gates R2.
    """
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in range(n - 1, -1, -1):
        next_value = v[t + 1] if t + 1 < n else 0.0
        r = reward if t == n - 1 else 0.0
        delta = r + gamma * next_value - v[t]
        last = delta + gamma * lam * last
        adv[t] = last

    return adv.astype(np.float32), (adv + v).astype(np.float32)


def build_batch(
    trajectories: Sequence[Trajectory],
    *,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> RolloutBatch:
    """Flatten finished games into one training batch, with GAE applied per game.

    Advantages are **not** normalised here.  §9.1 normalises per minibatch over
    decision points, and doing it twice — once per buffer, once per minibatch —
    silently rescales the effective learning rate.
    """
    feats: list[dict[str, np.ndarray]] = []
    a_idx, a_len, lp, vals, advs, vtargs, turns = [], [], [], [], [], [], []

    for traj in trajectories:
        if not traj.decisions:
            continue
        v = [d.value for d in traj.decisions]
        adv, vtarget = compute_gae(v, traj.reward, gamma=gamma, lam=lam)
        for i, d in enumerate(traj.decisions):
            feats.append(d.features)
            a_idx.append(d.action_idx)
            a_len.append(d.action_len)
            lp.append(d.logp)
            vals.append(d.value)
            advs.append(adv[i])
            vtargs.append(vtarget[i])
            turns.append(d.turn)
    
    #  在這裡將收集到的「全量 Rollout Advantage」轉成 numpy 後做標準化
    advs_np = np.asarray(advs, dtype=np.float32)
    advs_np = (advs_np - advs_np.mean()) / (advs_np.std() + 1e-8)
    if not feats:
        raise ValueError("no decision points in the collected trajectories")

    return RolloutBatch(
        features=feats,
        action_idx=np.stack(a_idx).astype(np.int64),
        action_len=np.asarray(a_len, dtype=np.int64),
        logp_old=np.asarray(lp, dtype=np.float32),
        value_old=np.asarray(vals, dtype=np.float32),
        advantage=advs_np,
        value_target=np.asarray(vtargs, dtype=np.float32),
        turn=np.asarray(turns, dtype=np.int64),
    )


def normalize_advantages(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance advantages over the minibatch's decision points.

    Per decision point rather than per episode: episode length varies ~3×, so a
    per-episode normalisation would weight short games more heavily for no
    reason connected to the policy.
    """
    adv = adv.float()
    if adv.numel() < 2:
        return torch.zeros_like(adv)
    return (adv - adv.mean()) / (adv.std(unbiased=False) + eps)


class PolicyActor:
    """Serves batched actions to :meth:`RolloutPool.collect`.

    One instance holds the acting policy and the vocab.  Calling it with the
    pending requests featurizes them all, runs **one** forward pass, and samples
    with the shared masking from :mod:`ptcg_rl.actor`.

    Acting always samples from ``π_θ``.  If anything else picked the action —
    a search, a heuristic, an opponent model — the PPO ratio would be an
    importance weight for a distribution nobody sampled, and the update would be
    biased in a way no hyperparameter fixes (§8.2).
    """

    def __init__(
        self,
        policy,
        vocab: dict,
        *,
        device: str = "cpu",
        temperature: float = 1.0,
        greedy: bool = False,
        bf16: bool = False,
        seed: int = 0,
        engine_card_features: dict | None = None,
        engine_attack_features: dict | None = None,
    ):
        self.policy = policy
        self.vocab = vocab
        self.device = torch.device(device)
        self.temperature = temperature
        self.greedy = greedy
        self.bf16 = bf16 and self.device.type == "cuda"
        self.engine_card_features = engine_card_features
        self.engine_attack_features = engine_attack_features
        # The generator must live on the same device as the tensors it samples
        # from: `torch.multinomial` rejects a CPU generator for a CUDA input
        # outright rather than falling back.
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.n_featurize_failures = 0
        # Perf counters (reset each collect cycle)
        self.perf_n_calls = 0
        self.perf_t_featurize = 0.0
        self.perf_t_forward = 0.0
        # `_collate` builds the tensors and pushes them across PCIe.  It used
        # to fall between the featurize and forward timers and land in neither,
        # so the one phase whose cost is bus-bound was invisible.
        self.perf_t_collate = 0.0
        # The trailing `.cpu()` readback plus reply assembly — likewise after
        # the forward timer closed.  On CUDA this is where the queued forward
        # actually gets waited on, so a large value here means the GPU is the
        # bottleneck even though `perf_t_forward` looks small.
        self.perf_t_d2h = 0.0
        self.perf_n_rows = 0
        # Wall-clock attribution across a CUDA stream is only meaningful with a
        # sync at each boundary; without one the forward's cost migrates into
        # whatever next touches the result.  Opt-in because the sync itself
        # serialises the pipeline and slows the run it is measuring.
        self.perf_sync = (
            os.environ.get("PTCG_PERF_SYNC") == "1"
            and self.device.type == "cuda"
        )

    def __call__(self, requests: list[dict]) -> list[dict]:
        """One reply per request, in the same order.

        A request whose observation cannot be featurized still gets a reply —
        the first legal option — because a worker blocked forever on a missing
        action deadlocks the whole pool.  Those replies are never recorded.
        """
        import time as _time
        _t0 = _time.perf_counter()
        samples: list[dict | None] = []
        for req in requests:
            try:
                samples.append(featurize(
                    req["obs"], self.vocab,
                    engine_card_features=self.engine_card_features,
                    engine_attack_features=self.engine_attack_features))
            except (ValueError, TypeError, KeyError) as e:
                logger.debug("featurize failed, falling back to first legal: %s", e)
                self.n_featurize_failures += 1
                samples.append(None)
        _t1 = _time.perf_counter()
        self.perf_t_featurize += _t1 - _t0

        usable = [i for i, s in enumerate(samples) if s is not None]
        replies: list[dict] = [
            {"picks": _first_legal(requests[i]["obs"]), "features": None,
             "action_idx": None, "action_len": 0, "logp": 0.0, "value": 0.0}
            for i in range(len(requests))
        ]
        if not usable:
            # Still a call, and its featurize time is already banked — bailing
            # without counting it left perf_n_calls at 0, and mcts_train gates
            # the whole PERF report on `perf_n_calls > 0`.
            self.perf_n_calls += 1
            self.perf_n_rows += len(requests)
            return replies

        import time as _time2

        def _mark() -> float:
            if self.perf_sync:
                torch.cuda.synchronize(self.device)
            return _time2.perf_counter()

        _t_pre_collate = _mark()
        batch = _collate([samples[i] for i in usable], self.device)

        _t_pre_fwd = _mark()
        self.perf_t_collate += _t_pre_fwd - _t_pre_collate
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16 else _NullCtx()
        with torch.no_grad(), autocast:
            # Encode once for both the pointer AR loop and the value head.
            h, _history = self.policy._encode(batch)
            picks, logp, _entropy = sample_action(
                self.policy, batch,
                temperature=self.temperature,
                greedy=self.greedy,
                generator=self.generator,
                encoded=h,
            )
            value = self.policy.value(h[:, 0])
        _t_post_fwd = _mark()
        self.perf_t_forward += _t_post_fwd - _t_pre_fwd

        logp = logp.float().cpu().numpy()
        value = value.float().cpu().numpy()

        for slot, i in enumerate(usable):
            row_picks = picks[slot]
            stop_col = int(samples[i]["stop_column"])
            action_idx, action_len = _action_label(
                row_picks, stop_col, int(samples[i]["maxCount"]),
                width=samples[i]["action_idx"].shape[0],
            )
            replies[i] = {
                "picks": row_picks,
                "features": samples[i],
                "action_idx": action_idx,
                "action_len": action_len,
                "logp": float(logp[slot]),
                "value": float(value[slot]),
            }
        self.perf_t_d2h += _time2.perf_counter() - _t_post_fwd
        self.perf_n_calls += 1
        self.perf_n_rows += len(requests)
        return replies


class DuelActor:
    """Routes each decision to whichever policy owns that seat.

    :class:`RolloutPool` drives *both* players through one callback, so a single
    :class:`PolicyActor` makes every match a mirror of itself.  That is correct
    for R2 self-play and silently wrong for the gate, where the whole question is
    whether θ beats the frozen π_IL — a θ-vs-θ match sits at 50% by construction
    and would read as "the candidate did not improve".

    Requests carry ``actor`` (whose turn it is), so the split is exact.
    """

    def __init__(self, ours: PolicyActor, theirs: PolicyActor, our_player: int = 0):
        self.ours = ours
        self.theirs = theirs
        self.our_player = our_player

    def __call__(self, requests: list[dict]) -> list[dict]:
        mine = [i for i, r in enumerate(requests) if r["actor"] == self.our_player]
        yours = [i for i, r in enumerate(requests) if r["actor"] != self.our_player]

        replies: list[dict | None] = [None] * len(requests)
        for idx, actor in ((mine, self.ours), (yours, self.theirs)):
            if not idx:
                continue
            for slot, rep in zip(idx, actor([requests[i] for i in idx])):
                replies[slot] = rep
        # Every waiting worker must get exactly one action or the pool deadlocks.
        assert all(r is not None for r in replies)
        return replies  # type: ignore[return-value]


def _action_label(
    picks: Sequence[int], stop_column: int, max_count: int, width: int
) -> tuple[np.ndarray, int]:
    """Rebuild the AR target sequence the sampler implies.

    The STOP step is appended whenever a STOP column exists and the row did not
    exhaust ``maxCount`` — matching ``featurizer._build_label`` exactly, because
    ``recompute_logp`` must score the same sequence the sampler scored.  Getting
    this wrong makes the epoch-0 ratio canary fire, which is the point of it.
    """
    action_idx = np.full(width, -1, dtype=np.int64)
    n = 0
    for c in picks:
        if n >= width:
            break
        action_idx[n] = int(c)
        n += 1
    if stop_column >= 0 and len(picks) < max_count and n < width:
        action_idx[n] = stop_column
        n += 1
    return action_idx, n


def _first_legal(obs: dict) -> list[int]:
    """A syntactically valid action for an observation we could not featurize.

    Picks *distinct* leading indices, not ``[0] * minCount`` — the engine rejects
    a repeated option, so the naive version would turn a featurization failure
    into an engine error and lose the whole game rather than one decision.
    """
    select = obs.get("select") or {}
    options = select.get("option") or []
    if not options:
        return []
    min_count = max(1, int(select.get("minCount") or 1))
    return list(range(min(min_count, len(options))))


def _collate(samples: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Stack featurizer outputs into a batch on *device*.

    ``minCount``/``maxCount``/``stop_column`` come out of the featurizer as
    ``np.int64`` **scalars**, not arrays (``featurizer.py:664, 884``).  Stacking
    them as if they were arrays is the discrepancy that caused ~9% agent
    forfeits in the submission bug, so they are handled explicitly here.
    """
    out: dict[str, torch.Tensor] = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        first = vals[0]
        if np.isscalar(first) or (isinstance(first, np.generic)) or (
            isinstance(first, np.ndarray) and first.ndim == 0
        ):
            arr = np.asarray([np.asarray(v).item() for v in vals])
        else:
            arr = np.stack([np.asarray(v) for v in vals])
        out[key] = torch.as_tensor(arr).to(device)
    return out


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
