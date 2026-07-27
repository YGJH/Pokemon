"""``ptcg_rl.ppo``, ``ptcg_rl.gate``, ``ptcg_rl.critic`` — the numerical guards.

The ratio canary (§7) and the Wilson bound (§10.2) are both cases where being
*approximately* right is indistinguishable from being wrong, so both are tested
against exact reference values rather than against "looks reasonable".
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ptcg_il.model.policy import Policy  # noqa: E402
from ptcg_rl.config import RLConfig  # noqa: E402
from ptcg_rl.critic import CriticDiagnostics, diagnose  # noqa: E402
from ptcg_rl.gate import (  # noqa: E402
    evaluate_gate,
    paired_seeds,
    score_from_results,
    wilson_interval,
)
from ptcg_rl.ppo import (  # noqa: E402
    RatioCanaryError,
    check_ratio_canary,
    explained_variance,
    ppo_losses,
    update_beta,
)

from tests.test_rl_actor import _batch, _tiny_policy  # noqa: E402


# ── The ratio canary ────────────────────────────────────────────────────────


class TestRatioCanary:
    def test_identical_logprobs_pass(self):
        ratio = torch.ones(1000)
        assert check_ratio_canary(ratio, 1e-3) == pytest.approx(0.0, abs=1e-9)

    def test_bf16_sized_drift_is_rejected(self):
        """§7 measured p99 |Δlogp| ≈ 0.69 across precisions — ratio ≈ 0.5.

        That is 500% of the ε = 0.2 clip range, so clipping would be driven by
        rounding rather than by policy change.  The canary exists to make that
        loud.
        """
        ratio = torch.ones(1000)
        ratio[:20] = float(np.exp(-0.693))  # ln 2, the measured tie-induced tail
        with pytest.raises(RatioCanaryError, match=r"p99 \|ratio - 1\|"):
            check_ratio_canary(ratio, 1e-3)

    def test_nan_is_rejected(self):
        ratio = torch.ones(100)
        ratio[0] = float("nan")
        with pytest.raises(RatioCanaryError):
            check_ratio_canary(ratio, 1e-3)

    def test_tiny_drift_within_tolerance_passes(self):
        ratio = torch.ones(1000) * 1.0000001
        assert check_ratio_canary(ratio, 1e-3) < 1e-3

    def test_epoch_zero_ratios_are_one_on_a_real_policy(self):
        """End-to-end: recomputing logp_old from the same weights gives ratio 1.

        This is the canary doing its actual job — if the update path masked or
        batched differently from the rollout path, it would show up here.
        """
        from ptcg_rl.actor import recompute_logp

        policy = _tiny_policy()
        x = _batch(B=16, seed=4)
        with torch.no_grad():
            logp_old, _ = recompute_logp(policy, x)
            logp_new, _ = recompute_logp(policy, x)
        ratio = (logp_new - logp_old).exp()
        assert check_ratio_canary(ratio, 1e-3) < 1e-3


# ── Adaptive β ──────────────────────────────────────────────────────────────


class TestAdaptiveBeta:
    def test_kl_above_budget_raises_beta(self):
        cfg = RLConfig(kappa=0.02)
        assert update_beta(0.1, 0.10, cfg) > 0.1

    def test_kl_below_budget_lowers_beta(self):
        cfg = RLConfig(kappa=0.02)
        assert update_beta(0.1, 0.001, cfg) < 0.1

    def test_kl_at_budget_leaves_beta_alone(self):
        cfg = RLConfig(kappa=0.02)
        assert update_beta(0.1, 0.02, cfg) == pytest.approx(0.1, rel=1e-9)

    def test_beta_stays_within_bounds(self):
        cfg = RLConfig(kappa=0.02, beta_min=1e-4, beta_max=10.0)
        assert update_beta(10.0, 100.0, cfg) <= 10.0
        assert update_beta(1e-4, 0.0, cfg) >= 1e-4

    def test_pathological_kl_does_not_saturate_beta(self):
        """One NaN or huge KL upstream must not pin β at its bound forever."""
        cfg = RLConfig(kappa=0.02)
        assert update_beta(0.1, 1e9, cfg) <= cfg.beta_max

    def test_beta_converges_to_hold_the_budget(self):
        """The dual update must actually drive measured KL toward κ.

        Simulates a policy whose KL falls as β rises; β should settle rather
        than oscillate or run away.
        """
        cfg = RLConfig(kappa=0.02, beta_lr=0.5)
        beta = 0.1
        kl = 0.20
        for _ in range(400):
            beta = update_beta(beta, kl, cfg)
            kl = 0.02 * (0.1 / beta) ** 0.5   # KL falls as the penalty rises
        assert abs(kl - cfg.kappa) < 0.005, f"KL settled at {kl}, budget {cfg.kappa}"


# ── Value diagnostics ───────────────────────────────────────────────────────


class TestExplainedVariance:
    def test_perfect_prediction_is_one(self):
        t = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert explained_variance(t, t) == pytest.approx(1.0)

    def test_constant_prediction_is_zero(self):
        """A dead critic scores 0, where MSE would read a misleading ~1.0."""
        t = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert explained_variance(torch.zeros(4), t) == pytest.approx(0.0, abs=1e-6)

    def test_constant_target_does_not_divide_by_zero(self):
        assert explained_variance(torch.zeros(4), torch.ones(4)) == 0.0


# ── PPO losses ──────────────────────────────────────────────────────────────


class TestPPOLosses:
    def _inputs(self, B: int):
        return {
            "advantage": torch.randn(B),
            "value_target": torch.rand(B) * 2 - 1,
            "value_old": torch.zeros(B),
        }

    def test_runs_and_produces_finite_gradients(self):
        policy = _tiny_policy()
        x = _batch(B=8, seed=2)
        from ptcg_rl.actor import recompute_logp

        with torch.no_grad():
            logp_old, _ = recompute_logp(policy, x)

        loss, stats = ppo_losses(
            policy, None, x, logp_old=logp_old, beta=0.1,
            cfg=RLConfig(minibatch=8, rollout_buffer=8), check_canary=True,
            **self._inputs(8),
        )
        loss.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        assert grads, "no parameter received a gradient"
        assert all(torch.isfinite(g).all() for g in grads)
        assert stats.ratio_p99 < 1e-3

    def test_kl_anchor_against_itself_is_zero(self):
        """KL(π_θ ‖ π_IL) must vanish when the two policies are identical."""
        import copy

        policy = _tiny_policy()
        reference = copy.deepcopy(policy).eval()
        x = _batch(B=8, seed=6)
        from ptcg_rl.actor import recompute_logp

        with torch.no_grad():
            logp_old, _ = recompute_logp(policy, x)

        _loss, stats = ppo_losses(
            policy, reference, x, logp_old=logp_old, beta=1.0,
            cfg=RLConfig(minibatch=8, rollout_buffer=8), **self._inputs(8),
        )
        assert abs(stats.kl_to_il) < 1e-5, (
            f"a policy diverged from a copy of itself by {stats.kl_to_il}"
        )

    def test_k3_kl_is_non_negative_where_the_raw_mean_is_not(self):
        """β must be driven by an estimator that cannot go negative.

        ``E[log π_θ(a) − log π_IL(a)]`` with ``a ~ π_θ`` is unbiased for the KL
        but is not itself non-negative on a finite minibatch. A real 128-decision
        rollout produced raw −0.0204 against k3 +0.0362 — feeding the raw value
        to ``update_beta`` reads as "the anchor is slack" and pushes β *down*
        exactly when the policy has drifted past the budget.
        """
        import copy

        policy = _tiny_policy()
        reference = copy.deepcopy(policy).eval()
        # Perturb θ so the two policies genuinely differ.
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(torch.randn_like(p) * 0.05)

        x = _batch(B=32, seed=21)
        from ptcg_rl.actor import recompute_logp

        with torch.no_grad():
            logp_old, _ = recompute_logp(policy, x)

        _loss, stats = ppo_losses(
            policy, reference, x, logp_old=logp_old, beta=1.0,
            cfg=RLConfig(minibatch=32, rollout_buffer=32),
            **self._inputs(32),
        )
        assert stats.kl_to_il_k3 >= 0.0, (
            f"k3 estimator went negative ({stats.kl_to_il_k3}); it is "
            f"non-negative by construction, so this is a formula error"
        )

    def test_mismatched_reference_shape_raises(self):
        """Two policies must be scored on the same decision points (§9.3)."""
        policy = _tiny_policy()
        x = _batch(B=4, seed=8)

        class _ShortReference:
            D = policy.D

            def __getattr__(self, name):
                return getattr(policy, name)

        from ptcg_rl.actor import recompute_logp

        with torch.no_grad():
            logp_old, _ = recompute_logp(policy, x)

        # A reference that silently returns fewer rows must not be tolerated.
        import ptcg_rl.ppo as ppo_mod

        original = ppo_mod.recompute_logp
        ppo_mod.recompute_logp = lambda p, b, **kw: (
            (torch.zeros(2), torch.zeros(2)) if p is not policy else original(p, b, **kw)
        )
        try:
            with pytest.raises(ValueError, match="decision points"):
                ppo_losses(
                    policy, _ShortReference(), x, logp_old=logp_old, beta=1.0,
                    cfg=RLConfig(minibatch=4, rollout_buffer=4), **self._inputs(4),
                )
        finally:
            ppo_mod.recompute_logp = original

    def test_clip_fraction_is_zero_at_epoch_zero(self):
        policy = _tiny_policy()
        x = _batch(B=8, seed=12)
        from ptcg_rl.actor import recompute_logp

        with torch.no_grad():
            logp_old, _ = recompute_logp(policy, x)
        _loss, stats = ppo_losses(
            policy, None, x, logp_old=logp_old, beta=0.0,
            cfg=RLConfig(minibatch=8, rollout_buffer=8), **self._inputs(8),
        )
        assert stats.clip_fraction == 0.0, "unchanged weights should clip nothing"


# ── Config validation ───────────────────────────────────────────────────────


class TestRLConfig:
    def test_rejects_a_ragged_minibatch(self):
        with pytest.raises(ValueError, match="multiple of"):
            RLConfig(rollout_buffer=1000, minibatch=384)

    def test_rejects_beta_init_outside_bounds(self):
        with pytest.raises(ValueError, match="beta_init"):
            RLConfig(beta_init=100.0, beta_max=10.0)

    def test_rejects_non_positive_kappa(self):
        with pytest.raises(ValueError, match="kappa"):
            RLConfig(kappa=0.0)

    def test_defaults_match_the_spec(self):
        cfg = RLConfig()
        assert cfg.kappa == 0.02
        assert cfg.clip_eps == 0.2
        assert cfg.gamma == 1.0 and cfg.gae_lambda == 0.95
        assert cfg.lr <= 3e-5, "RL lr must stay well below IL's 3e-4 (§9.4)"


# ── Gate ────────────────────────────────────────────────────────────────────


class TestWilson:
    @pytest.mark.parametrize("n,expected_lb", [
        (100, 0.604), (200, 0.633), (400, 0.653), (1000, 0.671),
    ])
    def test_matches_the_spec_table(self, n, expected_lb):
        """RL_SPEC §10.2's published lower bounds at p̂ = 0.70."""
        lb, centre, _hi = wilson_interval(0.70 * n, n)
        assert lb == pytest.approx(expected_lb, abs=0.001)
        assert centre == pytest.approx(0.70, abs=1e-9)

    def test_bounds_stay_inside_the_unit_interval(self):
        """Where the normal approximation famously fails."""
        for wins, n in ((0, 20), (20, 20), (1, 400), (399, 400)):
            lb, _c, ub = wilson_interval(wins, n)
            assert 0.0 <= lb <= ub <= 1.0

    def test_zero_games(self):
        assert wilson_interval(0, 0) == (0.0, 0.0, 1.0)


class TestScoring:
    def test_draws_score_half(self):
        results = [{"winner": 0, "our_player": 0}, {"winner": -1, "our_player": 0}]
        assert score_from_results(results) == (1.5, 2)

    def test_our_player_is_respected(self):
        """Playing second must not be scored as a loss just because winner != 0."""
        results = [{"winner": 1, "our_player": 1}]
        assert score_from_results(results) == (1.0, 1)

    def test_paired_seeds_play_each_side_once(self):
        seeds = paired_seeds(3, base_seed=10)
        assert len(seeds) == 6
        assert sorted(seeds) == [(10, 0), (10, 1), (11, 0), (11, 1), (12, 0), (12, 1)]


class TestEvaluateGate:
    def _wins(self, k: int, n: int) -> list[dict]:
        return (
            [{"winner": 0, "our_player": 0}] * k
            + [{"winner": 1, "our_player": 0}] * (n - k)
        )

    def test_strong_candidate_passes(self):
        r = evaluate_gate(self._wins(620, 800), min_score=0.60, min_wilson_lb=0.55,
                          il_nontrivial_top1=0.75, il_baseline=0.76)
        assert r.passed, r.summary()

    def test_a_high_score_on_few_games_fails_the_wilson_floor(self):
        """The whole point of condition 2: a good point estimate is not enough."""
        r = evaluate_gate(self._wins(14, 20), min_score=0.60, min_wilson_lb=0.55)
        assert r.score >= 0.60
        assert not r.passed
        assert r.conditions["wilson_lb"] is False

    def test_il_regression_blocks_promotion(self):
        """Winning more games does not license forgetting the corpus."""
        r = evaluate_gate(self._wins(700, 800), min_score=0.60, min_wilson_lb=0.55,
                          il_nontrivial_top1=0.60, il_baseline=0.75,
                          max_il_regression_pts=5.0)
        assert not r.passed
        assert r.conditions["il_regression"] is False
        assert r.il_regression_pts == pytest.approx(-15.0)

    def test_a_missing_baseline_is_recorded_as_skipped_not_passed(self):
        """A check that could not run must never look like a check that passed."""
        r = evaluate_gate(self._wins(620, 800), min_score=0.60, min_wilson_lb=0.55)
        assert "il_regression" not in r.conditions
        assert any("SKIPPED" in n for n in r.notes)

    def test_kl_budget_condition(self):
        base = dict(min_score=0.60, min_wilson_lb=0.55)
        assert evaluate_gate(self._wins(620, 800), kl_to_il=0.05, max_kl=0.08,
                             **base).conditions["kl_budget"]
        assert not evaluate_gate(self._wins(620, 800), kl_to_il=0.5, max_kl=0.08,
                                 **base).conditions["kl_budget"]

    def test_summary_names_the_failing_conditions(self):
        r = evaluate_gate(self._wins(400, 800), min_score=0.60, min_wilson_lb=0.55)
        assert "FAIL" in r.summary()
        assert "score" in r.summary()


# ── Critic diagnostics ──────────────────────────────────────────────────────


class TestCriticDiagnostics:
    def test_detects_the_collapsed_head(self):
        """RL_SPEC §3's measured signature: std 0.0006 against target std 0.994."""
        rng = np.random.default_rng(0)
        targets = rng.choice([-1.0, 1.0], 2000)
        d = diagnose(rng.normal(0.0, 0.0006, 2000), targets)
        assert d.collapsed
        assert not d.passes()
        assert abs(d.corr) < 0.1
        assert d.mse == pytest.approx(1.0, abs=0.1)

    def test_accepts_a_healthy_head(self):
        rng = np.random.default_rng(1)
        targets = rng.choice([-1.0, 1.0], 4000)
        turns = rng.integers(0, 100, 4000)
        # Signal that sharpens as the game progresses.
        preds = np.tanh(targets * 0.9 + rng.normal(0, 1.0, 4000) * (1 - turns / 120))
        d = diagnose(preds, targets, turns)
        assert not d.collapsed
        assert d.passes(), d.summary()
        assert d.accuracy_rises_with_turn

    def test_a_flat_critic_fails_the_turn_check(self):
        """Equal accuracy early and late means nothing about progress was learned."""
        rng = np.random.default_rng(2)
        targets = rng.choice([-1.0, 1.0], 4000)
        turns = rng.integers(0, 100, 4000)
        preds = targets * 0.5 + rng.normal(0, 0.3, 4000)  # no turn dependence
        d = diagnose(preds, targets, turns)
        assert d.corr > 0.35 and d.pred_std > 0.3
        assert not d.accuracy_rises_with_turn
        assert not d.passes(), "a turn-flat critic should not clear the R1 gate"

    def test_constant_series_yields_zero_not_nan(self):
        d = diagnose([0.5] * 100, [1.0] * 100)
        assert d.corr == 0.0
        assert np.isfinite(d.corr)

    def test_empty_input(self):
        d = diagnose([], [])
        assert d.n_samples == 0
        assert not d.passes()

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="differ"):
            diagnose([1.0, 2.0], [1.0])

    def test_too_few_buckets_is_not_a_pass(self):
        """`accuracy_rises_with_turn` must not pass on insufficient data."""
        d = CriticDiagnostics(per_turn_accuracy=[(0, 0.9, 5)])
        assert not d.accuracy_rises_with_turn

    def test_a_perfect_critic_with_no_progress_data_cannot_pass(self):
        """Pins the shape of the gate: no progress data ⇒ no pass.

        This is deliberate — §3's third criterion is a real requirement — but it
        makes ``measure()`` supplying progress **load-bearing**. When it did not,
        ``per_turn_accuracy`` was always empty, ``accuracy_rises_with_turn`` was
        always False, and the R1 gate was unpassable no matter how good the
        critic got. The failure looked exactly like "the critic is still broken".
        """
        rng = np.random.default_rng(3)
        targets = rng.choice([-1.0, 1.0], 2000)
        perfect = diagnose(targets * 0.95, targets)   # corr ≈ 1, std ≈ 0.95
        assert perfect.corr > 0.9 and perfect.pred_std > 0.9
        assert not perfect.passes(), (
            "a near-perfect critic passed without any progress data — the third "
            "criterion is not being enforced"
        )
        assert perfect.per_turn_accuracy == []

    def test_progress_proxy_populates_the_buckets(self):
        """``_progress`` must yield enough spread to bucket.

        The shards carry no turn index, so game progress is derived from the two
        prize counts in ``cls_feat``. If that derivation breaks — a layout change,
        a wrong column — the symptom is not an exception but a permanently
        failing R1 gate.
        """
        pytest.importorskip("torch")
        from ptcg_rl.critic import CLS_OPP_PRIZES, CLS_OUR_PRIZES, _progress

        n = 500
        cls = torch.zeros(n, 93)
        # Prizes remaining fall from 1.0 to 0.0 as the game progresses.
        frac = torch.linspace(1.0, 0.0, n)
        cls[:, CLS_OUR_PRIZES] = frac
        cls[:, CLS_OPP_PRIZES] = frac
        prog = _progress({"cls_feat": cls})

        assert prog.min() == 0, "progress should start at zero prizes taken"
        assert prog.max() > 0, "progress never advanced"
        assert len(np.unique(prog)) >= 5, (
            f"only {len(np.unique(prog))} distinct progress values — too few to "
            f"bucket, so the turn criterion would never be evaluable"
        )


# ── The reference pass must share the update's kernels ──────────────────────


class TestReferenceLogpSharesTheUpdatePath:
    """``logp_old`` must be produced by the same kernels as the PPO update.

    ``nn.TransformerEncoderLayer`` takes a *fused* fast path only when the
    module is in eval mode **and** grad is disabled.  The PPO update is
    grad-enabled, so it never takes it; a reference pass under
    ``eval()`` + ``no_grad()`` always does.  The two kernels disagree by ~2e-4
    on the encoder output, which reaches the epoch-0 ratio as
    p99 |ratio − 1| ≈ 3.7e-3 and trips the canary at its 1e-3 tolerance.

    This is the §7 failure the canary exists for, and it is invisible in the
    loss: nothing is NaN, nothing is out of range, the ratios are merely not 1.
    Stage 5 could not reach it until the AR loop stopped OOM'ing first.
    """

    def test_encoder_fast_path_is_mode_dependent(self):
        """Document the PyTorch behaviour the bug rests on.

        If a future torch drops the fast path this goes green trivially — but
        the ratio test below is the one that actually guards the invariant.
        """
        from tests.test_rl_actor import _batch, _tiny_policy

        policy = _tiny_policy()
        x = _batch(B=4)

        policy.eval()
        with torch.no_grad():
            h_eval_nograd = policy._encode(x)[0]
        h_eval_grad = policy._encode(x)[0].detach()

        policy.train()
        with torch.no_grad():
            h_train_nograd = policy._encode(x)[0]
        h_train_grad = policy._encode(x)[0].detach()

        assert torch.equal(h_train_nograd, h_train_grad), (
            "train mode must use one kernel regardless of grad; if this fails "
            "the mode-matching fix below is not sufficient"
        )
        # The eval/no_grad combination is the odd one out.
        assert not torch.equal(h_eval_nograd, h_eval_grad), (
            "eval()+no_grad() no longer diverges from the grad path; the "
            "fast-path hazard may be gone, but check before relaxing anything"
        )

    def test_epoch_zero_ratio_is_exactly_one(self):
        """The canary's premise, end to end through ``_recompute_buffer_logp``."""
        from ptcg_rl.actor import recompute_logp
        from ptcg_rl.rollout import RolloutBatch
        from ptcg_rl.train import _recompute_buffer_logp
        from tests.test_rl_actor import _batch, _tiny_policy

        B = 8
        policy = _tiny_policy()
        x = _batch(B=B, seed=3)

        feats = [
            {k: v[b].numpy() for k, v in x.items()
             if torch.is_tensor(v) and k not in ("action_idx", "action_len")}
            for b in range(B)
        ]
        batch = RolloutBatch(
            features=feats,
            action_idx=x["action_idx"].numpy(),
            action_len=x["action_len"].numpy(),
            logp_old=np.zeros(B, dtype=np.float32),
            advantage=np.zeros(B, dtype=np.float32),
            value_target=np.zeros(B, dtype=np.float32),
            value_old=np.zeros(B, dtype=np.float32),
            turn=np.zeros(B, dtype=np.int64),
        )

        cfg = RLConfig(deck_archetype=0, minibatch=4, rollout_buffer=8)
        _recompute_buffer_logp(policy, batch, cfg, torch.device("cpu"))

        # Exactly what _ppo_epochs does next: grad-enabled recomputation.
        h, _ = policy._encode({k: v for k, v in x.items()})
        logp, _ = recompute_logp(policy, x, encoded=h)

        # Asserted *bitwise*, not against ratio_canary_tol.  The tolerance is a
        # property of the real model: on this tiny CPU policy the eval fast-path
        # divergence lands around 1e-5, comfortably inside 1e-3, so a tolerance
        # assertion here would pass with the bug still in place and prove
        # nothing.  "Same kernels" means identical, and the fix delivers that.
        ref = torch.as_tensor(batch.logp_old)
        assert torch.equal(logp.detach(), ref), (
            f"logp_old was not computed by the update's kernels; "
            f"max |Δlogp| = {float((logp.detach() - ref).abs().max()):.3e}, "
            f"max |ratio-1| = "
            f"{float(((logp.detach() - ref).exp() - 1.0).abs().max()):.3e}"
        )
