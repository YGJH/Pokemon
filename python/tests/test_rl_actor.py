"""``ptcg_rl.actor`` — joint-sequence log-probs and masked sampling.

The headline test is :class:`TestParityWithMultiselectCE`.  RL_SPEC §5 lists
"per-step instead of joint-sequence log-probs" as high-severity and *silent*:
training runs, the loss falls, and the policy drifts toward short selections
because every dropped STOP step made them cheaper.  Nothing catches that except
an equality check against the IL objective the anchor is defined in terms of.

Several tests count what they examined and assert the count is non-zero.  A test
that only checks a property when the fixture happens to contain the relevant
option type passes vacuously on an empty loop — which is how two convincing but
wrong belief-coverage results got past review during R0b.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ptcg_il.model.policy import Policy, multiselect_ce  # noqa: E402
from ptcg_rl.actor import (  # noqa: E402
    mask_step_logits,
    masked_entropy,
    masked_log_softmax,
    recompute_logp,
    sample_action,
)

from tests.test_model_policy import _make_synthetic_batch  # noqa: E402

from ptcg_il.model.pointer import O_MAX  # noqa: E402

# _make_synthetic_batch marks options 0..7 valid, and puts the STOP column at 7
# when max_count > 1 — so a multi-select decision has 7 real options.
N_REAL_OPTS = 7
STOP_COL = 7


def _tiny_policy(V: int = 100, A: int = 50, D: int = 64) -> Policy:
    """A small policy on the same vocab sizes ``_make_synthetic_batch`` uses."""
    torch.manual_seed(0)
    p = Policy(V=V, A=A, D=D, heads=4, layers=1, ff=128)
    p.eval()
    return p


def _batch(
    B: int = 6,
    *,
    multi: bool = True,
    min_count: int = 1,
    max_count: int = 3,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """A featurizer-shaped batch carrying a valid multi-select action label.

    Built on the IL test suite's own ``_make_synthetic_batch`` so the key set and
    tensor shapes cannot drift from what the embedder and pointer actually read;
    this only fills in the labels, which that helper leaves empty.
    """
    torch.manual_seed(seed)
    x = _make_synthetic_batch(B, max_count=max_count if multi else 1)
    x["minCount"] = torch.full((B,), min_count, dtype=torch.long)

    n_opts = N_REAL_OPTS if multi else 8
    action_idx = torch.full_like(x["action_idx"], -1)
    action_len = torch.zeros(B, dtype=torch.long)
    g = torch.Generator().manual_seed(seed + 1)
    for b in range(B):
        span = max(1, max_count - min_count + 1)
        k = min(min_count + (b % span), n_opts) if multi else 1
        for i, c in enumerate(torch.randperm(n_opts, generator=g)[:k].tolist()):
            action_idx[b, i] = c
        if multi:
            action_idx[b, k] = STOP_COL
            action_len[b] = k + 1
        else:
            action_len[b] = k

    x["action_idx"] = action_idx
    x["action_len"] = action_len
    return x


# ── The parity gate ─────────────────────────────────────────────────────────


class TestParityWithMultiselectCE:
    """``recompute_logp`` must be exactly ``−multiselect_ce`` at zero smoothing.

    ``multiselect_ce`` is the objective π_IL was fitted with, so if these two
    disagree the KL anchor compares θ against a distribution IL never
    represented, and the anchor silently stops anchoring.
    """

    def test_logp_equals_negative_ce(self):
        policy = _tiny_policy()
        x = _batch(B=8)

        with torch.no_grad():
            ce = multiselect_ce(policy, x, label_smoothing=0.0)
            logp, _ = recompute_logp(policy, x)

        assert torch.allclose(logp, -ce, atol=1e-5), (
            f"joint log-prob disagrees with multiselect_ce:\n"
            f"  logp = {logp}\n  -ce  = {-ce}"
        )

    def test_parity_holds_for_single_select(self):
        policy = _tiny_policy()
        x = _batch(B=6, multi=False, min_count=1, max_count=1)

        with torch.no_grad():
            ce = multiselect_ce(policy, x, label_smoothing=0.0)
            logp, _ = recompute_logp(policy, x)

        assert torch.allclose(logp, -ce, atol=1e-5)

    def test_stop_step_is_included(self):
        """Truncating the STOP target must change the log-prob.

        This is the direct test for the §5 failure: if ``recompute_logp``
        silently ignored the final STOP column, dropping it would be a no-op.
        """
        policy = _tiny_policy()
        x = _batch(B=4)

        with torch.no_grad():
            full, _ = recompute_logp(policy, x)

            truncated = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
            n_dropped = 0
            for b in range(x["action_len"].shape[0]):
                last = int(x["action_len"][b]) - 1
                if int(x["action_idx"][b, last]) == int(x["stop_column"][b]):
                    truncated["action_idx"][b, last] = -1
                    truncated["action_len"][b] = last
                    n_dropped += 1
            no_stop, _ = recompute_logp(policy, truncated)

        assert n_dropped > 0, "fixture contained no STOP steps — test proves nothing"
        assert not torch.allclose(full, no_stop), (
            "dropping the STOP target changed nothing, so it is not being scored"
        )
        # Every dropped step removes a negative term, so the truncated sum is larger.
        assert (no_stop >= full - 1e-6).all()


# ── Masking ─────────────────────────────────────────────────────────────────


class TestMasking:
    def test_illegal_options_get_zero_probability(self):
        logits = torch.randn(3, 8)
        picked = torch.zeros(3, 8, dtype=torch.bool)
        picked[:, :4] = True
        masked = mask_step_logits(
            logits, picked, t=0,
            min_count=torch.zeros(3, dtype=torch.long),
            stop_column=torch.full((3,), -1, dtype=torch.long),
        )
        probs = masked_log_softmax(masked).exp()
        assert torch.allclose(probs[:, 4:], torch.zeros(3, 4))
        assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-6)

    def test_stop_forbidden_below_min_count(self):
        """STOP must be illegal until minCount picks have been taken."""
        logits = torch.zeros(2, 6)
        picked = torch.ones(2, 6, dtype=torch.bool)
        min_count = torch.tensor([2, 2])
        stop_column = torch.tensor([5, 5])

        for t, expect_legal in ((0, False), (1, False), (2, True)):
            masked = mask_step_logits(logits, picked, t, min_count, stop_column)
            p_stop = masked_log_softmax(masked).exp()[:, 5]
            if expect_legal:
                assert (p_stop > 0).all(), f"STOP should be legal at t={t}"
            else:
                assert torch.allclose(p_stop, torch.zeros(2)), (
                    f"STOP was selectable at t={t} with minCount=2"
                )

    def test_no_stop_column_is_left_alone(self):
        """stop_column == -1 must not mask column -1 (i.e. the last option)."""
        logits = torch.zeros(2, 6)
        picked = torch.ones(2, 6, dtype=torch.bool)
        masked = mask_step_logits(
            logits, picked, t=0,
            min_count=torch.tensor([1, 1]),
            stop_column=torch.tensor([-1, -1]),
        )
        assert torch.isfinite(masked).all(), (
            "a -1 stop_column masked a real option via negative indexing"
        )

    def test_entropy_is_finite_and_bounded(self):
        """Masked entropy must never be NaN, and must be ≤ log(n_legal)."""
        logits = torch.randn(4, 10)
        picked = torch.zeros(4, 10, dtype=torch.bool)
        picked[:, :3] = True
        log_probs = masked_log_softmax(
            mask_step_logits(logits, picked, 0,
                             torch.zeros(4, dtype=torch.long),
                             torch.full((4,), -1, dtype=torch.long))
        )
        h = masked_entropy(log_probs)
        assert torch.isfinite(h).all(), "0 · -inf leaked a NaN into the entropy"
        assert (h <= np.log(3) + 1e-5).all()
        assert (h >= 0).all()

    def test_entropy_gradient_is_finite_at_masked_positions(self):
        """The value being finite is not enough — the *gradient* must be too.

        ``torch.where(finite, p·log p, 0)`` yields the correct entropy and a NaN
        gradient, because ``where`` evaluates both branches and autograd
        propagates through the discarded one.  Entropy is a loss term, so that
        NaN reaches every parameter in the model and training dies silently on
        the first step.
        """
        logits = torch.randn(4, 10, requires_grad=True)
        picked = torch.zeros(4, 10, dtype=torch.bool)
        picked[:, :3] = True
        log_probs = masked_log_softmax(
            mask_step_logits(logits, picked, 0,
                             torch.zeros(4, dtype=torch.long),
                             torch.full((4,), -1, dtype=torch.long))
        )
        masked_entropy(log_probs).sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all(), (
            "masked entropy produced a NaN gradient; every parameter would inherit it"
        )


# ── Sampling ────────────────────────────────────────────────────────────────


class TestSampleAction:
    def test_picks_are_legal_and_never_repeat(self):
        policy = _tiny_policy()
        x = _batch(B=16, max_count=3, seed=3)
        g = torch.Generator().manual_seed(7)

        n_checked = 0
        for _ in range(20):
            picks, _, _ = sample_action(policy, x, generator=g)
            for b, row in enumerate(picks):
                legal = set(torch.where(x["opt_mask"][b])[0].tolist())
                stop = int(x["stop_column"][b])
                assert set(row) <= legal, f"illegal pick {row} vs legal {sorted(legal)}"
                assert stop not in row, "STOP leaked into the engine-facing picks"
                assert len(row) == len(set(row)), f"repeated pick in {row}"
                assert len(row) <= int(x["maxCount"][b])
                n_checked += 1

        assert n_checked == 320, f"expected 320 sampled rows, checked {n_checked}"

    def test_logp_matches_teacher_forcing_of_what_was_sampled(self):
        """The returned log-prob must be the log-prob of the returned action.

        Scoring a different sequence than the one taken is the subtlest version
        of the §5 bug, and it survives every "is the action legal" check.
        """
        policy = _tiny_policy()
        x = _batch(B=8, max_count=3, seed=11)
        g = torch.Generator().manual_seed(1)

        picks, logp, _ = sample_action(policy, x, generator=g)

        # Rebuild the label the sampler implies, STOP step included.
        replay = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        replay["action_idx"] = torch.full_like(x["action_idx"], -1)
        replay["action_len"] = torch.zeros_like(x["action_len"])
        for b, row in enumerate(picks):
            for i, c in enumerate(row):
                replay["action_idx"][b, i] = c
            n = len(row)
            stop = int(x["stop_column"][b])
            if stop >= 0 and n < int(x["maxCount"][b]):
                replay["action_idx"][b, n] = stop
                n += 1
            replay["action_len"][b] = n

        with torch.no_grad():
            re_logp, _ = recompute_logp(policy, replay)

        assert torch.allclose(logp, re_logp, atol=1e-5), (
            f"sampled log-prob != teacher-forced log-prob of the same action:\n"
            f"  sampled = {logp}\n  replay  = {re_logp}"
        )

    def test_greedy_is_deterministic(self):
        policy = _tiny_policy()
        x = _batch(B=6, seed=5)
        a, la, _ = sample_action(policy, x, greedy=True)
        b, lb, _ = sample_action(policy, x, greedy=True)
        assert a == b
        assert torch.allclose(la, lb)

    def test_empty_selection_scores_zero_not_neg_inf(self):
        """A row that takes no picks contributes an empty sum (§5).

        ``-inf`` here would propagate through the PPO ratio and NaN the update.
        """
        policy = _tiny_policy()
        x = _batch(B=3, multi=False, max_count=1)
        x["action_idx"] = torch.full_like(x["action_idx"], -1)
        x["action_len"] = torch.zeros_like(x["action_len"])

        with torch.no_grad():
            logp, ent = recompute_logp(policy, x)

        assert torch.allclose(logp, torch.zeros(3)), f"expected 0.0, got {logp}"
        assert torch.isfinite(logp).all()
        assert torch.allclose(ent, torch.zeros(3))

    def test_temperature_changes_the_distribution_not_the_score(self):
        """Temperature is a proposal knob; the score stays the true density.

        If the log-prob were computed from the tempered distribution, the PPO
        ratio would be over a density the policy does not define.
        """
        policy = _tiny_policy()
        x = _batch(B=4, seed=9)
        picks, logp, _ = sample_action(
            policy, x, temperature=2.0, generator=torch.Generator().manual_seed(2)
        )

        replay = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        replay["action_idx"] = torch.full_like(x["action_idx"], -1)
        replay["action_len"] = torch.zeros_like(x["action_len"])
        for b, row in enumerate(picks):
            for i, c in enumerate(row):
                replay["action_idx"][b, i] = c
            n = len(row)
            stop = int(x["stop_column"][b])
            if stop >= 0 and n < int(x["maxCount"][b]):
                replay["action_idx"][b, n] = stop
                n += 1
            replay["action_len"][b] = n

        with torch.no_grad():
            re_logp, _ = recompute_logp(policy, replay)
        assert torch.allclose(logp, re_logp, atol=1e-5)

    def test_rejects_non_positive_temperature(self):
        policy = _tiny_policy()
        with pytest.raises(ValueError, match="temperature"):
            sample_action(policy, _batch(B=2), temperature=0.0)


# ── Cost of the autoregressive loop ─────────────────────────────────────────


def _record_pointer_rows(policy) -> list[int]:
    """Record the row count of every ``PointerHead`` forward, in order."""
    seen: list[int] = []
    inner = policy.pointer.forward

    def spy(h, *args, **kwargs):
        seen.append(int(h.shape[0]))
        return inner(h, *args, **kwargs)

    policy.pointer.forward = spy
    return seen


class TestARLoopRunsOnActiveRowsOnly:
    """Step *t* must score only the rows that still have a pick at step *t*.

    ``recompute_logp`` is the one AR loop that runs under ``enable_grad`` — the
    PPO update calls it — so every pointer forward it makes is retained for
    backward at ``B × O_MAX × 4D``.  Driving all ``B`` rows off the batch *max*
    ``action_len`` made that ``max(action_len)`` times larger than the work
    requires, and the cost is paid by the whole minibatch: in the real corpus
    94.4% of decisions take a single pick, so one 30-pick row used to drag every
    other row through 30 forwards.  Stage 5 OOM'd on a 16 GiB card at the first
    grad-enabled minibatch because of it.

    Nothing else catches this.  The math is unchanged — the parity gate above
    stays green either way — so it shows up only as memory.
    """

    def test_rows_per_step_track_the_active_set(self):
        policy = _tiny_policy()
        x = _batch(B=6, min_count=1, max_count=3)
        action_len = x["action_len"]

        n_steps = int(action_len.max())
        expected = [int((action_len > t).sum()) for t in range(n_steps)]

        # Guard against a fixture where every row happens to be the same length:
        # the assertion below would then hold vacuously for the old loop too.
        assert len(set(expected)) > 1, (
            f"fixture has uniform action_len {action_len.tolist()}; this test "
            f"cannot distinguish per-step subsetting from a full-batch loop"
        )

        seen = _record_pointer_rows(policy)
        with torch.no_grad():
            recompute_logp(policy, x)

        assert seen == expected, (
            f"pointer ran on {seen} rows per step, expected {expected}.\n"
            f"action_len = {action_len.tolist()}"
        )

    def test_a_single_long_row_does_not_drag_the_batch(self):
        """The corpus shape: one long multi-select among single-pick rows.

        Total row-steps must stay ``sum(action_len)``, not ``B × max``.
        """
        policy = _tiny_policy()
        x = _batch(B=8, min_count=1, max_count=1, multi=False)
        # Give row 3 alone a 5-pick sequence; every other row keeps its 1 pick.
        x["maxCount"] = torch.full((8,), 5, dtype=torch.long)
        x["action_idx"][3, :5] = torch.arange(5)
        x["action_len"][3] = 5

        seen = _record_pointer_rows(policy)
        with torch.no_grad():
            recompute_logp(policy, x)

        assert sum(seen) == int(x["action_len"].sum()) == 12, (
            f"ran {sum(seen)} row-steps ({seen}) for "
            f"{int(x['action_len'].sum())} picks; the old loop would run "
            f"{8 * 5} = B x max(action_len)"
        )

    def test_log_probs_are_unchanged_by_subsetting(self):
        """Scoring a subset must give the same answer as scoring the batch.

        Runs each row alone and compares to the batched call: the batched path
        indexes and scatters, and an off-by-one in either would show up here as
        a row scored with another row's GRU state.
        """
        policy = _tiny_policy()
        x = _batch(B=5, min_count=1, max_count=3, seed=11)

        with torch.no_grad():
            logp, ent = recompute_logp(policy, x)
            for b in range(x["action_len"].shape[0]):
                one = {k: (v[b : b + 1] if torch.is_tensor(v) else v)
                       for k, v in x.items()}
                lp_b, ent_b = recompute_logp(policy, one)
                assert torch.allclose(logp[b], lp_b[0], atol=1e-5), (
                    f"row {b}: batched {float(logp[b])} != alone {float(lp_b[0])}"
                )
                assert torch.allclose(ent[b], ent_b[0], atol=1e-5)
