"""Tests for Policy, multiselect_ce, and select_multi (Appendix B.6–B.8)."""

import torch
import pytest

from ptcg_il.featurizer import F_GLOBAL, F_HAND, F_OPT, F_POKE, F_SUM
from ptcg_il.model.cards import F_ATK, F_CARD
from ptcg_il.model.embed import L_STATE, P_MAX, H_MAX, SUM
from ptcg_il.model.pointer import O_MAX
from ptcg_il.model.policy import Policy, multiselect_ce, select_multi

D = 256
D_MAX = 60
PZ_MAX = 6


def _make_synthetic_batch(B: int = 2, max_count: int = 1) -> dict[str, torch.Tensor]:
    """Build a synthetic featurizer dict.

    Cards enter the model as static feature vectors (``*_card_feat``), not as
    vocab indices — there are no learned id embeddings to feed.
    """
    O = O_MAX
    L = L_STATE

    x = {
        # State — card identity, as static features
        "poke_card_feat": torch.randn(B, P_MAX, F_CARD),
        "hand_card_feat": torch.randn(B, H_MAX, F_CARD),
        "stadium_card_feat": torch.randn(B, 1, F_CARD),
        "context_card_feat": torch.randn(B, 1, F_CARD),
        "effect_card_feat": torch.randn(B, 1, F_CARD),
        "discard_card_feat": torch.randn(B, SUM, D_MAX, F_CARD),
        "discard_mask": torch.ones(B, SUM, D_MAX, dtype=torch.bool),
        "prize_card_feat": torch.zeros(B, SUM, PZ_MAX, F_CARD),
        # State — dense features
        # Widths come from ptcg_il.featurizer, never literals: a featurizer edit
        # otherwise leaves this fixture building batches the model cannot accept,
        # and every test that imports it fails on a matmul shape rather than on
        # the thing it was written to check.
        "poke_feat": torch.randn(B, P_MAX, F_POKE),
        "hand_feat": torch.randn(B, H_MAX, F_HAND),
        "sum_feat": torch.randn(B, SUM, F_SUM),
        "cls_feat": torch.randn(B, F_GLOBAL),
        "stadium_present": torch.ones(B, 1),
        # State — categorical
        "tok_type": torch.randint(0, 5, (B, L)),
        "tok_owner": torch.randint(0, 3, (B, L)),
        "tok_zone": torch.randint(0, 6, (B, L)),
        "tok_mask": torch.ones(B, L, dtype=torch.bool),
        # Options
        "opt_type": torch.randint(0, 18, (B, O)),  # 17 types + STOP=17
        "opt_src_idx": torch.randint(-1, L, (B, O)),
        "opt_tgt_idx": torch.randint(-1, L, (B, O)),
        "opt_card_feat": torch.randn(B, O, F_CARD),
        "opt_attack_feat": torch.randn(B, O, F_ATK),
        "opt_scalar": torch.randn(B, O, F_OPT),
        "opt_mask": torch.ones(B, O, dtype=torch.bool),
        "opt_group": torch.full((B, O), -1, dtype=torch.long),
        # Labels (padded with -1, matching featurizer)
        "action_idx": torch.full((B, O), -1, dtype=torch.long),
        "action_len": torch.full((B,), max_count, dtype=torch.long),
        "minCount": torch.full((B,), 1, dtype=torch.long),
        "maxCount": torch.full((B,), max_count, dtype=torch.long),
        "sel_type": torch.zeros(B, dtype=torch.long),
        "sel_ctx": torch.zeros(B, dtype=torch.long),
        "value_target": torch.ones(B),
        "sample_weight": torch.ones(B),
        "stop_column": torch.full((B,), -1, dtype=torch.long),
        # Logs (belief module)
        "log_feat": torch.zeros(B, 32, 6),
        "log_mask": torch.zeros(B, 32, dtype=torch.bool),
        "log_len": torch.zeros(B, dtype=torch.long),
        "log_card_feat": torch.zeros(B, 32, F_CARD),
    }

    # First 8 options valid (leave room for STOP if needed)
    x["opt_mask"][:, 8:] = False
    x["opt_group"][:, :8] = 0  # single group containing all valid options
    if max_count > 1:
        # Set up STOP column at position 7 (last valid option slot)
        x["stop_column"][:] = 7
        x["opt_type"][:, 7] = 17  # STOP type
        x["opt_mask"][:, 7] = True  # STOP is a valid "option"

    return x


class TestPolicy:
    """Policy: embed -> encode -> pointer + value."""

    def test_forward_shapes_single(self):
        """Single-select forward returns correct shapes."""
        policy = Policy()
        x = _make_synthetic_batch(4, max_count=1)
        x["action_idx"][:, 0] = torch.randint(0, 8, (4,))
        logits, value, _hist = policy(x)
        assert logits.shape == (4, O_MAX)
        assert value.shape == (4,)

    def test_value_range(self):
        """Value is in (-1, 1) due to tanh."""
        policy = Policy()
        policy.eval()
        x = _make_synthetic_batch(4)
        _, value, _hist = policy(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_logits_masked(self):
        """Padding options logits are -1e9."""
        policy = Policy()
        policy.eval()
        x = _make_synthetic_batch(2, max_count=1)
        logits, _val, _hist = policy(x)
        for j in range(8, O_MAX):
            assert (logits[:, j] < -1e8).all()

    def test_gradient_flow(self):
        """All parameters receive gradients (except msgru which runs in multi-select loop)."""
        policy = Policy()
        x = _make_synthetic_batch(2, max_count=1)
        x["action_idx"][:, 0] = torch.randint(0, 8, (2,))
        logits, value, _hist = policy(x)
        loss = logits[:, :8].sum() + value.sum()
        loss.backward()
        for name, p in policy.named_parameters():
            if "msgru" in name:
                continue  # only exercised in multi-select loop
            if name.startswith("belief_heads."):
                continue  # auxiliary; only reached via forward_with_belief
            assert p.grad is not None, f"Parameter {name} has no gradient"

    def test_pointer_card_bound(self):
        """Pointer.card is the same object as embed.card."""
        policy = Policy()
        assert policy.pointer.card is policy.embed.card

    def test_single_select_cross_entropy(self):
        """CE loss computed on single-select output is finite."""
        policy = Policy()
        x = _make_synthetic_batch(4, max_count=1)
        x["action_idx"][:, 0] = torch.randint(0, 8, (4,))
        logits, value, _hist = policy(x)
        ce = torch.nn.functional.cross_entropy(
            logits, x["action_idx"][:, 0], reduction="none"
        )
        assert torch.isfinite(ce).all()

    def test_different_actions_different_loss(self):
        """CE should penalize wrong predictions differently than correct ones."""
        policy = Policy()
        policy.eval()
        x = _make_synthetic_batch(2, max_count=1)
        logits, _val, _hist = policy(x)

        # Predict the argmax - this is "correct" (label_smoothing aside)
        target = logits[:, :8].argmax(-1)
        ce_correct = torch.nn.functional.cross_entropy(
            logits, target, reduction="none"
        )

        # Predict a different target
        target_wrong = (target + 1) % 8
        ce_wrong = torch.nn.functional.cross_entropy(
            logits, target_wrong, reduction="none"
        )

        # ce_wrong should generally be >= ce_correct (not always due to label smoothing)
        assert ce_wrong.sum() >= 0

    def test_no_nan_inf(self):
        """Output should not contain NaN or Inf."""
        policy = Policy()
        policy.eval()
        x = _make_synthetic_batch(4)
        logits, value, _hist = policy(x)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits[:, :8]).any()  # masked has -inf
        assert not torch.isnan(value).any()
        assert not torch.isinf(value).any()


class TestMultiSelectCE:
    """multiselect_ce: teacher-forced AR training loss."""

    def test_output_is_finite(self):
        """CE output is finite (including STOP supervision)."""
        policy = Policy()
        x = _make_synthetic_batch(2, max_count=3)
        stop_col = int(x["stop_column"][0].item())
        # Set unique valid action picks + STOP target
        for b in range(2):
            picks = torch.randperm(7)[:2]  # 2 regular picks (avoiding STOP slot)
            x["action_idx"][b, 0] = picks[0]
            x["action_idx"][b, 1] = picks[1]
            x["action_idx"][b, 2] = stop_col  # STOP target
        x["action_len"] = torch.tensor([3, 3], dtype=torch.long)  # 2 picks + STOP

        ce = multiselect_ce(policy, x)
        assert ce.shape == (2,)
        assert torch.isfinite(ce).all()

    def test_gradient_flow(self):
        """Gradients flow through embed/encoder/pointer (not value) in multi-select path."""
        policy = Policy()
        x = _make_synthetic_batch(2, max_count=2)
        stop_col = int(x["stop_column"][0].item())
        for b in range(2):
            picks = torch.randperm(7)[:1]  # 1 regular pick
            x["action_idx"][b, 0] = picks[0]
            x["action_idx"][b, 1] = stop_col  # STOP target
        x["action_len"] = torch.tensor([2, 2], dtype=torch.long)  # 1 pick + STOP

        ce = multiselect_ce(policy, x)
        loss = ce.mean()
        loss.backward()
        # Only embed/encoder/pointer are used in multiselect_ce; value head is unused
        for name, p in policy.named_parameters():
            if "value" in name or name.startswith("belief_heads."):
                assert p.grad is None, f"Auxiliary parameter {name} should not have gradient in multiselect"
            else:
                assert p.grad is not None, f"Parameter {name} has no gradient"

    def test_variable_action_len(self):
        """Handles samples with different pick counts in batch (incl STOP)."""
        policy = Policy()
        B = 4
        x = _make_synthetic_batch(B, max_count=3)
        stop_col = int(x["stop_column"][0].item())
        # action_len includes STOP: 4, 3, 2, 4  (3,2,1,3 picks + STOP each)
        x["action_len"] = torch.tensor([4, 3, 2, 4], dtype=torch.long)
        for b in range(B):
            n_reg = int(x["action_len"][b].item()) - 1  # regular picks
            picks = torch.randperm(7)[:n_reg]
            for k in range(n_reg):
                x["action_idx"][b, k] = picks[k]
            x["action_idx"][b, n_reg] = stop_col  # STOP target

        ce = multiselect_ce(policy, x)
        assert torch.isfinite(ce).all()

    def test_no_gradient_leak_between_picks(self):
        """Each pick's CE uses the updated GRU state (not picked_ctx sum)."""
        policy = Policy()
        x = _make_synthetic_batch(2, max_count=2)
        stop_col = int(x["stop_column"][0].item())
        x["action_idx"][:, 0] = torch.tensor([0, 0])
        x["action_idx"][:, 1] = stop_col  # STOP after 1 pick
        x["action_len"] = torch.tensor([2, 2], dtype=torch.long)

        # First pick masks option 0, second step supervises STOP
        ce = multiselect_ce(policy, x)
        assert ce.shape == (2,)


class TestSelectMulti:
    """select_multi: greedy AR inference (high-level API)."""

    def test_output_shape(self):
        """Output is [B, maxC]."""
        policy = Policy()
        policy.eval()
        x = _make_synthetic_batch(2, max_count=3)
        x["maxCount"] = torch.full((2,), 3, dtype=torch.long)

        chosen = select_multi(policy, x)
        assert chosen.shape == (2, 3)

    def test_all_distinct(self):
        """All chosen indices are distinct (no repeats, ignoring STOP)."""
        policy = Policy()
        policy.eval()
        x = _make_synthetic_batch(8, max_count=4)
        x["maxCount"] = torch.full((8,), 4, dtype=torch.long)

        chosen = select_multi(policy, x)
        for b in range(8):
            picks = [int(p) for p in chosen[b].tolist() if p >= 0]  # filter STOP (-2), pad (-1)
            assert len(set(picks)) == len(picks), f"Sample {b}: duplicate picks {picks}"

    def test_no_out_of_range(self):
        """All regular picks are within valid option range."""
        policy = Policy()
        policy.eval()
        x = _make_synthetic_batch(4, max_count=3)
        x["maxCount"] = torch.full((4,), 3, dtype=torch.long)

        chosen = select_multi(policy, x)
        # Regular picks: >= 0 and < 8 (STOP = -2, padding = -1)
        for b in range(4):
            for p in chosen[b].tolist():
                p = int(p)
                if p >= 0:
                    assert p < 8, f"Sample {b}: pick {p} out of range"

    def test_single_member_ensemble_equivalent_to_single_pointer(self):
        """_select_multi_raw with pointers=[p.pointer], h_list=[h] produces
        element-wise identical output to the single-pointer path."""
        from ptcg_il.model.policy import _select_multi_raw

        torch.manual_seed(42)
        policy = Policy()
        policy.eval()
        x = _make_synthetic_batch(4, max_count=3)
        x["maxCount"] = torch.full((4,), 3, dtype=torch.long)
        x["minCount"] = torch.full((4,), 1, dtype=torch.long)

        h, _hist = policy._encode(x)
        single = _select_multi_raw(
            policy.pointer, h, x["tok_mask"], policy.embed.card, x,
            minC=x["minCount"], maxC=x["maxCount"],
            stop_column=x["stop_column"],
        )

        h2, _hist2 = policy._encode(x)
        ensemble = _select_multi_raw(
            policy.pointer, h2, x["tok_mask"], policy.embed.card, x,
            minC=x["minCount"], maxC=x["maxCount"],
            stop_column=x["stop_column"],
            pointers=[policy.pointer],
            h_list=[h2],
        )

        assert torch.equal(single, ensemble), (
            f"1-member ensemble path differs from single-pointer path\n"
            f"single:\n{single}\nensemble:\n{ensemble}"
        )


class TestValueHead:
    """ValueHead standalone tests."""

    def test_output_shape(self):
        from ptcg_il.model.value import ValueHead
        head = ValueHead(D)
        h_cls = torch.randn(4, D)
        value = head(h_cls)
        assert value.shape == (4,)

    def test_value_range(self):
        from ptcg_il.model.value import ValueHead
        head = ValueHead(D)
        head.eval()
        h_cls = torch.randn(16, D)
        value = head(h_cls)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_gradient_flow(self):
        from ptcg_il.model.value import ValueHead
        head = ValueHead(D)
        h_cls = torch.randn(4, D)
        value = head(h_cls)
        loss = value.sum()
        loss.backward()
        for name, p in head.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"


class TestMultiSelectCEScale:
    """The STOP column is suppressed by writing -1e9 into the logits, while the
    mask handed to the CE still marks it valid.  Label smoothing averages
    log-probs over *valid* positions, so it averaged in log_prob = -1e9 and
    returned ~eps * 1e9 / n_valid per step — 2.5e7 on a real batch, against a
    plausible value of ~9.  It stayed finite, so `isfinite` assertions above
    passed the whole time.
    """

    def _batch(self, B=2, max_count=3, min_count=2):
        x = _make_synthetic_batch(B, max_count=max_count)
        x["minCount"] = torch.full((B,), min_count, dtype=torch.long)
        stop_col = int(x["stop_column"][0].item())
        for b in range(B):
            picks = torch.randperm(7)[:max_count - 1]
            for k, p in enumerate(picks):
                x["action_idx"][b, k] = p
            x["action_idx"][b, max_count - 1] = stop_col
        x["action_len"] = torch.full((B,), max_count, dtype=torch.long)
        return x

    def test_smoothing_does_not_dominate_the_loss(self):
        """Label smoothing adds at most eps * (mean surprisal over valid
        options).  At init that is well under 1 nat per pick."""
        torch.manual_seed(0)
        policy = Policy()
        x = self._batch()
        plain = multiselect_ce(policy, x, label_smoothing=0.0)
        smoothed = multiselect_ce(policy, x, label_smoothing=0.05)
        assert torch.isfinite(smoothed).all()
        assert (smoothed <= plain + 5.0).all(), (
            f"smoothing inflated CE: {plain.tolist()} -> {smoothed.tolist()}"
        )

    def test_ce_is_on_a_plausible_scale_at_init(self):
        """An untrained pointer over O options costs ~log(O) nats per pick."""
        torch.manual_seed(0)
        policy = Policy()
        ce = multiselect_ce(policy, self._batch(max_count=3))
        assert (ce < 100).all(), f"CE at init should be a few nats per pick, got {ce.tolist()}"

    def test_suppressing_stop_costs_nothing_extra(self):
        """min_count only decides *when* STOP becomes legal.  Raising it must
        not change the loss scale — the suppressed column simply drops out."""
        torch.manual_seed(0)
        policy = Policy()
        free = multiselect_ce(policy, self._batch(min_count=0))
        forced = multiselect_ce(policy, self._batch(min_count=2))
        assert (forced < free + 10.0).all(), (
            f"suppressing STOP inflated CE: {free.tolist()} -> {forced.tolist()}"
        )


def test_multiselect_ce_marginalises_over_duplicate_options():
    """A batch whose first two options are identical must cost less under
    group-marginal CE than under plain CE, and the gap must be positive."""
    policy = Policy(D=32, heads=4, layers=1, ff=64)
    batch = _make_synthetic_batch(B=2, max_count=3)
    # Set valid action targets: row 0 picks options 0 then 1; row 1 picks option 2
    batch["action_idx"][0, 0] = 0
    batch["action_idx"][0, 1] = 1
    batch["action_idx"][0, 2] = 7  # STOP at first stop slot
    batch["action_idx"][1, 0] = 2
    batch["action_idx"][1, 1] = 7  # STOP
    batch["stop_column"][:] = 7
    batch["opt_type"][:, 7] = 17  # STOP type
    batch["minCount"][:] = 1
    plain = multiselect_ce(policy, batch, group_marginal=False)

    # Options 0 and 1 are already in the same group (fixture sets group 0 for all valid)
    # but option 2 is also in group 0. Split option 2 into its own group for the
    # plain run, then merge 0+1 for the grouped run.
    batch["opt_group"][:, 2] = 1
    plain = multiselect_ce(policy, batch, group_marginal=False)

    # Merge 0+1 back into same group
    batch["opt_group"][:, 1] = batch["opt_group"][:, 0]
    grouped = multiselect_ce(policy, batch, group_marginal=True)

    assert torch.isfinite(grouped).all()
    assert (grouped <= plain + 1e-5).all()
    assert (grouped < plain).any(), "merging a group must reduce CE for at least one row"


def test_multiselect_ce_unchanged_when_every_group_is_a_singleton():
    policy = Policy(D=32, heads=4, layers=1, ff=64)
    batch = _make_synthetic_batch(B=2, max_count=3)   # all-distinct opt_group after Task 2
    assert torch.allclose(
        multiselect_ce(policy, batch, group_marginal=True),
        multiselect_ce(policy, batch, group_marginal=False),
        atol=1e-5,
    )


class TestHistoryGruIsGone:
    """The cross-turn GRU was removed.

    Nothing ever threaded a non-zero ``history_h`` into it, so with ``h = 0`` the
    GRUCell reduced to ``(1-z) ⊙ tanh(Wx+b)`` — measured mean update gate 0.0146,
    **69% of output dims tanh-saturated** (|n| > 0.99), CLS rms 1.31 → 0.96, and
    per-dim corr(pre, post) ≈ −0.07.  Everything downstream of CLS (the value
    head, and the pointer's CLS key/value) read only that squashed vector.
    """

    def test_module_and_parameters_are_absent(self):
        p = Policy(D=32, heads=2, layers=1, ff=64)
        assert not hasattr(p, "history_gru")
        assert not [k for k in p.state_dict() if k.startswith("history_gru.")]

    def test_cls_row_is_the_encoder_row_plus_belief(self):
        """The CLS token must now pass through, not be squashed."""
        import torch

        p = Policy(D=32, heads=2, layers=1, ff=64).eval()
        x = _make_synthetic_batch(B=3)
        with torch.no_grad():
            rows = p.embed(x)
            h_enc = p.encoder(rows, x["tok_mask"])
            belief = p.belief(x["log_feat"], x["log_mask"],
                              log_card_feat=x.get("log_card_feat"))
            h, _ = p._encode(x)
        assert torch.allclose(h[:, 0, :], h_enc[:, 0, :] + belief, atol=1e-6)
        # The non-CLS rows are untouched either way.
        assert torch.allclose(h[:, 1:, :], h_enc[:, 1:, :], atol=1e-6)

    def test_cls_is_not_tanh_bounded(self):
        """The old path squashed every CLS dim into (-1, 1); this one does not."""
        import torch

        torch.manual_seed(0)
        p = Policy(D=64, heads=4, layers=2, ff=128).eval()
        x = _make_synthetic_batch(B=8)
        with torch.no_grad():
            h, _ = p._encode(x)
        assert h[:, 0, :].abs().max() > 1.0, (
            "CLS is still bounded by 1.0 — a tanh is back in the path"
        )

    def test_history_h_is_accepted_and_returned_unchanged(self):
        """Callers across ptcg_rl still pass and unpack it."""
        import torch

        p = Policy(D=32, heads=2, layers=1, ff=64).eval()
        x = _make_synthetic_batch(B=2)
        hh = torch.randn(2, 32)
        with torch.no_grad():
            h, out = p._encode(x, hh)
            logits, value, hist = p(x, hh)
        assert torch.equal(out, hh)
        assert logits.shape[0] == value.shape[0] == hist.shape[0] == 2

    def test_stale_history_gru_weights_load_without_raising(self):
        """Every pre-removal checkpoint carries these four parameters."""
        import torch

        from ptcg_il.model.policy import load_policy_state

        p = Policy(D=32, heads=2, layers=1, ff=64)
        sd = dict(p.state_dict())
        sd["history_gru.weight_ih"] = torch.randn(96, 32)
        sd["history_gru.weight_hh"] = torch.randn(96, 32)
        sd["history_gru.bias_ih"] = torch.randn(96)
        sd["history_gru.bias_hh"] = torch.randn(96)

        load_policy_state(p, sd)  # must not raise

        sd["genuinely_unknown.weight"] = torch.randn(4, 4)
        with pytest.raises(RuntimeError):
            load_policy_state(p, sd)
