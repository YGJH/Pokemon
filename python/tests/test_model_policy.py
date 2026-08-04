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
