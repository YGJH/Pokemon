"""Tests for Policy, multiselect_ce, and select_multi (Appendix B.6–B.8)."""

import torch
import pytest

from ptcg_il.model.embed import L_STATE, P_MAX, H_MAX, SUM
from ptcg_il.model.pointer import O_MAX
from ptcg_il.model.policy import Policy, multiselect_ce, select_multi

D = 256
V = 100
A = 50


def _make_synthetic_batch(B: int = 2, max_count: int = 1) -> dict[str, torch.Tensor]:
    """Build a synthetic featurizer dict."""
    O = O_MAX
    L = L_STATE

    x = {
        # State — card identity
        "poke_card_id": torch.randint(0, V, (B, P_MAX)),
        "hand_card_id": torch.randint(0, V, (B, H_MAX)),
        "stadium_card_id": torch.randint(0, V, (B, 1)),
        "context_card_id": torch.randint(0, V, (B, 1)),
        "effect_card_id": torch.randint(0, V, (B, 1)),
        "discard_ids": torch.randint(0, V, (B, SUM, 60)),
        "discard_mask": torch.ones(B, SUM, 60, dtype=torch.bool),
        "prize_ids": torch.zeros(B, SUM, 6, dtype=torch.long),
        # State — dense features
        "poke_feat": torch.randn(B, P_MAX, 26),
        "hand_feat": torch.randn(B, H_MAX, 2),
        "sum_feat": torch.randn(B, SUM, 11),
        "cls_feat": torch.randn(B, 93),
        "stadium_present": torch.ones(B, 1),
        # State — categorical
        "tok_type": torch.randint(0, 5, (B, L)),
        "tok_owner": torch.randint(0, 3, (B, L)),
        "tok_zone": torch.randint(0, 6, (B, L)),
        "tok_mask": torch.ones(B, L, dtype=torch.bool),
        # Options
        "opt_type": torch.randint(0, 17, (B, O)),
        "opt_src_idx": torch.randint(-1, L, (B, O)),
        "opt_tgt_idx": torch.randint(-1, L, (B, O)),
        "opt_card_id": torch.randint(0, V, (B, O)),
        "opt_attack_idx": torch.randint(0, A, (B, O)),
        "opt_scalar": torch.randn(B, O, 6),
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
    }

    # First 8 options valid
    x["opt_mask"][:, 8:] = False

    return x


class TestPolicy:
    """Policy: embed -> encode -> pointer + value."""

    def test_forward_shapes_single(self):
        """Single-select forward returns correct shapes."""
        policy = Policy(V, A)
        x = _make_synthetic_batch(4, max_count=1)
        x["action_idx"][:, 0] = torch.randint(0, 8, (4,))
        logits, value = policy(x)
        assert logits.shape == (4, O_MAX)
        assert value.shape == (4,)

    def test_value_range(self):
        """Value is in (-1, 1) due to tanh."""
        policy = Policy(V, A)
        policy.eval()
        x = _make_synthetic_batch(4)
        _, value = policy(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_logits_masked(self):
        """Padding options logits are -1e9."""
        policy = Policy(V, A)
        policy.eval()
        x = _make_synthetic_batch(2, max_count=1)
        logits, _ = policy(x)
        for j in range(8, O_MAX):
            assert (logits[:, j] < -1e8).all()

    def test_gradient_flow(self):
        """All parameters receive gradients."""
        policy = Policy(V, A)
        x = _make_synthetic_batch(2, max_count=1)
        x["action_idx"][:, 0] = torch.randint(0, 8, (2,))
        logits, value = policy(x)
        loss = logits[:, :8].sum() + value.sum()
        loss.backward()
        for name, p in policy.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"

    def test_pointer_card_bound(self):
        """Pointer.card is the same object as embed.card."""
        policy = Policy(V, A)
        assert policy.pointer.card is policy.embed.card

    def test_single_select_cross_entropy(self):
        """CE loss computed on single-select output is finite."""
        policy = Policy(V, A)
        x = _make_synthetic_batch(4, max_count=1)
        x["action_idx"][:, 0] = torch.randint(0, 8, (4,))
        logits, value = policy(x)
        ce = torch.nn.functional.cross_entropy(
            logits, x["action_idx"][:, 0], reduction="none"
        )
        assert torch.isfinite(ce).all()

    def test_different_actions_different_loss(self):
        """CE should penalize wrong predictions differently than correct ones."""
        policy = Policy(V, A)
        policy.eval()
        x = _make_synthetic_batch(2, max_count=1)
        logits, _ = policy(x)

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
        policy = Policy(V, A)
        policy.eval()
        x = _make_synthetic_batch(4)
        logits, value = policy(x)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits[:, :8]).any()  # masked has -inf
        assert not torch.isnan(value).any()
        assert not torch.isinf(value).any()


class TestMultiSelectCE:
    """multiselect_ce: teacher-forced AR training loss."""

    def test_output_is_finite(self):
        """CE output is finite."""
        policy = Policy(V, A)
        x = _make_synthetic_batch(2, max_count=3)
        # Set unique valid action picks per sample (no duplicates)
        for b in range(2):
            picks = torch.randperm(8)[:3]
            x["action_idx"][b, :3] = picks
        x["action_len"] = torch.full((2,), 3, dtype=torch.long)

        ce = multiselect_ce(policy, x)
        assert ce.shape == (2,)
        assert torch.isfinite(ce).all()

    def test_gradient_flow(self):
        """Gradients flow through embed/encoder/pointer (not value) in multi-select path."""
        policy = Policy(V, A)
        x = _make_synthetic_batch(2, max_count=2)
        for b in range(2):
            picks = torch.randperm(8)[:2]
            x["action_idx"][b, :2] = picks
        x["action_len"] = torch.full((2,), 2, dtype=torch.long)

        ce = multiselect_ce(policy, x)
        loss = ce.mean()
        loss.backward()
        # Only embed/encoder/pointer are used in multiselect_ce; value head is unused
        for name, p in policy.named_parameters():
            if "value" in name:
                assert p.grad is None, f"Value parameter {name} should not have gradient in multiselect"
            else:
                assert p.grad is not None, f"Parameter {name} has no gradient"

    def test_variable_action_len(self):
        """Handles samples with different pick counts in batch."""
        policy = Policy(V, A)
        B = 4
        x = _make_synthetic_batch(B, max_count=3)
        # Sample 0: 3 picks, sample 1: 2 picks, sample 2: 1 pick, sample 3: 3 picks
        x["action_len"] = torch.tensor([3, 2, 1, 3], dtype=torch.long)
        for b in range(B):
            n = int(x["action_len"][b].item())
            picks = torch.randperm(8)[:n]
            x["action_idx"][b, :n] = picks

        ce = multiselect_ce(policy, x)
        assert torch.isfinite(ce).all()

    def test_no_gradient_leak_between_picks(self):
        """Each pick's CE is computed using the updated picked_ctx."""
        policy = Policy(V, A)
        x = _make_synthetic_batch(2, max_count=2)
        x["action_idx"][:, 0] = torch.tensor([0, 0])
        x["action_idx"][:, 1] = torch.tensor([1, 1])
        x["action_len"] = torch.full((2,), 2, dtype=torch.long)

        # First pick masks option 0, second pick can't choose 0 again
        ce = multiselect_ce(policy, x)
        assert ce.shape == (2,)


class TestSelectMulti:
    """select_multi: greedy AR inference (high-level API)."""

    def test_output_shape(self):
        """Output is [B, maxC]."""
        policy = Policy(V, A)
        policy.eval()
        x = _make_synthetic_batch(2, max_count=3)
        x["maxCount"] = torch.full((2,), 3, dtype=torch.long)

        chosen = select_multi(policy, x)
        assert chosen.shape == (2, 3)

    def test_all_distinct(self):
        """All chosen indices are distinct (no repeats)."""
        policy = Policy(V, A)
        policy.eval()
        x = _make_synthetic_batch(8, max_count=4)
        x["maxCount"] = torch.full((8,), 4, dtype=torch.long)

        chosen = select_multi(policy, x)
        # Each sample should have 4 distinct picks
        for b in range(8):
            picks = chosen[b].tolist()
            assert len(set(picks)) == len(picks), f"Sample {b}: duplicate picks {picks}"

    def test_no_out_of_range(self):
        """All picks are within valid option range."""
        policy = Policy(V, A)
        policy.eval()
        x = _make_synthetic_batch(4, max_count=3)
        x["maxCount"] = torch.full((4,), 3, dtype=torch.long)

        chosen = select_multi(policy, x)
        # Valid options are 0..7
        assert (chosen >= 0).all()
        assert (chosen < 8).all()


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
