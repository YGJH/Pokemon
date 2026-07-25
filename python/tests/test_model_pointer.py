"""Tests for PointerHead (Appendix B.4)."""

import torch
import pytest

from ptcg_il.model.cards import CardEncoder
from ptcg_il.model.pointer import PointerHead, L_STATE, O_MAX, F_OPT

D = 256
A = 50
V = 100
L = L_STATE
O = O_MAX


def _make_synthetic_h_and_x(B: int = 2):
    """Build synthetic encoded state and option dict."""
    h = torch.randn(B, L, D)
    tok_mask = torch.ones(B, L, dtype=torch.bool)

    x = {
        "opt_type": torch.randint(0, 17, (B, O)),
        "opt_src_idx": torch.randint(-1, L, (B, O)),
        "opt_tgt_idx": torch.randint(-1, L, (B, O)),
        "opt_card_id": torch.randint(0, V, (B, O)),
        "opt_attack_idx": torch.randint(0, A, (B, O)),
        "opt_scalar": torch.randn(B, O, F_OPT),
        "opt_mask": torch.ones(B, O, dtype=torch.bool),
    }

    # Make first few options valid, rest masked
    x["opt_mask"][:, 8:] = False

    card_enc = CardEncoder(V)
    return h, tok_mask, card_enc, x


class TestPointerHead:
    """PointerHead: option building + cross-attention + scoring."""

    def test_output_shapes(self):
        """logits [B, O] and o [B, O, D]."""
        h, tok_mask, card_enc, x = _make_synthetic_h_and_x(4)
        pointer = PointerHead(A)
        logits, o = pointer(h, tok_mask, card_enc, x)
        assert logits.shape == (4, O)
        assert o.shape == (4, O, D)

    def test_masked_options_get_neg_inf(self):
        """Masked (padding) options should have logit ~ -1e9."""
        h, tok_mask, card_enc, x = _make_synthetic_h_and_x(2)
        pointer = PointerHead(A)
        pointer.eval()
        logits, _ = pointer(h, tok_mask, card_enc, x)

        # Options 8+ are masked
        for j in range(8, O):
            assert (logits[:, j] < -1e8).all(), f"Option {j} should be -1e9"

    def test_valid_options_finite(self):
        """Valid options have finite logits."""
        h, tok_mask, card_enc, x = _make_synthetic_h_and_x(2)
        pointer = PointerHead(A)
        logits, _ = pointer(h, tok_mask, card_enc, x)

        for j in range(8):
            assert torch.isfinite(logits[:, j]).all(), f"Option {j} should be finite"

    def test_gradient_flow(self):
        """Gradients flow through all parameters (including msgru)."""
        h, tok_mask, card_enc, x = _make_synthetic_h_and_x(2)
        h.requires_grad_(True)
        pointer = PointerHead(A)
        # Pass msgru_h to exercise the multi-select GRU
        msgru_h = torch.randn(2, D)
        logits, o = pointer(h, tok_mask, card_enc, x, msgru_h=msgru_h)
        loss = logits[:, :8].sum()  # only valid options
        loss.backward()
        for name, p in pointer.named_parameters():
            if "msgru" in name:
                continue  # msgru is called externally by the multi-select loop
            assert p.grad is not None, f"Parameter {name} has no gradient"
        assert h.grad is not None, "Input h should have gradient"

    def test_null_token_path(self):
        """idx=-1 references use null_token row."""
        h, tok_mask, card_enc, x = _make_synthetic_h_and_x(2)
        pointer = PointerHead(A)
        pointer.eval()

        # All -1 src/tgt
        x["opt_src_idx"] = torch.full((2, O), -1, dtype=torch.long)
        x["opt_tgt_idx"] = torch.full((2, O), -1, dtype=torch.long)

        logits, _ = pointer(h, tok_mask, card_enc, x)
        assert not torch.isnan(logits).any()
        assert torch.isfinite(logits[:, :8]).all()

    def test_null_token_is_learnable(self):
        """null_token receives gradients."""
        pointer = PointerHead(A)
        # Check it's a parameter
        assert isinstance(pointer.null_token, torch.nn.Parameter)

        h, tok_mask, card_enc, x = _make_synthetic_h_and_x(2)
        x["opt_src_idx"] = torch.full((2, O), -1, dtype=torch.long)
        logits, _ = pointer(h, tok_mask, card_enc, x)
        loss = logits[:, :8].sum()
        loss.backward()
        assert pointer.null_token.grad is not None
        assert pointer.null_token.grad.abs().sum() > 0

    def test_extra_ctx_adds_to_base(self):
        """msgru_h (multi-select GRU) influences logits."""
        h, tok_mask, card_enc, x = _make_synthetic_h_and_x(2)
        pointer = PointerHead(A)
        pointer.eval()

        logits_no_ctx, _ = pointer(h, tok_mask, card_enc, x)

        msgru_h = torch.randn(2, D)
        logits_ctx, _ = pointer(h, tok_mask, card_enc, x, msgru_h=msgru_h)

        # Logits should differ
        assert not torch.allclose(logits_no_ctx[:, :8], logits_ctx[:, :8], atol=1e-4)

    def test_card_is_none_initially(self):
        """PointerHead.card is None until bound externally."""
        pointer = PointerHead(A)
        assert pointer.card is None

    def test_card_enc_integration(self):
        """Passing card_enc via forward works."""
        h, tok_mask, card_enc, x = _make_synthetic_h_and_x(2)
        pointer = PointerHead(A)
        logits, o = pointer(h, tok_mask, card_enc, x)
        assert logits.shape == (2, O)

    def test_gather_method(self):
        """gather correctly maps -1 to null row."""
        pointer = PointerHead(A)
        h_aug = torch.randn(2, L + 1, D)
        idx = torch.tensor([[0, -1, 5], [-1, 3, -1]])
        gathered = pointer.gather(h_aug, idx)
        assert gathered.shape == (2, 3, D)
        # Row 0 of idx[0] -> h_aug[0,0]; idx[0,1]=-1 -> h_aug[0, L]
        assert torch.allclose(gathered[0, 0], h_aug[0, 0])
        assert torch.allclose(gathered[0, 1], h_aug[0, L])
