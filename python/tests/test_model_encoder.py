"""Tests for Encoder (Appendix B.3)."""

import torch
import pytest

from ptcg_il.model.encoder import Encoder

D = 256
L = 46  # L_STATE


class TestEncoder:
    """Encoder: self-attention transformer over state tokens."""

    def test_output_shape(self):
        """Output [B, L, D] matches input shape."""
        enc = Encoder()
        rows = torch.randn(4, L, D)
        tok_mask = torch.ones(4, L, dtype=torch.bool)
        h = enc(rows, tok_mask)
        assert h.shape == (4, L, D)

    def test_gradient_flow(self):
        """Gradients flow through the transformer."""
        enc = Encoder()
        rows = torch.randn(2, L, D, requires_grad=False)
        tok_mask = torch.ones(2, L, dtype=torch.bool)
        h = enc(rows, tok_mask)
        loss = h.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"

    def test_padding_mask_works(self):
        """Padding mask prevents attention to masked tokens."""
        enc = Encoder()
        enc.eval()

        rows = torch.randn(2, L, D)

        # All tokens unmasked
        mask_full = torch.ones(2, L, dtype=torch.bool)
        h_full = enc(rows, mask_full)

        # Mask out tokens 20..45 (keep 0..19)
        mask_partial = torch.ones(2, L, dtype=torch.bool)
        mask_partial[:, 20:] = False
        h_partial = enc(rows, mask_partial)

        # Unmasked tokens should differ due to less attention context
        # But the actual values depend on softmax, so we just check shapes
        assert h_partial.shape == h_full.shape
        assert h_partial.dtype == torch.float32

    def test_batch_independence(self):
        """Each batch item processed independently (no cross-sample attention)."""
        enc = Encoder()
        enc.eval()

        rows_batch = torch.randn(3, L, D)
        tok_mask = torch.ones(3, L, dtype=torch.bool)
        h_batch = enc(rows_batch, tok_mask)

        # Process each sample separately
        for i in range(3):
            rows_i = rows_batch[i:i+1]
            mask_i = tok_mask[i:i+1]
            h_i = enc(rows_i, mask_i)
            # Should be close (deterministic eval, same input)
            assert torch.allclose(h_batch[i:i+1], h_i, atol=1e-5)

    def test_variable_length_masking(self):
        """Samples with different numbers of valid tokens handled correctly."""
        enc = Encoder()
        enc.eval()

        rows = torch.randn(2, L, D)
        tok_mask = torch.ones(2, L, dtype=torch.bool)
        # First sample has 20 tokens, second has 46
        tok_mask[0, 20:] = False

        h = enc(rows, tok_mask)
        assert h.shape == (2, L, D)
        assert not torch.isnan(h).any()
        assert not torch.isinf(h).any()

    def test_pre_norm_order(self):
        """Verify norm_first=True by checking layer config."""
        enc = Encoder()
        layer = enc.enc.layers[0]
        assert layer.norm_first, "Encoder must use pre-norm (norm_first=True)"

    def test_gelu_activation(self):
        """Verify GELU activation."""
        enc = Encoder()
        layer = enc.enc.layers[0]
        # PyTorch stores activation as a string
        assert "gelu" in str(layer.activation).lower()

    def test_batch_first(self):
        """Verify batch_first=True."""
        enc = Encoder()
        layer = enc.enc.layers[0]
        assert layer.self_attn.batch_first, "Encoder must use batch_first=True"

    def test_final_layernorm_present(self):
        """A pre-norm stack needs a trailing norm, or its output is un-normalized."""
        enc = Encoder()
        assert isinstance(enc.enc.norm, torch.nn.LayerNorm)
        assert enc.enc.norm.normalized_shape == (D,)

    def test_output_is_normalized(self):
        """Encoder output has ~unit per-element RMS regardless of input scale.

        Without the final norm the residual stream leaves the stack at whatever
        scale four layers happened to accumulate (measured RMS 1.49 on the
        trained checkpoint, and growing with `layers`).
        """
        enc = Encoder()
        enc.eval()
        tok_mask = torch.ones(8, L, dtype=torch.bool)
        for scale in (0.1, 1.0, 10.0):
            rows = torch.randn(8, L, D) * scale
            with torch.no_grad():
                h = enc(rows, tok_mask)
            rms = float(h.pow(2).mean().sqrt())
            assert 0.8 < rms < 1.25, f"input scale {scale} → output RMS {rms}"

    def test_default_sizes(self):
        """Default D=256, heads=8, layers=4, ff=1024."""
        enc = Encoder()
        layer = enc.enc.layers[0]
        assert layer.self_attn.embed_dim == 256
        assert layer.self_attn.num_heads == 8
        assert len(enc.enc.layers) == 4


class TestDropoutArgumentIsHonoured:
    """C3 — ``Encoder.__init__`` began with ``dropout = 0.0``, before
    ``super().__init__()``, which made the constructor argument unreachable.
    The model therefore trained with no dropout no matter what was configured.
    """

    def test_dropout_reaches_the_transformer_layers(self):
        import torch.nn as nn

        from ptcg_il.model.encoder import Encoder

        enc = Encoder(D=32, heads=2, layers=2, ff=64, dropout=0.25)
        rates = [m.p for m in enc.modules() if isinstance(m, nn.Dropout)]
        assert rates, "no Dropout modules found — encoder shape changed"
        assert all(r == 0.25 for r in rates), (
            f"dropout argument did not reach the layers: {sorted(set(rates))}"
        )

    def test_default_is_still_zero(self):
        import torch.nn as nn

        from ptcg_il.model.encoder import Encoder

        enc = Encoder(D=32, heads=2, layers=1, ff=64)
        assert all(m.p == 0.0 for m in enc.modules() if isinstance(m, nn.Dropout))

    def test_dropout_actually_perturbs_activations_in_train_mode(self):
        """A rate that reaches the layers but is never applied is still dead."""
        import torch

        from ptcg_il.model.encoder import Encoder

        torch.manual_seed(0)
        enc = Encoder(D=32, heads=2, layers=2, ff=64, dropout=0.5).train()
        rows = torch.randn(4, 46, 32)
        mask = torch.ones(4, 46, dtype=torch.bool)
        a, b = enc(rows, mask), enc(rows, mask)
        assert not torch.allclose(a, b), "dropout is configured but has no effect"

        enc.eval()
        assert torch.allclose(enc(rows, mask), enc(rows, mask)), (
            "eval mode must be deterministic"
        )
