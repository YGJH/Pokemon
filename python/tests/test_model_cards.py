"""Tests for CardEncoder and AttackEncoder (Appendix B.1)."""

import torch
import pytest

from ptcg_il.model.cards import CardEncoder, AttackEncoder, F_CARD, F_ATK


V = 100
A = 50
D = 256


class TestCardEncoder:
    """CardEncoder: id embedding + static-feature MLP."""

    def test_output_shape_1d(self):
        """1-d input -> [N, D]."""
        enc = CardEncoder(V)
        ids = torch.randint(2, V, (16,))
        out = enc(ids)
        assert out.shape == (16, D)

    def test_output_shape_2d(self):
        """2-d input -> [B, N, D]."""
        enc = CardEncoder(V)
        ids = torch.randint(2, V, (4, 12))
        out = enc(ids)
        assert out.shape == (4, 12, D)

    def test_padding_idx_in_embedding(self):
        """PAD=0 -> id_emb[0] is zero; static_mlp may add bias."""
        enc = CardEncoder(V)
        # The id_embedding for PAD should be all zeros
        id_part = enc.id_emb(torch.tensor([0]))
        assert torch.allclose(id_part, torch.zeros_like(id_part), atol=1e-7)
        # Full output has static_mlp contribution (with bias), shape correct
        out = enc(torch.tensor([0]))
        assert out.shape == (1, D)

    def test_gradient_flow(self):
        """Gradients flow through both id_emb and static_mlp."""
        enc = CardEncoder(V)
        ids = torch.randint(2, V, (8,))
        out = enc(ids)
        loss = out.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            if p.grad is None:
                raise AssertionError(f"Parameter {name} has no gradient")

    def test_different_ids_different_embeddings(self):
        """Different ids produce different vectors."""
        enc = CardEncoder(V)
        ids1 = torch.tensor([2, 3, 4])
        ids2 = torch.tensor([5, 6, 7])
        out1 = enc(ids1)
        out2 = enc(ids2)
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_static_table_integration(self):
        """Custom static table is used correctly."""
        static = torch.randn(V, F_CARD)
        enc = CardEncoder(V, card_static_table=static)
        ids = torch.tensor([10, 20])
        out = enc(ids)
        # Output should depend on static table
        assert out.shape == (2, D)
        assert out.requires_grad  # MLP params have grad

    def test_unknown_index_uses_static(self):
        """UNKNOWN=1 produces non-zero embedding (from static features)."""
        enc = CardEncoder(V)
        ids = torch.tensor([1])  # UNKNOWN
        out = enc(ids)
        # id_emb[1] is not zero (no padding_idx for UNKNOWN), but untrained
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-7)

    def test_batch_independence(self):
        """Each sample is independent."""
        enc = CardEncoder(V).eval()
        ids_batch = torch.randint(2, V, (3, 4))
        out_batch = enc(ids_batch)
        for i in range(3):
            out_single = enc(ids_batch[i:i+1])
            assert torch.allclose(out_batch[i], out_single[0], atol=1e-6)


class TestAttackEncoder:
    """AttackEncoder: attack id embedding + static-feature MLP."""

    def test_output_shape_1d(self):
        """1-d input -> [N, D]."""
        enc = AttackEncoder(A)
        idx = torch.randint(1, A, (8,))
        out = enc(idx)
        assert out.shape == (8, D)

    def test_output_shape_2d(self):
        """2-d input -> [B, N, D]."""
        enc = AttackEncoder(A)
        idx = torch.randint(1, A, (4, 64))
        out = enc(idx)
        assert out.shape == (4, 64, D)

    def test_padding_idx_in_embedding(self):
        """PAD_ATTACK=0 -> id_emb[0] is zero; static_mlp may add bias."""
        enc = AttackEncoder(A)
        id_part = enc.id_emb(torch.tensor([0]))
        assert torch.allclose(id_part, torch.zeros_like(id_part), atol=1e-7)
        out = enc(torch.tensor([0]))
        assert out.shape == (1, D)
    def test_gradient_flow(self):
        """Gradients flow."""
        enc = AttackEncoder(A)
        idx = torch.randint(1, A, (8,))
        out = enc(idx)
        loss = out.sum()
        loss.backward()
        for name, p in enc.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"

    def test_static_table_integration(self):
        """Custom static table."""
        static = torch.randn(A, F_ATK)
        enc = AttackEncoder(A, attack_static_table=static)
        idx = torch.tensor([1, 2])
        out = enc(idx)
        assert out.shape == (2, D)
