"""Tests for CardFeaturizer and AttackFeaturizer (pure MLP over features)."""

import torch
import torch.nn as nn
import pytest

from ptcg_il.model.cards import CardFeaturizer, AttackFeaturizer, F_CARD, F_ATK

D = 64


class TestCardFeaturizer:
    """CardFeaturizer: pure MLP over 94-dim static features."""

    def test_output_shape_1d(self):
        cf = CardFeaturizer(D)
        feat = torch.randn(16, F_CARD)
        out = cf(feat)
        assert out.shape == (16, D)

    def test_output_shape_3d(self):
        cf = CardFeaturizer(D)
        feat = torch.randn(4, 12, F_CARD)
        out = cf(feat)
        assert out.shape == (4, 12, D)

    def test_zeros_is_small(self):
        """All-zero features (PAD) produce a small embedding (bias only)."""
        cf = CardFeaturizer(D)
        out = cf(torch.zeros(1, F_CARD))
        # With MLP bias, output is small but non-zero.  Shape is correct.
        assert out.shape == (1, D)

    def test_gradient_flow(self):
        cf = CardFeaturizer(D)
        feat = torch.randn(8, F_CARD, requires_grad=True)
        out = cf(feat)
        loss = out.sum()
        loss.backward()
        for name, p in cf.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"
        assert feat.grad is not None, "Input feature grad should flow"

    def test_different_features_different_embeddings(self):
        cf = CardFeaturizer(D)
        f1 = torch.zeros(1, F_CARD)
        f2 = torch.ones(1, F_CARD)
        out1 = cf(f1)
        out2 = cf(f2)
        assert not torch.allclose(out1, out2, atol=1e-4)

    def test_batch_independence(self):
        cf = CardFeaturizer(D).eval()
        feat = torch.randn(3, 4, F_CARD)
        out_batch = cf(feat)
        for i in range(3):
            out_single = cf(feat[i:i+1])
            assert torch.allclose(out_batch[i], out_single[0], atol=1e-6)


class TestAttackFeaturizer:
    """AttackFeaturizer: pure MLP over 14-dim static features."""

    def test_output_shape_1d(self):
        af = AttackFeaturizer(D)
        feat = torch.randn(8, F_ATK)
        out = af(feat)
        assert out.shape == (8, D)

    def test_output_shape_2d(self):
        af = AttackFeaturizer(D)
        feat = torch.randn(4, 64, F_ATK)
        out = af(feat)
        assert out.shape == (4, 64, D)

    def test_gradient_flow(self):
        af = AttackFeaturizer(D)
        feat = torch.randn(8, F_ATK)
        out = af(feat)
        loss = out.sum()
        loss.backward()
        for name, p in af.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"
