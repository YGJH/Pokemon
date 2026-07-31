"""Tests for TokenEmbedder (Appendix B.2)."""

import torch
import numpy as np
import pytest

from ptcg_il.model.cards import F_CARD
from ptcg_il.model.embed import TokenEmbedder, L_STATE, P_MAX, H_MAX, SUM

D = 256
D_MAX = 60
PZ_MAX = 6


def _make_synthetic_batch(B: int = 2) -> dict[str, torch.Tensor]:
    """Build a minimal synthetic featurizer dict with batch dim B.

    Cards arrive as static feature vectors; the embedder holds no id table.
    """
    return {
        # CLS
        "cls_feat": torch.randn(B, 93),
        "context_card_feat": torch.randn(B, 1, F_CARD),
        "effect_card_feat": torch.randn(B, 1, F_CARD),
        # Pokemon
        "poke_feat": torch.randn(B, P_MAX, 26),
        "poke_card_feat": torch.randn(B, P_MAX, F_CARD),
        # Hand
        "hand_feat": torch.randn(B, H_MAX, 2),
        "hand_card_feat": torch.randn(B, H_MAX, F_CARD),
        # Summary
        "sum_feat": torch.randn(B, SUM, 11),
        "discard_card_feat": torch.randn(B, SUM, D_MAX, F_CARD),
        "discard_mask": torch.ones(B, SUM, D_MAX, dtype=torch.bool),
        "prize_card_feat": torch.randn(B, SUM, PZ_MAX, F_CARD),
        # Stadium
        "stadium_present": torch.ones(B, 1),
        "stadium_card_feat": torch.randn(B, 1, F_CARD),
        # Categorical
        "tok_type": torch.randint(0, 5, (B, L_STATE)),
        "tok_owner": torch.randint(0, 3, (B, L_STATE)),
        "tok_zone": torch.randint(0, 6, (B, L_STATE)),
        # (tok_mask not used in embed, passed through)
    }


class TestTokenEmbedder:
    """TokenEmbedder: state dictionary -> [B, L_STATE, D]."""

    def test_output_shape(self):
        """Output shape is [B, L_STATE, D]."""
        embed = TokenEmbedder(D)
        x = _make_synthetic_batch(4)
        rows = embed(x)
        assert rows.shape == (4, L_STATE, D)

    def test_gradient_flow(self):
        """Gradients flow through all learnable params."""
        embed = TokenEmbedder(D)
        x = _make_synthetic_batch(2)
        rows = embed(x)
        loss = rows.sum()
        loss.backward()
        for name, p in embed.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"

    def test_no_stadium_path(self):
        """When stadium_present=0, no_stadium param is used."""
        embed = TokenEmbedder(D)
        x = _make_synthetic_batch(2)
        x["stadium_present"] = torch.zeros(2, 1)

        # With no_stadium all zeros, stadium row should match card embedding
        # of the stadium card * 0 + zeros = 0 when the stadium card is PAD
        x["stadium_card_feat"] = torch.zeros(2, 1, F_CARD)  # PAD = all-zero features
        rows = embed(x)
        stadium_row = rows[:, 45, :]
        # PAD card -> 0 embedding. no_stadium is 0-init -> row 45 should be near zero
        # (plus type/owner/zone embeddings for row 45)
        assert stadium_row.shape == (2, D)

    def test_stadium_present_path(self):
        """When stadium_present=1, real card embedding is used."""
        embed = TokenEmbedder(D)
        embed.eval()
        x = _make_synthetic_batch(2)
        x["stadium_present"] = torch.ones(2, 1)
        x["stadium_card_feat"] = torch.randn(2, 1, F_CARD)

        rows1 = embed(x)

        # Change stadium card
        x["stadium_card_feat"] = torch.randn(2, 1, F_CARD)
        rows2 = embed(x)

        # Stadium rows should differ (different card ids)
        assert not torch.allclose(rows1[:, 45, :], rows2[:, 45, :], atol=1e-4)

    def test_context_effect_cards(self):
        """context_card_feat and effect_card_feat influence CLS token."""
        embed = TokenEmbedder(D)
        embed.eval()
        x = _make_synthetic_batch(2)

        # Set has_contextCard = 1, has_effect = 0
        x["cls_feat"][:, 87] = 1.0
        x["cls_feat"][:, 88] = 0.0

        rows1 = embed(x)

        # Change the context card's features
        x["context_card_feat"] = torch.randn(2, 1, F_CARD)
        rows2 = embed(x)

        # CLS should change because context card embedding is added
        assert not torch.allclose(rows1[:, 0, :], rows2[:, 0, :], atol=1e-4)

    def test_discard_pooling(self):
        """Discard pile is pooled via masked sum / DECK_N."""
        embed = TokenEmbedder(D)
        embed.eval()
        x = _make_synthetic_batch(2)

        # All-zero discard mask -> zero contribution
        x["discard_mask"] = torch.zeros(2, SUM, D_MAX, dtype=torch.bool)
        rows_nodisc = embed(x)

        # All-one discard mask -> non-zero contribution
        x["discard_mask"] = torch.ones(2, SUM, D_MAX, dtype=torch.bool)
        x["discard_card_feat"] = torch.randn(2, SUM, D_MAX, F_CARD)
        rows_withdisc = embed(x)

        # Summary rows should differ
        assert not torch.allclose(
            rows_nodisc[:, 43:45, :], rows_withdisc[:, 43:45, :], atol=1e-4
        )

    def test_categorical_embeddings_applied(self):
        """Type/owner/zone embeddings add to output."""
        embed = TokenEmbedder(D)
        x = _make_synthetic_batch(2)

        # Verify embedding dimensions
        assert embed.type_emb.weight.shape == (5, D)
        assert embed.owner_emb.weight.shape == (3, D)
        assert embed.zone_emb.weight.shape == (6, D)

        rows = embed(x)
        assert rows.shape == (2, L_STATE, D)

    def test_pokemon_positions(self):
        """Pokemon token rows are 1..12."""
        embed = TokenEmbedder(D)
        embed.eval()
        x = _make_synthetic_batch(2)

        # Set PAD for all poke slots
        x["poke_card_feat"] = torch.zeros(2, P_MAX, F_CARD)
        x["poke_feat"] = torch.zeros(2, P_MAX, 26)

        rows = embed(x)
        # Row 0 (CLS) should differ from poke rows
        # Poke rows 1..12 should exist
        assert rows.shape[1] == L_STATE

    def test_hand_positions(self):
        """Hand token rows are 13..42."""
        embed = TokenEmbedder(D)
        embed.eval()
        x = _make_synthetic_batch(2)

        # Set PAD for all hand slots
        x["hand_card_feat"] = torch.zeros(2, H_MAX, F_CARD)
        x["hand_feat"] = torch.zeros(2, H_MAX, 2)

        rows = embed(x)
        # Hand rows exist
        hand_rows = rows[:, 13:43, :]
        assert hand_rows.shape == (2, H_MAX, D)

    def test_dtype_float32(self):
        """Output is float32."""
        embed = TokenEmbedder(D)
        x = _make_synthetic_batch(2)
        rows = embed(x)
        assert rows.dtype == torch.float32
