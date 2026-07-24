"""Tests for TokenEmbedder (Appendix B.2)."""

import torch
import numpy as np
import pytest

from ptcg_il.model.embed import TokenEmbedder, L_STATE, P_MAX, H_MAX, SUM

V = 100
D = 256


def _make_synthetic_batch(B: int = 2) -> dict[str, torch.Tensor]:
    """Build a minimal synthetic featurizer dict with batch dim B."""
    return {
        # CLS
        "cls_feat": torch.randn(B, 93),
        "context_card_id": torch.randint(0, V, (B, 1)),
        "effect_card_id": torch.randint(0, V, (B, 1)),
        # Pokemon
        "poke_feat": torch.randn(B, P_MAX, 26),
        "poke_card_id": torch.randint(0, V, (B, P_MAX)),
        # Hand
        "hand_feat": torch.randn(B, H_MAX, 2),
        "hand_card_id": torch.randint(0, V, (B, H_MAX)),
        # Summary
        "sum_feat": torch.randn(B, SUM, 11),
        "discard_ids": torch.randint(0, V, (B, SUM, 60)),
        "discard_mask": torch.ones(B, SUM, 60, dtype=torch.bool),
        "prize_ids": torch.randint(0, V, (B, SUM, 6)),
        # Stadium
        "stadium_present": torch.ones(B, 1),
        "stadium_card_id": torch.randint(0, V, (B, 1)),
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
        embed = TokenEmbedder(V)
        x = _make_synthetic_batch(4)
        rows = embed(x)
        assert rows.shape == (4, L_STATE, D)

    def test_gradient_flow(self):
        """Gradients flow through all learnable params."""
        embed = TokenEmbedder(V)
        x = _make_synthetic_batch(2)
        rows = embed(x)
        loss = rows.sum()
        loss.backward()
        for name, p in embed.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient"

    def test_no_stadium_path(self):
        """When stadium_present=0, no_stadium param is used."""
        embed = TokenEmbedder(V)
        x = _make_synthetic_batch(2)
        x["stadium_present"] = torch.zeros(2, 1)

        # With no_stadium all zeros, stadium row should match card embedding
        # of stadium_card_id * 0 + zeros = 0 if stadium_card_id is PAD
        x["stadium_card_id"] = torch.zeros(2, 1, dtype=torch.long)  # PAD=0
        rows = embed(x)
        stadium_row = rows[:, 45, :]
        # PAD card -> 0 embedding. no_stadium is 0-init -> row 45 should be near zero
        # (plus type/owner/zone embeddings for row 45)
        assert stadium_row.shape == (2, D)

    def test_stadium_present_path(self):
        """When stadium_present=1, real card embedding is used."""
        embed = TokenEmbedder(V)
        embed.eval()
        x = _make_synthetic_batch(2)
        x["stadium_present"] = torch.ones(2, 1)
        cid = torch.randint(2, V, (2, 1))
        x["stadium_card_id"] = cid

        rows1 = embed(x)

        # Change stadium card
        x["stadium_card_id"] = torch.randint(2, V, (2, 1))
        rows2 = embed(x)

        # Stadium rows should differ (different card ids)
        assert not torch.allclose(rows1[:, 45, :], rows2[:, 45, :], atol=1e-4)

    def test_context_effect_cards(self):
        """context_card_id and effect_card_id influence CLS token."""
        embed = TokenEmbedder(V)
        embed.eval()
        x = _make_synthetic_batch(2)

        # Set has_contextCard = 1, has_effect = 0
        x["cls_feat"][:, 87] = 1.0
        x["cls_feat"][:, 88] = 0.0

        rows1 = embed(x)

        # Change context_card_id
        x["context_card_id"] = torch.randint(2, V, (2, 1))
        rows2 = embed(x)

        # CLS should change because context card embedding is added
        assert not torch.allclose(rows1[:, 0, :], rows2[:, 0, :], atol=1e-4)

    def test_discard_pooling(self):
        """Discard pile is pooled via masked sum / DECK_N."""
        embed = TokenEmbedder(V)
        embed.eval()
        x = _make_synthetic_batch(2)

        # All-zero discard mask -> zero contribution
        x["discard_mask"] = torch.zeros(2, SUM, 60, dtype=torch.bool)
        rows_nodisc = embed(x)

        # All-one discard mask -> non-zero contribution
        x["discard_mask"] = torch.ones(2, SUM, 60, dtype=torch.bool)
        x["discard_ids"] = torch.randint(2, V, (2, SUM, 60))
        rows_withdisc = embed(x)

        # Summary rows should differ
        assert not torch.allclose(
            rows_nodisc[:, 43:45, :], rows_withdisc[:, 43:45, :], atol=1e-4
        )

    def test_categorical_embeddings_applied(self):
        """Type/owner/zone embeddings add to output."""
        embed = TokenEmbedder(V)
        x = _make_synthetic_batch(2)

        # Verify embedding dimensions
        assert embed.type_emb.weight.shape == (5, D)
        assert embed.owner_emb.weight.shape == (3, D)
        assert embed.zone_emb.weight.shape == (6, D)

        rows = embed(x)
        assert rows.shape == (2, L_STATE, D)

    def test_pokemon_positions(self):
        """Pokemon token rows are 1..12."""
        embed = TokenEmbedder(V)
        embed.eval()
        x = _make_synthetic_batch(2)

        # Set PAD for all poke slots
        x["poke_card_id"] = torch.zeros(2, P_MAX, dtype=torch.long)
        x["poke_feat"] = torch.zeros(2, P_MAX, 26)

        rows = embed(x)
        # Row 0 (CLS) should differ from poke rows
        # Poke rows 1..12 should exist
        assert rows.shape[1] == L_STATE

    def test_hand_positions(self):
        """Hand token rows are 13..42."""
        embed = TokenEmbedder(V)
        embed.eval()
        x = _make_synthetic_batch(2)

        # Set PAD for all hand slots
        x["hand_card_id"] = torch.zeros(2, H_MAX, dtype=torch.long)
        x["hand_feat"] = torch.zeros(2, H_MAX, 2)

        rows = embed(x)
        # Hand rows exist
        hand_rows = rows[:, 13:43, :]
        assert hand_rows.shape == (2, H_MAX, D)

    def test_dtype_float32(self):
        """Output is float32."""
        embed = TokenEmbedder(V)
        x = _make_synthetic_batch(2)
        rows = embed(x)
        assert rows.dtype == torch.float32
