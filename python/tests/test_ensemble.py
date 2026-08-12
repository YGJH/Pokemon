"""Tests for EnsemblePolicy — from_checkpoints, forward, select_multi."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ptcg_il.ensemble import EnsemblePolicy
from ptcg_il.model.policy import Policy, select_multi

# Reuse the synthetic batch builder from test_model_policy
from tests.test_model_policy import (
    _make_synthetic_batch, make_policy, make_static_tables,
)


def _from_checkpoints(paths, **kw):
    """``EnsemblePolicy.from_checkpoints`` with the fixtures' static tables.

    Members are saved by ``make_policy``, which attaches those tables, so the
    card matrix is in their state dicts; rebuilding without it makes
    ``belief_heads.all_card_feat`` an unexpected key.
    """
    card, atk = make_static_tables()
    kw.setdefault("all_card_feat", card)
    kw.setdefault("all_attack_feat", atk)
    return EnsemblePolicy.from_checkpoints(paths, **kw)



def _make_member(D=256, heads=8, layers=4, ff=1024, seed=0):
    """Build a Policy with a specific seed for deterministic comparison."""
    torch.manual_seed(seed)
    from ptcg_il.model import init_weights
    policy = make_policy(D=D, heads=heads, layers=layers, ff=ff)
    init_weights(policy)
    return policy


class TestEnsemblePolicyConstruction:
    """from_checkpoints and basic construction."""

    def test_from_two_different_members(self):
        """Two members with same arch but different seeds."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        assert len(ensemble.members) == 2
        # Different parameters due to different seeds — check a real weight,
        # not the zero-initialised bias that happens to be first in order.
        w0 = m0.embed.card.mlp[0].weight.clone()
        w1 = m1.embed.card.mlp[0].weight.clone()
        assert not torch.equal(w0, w1), "Different seeds produced identical weights"

    def test_from_two_same_seed_members_are_identical(self):
        """Two members with same seed -> identical weights (deterministic init)."""
        m0 = _make_member(seed=42)
        m1 = _make_member(seed=42)
        for name0, p0 in m0.named_parameters():
            p1 = dict(m1.named_parameters())[name0]
            assert torch.equal(p0, p1), (
                f"Same seed produced different params for {name0}"
            )

    def test_from_checkpoints_with_saved_files(self):
        """from_checkpoints loads two saved .pt files correctly."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)

        with tempfile.TemporaryDirectory() as tmp:
            path0 = Path(tmp) / "ckpt0.pt"
            path1 = Path(tmp) / "ckpt1.pt"
            deck = {
                "deck": [1] * 60,
                "vocab_sha1": "abc123",
                "archetypes_sha1": "def456",
            }
            torch.save({
                "model_state_dict": m0.state_dict(),
                "config": m0.config,
                "deck": deck,
            }, path0)
            torch.save({
                "model_state_dict": m1.state_dict(),
                "config": m1.config,
                "deck": deck,
            }, path1)

            ensemble = _from_checkpoints(
                [str(path0), str(path1)],
            )
            assert len(ensemble.members) == 2

    def test_from_checkpoints_prefers_ema_shadow(self):
        """When ema_state_dict.shadow exists, it is used over model_state_dict.

        The shadow is built the same way _EMA.__init__ builds it — via
        named_parameters() with remove_duplicate=True (the default), so tied
        parameter aliases (pointer.card, belief.card_emb) are absent.
        """
        m0 = _make_member(seed=0)
        # Build shadow matching _EMA.__init__: named_parameters() deduplicates
        # tied params (pointer.card = embed.card), so alias names are absent.
        ema_shadow = {}
        for n, p in m0.named_parameters():
            if p.requires_grad:
                ema_shadow[n] = p.data.clone().detach() + 0.1

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            deck = {"deck": [1] * 60, "vocab_sha1": "abc", "archetypes_sha1": "def"}
            torch.save({
                "model_state_dict": m0.state_dict(),
                "ema_state_dict": {"shadow": ema_shadow, "decay": 0.999},
                "config": m0.config,
                "deck": deck,
            }, path)

            ensemble = _from_checkpoints([str(path)])
            # Check a real weight (not the zero-initialised bias first in order)
            loaded = ensemble.members[0].embed.card.mlp[0].weight
            original = m0.embed.card.mlp[0].weight
            # EMA shadow was offset by 0.1, so loaded params should differ
            assert not torch.allclose(loaded, original, atol=0.05), (
                "EMA shadow should have been preferred"
            )

    def test_deck_mismatch_raises(self):
        """Members with different decklists must raise."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)

        with tempfile.TemporaryDirectory() as tmp:
            path0 = Path(tmp) / "ckpt0.pt"
            path1 = Path(tmp) / "ckpt1.pt"
            torch.save({
                "model_state_dict": m0.state_dict(),
                "config": m0.config,
                "deck": {"deck": [1] * 60, "vocab_sha1": "abc", "archetypes_sha1": "def"},
            }, path0)
            torch.save({
                "model_state_dict": m1.state_dict(),
                "config": m1.config,
                "deck": {"deck": [2] * 60, "vocab_sha1": "abc", "archetypes_sha1": "def"},
            }, path1)

            with pytest.raises(ValueError, match="decklist differs"):
                _from_checkpoints([str(path0), str(path1)])

    def test_missing_both_state_dicts_raises(self):
        """Checkpoint with neither EMA shadow nor model_state_dict raises."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.pt"
            torch.save({"config": {"D": 256, "heads": 8, "layers": 4, "ff": 1024}}, path)
            with pytest.raises(KeyError, match="missing both"):
                _from_checkpoints([str(path)])

    def test_empty_paths_raises(self):
        """Zero paths raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            _from_checkpoints([])


class TestEnsemblePolicyForward:
    """Ensemble forward pass."""

    def test_forward_shapes(self):
        """Ensemble forward returns correct shapes."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(4, max_count=1)
        logits, value, hist = ensemble(x)
        from ptcg_il.model.pointer import O_MAX
        assert logits.shape == (4, O_MAX)
        assert value.shape == (4,)
        assert hist.shape == (4, m0.D)

    def test_forward_value_is_mean_of_members(self):
        """Value should be the mean of individual member values."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        m0.eval()
        m1.eval()
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(2, max_count=1)
        _, v0, _ = m0(x)
        _, v1, _ = m1(x)
        _, v_ens, _ = ensemble(x)

        expected = (v0 + v1) / 2.0
        assert torch.allclose(v_ens, expected, atol=1e-6), (
            f"Ensemble value {v_ens} != mean {expected}"
        )

    def test_forward_masked_padding(self):
        """Padding option logits are ~-1e9 (very negative)."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(2, max_count=1)
        logits, _, _ = ensemble(x)
        from ptcg_il.model.pointer import O_MAX
        for j in range(8, O_MAX):
            assert (logits[:, j] < -1e8).all(), f"Column {j} not masked"

    def test_single_member_ensemble_argmax_matches_policy(self):
        """A 1-member EnsemblePolicy argmax matches the Policy's argmax."""
        torch.manual_seed(42)
        m0 = _make_member(seed=42)
        m0.eval()
        ensemble = EnsemblePolicy(nn.ModuleList([m0]))
        ensemble.eval()

        x = _make_synthetic_batch(4, max_count=1)
        logits_p, _, _ = m0(x)
        logits_e, _, _ = ensemble(x)

        # Single member: softmax -> log should preserve argmax
        assert torch.equal(logits_p.argmax(-1), logits_e.argmax(-1)), (
            "Single-member ensemble argmax differs from policy"
        )


class TestEnsemblePolicySelectMulti:
    """Ensemble AR multi-select inference."""

    def test_output_shape(self):
        """Output is [B, maxC]."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(2, max_count=3)
        x["maxCount"] = torch.full((2,), 3, dtype=torch.long)
        x["minCount"] = torch.full((2,), 1, dtype=torch.long)

        chosen = select_multi(ensemble, x)
        assert chosen.shape == (2, 3)

    def test_all_distinct(self):
        """All chosen indices are distinct (no repeats, ignoring STOP)."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(8, max_count=4)
        x["maxCount"] = torch.full((8,), 4, dtype=torch.long)
        x["minCount"] = torch.full((8,), 1, dtype=torch.long)

        chosen = select_multi(ensemble, x)
        for b in range(8):
            picks = [int(p) for p in chosen[b].tolist() if p >= 0]
            assert len(set(picks)) == len(picks), f"Sample {b}: duplicate picks {picks}"

    def test_no_out_of_range(self):
        """All regular picks are within valid option range."""
        m0 = _make_member(seed=0)
        m1 = _make_member(seed=1)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1]))
        ensemble.eval()

        x = _make_synthetic_batch(4, max_count=3)
        x["maxCount"] = torch.full((4,), 3, dtype=torch.long)
        x["minCount"] = torch.full((4,), 1, dtype=torch.long)

        chosen = select_multi(ensemble, x)
        for b in range(4):
            for p in chosen[b].tolist():
                p = int(p)
                if p >= 0:
                    assert p < 8, f"Sample {b}: pick {p} out of range"

    def test_heterogeneous_dims(self):
        """Members with different D multi-select together.

        ``from_checkpoints`` advertises heterogeneous architectures, and the
        real ensembles are heterogeneous (archetype 0's seeds are D=128/256/512).
        Every D-dependent term inside the AR loop must therefore come from the
        member it is added to: the card encoder feeding ``pointers[i]`` and the
        msgru hidden state, neither of which member 0 can supply for member i.
        """
        m0 = _make_member(D=256, heads=8, layers=2, ff=512, seed=0)
        m1 = _make_member(D=128, heads=4, layers=2, ff=256, seed=1)
        m2 = _make_member(D=512, heads=8, layers=2, ff=1024, seed=2)
        ensemble = EnsemblePolicy(nn.ModuleList([m0, m1, m2]))
        ensemble.eval()

        x = _make_synthetic_batch(4, max_count=3)
        x["maxCount"] = torch.full((4,), 3, dtype=torch.long)
        x["minCount"] = torch.full((4,), 1, dtype=torch.long)

        chosen = select_multi(ensemble, x)
        assert chosen.shape == (4, 3)
        for b in range(4):
            picks = [int(p) for p in chosen[b].tolist() if p >= 0]
            assert len(set(picks)) == len(picks), f"Sample {b}: duplicate picks {picks}"
            for p in picks:
                assert p < 8, f"Sample {b}: pick {p} out of range"

    def test_each_member_pointer_gets_its_own_card_encoder(self):
        """Member *i*'s pointer is called with member *i*'s card encoder.

        Same-D members hide a shared encoder: it still runs and only changes
        the numbers.  Perturbation cannot isolate it either — ``embed.card``
        also feeds the state tokens — so this asserts identity at the call.
        """
        members = [_make_member(D=256, heads=8, layers=2, ff=512, seed=s)
                   for s in range(3)]
        ensemble = EnsemblePolicy(nn.ModuleList(members))
        ensemble.eval()

        seen: dict[int, list] = {}

        def spy(i, member):
            orig = member.pointer.forward

            def fwd(h, tok_mask, card_enc, x, msgru_h=None):
                seen.setdefault(i, []).append(card_enc)
                return orig(h, tok_mask, card_enc, x, msgru_h=msgru_h)
            member.pointer.forward = fwd

        for i, m in enumerate(members):
            spy(i, m)

        x = _make_synthetic_batch(4, max_count=3)
        x["maxCount"] = torch.full((4,), 3, dtype=torch.long)
        x["minCount"] = torch.full((4,), 1, dtype=torch.long)
        select_multi(ensemble, x)

        assert set(seen) == {0, 1, 2}, f"not every member's pointer ran: {sorted(seen)}"
        for i, m in enumerate(members):
            assert seen[i], f"member {i}'s pointer was never called"
            for enc in seen[i]:
                assert enc is m.embed.card, (
                    f"member {i}'s pointer got a card encoder belonging to "
                    f"another member"
                )

    def test_single_member_ensemble_matches_policy(self):
        """1-member ensemble select_multi matches single Policy select_multi."""
        torch.manual_seed(42)
        m0 = _make_member(seed=42)
        m0.eval()
        ensemble = EnsemblePolicy(nn.ModuleList([m0]))
        ensemble.eval()

        x = _make_synthetic_batch(4, max_count=3)
        x["maxCount"] = torch.full((4,), 3, dtype=torch.long)
        x["minCount"] = torch.full((4,), 1, dtype=torch.long)

        chosen_policy = select_multi(m0, x)
        chosen_ensemble = select_multi(ensemble, x)

        assert torch.equal(chosen_policy, chosen_ensemble), (
            f"1-member ensemble select_multi differs from policy:\n"
            f"policy:\n{chosen_policy}\nensemble:\n{chosen_ensemble}"
        )
