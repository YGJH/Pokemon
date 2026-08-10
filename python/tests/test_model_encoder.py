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


def _ffn_rates(enc):
    """The three ``nn.Dropout`` modules per layer: FFN-internal + 2 residual."""
    import torch.nn as nn

    return [m.p for m in enc.modules() if isinstance(m, nn.Dropout)]


def _attn_rates(enc):
    """``MultiheadAttention.dropout`` — a plain float, not a module, so it is
    invisible to a ``modules()`` sweep.  That is exactly why the two rates can
    be set independently, and why a test that only walks modules would not
    notice ``attn_dropout`` silently doing nothing."""
    return [layer.self_attn.dropout for layer in enc.enc.layers]


class TestDropoutArgumentIsHonoured:
    """C3 — ``Encoder.__init__`` began with ``dropout = 0.0``, before
    ``super().__init__()``, which made the constructor argument unreachable.
    The model therefore trained with no dropout no matter what was configured.

    The single rate is now split per site (``attn_dropout``/``ffn_dropout``),
    so the same regression is pinned once for each.
    """

    def test_ffn_dropout_reaches_the_transformer_layers(self):
        from ptcg_il.model.encoder import Encoder

        enc = Encoder(D=32, heads=2, layers=2, ff=64, ffn_dropout=0.25)
        rates = _ffn_rates(enc)
        assert rates, "no Dropout modules found — encoder shape changed"
        assert all(r == 0.25 for r in rates), (
            f"ffn_dropout did not reach the layers: {sorted(set(rates))}"
        )

    def test_attn_dropout_reaches_the_attention_weights(self):
        from ptcg_il.model.encoder import Encoder

        enc = Encoder(D=32, heads=2, layers=2, ff=64, attn_dropout=0.25)
        rates = _attn_rates(enc)
        assert rates, "no attention layers found — encoder shape changed"
        assert all(r == 0.25 for r in rates), (
            f"attn_dropout did not reach the attention weights: {sorted(set(rates))}"
        )

    def test_the_two_rates_are_independent(self):
        """The whole point of the split.  ``nn.TransformerEncoderLayer`` drives
        all four sites from one constructor argument, so the attention rate has
        to be re-pointed after ``nn.TransformerEncoder`` deep-copies the layer.
        Setting it on the prototype instead would reach nothing.
        """
        from ptcg_il.model.encoder import Encoder

        enc = Encoder(D=32, heads=2, layers=3, ff=64,
                      attn_dropout=0.0, ffn_dropout=0.1)
        assert _attn_rates(enc) == [0.0, 0.0, 0.0], (
            f"attn_dropout followed ffn_dropout: {_attn_rates(enc)}"
        )
        assert all(r == 0.1 for r in _ffn_rates(enc)), (
            f"ffn_dropout did not survive the split: {sorted(set(_ffn_rates(enc)))}"
        )

        # ...and the other way round, so neither is hardwired to the other.
        flipped = Encoder(D=32, heads=2, layers=3, ff=64,
                          attn_dropout=0.2, ffn_dropout=0.0)
        assert _attn_rates(flipped) == [0.2, 0.2, 0.2]
        assert all(r == 0.0 for r in _ffn_rates(flipped))

    def test_both_defaults_are_still_zero(self):
        from ptcg_il.model.encoder import Encoder

        enc = Encoder(D=32, heads=2, layers=1, ff=64)
        assert all(r == 0.0 for r in _ffn_rates(enc))
        assert all(r == 0.0 for r in _attn_rates(enc))

    def test_ffn_dropout_perturbs_activations_in_train_mode(self):
        """A rate that reaches the layers but is never applied is still dead."""
        import torch

        from ptcg_il.model.encoder import Encoder

        torch.manual_seed(0)
        enc = Encoder(D=32, heads=2, layers=2, ff=64, ffn_dropout=0.5).train()
        rows = torch.randn(4, 46, 32)
        mask = torch.ones(4, 46, dtype=torch.bool)
        a, b = enc(rows, mask), enc(rows, mask)
        assert not torch.allclose(a, b), "ffn_dropout is configured but has no effect"

        enc.eval()
        assert torch.allclose(enc(rows, mask), enc(rows, mask)), (
            "eval mode must be deterministic"
        )

    def test_attn_dropout_perturbs_activations_in_train_mode(self):
        """Same check for the site that is *off* by default — otherwise the
        default could be masking a knob that never worked in the first place.
        """
        import torch

        from ptcg_il.model.encoder import Encoder

        torch.manual_seed(0)
        enc = Encoder(D=32, heads=2, layers=2, ff=64,
                      attn_dropout=0.5, ffn_dropout=0.0).train()
        rows = torch.randn(4, 46, 32)
        mask = torch.ones(4, 46, dtype=torch.bool)
        a, b = enc(rows, mask), enc(rows, mask)
        assert not torch.allclose(a, b), "attn_dropout is configured but has no effect"

        enc.eval()
        assert torch.allclose(enc(rows, mask), enc(rows, mask)), (
            "eval mode must be deterministic"
        )

    def test_the_shipped_default_pair_is_deterministic_in_eval(self):
        """``--attn-dropout 0.0 --ffn-dropout 0.1`` is what training will use."""
        import torch

        from ptcg_il.model.encoder import Encoder

        torch.manual_seed(0)
        enc = Encoder(D=32, heads=2, layers=2, ff=64,
                      attn_dropout=0.0, ffn_dropout=0.1)
        rows = torch.randn(4, 46, 32)
        mask = torch.ones(4, 46, dtype=torch.bool)

        enc.train()
        assert not torch.allclose(enc(rows, mask), enc(rows, mask)), (
            "ffn_dropout=0.1 must still regularise with attn_dropout off"
        )
        enc.eval()
        assert torch.allclose(enc(rows, mask), enc(rows, mask))


class TestDropoutReachesTheEncoderFromAbove:
    """``Encoder`` honoured its rate (see above) but nothing ever passed one.

    ``Policy.__init__`` took no such argument and built ``Encoder(D, heads,
    layers, ff)``, so every model trained at 0.0 while ``ptcg_il.cli`` parsed
    ``--dropout`` (default 0.4) and dropped it on the floor.  Testing the
    ``Encoder`` constructor alone cannot see this: the dead link is one level up.
    """

    def test_policy_forwards_both_rates_to_its_encoder(self):
        from tests.test_model_policy import make_policy

        policy = make_policy(D=32, heads=2, layers=2, ff=64,
                             attn_dropout=0.2, ffn_dropout=0.25)
        ffn = _ffn_rates(policy.encoder)
        assert ffn, "no Dropout modules in Policy.encoder — encoder shape changed"
        assert all(r == 0.25 for r in ffn), (
            f"Policy did not forward ffn_dropout: {sorted(set(ffn))}"
        )
        assert all(r == 0.2 for r in _attn_rates(policy.encoder)), (
            f"Policy did not forward attn_dropout: {_attn_rates(policy.encoder)}"
        )

    def test_policy_does_not_collapse_the_two(self):
        """Passing only one must not move the other — the failure this split
        exists to prevent is attention dropout riding along with the FFN rate.
        """
        from tests.test_model_policy import make_policy

        policy = make_policy(D=32, heads=2, layers=2, ff=64, ffn_dropout=0.1)
        assert all(r == 0.0 for r in _attn_rates(policy.encoder)), (
            "ffn_dropout switched on attention dropout: "
            f"{_attn_rates(policy.encoder)}"
        )

    def test_policy_defaults_are_zero(self):
        """Inference and RL build Policy positionally; the defaults must be inert."""
        from tests.test_model_policy import make_policy

        policy = make_policy(D=32, heads=2, layers=2, ff=64)
        assert all(r == 0.0 for r in _ffn_rates(policy.encoder))
        assert all(r == 0.0 for r in _attn_rates(policy.encoder))

    def test_dropout_stays_out_of_the_heads(self):
        """Encoder-only was the decision.  The MLP heads keep their own 0.0."""
        from tests.test_model_policy import make_policy

        policy_mods = None
        policy = make_policy(D=32, heads=2, layers=2, ff=64,
                             attn_dropout=0.2, ffn_dropout=0.25)
        for name in ("embed", "pointer", "value", "belief", "belief_heads"):
            sub = getattr(policy, name, None)
            if sub is None:
                continue
            policy_mods = True
            assert all(r == 0.0 for r in _ffn_rates(sub)), (
                f"dropout leaked into {name}; the decision was encoder-only"
            )
        assert policy_mods, "no head submodules examined — Policy layout changed"

    def test_config_records_both_rates(self):
        from tests.test_model_policy import make_policy

        policy = make_policy(D=32, heads=2, layers=2, ff=64,
                             attn_dropout=0.2, ffn_dropout=0.25)
        assert policy.config["attn_dropout"] == 0.2
        assert policy.config["ffn_dropout"] == 0.25

    def test_policy_from_config_rebuilds_at_zero(self):
        """RL's ``logp_old`` pass runs under ``policy.train()`` on purpose, to
        match the PPO update's non-fused encoder kernel.  Restoring the trained
        rates here would put live dropout in that pass and randomise ``logp_old``.
        """
        from ptcg_il.model.policy import policy_from_config
        from tests.test_model_policy import make_policy, make_static_tables

        card, atk = make_static_tables()
        trained = make_policy(D=32, heads=2, layers=2, ff=64,
                              attn_dropout=0.2, ffn_dropout=0.25)

        rebuilt = policy_from_config(
            trained.config, all_card_feat=card, all_attack_feat=atk,
        )
        assert all(r == 0.0 for r in _ffn_rates(rebuilt.encoder)), (
            "policy_from_config restored ffn_dropout; RL's train()-mode "
            "logp_old recompute is no longer deterministic"
        )
        assert all(r == 0.0 for r in _attn_rates(rebuilt.encoder)), (
            "policy_from_config restored attn_dropout; RL's train()-mode "
            "logp_old recompute is no longer deterministic"
        )

    def test_rates_do_not_change_the_state_dict(self):
        """Why not restoring them is safe: dropout carries no parameters, so a
        checkpoint trained at any rate loads into a policy built at any other.
        """
        from ptcg_il.model.policy import load_policy_state
        from tests.test_model_policy import make_policy

        trained = make_policy(D=32, heads=2, layers=2, ff=64,
                              attn_dropout=0.2, ffn_dropout=0.25)
        inference = make_policy(D=32, heads=2, layers=2, ff=64)
        assert set(trained.state_dict()) == set(inference.state_dict())
        load_policy_state(inference, trained.state_dict())


def _data_dir_with_static_tables(tmp_path):
    """``_build_policy`` loads the engine tables off disk, not from artifacts.

    Both artifacts are pickled ``{engine_id: feature_list}`` dicts, which
    ``load_static_tables`` densifies via ``build_static_table``.
    """
    import numpy as np

    from ptcg_il.featurizer import F_ATK, F_CARD

    for name, dim in (("engine_card_features.npy", F_CARD),
                      ("engine_attack_features.npy", F_ATK)):
        feats = {i: np.zeros(dim, dtype=np.float32) for i in range(1, 8)}
        np.save(tmp_path / name, feats, allow_pickle=True)
    return tmp_path


class TestCliPassesDropoutToThePolicy:
    """The flag existed and was parsed; ``args.dropout`` was read nowhere."""

    def test_shipped_defaults_are_the_split_pair(self):
        from ptcg_il.cli import DEFAULTS

        assert "dropout" not in DEFAULTS, (
            "the single rate was replaced by the per-site pair; a leftover key "
            "is a knob nothing reads, which is the bug this started as"
        )
        assert DEFAULTS["attn_dropout"] == 0.0, (
            "attention-weight dropout over 46 structured entity tokens removes "
            "facts rather than adding noise; it stays off by default"
        )
        assert DEFAULTS["ffn_dropout"] == 0.1

    def test_the_flags_parse_and_are_independent(self):
        from ptcg_il.cli import _build_parser

        args = _build_parser().parse_args(["train"])
        assert (args.attn_dropout, args.ffn_dropout) == (0.0, 0.1)

        args = _build_parser().parse_args(
            ["train", "--attn-dropout", "0.05", "--ffn-dropout", "0.2"])
        assert (args.attn_dropout, args.ffn_dropout) == (0.05, 0.2)

        # Setting one must leave the other at its default.
        args = _build_parser().parse_args(["train", "--ffn-dropout", "0.3"])
        assert (args.attn_dropout, args.ffn_dropout) == (0.0, 0.3)

    def test_build_policy_applies_both_flags(self, tmp_path):
        import argparse

        from ptcg_il.cli import _build_policy

        args = argparse.Namespace(
            d_model=32, heads=2, layers=2, ff=64, seed=1,
            data_dir=_data_dir_with_static_tables(tmp_path),
            attn_dropout=0.15, ffn_dropout=0.3,
        )
        policy = _build_policy({"n_opp_arch": 3}, args)
        ffn = _ffn_rates(policy.encoder)
        assert ffn and all(r == 0.3 for r in ffn), (
            f"--ffn-dropout did not reach the encoder: {sorted(set(ffn))}"
        )
        assert all(r == 0.15 for r in _attn_rates(policy.encoder)), (
            f"--attn-dropout did not reach the encoder: "
            f"{_attn_rates(policy.encoder)}"
        )
        assert policy.config["attn_dropout"] == 0.15
        assert policy.config["ffn_dropout"] == 0.3

    def test_absent_flags_mean_no_dropout(self, tmp_path):
        """Eval-only and tests build the Namespace by hand; a missing flag must
        not raise, and must not silently enable regularisation either."""
        import argparse

        from ptcg_il.cli import _build_policy

        args = argparse.Namespace(
            d_model=32, heads=2, layers=2, ff=64, seed=1,
            data_dir=_data_dir_with_static_tables(tmp_path),
        )
        policy = _build_policy({}, args)
        assert all(r == 0.0 for r in _ffn_rates(policy.encoder))
        assert all(r == 0.0 for r in _attn_rates(policy.encoder))
