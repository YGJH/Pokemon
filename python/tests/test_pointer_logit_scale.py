"""The pointer's scoring input must be normalised.

`PointerHead` scores an option by projecting the output of a residual add
(``o = o + ffn(o)``).  Without a LayerNorm between that add and ``score``,
logit magnitude tracks the FFN output weights, which drift upward over long
runs — measured 21.9 → 243.5 over 80k steps on archetype 0, taking the scorer
0.27 → 10.1 with it and the val top-1 0.698 → 0.601.  The encoder already
carries a final ``nn.LayerNorm`` for exactly this reason (see ``encoder.py``);
these tests pin the same property on the pointer.

Mutation check: deleting ``self.ln_out`` from ``PointerHead.forward`` makes
``test_logit_scale_survives_ffn_blowup`` fail with a ratio in the hundreds.
"""

import torch

from ptcg_il.model.pointer import PointerHead
from ptcg_il.model.policy import Policy


def _pointer_inputs(B=4, L=46, O=12, D=64):
    from ptcg_il.featurizer import F_ATK, F_CARD, F_OPT

    torch.manual_seed(0)
    h = torch.randn(B, L, D)
    tok_mask = torch.ones(B, L, dtype=torch.bool)
    x = {
        "opt_type": torch.randint(0, 18, (B, O)),
        "opt_src_idx": torch.randint(-1, L, (B, O)),
        "opt_tgt_idx": torch.randint(-1, L, (B, O)),
        "opt_card_feat": torch.randn(B, O, F_CARD),
        "opt_attack_feat": torch.randn(B, O, F_ATK),
        "opt_scalar": torch.randn(B, O, F_OPT),
        "opt_mask": torch.ones(B, O, dtype=torch.bool),
    }
    return h, tok_mask, x


def _card_enc(D=64):
    from ptcg_il.model.cards import CardFeaturizer

    return CardFeaturizer(D)


def test_logit_scale_survives_ffn_blowup():
    """Inflating the FFN output projection must not inflate the logits.

    This is the failure mode observed in run 7e2wozk9: ``pointer.ffn.3.weight``
    grew 11× and the logits followed it out of the range where softmax is
    meaningful.
    """
    D = 64
    torch.manual_seed(0)
    ptr = PointerHead(D=D, heads=4)
    ptr.card = _card_enc(D)
    ptr.eval()
    h, tok_mask, x = _pointer_inputs(D=D)

    with torch.no_grad():
        base, _ = ptr(h, tok_mask, ptr.card, x)
        # Reproduce the measured drift: FFN output weights blown up 100×.
        ptr.ffn[3].weight.mul_(100.0)
        ptr.ffn[3].bias.mul_(100.0)
        blown, _ = ptr(h, tok_mask, ptr.card, x)

    base_spread = float(base.std())
    blown_spread = float(blown.std())
    assert base_spread > 0, "degenerate fixture: base logits have no spread"
    ratio = blown_spread / base_spread
    assert ratio < 10.0, (
        f"logit spread grew {ratio:.1f}× when ffn.3 grew 100× — the scoring "
        f"input is unnormalised (base std {base_spread:.3f} → {blown_spread:.3f})"
    )


def test_score_reads_a_normalised_activation():
    """The tensor handed to ``score`` stays ~unit-scale even as the FFN drifts."""
    D = 64
    torch.manual_seed(0)
    ptr = PointerHead(D=D, heads=4)
    ptr.card = _card_enc(D)
    ptr.eval()

    captured = []
    ptr.score.register_forward_pre_hook(lambda _m, inp: captured.append(inp[0].detach()))

    h, tok_mask, x = _pointer_inputs(D=D)
    with torch.no_grad():
        ptr(h, tok_mask, ptr.card, x)
        ptr.ffn[3].weight.mul_(100.0)
        ptr.ffn[3].bias.mul_(100.0)
        ptr(h, tok_mask, ptr.card, x)

    assert len(captured) == 2, f"hook did not fire as expected: {len(captured)}"
    for i, t in enumerate(captured):
        rms = float(t.pow(2).mean().sqrt())
        assert 0.1 < rms < 10.0, f"score input {i} has RMS {rms:.3f}, not normalised"


def _policy_batch(B=2, O=16):
    """A batch shaped from the live featurizer constants (matches real shards)."""
    from ptcg_il.featurizer import (
        F_ATK, F_CARD, F_GLOBAL, F_HAND, F_OPT, F_POKE, F_SUM,
    )

    torch.manual_seed(1)
    return {
        "tok_type": torch.randint(0, 5, (B, 46)),
        "tok_owner": torch.randint(0, 2, (B, 46)),
        "tok_zone": torch.randint(0, 4, (B, 46)),
        "tok_mask": torch.ones(B, 46, dtype=torch.bool),
        "poke_card_feat": torch.randn(B, 12, F_CARD),
        "poke_feat": torch.randn(B, 12, F_POKE),
        "hand_card_feat": torch.randn(B, 30, F_CARD),
        "hand_feat": torch.randn(B, 30, F_HAND),
        "sum_feat": torch.randn(B, 2, F_SUM),
        "cls_feat": torch.randn(B, F_GLOBAL),
        "context_card_feat": torch.randn(B, 1, F_CARD),
        "effect_card_feat": torch.randn(B, 1, F_CARD),
        "stadium_card_feat": torch.randn(B, 1, F_CARD),
        "stadium_present": torch.ones(B, 1),
        "discard_card_feat": torch.randn(B, 2, 60, F_CARD),
        "discard_mask": torch.ones(B, 2, 60, dtype=torch.bool),
        "prize_card_feat": torch.randn(B, 2, 6, F_CARD),
        "opt_type": torch.randint(0, 18, (B, O)),
        "opt_src_idx": torch.randint(-1, 46, (B, O)),
        "opt_tgt_idx": torch.randint(-1, 46, (B, O)),
        "opt_card_feat": torch.randn(B, O, F_CARD),
        "opt_attack_feat": torch.randn(B, O, F_ATK),
        "opt_scalar": torch.randn(B, O, F_OPT),
        "opt_mask": torch.ones(B, O, dtype=torch.bool),
    }


def test_policy_logits_bounded_under_pointer_drift():
    """End-to-end: the same blowup through a full Policy stays in softmax range."""
    torch.manual_seed(0)
    pol = Policy(D=64, heads=4, layers=2, ff=128)
    pol.eval()
    x = _policy_batch()
    with torch.no_grad():
        pol.pointer.ffn[3].weight.mul_(100.0)
        pol.pointer.ffn[3].bias.mul_(100.0)
        logits, _value, _h = pol(x)
    finite = logits[x["opt_mask"]]
    assert torch.isfinite(finite).all()
    # Measured: 1.03 with ln_out, 21.9 without, at this mutation strength.
    # ln_out bounds what reaches ``score``; it cannot bound ``score.weight``
    # itself, so this pins the activation path only.
    assert float(finite.abs().max()) < 5.0, (
        f"pointer logits reached {float(finite.abs().max()):.1f} after an FFN "
        "blowup; softmax is saturated and CE becomes a scale readout"
    )
