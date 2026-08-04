"""PointerHead — option-token builder + cross-attention + scoring (Appendix B.4).

Options gather encoded state-token rows, cross-attend into the full state
sequence, and produce per-option logits.  A learned ``null_token`` handles
index==-1 references.  ``msgru_h`` enables multi-select re-scoring (B.7).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptcg_il.model import MLP
from ptcg_il.model.cards import AttackFeaturizer, CardFeaturizer

L_STATE = 46
O_MAX = 64
from ptcg_il.featurizer import F_OPT


class PointerHead(nn.Module):
    """Pointer head: build option queries, cross-attend into state, score.

    ``self.card`` is bound externally by ``Policy`` to the shared CardEncoder
    (see B.6).

    Parameters
    ----------
    A : int
        Attack vocab size.
    D : int
        Model dimension (256).
    heads : int
        Cross-attention heads (8).
    attack_static_table : Tensor[A, 14] or None
        Prebuilt table for AttackEncoder.
    """

    def __init__(self, D: int = 256, heads: int = 8):
        super().__init__()
        self.opt_type_emb = nn.Embedding(18, D)  # OptionType 0..16 + STOP=17
        self.card: CardFeaturizer | None = None   # bound externally by Policy
        self.attack = AttackFeaturizer(D)
        self.null_token = nn.Parameter(torch.zeros(D))
        self.msgru = nn.GRUCell(D, D)             # multi-select memory
        self.opt_in = nn.Linear(D + F_OPT, D)
        self.cross = nn.MultiheadAttention(D, heads, batch_first=True)
        self.ln_q = nn.LayerNorm(D)
        self.ln_o = nn.LayerNorm(D)
        self.ffn = MLP(D, 4 * D, D)
        # Normalises the residual stream on the way *out*, before scoring.  See
        # the note in ``forward`` -- without it the logit scale is unbounded.
        self.ln_out = nn.LayerNorm(D)
        self.score = nn.Linear(D, 1)

    def gather(self, h_aug: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather rows from h_aug, mapping idx==-1 to the null (last) row.

        Parameters
        ----------
        h_aug : Tensor[B, L+1, D]
            Encoded state rows with null row appended at position L.
        idx : int64 Tensor[B, O]
            Index tensor (may contain -1).

        Returns
        -------
        Tensor[B, O, D]
        """
        L_aug = h_aug.shape[1] - 1  # L
        idx_safe = torch.where(idx < 0, torch.full_like(idx, L_aug), idx)
        return torch.gather(
            h_aug, 1, idx_safe.unsqueeze(-1).expand(-1, -1, h_aug.shape[-1])
        )

    def forward(
        self,
        h: torch.Tensor,
        tok_mask: torch.Tensor,
        card_enc: CardFeaturizer,
        x: dict[str, torch.Tensor],
        msgru_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score options given encoded state.

        Parameters
        ----------
        h : Tensor[B, L, D]
            Encoded state tokens from Encoder.
        tok_mask : bool Tensor[B, L]
            True = real state token.
        card_enc : CardFeaturizer
            Shared card featurizer (for opt_card_feat embedding).
        x : dict
            Option tensors: opt_type [B,O], opt_src_idx [B,O], opt_tgt_idx [B,O],
            opt_card_feat [B,O,F_CARD], opt_attack_feat [B,O,F_ATK],
            opt_scalar [B,O,F_OPT], opt_mask bool [B,O].
        msgru_h : Tensor[B, D] or None
            Multi-select GRU hidden state.  When None (single-select),
            no extra context is added.

        Returns
        -------
        logits : Tensor[B, O]
            Per-option scores, padded options filled with -1e9.
        o : Tensor[B, O, D]
            Per-option representations (for GRU update).
        """
        B = h.shape[0]
        D = h.shape[-1]

        # Append null row at position L
        h_aug = torch.cat([h, self.null_token.expand(B, 1, D)], dim=1)  # [B, L+1, D]

        src = self.gather(h_aug, x["opt_src_idx"])                       # [B, O, D]
        tgt = self.gather(h_aug, x["opt_tgt_idx"])                       # [B, O, D]

        base = (
            self.opt_type_emb(x["opt_type"])
            + src
            + tgt
            + card_enc(x["opt_card_feat"])
            + self.attack(x["opt_attack_feat"])
        )  # [B, O, D]

        if msgru_h is not None:
            base = base + msgru_h.unsqueeze(1)  # [B, O, D]

        # Option query
        q_input = torch.cat([base, x["opt_scalar"]], dim=-1)             # [B, O, D+F_OPT]
        q = self.ln_q(F.gelu(self.opt_in(q_input)))                       # [B, O, D]

        # Cross-attention: option queries attend to state tokens
        a, _ = self.cross(q, h, h, key_padding_mask=~tok_mask)            # [B, O, D]
        o = self.ln_o(q + a)
        o = o + self.ffn(o)                                               # [B, O, D]

        # Score.  ``o`` is a residual sum, so its scale is whatever ``ffn`` has
        # drifted to -- and ``score`` is a bare Linear, so the logits inherit
        # that drift directly.  Left unnormalised this diverges: over 80k steps
        # on archetype 0, ``ffn.3.weight`` grew 21.9 -> 243.5 and ``score.weight``
        # 0.27 -> 10.1, the softmax saturated, and val top-1 fell 0.698 -> 0.601
        # while training CE read in the thousands.  Grad clipping does not help
        # -- it bounds the update norm, not the direction, and the growth is
        # monotone from step 20k.  The encoder solved the same problem with a
        # final ``nn.LayerNorm`` (see encoder.py); its weight norm moved 1.00x
        # over the same run.  This is that norm, for the pointer.
        o = self.ln_out(o)                                                # [B, O, D]
        logits = self.score(o).squeeze(-1)                                # [B, O]
        logits = logits.masked_fill(~x["opt_mask"], -1e9)

        return logits, o
