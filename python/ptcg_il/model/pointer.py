"""PointerHead — option-token builder + cross-attention + scoring (Appendix B.4).

Options gather encoded state-token rows, cross-attend into the full state
sequence, and produce per-option logits.  A learned ``null_token`` handles
index==-1 references.  ``extra_ctx`` enables multi-select re-scoring (B.7).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptcg_il.model import MLP
from ptcg_il.model.cards import AttackEncoder, CardEncoder

L_STATE = 46
O_MAX = 128
F_OPT = 6


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

    def __init__(
        self,
        A: int,
        D: int = 256,
        heads: int = 8,
        attack_static_table: torch.Tensor | None = None,
    ):
        super().__init__()
        self.opt_type_emb = nn.Embedding(17, D)  # OptionType 0..16
        self.card: CardEncoder | None = None      # bound externally by Policy
        self.attack = AttackEncoder(A, D, attack_static_table)
        self.null_token = nn.Parameter(torch.zeros(D))
        self.opt_in = nn.Linear(D + F_OPT, D)
        self.cross = nn.MultiheadAttention(D, heads, batch_first=True)
        self.ln_q = nn.LayerNorm(D)
        self.ln_o = nn.LayerNorm(D)
        self.ffn = MLP(D, 4 * D, D)
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
        card_enc: CardEncoder,
        x: dict[str, torch.Tensor],
        extra_ctx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score options given encoded state.

        Parameters
        ----------
        h : Tensor[B, L, D]
            Encoded state tokens from Encoder.
        tok_mask : bool Tensor[B, L]
            True = real state token.
        card_enc : CardEncoder
            Shared card encoder (for opt_card_id embedding).
        x : dict
            Option tensors: opt_type [B,O], opt_src_idx [B,O], opt_tgt_idx [B,O],
            opt_card_id [B,O], opt_attack_idx [B,O], opt_scalar [B,O,F_OPT],
            opt_mask bool [B,O].
        extra_ctx : Tensor[B, D] or None
            Running sum of already-picked option reprs (multi-select B.7).

        Returns
        -------
        logits : Tensor[B, O]
            Per-option scores, padded options filled with -1e9.
        o : Tensor[B, O, D]
            Per-option representations (for multi-select pooling).
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
            + card_enc(x["opt_card_id"])
            + self.attack(x["opt_attack_idx"])
        )  # [B, O, D]

        if extra_ctx is not None:
            base = base + extra_ctx.unsqueeze(1)  # [B, O, D]

        # Option query
        q_input = torch.cat([base, x["opt_scalar"]], dim=-1)             # [B, O, D+F_OPT]
        q = self.ln_q(F.gelu(self.opt_in(q_input)))                       # [B, O, D]

        # Cross-attention: option queries attend to state tokens
        a, _ = self.cross(q, h, h, key_padding_mask=~tok_mask)            # [B, O, D]
        o = self.ln_o(q + a)
        o = o + self.ffn(o)                                               # [B, O, D]

        # Score
        logits = self.score(o).squeeze(-1)                                # [B, O]
        logits = logits.masked_fill(~x["opt_mask"], -1e9)

        return logits, o
