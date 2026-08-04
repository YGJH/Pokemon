"""TokenEmbedder — featurizer dict -> [B, L_STATE, D] (Appendix B.2).

Assembles state tokens at the fixed A.1 positions with per-group feature
projections, the shared CardFeaturizer, additive categorical embeddings,
and discard-pile bag-of-cards pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptcg_il.model import MLP
from ptcg_il.model.cards import CardFeaturizer

# Fixed capacities (A.1)
L_STATE = 46
P_MAX = 12
H_MAX = 30
SUM = 2

from ptcg_il.featurizer import F_GLOBAL, F_HAND, F_POKE, F_SUM

# Normalizer for discard sum-pooling (A.2)
DECK_N = 60.0


class TokenEmbedder(nn.Module):
    """Convert featurizer tensor dict to a [B, L_STATE, D] state-token sequence.

    Cards are represented purely by their static feature vectors — no vocab
    indices or id embeddings.

    Parameters
    ----------
    D : int
        Output embedding dimension (256).
    """

    def __init__(self, D: int = 256):
        super().__init__()
        self.card = CardFeaturizer(D)

        self.type_emb = nn.Embedding(5, D)   # CLS, POKE, HAND, SUMMARY, STADIUM
        self.owner_emb = nn.Embedding(3, D)  # none, self, opp
        self.zone_emb = nn.Embedding(6, D)   # cls, active, bench, hand, summary, stadium

        self.poke_mlp = MLP(F_POKE, D, D)
        self.hand_mlp = MLP(F_HAND, D, D)
        self.sum_mlp = MLP(F_SUM, D, D)
        self.cls_mlp = MLP(F_GLOBAL, D, D)

        self.no_stadium = nn.Parameter(torch.zeros(D))

        self.D = D

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the [B, L_STATE, D] token sequence.

        Parameters
        ----------
        x : dict
            Featurizer tensor dict with batch dim B.  Requires keys:
            cls_feat, context_card_feat, effect_card_feat, poke_feat,
            poke_card_feat, hand_feat, hand_card_feat, sum_feat,
            discard_card_feat, discard_mask, prize_card_feat,
            stadium_present, stadium_card_feat, tok_type, tok_owner,
            tok_zone.  All ``*_card_feat`` keys are float32[..., F_CARD].

        Returns
        -------
        rows : Tensor[B, L_STATE, D]
        """
        B = x["tok_type"].shape[0]
        device = x["tok_type"].device
        rows = torch.zeros(B, L_STATE, self.D, device=device)

        # --- CLS token (row 0) ---
        x_has_context = x["cls_feat"][:, 87:88]   # [B, 1]
        x_has_effect = x["cls_feat"][:, 88:89]    # [B, 1]

        cls_tok = self.cls_mlp(x["cls_feat"])                                              # [B, D]
        cls_tok = cls_tok + self.card(x["context_card_feat"]).squeeze(1) * x_has_context  # [B, D]
        cls_tok = cls_tok + self.card(x["effect_card_feat"]).squeeze(1) * x_has_effect     # [B, D]
        rows[:, 0] = cls_tok

        # --- Pokemon tokens (rows 1..12) ---
        poke_emb = self.poke_mlp(x["poke_feat"]) + self.card(x["poke_card_feat"])  # [B, P_MAX, D]
        rows[:, 1:13] = poke_emb

        # --- Hand tokens (rows 13..42) ---
        hand_emb = self.hand_mlp(x["hand_feat"]) + self.card(x["hand_card_feat"])  # [B, H_MAX, D]
        rows[:, 13:43] = hand_emb

        # --- Summary tokens (rows 43..44) with discard + prize pooling ---
        # Mask-aware SUM of discard card embeddings, normalized by DECK_N
        discard_emb = self.card(x["discard_card_feat"])                            # [B, SUM, D_MAX, D]
        discard_mask = x["discard_mask"].to(discard_emb.dtype).unsqueeze(-1)       # [B, SUM, D_MAX, 1]
        disc = (discard_emb * discard_mask).sum(dim=2) / DECK_N                    # [B, SUM, D]
        summ = self.sum_mlp(x["sum_feat"]) + disc                                  # [B, SUM, D]

        # Mask-aware SUM of revealed prize card embeddings, normalized by DECK_N
        prize_emb = self.card(x["prize_card_feat"])                                 # [B, SUM, PZ_MAX, D]
        prize_mask = (x["prize_card_feat"].abs().sum(-1) > 0).to(prize_emb.dtype).unsqueeze(-1)  # [B, SUM, PZ_MAX, 1]
        prize = (prize_emb * prize_mask).sum(dim=2) / DECK_N                       # [B, SUM, D]
        summ = summ + prize                                                        # [B, SUM, D]

        rows[:, 43:45] = summ

        # --- Stadium token (row 45) ---
        stad_emb = self.card(x["stadium_card_feat"]).squeeze(1)                    # [B, D]
        stadium_present = x["stadium_present"].to(stad_emb.dtype)                  # [B, 1]
        stad = stad_emb * stadium_present + self.no_stadium * (1.0 - stadium_present)
        rows[:, 45] = stad

        # --- Additive categorical embeddings for ALL rows ---
        rows = rows + self.type_emb(x["tok_type"]) \
                    + self.owner_emb(x["tok_owner"]) \
                    + self.zone_emb(x["tok_zone"])

        return rows  # [B, L_STATE, D]
