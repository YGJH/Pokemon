"""BeliefModule — GRU over game logs for opponent-state inference.

Processes the sequence of ``Log`` entries since the last decision to build a
belief state about the opponent's hidden information (hand contents, deck
composition, prize cards).
"""

import torch
import torch.nn as nn

L_LOG_MAX = 32
LOG_FEAT_DIM = 6  # log_type, player_rel, card_id, area_from, area_to, scalar


class BeliefModule(nn.Module):
    """GRU-based belief encoder over game log entries.

    Each log entry is a compact 6-dim feature vector.  The module embeds the
    discrete fields (log_type, card_id) and feeds the concatenated features
    through a GRU.  The final hidden state is the belief vector.

    Parameters
    ----------
    V : int
        Card vocab size (for card_id embedding, shared with Policy).
    D : int
        Model dimension (256).
    """

    def __init__(self, V: int, D: int = 256):
        super().__init__()
        self.log_type_emb = nn.Embedding(24, 16)       # LogType 0..23 → 16 dims
        self.card_emb: nn.Embedding | None = None       # bound externally to shared CardEncoder
        self.area_emb = nn.Embedding(7, 4)              # 6 areas + 1 for None(-1→6)
        self.in_proj = nn.Linear(16 + D + 4 + 4 + 1 + 1, D)  # total → D
        self.gru = nn.GRU(D, D, num_layers=1, batch_first=True)

    def forward(
        self,
        log_feat: torch.Tensor,
        log_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode log sequence into a belief state.

        Parameters
        ----------
        log_feat : Tensor[B, L_MAX, 6]
            Raw log features: [log_type, player_rel, card_id, area_from,
            area_to, scalar].
        log_mask : bool Tensor[B, L_MAX]
            True = valid log entry.

        Returns
        -------
        belief : Tensor[B, D]
            Final GRU hidden state (belief vector).
        """
        B, L, _ = log_feat.shape
        device = log_feat.device

        lt = log_feat[:, :, 0].long()                                    # [B, L]
        player_rel = log_feat[:, :, 1:2]                                 # [B, L, 1]
        card_id = log_feat[:, :, 2].long()                               # [B, L]
        area_from = log_feat[:, :, 3].long().clamp(-1, 5) + 1            # [B, L], -1→0
        area_to = log_feat[:, :, 4].long().clamp(-1, 5) + 1              # [B, L], -1→0
        scalar = log_feat[:, :, 5:6]                                     # [B, L, 1]

        # Embed discrete fields
        lt_emb = self.log_type_emb(lt)                                   # [B, L, 16]
        if self.card_emb is not None:
            card_emb = self.card_emb(card_id)                            # [B, L, D]
        else:
            card_emb = torch.zeros(B, L, self.in_proj.weight.shape[0] // 4,
                                   device=device)  # fallback (should not happen)
        area_f_emb = self.area_emb(area_from)                            # [B, L, 4]
        area_t_emb = self.area_emb(area_to)                              # [B, L, 4]

        # Concatenate all features
        x = torch.cat([lt_emb, card_emb, area_f_emb, area_t_emb,
                       player_rel, scalar], dim=-1)                      # [B, L, D_in]
        x = self.in_proj(x)                                              # [B, L, D]

        # GRU over sequence (handle empty sequences)
        lengths = log_mask.sum(dim=1).clamp(min=1)                       # [B]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_last = self.gru(packed)                                     # [1, B, D]
        belief = h_last.squeeze(0)                                       # [B, D]

        return belief
