"""Encoder — self-attention transformer over state tokens (Appendix B.3).

``nn.TransformerEncoder`` with pre-norm (norm_first=True), GELU, batch_first.
Token count is small (~46) so no packing tricks.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Transformer encoder over L_STATE=46 state tokens.

    Parameters
    ----------
    D : int
        Model dimension (256).
    heads : int
        Attention heads (8).
    layers : int
        Number of transformer layers (4).
    ff : int
        Feed-forward hidden dim (1024).
    dropout : float
        Dropout probability (0.1).
    """

    def __init__(
        self,
        D: int = 256,
        heads: int = 8,
        layers: int = 4,
        ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(
        self, rows: torch.Tensor, tok_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode state tokens with self-attention.

        Parameters
        ----------
        rows : Tensor[B, L, D]
            Token embeddings from TokenEmbedder.
        tok_mask : bool Tensor[B, L]
            True = real token, False = padding.  Inverted to
            ``src_key_padding_mask`` where True = ignored.

        Returns
        -------
        h : Tensor[B, L, D]
            Contextualized token representations.
        """
        return self.enc(rows, src_key_padding_mask=~tok_mask)
