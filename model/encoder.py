"""Encoder — self-attention transformer over state tokens (Appendix B.3).

``nn.TransformerEncoder`` with pre-norm (norm_first=True), GELU, batch_first.
Token count is small (~46) so no packing tricks.

A pre-norm stack normalizes *into* each sublayer but never on the way out, so
the residual stream leaves the last layer un-normalized and its scale grows
with depth.  Measured on ``checkpoints_a0/ckpt-best.pt`` over 256 real samples,
per-element RMS runs 0.13 (embedder) → 0.43 → 0.77 → 1.22 → 1.49 across the
four layers, against the 1.0 a final norm would give.  The consumers care to
different degrees: the CLS row is re-derived by ``Policy.history_gru`` and
arrives tanh-bounded either way, but ``PointerHead`` cross-attends into these
rows as unnormalized keys/values and adds gathered rows straight into its
additive base, so their scale sets attention sharpness.  1.49 is mild at four
layers -- this is hygiene, not a repair -- but it costs one LayerNorm and stops
the drift from compounding if ``layers`` is ever raised.
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
    attn_dropout : float
        Dropout on the *attention weights* (``self_attn.dropout``), i.e. inside
        the softmax.  Held at 0.0 by default even when ``ffn_dropout`` is on --
        see the note above on why this site is the expensive one here.
    ffn_dropout : float
        Dropout on the other three sites: inside the FFN after the activation,
        and on each of the two residual branches (``dropout1``/``dropout2``).

    Both default to 0.0 so inference and RL rebuilds are deterministic;
    ``ptcg_il.cli`` supplies the training rates (``--attn-dropout`` 0.0,
    ``--ffn-dropout`` 0.1).
    """

    def __init__(
        self,
        D: int = 256,
        heads: int = 8,
        layers: int = 4,
        ff: int = 1024,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        # One constructor arg drives all four sites, so the FFN/residual rate
        # goes in here and the attention weights are re-pointed afterwards.
        layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=heads,
            dim_feedforward=ff,
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(D),
        )
        # After TransformerEncoder, not before: it deep-copies `layer` N times,
        # so a value set on the prototype would be copied but one set on
        # `layer.self_attn` after construction would reach nothing.
        # `MultiheadAttention.dropout` is a plain float read at forward time,
        # which is the only seam torch gives us for splitting the two rates.
        for enc_layer in self.enc.layers:
            enc_layer.self_attn.dropout = float(attn_dropout)
        self.attn_dropout = float(attn_dropout)
        self.ffn_dropout = float(ffn_dropout)

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
