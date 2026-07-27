"""ValueHead — predict episode outcome from CLS token (Appendix B.5).

``value = tanh(Linear(D,D) -> GELU -> Linear(D,1))`` applied to h_CLS.
Output in (-1,1) — predicts acting player's episode reward (±1).
"""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Predict win/loss value from CLS token.

    Parameters
    ----------
    D : int
        Model dimension (256).
    """

    def __init__(self, D: int = 256):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1),
        )

    def forward(self, h_cls: torch.Tensor) -> torch.Tensor:
        """Predict value from CLS representation.

        Parameters
        ----------
        h_cls : Tensor[B, D]
            CLS token (row 0) from the encoder output.

        Returns
        -------
        value : Tensor[B]
            Predicted outcome in (-1, 1).
        """
        return torch.tanh(self.f(h_cls)).squeeze(-1)
