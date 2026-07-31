"""CardFeaturizer and AttackFeaturizer — pure MLP over feature vectors (Appendix B.1).

``card_embed(feat) = CardStaticMLP(feat)``
``attack_embed(feat) = AtkStaticMLP(feat)``

No learned id embeddings.  Every card is represented purely by its static
features (HP, type, stage, weakness, resistance, retreat cost, special flags).
Two cards with identical features are truly identical to the model —
zero-shot generalisation to any card, known or unknown.

PAD cards (all-zeros features) map to ~zero embedding (modulo bias, which is
zero-initialised).
"""

import torch
import torch.nn as nn

from ptcg_il.model import MLP

F_CARD = 94  # 52 base + 3 attacks × 14
F_ATK = 14


class CardFeaturizer(nn.Module):
    """Map card static-feature vectors to D-dim embeddings via a pure MLP.

    Parameters
    ----------
    D : int
        Output embedding dimension.
    """

    def __init__(self, D: int = 256):
        super().__init__()
        self.mlp = MLP(F_CARD, D, D)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Embed card features.

        Parameters
        ----------
        feat : float32 Tensor[..., F_CARD]
            Static feature vectors.  All-zeros = PAD.

        Returns
        -------
        Tensor[..., D]
        """
        return self.mlp(feat)


class AttackFeaturizer(nn.Module):
    """Map attack static-feature vectors to D-dim embeddings via a pure MLP.

    Parameters
    ----------
    D : int
        Output embedding dimension.
    """

    def __init__(self, D: int = 256):
        super().__init__()
        self.mlp = MLP(F_ATK, D, D)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Embed attack features.

        Parameters
        ----------
        feat : float32 Tensor[..., F_ATK]
            Static feature vectors.  All-zeros = PAD.

        Returns
        -------
        Tensor[..., D]
        """
        return self.mlp(feat)
