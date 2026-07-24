"""CardEncoder and AttackEncoder — id embedding + static-feature MLP (Appendix B.1).

``card_embed(id) = IdEmb[id] + CardStaticMLP(card_static[id])``
``attack_embed(idx) = AtkIdEmb[idx] + AtkStaticMLP(attack_static[idx])``

The static tables ``register_buffer`` store prebuilt ``[V, F_CARD]`` / ``[A, F_ATK]``
rows so samples carry only integer ids.
"""

import torch
import torch.nn as nn

from ptcg_il.model import MLP

F_CARD = 52
F_ATK = 14


class CardEncoder(nn.Module):
    """Map card vocab ids to D-dim embeddings: IdEmb + StaticMLP(static_features).

    Parameters
    ----------
    V : int
        Vocab size (PAD=0, UNKNOWN=1, real cards 2..V-1).
    D : int
        Output embedding dimension.
    card_static_table : Tensor[V, F_CARD] or None
        Prebuilt static-feature table (row 0 = PAD zeros, row 1 = UNKNOWN mean).
        If None, uses a zero-filled buffer (for testing).
    """

    def __init__(
        self,
        V: int,
        D: int = 256,
        card_static_table: torch.Tensor | None = None,
    ):
        super().__init__()
        self.id_emb = nn.Embedding(V, D, padding_idx=0)
        self.static_mlp = MLP(F_CARD, D, D)

        if card_static_table is not None:
            self.register_buffer("static", card_static_table.to(torch.float32))
        else:
            self.register_buffer("static", torch.zeros(V, F_CARD))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed card ids.

        Parameters
        ----------
        ids : int64 Tensor[...,]
            Remapped vocab indices.  PAD=0 produces zero embedding.

        Returns
        -------
        Tensor[..., D]
        """
        return self.id_emb(ids) + self.static_mlp(self.static[ids])


class AttackEncoder(nn.Module):
    """Map attack vocab indices to D-dim embeddings: IdEmb + StaticMLP(static_features).

    Parameters
    ----------
    A : int
        Attack vocab size (PAD_ATTACK=0, real attacks 1..A-1).
    D : int
        Output embedding dimension.
    attack_static_table : Tensor[A, F_ATK] or None
        Prebuilt static-feature table (row 0 = PAD zeros).  If None, zero-filled.
    """

    def __init__(
        self,
        A: int,
        D: int = 256,
        attack_static_table: torch.Tensor | None = None,
    ):
        super().__init__()
        self.id_emb = nn.Embedding(A, D, padding_idx=0)
        self.static_mlp = MLP(F_ATK, D, D)

        if attack_static_table is not None:
            self.register_buffer("static", attack_static_table.to(torch.float32))
        else:
            self.register_buffer("static", torch.zeros(A, F_ATK))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Embed attack indices.

        Parameters
        ----------
        idx : int64 Tensor[...,]
            Remapped attack indices.  PAD=0 produces zero embedding.

        Returns
        -------
        Tensor[..., D]
        """
        return self.id_emb(idx) + self.static_mlp(self.static[idx])
