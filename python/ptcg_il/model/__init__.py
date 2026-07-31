"""Model modules: CardEncoder, TokenEmbedder, Encoder, PointerHead, ValueHead, Policy.

Implements nn.Module classes per TRANSFORMER_IL_SPEC.md Appendix B.
"""

import torch.nn as nn


def MLP(in_features: int, hidden_features: int, out_features: int, dropout: float = 0.0) -> nn.Sequential:
    """Two-layer MLP with GELU: Linear(i,h) -> GELU -> Dropout -> Linear(h,o).

    Matches the B.1 definition exactly.
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_features, out_features),
    )


from ptcg_il.model.cards import CardFeaturizer, AttackFeaturizer  # noqa: E402, F401
from ptcg_il.model.embed import TokenEmbedder  # noqa: E402, F401
from ptcg_il.model.encoder import Encoder  # noqa: E402, F401
from ptcg_il.model.pointer import PointerHead  # noqa: E402, F401
from ptcg_il.model.value import ValueHead  # noqa: E402, F401
from ptcg_il.model.policy import Policy, multiselect_ce, select_multi  # noqa: E402, F401


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """Apply spec B.8 init: trunc_normal_(std=0.02) to Linear/Embedding weights.

    LayerNorm weights stay at default (1.0).  Biases stay at default (0.0).
    ``null_token`` / ``no_stadium`` are already zero-initialized by
    ``nn.Parameter(torch.zeros(D))`` and are left unchanged.

    Call once after model construction, before training.
    """
    for name, param in module.named_parameters():
        if param.dim() >= 2 and "weight" in name:
            # nn.Linear and nn.Embedding weights
            nn.init.trunc_normal_(param, std=std)
        # LayerNorm weight (dim=1, named "weight") → leave at default 1.0
        # Bias → leave at default 0.0


__all__ = [
    "MLP",
    "CardFeaturizer",
    "AttackFeaturizer",
    "TokenEmbedder",
    "Encoder",
    "PointerHead",
    "ValueHead",
    "Policy",
    "multiselect_ce",
    "select_multi",
    "init_weights",
]
