"""Submission model package — inference-only."""
import torch.nn as nn


def MLP(
    in_features: int,
    hidden_features: int,
    out_features: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Two-layer MLP with GELU: Linear(i,h) -> GELU -> Dropout -> Linear(h,o).

    Matches the B.1 definition from TRANSFORMER_IL_SPEC.md exactly.
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_features, out_features),
    )


from model.cards import CardFeaturizer, AttackFeaturizer  # noqa: E402
from model.policy import Policy, select_multi  # noqa: E402
