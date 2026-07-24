"""Policy — end-to-end model + multi-select helpers (Appendix B.6–B.8).

``Policy.forward`` returns single-select logits.  ``multiselect_ce`` handles
teacher-forced AR training for multi-select decisions.  ``select_multi``
handles greedy AR inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptcg_il.model.embed import TokenEmbedder
from ptcg_il.model.encoder import Encoder
from ptcg_il.model.pointer import PointerHead
from ptcg_il.model.value import ValueHead


class Policy(nn.Module):
    """End-to-end imitation-learning policy: embed -> encode -> pointer + value.

    Parameters
    ----------
    V : int
        Card vocab size.
    A : int
        Attack vocab size.
    D : int
        Model dimension (256).
    heads : int
        Attention heads (8).
    layers : int
        Encoder layers (4).
    ff : int
        Feed-forward hidden dim (1024).
    card_static_table : Tensor[V, 52] or None
    attack_static_table : Tensor[A, 14] or None
    """

    def __init__(
        self,
        V: int,
        A: int,
        D: int = 256,
        heads: int = 8,
        layers: int = 4,
        ff: int = 1024,
        card_static_table: torch.Tensor | None = None,
        attack_static_table: torch.Tensor | None = None,
    ):
        super().__init__()
        self.embed = TokenEmbedder(V, D, card_static_table)
        self.encoder = Encoder(D, heads, layers, ff)
        self.pointer = PointerHead(A, D, heads, attack_static_table)
        self.pointer.card = self.embed.card
        self.value = ValueHead(D)
        self.D = D

    def forward(self, x: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-select forward pass.

        For multi-select decisions, use ``multiselect_ce`` (training) or
        ``select_multi`` (inference) instead.

        Parameters
        ----------
        x : dict
            Featurizer tensor dict with batch dim B.

        Returns
        -------
        logits : Tensor[B, O_MAX]
            Per-option logits (masked with -1e9 for padding options).
        value : Tensor[B]
            Predicted outcome in (-1, 1).
        """
        rows = self.embed(x)                                  # [B, L, D]
        h = self.encoder(rows, x["tok_mask"])                 # [B, L, D]
        logits, _ = self.pointer(h, x["tok_mask"], self.embed.card, x)  # [B, O]
        value = self.value(h[:, 0])                           # [B]
        return logits, value


def multiselect_ce(
    policy: Policy,
    x: dict[str, torch.Tensor],
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Teacher-forced cross-entropy for multi-select decisions (Appendix B.7).

    Re-scores after each pick, adding the true option's representation into
    ``extra_ctx`` and masking chosen options.  Handles variable pick counts
    via ``action_len``.

    Parameters
    ----------
    policy : Policy
        The policy module (embed/encoder/pointer must be accessible).
    x : dict
        Featurizer tensor dict.  Must contain multi-select labels
        (maxCount > 1).
    label_smoothing : float
        Smoothing factor for CE.

    Returns
    -------
    ce_per_sample : Tensor[B]
        Per-sample sum of per-pick cross-entropies (not reduced).
        Caller should weight (e.g. ``batch["sample_weight"] * ce``) and
        ``.mean()`` to get the scalar training loss.
    """
    B = x["tok_type"].shape[0]
    D = policy.D
    device = x["tok_type"].device

    rows = policy.embed(x)
    h = policy.encoder(rows, x["tok_mask"])

    action_idx = x["action_idx"]                          # [B, O_MAX]
    action_len = x["action_len"]                          # [B]  int64
    picked_mask = x["opt_mask"].clone()                   # [B, O_MAX]
    picked_ctx = torch.zeros(B, D, device=device)

    batch_max = int(action_len.max().item())

    total_ce = torch.zeros(B, device=device)

    for t in range(batch_max):
        logits, o = policy.pointer(h, x["tok_mask"], policy.embed.card, x, extra_ctx=picked_ctx)

        target = action_idx[:, t]                           # [B]
        valid = target >= 0                                 # [B] bool

        if valid.any():
            # Lazy import to avoid circular dependency with train.loop
            from ptcg_il.train.loop import masked_label_smoothed_ce
            valid_idx = torch.where(valid)[0]
            ce = masked_label_smoothed_ce(
                logits[valid_idx], target[valid_idx].clamp(min=0),
                picked_mask[valid_idx],
                label_smoothing=label_smoothing,
            )
            total_ce[valid_idx] = total_ce[valid_idx] + ce

            # Update picked context with true option repr (only where valid)
            valid_idx = torch.where(valid)[0]
            update = o[valid_idx, target[valid_idx]]        # [N_valid, D]
            picked_ctx[valid_idx] = picked_ctx[valid_idx] + update

            # Mask chosen option
            picked_mask[valid_idx, target[valid_idx]] = False

    return total_ce


def _select_multi_raw(
    pointer: PointerHead,
    h: torch.Tensor,
    tok_mask: torch.Tensor,
    card_enc: nn.Module,
    x: dict[str, torch.Tensor],
    minC: torch.Tensor,
    maxC: torch.Tensor,
) -> torch.Tensor:
    """Low-level greedy AR multi-select inference (Appendix B.7).

    v1 assumes fixed length per sample (most multi-selects have minC==maxC)
    and loops ``batch_max`` times.  Per-sample maxC tensors ensure
    correctness for heterogeneous batches (I2 fix).

    Parameters
    ----------
    pointer : PointerHead
    h : Tensor[B, L, D]
        Encoded state.
    tok_mask : bool Tensor[B, L]
    card_enc : CardEncoder
    x : dict
        Option tensors.
    minC : int64 Tensor[B]
        Minimum picks per sample (unused in v1 when minC==maxC).
    maxC : int64 Tensor[B]
        Maximum picks per sample.

    Returns
    -------
    chosen : Tensor[B, batch_max]
        Chosen option indices, -1-padded beyond per-sample maxC.
    """
    B = h.shape[0]
    D = h.shape[-1]
    batch_max = int(maxC.max().item())
    chosen_list: list[torch.Tensor] = []
    picked_mask = x["opt_mask"].clone()                         # [B, O_MAX]
    picked_ctx = torch.zeros(B, D, device=h.device)

    for t in range(batch_max):
        logits, o = pointer(h, tok_mask, card_enc, x, extra_ctx=picked_ctx)
        logits = logits.masked_fill(~picked_mask, -1e9)
        j = logits.argmax(-1)                                   # [B]
        chosen_list.append(j)

        # Only update picked state for samples that still need picks
        still_active = t < maxC                                 # [B] bool
        if still_active.any():
            active_idx = torch.where(still_active)[0]
            picked_mask[active_idx, j[active_idx]] = False
            picked_ctx[active_idx] = (
                picked_ctx[active_idx] + o[active_idx, j[active_idx]]
            )

    chosen = torch.stack(chosen_list, dim=1)  # [B, batch_max]
    # Mask out picks beyond per-sample maxC with -1 padding
    arange = torch.arange(batch_max, device=chosen.device).unsqueeze(0)  # [1, batch_max]
    valid = arange < maxC.unsqueeze(1)  # [B, batch_max]
    chosen = torch.where(valid, chosen, torch.full_like(chosen, -1))
    return chosen


def select_multi(
    policy: Policy,
    x: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Greedy autoregressive multi-select inference (Appendix B.7).

    High-level wrapper matching the spec: ``select_multi(policy, feat_dict)
    -> [B, batch_max]``.  Encodes the featurizer dict, then delegates to the
    low-level AR loop ``_select_multi_raw`` with per-sample minCount/maxCount
    tensors for heterogeneous-batch correctness.

    Parameters
    ----------
    policy : Policy
        The full policy module.
    x : dict
        Featurizer tensor dict.  Must contain ``tok_mask`` and per-sample
        ``minCount`` / ``maxCount`` tensors.

    Returns
    -------
    chosen : Tensor[B, batch_max]
        Chosen option indices (greedy), -1-padded beyond per-sample maxC.
    """
    rows = policy.embed(x)
    h = policy.encoder(rows, x["tok_mask"])
    return _select_multi_raw(
        policy.pointer, h, x["tok_mask"], policy.embed.card, x,
        minC=x["minCount"], maxC=x["maxCount"],
    )
