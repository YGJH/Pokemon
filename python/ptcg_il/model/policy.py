"""Policy — end-to-end model + multi-select helpers (Appendix B.6–B.8).

``Policy.forward`` returns single-select logits.  ``multiselect_ce`` handles
teacher-forced AR training for multi-select decisions.  ``select_multi``
handles greedy AR inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptcg_il.model.belief import BeliefModule
from ptcg_il.model.embed import TokenEmbedder
from ptcg_il.model.encoder import Encoder
from ptcg_il.model.pointer import PointerHead
from ptcg_il.model.value import ValueHead


class Policy(nn.Module):
    """End-to-end imitation-learning policy: embed -> encode -> pointer + value.

    Cross-turn history is handled by ``self.history_gru`` which processes the
    CLS token.  Pass ``history_h`` to ``forward`` for sequential inference;
    omit it (or pass None) during independent-sample training.

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
        self.belief = BeliefModule(V, D)
        self.belief.card_emb = self.embed.card  # share CardEncoder
        self.history_gru = nn.GRUCell(D, D)  # cross-turn memory on CLS token
        self.D = D

    def _encode(self, x: dict[str, torch.Tensor], history_h: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared encode path: embed → encode → history GRU on CLS.

        Returns ``(h, history_h)`` where *h* has the history-augmented CLS token.
        """
        B = x["tok_type"].shape[0]
        device = x["tok_type"].device
        rows = self.embed(x)                                 # [B, L, D]
        h = self.encoder(rows, x["tok_mask"])                # [B, L, D]

        # Belief module: encode logs into belief state
        if "log_feat" in x and x.get("log_mask") is not None:
            belief = self.belief(x["log_feat"], x["log_mask"])  # [B, D]
        else:
            belief = torch.zeros(B, self.D, device=device)

        # Cross-turn GRU on CLS token (augmented with belief)
        cls_token = h[:, 0, :] + belief                      # [B, D]
        if history_h is None:
            history_h = torch.zeros(B, self.D, device=device)
        cls_out = self.history_gru(cls_token, history_h)     # [B, D]
        # Replace CLS without in-place mutation (preserves autograd graph)
        h = torch.cat([cls_out.unsqueeze(1), h[:, 1:, :]], dim=1)  # [B, L, D]

        return h, history_h

    def forward(self, x: dict[str, torch.Tensor],
                history_h: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-select forward pass.

        For multi-select decisions, use ``multiselect_ce`` (training) or
        ``select_multi`` (inference) instead.

        Parameters
        ----------
        x : dict
            Featurizer tensor dict with batch dim B.
        history_h : Tensor[B, D] or None
            Cross-turn GRU hidden state (None = zero-init).

        Returns
        -------
        logits : Tensor[B, O_MAX]
            Per-option logits (masked with -1e9 for padding options).
        value : Tensor[B]
            Predicted outcome in (-1, 1).
        history_h : Tensor[B, D]
            Updated cross-turn hidden state (detached).
        """
        h, history_h = self._encode(x, history_h)
        logits, _ = self.pointer(h, x["tok_mask"], self.embed.card, x)  # [B, O]
        value = self.value(h[:, 0])                           # [B]
        return logits, value, history_h.detach()


def multiselect_ce(
    policy: Policy,
    x: dict[str, torch.Tensor],
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Teacher-forced cross-entropy for multi-select decisions (Appendix B.7).

    Uses ``policy.pointer.msgru`` for sequential memory instead of a running
    sum.  A STOP column (index in ``x["stop_column"]``) is supervised after
    the last expert pick for variable-length selects.

    Parameters
    ----------
    policy : Policy
        The policy module (embed/encoder/pointer must be accessible).
    x : dict
        Featurizer tensor dict.  Must contain multi-select labels
        (maxCount > 1) and ``stop_column``.
    label_smoothing : float
        Smoothing factor for CE.

    Returns
    -------
    ce_per_sample : Tensor[B]
        Per-sample sum of per-pick cross-entropies (not reduced).
    """
    B = x["tok_type"].shape[0]
    D = policy.D
    device = x["tok_type"].device

    # Encode once (history_h unused during independent-sample training)
    h, _history_h = policy._encode(x)

    action_idx = x["action_idx"]                          # [B, O_MAX]
    action_len = x["action_len"]                          # [B]  int64
    stop_column = x.get("stop_column")
    if stop_column is None:
        stop_column = torch.full((B,), -1, dtype=torch.long, device=device)
    min_count = x["minCount"]                             # [B]  int64
    picked_mask = x["opt_mask"].clone()                   # [B, O_MAX]

    # Multi-select GRU hidden state — fp32, GRU runs outside autocast
    msgru_h = torch.zeros(B, D, device=device, dtype=torch.float32)

    batch_max = int(action_len.max().item())

    total_ce = torch.zeros(B, device=device)

    # Lazy import to avoid circular dependency with train.loop
    from ptcg_il.train.loop import masked_label_smoothed_ce

    for t in range(batch_max):
        logits, o = policy.pointer(
            h, x["tok_mask"], policy.embed.card, x, msgru_h=msgru_h,
        )

        # Mask STOP column for samples that haven't reached minCount yet
        t_tensor = torch.tensor(t, device=device)
        stop_forbidden = (t_tensor < min_count) & (stop_column >= 0)  # [B]
        if stop_forbidden.any():
            fb_idx = torch.where(stop_forbidden)[0]
            logits[fb_idx, stop_column[fb_idx].clamp(min=0)] = -1e9

        target = action_idx[:, t]                           # [B]
        valid = target >= 0                                 # [B] bool

        if valid.any():
            valid_idx = torch.where(valid)[0]
            ce = masked_label_smoothed_ce(
                logits[valid_idx], target[valid_idx].clamp(min=0),
                picked_mask[valid_idx],
                label_smoothing=label_smoothing,
            )
            total_ce[valid_idx] = total_ce[valid_idx] + ce

            # Update multi-select GRU with chosen option repr (fp32, outside autocast)
            update = o[valid_idx, target[valid_idx]].float()  # [N_valid, D]
            with torch.amp.autocast(device.type if device.type in ("cuda", "cpu") else "cpu", enabled=False):
                msgru_h[valid_idx] = policy.pointer.msgru(update, msgru_h[valid_idx])

            # Mask chosen option (but NOT the STOP column — it stays available)
            is_regular = target[valid_idx] != stop_column[valid_idx]
            if is_regular.any():
                reg_idx = valid_idx[is_regular]
                picked_mask[reg_idx, target[reg_idx]] = False

    return total_ce


def _select_multi_raw(
    pointer: PointerHead,
    h: torch.Tensor,
    tok_mask: torch.Tensor,
    card_enc: nn.Module,
    x: dict[str, torch.Tensor],
    minC: torch.Tensor,
    maxC: torch.Tensor,
    stop_column: torch.Tensor | None = None,
) -> torch.Tensor:
    """Low-level greedy AR multi-select inference (Appendix B.7).

    Uses the PointerHead's ``msgru`` for sequential memory and a STOP column
    (when provided) for variable-length selection.  The STOP column is masked
    until ``t >= minC[b]``; selecting STOP ends the picks for that sample.

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
        Minimum picks per sample.
    maxC : int64 Tensor[B]
        Maximum picks per sample.
    stop_column : int64 Tensor[B] or None
        STOP column index per sample (-1 = no STOP / single-select).

    Returns
    -------
    chosen : Tensor[B, batch_max]
        Chosen option indices, -1-padded beyond per-sample maxC.
        STOP picks are recorded as -2 (distinguishable from padding).
    """
    B = h.shape[0]
    D = h.shape[-1]
    device = h.device
    batch_max = int(maxC.max().item())
    chosen_list: list[torch.Tensor] = []
    picked_mask = x["opt_mask"].clone()                         # [B, O_MAX]

    # Multi-select GRU hidden state — fp32, GRU runs outside autocast
    msgru_h = torch.zeros(B, D, device=device, dtype=torch.float32)

    # Track which samples are still picking
    active = torch.ones(B, dtype=torch.bool, device=device)

    for t in range(batch_max):
        logits, o = pointer(h, tok_mask, card_enc, x, msgru_h=msgru_h)

        # Mask STOP column for samples that haven't reached minCount yet
        if stop_column is not None:
            t_tensor = torch.tensor(t, device=device)
            stop_forbidden = active & (t_tensor < minC) & (stop_column >= 0)
            if stop_forbidden.any():
                fb_idx = torch.where(stop_forbidden)[0]
                logits[fb_idx, stop_column[fb_idx].clamp(min=0)] = -1e9

        # Mask already-picked options (but not STOP column)
        logits = logits.masked_fill(~picked_mask, -1e9)
        j = logits.argmax(-1)                                   # [B]
        chosen_list.append(j)

        # Check for STOP selection
        if stop_column is not None:
            chose_stop = active & (j == stop_column) & (stop_column >= 0)
            if chose_stop.any():
                cs_idx = torch.where(chose_stop)[0]
                active[cs_idx] = False  # these samples are done

        # Update state for still-active samples
        still_active = active & (t < maxC)
        if still_active.any():
            active_idx = torch.where(still_active)[0]
            # Don't mask STOP — it stays available for future steps
            is_regular = j[active_idx] != (
                stop_column[active_idx] if stop_column is not None else -1
            )
            if is_regular.any():
                reg_idx = active_idx[is_regular]
                picked_mask[reg_idx, j[reg_idx]] = False
            # Update GRU with chosen option repr (fp32, outside autocast)
            with torch.amp.autocast(device.type if device.type in ("cuda", "cpu") else "cpu", enabled=False):
                msgru_h[active_idx] = pointer.msgru(
                    o[active_idx, j[active_idx]].float(), msgru_h[active_idx],
                )

    chosen = torch.stack(chosen_list, dim=1)  # [B, batch_max]
    # Mask out picks beyond per-sample maxC with -1 padding
    arange = torch.arange(batch_max, device=device).unsqueeze(0)  # [1, batch_max]
    valid = arange < maxC.unsqueeze(1)  # [B, batch_max]
    chosen = torch.where(valid, chosen, torch.full_like(chosen, -1))
    # Mark STOP picks as -2
    if stop_column is not None:
        is_stop_pick = (chosen == stop_column.unsqueeze(1)) & (stop_column.unsqueeze(1) >= 0)
        chosen = torch.where(is_stop_pick, torch.full_like(chosen, -2), chosen)
    return chosen


def select_multi(
    policy: Policy,
    x: dict[str, torch.Tensor],
    history_h: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy autoregressive multi-select inference (Appendix B.7).

    Parameters
    ----------
    policy : Policy
        The full policy module.
    x : dict
        Featurizer tensor dict.  Must contain ``tok_mask``, ``minCount``,
        ``maxCount``, and ``stop_column`` tensors.
    history_h : Tensor[B, D] or None
        Cross-turn GRU hidden state.

    Returns
    -------
    chosen : Tensor[B, batch_max]
        Chosen option indices (greedy), -1-padded beyond per-sample maxC.
        STOP picks are recorded as -2.
    """
    h, _history_h = policy._encode(x, history_h)
    stop_col = x.get("stop_column")
    return _select_multi_raw(
        policy.pointer, h, x["tok_mask"], policy.embed.card, x,
        minC=x["minCount"], maxC=x["maxCount"],
        stop_column=stop_col,
    )
