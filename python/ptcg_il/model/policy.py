"""Policy — end-to-end model + multi-select helpers (Appendix B.6–B.8).

``Policy.forward`` returns single-select logits.  ``multiselect_ce`` handles
teacher-forced AR training for multi-select decisions.  ``select_multi``
handles greedy AR inference.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptcg_il.model.belief import BeliefHeads, BeliefModule
from ptcg_il.model.embed import TokenEmbedder
from ptcg_il.model.encoder import Encoder
from ptcg_il.model.pointer import PointerHead
from ptcg_il.model.value import ValueHead

logger = logging.getLogger(__name__)

#: Featurizer widths that determine input layer shapes.  These are module-level
#: constants in ``ptcg_il.featurizer`` baked into ``nn.Linear`` shapes at
#: construction (``cards.py`` ``MLP(F_CARD, D, D)``, ``embed.py``
#: ``MLP(F_POKE, ...)``, ``pointer.py`` ``nn.Linear(D + F_OPT, D)``), so editing
#: the featurizer silently redefines what a checkpoint's weights mean.  Recorded
#: in ``Policy.config`` and checked by :func:`policy_from_config`, for the same
#: reason ``ptcg_il.deck`` pins ``vocab_sha1``/``archetypes_sha1``.
FEATURE_DIM_KEYS: tuple[str, ...] = (
    "F_CARD", "F_ATK", "F_POKE", "F_HAND", "F_SUM", "F_GLOBAL", "F_OPT",
)


def current_feature_dims() -> dict[str, int]:
    """Snapshot the live ``ptcg_il.featurizer`` widths."""
    from ptcg_il import featurizer as _fz

    return {k: int(getattr(_fz, k)) for k in FEATURE_DIM_KEYS}


class Policy(nn.Module):
    """End-to-end imitation-learning policy: embed -> encode -> pointer + value.

    Cross-turn history is handled by ``self.history_gru`` which processes the
    CLS token.  Pass ``history_h`` to ``forward`` for sequential inference;
    omit it (or pass None) during independent-sample training.

    Cards and attacks are represented purely by their static features (no
    learned id embeddings), so the model generalises zero-shot to any card
    the engine knows about.

    Parameters
    ----------
    D : int
        Model dimension (256).
    heads : int
        Attention heads (8).
    layers : int
        Encoder layers (4).
    ff : int
        Feed-forward hidden dim (1024).
    n_opp_arch : int
        Number of opponent archetypes for belief head.
    n_all_cards : int
        Total number of engine cards (for belief card matrix).
    """

    def __init__(
        self,
        D: int = 256,
        heads: int = 8,
        layers: int = 4,
        ff: int = 1024,
        n_opp_arch: int = 0,
        n_all_cards: int = 0,
        all_card_feat: torch.Tensor | None = None,
    ):
        super().__init__()
        self.embed = TokenEmbedder(D)
        self.encoder = Encoder(D, heads, layers, ff)
        self.pointer = PointerHead(D, heads)
        self.pointer.card = self.embed.card
        self.value = ValueHead(D)
        self.belief = BeliefModule(D)
        self.belief.card_emb = self.embed.card  # share CardFeaturizer
        self.belief_heads = BeliefHeads(D, n_opp_arch, n_all_cards,
                                         card_emb=self.embed.card)
        if all_card_feat is not None:
            self.belief_heads.set_all_card_feat(all_card_feat)
        self.n_opp_arch = n_opp_arch
        self.history_gru = nn.GRUCell(D, D)  # cross-turn memory on CLS token
        self.D = D
        self.config: dict[str, int] = {
            "D": D,
            "heads": heads,
            "layers": layers,
            "ff": ff,
            "n_opp_arch": n_opp_arch,
            "n_all_cards": n_all_cards,
            "feat_dims": current_feature_dims(),
            "seed": 42,  # placeholder; set by caller after construction
        }

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
            belief = self.belief(
                x["log_feat"], x["log_mask"],
                log_card_feat=x.get("log_card_feat"),
            )  # [B, D]
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

    def belief_logits(
        self, x: dict[str, torch.Tensor], history_h: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Opponent-card belief logits: ``arch``, ``deck``, ``hidden``, ``hand``.

        Separate from :meth:`forward` because the two have different callers:
        training needs the belief heads alongside the action logits and gets
        them from :meth:`forward_with_belief`, which encodes once, whereas the
        MCTS planner needs *only* the belief and would otherwise pay for the
        pointer head on every determinization.
        """
        h, _ = self._encode(x, history_h)
        return self.belief_heads(h[:, 0])

    def forward_with_belief(
        self, x: dict[str, torch.Tensor], history_h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """:meth:`forward` plus the belief logits, sharing a single encode pass.

        Returns ``(logits, value, history_h, belief)``.  Calling ``forward`` and
        ``belief_logits`` separately would run the transformer twice per step.
        """
        h, history_h = self._encode(x, history_h)
        logits, _ = self.pointer(h, x["tok_mask"], self.embed.card, x)
        value = self.value(h[:, 0])
        belief = self.belief_heads(h[:, 0])
        return logits, value, history_h.detach(), belief


#: The belief archetype classifier's output layer.  Its width is
#: ``len(archetypes.json["opp_ids"])``, which grows when a seeded mining run
#: appends a new 𝒟_opp archetype (``ptcg_mine.archetype._append_only``).
_ARCH_HEAD_PREFIX = "belief_heads.arch_head."


def widen_belief_arch_head(policy: Policy, state_dict: dict) -> tuple[dict, int, int]:
    """Copy an older, narrower archetype head into this policy's wider one.

    ``𝒟_opp`` is append-only, so slot *i* means the same archetype it always
    did and the old head's rows are a strict prefix of the new one's.  Copying
    them keeps everything the old model learned about the archetypes it saw and
    leaves the appended rows at their initialisation.

    This is only sound *because* the ordering is append-only.  Against a
    re-baselined ``archetypes.json`` the prefix rows describe different decks,
    and a prefix copy would be worse than a fresh head — it would look trained.
    Callers must therefore opt in, and only when the artifacts share a lineage
    generation with the checkpoint.

    Returns ``(patched_state_dict, old_width, new_width)``; the dict is
    unchanged and the widths equal when there is nothing to widen.  Raises if
    the head *shrank*, which means a re-cluster, not an append.
    """
    model_sd = policy.state_dict()
    old_w = new_w = 0
    patched = state_dict
    for key, want in model_sd.items():
        if not key.startswith(_ARCH_HEAD_PREFIX):
            continue
        have = state_dict.get(key)
        if have is None or have.shape == want.shape:
            continue
        if have.shape[1:] != want.shape[1:]:
            raise RuntimeError(
                f"{key}: checkpoint shape {list(have.shape)} differs from "
                f"{list(want.shape)} in a dimension that is not the archetype "
                "count — this is not an appended 𝒟_opp slot."
            )
        if have.shape[0] > want.shape[0]:
            raise RuntimeError(
                f"{key}: checkpoint has {have.shape[0]} archetype slots but "
                f"this policy has {want.shape[0]}. 𝒟_opp shrank, which means "
                "the archetypes were re-clustered rather than appended to. The "
                "old rows now describe different decks; retrain the belief "
                "head instead of copying them."
            )
        if patched is state_dict:
            patched = dict(state_dict)
        grown = want.clone()
        grown[: have.shape[0]] = have
        patched[key] = grown
        old_w, new_w = int(have.shape[0]), int(want.shape[0])
    return patched, old_w, new_w


def load_policy_state(
    policy: Policy,
    state_dict: dict,
    allow_belief_widening: bool = False,
) -> list[str]:
    """``policy.load_state_dict`` that tolerates a pre-belief checkpoint.

    Every checkpoint written before :class:`~ptcg_il.model.belief.BeliefHeads`
    existed lacks the ``belief_heads.*`` parameters, and a strict load rejects
    it outright.  Those are the *only* keys allowed to be missing: anything
    else still raises, because a silently half-loaded policy evaluates as a
    plausible-looking but randomly-initialised model.

    With *allow_belief_widening*, a checkpoint whose archetype head is narrower
    than this policy's is accepted and its rows copied into the leading slots —
    see :func:`widen_belief_arch_head` for the condition that makes that sound.
    It is off by default: a shape mismatch is the only signal that the 𝒟_opp
    set moved, and swallowing it by default would let a re-baselined corpus load
    a checkpoint whose belief slots mean different decks.

    Returns the list of belief keys that were left at their initial values.
    """
    if allow_belief_widening:
        state_dict, old_w, new_w = widen_belief_arch_head(policy, state_dict)
        if new_w > old_w > 0:
            logger.warning(
                "belief archetype head widened %d → %d slots; rows 0..%d were "
                "copied from the checkpoint and %d appended slot(s) start "
                "untrained. This is only correct if archetypes.json was seeded, "
                "not re-baselined.",
                old_w, new_w, old_w - 1, new_w - old_w,
            )
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    belief_missing = [k for k in missing if k.startswith("belief_heads.")]
    other = [k for k in missing if not k.startswith("belief_heads.")]

    # Policy ties one CardEncoder into three places (pointer.card, belief.card_emb,
    # belief_heads all reference self.embed.card), and EMA shadows built via
    # named_parameters() (remove_duplicate=True, the default) omit the alias
    # names.  The tensors are already loaded through embed.card.* — compare object
    # identity rather than guessing at name prefixes.
    if other:
        params = dict(policy.named_parameters(remove_duplicate=False))
        params.update(dict(policy.named_buffers(remove_duplicate=False)))
        loaded_ids = {id(params[k]) for k in state_dict if k in params}
        other = [k for k in other
                 if k not in params or id(params[k]) not in loaded_ids]

    if other or unexpected:
        raise RuntimeError(
            f"checkpoint does not match this Policy: missing={other}, "
            f"unexpected={list(unexpected)}"
        )
    return belief_missing


def policy_from_config(config: dict,
                       all_card_feat: torch.Tensor | None = None) -> Policy:
    """Rebuild a :class:`Policy` from a checkpoint's ``config`` record.

    Supports both old configs (with ``V``/``A`` from the id_emb era) and new
    configs (pure-feature model).  ``V``/``A`` are ignored — the pure-feature
    model does not need them.
    """
    required = ["D", "heads", "layers", "ff"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(
            f"checkpoint config is missing {missing}; it predates Policy.config "
            "and the policy must be built from artifact sizes instead"
        )

    # A featurizer edit changes every input width at once, and the resulting
    # load failure names layer shapes rather than the cause.  Checkpoints from
    # before this record was added carry no feat_dims and are let through --
    # they fail later on shape, as they always did.
    recorded = config.get("feat_dims")
    if recorded:
        live = current_feature_dims()
        bad = {k: (int(v), live[k]) for k, v in recorded.items()
               if k in live and int(v) != live[k]}
        if bad:
            detail = ", ".join(
                f"{k}: checkpoint {was}, current {now}" for k, (was, now) in sorted(bad.items())
            )
            raise ValueError(
                f"checkpoint was trained with different featurizer widths ({detail}). "
                "ptcg_il/featurizer.py changed since it was written, so its weights "
                "no longer mean what the current features mean. Retrain, or check out "
                "the featurizer generation that produced it."
            )

    return Policy(
        D=int(config["D"]),
        heads=int(config["heads"]),
        layers=int(config["layers"]),
        ff=int(config["ff"]),
        n_opp_arch=int(config.get("n_opp_arch", 0)),
        n_all_cards=int(config.get("n_all_cards", 0)),
        all_card_feat=all_card_feat,
    )


def multiselect_ce(
    policy: Policy,
    x: dict[str, torch.Tensor],
    label_smoothing: float = 0.05,
    group_marginal: bool = True,
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
    from ptcg_il.train.loop import masked_label_smoothed_ce, target_group_mask

    opt_group = x.get("opt_group")
    use_groups = group_marginal and opt_group is not None

    for t in range(batch_max):
        logits, o = policy.pointer(
            h, x["tok_mask"], policy.embed.card, x, msgru_h=msgru_h,
        )

        # Mask STOP column for samples that haven't reached minCount yet.
        #
        # The suppression must land in the *mask*, not only in the logits.
        # `masked_label_smoothed_ce` spreads the smoothing mass uniformly over
        # positions the mask calls valid, so a column left valid while carrying
        # a -1e9 logit contributes log_prob = -1e9 to that average — roughly
        # eps * 1e9 / n_valid per step (2.5e7 on a real batch, against ~9).  It
        # stays finite, so it shows up as a loss in the millions rather than as
        # a NaN.  step_mask is per-step because picked_mask persists across the
        # AR loop, and STOP becomes legal again at t >= minCount.
        t_tensor = torch.tensor(t, device=device)
        stop_forbidden = (t_tensor < min_count) & (stop_column >= 0)  # [B]
        step_mask = picked_mask
        if stop_forbidden.any():
            fb_idx = torch.where(stop_forbidden)[0]
            fb_col = stop_column[fb_idx].clamp(min=0)
            logits[fb_idx, fb_col] = -1e9
            step_mask = picked_mask.clone()
            step_mask[fb_idx, fb_col] = False

        target = action_idx[:, t]                           # [B]
        valid = target >= 0                                 # [B] bool

        if valid.any():
            valid_idx = torch.where(valid)[0]
            tgt = target[valid_idx].clamp(min=0)
            # Intersect with step_mask, not opt_mask: a group member already
            # picked this step is no longer an alternative to the target.
            tg = (
                target_group_mask(opt_group[valid_idx], tgt, step_mask[valid_idx])
                if use_groups else None
            )
            ce = masked_label_smoothed_ce(
                logits[valid_idx], tgt, step_mask[valid_idx],
                label_smoothing=label_smoothing,
                target_group=tg,
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
    pointers: list[PointerHead] | None = None,
    h_list: list[torch.Tensor] | None = None,
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

    # Ensemble mode: pointers[i] paired with h_list[i].
    is_ensemble = pointers is not None and h_list is not None
    if is_ensemble:
        if len(pointers) != len(h_list):
            raise ValueError(
                f"pointers and h_list must have same length, "
                f"got {len(pointers)} and {len(h_list)}"
            )
        N = len(pointers)
    else:
        N = 1

    # Multi-select GRU hidden state(s) — fp32, GRU runs outside autocast
    if is_ensemble:
        msgru_h_list = [torch.zeros(B, D, device=device, dtype=torch.float32) for _ in range(N)]
    else:
        msgru_h = torch.zeros(B, D, device=device, dtype=torch.float32)

    # Track which samples are still picking
    active = torch.ones(B, dtype=torch.bool, device=device)

    for t in range(batch_max):
        if is_ensemble:
            # Each member computes logits independently
            all_logits: list[torch.Tensor] = []
            all_o: list[torch.Tensor] = []
            for i in range(N):
                li, oi = pointers[i](h_list[i], tok_mask, card_enc, x, msgru_h=msgru_h_list[i])
                # Mask STOP column for samples that haven't reached minCount yet
                if stop_column is not None:
                    t_tensor = torch.tensor(t, device=device)
                    stop_forbidden = active & (t_tensor < minC) & (stop_column >= 0)
                    if stop_forbidden.any():
                        fb_idx = torch.where(stop_forbidden)[0]
                        li[fb_idx, stop_column[fb_idx].clamp(min=0)] = -1e9
                # Mask already-picked options
                li = li.masked_fill(~picked_mask, -1e9)
                all_logits.append(li)
                all_o.append(oi)
            # Average probabilities → argmax
            probs = torch.stack([F.softmax(li, dim=-1) for li in all_logits], dim=0).mean(dim=0)
            j = probs.argmax(-1)                                   # [B]
        else:
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
            # Update GRU(s) with chosen option repr (fp32, outside autocast)
            with torch.amp.autocast(device.type if device.type in ("cuda", "cpu") else "cpu", enabled=False):
                if is_ensemble:
                    for i in range(N):
                        msgru_h_list[i][active_idx] = pointers[i].msgru(
                            all_o[i][active_idx, j[active_idx]].float(),
                            msgru_h_list[i][active_idx],
                        )
                else:
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
    # EnsemblePolicy defines its own select_multi; dispatch to it.
    if hasattr(policy, 'select_multi'):
        return policy.select_multi(x, history_h)

    h, _history_h = policy._encode(x, history_h)
    stop_col = x.get("stop_column")
    return _select_multi_raw(
        policy.pointer, h, x["tok_mask"], policy.embed.card, x,
        minC=x["minCount"], maxC=x["maxCount"],
        stop_column=stop_col,
    )
