"""BeliefModule — GRU over game logs for opponent-state inference.

Processes the sequence of ``Log`` entries since the last decision to build a
belief state about the opponent's hidden information (hand contents, deck
composition, prize cards).

:class:`BeliefHeads` turns that implicit belief into *explicit, supervised*
predictions of the opponent's cards, trained against the labels in
``ptcg_il.belief_labels``.  Two things consume them:

* the MCTS planner, which currently determinizes the opponent's deck by
  assuming a mirror of our own (``ptcg_search/src/guessing.rs``); and
* the policy itself, since before these heads existed the belief GRU had no
  supervision at all and only learned whatever the action-imitation gradient
  happened to teach it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptcg_il.model.cards import F_CARD

L_LOG_MAX = 32
LOG_FEAT_DIM = 6  # log_type, player_rel, card_id, area_from, area_to, scalar

#: Highest ``cg.api.AreaType`` value (LOOKING=12).  Areas are encoded as
#: ``area + 1`` so that "no area" (-1) lands on 0, giving MAX_AREA + 2 rows.
MAX_AREA = 12
N_AREA_EMB = MAX_AREA + 2  # 0 = none, 1..13 = AreaType 0..12


class BeliefModule(nn.Module):
    """GRU-based belief encoder over game log entries.

    Each log entry carries card features (F_CARD-dim) directly — no vocab
    remapping needed.  The module embeds log_type and areas, then feeds
    everything through a GRU.

    Parameters
    ----------
    D : int
        Model dimension (256).
    """

    def __init__(self, D: int = 256):
        super().__init__()
        self.log_type_emb = nn.Embedding(24, 16)       # LogType 0..23 → 16 dims
        self.card_emb = None                            # bound externally to shared CardFeaturizer
        # AreaType runs 1..12 (cg.api.AreaType), plus -1 for "no area", so the
        # encoding is area+1 over 0..13 and needs N_AREA_EMB rows.  This used to
        # be 7 rows behind a clamp(-1, 5), which silently folded BENCH(5),
        # PRIZE(6), STADIUM(7), ENERGY(8), TOOL(9), PRE_EVOLUTION(10),
        # PLAYER(11) and LOOKING(12) onto one vector -- the belief GRU could not
        # tell "drawn to hand" from "moved to prize".  Real logs reach 12.
        self.area_emb = nn.Embedding(N_AREA_EMB, 4)
        self.in_proj = nn.Linear(16 + D + 4 + 4 + 1 + 1 + F_CARD, D)  # total → D
        self.gru = nn.GRU(D, D, num_layers=1, batch_first=True)
        # `forward`'s no-card fallbacks size a zero tensor by D, which was only
        # ever a local in `__init__`.  Policy always binds `card_emb` and always
        # passes `log_card_feat`, so those branches never ran in training and
        # the NameError stayed latent.
        self.D = D

    def forward(
        self,
        log_feat: torch.Tensor,
        log_mask: torch.Tensor,
        log_card_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode log sequence into a belief state.

        Parameters
        ----------
        log_feat : Tensor[B, L_MAX, LOG_FEAT_DIM]
            Raw log features (card_id slot is ignored; use log_card_feat).
        log_mask : bool Tensor[B, L_MAX]
            True = valid log entry.
        log_card_feat : Tensor[B, L_MAX, F_CARD] or None
            Card static features at each log position (zero for no-card).

        Returns
        -------
        belief : Tensor[B, D]
            Final GRU hidden state (belief vector).
        """
        B, L, _ = log_feat.shape
        device = log_feat.device

        lt = log_feat[:, :, 0].long()                                    # [B, L]
        player_rel = log_feat[:, :, 1:2]                                 # [B, L, 1]
        area_from = log_feat[:, :, 3].long().clamp(-1, MAX_AREA) + 1     # [B, L], -1→0
        area_to = log_feat[:, :, 4].long().clamp(-1, MAX_AREA) + 1       # [B, L], -1→0
        scalar = log_feat[:, :, 5:6]                                     # [B, L, 1]

        lt_emb = self.log_type_emb(lt)                                   # [B, L, 16]
        area_f_emb = self.area_emb(area_from)                            # [B, L, 4]
        area_t_emb = self.area_emb(area_to)                              # [B, L, 4]

        # Card features — use the raw F_CARD-dim feat passed from the featurizer
        if log_card_feat is not None:
            card_emb_raw = log_card_feat                                 # [B, L, F_CARD]
            if self.card_emb is not None:
                card_emb = self.card_emb(card_emb_raw)                   # [B, L, D]
            else:
                card_emb = torch.zeros(B, L, self.D, device=device)
        else:
            card_emb = torch.zeros(B, L, self.D, device=device)
            card_emb_raw = torch.zeros(B, L, F_CARD, device=device)

        x = torch.cat([lt_emb, card_emb, area_f_emb, area_t_emb,
                       player_rel, scalar, card_emb_raw], dim=-1)        # [B, L, D_in]
        x = self.in_proj(x)                                              # [B, L, D]

        lengths = log_mask.sum(dim=1).clamp(min=1)                       # [B]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_last = self.gru(packed)                                     # [1, B, D]
        belief = h_last.squeeze(0)                                       # [B, D]

        return belief



class BeliefHeads(nn.Module):
    """Explicit predictions of the opponent's hidden cards.

    Reads the encoder's CLS token -- *after* :class:`BeliefModule` and the
    cross-turn GRU have been mixed into it -- so the heads see the visible board,
    the discard pile, hand sizes and the action log, not just the log.

    Four outputs:

    ``arch``   categorical over the retained 𝒟_opp archetypes.  The single most
               useful signal for the planner: an archetype id maps straight to a
               real 60-card decklist, replacing the mirror-deck fallback.
    ``deck``   distribution over the vocab: the opponent's full 60-card list.
    ``hidden`` distribution over the cards of that list we have *not* seen yet --
               the pool determinization must draw their deck, hand and prizes
               from.  This, not ``deck``, is what the sampler needs.
    ``hand``   distribution over the cards currently in their hand.

    The three card distributions are **softmaxes over the vocab, not
    independent per-card sigmoids.** Copy counts are not bounded by the 4-copy
    rule (basic Energy is exempt; decks in this corpus run up to 22 copies of
    one card), so per-card count buckets would either truncate Energy or spend
    most of their classes on impossible values.  A distribution has no such
    ceiling and composes directly with what the planner does: draw 60 cards from
    it and you have a decklist.

    Output projections are **tied to the shared CardFeaturizer** rather than being
    a free ``Linear(D, n_all_cards)``.  The card representation is a pure function
    of static features (HP, type, stage, ...), so tying lets every engine card
    receive sensible probability from its stats, and keeps the head's output
    space aligned with the encoder's input space.

    Parameters
    ----------
    D : int
        Model dimension.
    n_arch : int
        Number of opponent archetypes.
    n_all_cards : int
        Total number of engine cards (for the card matrix).
    """

    def __init__(self, D: int, n_arch: int, n_all_cards: int,
                 card_emb: nn.Module | None = None):
        super().__init__()
        self.D = D
        self.n_arch = n_arch
        self.n_all_cards = n_all_cards
        # Assigned by Policy to the shared CardFeaturizer; kept optional so the
        # module is testable standalone.
        self.card_emb = card_emb

        self.arch_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, max(n_arch, 1))
        )
        self.deck_proj = nn.Linear(D, D)
        self.hidden_proj = nn.Linear(D, D)
        self.hand_proj = nn.Linear(D, D)
        self.card_bias = nn.Parameter(torch.zeros(3, max(n_all_cards, 1)))

    def _card_matrix(self, device: torch.device,
                      all_card_feat: torch.Tensor | None = None) -> torch.Tensor:
        """[n_all_cards, D] representation of every engine card.

        *all_card_feat* must be a precomputed ``[n_all_cards, F_CARD]`` tensor
        of static features for every card the engine knows about, registered
        as a buffer or passed explicitly.
        """
        if all_card_feat is not None:
            if self.card_emb is None:
                raise RuntimeError(
                    "BeliefHeads.card_emb was never assigned; Policy must "
                    "point it at the shared CardFeaturizer."
                )
            return self.card_emb(all_card_feat)
        # Fallback for old code paths (should not happen)
        raise RuntimeError(
            "BeliefHeads._card_matrix requires all_card_feat; "
            "register it via set_all_card_feat() after model construction."
        )

    def set_all_card_feat(self, all_card_feat: torch.Tensor) -> None:
        """Register the [n_all_cards, F_CARD] feature matrix as a persistent buffer."""
        self.register_buffer("all_card_feat", all_card_feat.to(torch.float32))

    def forward(self, cls: torch.Tensor) -> dict[str, torch.Tensor]:
        """``cls`` is [B, D]; returns logits for each head."""
        if not hasattr(self, "all_card_feat") or self.all_card_feat is None:
            raise RuntimeError(
                "BeliefHeads.all_card_feat not set; call set_all_card_feat() "
                "after model construction."
            )
        cards = self._card_matrix(cls.device, all_card_feat=self.all_card_feat)  # [V, D]
        scale = self.D ** -0.5

        def score(proj: nn.Linear, which: int) -> torch.Tensor:
            logits = (proj(cls) @ cards.t()) * scale + self.card_bias[which]
            logits[:, 0] = float("-inf")  # PAD is not a card
            return logits

        return {
            "arch": self.arch_head(cls),                           # [B, n_arch]
            "deck": score(self.deck_proj, 0),                      # [B, n_all_cards]
            "hidden": score(self.hidden_proj, 1),                  # [B, n_all_cards]
            "hand": score(self.hand_proj, 2),                      # [B, n_all_cards]
        }


def soft_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Cross-entropy against a *distribution* target, masked by ``weight``.

    ``target`` rows are expected to sum to 1 (or to 0 for masked-out rows).
    Rows are weighted, summed, and divided by the total weight, so a batch in
    which every row is masked returns exactly 0 instead of NaN -- which matters
    because the hand label is absent for ~7% of decision points and would
    otherwise poison the whole loss.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    # target is 0 wherever it is masked, so the product is 0 there too; but
    # logp can be -inf at PAD, and 0 * -inf is NaN.  Zero those terms out.
    per_row = -(target * torch.nan_to_num(logp, neginf=0.0)).sum(-1)
    if weight is None:
        return per_row.mean()
    weight = weight.to(per_row.dtype)
    total = weight.sum()
    if total <= 0:
        return per_row.new_zeros(())
    return (per_row * weight).sum() / total


def belief_loss(
    preds: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    w_arch: float = 1.0,
    w_deck: float = 1.0,
    w_hidden: float = 1.0,
    w_hand: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted belief loss plus per-term scalars for logging.

    ``labels`` carries the densified distributions and validity masks produced
    by ``ptcg_il.belief_labels`` (see ``densify``).  Each term is masked
    independently: ``bel_valid`` gates deck/hidden/arch, and ``bel_hand_valid``
    additionally gates the hand term.
    """
    valid = labels["bel_valid"].float()
    parts: dict[str, float] = {}
    total = preds["deck"].new_zeros(())

    if w_deck:
        l_deck = soft_cross_entropy(preds["deck"], labels["bel_deck"], valid)
        total = total + w_deck * l_deck
        parts["belief/deck_ce"] = float(l_deck.detach())
    if w_hidden:
        l_hid = soft_cross_entropy(preds["hidden"], labels["bel_hidden"], valid)
        total = total + w_hidden * l_hid
        parts["belief/hidden_ce"] = float(l_hid.detach())
    if w_hand:
        hand_w = valid * labels["bel_hand_valid"].float()
        l_hand = soft_cross_entropy(preds["hand"], labels["bel_hand"], hand_w)
        total = total + w_hand * l_hand
        parts["belief/hand_ce"] = float(l_hand.detach())
    if w_arch and preds["arch"].shape[-1] > 1:
        # -1 marks an opponent whose archetype is outside the retained 𝒟_opp
        # set; ignore_index drops those rows rather than inventing a class.
        arch_t = labels["bel_arch"].long()
        arch_t = torch.where(valid > 0, arch_t, torch.full_like(arch_t, -1))
        l_arch = F.cross_entropy(
            preds["arch"].float(), arch_t, ignore_index=-1, reduction="mean"
        )
        if torch.isfinite(l_arch):
            total = total + w_arch * l_arch
            parts["belief/arch_ce"] = float(l_arch.detach())

    return total, parts
