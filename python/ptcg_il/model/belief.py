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

L_LOG_MAX = 32
LOG_FEAT_DIM = 6  # log_type, player_rel, card_id, area_from, area_to, scalar


class BeliefModule(nn.Module):
    """GRU-based belief encoder over game log entries.

    Each log entry is a compact 6-dim feature vector.  The module embeds the
    discrete fields (log_type, card_id) and feeds the concatenated features
    through a GRU.  The final hidden state is the belief vector.

    Parameters
    ----------
    V : int
        Card vocab size (for card_id embedding, shared with Policy).
    D : int
        Model dimension (256).
    """

    def __init__(self, V: int, D: int = 256):
        super().__init__()
        self.log_type_emb = nn.Embedding(24, 16)       # LogType 0..23 → 16 dims
        self.card_emb: nn.Embedding | None = None       # bound externally to shared CardEncoder
        self.area_emb = nn.Embedding(7, 4)              # 6 areas + 1 for None(-1→6)
        self.in_proj = nn.Linear(16 + D + 4 + 4 + 1 + 1, D)  # total → D
        self.gru = nn.GRU(D, D, num_layers=1, batch_first=True)

    def forward(
        self,
        log_feat: torch.Tensor,
        log_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode log sequence into a belief state.

        Parameters
        ----------
        log_feat : Tensor[B, L_MAX, 6]
            Raw log features: [log_type, player_rel, card_id, area_from,
            area_to, scalar].
        log_mask : bool Tensor[B, L_MAX]
            True = valid log entry.

        Returns
        -------
        belief : Tensor[B, D]
            Final GRU hidden state (belief vector).
        """
        B, L, _ = log_feat.shape
        device = log_feat.device

        lt = log_feat[:, :, 0].long()                                    # [B, L]
        player_rel = log_feat[:, :, 1:2]                                 # [B, L, 1]
        card_id = log_feat[:, :, 2].long()                               # [B, L]
        area_from = log_feat[:, :, 3].long().clamp(-1, 5) + 1            # [B, L], -1→0
        area_to = log_feat[:, :, 4].long().clamp(-1, 5) + 1              # [B, L], -1→0
        scalar = log_feat[:, :, 5:6]                                     # [B, L, 1]

        # Embed discrete fields
        lt_emb = self.log_type_emb(lt)                                   # [B, L, 16]
        if self.card_emb is not None:
            card_emb = self.card_emb(card_id)                            # [B, L, D]
        else:
            card_emb = torch.zeros(B, L, self.in_proj.weight.shape[0] // 4,
                                   device=device)  # fallback (should not happen)
        area_f_emb = self.area_emb(area_from)                            # [B, L, 4]
        area_t_emb = self.area_emb(area_to)                              # [B, L, 4]

        # Concatenate all features
        x = torch.cat([lt_emb, card_emb, area_f_emb, area_t_emb,
                       player_rel, scalar], dim=-1)                      # [B, L, D_in]
        x = self.in_proj(x)                                              # [B, L, D]

        # GRU over sequence (handle empty sequences)
        lengths = log_mask.sum(dim=1).clamp(min=1)                       # [B]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_last = self.gru(packed)                                     # [1, B, D]
        belief = h_last.squeeze(0)                                       # [B, D]

        return belief


PAD_INDEX = 0  # featurizer.PAD_CARD; duplicated to keep this module import-light


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

    Output projections are **tied to the shared CardEncoder** rather than being
    a free ``Linear(D, V)``.  The card representation already fuses the id
    embedding with static features (HP, type, stage, ...), so tying lets a card
    that never appeared in a training deck still receive sensible probability
    from its stats, and it keeps the head's output space aligned with the
    encoder's input space.
    """

    def __init__(self, V: int, D: int, n_arch: int, card_emb: nn.Module | None = None):
        super().__init__()
        self.V = V
        self.D = D
        self.n_arch = n_arch
        # Assigned by Policy to the shared CardEncoder; kept optional so the
        # module is testable standalone.
        self.card_emb = card_emb

        self.arch_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, max(n_arch, 1))
        )
        # One projection per card distribution, each scoring against the tied
        # card matrix.  Separate projections (rather than one shared) because
        # "what is in their deck" and "what is in their hand right now" are
        # genuinely different questions over the same card space.
        self.deck_proj = nn.Linear(D, D)
        self.hidden_proj = nn.Linear(D, D)
        self.hand_proj = nn.Linear(D, D)
        self.card_bias = nn.Parameter(torch.zeros(3, V))

    def _card_matrix(self, device: torch.device) -> torch.Tensor:
        """[V, D] representation of every vocab card, via the shared encoder."""
        ids = torch.arange(self.V, device=device)
        if self.card_emb is None:
            raise RuntimeError(
                "BeliefHeads.card_emb was never assigned; Policy must point it "
                "at the shared CardEncoder before the first forward pass."
            )
        return self.card_emb(ids)

    def forward(self, cls: torch.Tensor) -> dict[str, torch.Tensor]:
        """``cls`` is [B, D]; returns logits for each head."""
        cards = self._card_matrix(cls.device)                      # [V, D]
        # Scale like attention: without it the logit variance grows with D and
        # the softmax saturates before the head has learned anything.
        scale = self.D ** -0.5

        def score(proj: nn.Linear, which: int) -> torch.Tensor:
            logits = (proj(cls) @ cards.t()) * scale + self.card_bias[which]
            # PAD is not a card.  The targets place zero mass on it, so leaving
            # it unmasked would only teach the head to push PAD's logit down.
            logits[:, PAD_INDEX] = float("-inf")
            return logits

        return {
            "arch": self.arch_head(cls),                           # [B, n_arch]
            "deck": score(self.deck_proj, 0),                      # [B, V]
            "hidden": score(self.hidden_proj, 1),                  # [B, V]
            "hand": score(self.hand_proj, 2),                      # [B, V]
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
