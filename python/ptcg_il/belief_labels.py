"""Ground-truth labels for the opponent-card belief model.

The MCTS planner has to *determinize* the game before it can search: the
engine's ``search_begin`` demands concrete values for the opponent's deck, hand
and prizes.  Today ``ptcg_search/src/guessing.rs`` fills those in by assuming
the opponent plays a mirror of our own deck and sampling uniformly, which is
wrong in almost every game.  This module produces the supervision needed to
replace that guess with a learned posterior.

Everything here is derived from data an *episode replay* contains but a *live
agent* cannot see, which is exactly what makes it a usable training target:

* **The opponent's 60-card decklist** is submitted as their step-0 action, so
  ``deck_of(ep, 1 - p)`` is an exact label at every single decision point.
* **The opponent's hand** is visible in *their own* observations.  Each replay
  step has exactly one ACTIVE agent, so we read the hand from the opponent's
  next decision and carry it back — see :func:`opp_hand_at_next_decision` for
  why that offset is acceptable.

Representation choice: counts are stored as a **distribution over the vocab**
(count / total) rather than per-card integer buckets.  Card multiplicity is not
bounded by the usual 4-copy rule — basic Energy is exempt, and real decks in
this corpus go up to 22 copies of one card — so a fixed set of count buckets
either truncates Energy or wastes most of its classes.  A single softmax over V
sidesteps that, and it composes directly with how the planner consumes the
output: draw 60 cards from the distribution and you have a decklist.

Storage is sparse (index/count pairs, :data:`BELIEF_K` slots).  A 60-card deck
holds at most 26 distinct cards in this corpus, so dense ``float32[V]`` arrays
would be ~97% zeros and would dominate the shard size.
"""

from __future__ import annotations

import bisect

import numpy as np

from ptcg_il.featurizer import PAD_CARD, UNKNOWN_CARD

# Sparse-slot budget for one decklist.  Corpus max is 26 distinct cards in a
# 60-card deck; 32 leaves headroom without a second pass over the data.
BELIEF_K = 32

DECK_SIZE = 60

# Default weights of the four belief loss terms.  Lives here, in the
# torch-free module, so the CLI can name them without importing torch.
# ``arch`` dominates because it is the term the planner actually consumes -- an
# archetype id maps straight to a real decklist -- while the card
# distributions are the fallback for opponents outside the retained 𝒟_opp set.
# ``hand`` is smallest: its label is the opponent's hand at their *next* turn,
# so it carries a half-turn of noise the other three do not.
BELIEF_WEIGHTS: dict[str, float] = {
    "w_arch": 1.0,
    "w_deck": 0.5,
    "w_hidden": 0.5,
    "w_hand": 0.25,
}


def _accumulate(counts: dict[int, int], index: int, n: int = 1) -> None:
    counts[index] = counts.get(index, 0) + n


def deck_counts_dense(
    deck_ids: list[int], id_to_index: dict, vocab_size: int
) -> np.ndarray:
    """int32[vocab_size] copy-count of every card in *deck_ids*.

    Out-of-vocab ids all collapse onto :data:`UNKNOWN_CARD`, matching how the
    featurizer maps them, so the model's target lives in the same index space
    as its inputs.
    """
    out = np.zeros(vocab_size, dtype=np.int32)
    for cid in deck_ids:
        idx = int(id_to_index.get(cid, UNKNOWN_CARD))
        if 0 <= idx < vocab_size:
            out[idx] += 1
    return out


def opp_visible_counts(
    state: dict, your_index: int, id_to_index: dict, vocab_size: int
) -> np.ndarray:
    """int32[vocab_size] counts of opponent cards we can currently *see*.

    Covers every area the observation reveals: their active and bench Pokémon
    (including the pre-evolution stack underneath, attached Energy and Tools),
    their discard pile, any prize that has been flipped face-up, and **the
    Stadium in play if they are the one who played it**.

    The Stadium is easy to miss because it does not live under either player in
    the observation -- it sits at ``state["stadium"]`` and names its owner via
    ``playerIndex``.  Omitting it left the hidden pool one card too large in
    31% of the decision points measured on this corpus, since a Stadium stays
    in play for many turns.

    Attached cards carry a ``playerIndex``, so Energy *we* attached to *their*
    Pokémon is not miscounted as coming from their deck.

    A **face-down active** Pokémon is deliberately *not* counted.  Its location
    is known but its identity is not, and this function's job is to remove
    cards whose identity we have learned.  It therefore stays in the hidden
    pool, which is also what ``guessing.rs`` wants -- it guesses the face-down
    active separately via ``opp_active_face_down``.

    **Measured accuracy.** Cross-checking ``hidden_total`` against the
    observation's own ``deckCount + handCount + face-down prizes`` over 2058
    decision points: 98.4% agree exactly.  The 1.6% remainder is the face-down
    active above (by design) plus ~0.3% where the transient ``state["looking"]``
    reveal zone holds cards pulled out of the deck mid-effect.  Those return to
    the deck on the following step, so they are left uncounted rather than
    special-cased.
    """
    players = state.get("players") or []
    opp_index = 1 - your_index
    if opp_index >= len(players) or players[opp_index] is None:
        return np.zeros(vocab_size, dtype=np.int32)
    opp = players[opp_index]

    counts: dict[int, int] = {}

    def add_card(card, owner_must_be: int | None = opp_index) -> None:
        if not isinstance(card, dict):
            return
        # Attached cards name their owner; only count the opponent's own cards.
        if owner_must_be is not None:
            owner = card.get("playerIndex")
            if owner is not None and int(owner) != owner_must_be:
                return
        cid = card.get("id")
        if cid is None:
            return
        _accumulate(counts, int(id_to_index.get(cid, UNKNOWN_CARD)))

    def add_pokemon(poke) -> None:
        if not isinstance(poke, dict):
            return
        add_card(poke)
        for key in ("preEvolution", "energyCards", "tools"):
            for sub in poke.get(key) or []:
                add_card(sub)

    for poke in opp.get("active") or []:
        add_pokemon(poke)
    for poke in opp.get("bench") or []:
        add_pokemon(poke)
    for card in opp.get("discard") or []:
        add_card(card)
    # Face-down prizes serialize as null; only flipped ones have an id.
    for card in opp.get("prize") or []:
        add_card(card)
    # Stadium lives on the state, not on a player; playerIndex says whose it is.
    stadium = state.get("stadium")
    if isinstance(stadium, dict):
        stadium = [stadium]
    for card in stadium or []:
        add_card(card)

    out = np.zeros(vocab_size, dtype=np.int32)
    for idx, n in counts.items():
        if 0 <= idx < vocab_size:
            out[idx] = n
    return out


def opp_hand_timeline(ep: dict, opp_player: int) -> list[tuple[int, list[int]]]:
    """Every step at which *opp_player* was ACTIVE, with their hand card-ids.

    Built once per (episode, player) and queried with
    :func:`hand_after`.  Scanning the whole replay for each decision point
    instead would be quadratic in the step count -- ~21k step visits per player
    per episode on this corpus -- which is pure waste when one pass suffices.
    """
    out: list[tuple[int, list[int]]] = []
    for t, agents in enumerate(ep.get("steps") or []):
        if opp_player >= len(agents):
            continue
        agent = agents[opp_player]
        if agent.get("status") != "ACTIVE":
            continue
        cur = (agent.get("observation") or {}).get("current")
        if not isinstance(cur, dict):
            continue
        players = cur.get("players") or []
        yi = cur.get("yourIndex")
        if yi is None or yi >= len(players) or players[yi] is None:
            continue
        hand = players[yi].get("hand")
        if not isinstance(hand, list):
            continue
        out.append(
            (t, [c["id"] for c in hand if isinstance(c, dict) and c.get("id") is not None])
        )
    return out


def hand_after(
    timeline: list[tuple[int, list[int]]], after_step: int
) -> list[int] | None:
    """First entry of *timeline* strictly after ``after_step``, else ``None``.

    ``None`` means the opponent never acts again (the game ended on our action),
    and the caller must mask the hand loss out.

    See :func:`opp_hand_at_next_decision` for why a forward-looking hand is the
    label of choice.
    """
    steps = [t for t, _ in timeline]
    i = bisect.bisect_right(steps, after_step)
    if i >= len(timeline):
        return None
    return timeline[i][1]


def opp_hand_at_next_decision(
    ep: dict, opp_player: int, after_step: int
) -> list[int] | None:
    """The opponent's hand card-ids at their first decision *after* ``after_step``.

    Convenience wrapper over :func:`opp_hand_timeline` for single lookups and
    tests.  Bulk callers should build the timeline once and use
    :func:`hand_after`.

    Returns ``None`` when the opponent never acts again (the game ends on our
    action), in which case the caller must mask the hand loss out.

    **On the temporal offset.** A replay step exposes only the ACTIVE agent's
    observation, so the opponent's hand is never available at the same step we
    act on.  The nearest exact reading is their next decision, which sits one
    half-turn ahead.  Between our decision and theirs a hand can gain the
    start-of-turn draw and lose whatever they play, so this label is a
    *slightly forward-looking* hand rather than the current one.  It is used
    anyway because the alternative -- reconstructing the hand by replaying the
    log backwards -- depends on correctly modelling every card effect, and a
    silently wrong reconstruction is worse than a label with a known,
    bounded offset.  Deck and hidden-pool labels have no such offset.
    """
    return hand_after(opp_hand_timeline(ep, opp_player), after_step)


def _to_sparse(dense: np.ndarray, k: int = BELIEF_K) -> tuple[np.ndarray, np.ndarray]:
    """Top-*k* nonzero entries of *dense* as (index[k] int16, count[k] int16).

    Unused slots are ``(PAD_CARD, 0)``.  Entries beyond *k* are dropped
    largest-first, so if the budget is ever exceeded the loss is the rarest
    singleton copies rather than the deck's core.
    """
    idx_out = np.full(k, PAD_CARD, dtype=np.int16)
    cnt_out = np.zeros(k, dtype=np.int16)
    nz = np.nonzero(dense)[0]
    if nz.size == 0:
        return idx_out, cnt_out
    if nz.size > k:
        nz = nz[np.argsort(-dense[nz], kind="stable")[:k]]
        nz = np.sort(nz)
    idx_out[: nz.size] = nz.astype(np.int16)
    cnt_out[: nz.size] = dense[nz].astype(np.int16)
    return idx_out, cnt_out


def densify(
    idx: np.ndarray, cnt: np.ndarray, vocab_size: int, normalize: bool = True
) -> np.ndarray:
    """Inverse of :func:`_to_sparse`, batched over any leading dimensions.

    With ``normalize=True`` each row is divided by its own total, producing the
    distribution the belief heads are trained against.  Rows that are entirely
    empty stay all-zero -- callers detect them via the ``*_valid`` mask rather
    than by testing the sum, so an all-zero row is never renormalized into a
    uniform distribution by accident.
    """
    idx = np.asarray(idx)
    cnt = np.asarray(cnt).astype(np.float32)
    lead = idx.shape[:-1]
    flat_idx = idx.reshape(-1, idx.shape[-1])
    flat_cnt = cnt.reshape(-1, cnt.shape[-1])
    out = np.zeros((flat_idx.shape[0], vocab_size), dtype=np.float32)
    rows = np.repeat(np.arange(flat_idx.shape[0]), flat_idx.shape[1])
    np.add.at(out, (rows, flat_idx.reshape(-1).clip(0, vocab_size - 1)),
              flat_cnt.reshape(-1))
    # Slot padding writes into column PAD_CARD; its contribution is 0 anyway
    # because padded counts are 0, but clear it so PAD never carries mass.
    out[:, PAD_CARD] = 0.0
    if normalize:
        total = out.sum(axis=1, keepdims=True)
        np.divide(out, total, out=out, where=total > 0)
    return out.reshape(*lead, vocab_size)


def empty_belief_labels() -> dict[str, np.ndarray]:
    """All-invalid belief labels, for decision points where no label exists.

    ``_write_shard`` stacks samples by taking the key set of the *first* sample
    in the buffer, so a sample that omits these keys either crashes the stack or
    -- worse, if it happens to be first -- silently drops the labels for the
    whole shard.  Every sample therefore carries the full key set and the
    ``*_valid`` masks decide what contributes to the loss.
    """
    return {
        "bel_deck_idx": np.full(BELIEF_K, PAD_CARD, dtype=np.int16),
        "bel_deck_cnt": np.zeros(BELIEF_K, dtype=np.int16),
        "bel_hidden_idx": np.full(BELIEF_K, PAD_CARD, dtype=np.int16),
        "bel_hidden_cnt": np.zeros(BELIEF_K, dtype=np.int16),
        "bel_hand_idx": np.full(BELIEF_K, PAD_CARD, dtype=np.int16),
        "bel_hand_cnt": np.zeros(BELIEF_K, dtype=np.int16),
        "bel_hand_valid": np.zeros((), dtype=np.bool_),
        "bel_arch": np.array(-1, dtype=np.int16),
        "bel_hidden_total": np.zeros((), dtype=np.int16),
        "bel_valid": np.zeros((), dtype=np.bool_),
    }


def build_belief_labels(
    state: dict,
    your_index: int,
    opp_deck: list[int],
    id_to_index: dict,
    vocab_size: int,
    opp_arch_index: int,
    opp_hand_ids: list[int] | None,
) -> dict[str, np.ndarray]:
    """All belief targets for one decision point.

    Keys written into the shard:

    ``bel_deck_idx`` / ``bel_deck_cnt``
        The opponent's full 60-card decklist.  Exact, available always.
    ``bel_hidden_idx`` / ``bel_hidden_cnt``
        Decklist minus what we have seen -- the pool determinization must draw
        the opponent's deck, hand and prizes from.  Exact, available always.
    ``bel_hand_idx`` / ``bel_hand_cnt`` / ``bel_hand_valid``
        The opponent's hand at their next decision (see
        :func:`opp_hand_at_next_decision`).  Masked when they never act again.
    ``bel_arch``
        Contiguous index of the opponent's archetype, or -1 when unmapped.
        Cross-entropy ignores -1, so unmapped rows cost nothing.
    ``bel_hidden_total``
        Size of the hidden pool, kept because the planner needs an absolute
        count and the distribution alone cannot supply one.
    ``bel_valid``
        True for every row this function produces.  Rows filled in by
        :func:`empty_belief_labels` carry False.
    """
    deck_dense = deck_counts_dense(opp_deck, id_to_index, vocab_size)
    visible = opp_visible_counts(state, your_index, id_to_index, vocab_size)
    # A card we have seen cannot still be hidden.  clip() rather than plain
    # subtraction because our own cards can end up in their discard (Boss's
    # Orders on an attached Tool, and similar), which would otherwise drive a
    # count negative and silently corrupt the distribution.
    hidden_dense = np.clip(deck_dense - visible, 0, None)

    deck_idx, deck_cnt = _to_sparse(deck_dense)
    hid_idx, hid_cnt = _to_sparse(hidden_dense)

    if opp_hand_ids is None:
        hand_idx = np.full(BELIEF_K, PAD_CARD, dtype=np.int16)
        hand_cnt = np.zeros(BELIEF_K, dtype=np.int16)
        hand_valid = np.zeros((), dtype=np.bool_)
    else:
        hand_dense = deck_counts_dense(opp_hand_ids, id_to_index, vocab_size)
        hand_idx, hand_cnt = _to_sparse(hand_dense)
        # An empty hand is a legitimate observation but carries no signal for a
        # distribution target, so treat it as invalid rather than as uniform.
        hand_valid = np.array(hand_dense.sum() > 0, dtype=np.bool_)

    return {
        "bel_deck_idx": deck_idx,
        "bel_deck_cnt": deck_cnt,
        "bel_hidden_idx": hid_idx,
        "bel_hidden_cnt": hid_cnt,
        "bel_hand_idx": hand_idx,
        "bel_hand_cnt": hand_cnt,
        "bel_hand_valid": hand_valid,
        "bel_arch": np.array(opp_arch_index, dtype=np.int16),
        "bel_hidden_total": np.array(hidden_dense.sum(), dtype=np.int16),
        "bel_valid": np.ones((), dtype=np.bool_),
    }
