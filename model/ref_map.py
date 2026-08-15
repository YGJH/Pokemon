"""Reference resolution map for option → state-token row lookup (Appendix A.7).

`build_ref_map` precomputes `(area, playerIndex, index) -> state_token_row` using
the fixed token layout (A.1), covering:
  * active / bench for both players
  * the acting player's hand cards
  * stadium

`card_id_at` resolves the *card identity* at a location, including zones that have
no state token (hand, discard, prize, `looking`).  Options almost never carry a
`cardId` field (measured: 0.01%), so this dereference is the only way `opt_card_id`
gets populated.  It deliberately returns None for genuinely hidden cards.
"""

# Engine AreaType values used in option refs.
#
# Only 2/4/5/7 are needed for token-row lookup (the tokenized zones).  The rest
# are required by `card_id_at` below, which also resolves non-tokenized zones.
# Values 1/3/6/12 were identified empirically from the corpus by matching option
# index ranges against each container's length; see RL_SPEC.md §6.4.1.
_DECK = 1
_HAND = 2
_DISCARD = 3
_ACTIVE = 4
_BENCH = 5
_PRIZE = 6
_STADIUM = 7
_LOOKING = 12

# Fixed state-token row layout (A.1)
_MY_ACTIVE = 1
_MY_BENCH_START = 2
_OPP_ACTIVE = 7
_OPP_BENCH_START = 8
_HAND_START = 13
_STADIUM_ROW = 45


def build_ref_map(observation: dict) -> dict:
    """Build a reference-resolution map from an Observation dict.

    Returns a dict keyed by ``(area, playerIndex, index)`` whose values are
    state-token row indices (1-based per A.1 fixed layout).

    It also carries one non-tuple key, ``"_serials"``: ``{serial: row}`` over
    every in-play Pokémon of both players.  ``serial`` is unique within a match
    (``AGENT_SPEC`` §2.1) and is the *only* field that distinguishes one
    ``SKILL`` option from another — two copies of the same card in play share
    ``cardId`` and differ solely by ``serial``.  Without this map those options
    are byte-identical to the pointer head, which is what
    :func:`ptcg_il.featurizer.option_groups` was merging them on.

    Parameters
    ----------
    observation : dict
        An observation dict with keys ``"current"`` (State) and ``"select"``
        (SelectData).  ``current`` must not be None.

    Returns
    -------
    dict
        ``{(area, playerIndex, index): state_token_row}``
    """
    state = observation["current"]
    your_index = state["yourIndex"]
    ref_map: dict[tuple, int] = {}

    # --- My active (row 1) ---
    my_active = state["players"][your_index]["active"]
    if len(my_active) > 0 and my_active[0] is not None:
        ref_map[(_ACTIVE, your_index, 0)] = _MY_ACTIVE

    # --- My bench (rows 2..6) ---
    my_bench = state["players"][your_index]["bench"]
    for i in range(len(my_bench)):
        ref_map[(_BENCH, your_index, i)] = _MY_BENCH_START + i

    # --- Opp active (row 7) ---
    opp_idx = 1 - your_index
    opp_active = state["players"][opp_idx]["active"]
    if len(opp_active) > 0 and opp_active[0] is not None:
        ref_map[(_ACTIVE, opp_idx, 0)] = _OPP_ACTIVE

    # --- Opp bench (rows 8..12) ---
    opp_bench = state["players"][opp_idx]["bench"]
    for i in range(len(opp_bench)):
        ref_map[(_BENCH, opp_idx, i)] = _OPP_BENCH_START + i

    # --- My hand (rows 13..42) ---
    my_hand = state["players"][your_index].get("hand")
    if my_hand is not None:
        for i in range(min(len(my_hand), 30)):  # H_MAX = 30
            ref_map[(_HAND, your_index, i)] = _HAND_START + i

    # --- Stadium (row 45) ---
    stadium = state.get("stadium", [])
    if len(stadium) > 0:
        ref_map[(_STADIUM, -1, 0)] = _STADIUM_ROW

    # --- serial -> token row, for every in-play Pokémon of both players ---
    serials: dict[int, int] = {}
    for pidx, base_active, base_bench in (
        (your_index, _MY_ACTIVE, _MY_BENCH_START),
        (opp_idx, _OPP_ACTIVE, _OPP_BENCH_START),
    ):
        player = state["players"][pidx]
        active = player["active"]
        if len(active) > 0 and active[0] is not None:
            _put_serial(serials, active[0], base_active)
        for i, poke in enumerate(player["bench"][:5]):
            _put_serial(serials, poke, base_bench + i)
    ref_map["_serials"] = serials

    return ref_map


def _put_serial(serials: dict, poke, row: int) -> None:
    """Record ``poke["serial"] -> row``, skipping entities that carry no serial.

    A face-down Pokémon is ``None`` and an opponent's may omit the field; both
    simply do not get an entry, so :func:`serial_row` returns -1 and the option
    falls back to the null token exactly as it did before.
    """
    if not isinstance(poke, dict):
        return
    serial = poke.get("serial")
    if isinstance(serial, int):
        serials[serial] = row


def serial_row(ref_map: dict, serial) -> int:
    """State-token row of the in-play Pokémon with this ``serial``, or -1."""
    if serial is None:
        return -1
    try:
        return int(ref_map.get("_serials", {}).get(int(serial), -1))
    except (TypeError, ValueError):
        return -1


def card_id_at(state: dict, area, player_idx, index, select_deck=None) -> int | None:
    """Resolve `(area, playerIndex, index)` to a raw engine card id.

    Returns ``None`` when the referenced card is **genuinely hidden** or the
    location is out of range.  Callers must leave ``opt_card_id`` at PAD in that
    case: writing an id for a face-down card would hand the policy information
    the live agent cannot see, which inflates offline metrics and collapses at
    live-eval.

    Hidden by design:
      * ``_PRIZE`` — ``prize`` is ``[Card | None]``; face-down slots are ``None``.
      * face-down active (``active[0] is None``).
      * ``_DECK`` — *unless* `select_deck` is supplied; see below.

    **`select_deck` is the search payload, not a peek at the deck.**  The state
    carries ``deckCount`` and nothing else, so `_DECK` is unresolvable from
    *state* alone and guessing would leak.  But when an effect makes you search
    your own deck, the engine puts the searchable cards in ``select["deck"]``
    and the options index into *that list* — which is how the real game works,
    and which the live agent receives verbatim in its own observation.  Passing
    it here is therefore not a leak; refusing it is the leak's mirror image,
    where the agent is denied information it legitimately has.

    Measured on the corpus: 100% of ``area == _DECK`` option indices fall inside
    ``select["deck"]``, and 100% of those entries carry the acting player's own
    ``playerIndex``.  The `playerIndex` guard below keeps it that way — an entry
    belonging to the other player is not ours to read.

    Note the asymmetry with options: state containers key the card id as
    ``card["id"]``, while options (rarely) use ``cardId``.
    """
    try:
        area = int(area)
        index = int(index)
    except (TypeError, ValueError):
        return None

    if area == _DECK:
        return _deck_card_id(select_deck, player_idx, index)

    if area == _STADIUM:
        card = _first(state.get("stadium"))
    elif area == _LOOKING:
        # The reveal buffer may sit on the state or on the player.
        looking = state.get("looking")
        if not looking:
            player = _player(state, player_idx)
            looking = (player or {}).get("looking")
        card = _at(looking, index)
    else:
        player = _player(state, player_idx)
        if player is None:
            return None
        if area == _HAND:
            card = _at(player.get("hand"), index)
        elif area == _DISCARD:
            card = _at(player.get("discard"), index)
        elif area == _ACTIVE:
            card = _at(player.get("active"), index)
        elif area == _BENCH:
            card = _at(player.get("bench"), index)
        elif area == _PRIZE:
            card = _at(player.get("prize"), index)
        else:
            return None

    if not isinstance(card, dict):
        return None
    cid = card.get("id")
    return int(cid) if isinstance(cid, int) else None


def _deck_card_id(select_deck, player_idx, index) -> int | None:
    """Card id at *index* of a ``select["deck"]`` payload, or None.

    Returns None when there is no payload (the ordinary case — the deck is
    hidden), when the index is out of range, or when the entry names a player
    other than the one the option referenced.
    """
    card = _at(select_deck, index)
    if not isinstance(card, dict):
        return None
    owner = card.get("playerIndex")
    if owner is not None and player_idx is not None:
        try:
            if int(owner) != int(player_idx):
                return None
        except (TypeError, ValueError):
            return None
    cid = card.get("id")
    return int(cid) if isinstance(cid, int) else None


def _player(state: dict, player_idx) -> dict | None:
    try:
        return state["players"][int(player_idx)]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _at(container, index):
    """Element `index` of `container`, or None if absent/out of range."""
    if not isinstance(container, list) or not 0 <= index < len(container):
        return None
    return container[index]


def _first(container):
    """Element 0 of `container`, or None."""
    return _at(container, 0)
