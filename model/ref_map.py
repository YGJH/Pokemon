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
    state-token row indices (1-based per A.1 fixed layout).  Also stores a
    ``_card_ids`` sub-dict ``{(area, playerIndex, index): card_id}`` for
    entities in non-tokenized zones.

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

    return ref_map


def card_id_at(state: dict, area, player_idx, index) -> int | None:
    """Resolve `(area, playerIndex, index)` to a raw engine card id.

    Returns ``None`` when the referenced card is **genuinely hidden** or the
    location is out of range.  Callers must leave ``opt_card_id`` at PAD in that
    case: writing an id for a face-down card would hand the policy information
    the live agent cannot see, which inflates offline metrics and collapses at
    live-eval.

    Hidden by design:
      * ``_DECK``  — the observation carries ``deckCount`` only, never the cards.
      * ``_PRIZE`` — ``prize`` is ``[Card | None]``; face-down slots are ``None``.
      * face-down active (``active[0] is None``).

    Note the asymmetry with options: state containers key the card id as
    ``card["id"]``, while options (rarely) use ``cardId``.
    """
    try:
        area = int(area)
        index = int(index)
    except (TypeError, ValueError):
        return None

    if area == _DECK:
        # Not resolvable, and must not be guessed.
        return None

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
