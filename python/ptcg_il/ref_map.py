"""Reference resolution map for option → state-token row lookup (Appendix A.7).

Precomputes a map `(area, playerIndex, index) -> state_token_row` using the
fixed token layout (A.1).  Also stores per-entity card_id for CARD / ENERGY /
TOOL_CARD resolution when the referenced entity lives in a non-tokenized zone
(e.g. DECK, DISCARD).

The map is built from an `Observation` dict and covers:
  * active / bench for both players
  * the acting player's hand cards
  * stadium
"""

# Engine AreaType values used in option refs
_HAND = 2
_ACTIVE = 4
_BENCH = 5
_STADIUM = 7

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
