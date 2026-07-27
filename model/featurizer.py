"""Featurizer core: obs_dict → tensor dict per TRANSFORMER_IL_SPEC.md Appendix A.

Pure NumPy — no PyTorch dependency.  One call per decision point.

Usage::

    tensors = featurize(obs_dict, vocab, action,
                        value_target=1.0, sample_weight=1.0)

Raises ``ValueError`` if ``select is None`` (deck-selection steps are excluded
— the deck is fixed, not predicted).
"""

import numpy as np

from model.ref_map import build_ref_map, card_id_at

# ============================================================
# Normalizers (A.2)
# ============================================================
HP_N = 400.0
RETREAT_N = 4.0
ATKDMG_N = 350.0
ENERGY_N = 12.0
DECK_N = 60.0
HAND_N = 30.0
TURN_N = 50.0
COUNT_N = 20.0
PRIZE_N = 6.0
BENCH_N = 8.0
ATKCOST_N = 5.0
DMGCTR_N = 20.0

# ============================================================
# Capacities (A.1)
# ============================================================
P_MAX = 12
H_MAX = 30
D_MAX = 60
PZ_MAX = 6
SUM = 2
STAD = 1
CLS = 1
L_STATE = 46  # CLS + P_MAX + H_MAX + SUM + STAD
O_MAX = 64
L_LOG_MAX = 32   # max log entries per decision
LOG_FEAT_DIM = 6 # log_type, player_rel, card_id, area_from, area_to, scalar

# ============================================================
# Feature dims (A.1)
# ============================================================
F_CARD = 52
F_ATK = 14
F_POKE = 26
F_HAND = 2
F_SUM = 11
F_GLOBAL = 93
F_OPT = 6

# ============================================================
# Enum sizes (A.1)
# ============================================================
N_SELTYPE = 11
N_SELCTX = 49
N_OPTTYPE = 18  # 17 original types + STOP
STOP_OPT_TYPE = 17
N_COND = 5
N_ENERGY = 12

# ============================================================
# Token-type / owner / zone vocab (int encodings)
# ============================================================
TOK_TYPE = {"CLS": 0, "POKE": 1, "HAND": 2, "SUMMARY": 3, "STADIUM": 4}
TOK_OWNER = {"none": 0, "self": 1, "opp": 2}
TOK_ZONE = {"cls": 0, "active": 1, "bench": 2, "hand": 3, "summary": 4, "stadium": 5}

# ============================================================
# Special sentinel values
# ============================================================
PAD_CARD = 0
UNKNOWN_CARD = 1
PAD_ATTACK = 0


def normalize_vocab(vocab: dict) -> dict:
    """Return *vocab* with ``id_to_index`` / ``attack_id_to_index`` keyed by int.

    ``vocab.json`` round-trips through JSON, which forces all object keys to
    strings (``{"1152": 6}``).  Card and attack ids coming out of the engine
    observation are **ints**, so a raw ``json.load`` result silently misses on
    every lookup and maps every card to ``UNKNOWN_CARD`` / every attack to
    ``PAD_ATTACK``.  Every caller that hands a vocab to :func:`featurize` must
    pass it through here first.

    Idempotent — safe to call on an already-normalized dict.
    """
    out = dict(vocab)
    for key in ("id_to_index", "attack_id_to_index"):
        mapping = vocab.get(key)
        if mapping:
            out[key] = {int(k): int(v) for k, v in mapping.items()}
    return out

# ============================================================
# Helpers
# ============================================================


def _remap_card(card_id, id_to_index: dict) -> int:
    """Remap a real card id to a vocab index.  None → PAD, OOV → UNKNOWN."""
    if card_id is None:
        return PAD_CARD
    return int(id_to_index.get(card_id, UNKNOWN_CARD))


def _remap_attack(attack_id, attack_id_to_index: dict) -> int:
    """Remap a real attack id to a vocab index.  None → PAD_ATTACK."""
    if attack_id is None:
        return PAD_ATTACK
    return int(attack_id_to_index.get(attack_id, PAD_ATTACK))


def _energy_histogram(energies: list, n: int = N_ENERGY) -> np.ndarray:
    """float32[n] histogram of energy-type counts, divided by ENERGY_N."""
    hist = np.zeros(n, dtype=np.float32)
    for e in energies:
        ei = int(e)
        if 0 <= ei < n:
            hist[ei] += 1.0
    hist /= ENERGY_N
    return hist


def _onehot(value: int | None, size: int) -> np.ndarray:
    """float32[size] one-hot; all-zero if value is None or out of range."""
    vec = np.zeros(size, dtype=np.float32)
    if value is not None and 0 <= int(value) < size:
        vec[int(value)] = 1.0
    return vec


def _clip_norm(value: float, norm: float) -> float:
    """Clip value/norm to [0, 1] (float counts) or [-1, 1] (signed).

    Pure-Python arithmetic on purpose: this is the hottest function in the
    featurizer (~125 calls per decision) and `np.clip` on a *scalar* pays the
    full array-dispatch cost -- ~2.6 us a call, 69% of total featurize time.
    The branch form is bitwise-identical, NaN and +-inf included.
    """
    x = value / norm
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return float(x)


# ============================================================
# Sub-builders
# ============================================================


def _build_poke_tokens(
    state: dict, your_index: int, id_to_index: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Build poke_card_id[P_MAX] int64 and poke_feat[P_MAX, F_POKE] float32.

    Slot layout (A.1): 0=my_active, 1..5=my_bench[0..4],
    6=opp_active, 7..11=opp_bench[0..4].
    """
    poke_card_id = np.full(P_MAX, PAD_CARD, dtype=np.int64)
    poke_feat = np.zeros((P_MAX, F_POKE), dtype=np.float32)

    for pi, player_idx in enumerate([your_index, 1 - your_index]):
        player = state["players"][player_idx]
        is_me = pi == 0
        base_row = 0 if is_me else 6

        # Active slot
        active_list = player["active"]
        if len(active_list) > 0 and active_list[0] is not None:
            poke = active_list[0]
            row = base_row
            poke_card_id[row] = _remap_card(poke["id"], id_to_index)
            f = poke_feat[row]
            f[0] = _clip_norm(poke["hp"], HP_N)
            f[1] = _clip_norm(poke["maxHp"], HP_N)
            f[2] = _clip_norm(poke["hp"], max(poke["maxHp"], 1))
            f[3:15] = _energy_histogram(poke.get("energies", []))
            f[15] = _clip_norm(len(poke.get("energies", [])), ENERGY_N)
            f[16] = _clip_norm(len(poke.get("energyCards", [])), ENERGY_N)
            f[17] = _clip_norm(len(poke.get("tools", [])), 2.0)
            f[18] = _clip_norm(len(poke.get("preEvolution", [])), 2.0)
            f[19] = 1.0 if poke.get("appearThisTurn", False) else 0.0
            f[20] = 1.0  # is_active
            # Condition flags (active only, A.5)
            f[21] = 1.0 if player.get("poisoned", False) else 0.0
            f[22] = 1.0 if player.get("burned", False) else 0.0
            f[23] = 1.0 if player.get("asleep", False) else 0.0
            f[24] = 1.0 if player.get("paralyzed", False) else 0.0
            f[25] = 1.0 if player.get("confused", False) else 0.0

        # Bench slots
        bench = player["bench"]
        for i, poke in enumerate(bench):
            if i >= 5:  # safety cap
                break
            row = base_row + 1 + i
            poke_card_id[row] = _remap_card(poke["id"], id_to_index)
            f = poke_feat[row]
            f[0] = _clip_norm(poke["hp"], HP_N)
            f[1] = _clip_norm(poke["maxHp"], HP_N)
            f[2] = _clip_norm(poke["hp"], max(poke["maxHp"], 1))
            f[3:15] = _energy_histogram(poke.get("energies", []))
            f[15] = _clip_norm(len(poke.get("energies", [])), ENERGY_N)
            f[16] = _clip_norm(len(poke.get("energyCards", [])), ENERGY_N)
            f[17] = _clip_norm(len(poke.get("tools", [])), 2.0)
            f[18] = _clip_norm(len(poke.get("preEvolution", [])), 2.0)
            f[19] = 1.0 if poke.get("appearThisTurn", False) else 0.0
            f[20] = 0.0  # is_active = False on bench
            # Condition flags zero on bench (A.5)

    return poke_card_id, poke_feat


def _build_hand_tokens(
    state: dict, your_index: int, id_to_index: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Build hand_card_id[H_MAX] int64 and hand_feat[H_MAX, F_HAND] float32.

    Cards are packed to the front; remaining slots are PAD.
    hand_feat = [idx / H_MAX, dup_count / COUNT_N].
    """
    hand_card_id = np.full(H_MAX, PAD_CARD, dtype=np.int64)
    hand_feat = np.zeros((H_MAX, F_HAND), dtype=np.float32)

    hand = state["players"][your_index].get("hand")
    if hand is None:
        return hand_card_id, hand_feat

    # Count duplicates in hand for dup_count feature
    id_counts: dict[int, int] = {}
    for card in hand:
        cid = card["id"]
        id_counts[cid] = id_counts.get(cid, 0) + 1

    for i, card in enumerate(hand):
        if i >= H_MAX:
            break
        cid = card["id"]
        hand_card_id[i] = _remap_card(cid, id_to_index)
        hand_feat[i, 0] = _clip_norm(float(i), HAND_N)
        hand_feat[i, 1] = _clip_norm(float(id_counts.get(cid, 1)), COUNT_N)

    return hand_card_id, hand_feat


def _build_summary_tokens(state: dict, your_index: int) -> np.ndarray:
    """Build sum_feat[SUM, F_SUM] float32.

    Row 0 = me, row 1 = opp.
    F_SUM = 11: [is_me, deckCount, handCount, bench_len, benchMax,
                  prizes_left, discard_len, poisoned, burned, asleep, paralyzed]
    """
    sum_feat = np.zeros((SUM, F_SUM), dtype=np.float32)

    for pi, player_idx in enumerate([your_index, 1 - your_index]):
        player = state["players"][player_idx]
        is_me = pi == 0
        f = sum_feat[pi]
        f[0] = 1.0 if is_me else 0.0
        f[1] = _clip_norm(float(player["deckCount"]), DECK_N)
        f[2] = _clip_norm(float(player["handCount"]), HAND_N)
        f[3] = _clip_norm(float(len(player["bench"])), BENCH_N)
        f[4] = _clip_norm(float(player["benchMax"]), BENCH_N)
        f[5] = _clip_norm(float(len(player["prize"])), PRIZE_N)
        f[6] = _clip_norm(float(len(player["discard"])), DECK_N)
        f[7] = 1.0 if player.get("poisoned", False) else 0.0
        f[8] = 1.0 if player.get("burned", False) else 0.0
        f[9] = 1.0 if player.get("asleep", False) else 0.0
        f[10] = 1.0 if player.get("paralyzed", False) else 0.0

    return sum_feat


def _build_stadium_token(
    state: dict, id_to_index: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Build stadium_card_id[1] int64, stadium_present[1] float32."""
    stadium = state.get("stadium", [])
    if len(stadium) > 0:
        sid = _remap_card(stadium[0]["id"], id_to_index)
        return np.array([sid], dtype=np.int64), np.array([1.0], dtype=np.float32)
    return np.array([PAD_CARD], dtype=np.int64), np.array([0.0], dtype=np.float32)


def _build_cls_features(
    state: dict, select: dict, your_index: int, id_to_index: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cls_feat[F_GLOBAL] float32, context_card_id[1] int64, effect_card_id[1] int64.

    Per A.6 layout exactly.
    """
    cls_feat = np.zeros(F_GLOBAL, dtype=np.float32)

    # [0] turn
    cls_feat[0] = _clip_norm(float(state["turn"]), TURN_N)
    # [1] turn % 2
    cls_feat[1] = float(state["turn"] % 2)
    # [2] yourIndex
    cls_feat[2] = float(your_index)
    # [3] turnActionCount
    cls_feat[3] = _clip_norm(float(state["turnActionCount"]), COUNT_N)
    # [4:7] firstPlayer one-hot {-1→[1,0,0], 0→[0,1,0], 1→[0,0,1]}
    fp = state["firstPlayer"]
    if fp == -1:
        cls_feat[4] = 1.0
    elif fp == 0:
        cls_feat[5] = 1.0
    elif fp == 1:
        cls_feat[6] = 1.0
    # [7:11] per-turn flags
    cls_feat[7] = 1.0 if state.get("supporterPlayed", False) else 0.0
    cls_feat[8] = 1.0 if state.get("stadiumPlayed", False) else 0.0
    cls_feat[9] = 1.0 if state.get("energyAttached", False) else 0.0
    cls_feat[10] = 1.0 if state.get("retreated", False) else 0.0
    # [11:13] prizes left
    cls_feat[11] = _clip_norm(
        float(len(state["players"][your_index]["prize"])), PRIZE_N
    )
    cls_feat[12] = _clip_norm(
        float(len(state["players"][1 - your_index]["prize"])), PRIZE_N
    )
    # [13:24] select.type one-hot (N_SELTYPE=11)
    sel_type = select["type"]
    cls_feat[13 + int(sel_type)] = 1.0
    # [24:73] select.context one-hot (N_SELCTX=49)
    sel_ctx = select["context"]
    cls_feat[24 + int(sel_ctx)] = 1.0
    # [73:77] minCount, maxCount, remainEnergyCost, remainDamageCounter
    cls_feat[73] = _clip_norm(float(select["minCount"]), COUNT_N)
    cls_feat[74] = _clip_norm(float(select["maxCount"]), COUNT_N)
    cls_feat[75] = _clip_norm(
        float(select.get("remainEnergyCost", 0)), ATKCOST_N
    )
    cls_feat[76] = _clip_norm(
        float(select.get("remainDamageCounter", 0)), DMGCTR_N
    )
    # [77:87] conditions: my-active(5) + opp-active(5)
    for pi, pidx in enumerate([your_index, 1 - your_index]):
        player = state["players"][pidx]
        base = 77 + pi * 5
        cls_feat[base + 0] = 1.0 if player.get("poisoned", False) else 0.0
        cls_feat[base + 1] = 1.0 if player.get("burned", False) else 0.0
        cls_feat[base + 2] = 1.0 if player.get("asleep", False) else 0.0
        cls_feat[base + 3] = 1.0 if player.get("paralyzed", False) else 0.0
        cls_feat[base + 4] = 1.0 if player.get("confused", False) else 0.0
    # [87:89] has_contextCard, has_effect
    cls_feat[87] = 1.0 if select.get("contextCard") is not None else 0.0
    cls_feat[88] = 1.0 if select.get("effect") is not None else 0.0
    # [89:93] reserved (zeros already)

    # context_card_id, effect_card_id
    ctx_card = select.get("contextCard")
    eff_card = select.get("effect")
    context_card_id = np.array(
        [_remap_card(ctx_card["id"] if ctx_card else None, id_to_index)],
        dtype=np.int64,
    )
    effect_card_id = np.array(
        [_remap_card(eff_card["id"] if eff_card else None, id_to_index)],
        dtype=np.int64,
    )

    return cls_feat, context_card_id, effect_card_id


def _build_discard_prizes(
    state: dict, your_index: int, id_to_index: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build discard_ids[SUM, D_MAX], discard_mask[SUM, D_MAX], prize_ids[SUM, PZ_MAX].

    Prize ids are only for *revealed* (non-None) cards; face-down → PAD.
    """
    discard_ids = np.full((SUM, D_MAX), PAD_CARD, dtype=np.int64)
    discard_mask = np.zeros((SUM, D_MAX), dtype=bool)
    prize_ids = np.full((SUM, PZ_MAX), PAD_CARD, dtype=np.int64)

    for pi, player_idx in enumerate([your_index, 1 - your_index]):
        player = state["players"][player_idx]

        # Discard pile
        disc = player["discard"]
        for i, card in enumerate(disc):
            if i >= D_MAX:
                break
            discard_ids[pi, i] = _remap_card(card["id"], id_to_index)
            discard_mask[pi, i] = True

        # Revealed prizes
        prizes = player["prize"]
        for i, card in enumerate(prizes):
            if i >= PZ_MAX:
                break
            if card is not None:
                prize_ids[pi, i] = _remap_card(card["id"], id_to_index)
            # else stays PAD

    return discard_ids, discard_mask, prize_ids


def _build_tok_attrs(
    poke_card_id: np.ndarray,
    hand_card_id: np.ndarray,
    stadium_present: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build tok_type[L_STATE], tok_owner[L_STATE], tok_zone[L_STATE], tok_mask[L_STATE].

    All int64 except tok_mask bool.
    """
    tok_type = np.zeros(L_STATE, dtype=np.int64)
    tok_owner = np.zeros(L_STATE, dtype=np.int64)
    tok_zone = np.zeros(L_STATE, dtype=np.int64)
    tok_mask = np.zeros(L_STATE, dtype=bool)

    # CLS (row 0)
    tok_type[0] = TOK_TYPE["CLS"]
    tok_owner[0] = TOK_OWNER["none"]
    tok_zone[0] = TOK_ZONE["cls"]
    tok_mask[0] = True

    # Pokémon-in-play (rows 1..12)
    for i in range(P_MAX):
        row = 1 + i
        if poke_card_id[i] == PAD_CARD:
            continue  # empty slot — stays all-zero / False
        tok_type[row] = TOK_TYPE["POKE"]
        is_my = i <= 5  # rows 1..6 are mine, 7..12 are opp
        tok_owner[row] = TOK_OWNER["self"] if is_my else TOK_OWNER["opp"]
        is_active_slot = i == 0 or i == 6
        tok_zone[row] = TOK_ZONE["active"] if is_active_slot else TOK_ZONE["bench"]
        tok_mask[row] = True

    # Hand (rows 13..42)
    for i in range(H_MAX):
        row = 13 + i
        if hand_card_id[i] == PAD_CARD:
            continue
        tok_type[row] = TOK_TYPE["HAND"]
        tok_owner[row] = TOK_OWNER["self"]
        tok_zone[row] = TOK_ZONE["hand"]
        tok_mask[row] = True

    # Summary (rows 43..44)
    for pi in range(SUM):
        row = 43 + pi
        tok_type[row] = TOK_TYPE["SUMMARY"]
        tok_owner[row] = TOK_OWNER["self"] if pi == 0 else TOK_OWNER["opp"]
        tok_zone[row] = TOK_ZONE["summary"]
        tok_mask[row] = True

    # Stadium (row 45)
    if stadium_present[0] > 0:
        row = 45
        tok_type[row] = TOK_TYPE["STADIUM"]
        tok_owner[row] = TOK_OWNER["none"]
        tok_zone[row] = TOK_ZONE["stadium"]
        tok_mask[row] = True

    return tok_type, tok_owner, tok_zone, tok_mask


def _build_option_tokens(
    select: dict,
    ref_map: dict,
    your_index: int,
    id_to_index: dict,
    attack_id_to_index: dict,
    action: list[int] | None = None,
    state: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int], int]:
    """Build option tensors per A.4/A.7.

    When the raw option list exceeds O_MAX, the chosen actions are always
    included within the O_MAX window so that no training samples are silently
    dropped.  Returns an *index_remap* dict mapping old option indices to new
    positions (identity when no reordering occurred).

    For multi-select decisions (maxCount > 1), a STOP column is appended
    after the regular options so the model can learn when to stop picking.
    *stop_column* is the index of that column (or -1 for single-select).

    Returns (opt_type, opt_src_idx, opt_tgt_idx, opt_card_id, opt_attack_idx,
             opt_scalar, opt_mask, index_remap, stop_column).
    """
    options = select["option"]
    n_total = len(options)
    max_count = int(select.get("maxCount", 1))
    is_multi = max_count > 1

    # ---- choose which option indices to keep (old → new) ----
    # For multi-select, reserve one slot for the STOP column
    max_regular = (O_MAX - 1) if is_multi else O_MAX

    if n_total <= max_regular:
        indices_to_keep = list(range(n_total))
        index_remap = {i: i for i in range(n_total)}
    else:
        # Prioritise action-chosen options so the model can always predict them
        chosen: set[int] = set()
        if action is not None:
            for a in action:
                a_int = int(a)
                if 0 <= a_int < n_total:
                    chosen.add(a_int)
        indices_to_keep = list(chosen)
        for i in range(n_total):
            if len(indices_to_keep) >= max_regular:
                break
            if i not in chosen:
                indices_to_keep.append(i)
        index_remap = {old: new for new, old in enumerate(indices_to_keep)}

    n_opts = len(indices_to_keep)

    opt_type = np.zeros(O_MAX, dtype=np.int64)
    opt_src_idx = np.full(O_MAX, -1, dtype=np.int64)
    opt_tgt_idx = np.full(O_MAX, -1, dtype=np.int64)
    opt_card_id = np.full(O_MAX, PAD_CARD, dtype=np.int64)
    opt_attack_idx = np.zeros(O_MAX, dtype=np.int64)
    opt_scalar = np.zeros((O_MAX, F_OPT), dtype=np.float32)
    opt_mask = np.zeros(O_MAX, dtype=bool)

    for new_j, old_j in enumerate(indices_to_keep):
        opt = options[old_j]
        otype = int(opt["type"])
        opt_type[new_j] = otype
        opt_mask[new_j] = True

        # Fill scalar features
        opt_scalar[new_j, 0] = _clip_norm(float(opt.get("number", 0)), COUNT_N)
        opt_scalar[new_j, 1] = _clip_norm(float(opt.get("count", 0)), COUNT_N)
        opt_scalar[new_j, 2] = _clip_norm(
            float(opt.get("energyIndex", 0)), float(N_ENERGY)
        )
        opt_scalar[new_j, 3] = _clip_norm(float(opt.get("toolIndex", 0)), 2.0)
        opt_scalar[new_j, 4] = _clip_norm(
            float(select.get("remainEnergyCost", 0)), ATKCOST_N
        )
        opt_scalar[new_j, 5] = _clip_norm(
            float(select.get("remainDamageCounter", 0)), DMGCTR_N
        )

        # Built-in ref helper
        def _ref(area, player_idx, idx):
            return ref_map.get((int(area), int(player_idx), int(idx)), -1)

        def _set_card(area, player_idx, idx):
            """Populate opt_card_id by dereferencing a location (A.7).

            Options almost never carry `cardId` (measured 0.01% of the corpus),
            so card identity has to come from the state.  Leaves the entry at
            PAD when `card_id_at` reports the card is hidden -- deck slots and
            face-down prizes must stay PAD or we leak.
            """
            if state is None or area is None or idx is None:
                return
            raw = card_id_at(state, area, player_idx, idx)
            if raw is not None:
                opt_card_id[new_j] = _remap_card(raw, id_to_index)

        # Resolve per option type (A.7)
        if otype == 7:  # PLAY
            # Bare `index` into the actor's hand -- no area/playerIndex given.
            src = _ref(2, your_index, opt.get("index", -1))
            opt_src_idx[new_j] = src if src != -1 else -1
            _set_card(2, your_index, opt.get("index"))

        elif otype == 8:  # ATTACH
            area = opt.get("area")
            index = opt.get("index")
            if area is not None and index is not None:
                opt_src_idx[new_j] = _ref(area, your_index, index)
                _set_card(area, your_index, index)
            elif index is not None:
                # Bare `index` is the actor's hand (the card being attached).
                _set_card(2, your_index, index)
            in_play_area = opt.get("inPlayArea")
            in_play_idx = opt.get("inPlayIndex")
            if in_play_area is not None and in_play_idx is not None:
                opt_tgt_idx[new_j] = _ref(in_play_area, your_index, in_play_idx)

        elif otype == 9:  # EVOLVE
            area = opt.get("area")
            index = opt.get("index")
            if area is not None and index is not None:
                opt_src_idx[new_j] = _ref(area, your_index, index)
                _set_card(area, your_index, index)
            elif index is not None:
                # Bare `index` is the actor's hand (the evolution card).
                _set_card(2, your_index, index)
            in_play_area = opt.get("inPlayArea")
            in_play_idx = opt.get("inPlayIndex")
            if in_play_area is not None and in_play_idx is not None:
                opt_tgt_idx[new_j] = _ref(in_play_area, your_index, in_play_idx)

        elif otype in (10, 11):  # ABILITY, DISCARD
            area = opt.get("area")
            player_idx = opt.get("playerIndex")
            index = opt.get("index")
            if area is not None and player_idx is not None and index is not None:
                opt_src_idx[new_j] = _ref(area, player_idx, index)
                _set_card(area, player_idx, index)
            elif area is not None and index is not None:
                # ABILITY often omits playerIndex -- it is the actor's own board.
                opt_src_idx[new_j] = _ref(area, your_index, index)
                _set_card(area, your_index, index)

        elif otype == 12:  # RETREAT
            opt_src_idx[new_j] = 1  # my active (row 1)

        elif otype == 13:  # ATTACK
            opt_src_idx[new_j] = 1  # my active (row 1)
            opt_attack_idx[new_j] = _remap_attack(
                opt.get("attackId"), attack_id_to_index
            )
            # Some attacks target specific bench slots (snipe effects)
            in_play_area = opt.get("inPlayArea")
            in_play_idx = opt.get("inPlayIndex")
            if in_play_area is not None and in_play_idx is not None:
                opt_tgt_idx[new_j] = _ref(in_play_area, 1 - your_index, in_play_idx)

        elif otype == 3:  # CARD
            area = opt.get("area")
            player_idx = opt.get("playerIndex", your_index)
            index = opt.get("index")
            if area is not None and index is not None:
                opt_src_idx[new_j] = _ref(area, player_idx, index)
            # Always set card_id for disambiguation (A.7).  `cardId` is present
            # on ~0% of real options, so the location dereference does the work;
            # it stays PAD for deck slots and face-down prizes.
            cid = opt.get("cardId")
            if cid is not None:
                opt_card_id[new_j] = _remap_card(cid, id_to_index)
            else:
                _set_card(area, player_idx, index)

        elif otype in (4, 5, 6):  # TOOL_CARD, ENERGY_CARD, ENERGY
            area = opt.get("area")
            player_idx = opt.get("playerIndex", your_index)
            index = opt.get("index")
            if area is not None and index is not None:
                opt_src_idx[new_j] = _ref(area, player_idx, index)
            # Card id for attachment disambiguation (A.7)
            cid = opt.get("cardId")
            if cid is not None:
                opt_card_id[new_j] = _remap_card(cid, id_to_index)
            else:
                _set_card(area, player_idx, index)

        elif otype in (1, 2, 14, 15, 16):  # YES, NO, END, SKILL, SPECIAL_CONDITION
            # src = -1, tgt = -1 (constant-type options)
            cid = opt.get("cardId")
            if cid is not None:
                opt_card_id[new_j] = _remap_card(cid, id_to_index)

        elif otype == 0:  # NUMBER
            # src = -1, tgt = -1; scalar carries the number
            cid = opt.get("cardId")
            if cid is not None:
                opt_card_id[new_j] = _remap_card(cid, id_to_index)

    # ---- STOP column for multi-select ----
    stop_column = -1
    if is_multi and n_opts < O_MAX:
        stop_column = n_opts
        opt_type[n_opts] = STOP_OPT_TYPE
        opt_mask[n_opts] = True
        # src/tgt = -1, card_id = PAD, attack_idx = 0, scalar = 0 (all defaults)

    return (
        opt_type,
        opt_src_idx,
        opt_tgt_idx,
        opt_card_id,
        opt_attack_idx,
        opt_scalar,
        opt_mask,
        index_remap,
        stop_column,
    )


def _build_label(
    action: list[int],
    select: dict,
    index_remap: dict[int, int] | None = None,
    stop_column: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build action_idx[O_MAX] int64 and action_len scalar int64.

    action_idx stores the expert's picks in order, padded with -1.
    When *index_remap* is provided, action indices are translated from
    original option positions to their new (possibly reordered) positions.

    For multi-select with a STOP column, the STOP target is appended
    after the last expert pick, so the model learns to stop after the
    correct number of selections.
    """
    action_idx = np.full(O_MAX, -1, dtype=np.int64)
    max_count = select["maxCount"]

    n_picks = min(len(action), max_count)
    write_pos = 0
    for k in range(n_picks):
        old_idx = int(action[k])
        if index_remap is not None:
            new_idx = index_remap.get(old_idx)
            if new_idx is None:
                continue  # option was truncated away (should not happen with remap logic)
        else:
            new_idx = old_idx
        action_idx[write_pos] = new_idx
        write_pos += 1

    # Append STOP target for multi-select (after last expert pick)
    if stop_column >= 0 and write_pos < O_MAX:
        action_idx[write_pos] = stop_column
        write_pos += 1

    action_len = np.int64(write_pos)
    return action_idx, action_len


# ============================================================
# Log tokens (belief module)
# ============================================================


def _build_log_tokens(
    logs: list[dict],
    your_index: int,
    id_to_index: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-size log feature tensor for the belief module.

    Each log entry is encoded as [log_type, player_rel, card_id, area_from,
    area_to, scalar_value].  The sequence is truncated to L_LOG_MAX entries.

    Returns (log_feat[L_LOG_MAX, LOG_FEAT_DIM], log_mask[L_LOG_MAX], log_len).
    """
    log_feat = np.zeros((L_LOG_MAX, LOG_FEAT_DIM), dtype=np.float32)
    log_mask = np.zeros(L_LOG_MAX, dtype=bool)
    n_logs = min(len(logs), L_LOG_MAX)

    for i in range(n_logs):
        log = logs[i]
        lt = int(log.get("type", 0))

        # player relative: -1=self, 1=opponent, 0=unknown
        pidx = log.get("playerIndex")
        player_rel = 0
        if pidx is not None:
            player_rel = 1 if int(pidx) != your_index else -1

        # card id
        cid = log.get("cardId")
        card_idx = PAD_CARD
        if cid is not None:
            card_idx = id_to_index.get(int(cid), UNKNOWN_CARD)

        # area from/to
        area_from = log.get("fromArea", -1)
        area_to = log.get("toArea", -1)
        if area_from is None:
            area_from = -1
        if area_to is None:
            area_to = -1

        # scalar: pick the most informative numeric field
        scalar = 0.0
        if lt == 16:  # HP_CHANGE
            scalar = _clip_norm(float(log.get("value", 0)), 200.0)
        elif lt == 22:  # COIN
            scalar = 1.0 if log.get("head") else -1.0
        elif lt == 23:  # RESULT
            scalar = float(log.get("result", 2)) / 2.0  # 0,1,2 → 0,0.5,1

        log_feat[i] = [lt, player_rel, card_idx, area_from, area_to, scalar]
        log_mask[i] = True

    log_len = np.int64(n_logs)
    return log_feat, log_mask, log_len


# ============================================================
# Top-level featurize()
# ============================================================


def featurize(
    obs_dict: dict,
    vocab: dict,
    action: list[int] | None = None,
    value_target: float = 0.0,
    sample_weight: float = 1.0,
) -> dict[str, np.ndarray]:
    """Convert one observation + expert action to the Appendix A.4 tensor dict.

    Parameters
    ----------
    obs_dict : dict
        Raw observation dict (keys: ``"select"``, ``"current"``, ``"logs"``).
    action : list[int] or None
        Expert's chosen option indices (from the off-by-one pairing rule).
        If None (inference mode), placeholder label tensors are returned.
    vocab : dict
        Mined vocab dict with keys ``"id_to_index"`` and
        ``"attack_id_to_index"``.
    value_target : float
        +1 for win, -1 for loss (§1.1).
    sample_weight : float
        Per-sample weight (computed from context/archetype/outcome, C.3).

    Returns
    -------
    dict[str, np.ndarray]
        All Appendix A.4 keys with exact shapes and dtypes.

    Raises
    ------
    ValueError
        If ``select is None`` (deck-selection step — excluded).
    """
    select = obs_dict.get("select")
    if select is None:
        raise ValueError(
            "featurize called with select=None (deck-selection step). "
            "Deck-selection steps are excluded — the deck is fixed, not predicted."
        )

    state = obs_dict.get("current")
    if state is None:
        raise ValueError("featurize called with current=None")

    id_to_index = vocab["id_to_index"]
    attack_id_to_index = vocab.get("attack_id_to_index", {})

    your_index = state["yourIndex"]

    # Build ref map for option resolution
    ref_map = build_ref_map(obs_dict)

    # --- State: card identity ---
    poke_card_id, poke_feat = _build_poke_tokens(state, your_index, id_to_index)
    hand_card_id, hand_feat = _build_hand_tokens(state, your_index, id_to_index)
    stadium_card_id, stadium_present = _build_stadium_token(state, id_to_index)
    cls_feat, context_card_id, effect_card_id = _build_cls_features(
        state, select, your_index, id_to_index
    )
    sum_feat = _build_summary_tokens(state, your_index)
    discard_ids, discard_mask, prize_ids = _build_discard_prizes(
        state, your_index, id_to_index
    )

    # --- Logs: belief-module input ---
    logs = obs_dict.get("logs", [])
    log_feat, log_mask, log_len = _build_log_tokens(logs, your_index, id_to_index)

    # --- State: categorical token attributes ---
    tok_type, tok_owner, tok_zone, tok_mask = _build_tok_attrs(
        poke_card_id, hand_card_id, stadium_present
    )

    # --- Action: option references ---
    (
        opt_type,
        opt_src_idx,
        opt_tgt_idx,
        opt_card_id,
        opt_attack_idx,
        opt_scalar,
        opt_mask,
        index_remap,
        stop_column,
    ) = _build_option_tokens(
        select, ref_map, your_index, id_to_index, attack_id_to_index, action,
        state=state,
    )

    # --- Labels ---
    if action is not None:
        action_idx, action_len = _build_label(action, select, index_remap, stop_column)
    else:
        action_idx = np.full(O_MAX, -1, dtype=np.int64)
        action_len = np.int64(0)

    # --- Assemble full dict (A.4) ---
    return {
        # State — card identity & structural
        "poke_card_id": poke_card_id,
        "hand_card_id": hand_card_id,
        "stadium_card_id": stadium_card_id,
        "context_card_id": context_card_id,
        "effect_card_id": effect_card_id,
        "discard_ids": discard_ids,
        "discard_mask": discard_mask,
        "prize_ids": prize_ids,
        # State — dense features
        "poke_feat": poke_feat,
        "hand_feat": hand_feat,
        "sum_feat": sum_feat,
        "cls_feat": cls_feat,
        "stadium_present": stadium_present,
        # State — categorical token attributes
        "tok_type": tok_type,
        "tok_owner": tok_owner,
        "tok_zone": tok_zone,
        "tok_mask": tok_mask,
        # Action — option references & features
        "opt_type": opt_type,
        "opt_src_idx": opt_src_idx,
        "opt_tgt_idx": opt_tgt_idx,
        "opt_card_id": opt_card_id,
        "opt_attack_idx": opt_attack_idx,
        "opt_scalar": opt_scalar,
        "opt_mask": opt_mask,
        # Labels & bookkeeping
        "action_idx": action_idx,
        "action_len": action_len,
        "minCount": np.int64(select["minCount"]),
        "maxCount": np.int64(select["maxCount"]),
        "sel_type": np.int64(select["type"]),
        "sel_ctx": np.int64(select["context"]),
        "value_target": np.float32(value_target),
        "sample_weight": np.float32(sample_weight),
        "stop_column": np.int64(stop_column),
        # Logs (belief module)
        "log_feat": log_feat,
        "log_mask": log_mask,
        "log_len": log_len,
    }
