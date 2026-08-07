"""Featurizer core: obs_dict → tensor dict per TRANSFORMER_IL_SPEC.md Appendix A.

Pure NumPy — no PyTorch dependency.  One call per decision point.

Usage::

    tensors = featurize(obs_dict, vocab, action,
                        value_target=1.0, sample_weight=1.0)

Raises ``ValueError`` if ``select is None`` (deck-selection steps are excluded
— the deck is fixed, not predicted).
"""

import numpy as np

from ptcg_il.ref_map import build_ref_map, card_id_at

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
#: Attached-card capacities per Pokémon slot.  Both are taken from the divisors
#: the featurizer already applies to the corresponding counts, rather than
#: invented: ``poke_feat[17]`` normalises ``len(tools)`` by 2.0 and
#: ``poke_feat[16]`` normalises ``len(energyCards)`` by ``ENERGY_N``.  Measured
#: maxima over 50 906 Pokémon-observations are 1 and 5, so both caps have
#: headroom; overflow truncates, exactly as ``bench`` already does at 5.
T_MAX = 2
E_MAX = 12
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
F_CARD = 212  # 52 base + 29 ability keywords + 2 counts + 3 attacks × 43
F_ATK = 43    # 14 numeric + 29 attack keywords
F_POKE = 26
F_HAND = 7
F_SUM = 11
F_GLOBAL = 97
F_OPT = 8

# Columns 9:12 of a card static row are (basic, stage1, stage2) — see
# ``ptcg_mine.cards.card_static_row``.  Named here rather than written as a
# literal at each use because the MCTS determinizer reads it to decide which
# cards can legally be a face-down active: pointing at column 10 instead yields
# a plausible non-empty set that the engine then refuses one search root at a
# time.  ``search_infer`` ships standalone in the submission bundle and cannot
# import ``ptcg_rl``, so this module is the one place both sides can share.
CARD_FEAT_BASIC_COL = 9

#: ``card_static_row[2:9]`` is the cardType one-hot (see ptcg_mine.cards).
CARD_FEAT_CARDTYPE_START = 2
CARD_TYPE_ITEM = 1
CARD_TYPE_SUPPORTER = 3
CARD_TYPE_STADIUM = 4
CARD_TYPE_BASIC_ENERGY = 5
CARD_TYPE_SPECIAL_ENERGY = 6

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


def _raw_card(card_id) -> int:
    """Pass an engine card id straight through; ``None`` → PAD.

    Card identity reaches the model as static features looked up by **engine
    card id**, not as a vocab index.  Going via the vocab is what used to
    destroy out-of-vocab cards: ``id_to_index.get(id, UNKNOWN_CARD)`` mapped
    every unseen card to index 1, and index 1 dereferences to the string
    ``"UNKNOWN"``, which has no engine features — so the card arrived as an
    all-zero row, indistinguishable from an empty slot.  ``engine_card_features``
    covers all 1267 engine cards against a ~294-card vocab, so 973 of them were
    unrepresentable.  That never fires on the training corpus (vocab is built
    ``all_corpus``) and only shows up in live play.

    PAD stays 0 and no real engine card id is 0, so mask logic keyed on
    ``== PAD_CARD`` is unaffected.
    """
    if card_id is None:
        return PAD_CARD
    return int(card_id)


def _raw_attack(attack_id) -> int:
    """Pass an engine attack id straight through; ``None`` → PAD_ATTACK.

    Same reasoning as :func:`_raw_card` — 1341 of 1556 engine attacks sit
    outside the mined attack vocab and used to collapse to PAD_ATTACK.
    """
    if attack_id is None:
        return PAD_ATTACK
    return int(attack_id)


def _card_feat(card_id, engine_features: dict | None) -> np.ndarray:
    """Return [F_CARD] static features for *card_id*, or zeros if unknown/None."""
    if engine_features is None or card_id is None:
        return np.zeros(F_CARD, dtype=np.float32)
    feat = engine_features.get(card_id)
    if feat is not None:
        return np.asarray(feat, dtype=np.float32)
    return np.zeros(F_CARD, dtype=np.float32)


def _attack_feat(attack_id, engine_features: dict | None) -> np.ndarray:
    """Return [14] static features for *attack_id*, or zeros if unknown/None."""
    if engine_features is None or attack_id is None:
        return np.zeros(F_ATK, dtype=np.float32)
    feat = engine_features.get(attack_id)
    if feat is not None:
        return np.asarray(feat, dtype=np.float32)
    return np.zeros(F_ATK, dtype=np.float32)


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


# Duplicated from ptcg_mine.damage rather than imported: this module is vendored
# into the Kaggle bundle, where ptcg_mine does not exist.  The constants are
# engine-measured; python/tests/test_damage.py pins them.
_WEAKNESS_MULT = 2.0
_RESISTANCE_DELTA = -30.0

# Slices into a card_static_row (see ptcg_mine.cards).  Named because an
# off-by-one here reads the wrong energy type and silently mis-scores weakness.
_CARD_ENERGYTYPE = slice(12, 24)
_CARD_WEAKNESS = slice(24, 36)
_CARD_RESISTANCE = slice(36, 48)
# Derived from F_CARD and F_ATK so a future K_EFFECT change can't drift the
# offset without also bumping both dims.
CARD_ATTACK_BLOCK_START = F_CARD - 3 * F_ATK  # == 83


def _onehot_index(vec) -> int | None:
    """Index of the set bit in a one-hot slice, or None if all-zero."""
    nz = np.flatnonzero(np.asarray(vec) > 0.5)
    return int(nz[0]) if nz.size else None


def _effective_damage(base, atk_type, weakness, resistance) -> float:
    """Base damage adjusted for weakness/resistance.  Mirrors ptcg_mine.damage."""
    dmg = float(base)
    if dmg <= 0.0:
        return 0.0
    if weakness is not None and atk_type is not None and weakness == atk_type:
        dmg *= _WEAKNESS_MULT
    if resistance is not None and atk_type is not None and resistance == atk_type:
        dmg += _RESISTANCE_DELTA
    return max(dmg, 0.0)


def _attack_is_affordable(cost_hist, attached_hist) -> bool:
    """Colorless (index 0) accepts any energy; colored costs need their colour."""
    cost = np.asarray(cost_hist, dtype=np.float64)
    have = np.asarray(attached_hist, dtype=np.float64)
    if cost[1:].sum() > 0 and np.any(have[1:] < cost[1:]):
        return False
    return have.sum() >= cost.sum()


def _best_damage(attacker_row, defender_row, attached_hist, require_affordable):
    """Best damage the attacker can deal to the defender, in raw HP units.

    *attacker_row* / *defender_row* are card_static_rows; *attached_hist* is the
    attacker's 12-wide attached-energy histogram in raw counts.  Reads the three
    embedded attack blocks rather than the attack table, so no attack-id lookup
    is needed.
    """
    if attacker_row is None or defender_row is None:
        return 0.0
    atk_type = _onehot_index(attacker_row[_CARD_ENERGYTYPE])
    weakness = _onehot_index(defender_row[_CARD_WEAKNESS])
    resistance = _onehot_index(defender_row[_CARD_RESISTANCE])

    best = 0.0
    for i in range(3):
        off = CARD_ATTACK_BLOCK_START + i * F_ATK
        block = attacker_row[off:off + F_ATK]
        base = float(block[0]) * ATKDMG_N
        if base <= 0.0:
            continue
        if require_affordable:
            cost = np.asarray(block[1:13], dtype=np.float64) * ATKCOST_N
            if not _attack_is_affordable(cost, attached_hist):
                continue
        best = max(best, _effective_damage(base, atk_type, weakness, resistance))
    return best


def _ko_pressure(poke_card_feat, poke_feat) -> tuple[float, float, float, float]:
    """(my_ratio, can_ko_opp, opp_ratio, opp_can_ko_me).

    Poke slot layout (A.1): 0 = my active, 6 = opp active.
    My side requires affordability; opp side does not.
    """
    mine, opp = poke_card_feat[0], poke_card_feat[6]
    if not np.asarray(mine).any() or not np.asarray(opp).any():
        return 0.0, 0.0, 0.0, 0.0

    my_energy = np.asarray(poke_feat[0][3:15], dtype=np.float64) * ENERGY_N
    my_hp = float(poke_feat[0][0]) * HP_N
    opp_hp = float(poke_feat[6][0]) * HP_N

    my_dmg = _best_damage(mine, opp, my_energy, require_affordable=True)
    opp_dmg = _best_damage(opp, mine, None, require_affordable=False)

    my_ratio = min(my_dmg / opp_hp, 2.0) if opp_hp > 0 else 0.0
    opp_ratio = min(opp_dmg / my_hp, 2.0) if my_hp > 0 else 0.0
    return (my_ratio, 1.0 if my_ratio >= 1.0 else 0.0,
            opp_ratio, 1.0 if opp_ratio >= 1.0 else 0.0)


def _attack_damage_ratio(attack_id, tgt_slot, poke_card_feat, poke_feat,
                         engine_attack_features) -> float:
    """Damage this attack would deal to *tgt_slot*, over that slot's current HP.

    Clipped to [0, 2].  Returns 0.0 when the attack or target is unresolvable,
    which is the same "no information" signal a PAD row carries.
    """
    if engine_attack_features is None or attack_id is None:
        return 0.0
    arow = engine_attack_features.get(int(attack_id))
    if arow is None:
        return 0.0
    if not (0 <= tgt_slot < P_MAX):
        return 0.0

    defender = poke_card_feat[tgt_slot]
    attacker = poke_card_feat[0]            # my active is always the attacker
    if not np.asarray(defender).any() or not np.asarray(attacker).any():
        return 0.0

    tgt_hp = float(poke_feat[tgt_slot][0]) * HP_N
    if tgt_hp <= 0.0:
        return 0.0

    dmg = _effective_damage(
        float(arow[0]) * ATKDMG_N,
        _onehot_index(attacker[_CARD_ENERGYTYPE]),
        _onehot_index(defender[_CARD_WEAKNESS]),
        _onehot_index(defender[_CARD_RESISTANCE]),
    )
    return min(dmg / tgt_hp, 2.0)


def _attached_ids(cards, capacity: int) -> np.ndarray:
    """Engine card ids of an attached-card list, PAD-padded to *capacity*.

    ``tools`` and ``energyCards`` are both ``[Card]`` on a Pokémon, and the
    featurizer used to keep only their lengths (``poke_feat[17]`` /
    ``poke_feat[16]``).  A count cannot distinguish a defensive Tool from an
    offensive one, nor a Special Energy from a basic of the same type — the
    energy *type* histogram covers the latter only for basics.
    """
    out = np.full(capacity, PAD_CARD, dtype=np.int64)
    for i, card in enumerate(cards or []):
        if i >= capacity:
            break
        if isinstance(card, dict):
            out[i] = _raw_card(card.get("id"))
    return out


def _build_poke_tokens(
    state: dict, your_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the per-Pokémon-slot tensors.

    Returns ``(poke_card_id[P_MAX], poke_feat[P_MAX, F_POKE],
    poke_tool_ids[P_MAX, T_MAX], poke_energy_ids[P_MAX, E_MAX])``.

    ``poke_card_id`` holds raw **engine** card ids (0 = empty slot); ``featurize``
    turns them into ``poke_card_feat``.  They are not vocab indices.  The two
    attachment arrays work the same way and become ``poke_tool_feat`` /
    ``poke_energy_feat``.

    Slot layout (A.1): 0=my_active, 1..5=my_bench[0..4],
    6=opp_active, 7..11=opp_bench[0..4].
    """
    poke_card_id = np.full(P_MAX, PAD_CARD, dtype=np.int64)
    poke_feat = np.zeros((P_MAX, F_POKE), dtype=np.float32)
    poke_tool_ids = np.full((P_MAX, T_MAX), PAD_CARD, dtype=np.int64)
    poke_energy_ids = np.full((P_MAX, E_MAX), PAD_CARD, dtype=np.int64)

    for pi, player_idx in enumerate([your_index, 1 - your_index]):
        player = state["players"][player_idx]
        is_me = pi == 0
        base_row = 0 if is_me else 6

        # Active slot
        active_list = player["active"]
        if len(active_list) > 0 and active_list[0] is not None:
            poke = active_list[0]
            row = base_row
            poke_card_id[row] = _raw_card(poke["id"])
            poke_tool_ids[row] = _attached_ids(poke.get("tools"), T_MAX)
            poke_energy_ids[row] = _attached_ids(poke.get("energyCards"), E_MAX)
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
            poke_card_id[row] = _raw_card(poke["id"])
            poke_tool_ids[row] = _attached_ids(poke.get("tools"), T_MAX)
            poke_energy_ids[row] = _attached_ids(poke.get("energyCards"), E_MAX)
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

    return poke_card_id, poke_feat, poke_tool_ids, poke_energy_ids


def _build_hand_tokens(
    state: dict, your_index: int,
    engine_card_features: dict | None = None,
    evolution_map: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build hand_card_id[H_MAX] int64 and hand_feat[H_MAX, F_HAND] float32.

    Cards are packed to the front; remaining slots are PAD (all-zero).

    hand_feat = [idx/H_MAX, dup_count/COUNT_N, can_bench, can_evolve,
                 can_attach_energy, can_play_supporter, can_play_stadium].

    The five legality flags are deterministic and the engine already enumerates
    legal plays at a MAIN select -- their value is at *non-MAIN* decisions and
    for lookahead, e.g. seeing while choosing a discard that one candidate is a
    Supporter not yet played this turn.
    """
    hand_card_id = np.full(H_MAX, PAD_CARD, dtype=np.int64)
    hand_feat = np.zeros((H_MAX, F_HAND), dtype=np.float32)

    player = state["players"][your_index]
    hand = player.get("hand")
    if hand is None:
        return hand_card_id, hand_feat

    id_counts: dict[int, int] = {}
    for card in hand:
        cid = card["id"]
        id_counts[cid] = id_counts.get(cid, 0) + 1

    bench_has_room = len(player["bench"]) < int(player.get("benchMax", 5))
    energy_free = not state.get("energyAttached", False)
    supporter_free = not state.get("supporterPlayed", False)
    stadium_free = not state.get("stadiumPlayed", False)

    # Ids of my in-play Pokemon that did not arrive this turn -- only those can
    # be evolved.
    evolvable_ids: set[int] = set()
    in_play = list(player["active"] or []) + list(player["bench"] or [])
    for poke in in_play:
        if poke is None or poke.get("appearThisTurn", False):
            continue
        evolvable_ids.add(int(poke["id"]))

    for i, card in enumerate(hand):
        if i >= H_MAX:
            break
        cid = card["id"]
        hand_card_id[i] = _raw_card(cid)
        f = hand_feat[i]
        f[0] = _clip_norm(float(i), HAND_N)
        f[1] = _clip_norm(float(id_counts.get(cid, 1)), COUNT_N)

        crow = None
        if engine_card_features is not None and cid is not None:
            crow = engine_card_features.get(int(cid))
        if crow is None:
            continue

        is_basic = crow[CARD_FEAT_BASIC_COL] > 0.5
        ctype_slice = crow[CARD_FEAT_CARDTYPE_START:CARD_FEAT_CARDTYPE_START + 7]

        def _is(ct):
            return ctype_slice[ct] > 0.5

        f[2] = 1.0 if (is_basic and _is(0) and bench_has_room) else 0.0
        if evolution_map is not None:
            pre = evolution_map.get(int(cid)) or ()
            f[3] = 1.0 if any(p in evolvable_ids for p in pre) else 0.0
        f[4] = 1.0 if ((_is(CARD_TYPE_BASIC_ENERGY) or _is(CARD_TYPE_SPECIAL_ENERGY))
                       and energy_free) else 0.0
        f[5] = 1.0 if (_is(CARD_TYPE_SUPPORTER) and supporter_free) else 0.0
        f[6] = 1.0 if (_is(CARD_TYPE_STADIUM) and stadium_free) else 0.0

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
    state: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build stadium_card_id[1] int64 and stadium_present[1] float32."""
    stadium = state.get("stadium", [])
    if len(stadium) > 0:
        sid = _raw_card(stadium[0]["id"])
        return (
            np.array([sid], dtype=np.int64),
            np.array([1.0], dtype=np.float32),
        )
    return (
        np.array([PAD_CARD], dtype=np.int64),
        np.array([0.0], dtype=np.float32),
    )


def _build_cls_features(
    state: dict, select: dict, your_index: int,
    poke_card_feat: np.ndarray | None = None,
    poke_feat: np.ndarray | None = None,
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
        [_raw_card(ctx_card["id"] if ctx_card else None)],
        dtype=np.int64,
    )
    effect_card_id = np.array(
        [_raw_card(eff_card["id"] if eff_card else None)],
        dtype=np.int64,
    )

    # [93:97] KO pressure.  Zero when either feature table is absent -- an
    # all-zero block is the same "no information" signal a PAD row carries.
    if poke_card_feat is not None and poke_feat is not None:
        cls_feat[93:97] = _ko_pressure(poke_card_feat, poke_feat)

    return cls_feat, context_card_id, effect_card_id


def _build_discard_prizes(
    state: dict, your_index: int
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
            discard_ids[pi, i] = _raw_card(card["id"])
            discard_mask[pi, i] = True

        # Revealed prizes
        prizes = player["prize"]
        for i, card in enumerate(prizes):
            if i >= PZ_MAX:
                break
            if card is not None:
                prize_ids[pi, i] = _raw_card(card["id"])
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
    action: list[int] | None = None,
    state: dict | None = None,
    poke_card_feat: np.ndarray | None = None,
    poke_feat: np.ndarray | None = None,
    engine_attack_features: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int], int]:
    """Build option tensors per A.4/A.7.

    When the raw option list exceeds O_MAX, the chosen actions are always
    included within the O_MAX window so that no training samples are silently
    dropped.  Returns an *index_remap* dict mapping old option indices to new
    positions (identity when no reordering occurred).

    For multi-select decisions (maxCount > 1), a STOP column is appended
    after the regular options so the model can learn when to stop picking.
    *stop_column* is the index of that column (or -1 for single-select).

    Returns (opt_type, opt_src_idx, opt_tgt_idx, opt_card_id,
             opt_attack_idx, opt_scalar, opt_mask, index_remap, stop_column).
    """
    options = select["option"]
    n_total = len(options)
    max_count = int(select.get("maxCount", 1))
    min_count = int(select.get("minCount", 1))
    is_multi = max_count > 1
    # A STOP column is what makes "take nothing" expressible.  Multi-select has
    # always had one; `minCount == 0` needs one just as much even when
    # `maxCount == 1`, because declining is legal there too -- 11.9% of
    # single-selects on this corpus, 86% of them deck searches.  Without it the
    # engine's own "you may" is unrepresentable: the argmax has to name a real
    # option, and an expert who declined contributes no target at all.
    wants_stop = is_multi or min_count == 0

    # ---- choose which option indices to keep (old → new) ----
    # Reserve one slot for the STOP column whenever there will be one
    max_regular = (O_MAX - 1) if wants_stop else O_MAX

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
            PAD when `card_id_at` reports the card is hidden -- face-down prizes
            must stay PAD or we leak.

            Deck slots are the one zone the *select* can resolve where the state
            cannot: during a search the engine hands the acting player their own
            deck in `select["deck"]` and the options index into it.  Without
            that payload every option in a search is byte-identical
            (`type=3, src=-1, tgt=-1, card=PAD, scalar=0`), so `option_groups`
            merges them all into one class, the group-marginal NLL is exactly
            `-log(1) = 0`, and 11% of the corpus produces no gradient at all
            while the agent picks a card it cannot see.
            """
            if state is None or area is None or idx is None:
                return
            raw = card_id_at(state, area, player_idx, idx,
                             select_deck=select.get("deck"))
            if raw is not None:
                opt_card_id[new_j] = _raw_card(raw)

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
            opt_attack_idx[new_j] = _raw_attack(opt.get("attackId"))
            # Some attacks target specific bench slots (snipe effects)
            in_play_area = opt.get("inPlayArea")
            in_play_idx = opt.get("inPlayIndex")
            if in_play_area is not None and in_play_idx is not None:
                opt_tgt_idx[new_j] = _ref(in_play_area, 1 - your_index, in_play_idx)
            # Damage preview.  Target is the snipe slot when the option names
            # one, else the opponent's Active (poke slot 6).  opt_tgt_idx is a
            # state-token row index (1..12), so poke slot = row - 1.
            tgt_slot = (int(opt_tgt_idx[new_j]) - 1) if opt_tgt_idx[new_j] > 0 else 6
            ratio = _attack_damage_ratio(
                opt.get("attackId"), tgt_slot, poke_card_feat, poke_feat,
                engine_attack_features,
            )
            opt_scalar[new_j, 6] = ratio
            opt_scalar[new_j, 7] = 1.0 if ratio >= 1.0 else 0.0

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
                opt_card_id[new_j] = _raw_card(cid)
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
                opt_card_id[new_j] = _raw_card(cid)
            else:
                _set_card(area, player_idx, index)

        elif otype in (1, 2, 14, 15, 16):  # YES, NO, END, SKILL, SPECIAL_CONDITION
            # src = -1, tgt = -1 (constant-type options)
            cid = opt.get("cardId")
            if cid is not None:
                opt_card_id[new_j] = _raw_card(cid)

        elif otype == 0:  # NUMBER
            # src = -1, tgt = -1; scalar carries the number
            cid = opt.get("cardId")
            if cid is not None:
                opt_card_id[new_j] = _raw_card(cid)

    # ---- STOP column (multi-select, or any select that may be declined) ----
    stop_column = -1
    if wants_stop and n_opts < O_MAX:
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


def option_groups(
    opt_type: np.ndarray,
    opt_src_idx: np.ndarray,
    opt_tgt_idx: np.ndarray,
    opt_card_feat: np.ndarray,
    opt_attack_feat: np.ndarray,
    opt_scalar: np.ndarray,
    opt_mask: np.ndarray,
) -> np.ndarray:
    """Equivalence classes over option slots: same id == identical model input.

    The pointer head reads exactly ``opt_type``, the gathered rows named by
    ``opt_src_idx``/``opt_tgt_idx``, ``opt_card_feat``, ``opt_attack_feat`` and
    ``opt_scalar`` (``model/pointer.py:117-130``).  Two options agreeing on all
    of them produce the same logit by construction, so a label that names one
    of them and not the other asks for a distinction the network cannot make.
    Measured on ``train-00000``: 10.0% of options and 15.6% of samples contain
    such a pair, and 13.9% of samples have a label that splits one.

    Card and attack features are compared **at fp32**, because that is the
    precision the model reads: shards store the ids (``CARD_FEAT_SOURCES``) and
    ``ShardDataset`` re-gathers from the fp32 static table.  They used to be
    stored materialised at fp16, and grouping had to match that -- comparing at
    fp32 then would have called two options distinct that were byte-identical
    by the time they reached training.  Now the reverse holds: comparing at
    fp16 would merge two options the network *can* tell apart, and a
    group-marginal CE would stop asking it to.

    Returns ``int64[O_MAX]``: ``0..G-1`` in first-appearance order for valid
    slots, ``-1`` for masked slots (so a padded slot never matches anything).
    """
    groups = np.full(O_MAX, -1, dtype=np.int64)
    valid = np.flatnonzero(opt_mask)
    if valid.size == 0:
        return groups

    ints = np.stack(
        [opt_type[valid], opt_src_idx[valid], opt_tgt_idx[valid]], axis=1
    ).astype(np.int64)
    feats = np.concatenate(
        [
            opt_card_feat[valid].astype(np.float32),
            opt_attack_feat[valid].astype(np.float32),
        ],
        axis=1,
    )
    scal = opt_scalar[valid].astype(np.float32)

    lookup: dict[tuple[bytes, bytes, bytes], int] = {}
    for pos, slot in enumerate(valid):
        key = (ints[pos].tobytes(), feats[pos].tobytes(), scal[pos].tobytes())
        gid = lookup.setdefault(key, len(lookup))
        groups[slot] = gid
    return groups


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

    Single-select is different: it is *one* step, so STOP is the label only when
    the expert declined outright and is never appended after a pick.  Appending
    it would make ``action_len == 2`` for a decision that admits one pick, which
    every ``maxCount == 1`` consumer (``train/loop.py``, ``train/eval.py``,
    ``meta.parquet``) reads as a multi-select.
    """
    action_idx = np.full(O_MAX, -1, dtype=np.int64)
    max_count = select["maxCount"]
    is_multi = int(max_count) > 1

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

    # Append STOP: after the last pick for multi-select, or *as* the label for a
    # single-select the expert declined (write_pos == 0 means no pick was made).
    if stop_column >= 0 and write_pos < O_MAX and (is_multi or write_pos == 0):
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

        # card id — raw engine id, so log_card_feat resolves OOV cards too
        card_idx = _raw_card(log.get("cardId"))

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


def _ids_to_feat(ids: np.ndarray, engine_features: dict | None,
                 feat_dim: int,
                 index_to_id: dict | list | None = None) -> np.ndarray:
    """Convert int64 vocab-index array to float32 feature array.

    *ids* contains vocab indices (or engine card IDs for attacks).  Uses
    *index_to_id* to map vocab index → engine card id → static features.
    For attack indices or when *index_to_id* is None, treats *ids* as
    engine IDs directly.

    *index_to_id* may be either form the corpus produces: ``vocab.json`` stores
    it as the **list** ``["PAD", "UNKNOWN", id2, ...]`` (so slots 0 and 1 are
    strings, not ids), while the attack mapping is inverted into a dict.  The
    lookup is resolved once, outside the per-element loop.

    Returns zeros for PAD (0), unknown, or None entries.
    """
    result = np.zeros(ids.shape + (feat_dim,), dtype=np.float32)
    if engine_features is None:
        return result

    if index_to_id is None:
        def lookup(v):
            return v
    elif isinstance(index_to_id, dict):
        lookup = index_to_id.get
    else:
        def lookup(v):
            return index_to_id[v] if 0 <= v < len(index_to_id) else None

    for idx in np.ndindex(ids.shape):
        raw_id = ids[idx]
        if raw_id is None or int(raw_id) <= 0:
            continue
        # Map vocab index → engine card id, then look up features.  UNKNOWN
        # (index 1) resolves to the string "UNKNOWN" in the list form and has
        # no engine features, so it correctly stays zeros.
        engine_id = lookup(int(raw_id))
        if isinstance(engine_id, (int, np.integer)):
            feat = engine_features.get(int(engine_id))
            if feat is not None:
                result[idx] = feat
    return result


#: Shard storage contract: ``feat_key -> (id_key, table)``.
#:
#: Every one of these is a gather from a frozen static table -- 1.1 MB for
#: cards, 0.3 MB for attacks -- so storing the *result* per row is pure
#: redundancy.  It is also the dominant cost: materialised, these keys are 93%
#: of a shard's decompressed bytes (5.8 GB of 6.2 GB for 50k samples), which
#: put the train split's mmap cache at 47 GB against 30 GB of RAM.  ``.npz``
#: hides this on disk because the arrays are ~0.5% nonzero and compress ~240x.
#:
#: So the writer stores the id and the reader re-gathers, exactly as the
#: ``bel_*`` labels are stored sparsely and densified on the way into a batch.
#: ``log_card_feat`` is absent by design -- ``log_feat[:, 2]`` already carries
#: its ids, so it needs no key of its own.
CARD_FEAT_SOURCES: dict[str, tuple[str, str]] = {
    "poke_card_feat": ("poke_card_id", "card"),
    "poke_tool_feat": ("poke_tool_ids", "card"),
    "poke_energy_feat": ("poke_energy_ids", "card"),
    "hand_card_feat": ("hand_card_id", "card"),
    "stadium_card_feat": ("stadium_card_id", "card"),
    "context_card_feat": ("context_card_id", "card"),
    "effect_card_feat": ("effect_card_id", "card"),
    "discard_card_feat": ("discard_ids", "card"),
    "prize_card_feat": ("prize_ids", "card"),
    "opt_card_feat": ("opt_card_id", "card"),
    "opt_attack_feat": ("opt_attack_idx", "attack"),
}

#: ``log_card_feat`` is rebuilt from this column of ``log_feat`` instead.
LOG_CARD_ID_COLUMN = 2


def build_static_table(engine_features: dict | None, feat_dim: int) -> np.ndarray:
    """Dense ``float32[max_id + 1, feat_dim]`` view of a sparse feature dict.

    Row *i* holds the features of engine id *i*, or zeros when the engine has
    none -- which is what makes the gather below a drop-in for the
    ``engine_features.get(id) is None`` branch of ``_ids_to_feat``.
    """
    n = (max(engine_features) + 1) if engine_features else 1
    table = np.zeros((max(n, 1), feat_dim), dtype=np.float32)
    for cid, feat in (engine_features or {}).items():
        table[int(cid)] = feat
    return table


def gather_static_feats(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Vectorised ``_ids_to_feat(ids, ..., index_to_id=None)``.

    Returns ``float32[*ids.shape, feat_dim]``.  PAD (0), negative ids, and ids
    past the end of *table* all gather zeros, matching the loop's skip
    conditions -- an out-of-range id must not wrap around to another card's
    features, which is why the clamp is paired with an explicit re-zero.
    """
    ids = np.asarray(ids)
    valid = (ids > 0) & (ids < table.shape[0])
    out = table[np.where(valid, ids, 0)]
    out[~valid] = 0.0
    return out


def featurize(
    obs_dict: dict,
    vocab: dict,
    action: list[int] | None = None,
    value_target: float = 0.0,
    sample_weight: float = 1.0,
    engine_card_features: dict | None = None,
    engine_attack_features: dict | None = None,
    evolution_map: dict | None = None,
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
        Mined vocab dict.  Retained for API compatibility and callers that
        still inspect it; card and attack identity no longer routes through it
        — features are looked up by raw engine id (see :func:`_raw_card`).
    value_target : float
        +1 for win, -1 for loss (§1.1).
    sample_weight : float
        Per-sample weight (computed from context/archetype/outcome, C.3).
    engine_card_features : dict or None
        ``{engine_card_id: np.ndarray[F_CARD]}`` for ALL engine cards.  Every
        card is represented purely by these static features — there are no
        learned id embeddings, so a card the vocab has never seen is still
        described exactly.  When None, all ``*_card_feat`` keys are zeros.

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

    your_index = state["yourIndex"]

    # Build ref map for option resolution
    ref_map = build_ref_map(obs_dict)

    # --- State: card identity ---
    poke_card_id, poke_feat, poke_tool_ids, poke_energy_ids = _build_poke_tokens(
        state, your_index
    )
    # Hoist the card-feature conversion early — _build_cls_features and
    # _build_option_tokens both need the same array, and the result dict
    # reuses it instead of calling _cfeat(poke_card_id) a second time.
    poke_card_feat = _ids_to_feat(poke_card_id, engine_card_features, F_CARD, None)
    hand_card_id, hand_feat = _build_hand_tokens(
        state, your_index, engine_card_features, evolution_map,
    )
    stadium_card_id, stadium_present = _build_stadium_token(state)
    cls_feat, context_card_id, effect_card_id = _build_cls_features(
        state, select, your_index, poke_card_feat, poke_feat,
    )
    sum_feat = _build_summary_tokens(state, your_index)
    discard_ids, discard_mask, prize_ids = _build_discard_prizes(
        state, your_index
    )

    # --- Logs: belief-module input ---
    logs = obs_dict.get("logs", [])
    log_feat, log_mask, log_len = _build_log_tokens(logs, your_index)

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
        select, ref_map, your_index, action,
        state=state,
        poke_card_feat=poke_card_feat,
        poke_feat=poke_feat,
        engine_attack_features=engine_attack_features,
    )

    # --- Labels ---
    if action is not None:
        action_idx, action_len = _build_label(action, select, index_remap, stop_column)
    else:
        action_idx = np.full(O_MAX, -1, dtype=np.int64)
        action_len = np.int64(0)

    # --- Assemble full dict (A.4) ---
    # ── Convert all card/attack IDs to feature vectors ─────────────────
    # The ids above are raw *engine* ids, so no index_to_id hop: passing None
    # makes _ids_to_feat look them up directly.  Every card the engine knows
    # about therefore gets its real features, in-vocab or not.
    _cfeat = lambda ids: _ids_to_feat(ids, engine_card_features, F_CARD, None)
    _afeat = lambda ids: _ids_to_feat(ids, engine_attack_features, F_ATK, None)
    opt_card_feat = _cfeat(opt_card_id)
    opt_attack_feat = _afeat(opt_attack_idx)
    opt_group = option_groups(
        opt_type, opt_src_idx, opt_tgt_idx,
        opt_card_feat, opt_attack_feat, opt_scalar, opt_mask,
    )
    result = {
        # State — card features (float32, not int64 ids)
        "poke_card_feat": poke_card_feat,
        # Attached Tools / Energy cards, per Pokémon slot.  Pooled into the
        # Pokémon token by TokenEmbedder, the same way the discard pile is
        # pooled into the summary token.
        "poke_tool_feat": _cfeat(poke_tool_ids),
        "poke_energy_feat": _cfeat(poke_energy_ids),
        "hand_card_feat": _cfeat(hand_card_id),
        "stadium_card_feat": _cfeat(stadium_card_id),
        "context_card_feat": _cfeat(context_card_id),
        "effect_card_feat": _cfeat(effect_card_id),
        "discard_card_feat": _cfeat(discard_ids),
        "discard_mask": discard_mask,
        "prize_card_feat": _cfeat(prize_ids),
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
        "opt_card_feat": opt_card_feat,
        "opt_attack_feat": opt_attack_feat,
        "opt_scalar": opt_scalar,
        "opt_mask": opt_mask,
        "opt_group": opt_group,
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
        # Logs (belief module) — card features extracted from log entries
        "log_feat": log_feat,
        "log_mask": log_mask,
        "log_len": log_len,
        # log_feat is per-sample [L_LOG_MAX, LOG_FEAT_DIM]; column 2 is the
        # card vocab index (belief.py does the batched [:, :, 2] equivalent).
        "log_card_feat": _cfeat(log_feat[:, LOG_CARD_ID_COLUMN].astype(np.int64)),
        # Raw ids behind every *_card_feat above.  The model never reads these
        # -- it consumes the features -- but the shard writer stores them
        # *instead of* the features (see CARD_FEAT_SOURCES) and ShardDataset
        # re-gathers on read.  diagnose.py reads them too, to report PAD and
        # UNKNOWN rates that a materialised feature row cannot distinguish.
        "poke_card_id": poke_card_id,
        "poke_tool_ids": poke_tool_ids,
        "poke_energy_ids": poke_energy_ids,
        "hand_card_id": hand_card_id,
        "stadium_card_id": stadium_card_id,
        "context_card_id": context_card_id,
        "effect_card_id": effect_card_id,
        "discard_ids": discard_ids,
        "prize_ids": prize_ids,
        "opt_card_id": opt_card_id,
        "opt_attack_idx": opt_attack_idx,
    }
    return result
