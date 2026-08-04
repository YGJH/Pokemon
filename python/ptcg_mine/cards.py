"""Engine-derived static feature tables for cards and attacks.

Layout is exact Appendix A.3 (feature slices) / A.2 (normalizers) of
TRANSFORMER_IL_SPEC.md:
  card_static_row[94]  = base[52] + 3 × attack_static_row[14]
  base[52]             = [hp/HP_N, retreat/RETREAT_N, cardType-onehot(7),
                          stage-onehot(3), energyType-onehot(12),
                          weakness-onehot(12), resistance-onehot(12),
                          [ex, megaEx, tera, aceSpec](4)]
  attack_static_row[14] = [damage/ATKDMG_N, energy-cost histogram(12)/ATKCOST_N,
                           len(energies)/ATKCOST_N]

Engine access: `all_card_data()` / `all_attack()` live in the bundled `cg`
package at pokemon-tcg-ai-battle/sample_submission/sample_submission/cg;
`load_engine()` adds that directory's parent to sys.path and imports them.
"""

import sys
from pathlib import Path

import numpy as np

HP_N = 400.0
RETREAT_N = 4.0
ATKDMG_N = 350.0
ATKCOST_N = 5.0

N_CARDTYPE = 7
N_ENERGY = 12

from ptcg_il.featurizer import F_ATK, F_CARD
from ptcg_mine.keywords import K_EFFECT, ability_keyword_row, attack_keyword_row

# ptcg_il.featurizer owns the dims but cannot import K_EFFECT (it is vendored
# into the Kaggle bundle, where ptcg_mine does not exist).  Assert agreement
# here instead: appending a keyword without bumping the featurizer would
# otherwise emit a row of the old width, which every downstream shape check
# accepts until the first forward pass.
assert F_ATK == 14 + K_EFFECT, (
    f"F_ATK={F_ATK} in ptcg_il.featurizer disagrees with K_EFFECT={K_EFFECT} "
    f"in ptcg_mine.keywords (expected {14 + K_EFFECT})"
)
assert F_CARD == 52 + K_EFFECT + 2 + 3 * F_ATK, (
    f"F_CARD={F_CARD} in ptcg_il.featurizer disagrees with K_EFFECT={K_EFFECT} "
    f"(expected {52 + K_EFFECT + 2 + 3 * F_ATK})"
)

#: Offset of the first embedded attack block inside ``card_static_row``.
CARD_ATTACK_BLOCK_START = 52 + K_EFFECT + 2   # == 83

import logging

logger = logging.getLogger(__name__)

_ENGINE_DIR = (
    Path(__file__).resolve().parent.parent
    / "pokemon-tcg-ai-battle"
    / "sample_submission"
    / "sample_submission"
)


def load_engine():
    """Add the bundled `cg` package to sys.path and return
    (all_card_data(), all_attack())."""
    engine_dir = str(_ENGINE_DIR)
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    from cg.api import all_attack, all_card_data

    return all_card_data(), all_attack()


def _onehot(index: int | None, size: int) -> np.ndarray:
    """One-hot vector of length `size`; all-zero if index is None."""
    vec = np.zeros(size, dtype=np.float32)
    if index is not None:
        vec[int(index)] = 1.0
    return vec


def card_static_row(card, attacks_by_id: dict) -> np.ndarray:
    """float32[94] static feature row for a CardData.

    Layout: 52 base features + 3 attacks × 14 (damage, energy cost, cost count).
    Attacks beyond the card's actual attacks are zero-padded.

    ``card.attacks`` holds attack *ids*, not Attack objects, so *attacks_by_id*
    (``{attackId: Attack}``) is required to resolve them.  It is a required
    argument on purpose: defaulting it to ``{}`` would silently emit a 94-dim
    row whose attack half is all zeros, which no assertion downstream catches.
    """
    row = np.zeros(F_CARD, dtype=np.float32)
    # Base features (0:52)
    row[0] = card.hp / HP_N
    row[1] = card.retreatCost / RETREAT_N
    row[2:9] = _onehot(card.cardType, N_CARDTYPE)
    row[9:12] = [float(card.basic), float(card.stage1), float(card.stage2)]
    row[12:24] = _onehot(card.energyType, N_ENERGY)
    row[24:36] = _onehot(card.weakness, N_ENERGY)
    row[36:48] = _onehot(card.resistance, N_ENERGY)
    row[48:52] = [float(card.ex), float(card.megaEx), float(card.tera), float(card.aceSpec)]
    # Ability keywords + counts (52:83)
    row[52:52 + K_EFFECT] = ability_keyword_row(card)
    skills = getattr(card, "skills", None) or []
    attack_ids = getattr(card, "attacks", []) or []
    row[52 + K_EFFECT] = min(len(skills), 3) / 3.0
    row[52 + K_EFFECT + 1] = min(len(attack_ids), 3) / 3.0
    # Attack features (83:212) — up to 3 attacks, each F_ATK, same layout as
    # attack_static_row so the two views of an attack cannot drift apart.
    for ai in range(min(len(attack_ids), 3)):
        atk = attacks_by_id.get(int(attack_ids[ai]))
        if atk is None:
            continue
        offset = CARD_ATTACK_BLOCK_START + ai * F_ATK
        row[offset:offset + F_ATK] = attack_static_row(atk)
    return row


def attack_static_row(attack) -> np.ndarray:
    """float32[14] static feature row for an Attack, per Appendix A.3.

    The energy-cost histogram is divided by ``ATKCOST_N``, matching the
    ``count/ENERGY_N`` treatment the featurizer already gives the *attached*
    energy histogram in ``poke_feat[3:15]``.  Spec A.3 originally wrote this
    histogram with no divisor while A.1 divided the other one, which left raw
    counts up to 5.0 sitting in 36 of the 94 card-feature dims next to
    everything else in [0, 1].  ``ATKCOST_N`` rather than ``ENERGY_N`` because
    a cost is bounded by its own total, which ``row[13]`` already normalizes
    the same way -- the two halves of the cost then share one scale.
    """
    row = np.zeros(F_ATK, dtype=np.float32)
    row[0] = attack.damage / ATKDMG_N
    hist = np.zeros(N_ENERGY, dtype=np.float32)
    for e in attack.energies:
        assert int(e) < N_ENERGY, f"attack energy index {e} >= N_ENERGY={N_ENERGY}"
        hist[int(e)] += 1.0
    row[1:13] = hist / ATKCOST_N
    row[13] = len(attack.energies) / ATKCOST_N
    # Effect keywords from the attack's oracle text (14:43).  Living here rather
    # than only in card_static_row is deliberate: opt_attack_feat is built from
    # this row, so ATTACK options gain their effect text for free -- the decision
    # where text matters most.
    row[14:14 + K_EFFECT] = attack_keyword_row(attack)
    return row


def build_static_tables(vocab: dict, card_data: list, attack_data: list):
    """Build the [V,94] card table and [A,14] attack table for the given vocab.

    Returns (card_table, attack_id_to_index, attack_table):
      - card_table[V,94] float32: row 0 = PAD (zeros), row 1 = UNKNOWN (mean of
        in-vocab card rows), rows 2..V-1 = card_static_row(card) per vocab id.
      - attack_id_to_index: {attackId: index}, index >= 1 (0 is PAD).
      - attack_table[A,14] float32: row 0 = PAD (zeros); A = 1 + number of
        distinct attackIds referenced by vocab cards.
    """
    cards_by_id = {c.cardId: c for c in card_data}
    attacks_by_id = {a.attackId: a for a in attack_data}

    index_to_id = vocab["index_to_id"]
    v_size = vocab["size"]

    card_table = np.zeros((v_size, F_CARD), dtype=np.float32)
    for idx in range(2, v_size):
        cid = index_to_id[idx]
        card = cards_by_id.get(cid)
        if card is not None:
            card_table[idx] = card_static_row(card, attacks_by_id)
    if v_size > 2:
        card_table[1] = card_table[2:].mean(axis=0)

    referenced_attack_ids: list[int] = []
    seen = set()
    for idx in range(2, v_size):
        cid = index_to_id[idx]
        card = cards_by_id.get(cid)
        if card is None:
            continue
        for aid in card.attacks:
            if aid not in seen:
                seen.add(aid)
                referenced_attack_ids.append(aid)

    attack_id_to_index = {aid: i + 1 for i, aid in enumerate(referenced_attack_ids)}
    attack_table = np.zeros((len(referenced_attack_ids) + 1, F_ATK), dtype=np.float32)
    for aid, idx in attack_id_to_index.items():
        attack = attacks_by_id.get(aid)
        if attack is not None:
            attack_table[idx] = attack_static_row(attack)

    return card_table, attack_id_to_index, attack_table


def build_engine_card_features(card_data: list,
                               attack_data: list) -> dict[int, np.ndarray]:
    """Build ``{card_id: static_row_94}`` for ALL engine cards.

    Unlike :func:`build_static_tables` which only covers vocab cards, this
    dict maps every card the engine knows about to its 94-dim static
    features.  Used at inference time to represent every card purely by its
    features (no learned id embeddings).
    """
    attacks_by_id = {a.attackId: a for a in attack_data}
    return {c.cardId: card_static_row(c, attacks_by_id) for c in card_data}


def build_engine_attack_features(attack_data: list) -> dict[int, np.ndarray]:
    """Build ``{attack_id: static_row_43}`` for ALL engine attacks."""
    return {a.attackId: attack_static_row(a) for a in attack_data}


def build_evolution_map(card_data: list) -> dict[int, list[int]]:
    """``{card_id: [pre_evolution_card_ids]}`` resolved from ``evolvesFrom``.

    ``CardData.evolvesFrom`` is a *name*, and names are not in the feature row,
    so the hand-playability flag needs this side table.  Several distinct card
    ids share a name (reprints), hence the list value — any of them in play
    makes the evolution legal.

    An unresolvable name is logged rather than dropped: silently returning an
    empty list is indistinguishable downstream from "this card evolves from
    nothing", which would teach the model the evolution is never available.
    """
    by_name: dict[str, list[int]] = {}
    for c in card_data:
        by_name.setdefault(c.name, []).append(int(c.cardId))

    out: dict[int, list[int]] = {}
    for c in card_data:
        pre = getattr(c, "evolvesFrom", None)
        if not pre:
            out[int(c.cardId)] = []
            continue
        ids = by_name.get(pre)
        if ids is None:
            logger.warning(
                "evolution_map: %s (id %d) evolves from %r, which matches no "
                "card name in the engine tables", c.name, int(c.cardId), pre)
            out[int(c.cardId)] = []
        else:
            out[int(c.cardId)] = sorted(ids)
    return out
