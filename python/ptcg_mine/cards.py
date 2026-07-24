"""Engine-derived static feature tables for cards and attacks.

Layout is exact Appendix A.3 (feature slices) / A.2 (normalizers) of
TRANSFORMER_IL_SPEC.md:
  card_static_row[52]  = [hp/HP_N, retreat/RETREAT_N, cardType-onehot(7),
                          stage-onehot(3), energyType-onehot(12),
                          weakness-onehot(12), resistance-onehot(12),
                          [ex, megaEx, tera, aceSpec](4)]
  attack_static_row[14] = [damage/ATKDMG_N, energy-cost histogram(12),
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

F_CARD = 52
F_ATK = 14

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


def card_static_row(card) -> np.ndarray:
    """float32[52] static feature row for a CardData, per Appendix A.3."""
    row = np.zeros(F_CARD, dtype=np.float32)
    row[0] = card.hp / HP_N
    row[1] = card.retreatCost / RETREAT_N
    row[2:9] = _onehot(card.cardType, N_CARDTYPE)
    row[9:12] = [float(card.basic), float(card.stage1), float(card.stage2)]
    row[12:24] = _onehot(card.energyType, N_ENERGY)
    row[24:36] = _onehot(card.weakness, N_ENERGY)
    row[36:48] = _onehot(card.resistance, N_ENERGY)
    row[48:52] = [float(card.ex), float(card.megaEx), float(card.tera), float(card.aceSpec)]
    return row


def attack_static_row(attack) -> np.ndarray:
    """float32[14] static feature row for an Attack, per Appendix A.3."""
    row = np.zeros(F_ATK, dtype=np.float32)
    row[0] = attack.damage / ATKDMG_N
    hist = np.zeros(N_ENERGY, dtype=np.float32)
    for e in attack.energies:
        assert int(e) < N_ENERGY, f"attack energy index {e} >= N_ENERGY={N_ENERGY}"
        hist[int(e)] += 1.0
    row[1:13] = hist
    row[13] = len(attack.energies) / ATKCOST_N
    return row


def build_static_tables(vocab: dict, card_data: list, attack_data: list):
    """Build the [V,52] card table and [A,14] attack table for the given vocab.

    Returns (card_table, attack_id_to_index, attack_table):
      - card_table[V,52] float32: row 0 = PAD (zeros), row 1 = UNKNOWN (mean of
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
            card_table[idx] = card_static_row(card)
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
