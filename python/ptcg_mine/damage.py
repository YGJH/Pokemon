"""Deterministic damage arithmetic shared by the CLS threat block and the
option damage preview.

``WEAKNESS_MULT`` and ``RESISTANCE_DELTA`` are **measured**, not assumed.
Weakness and resistance resolution lives in libcg.so and no Python binding states
the rule, so the values below come from driving vanilla attacks (empty
``Attack.text``, so nothing modifies the damage) into defenders whose weakness
or resistance matches the attacker's energyType, then reading the applied damage
off ``Log.value`` (LogType.HP_CHANGE, type 16).

Record of the measurement — update this block if it is ever redone:
    Weakness:  scanned 500 corpus episodes, 51 clean hits (empty attack text,
               weakness matches attacker energyType).  Every hit:
                   |Log.value| / Attack.damage == 2.0 exactly.
               => WEAKNESS_MULT = 2.0
    Resistance: scanned same corpus, inconclusive (all hits shared the same
                base damage, delta varied by other effects).  The engine
                implements standard Sword & Shield / Scarlet & Violet rules,
                which set Resistance to -30.  This is the value used; it
                should be re-measured against a controlled pair.
                => RESISTANCE_DELTA = -30.0

**Base damage is not real damage.**  315 of 1556 attacks carry
``does N more damage`` or ``damage for each ...`` text, so any KO prediction
built on ``Attack.damage`` alone is systematically wrong on those.  Callers must
treat the output as a graded signal, not a verdict — which is why the features
that consume it are continuous ratios with the booleans derived from them, and
why ``keywords.damage_scaling`` sits in the same feature row so the model can
learn to discount the ratio when it fires.
"""

import numpy as np

#: MEASURED — see module docstring.  Do not change without redoing the measurement.
WEAKNESS_MULT: float = 2.0
#: From standard TCG rules (Sword & Shield / Scarlet & Violet).  Should be
#: re-measured; corpus scan was inconclusive.
RESISTANCE_DELTA: float = -30.0

COLORLESS = 0


def effective_damage(base, attacker_energy_type, defender_weakness,
                     defender_resistance) -> float:
    """Base damage adjusted for weakness and resistance.  Never negative."""
    dmg = float(base)
    if dmg <= 0.0:
        return 0.0
    if defender_weakness is not None and int(defender_weakness) == int(attacker_energy_type):
        dmg *= WEAKNESS_MULT
    if defender_resistance is not None and int(defender_resistance) == int(attacker_energy_type):
        dmg += RESISTANCE_DELTA
    return max(dmg, 0.0)


def attack_is_affordable(cost_hist: np.ndarray, attached_hist: np.ndarray) -> bool:
    """Can a Pokemon holding *attached_hist* energy pay *cost_hist*?

    Both are length-12 histograms over ``EnergyType`` with COLORLESS at index 0.
    A colorless requirement accepts any energy, so it is checked only against the
    total; every colored requirement must be met in its own colour.

    Exact except for the 12 special Energy cards that provide multiple or
    arbitrary types (Prism, Legacy, Neo Upper, ...), which the engine resolves
    and this does not.
    """
    cost = np.asarray(cost_hist, dtype=np.float64)
    have = np.asarray(attached_hist, dtype=np.float64)
    if cost[1:].sum() > 0 and np.any(have[1:] < cost[1:]):
        return False
    return have.sum() >= cost.sum()
