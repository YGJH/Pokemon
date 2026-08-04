import numpy as np
import pytest

from ptcg_mine import damage as dmg


def test_no_weakness_no_change():
    assert dmg.effective_damage(100, 2, None, None) == 100.0


def test_weakness_applies_only_on_type_match():
    assert dmg.effective_damage(100, 2, 2, None) == 100.0 * dmg.WEAKNESS_MULT
    assert dmg.effective_damage(100, 2, 3, None) == 100.0


def test_resistance_applies_only_on_type_match():
    assert dmg.effective_damage(100, 2, None, 2) == 100.0 + dmg.RESISTANCE_DELTA
    assert dmg.effective_damage(100, 2, None, 3) == 100.0


def test_damage_never_goes_negative():
    assert dmg.effective_damage(10, 2, None, 2) == 0.0


def test_zero_base_stays_zero_even_against_weakness():
    """Status-only attacks have damage 0; weakness must not manufacture damage."""
    assert dmg.effective_damage(0, 2, 2, None) == 0.0


def test_weakness_mult_is_measured():
    """Pin: measured from 51 clean engine hits, all exactly x2."""
    assert dmg.WEAKNESS_MULT == 2.0


def _hist(**kw):
    h = np.zeros(12, dtype=np.float32)
    for k, v in kw.items():
        h[int(k[1:])] = v
    return h


def test_colorless_cost_accepts_any_energy():
    assert dmg.attack_is_affordable(_hist(t0=2), _hist(t5=2))


def test_colored_cost_needs_its_own_colour():
    """Two Psychic does not pay a two-Fire cost even though the total matches."""
    assert not dmg.attack_is_affordable(_hist(t2=2), _hist(t5=2))
    assert dmg.attack_is_affordable(_hist(t2=2), _hist(t2=2))


def test_mixed_cost_checks_colour_then_total():
    cost = _hist(t2=1, t0=2)          # 1 Fire + 2 Colorless
    assert not dmg.attack_is_affordable(cost, _hist(t2=1, t5=1))   # total 2 < 3
    assert dmg.attack_is_affordable(cost, _hist(t2=1, t5=2))       # total 3, Fire met


def test_empty_cost_is_always_affordable():
    assert dmg.attack_is_affordable(np.zeros(12), np.zeros(12))
