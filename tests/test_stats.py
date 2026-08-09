"""Tests for ptcg_mine.stats.select_experts, incl. best-effort relaxation of
g_min when too little data exists to fill k_experts at the configured floor."""

import logging

import pytest

from ptcg_mine.stats import (
    select_experts,
    skill_weight,
    team_skill_weights,
    wilson_lower_bound,
)


def _lb(**teams):
    """Build a leaderboard dict from name=(games, win_rate) kwargs."""
    return {
        name: {"games": games, "wins": round(games * wr), "win_rate": wr}
        for name, (games, wr) in teams.items()
    }


def test_select_experts_returns_top_k_by_win_rate_when_enough_qualify():
    # A, B, C all clear g_min=50; D has a great win-rate but too few games.
    lb = _lb(A=(60, 0.55), B=(55, 0.70), C=(50, 0.60), D=(10, 0.99))
    assert select_experts(lb, k_experts=2, g_min=50) == ["B", "C"]


def test_select_experts_relaxes_threshold_when_too_few_qualify():
    # Nobody reaches g_min=50; relax to admit the 3 most-played teams
    # (games >= 5: A, B, C) then rank that pool by win-rate. D (2 games) excluded.
    lb = _lb(A=(10, 0.4), B=(8, 0.9), C=(5, 0.7), D=(2, 1.0))
    assert select_experts(lb, k_experts=3, g_min=50) == ["B", "C", "A"]


def test_select_experts_relaxes_to_all_teams_when_fewer_than_k():
    """Relaxation admits every team; the bound then orders them.

    ``A`` outranks the higher-rate ``B`` because one game supports no claim at
    all -- under the old raw-win_rate ranking this asserted ``["B", "A"]``.
    The counts are chosen so the two bounds are far apart (~0.30 vs ~0.21);
    at single-digit *n* the bound compresses everything toward zero and a
    closer fixture would be a coin flip rather than a statement about ranking.
    """
    lb = _lb(A=(20, 0.5), B=(1, 1.0))
    assert select_experts(lb, k_experts=10, g_min=50) == ["A", "B"]


def test_select_experts_empty_leaderboard_returns_empty():
    assert select_experts({}, k_experts=10, g_min=50) == []


def test_select_experts_warns_when_relaxing_below_g_min(caplog):
    lb = _lb(A=(3, 0.33))
    with caplog.at_level(logging.WARNING):
        select_experts(lb, k_experts=10, g_min=50)
    assert any("g_min" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Wilson lower bound: ranking experts on evidence, not on point estimates
# ---------------------------------------------------------------------------


def test_wilson_lower_bound_is_below_point_estimate_and_in_range():
    for wins, n in ((30, 50), (2486, 4148), (1, 1), (0, 10)):
        lb = wilson_lower_bound(wins, n)
        assert 0.0 <= lb <= 1.0
        assert lb <= wins / n


def test_wilson_lower_bound_of_zero_games_is_zero():
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_lower_bound_tightens_toward_p_as_n_grows():
    """Same observed rate, more games -> bound moves up toward the estimate.

    This monotonicity is the whole reason the bound is used for ranking: it is
    what makes a 52-game 0.635 lose to a 4148-game 0.599.
    """
    bounds = [wilson_lower_bound(round(0.6 * n), n) for n in (50, 200, 1000, 5000)]
    assert bounds == sorted(bounds)
    assert bounds[-1] > bounds[0]
    assert bounds[-1] == pytest.approx(0.6, abs=0.02)


def test_select_experts_prefers_well_measured_team_over_small_sample_fluke():
    """Regression for the real corpus: 92% of episodes were discarded and four
    of ten expert slots went to teams with 51-54 games whose win rate was
    statistically indistinguishable from average, displacing teams with
    thousands of games.  Raw win_rate ranks Fluke first; the Wilson bound does
    not, because at n=52 a 0.635 rate carries a ~+-0.13 margin.
    """
    lb = _lb(Measured=(4148, 0.599), Fluke=(52, 0.635))
    assert lb["Fluke"]["win_rate"] > lb["Measured"]["win_rate"]  # fixture is real
    assert select_experts(lb, k_experts=1, g_min=50) == ["Measured"]


def test_select_experts_ranks_by_bound_not_by_win_rate():
    """Full ordering differs from the raw-win_rate ordering.

    Fails (goes green on the old code) if ranking reverts to ``win_rate``.
    """
    lb = _lb(Big=(4000, 0.58), Mid=(300, 0.60), Tiny=(51, 0.66))
    by_win_rate = sorted(lb, key=lambda t: -lb[t]["win_rate"])
    assert by_win_rate == ["Tiny", "Mid", "Big"]
    assert select_experts(lb, k_experts=3, g_min=50) == ["Big", "Mid", "Tiny"]


def test_select_experts_still_excludes_below_g_min_however_strong_the_bound():
    """The bound ranks the eligible pool; it does not widen it.

    ``k_experts=1`` is load-bearing: asking for more than the pool holds would
    trigger the best-effort relaxation and admit ``Strong`` on its own merits,
    which is a different rule being tested elsewhere.
    """
    lb = _lb(A=(60, 0.55), Strong=(49, 0.95))
    assert select_experts(lb, k_experts=1, g_min=50) == ["A"]


# ============================================================
# skill_weight — the soft replacement for the top-K expert filter
# ============================================================


def test_skill_weight_is_one_for_a_perfectly_average_infinite_team():
    """The exponent is signed skill relative to 0.5, so an average team weighs 1.

    0.5 is the corpus's pooled win rate by construction (every game has a
    winner and a loser), which is what makes 1.0 the natural neutral point
    rather than an arbitrary scale choice.  Approached from below, because the
    Wilson bound only reaches 0.5 in the limit.
    """
    weights = [skill_weight(n // 2, n) for n in (1_000, 100_000, 10_000_000)]
    assert weights == sorted(weights), "should approach 1.0 from below, not oscillate"
    assert weights[-1] == pytest.approx(1.0, abs=1e-2)
    assert all(w < 1.0 for w in weights)


def test_skill_weight_is_monotone_in_wins_at_fixed_games():
    n = 500
    weights = [skill_weight(w, n) for w in range(0, n + 1, 50)]
    assert len(weights) > 1
    assert weights == sorted(weights)
    assert weights[0] < weights[-1]


def test_skill_weight_charges_for_uncertainty_not_just_win_rate():
    """A 3-for-3 team must not outweigh a well-measured strong team.

    This is the property that makes ``g_min`` unnecessary once skill is a
    weight: the raw win rate ranks the fluke first (1.000 vs 0.550), and only
    the bound inside the exponent puts it in its place.  Without it, the ~500
    one-and-two-game teams in the real corpus would each carry more weight than
    Majkel1337's 4148 games.
    """
    fluke = skill_weight(3, 3)
    measured = skill_weight(1100, 2000)  # 0.550 over 2000 games
    assert fluke < measured
    # A perfect record on 3 games lands *below* an average team (1.0), not above
    # it, which is what a raw-win_rate weight would do.
    assert fluke < 0.5
    # And the ~500 single-game teams that dominate the leaderboard by count
    # contribute essentially nothing.
    assert skill_weight(1, 1) < 0.01


def test_skill_weight_spread_actually_separates_the_real_win_rate_band():
    """Sharpness has to beat the narrow band real win rates live in.

    Measured on the 52.2k-episode corpus, the best team's Wilson bound is 0.584
    and the median qualifying team's is ~0.47.  Weighting linearly in the bound
    separates those by 1.24x, which is why the weight is exponential; at
    sharpness 20 the same pair separates by ~10x.
    """
    best, median = 0.584, 0.47
    n = 4000  # large enough that the bound sits essentially at the point estimate

    linear_ratio = best / median
    assert linear_ratio < 1.3, "linear-in-the-bound cannot separate these"

    exponential_ratio = skill_weight(round(best * n), n) / skill_weight(
        round(median * n), n
    )
    assert exponential_ratio > 8.0


def test_team_skill_weights_covers_every_team_including_the_weak_ones():
    """No team is dropped — that is the whole point versus a top-K filter.

    ``build_shards`` indexes this map directly (a miss raises), so a team the
    leaderboard knows about and this map does not would abort a 7-hour build.
    """
    lb = _lb(strong=(2000, 0.58), average=(2000, 0.50), weak=(2000, 0.42), tiny=(1, 1.0))
    weights = team_skill_weights(lb)

    assert set(weights) == set(lb)
    assert all(w > 0.0 for w in weights.values()), "no team may be zeroed silently"
    assert weights["strong"] > weights["average"] > weights["weak"]
    assert weights["tiny"] < weights["weak"], "1 game must not outrank 2000 at 0.42"


def test_team_skill_weights_sharpness_is_tunable_and_monotone():
    lb = _lb(strong=(2000, 0.58), weak=(2000, 0.42))
    ratios = []
    for sharpness in (0.0, 8.0, 20.0, 40.0):
        w = team_skill_weights(lb, sharpness=sharpness)
        ratios.append(w["strong"] / w["weak"])
    assert ratios[0] == pytest.approx(1.0), "sharpness 0 must be uniform weighting"
    assert ratios == sorted(ratios)
