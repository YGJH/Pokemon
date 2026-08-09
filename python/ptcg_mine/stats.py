"""Phase 2 team stats: leaderboard and expert selection.

See docs/plans/corpus-mining-plan.md "Global Constraints" (Experts bullet):
  EXPERTS = top-K teams by win_rate = wins/games with games >= G_MIN.
  A draw (r == 0) counts toward games but not wins.

Ranking uses the Wilson *lower bound* of that win rate rather than the rate
itself -- see :func:`select_experts` for the measurement that motivated it.
"""

import logging
import math
from collections import defaultdict

from ptcg_mine.episode import rewards, teams

log = logging.getLogger(__name__)

#: z for a 95% one-sided-ish bound.  Matches ``ptcg_rl.gate.wilson_interval``
#: and ``ptcg_il.live_eval.wilson_interval``, which this deliberately does not
#: import: ``ptcg_mine`` is upstream of both and carries no torch dependency,
#: and this is six lines of stdlib math.
WILSON_Z = 1.96

#: Steepness of the exponential skill weight.  See :func:`skill_weight` for the
#: measurement behind the value.
SKILL_SHARPNESS = 20.0


def wilson_lower_bound(wins: float, n: int, z: float = WILSON_Z) -> float:
    """Lower end of the Wilson score interval for ``wins / n``.

    Wilson rather than the normal approximation: the latter is badly wrong for
    small *n* and can return bounds outside [0, 1], which is exactly the regime
    that a team scraping past ``g_min`` sits in.

    Returns 0.0 for ``n <= 0`` so an unplayed team can never rank.
    """
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin)


def team_leaderboard(episodes: list[dict]) -> dict[str, dict]:
    """Aggregate per-team games/wins/win_rate over a list of parsed episodes.

    Draws (reward == 0) count toward `games` but not `wins`.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"games": 0, "wins": 0})
    for ep in episodes:
        t0, t1 = teams(ep)
        r0, r1 = rewards(ep)
        for team, r in ((t0, r0), (t1, r1)):
            counts[team]["games"] += 1
            if r == 1:
                counts[team]["wins"] += 1

    leaderboard: dict[str, dict] = {}
    for team, d in counts.items():
        games, wins = d["games"], d["wins"]
        leaderboard[team] = {
            "games": games,
            "wins": wins,
            "win_rate": wins / games if games else 0.0,
        }
    return leaderboard


def skill_weight(wins: float, n: int, sharpness: float = SKILL_SHARPNESS) -> float:
    """Per-team training weight ``exp(sharpness * (wilson_lb - 0.5))``.

    This is the **soft replacement for the top-K expert filter**, which kept
    only 8027 of 104418 (episode, player) pairs -- 7.7% of the corpus, against
    2.7% discarded by both archetype filters combined.  A hard filter also has
    to answer "how many experts" with a constant, and ``k_experts=10`` had no
    relationship to how many teams on a given corpus actually play well.

    Weighting instead of filtering means a strong team just under whatever line
    a filter would have drawn still contributes, in proportion to how strong it
    is.  It also makes ``g_min`` unnecessary: the Wilson lower bound already
    charges for uncertainty, so a perfect 3-for-3 record bounds at 0.439 and
    weighs 0.29 -- less than an average team, not more -- and the ~500 teams
    with a single game apiece weigh 0.003.  A raw win rate would hand all of
    them 1.0, or more than the best-measured team on the board.

    Why exponential and why 0.5.  0.5 is the corpus's pooled win rate by
    construction -- every game has a winner and a loser -- so the exponent is
    signed skill relative to the average opponent, and an average team weighs
    exactly 1.  Linear-in-the-bound does not discriminate at all: measured on
    the 52.2k-episode corpus the spread between the best team and the median is
    1.4x, and the weighted corpus represents 0.512 play, barely above the 0.500
    it started at.  Win rates here live in a narrow band, so the weight has to
    be exponential in it to separate anything.

    ``sharpness=20`` from measuring Kish effective sample size against the win
    rate the weighted corpus represents::

        uniform                 ESS = 104418 pairs (13.0x)   wr 0.500
        exp( 8*(wlb-0.5))       ESS =  80934 pairs (10.1x)   wr 0.530
        exp(20*(wlb-0.5))       ESS =  39311 pairs ( 4.9x)   wr 0.555
        exp(40*(wlb-0.5))       ESS =  13031 pairs ( 1.6x)   wr 0.581
        top-10 expert filter    ESS =   8027 pairs ( 1.0x)   wr 0.595

    20 buys 4.9x the effective data for 4 points of represented skill; 40 is
    within noise of the filter it replaces, and 8 mostly imitates the median
    player.  Note the ESS column is why the extra rows are not free at training
    time: 5.2M rows at a 37x weight spread carry the gradient signal of ~39k
    uniformly-weighted pairs, so batch size buys less than the row count says.
    """
    return math.exp(sharpness * (wilson_lower_bound(wins, n) - 0.5))


def team_skill_weights(
    leaderboard: dict[str, dict], sharpness: float = SKILL_SHARPNESS
) -> dict[str, float]:
    """``{team: skill_weight(...)}`` for every team on *leaderboard*."""
    return {
        team: skill_weight(d["wins"], d["games"], sharpness)
        for team, d in leaderboard.items()
    }


def select_experts(leaderboard: dict[str, dict], k_experts: int, g_min: int) -> list[str]:
    """Top-K team names by **Wilson lower bound** on win_rate, games >= g_min.

    Deterministic tie-break by team name (ascending).

    Why the bound and not the rate itself.  ``g_min`` admits a team to the pool
    but says nothing about how precisely its rate is measured, and ranking the
    pool by a raw point estimate hands the top slots to whoever got luckiest at
    the threshold.  Measured on the 53.7k-episode corpus: four of the ten
    selected experts qualified on 51-54 games at rates of 0.608-0.648, whose
    95% intervals all straddle 0.5, while ``Majkel1337`` (4183 games, 0.599,
    SE 0.008) ranked *fifth* and teams with 2000+ games at 0.55-0.57 were
    excluded outright.  The four flukes contributed ~211 games.

    Since the expert filter is by far the most aggressive stage of corpus
    construction -- it alone discards 92.4% of (episode, player) pairs, against
    2.7% for both archetype filters combined -- that misranking is expensive.
    Switching to the bound at unchanged ``k_experts=10`` raises kept pairs from
    5273 to 8429 (~402k -> ~642k training rows, 1.6x) *and* lifts the weakest
    selected expert from 51 games to 297.

    The bound only ever reorders the eligible pool; ``g_min`` still decides who
    is in it.

    Best-effort on thin data: if fewer than `k_experts` teams reach `g_min`,
    the effective threshold is lowered just enough to admit up to `k_experts`
    of the most-played teams (floored at >= 1 game), then that pool is ranked
    as usual. This never drops a team that already qualified at `g_min`; it
    only widens the pool. A warning is logged whenever the effective threshold
    falls below the configured `g_min`, so callers can see the experts were
    chosen from an under-powered corpus. Returns [] only for an empty
    leaderboard (i.e. no episodes at all).
    """
    if not leaderboard:
        return []

    eligible = [team for team, d in leaderboard.items() if d["games"] >= g_min]
    if len(eligible) < k_experts:
        by_games = sorted(leaderboard.items(), key=lambda kv: (-kv[1]["games"], kv[0]))
        cutoff = min(k_experts, len(by_games)) - 1
        effective_g_min = max(1, by_games[cutoff][1]["games"])
        if effective_g_min < g_min:
            log.warning(
                "select_experts: only %d of %d teams reached g_min=%d; relaxing "
                "effective threshold to %d game(s) (best-effort on thin data)",
                len(eligible),
                len(leaderboard),
                g_min,
                effective_g_min,
            )
        eligible = [team for team, d in leaderboard.items() if d["games"] >= effective_g_min]

    eligible.sort(
        key=lambda team: (
            -wilson_lower_bound(leaderboard[team]["wins"], leaderboard[team]["games"]),
            team,
        )
    )
    return eligible[:k_experts]
