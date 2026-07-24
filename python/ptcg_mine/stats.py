"""Phase 2 team stats: leaderboard and expert selection.

See docs/plans/corpus-mining-plan.md "Global Constraints" (Experts bullet):
  EXPERTS = top-K teams by win_rate = wins/games with games >= G_MIN.
  A draw (r == 0) counts toward games but not wins.
"""

import logging
from collections import defaultdict

from ptcg_mine.episode import rewards, teams

log = logging.getLogger(__name__)


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


def select_experts(leaderboard: dict[str, dict], k_experts: int, g_min: int) -> list[str]:
    """Top-K team names by win_rate among teams with games >= g_min.

    Deterministic tie-break by team name (ascending).

    Best-effort on thin data: if fewer than `k_experts` teams reach `g_min`,
    the effective threshold is lowered just enough to admit up to `k_experts`
    of the most-played teams (floored at >= 1 game), then that pool is ranked
    by win_rate as usual. This never drops a team that already qualified at
    `g_min`; it only widens the pool. A warning is logged whenever the
    effective threshold falls below the configured `g_min`, so callers can see
    the experts were chosen from an under-powered corpus. Returns [] only for
    an empty leaderboard (i.e. no episodes at all).
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

    eligible.sort(key=lambda team: (-leaderboard[team]["win_rate"], team))
    return eligible[:k_experts]
