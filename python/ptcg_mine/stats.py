"""Phase 2 team stats: leaderboard and expert selection.

See docs/plans/corpus-mining-plan.md "Global Constraints" (Experts bullet):
  EXPERTS = top-K teams by win_rate = wins/games with games >= G_MIN.
  A draw (r == 0) counts toward games but not wins.
"""

from collections import defaultdict

from ptcg_mine.episode import rewards, teams


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
    """
    eligible = [team for team, d in leaderboard.items() if d["games"] >= g_min]
    eligible.sort(key=lambda team: (-leaderboard[team]["win_rate"], team))
    return eligible[:k_experts]
