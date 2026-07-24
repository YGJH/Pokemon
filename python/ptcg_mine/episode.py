"""Parsing and validation for Kaggle Pokemon TCG episode replay JSON.

Episode format (see docs/plans/corpus-mining-plan.md "Global Constraints"):
  - ep["info"]["TeamNames"] = [team0, team1]
  - ep["rewards"] = [r0, r1] with +1 win / -1 loss / 0 draw
  - ep["steps"] is list[[recordP0, recordP1]]
  - deck_of(ep, p) = ep["steps"][1][p]["action"], a 60-int decklist
"""

import json
from pathlib import Path


def project_for_selection(ep: dict) -> dict:
    """Reduce an episode to the fields the selection stages actually read.

    Expert selection, archetype clustering, vocab building and the kept-game
    filter only ever touch TeamNames, rewards, statuses and the two opening
    deck actions at ``steps[1][p]``. Everything else in ``steps`` is the
    turn-by-turn game log: ~4 MB on disk per episode, ~14.5 MB once parsed
    into Python objects. Retaining whole episodes therefore costs ~145 GB at
    the 10k-episode target, which the miner cannot survive.

    The projection is ~1 KB, so a caller holding one per episode scales with
    episode *count*, not corpus bytes. ``steps`` keeps its two-element shape so
    ``validate_episode``'s ``len(steps) >= 2`` check and ``deck_of``'s
    ``steps[1][p]["action"]`` indexing keep working against the projection.

    Stages that need the real game log (Phase 3 featurization) must re-read the
    episode from disk rather than expect it here.
    """
    return {
        "info": ep.get("info", {}),
        "statuses": ep.get("statuses"),
        "rewards": ep.get("rewards"),
        "steps": [[], [{"action": ep["steps"][1][0]["action"]},
                       {"action": ep["steps"][1][1]["action"]}]],
    }


def load_episode(path: str | Path) -> dict:
    """Load an episode JSON file into a dict."""
    with open(path, "r") as f:
        return json.load(f)


def teams(ep: dict) -> tuple[str, str]:
    """Return the (team0, team1) names from ep["info"]["TeamNames"]."""
    t0, t1 = ep["info"]["TeamNames"]
    return (t0, t1)


def rewards(ep: dict) -> tuple[int, int]:
    """Return the (r0, r1) final rewards from ep["rewards"]."""
    r0, r1 = ep["rewards"]
    return (r0, r1)


def deck_of(ep: dict, p: int) -> list[int]:
    """Return player p's 60-card decklist: ep["steps"][1][p]["action"].

    Raises ValueError if the extracted deck is not exactly 60 cards.
    """
    deck = ep["steps"][1][p]["action"]
    if len(deck) != 60:
        raise ValueError(
            f"deck_of(ep, {p}) expected 60 cards, got {len(deck)}"
        )
    return deck


def validate_episode(ep: dict) -> bool:
    """An episode is usable iff:
      - statuses == ["DONE", "DONE"]
      - len(steps) >= 2
      - len(rewards) == 2
      - both decks (deck_of(ep, 0), deck_of(ep, 1)) have length 60
    """
    if ep.get("statuses") != ["DONE", "DONE"]:
        return False
    if len(ep.get("steps", [])) < 2:
        return False
    if len(ep.get("rewards", [])) != 2:
        return False
    for p in (0, 1):
        try:
            deck_of(ep, p)
        except (ValueError, KeyError, IndexError, TypeError):
            return False
    return True
