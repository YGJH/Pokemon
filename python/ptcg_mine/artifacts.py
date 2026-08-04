"""Frozen artifact writers: vocab.json, archetypes.json, mining_report.md.

See docs/plans/corpus-mining-plan.md "Global Constraints" and
TRANSFORMER_IL_SPEC.md Appendix A for the exact constants recorded here.
"""

import json
from pathlib import Path

from ptcg_mine.cards import ATKCOST_N, ATKDMG_N, HP_N, RETREAT_N

from ptcg_il.featurizer import (F_ATK, F_CARD, F_GLOBAL, F_HAND, F_OPT,
                                 F_POKE, F_SUM)


def write_vocab_json(path, vocab: dict, attack_id_to_index: dict, config) -> None:
    """Write the frozen vocab artifact: remaps, norm constants, caps, F_* dims, w_lost."""
    data = {
        "id_to_index": vocab["id_to_index"],
        "index_to_id": vocab["index_to_id"],
        "freq": vocab["freq"],
        "size": vocab["size"],
        "freq_coverage": vocab["freq_coverage"],
        "attack_id_to_index": attack_id_to_index,
        "norm": {
            "HP_N": HP_N,
            "RETREAT_N": RETREAT_N,
            "ATKDMG_N": ATKDMG_N,
            "ATKCOST_N": ATKCOST_N,
        },
        "caps": {
            "h_max": config.h_max,
            "o_max": config.o_max,
            "d_max": config.d_max,
        },
        "F_CARD": F_CARD,
        "F_ATK": F_ATK,
        "F_POKE": F_POKE,
        "F_HAND": F_HAND,
        "F_SUM": F_SUM,
        "F_GLOBAL": F_GLOBAL,
        "F_OPT": F_OPT,
        "w_lost": config.w_lost,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_archetypes_json(
    path,
    self_ids: list[int],
    opp_ids: list[int],
    archetypes: list,
    fixed_deck: list[int],
    lineage: dict | None = None,
) -> None:
    """Write archetype signatures (representative decklists), D_self/D_opp id
    lists, and FIXED_DECK (60 ids).

    *lineage* records which id generation this file belongs to: whether it was
    seeded from an earlier ``archetypes.json`` and which one.  It is written so
    the question "can this checkpoint still be trusted against these artifacts"
    is answerable from the artifact itself — the SHA pins in each checkpoint's
    deck record tell you the file *changed*, but not whether the ids inside it
    still mean the same decks, which is the thing that actually matters.
    """
    data = {
        "self_ids": self_ids,
        "opp_ids": opp_ids,
        "fixed_deck": fixed_deck,
        "lineage": lineage or {"seeded": False, "baseline_sha1": None,
                               "generation": 0},
        "archetypes": [
            {
                "id": arch.id,
                "representative": list(arch.representative),
                "frequency": arch.frequency,
                "n_members": len(arch.members),
            }
            for arch in archetypes
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_mining_report(
    path,
    leaderboard: dict,
    experts: list[str],
    archetypes: list,
    self_ids: list[int],
    opp_ids: list[int],
    vocab: dict,
    counts: dict,
) -> None:
    """Write a Markdown summary report: team leaderboard, archetype table,
    vocab size + coverage, kept-game counts."""
    lines: list[str] = ["# Corpus Mining Report", ""]

    lines.append("## Kept-game counts")
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Team leaderboard (top teams by win-rate)")
    lines.append("| team | games | wins | win_rate | expert |")
    lines.append("|---|---|---|---|---|")
    expert_set = set(experts)
    ranked = sorted(leaderboard.items(), key=lambda kv: (-kv[1]["win_rate"], kv[0]))
    for team, d in ranked:
        marker = "yes" if team in expert_set else ""
        lines.append(f"| {team} | {d['games']} | {d['wins']} | {d['win_rate']:.3f} | {marker} |")
    lines.append("")

    lines.append("## Archetypes")
    lines.append("| id | frequency | n_members | in D_self | in D_opp |")
    lines.append("|---|---|---|---|---|")
    self_set = set(self_ids)
    opp_set = set(opp_ids)
    for arch in sorted(archetypes, key=lambda a: -a.frequency):
        lines.append(
            f"| {arch.id} | {arch.frequency} | {len(arch.members)} "
            f"| {'yes' if arch.id in self_set else ''} | {'yes' if arch.id in opp_set else ''} |"
        )
    lines.append("")

    lines.append("## Vocab")
    lines.append(f"- vocab size (V): {vocab['size']}")
    coverage = vocab.get("freq_coverage") or []
    if coverage:
        top_n = min(50, len(coverage))
        top_coverage = coverage[top_n - 1][2]
        lines.append(f"- top-{top_n} ids cover {top_coverage:.1%} of corpus card occurrences")
    lines.append("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
