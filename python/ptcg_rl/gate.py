"""The promotion gate: paired games, Wilson bound, IL-regression (RL_SPEC §10.2).

A candidate replaces the champion only if **all** the configured conditions hold.
Each exists because the others do not catch its failure:

1. **score ≥ threshold** — the headline bar.
2. **Wilson 95% lower bound** — the anti-noise floor.  At n = 400 paired games a
   point estimate of 0.70 is compatible with a true rate of 0.653, so a bare
   point estimate promotes on variance roughly as often as on strength.
3. **IL-regression** — the real forgetting guard.  A KL penalty bounds *average*
   divergence; it says nothing about the rare situations the corpus covers, and
   those are exactly where a drifted policy fails.  The held-out data already
   exists, so this check is nearly free.
4. **KL to π_IL ≤ 4κ at evaluation temperature** — catches a policy that met the
   budget in expectation during training but drifted when sampled greedily.

Games are **paired**: same seed, swapping who goes first.  First-player advantage
is large here, and unpaired sampling spends a large share of its games measuring
that instead of the policy difference.

Evaluation is with the **greedy policy network alone** — no MCTS, no sampling.
That is the deployment procedure by decision (§8.3), so the gate measures exactly
the artifact that ships, and its cost does not depend on any search budget.
"""

from __future__ import annotations

import logging
from rich.logging import RichHandler
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(show_time=False)])
logger = logging.getLogger(__name__)


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """``(low, centre, high)`` Wilson score interval for a binomial proportion.

    Wilson rather than normal-approximation: the normal interval is badly wrong
    near 0 and 1 and can produce bounds outside [0, 1], which is precisely the
    regime a strong candidate lands in.

    *wins* is a float so half-credit draws work, even though draws appear not to
    occur in this game (§4.1).
    """
    if n <= 0:
        return 0.0, 0.0, 1.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), p, min(1.0, centre + margin)


@dataclass
class GateResult:
    """The verdict, with every condition's outcome recorded separately.

    Keeping the individual booleans matters for diagnosis: a candidate that
    passes 1–2 but fails 3 is the signal to *tighten* κ, not to lower the bar.
    """

    passed: bool = False
    n_paired: int = 0
    wins: float = 0.0
    score: float = 0.0
    wilson_lb: float = 0.0
    wilson_ub: float = 1.0
    il_nontrivial_top1: float | None = None
    il_baseline: float | None = None
    il_regression_pts: float | None = None
    kl_to_il: float | None = None
    conditions: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"score={self.score:.4f}", f"wilson_lb={self.wilson_lb:.4f}"]
        if self.il_regression_pts is not None:
            bits.append(f"il_regression={self.il_regression_pts:+.2f}pts")
        if self.kl_to_il is not None:
            bits.append(f"kl={self.kl_to_il:.4f}")
        failed = [k for k, v in self.conditions.items() if not v]
        verdict = "PASS" if self.passed else f"FAIL ({', '.join(failed)})"
        return f"{verdict}: " + ", ".join(bits)


def score_from_results(results: list[dict[str, Any]], our_player_key: str = "our_player"
                       ) -> tuple[float, int]:
    """``(wins-with-half-credit-draws, n_games)`` from per-game result dicts.

    Each entry needs ``winner`` (0, 1, or −1 for no winner) and which side we
    played.  A game with no winner scores half, matching
    ``score = (wins + 0.5·draws)/n``.
    """
    wins = 0.0
    n = 0
    for r in results:
        winner = r.get("winner", -1)
        ours = r.get(our_player_key, 0)
        n += 1
        if winner == ours:
            wins += 1.0
        elif winner not in (0, 1):
            wins += 0.5
    return wins, n


def paired_seeds(n_paired: int, base_seed: int = 0) -> list[tuple[int, int]]:
    """``[(seed, first_player), ...]`` — each seed played once from each side.

    The pairing is what makes the comparison efficient: the same seed produces
    the same shuffles and the same determinization stream, so the only thing
    that differs between the two games of a pair is who moved first.
    """
    out: list[tuple[int, int]] = []
    for i in range(n_paired):
        out.append((base_seed + i, 0))
        out.append((base_seed + i, 1))
    return out


def evaluate_gate(
    results: list[dict[str, Any]],
    *,
    min_score: float,
    min_wilson_lb: float,
    il_nontrivial_top1: float | None = None,
    il_baseline: float | None = None,
    max_il_regression_pts: float = 5.0,
    kl_to_il: float | None = None,
    max_kl: float | None = None,
) -> GateResult:
    """Apply every configured condition and return the verdict.

    Conditions with no data supplied are **skipped and recorded as skipped**,
    never silently passed — a gate that reports ``passed=True`` because a check
    could not run is worse than no gate, since it looks like evidence.
    """
    wins, n = score_from_results(results)
    lb, score, ub = wilson_interval(wins, n)

    res = GateResult(
        passed=False, n_paired=n // 2, wins=wins, score=score,
        wilson_lb=lb, wilson_ub=ub,
        il_nontrivial_top1=il_nontrivial_top1, il_baseline=il_baseline,
        kl_to_il=kl_to_il,
    )

    res.conditions["score"] = score >= min_score
    res.conditions["wilson_lb"] = lb >= min_wilson_lb

    if il_nontrivial_top1 is not None and il_baseline is not None:
        drop_pts = (il_baseline - il_nontrivial_top1) * 100.0
        res.il_regression_pts = -drop_pts
        res.conditions["il_regression"] = drop_pts <= max_il_regression_pts
    else:
        res.notes.append(
            "IL-regression check SKIPPED (no baseline supplied) — this is the "
            "forgetting guard; a pass without it is not evidence of one"
        )

    if kl_to_il is not None and max_kl is not None:
        res.conditions["kl_budget"] = kl_to_il <= max_kl
    elif max_kl is not None:
        res.notes.append("KL-budget check SKIPPED (no measured KL supplied)")

    res.passed = bool(res.conditions) and all(res.conditions.values())
    return res


def load_il_baseline(
    data_dir: str | Path,
    archetype: int | None,
    il_ckpt: str | Path,
) -> float:
    """The recorded non-trivial top-1 for *archetype*, verified against *il_ckpt*.

    Delegates the SHA-1 pin to :mod:`ptcg_il.baselines`, so a baseline recorded
    from a different checkpoint raises rather than being compared against.  This
    is the mitigation for RL_SPEC §13's "stale IL baselines after the feature
    fix", which is listed high-severity precisely because stale thresholds do
    not raise on their own — they quietly pass or fail the wrong candidate.
    """
    from ptcg_il.baselines import get_baseline

    record = get_baseline(data_dir, archetype, il_ckpt)
    if "nontrivial_top1" not in record:
        raise KeyError(
            f"IL baseline for archetype {archetype} has no 'nontrivial_top1'; "
            f"re-record it with `ptcg_il.cli train --eval-only --eval-split test "
            f"--record-baseline`"
        )
    return float(record["nontrivial_top1"])
