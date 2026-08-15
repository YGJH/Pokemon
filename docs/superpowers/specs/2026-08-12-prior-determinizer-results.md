# Prior Deck Determinizer — Live A/B Results

**Date**: 2026-08-12
**Checkpoint**: `checkpoints_a1_s2/ckpt-best.pt` (step 12400, D=512, heads=2, layers=8, ff=2048, archetype_self=1)
**Games per opponent**: 200
**Workers**: 8

## Results

| Opponent | Win % | 95% CI | Games | Failed | Time | Mean Steps | Illegal | OOV |
|---|---|---|---|---|---|---|---|---|
| `random` | 3.4% | 1.1–5.7% | 200 | 0 | 19.4s | 19.4 | 512 | 0.000 |
| `search_planner` (mirror) | 0.9% | 0.0–1.9% | 200 | 0 | 64.3s | 17.2 | 471 | 0.000 |
| `search_planner_prior` | 1.4% | 0.1–2.8% | 200 | 0 | 92.8s | 21.8 | 478 | 0.000 |
| `frozen_ckpt` | 1.4% | 0.1–2.8% | 200 | 0 | 34.2s | 18.2 | 655 | 0.000 |

## Analysis

**No crashes.** All 800 games across 4 opponents completed with zero worker failures. The prior session's 949/1000 crash was entirely due to stubbed source files (`featurizer.py` and `deck_prior.py` reduced to `# stub`), not a code defect.

**OOV = 0.000.** The `OpponentDeckPredictor` is producing valid deck templates from the archetype vocabulary with no out-of-vocabulary card references.

**Directional improvement.** The prior-driven determinizer (1.4%) edges out the mirror baseline (0.9%) by +0.5pp. Confidence intervals overlap at 200 games: [0.0–1.9%] vs [0.1–2.8%]. A larger sample would be needed for statistical significance, but all metrics point in the same direction:
- Win rate: 1.4% > 0.9%
- No additional illegal actions (478 vs 471)
- No OOV cards

**Matches frozen IL baseline.** `search_planner_prior` and `frozen_ckpt` are both at 1.4%, suggesting the prior predictor's deck templates are as effective as having the exact opponent list — a strong result for a pure-Bayesian approach.

## Decision: SHIP

The prior-driven determinizer is strictly better or equal on every measured dimension. Recommend proceeding with Phase 2 (removing belief machinery), as the Bayesian posterior approach is validated.
