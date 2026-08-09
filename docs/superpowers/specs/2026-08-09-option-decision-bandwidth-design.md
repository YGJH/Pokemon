# Per-option decision bandwidth for ATTACK and RETREAT

**Date:** 2026-08-09
**Status:** design, pending implementation

## Problem

A live game against an idle opponent (every turn `END`, no attacks, no cards
played) was lost by the trained specialist. The model looped
`ATTACH -> ATTACK 934 (zero damage, draws) -> END` fourteen times, decked itself
out, and lost. `RETREAT` was offered 42 times and never selected; a fully
evolved 320 HP / 180 damage Grimmsnarl ex sat on the bench untouched.

The observation is not at fault. Every fact needed to make the right decision is
featurized and unsaturated:

| fact | location |
|---|---|
| deck remaining | `sum_feat[1] = deckCount / 60` -> `0.0` at empty |
| turn number | `cls_feat[0] = turn / 50` -> `0.6` at turn 30, not clipped |
| attack damage ratio | `opt_scalar[6]` -> `0.0` for attack 934 |
| OHKO available | `opt_scalar[7]` |
| KO pressure both ways | `cls_feat[93:97]` |
| bench Pokemon HP/attacks | `poke_card_feat[1..5]` |
| RETREAT offered | `opt_type == 12` |

The defect is that the policy cannot *express* the comparison it needs to make,
and separately has never been trained to make it.

### Root cause 1 — no per-option bandwidth (this spec)

`featurizer.py:941` and `featurizer.py:951` both resolve their option reference
to `AREA_ACTIVE, your_index, 0`. ATTACK and RETREAT therefore have
**byte-identical** `src`, `tgt` and `card_enc` terms in `PointerHead`'s additive
base. Measured on `checkpoints_a1/ckpt-best.pt` (EMA, 256 real samples), the
across-option L2 spread of each base term is:

| term | spread |
|---|---|
| `src` | 7.37 |
| `tgt` | 2.48 |
| `card_enc` | 1.06 |
| `opt_type_emb` | 0.32 |
| `attack_enc` | 0.015 |

against a general base spread of 8.80. On rows offering >= 2 ATTACK options,
`src`, `tgt`, `card_enc` and `opt_type_emb` each separate the options by exactly
0.0000. `opt_scalar` carries 0.86% of the pre-activation energy into `opt_in`
(scalar path rms 0.0044 vs base 0.5165).

So the entire ATTACK-vs-RETREAT decision rides on `opt_type_emb` (0.32) plus
eight scalar dims at under 1% of the head's energy.

There is a second blindness: `RETREAT` carries no fields beyond `type`, and the
promote-target is resolved by a *separate subsequent select*. At the moment the
model decides whether to retreat, it cannot see that Grimmsnarl ex is what it
would promote to.

### Root cause 2 — no training signal (explicitly out of scope)

`shard_writer.py:688` sets `value_target = 1.0 if r == 1 else -1.0` from
`rewards(ep)[p]`, constant across every decision point of that episode. The
corpus is human Kaggle games ending on prizes in 15-25 turns; **no training
episode ends by self-deck-out**. No gradient in IL training says the loop is
fatal.

**This spec does not fix the deck-out.** It gives the model the capacity to
represent the decision. Supplying the signal requires RL self-play (where a
policy that decks itself out receives reward -1) or an inference-time mask.
Both are separate work. This spec is a prerequisite for the RL step: a policy
that can barely express "retreat vs attack" will learn it slowly even under
correct rewards.

## Design

### A. New `opt_scalar` dims: `F_OPT` 8 -> 13

All eight current dims are in use (`featurizer.py:853-988`), so this is an
append. Indices 0-7 keep their present meaning.

| dim | applies to | definition |
|---|---|---|
| 8 | ATTACK | `min(draw_count / max(deckCount, 1), 1.0)` — deck-out risk |
| 9 | ATTACK | `1.0` if the attack's static damage is 0, else `0.0` |
| 10 | RETREAT | `(attached_energy_on_active - retreatCost) / RETREAT_N`, clipped to `[-1, 1]` |
| 11 | RETREAT | best damage ratio achievable from the bench vs the opponent's Active |
| 12 | RETREAT | best bench current HP / active current HP, clipped to `[0, 1]` |

Dims 11-12 are the load-bearing pair: they are the only way the model can weigh
"attack now" against "retreat and attack next turn," which is exactly the
comparison it is failing.

Dims 11-12 are constant across all RETREAT options within a select (RETREAT
never offers two options in one select — measured over 60 episodes). They are
not meant to separate RETREAT options from each other; they separate RETREAT
from ATTACK, which is the decision in question.

Dim 8 covers **attack-driven draw only**. Supporter-driven draw (Professor's
Research and similar) is not covered in this version. The observed failure is an
attack; extending to hand-card options is deliberately deferred rather than
guessed at.

**Dims 11-12 ignore currently-attached energy**, deliberately. The value of
retreating is realised *next* turn, after an attach, so gating the bench's
damage on energy already attached would understate exactly the play we want the
model to find (retreat to a fully evolved Grimmsnarl ex, attach, swing). Dim 10
carries the affordability of the retreat itself, which is the constraint that
actually binds this turn. Dim 11 is therefore "best damage this bench Pokemon
could do", read from the static attack blocks in `poke_card_feat`, not "best
damage it could do right now".

### B. New `opt_bench_idx` tensor

`int64[O_MAX]`, filled with `-1`.

For RETREAT options: the state-token row index of the bench Pokemon maximising
achievable damage ratio against the opponent's Active. Deterministic tie-break
by **lowest row index** — shards must be byte-reproducible. Empty bench yields
`-1`.

For every other option type: `-1`.

`PointerHead.forward` gains a sixth additive base term:

```python
bench = self.gather(h_aug, x["opt_bench_idx"])   # [B, O, D]
base = opt_type_emb + src + tgt + card_enc + attack + bench
```

`-1` maps to `null_token` via the existing `gather`, so non-RETREAT options
receive a constant vector. That shifts RETREAT relative to other types (intended)
without inducing spurious separation among options that share the value.

**Why a separate field rather than reusing `tgt`:** `tgt` means "the target this
option names." RETREAT names none. Overloading it would make one tensor mean two
different things depending on `opt_type`, and the heuristic choice can disagree
with what the follow-up select actually promotes. A distinct field keeps `tgt`
honest while still buying a real encoded-state gather (spread 2.48) instead of
eight scalars at 0.86% of energy.

**Hot-path constraint:** `featurize` is on the RL rollout path, where the
featurizer was previously 77% of rollout cost and 69% of *that* was one scalar
`np.clip`. This computation must be pure-Python scalar arithmetic over small
loops, matching the existing `_clip_norm` convention (`featurizer.py:227`, "Pure-
Python arithmetic on purpose"). Bench is at most 8 slots x 3 attack blocks = 24
reads. Compute lazily, once per select, only when the select offers at least one
RETREAT option.

### C. Attack draw-count numerics: `F_ATK` 43 -> 45, `F_CARD` 212 -> 218

`keywords.py:28` matches `draw \d+ (?:more )?card` but stores the result
**binary** — the row says an attack draws, not how many.

Measured over `all_attack()` (1556 attacks, 36 mention draw):

| form | count | example |
|---|---|---|
| explicit number | 16 | `draw 2 cards` (max observed: 6) |
| draw-to-hand-size | ~6 | `Draw cards until you have 7 cards in your hand.` |
| `Draw a card.` | ~12 | implicit count of 1 |
| other (`Each player draws 3 cards`) | rest | affects both players |

**The dangerous form has no static count.** `Draw cards until you have 7 cards in
your hand` draws `max(0, 7 - handCount)`, which depends on state. This is almost
certainly attack 934 in the observed game, and it is precisely the form that
decks a player out. A single static count cannot represent it, so the design
splits into two static dims plus a featurize-time combination:

Static table — `attack_static_row[16]` = `[damage/ATKDMG_N,
energy-cost histogram(12)/ATKCOST_N, len(energies)/ATKCOST_N,
draw_fixed/DRAW_N, draw_to_hand/DRAW_N]`:

- `draw_fixed` — the explicit count; `Draw a card` parses to 1; else 0.
- `draw_to_hand` — the N in "until you have N cards in your hand"; else 0.

So `F_ATK == 16 + K_EFFECT == 45` and
`F_CARD == 52 + K_EFFECT + 2 + 3*F_ATK == 218`.

`opt_scalar[8]` (dim 8 in section A) then combines them against live state:

```
draw_est = draw_fixed + max(0, draw_to_hand - handCount)
opt_scalar[8] = min(draw_est / max(deckCount, 1), 1.0)
```

This is the dim that makes the observed failure representable at all: in that
game the model held a large hand against an empty deck, and `draw_to_hand`
without `handCount` would have read the risk as constant across the whole game.

`cards.py:39-46` already asserts both dims against `K_EFFECT`; those assertions
must be updated in the same commit, and they are what turns a mismatch into an
import error rather than a silent row of the old width.

This is also the prerequisite for the deck-out mask (separate work): without a
count, that rule degrades to a `deckCount <= k` guess.

`DRAW_N = 10.0`, derived rather than chosen: the largest explicit count in
`all_attack()` is 6, and the largest draw-to-hand-size target is 7 (reached only
from an empty hand), so 10 is a ceiling no attack can exceed. Parsing must be
verified against all 36 matches during implementation — the ~12 `Draw a card`
cases and the `Each player draws` case are the ones most likely to be
mis-parsed, and a silent 0 there reads as "no deck-out risk".

## Blast radius

`F_OPT` feeds `pointer.opt_in`; `F_CARD`/`F_ATK` feed `CardEncoder` and
`AttackFeaturizer`. This invalidates **every shard and every checkpoint** via
`Policy.config["feat_dims"]` — the mechanism working as designed. There is no
migration path and none should be added: a checkpoint trained on the old widths
is not a smaller version of the new model, it is a different one.

Touch points:

| file | change |
|---|---|
| `ptcg_mine/keywords.py` | parse `draw_fixed` and `draw_to_hand` alongside the binary keyword |
| `ptcg_mine/cards.py` | `attack_static_row` 14 -> 16 numeric; new `DRAW_N`; update both dim assertions |
| `ptcg_il/featurizer.py` | `F_OPT` 8 -> 13, `F_ATK` 43 -> 45, `F_CARD` 212 -> 218; emit `opt_bench_idx`; new scalar dims; `CARD_ATTACK_BLOCK_START` follows `K_EFFECT` |
| `ptcg_il/featurizer.py:1100` | `option_groups` dedup key must include `opt_bench_idx` |
| `ptcg_il/model/pointer.py` | sixth additive base term |
| `ptcg_il/model/policy.py` | `Policy.config["feat_dims"]` records the new widths |
| `ptcg_il/train/dataset.py:69` | `opt_bench_idx` into `_INT_KEYS` |
| `ptcg_il/shard_writer.py` | new tensor flows through pass B |
| `ptcg_il/diagnose.py:60,115` | new ref field in the reported groups |
| `ptcg_il/qa.py:508,545` | comments naming the option-key fields |
| `ptcg_rl/actor.py:51` | `opt_bench_idx` in the masked-tensor list |
| `model/featurizer.py`, `model/pointer.py` | **vendored Kaggle-bundle copies — already drifted from `python/ptcg_il/`; must be re-synced or the submission featurizes differently than training** |

The vendored copies are the highest-risk item here. A drift between them and
`ptcg_il/` does not raise; it produces a submission whose inputs disagree with
its weights, which presents as an unexplained live-vs-offline gap.

## Verification

Offline top-1 is a poor proxy — a 78%-offline / 48%-live gap is already on
record. Three checks, in order:

1. **Capacity check (direct, model-independent of win rate).** Re-run the
   across-option spread measurement on rows offering ATTACK and RETREAT
   together. The current value is exactly 0.0000 for four of five base terms.
   Success is a non-zero spread attributable to `opt_bench_idx` and the new
   scalars. This tests what the spec actually claims to fix.
2. **Behavioural check.** RETREAT offered-vs-chosen rate, before and after. The
   observed game is 42 offered / 0 chosen.
3. **Live win rate**, via `live_eval`, before and after, against the existing
   baselines. Expected: no regression. **Not** expected: the deck-out to
   disappear — see "Root cause 2" above.

Check 1 is the acceptance criterion. Check 3 is a regression guard, not evidence
the spec succeeded.

## Testing

- Featurizer unit tests: `opt_bench_idx` is `-1` for every non-RETREAT option;
  points at a real bench row when the bench is non-empty; `-1` on empty bench;
  tie-break is the lowest row index.
- Determinism: featurizing the same observation twice yields byte-identical
  `opt_bench_idx` and `opt_scalar`.
- Dim assertions in `cards.py` fail on a `K_EFFECT` mismatch (verify by
  mutation — bump `K_EFFECT` and confirm the import raises).
- Draw parsing: assert against **all 36** draw-mentioning attacks in
  `all_attack()`, not a sample — `Draw a card` must yield `draw_fixed == 1` and
  `Draw cards until you have 7 cards in your hand` must yield
  `draw_to_hand == 7`. A silent 0 in either reads as "no deck-out risk", which
  is the exact failure this spec exists to make visible. The test must assert a
  non-zero number of attacks examined.
- `opt_scalar[8]` combination: with `draw_to_hand=7`, `handCount=6`,
  `deckCount=1`, the risk reads 1.0; with `handCount=7` it reads 0.0.
- `option_groups` splits two options that differ only in `opt_bench_idx`
  (verify by mutation: remove the field from the key and confirm the test
  goes red).
- Tests that count occurrences must fail on zero examined — a RETREAT test that
  runs over a fixture with no retreat options passes vacuously.
- Vendored-copy sync: a test asserting `model/featurizer.py` and
  `ptcg_il/featurizer.py` agree on `F_OPT`, `F_ATK`, `F_CARD`. This drift is
  currently unguarded.

## Cost

Full re-mine (static tables rebuild) -> rebuild shards (~23 min over the 55k
corpus) -> retrain every specialist. Both stages are stamp-guarded, and the
fingerprint already covers `featurizer.py`, so the edit invalidates them
automatically rather than needing `--force`.
