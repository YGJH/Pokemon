# Prior-based opponent deck determinization

**Date:** 2026-08-12
**Status:** approved, pending implementation plan

Replace the learned belief heads' inference-time role with a pure Bayesian
posterior over all 362 mined archetypes, seeded by their mined frequency. The
determinizer picks the closest archetype and searches against its
representative decklist. The belief heads remain as a training-time auxiliary
loss; nothing at inference reads them any more.

---

## 1. Motivation

### 1.1 What the current path does

`build_submission.py`'s `main.py` template calls
`ptcg_il.search_infer.predict_opponent_deck` on every decision. That function
runs a belief-head forward pass and walks a four-tier fallback chain:

1. `arch` head argmax, if p ≥ 0.5 — classifies over the **9** ids in
   `archetypes.json:opp_ids`
2. `ArchetypePosterior` over the same 9 ids, uniform prior, if p ≥ 0.80
3. `deck` head's card distribution, rounded into a 60-card bag
4. `[]` → the Rust determinizer assumes the opponent mirrors our own deck

### 1.2 The hypothesis set is the binding constraint, not the model

Measured against the real `data/archetypes.json` (362 clusters, every one
carrying a 60-card `representative`). Metric is **how many of the opponent's
60 cards the determinizer's template gets right** — that is what decides
whether a rollout passes through a possible world. Opponents are drawn
weighted by mined `frequency`; observed cards are drawn without replacement
from the true deck.

| k cards seen | mirror (Rust fallback) | posterior over 9 `opp_ids` | *oracle* best of those 9 | **posterior over all 362** |
|---|---|---|---|---|
| 1 | 19.4 | 42.7 | 51.8 | **44.6** |
| 2 | 19.9 | 47.7 | 51.6 | **51.9** |
| 4 | 19.4 | 50.0 | 51.5 | **57.1** |
| 8 | 19.5 | 50.7 | 51.3 | **59.1** |
| 12 | 19.9 | 52.2 | 52.5 | **59.4** |
| 20 | 19.4 | 51.6 | 51.8 | **59.6** |

The 9 `opp_ids` cover only **68.7%** of games by frequency, so a *perfect*
classifier over them caps at ~52/60. Widening the hypothesis set to all 362
beats that ceiling from k=4 onward and reaches 59.6/60. No model change is
involved — the ceiling was in the label space.

### 1.3 The frequency prior is the dominant lever

Top-1 archetype identification, same simulation:

| k | uniform prior | frequency prior |
|---|---|---|
| 1 | 0.295 | **0.560** |
| 4 | 0.255 | **0.818** |
| 8 | 0.343 | **0.885** |
| 20 | 0.590 | **0.943** |

The frequency distribution is heavily skewed — the top cluster is 37% of all
153,702 player-slots — so the prior carries early turns and multiset
elimination takes over later. Uniform is 15 points of overlap and 26 points of
top-1 worse at k=1.

**Accepted risk:** this bakes in the training corpus's metagame. Kaggle's
opponent pool is other submissions, not the mined corpus, so early-turn
guesses are confidently skewed toward whatever the corpus favoured. The
posterior recovers within a few observed cards via elimination; the exposure is
turn 1–2.

### 1.4 Confidence-thresholded abstention is actively harmful

Falling back to mirror below p ≥ 0.80 *lowers* mean overlap, because the thing
being fallen back to is worth 19.4:

| k | always argmax | abstain to mirror below 0.80 |
|---|---|---|
| 1 | **44.6** | 27.3 |
| 4 | **57.1** | 42.5 |
| 8 | **59.1** | 49.7 |
| 20 | **59.6** | 55.8 |

The old threshold made sense when the alternative was a coherent-ish learned
card bag. Against a mirror fallback there is no k at which abstaining pays.
**Decision: never abstain.**

### 1.5 `epsilon` is not a knob worth tuning

`epsilon` is the mass a hypothesis keeps per observed copy it cannot explain.
Tested under opponents playing *variants* of their cluster (d cards swapped for
random other cards), which is the realistic regime — the simulations above
assume the opponent plays the representative exactly, which real opponents do
not:

mean overlap/60 of the argmax template with the opponent's **actual** deck

| d swapped | k | ε=1e-6 | **ε=1e-3** | ε=1e-2 | ε=0.05 | ε=0.2 |
|---|---|---|---|---|---|---|
| 0 | 4 | 57.6 | **56.8** | 53.5 | 31.9 | 13.0 |
| 0 | 16 | 59.8 | **59.6** | 59.2 | 41.3 | 7.2 |
| 4 | 4 | 53.5 | **53.8** | 50.8 | 30.1 | 13.1 |
| 4 | 16 | 55.7 | **55.7** | 54.9 | 38.5 | 7.3 |
| 10 | 4 | 47.0 | **47.6** | 43.9 | 28.3 | 13.0 |
| 10 | 16 | 50.5 | **50.6** | 49.9 | 36.1 | 6.9 |

The existing default `1e-3` is at or within 0.8 of the best value in every
cell, and is strictly better than `1e-6` once opponents deviate at all. Large
epsilon degrades *with more observations* (ε=0.2 falls 13.0 → 6.9 as k rises),
because the prior swamps the evidence — a useful sanity signal if it ever
regresses. **Decision: keep `DEFAULT_EPSILON = 1e-3`, unchanged.**

Even at d=10 — a sixth of the deck differing from its cluster representative —
the template still lands ~50/60 against the mirror's 19.4.

### 1.6 Cost

Measured on the 362-archetype set: **3.34 ms** to construct, **0.55 ms** per
`posterior()` call with 15 cards observed. The current code constructs a fresh
`ArchetypePosterior` *inside* `predict_opponent_deck` on every decision; the
new design builds it once. Replacing a belief-head forward pass with 0.55 ms
makes the decision path **faster**, which matters against the MCTS iteration
budget.

---

## 2. Pre-existing bug this change fixes

`ptcg_rl/train.py:_sample_opp_deck` constructs
`OpponentDeckOracle(policy=None, vocab=..., archetypes=...)`. `__init__`
evaluates `self.device = device or str(next(policy.parameters()).device)`,
which raises `AttributeError: 'NoneType' object has no attribute 'parameters'`
— absorbed by the function's own bare `except Exception: return []`.

Verified by direct construction. **RL's K=8 determinizations per state have all
been running on the Rust mirror heuristic since the code was written**, at
19.4/60, with no error surfaced. There is consequently no belief-driven RL
determinization behaviour for this change to invalidate.

---

## 3. Architecture

### 3.1 New module: `python/ptcg_il/deck_prior.py`

`ArchetypePosterior` moves here from `ptcg_rl/belief.py`, joined by the new
predictor. This corrects the current backwards layering — `ptcg_il/belief_infer.py`
imports from `ptcg_rl`, while every other cross-package dependency runs
IL ← RL. The module must stay **torch-free and numpy-optional**: it reads only
`archetypes.json` and observation dicts, and it is copied flat into the
submission bundle.

```
python/ptcg_il/deck_prior.py
    DEFAULT_EPSILON = 1e-3
    ArchetypePosterior          # moved verbatim from ptcg_rl/belief.py
    all_archetype_ids()         # every id in archetypes.json, ordered
    frequency_prior()           # per-id `frequency`, for the `priors` argument
    extract_opp_visible_cards() # moved from search_infer.py (pure dict parsing)
    ObservedOpponentCards       # per-card-id running max across observations
    OpponentDeckPredictor       # the thing call sites use
```

`ptcg_rl/belief.py` is deleted. `python/tests/test_rl_belief.py` moves to
`python/tests/test_deck_prior.py` with its imports repointed; its existing
assertions about the posterior stay valid, since the class itself is unchanged.

### 3.2 `ArchetypePosterior` — moved, not modified

The multiset-aware likelihood is already correct: it counts copies, penalises a
third copy of a card an archetype runs two of, and reserves `epsilon` per
unexplained copy so a near-miss degrades instead of dying. It already accepts
`opp_ids` and `priors` arguments. Everything this design needs is
constructor-level configuration, not a code change.

The one thing to preserve carefully during the move: the `opp_ids` argument
order **is** the class order for anything that indexes by position. Nothing in
the new design indexes by position (the predictor works in archetype ids), but
the constraint is inherited and must not be quietly dropped.

### 3.3 `ObservedOpponentCards` — accumulation

Holds a `Counter[int]` and, per observation, takes the **element-wise maximum**
against that observation's `extract_opp_visible_cards` snapshot. Not a sum: a
card that moves between zones across turns would otherwise manufacture a
phantom copy, and each phantom copy the true archetype cannot explain costs it
a factor of `epsilon` — a manufactured elimination of the correct answer.

`reset()` is called when a new game starts. In the submission that is
`obs.select is None` (the deck-selection step), matching the existing
`_opp_visible_cards = []` reset already in the `main.py` template.

Effective k matters: k=2 → 47.2 overlap vs k=8 → 55.1 in the d=4 variant
regime, so accumulation is worth roughly +8 cards over a per-turn snapshot.

**Resolved during planning — no hidden-information leak.** `AGENT_SPEC.md:57`
specifies `prize: [Card|None]` with face-down cards (`active[0]`, `prize[i]`)
arriving as `None`, so the existing `isinstance(card, dict)` guard is what
excludes them. Every number in §1 stands.

**Two real defects found in the same function, both fixed by consolidation.**
There are currently three implementations of "which opponent cards can I see":
`search_infer.extract_opp_visible_cards`, `vec_env._extract_opp_visible_card_ids`
(a byte-identical copy), and `belief_labels.opp_visible_counts`. Only the third
is right, and it is the one measured at 98.4% agreement with the observation's
own `deckCount + handCount + face-down prizes`. The other two:

1. **Never filter on `playerIndex`.** The Stadium is a shared zone; crediting
   one *we* played to the opponent is false evidence, and an observed card the
   true archetype cannot explain costs it a factor of `epsilon` — a
   manufactured elimination of the correct hypothesis.
2. **Iterate the Stadium as a list.** `state["stadium"]` is a bare dict;
   `for card in (... or [])` over it walks its string keys, and the
   `isinstance` guard then rejects every one. The Stadium currently contributes
   nothing at all. Fixing the shape without also adding the owner filter would
   convert a silent no-op into active false evidence — both or neither.

They also discard evidence: `energyCards` and `tools` the opponent attached
come out of their deck and are as much signal as their discard pile.

`deck_prior` therefore adopts `belief_labels.opp_visible_counts`' rules rather
than becoming a fourth variant, and the two old copies are deleted with their
callers.

### 3.4 `OpponentDeckPredictor` — the call-site API

```python
class OpponentDeckPredictor:
    def __init__(self, archetypes, *, ids=None, use_frequency_prior=True,
                 epsilon=DEFAULT_EPSILON): ...
    def reset(self) -> None: ...
    def observe(self, obs_dict) -> None: ...
    def template(self) -> list[int]: ...
    def sample_templates(self, k, rng) -> list[list[int]]: ...
    def posterior(self) -> dict[int, float]: ...
```

- Defaults: **all 362 ids**, **frequency prior**, `epsilon=1e-3`.
- `template()` always commits to the argmax representative. No threshold, no
  empty-list return, no mirror fallback (§1.4).
- `sample_templates(k, rng)` draws `k` archetype ids from the posterior
  categorical and returns their representatives.
- `posterior()` is exposed for logging and tests, not for control flow.

### 3.5 Two consumers, two policies

| Consumer | Templates needed | Policy |
|---|---|---|
| Submission `main.py`, `live_eval` | 1 | `template()` — argmax |
| `ptcg_rl.train` determinization | K=8 (`mcts_k_determinizations`) | `sample_templates(K, rng)` |

The RL site is a deliberate departure from "always argmax". K determinizations
exist to cover uncertainty across *different* worlds; handing the aggregator 8
identical worlds would collapse `mcts_k_determinizations` to 1 while leaving
the config knob looking effective. Sampling the categorical is still pure prior
calculation with no belief model — it takes the distribution's draws rather
than its mode.

`ptcg_rl/search.py:1114` is untouched: that path is self-play against a *known*
opponent decklist and needs no inference at all.

---

## 4. Call-site changes

### 4.1 `ptcg_il/search_infer.py`

- Delete `predict_opponent_deck` (all four tiers).
- Move `extract_opp_visible_cards` out to `deck_prior.py` with no re-export
  left behind. Every caller is in this repo and is updated in the same change;
  a compatibility alias would only preserve an import path that the bundle
  rewrite rules would then also have to know about.
- No remaining reference to `policy.belief_logits`, and no
  `from model.belief_posterior import ...` inside a function body.

### 4.2 `scripts/build_submission.py`

- `MAIN_PY_TEMPLATE`: replace the `predict_opponent_deck(...)` call with a
  module-level `OpponentDeckPredictor` built once at import from the bundled
  `archetypes.json`; `agent()` calls `.observe(obs_dict)` then `.template()`.
  Keep the existing `obs.select is None` branch, changing
  `_opp_visible_cards = []` to `_predictor.reset()`.
- `EXTRA_FILES`: `("ptcg_rl/belief.py", "belief_posterior.py")` →
  `("ptcg_il/deck_prior.py", "deck_prior.py")`.
- Delete the special-case line rewriting
  `from ptcg_rl.belief import` → `from model.belief_posterior import`.
- `REWRITE_RULES`: add **both** import forms —
  `from ptcg_il\.deck_prior import` → `from model.deck_prior import`, and
  `from ptcg_il import deck_prior` → `from model import deck_prior`. Missing
  the second form is a known blind spot in this rewrite table and fails only
  at agent runtime.
- The bundle no longer needs the belief heads at inference, but `model.pt`
  still carries their weights and `model/belief.py` is still needed to
  construct the `Policy`. **No change to `MODEL_FILES`.**
- `archetypes.json` becomes *more* load-bearing, not less: it is now the entire
  opponent model. The existing `archetypes_sha1` pin and the MCTS-only bundling
  rule both stay.

### 4.3 `ptcg_il/belief_infer.py`

Deleted. `OpponentDeckOracle` and `deck_from_distribution` lose their only
callers. `python/tests/test_belief.py`'s `TestOpponentDeckOracle` and
`deck_from_distribution` tests go with them; any test in that file covering the
belief *heads themselves* stays, since the heads still train.

### 4.4 `ptcg_il/live_eval.py`

`SearchPlannerAgent` keeps its shape but takes an `OpponentDeckPredictor`
instead of an oracle, and calls `observe()` / `template()` rather than
`predict()`. Its `or None` coalescing goes: `template()` never returns empty.

### 4.5 `ptcg_il/cli.py`

The `search_planner_belief` opponent drops its
`if _belief_weights(args) is not None:` guard and its CPU `deepcopy` of the
policy — there is no NN in the path, so it works under `--no-belief` too, and
the fork-safety comment about CUDA tensors becomes moot. The opponent key
`search_planner_belief` becomes `search_planner_prior`; it appears in eval
output, so the rename is visible in logs and in any downstream comparison
against previously recorded numbers.

### 4.6 `ptcg_rl/train.py`

`_sample_opp_deck` is rewritten against `OpponentDeckPredictor`, built once
rather than per call (it currently reloads `vocab.json` and `archetypes.json`
on **every one of K×batch calls**). Its bare `except Exception: return []`
must go: that clause is what hid the `policy=None` crash for the lifetime of
the code. Failures should raise.

`vocab.json` is no longer needed here — the predictor works in engine card ids
throughout, with no vocab-index hop. (That hop was itself the source of a
previously-fixed bug in tier 3.)

---

## 5. Phase 2 — removing the belief machinery entirely

Scope decision taken 2026-08-12, after this design was approved: the belief
route is removed from the codebase, not merely disconnected from inference.
Phase 1 (§1–§4) removes the heads' consumers; Phase 2 removes the heads.

### 5.1 It is two objects, not one

`ptcg_il/model/belief.py` defines two unrelated things sharing a name.

**`BeliefHeads`** — four auxiliary prediction heads reading the CLS token,
supervised by the `bel_*` shard labels through `belief_loss`. After Phase 1
nothing consumes its output. Already gated by `--no-belief`.

**`BeliefModule`** — encodes `log_feat` (the game log) and `policy.py:299`
**adds it to the CLS token**: `cls_token = h[:, 0, :] + belief`. The value head
and the pointer's CLS key/value read that token, so it is a **policy input, not
a head**. `--no-belief` does not gate it, and since `history_gru` was removed
(`policy.py:268-278`) it is the model's only remaining cross-turn mechanism.

### 5.2 Measured cost

On a real 26,586-row shard:

| | shard bytes | loader, per batch @ bs=1024 | model fwd+bwd |
|---|---|---|---|
| `BeliefHeads` (`bel_*`) | 9.89 MiB / 2.9% | 14.9 MiB densify | 0.5% |
| `BeliefModule` (`log_*`) | 20.49 MiB / 6.0% | 26.5 MiB `log_card_feat` rebuild | included |
| both | 30.37 MiB / **8.9%** | **41.4 MiB** of ~437 MB/batch (~9.5%) | ~0.5% |

Plus ~7% of build-shards pass B for label construction. The heads are 13.7% of
parameters but 0.5% of compute — the trunk dominates, and the heads hang off an
encode pass that already happened.

`_densify_belief` (`dataset.py:796`) runs **unconditionally**, so `--no-belief`
never removed the loader-side cost; that is the one un-gated term.

### 5.3 Removing `BeliefModule` is an ablation, and is treated as one

It deletes a live policy input and makes the policy purely Markov on the
current observation. It is also the larger share of the speedup. Plan Task 15
measures it against the pre-change baseline (same archetype, seed, steps and
batch size) rather than assuming it is free, and prescribes restoring
`BeliefModule` alone if accuracy regresses — the two halves are independent.

### 5.4 Compatibility

`load_policy_state`'s existing `history_gru` stale-prefix drop
(`policy.py:444-458`) is extended to `belief.*` and `belief_heads.*`, so
checkpoints trained before the removal keep loading. This is plan Task 10 and
runs *first*, because a training run was in flight when the decision was taken
and its output must remain usable.

`feat_dims` is compared with `if k in live` (`policy.py:530`), so *removing*
`LOG_FEAT_DIM` is tolerated by construction. Removing a key is safe; changing a
surviving key's value is not.

Existing shards keep working with the post-removal code — `ShardDataset` reads
an explicit key list, so the `bel_*` and `log_*` arrays are simply never read.
The code change and the ~23-minute corpus rebuild are therefore decoupled.

## 6. What is explicitly *not* changing

- `ptcg_rl/search.py`'s known-deck self-play determinization.
- The Rust `libptcg_search.so` interface. It still takes one
  `opp_deck_json` template per root; nothing about the FFI changes.
- `opp_ids` stays in `archetypes.json` as unused metadata. It was the belief
  head's class order; removing it from the artifact has its own lineage
  consequences and is a separate change.
- `ptcg_mine` has no belief *code* — every mention is prose explaining why
  archetype ids are append-only. The rationale is rewritten (ids are pinned by
  trained checkpoints and `--archetype-self N`, which remains true), not the
  behaviour.

---

## 7. Testing

### 6.1 Unit

- `ArchetypePosterior` after the move: the existing `test_rl_belief.py`
  assertions, repointed.
- `frequency_prior` / `all_archetype_ids` against a fixture `archetypes.json`,
  including the id-keyed-dict vs list container forms `_representative_of`
  already handles.
- `ObservedOpponentCards`: a card appearing in two zones across two
  observations yields count 1, not 2 — assert against the summing form
  explicitly, since that is the failure that silently eliminates the correct
  archetype.
- `OpponentDeckPredictor.template()` returns 60 cards with **zero**
  observations (the never-abstain guarantee), and never returns `[]`.
- `sample_templates(k, rng)` is reproducible under a seeded rng and returns
  varied templates when the posterior is diffuse.
- Determinism: `template()` is a pure function of the observation sequence.

Per `CLAUDE.md`, any test that counts occurrences must fail on zero examined,
and each new guard is validated by **mutation** — in particular the
max-vs-sum accumulator and the removal of the `except Exception` in
`_sample_opp_deck`.

### 6.2 Bundle

`tests/test_build_submission.py:222` currently asserts
`"predict_opponent_deck" not in called` for the `--no-mcts` build. Update to
the new symbol. Add the standing bundle check from prior experience: exec the
generated `main.py` in a namespace with **no `__file__`** and confirm the
predictor constructs — an `import main` smoke test binds `__file__` and hides
exactly this class of failure.

### 6.3 Live A/B — the gate

Unit tests cannot validate this change; every number in §1 is a simulation
against `archetypes.json`, not a played game. Run `live_eval` with both
planners as separate opponents — the mirror-determinized `search_planner` and
the new prior-backed one — and compare win rate on real engine games. The two
are already wired as distinct opponent keys for precisely this comparison.

Ship on the win-rate delta, not on the overlap numbers.

---

## 8. Open items

None blocking. The prize-zone question raised against §3.3 was resolved during
planning against `AGENT_SPEC.md:57` — face-down cards arrive as `None` and are
already excluded. See §3.3 for that resolution and for the two genuine defects
found in the same function.

One item is deliberately unresolved and carried as a measured risk: whether
removing `BeliefModule` (§5.3) costs accuracy. It is taken knowingly, and plan
Task 15 is the experiment that answers it.

Implementation plan: `docs/superpowers/plans/2026-08-12-prior-deck-determinizer.md`
