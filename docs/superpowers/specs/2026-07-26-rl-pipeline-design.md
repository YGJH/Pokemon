# Five-stage training pipeline with PPO self-play

**Date:** 2026-07-26
**Status:** approved, ready for implementation planning
**Governing spec:** `RL_SPEC.md` (revision 4). Where this document and `RL_SPEC.md`
disagree on hyperparameters, tensor layout, or gate thresholds, `RL_SPEC.md` wins.

---

## 1. Goal

Turn `scripts/run_pipeline.sh` into a five-stage pipeline that runs end to end:
build the Rust engine, collect the corpus, featurize it, train two per-deck IL
specialists plus the opponent-belief models, and then improve one specialist by
PPO self-play anchored to its IL prior.

This delivers `RL_SPEC.md` phases **R0c** (rebuild shards, retrain specialists,
re-record baselines), **R1** (critic repair), and **R2** (PPO + KL anchor, one
deck, self-play, no MCTS in the loss).

Out of scope, deferred to follow-up specs:

- **R3** — MCTS distillation (`L_search`, `ptcg_rl/search.py`).
- **R4** — two-deck alternating league, Elo, promotion gate.
- **R0e** — Rust `VecEnv`/featurizer port. `RL_SPEC.md` §6.5 downgraded this from
  prerequisite to optional: six workers of the post-R0d NumPy featurizer already
  saturate the GPU at 61.6k dec/s, and the machine has sixteen cores. Do not port
  on principle; re-measure at R3.

MCTS *machinery* is in scope even though MCTS *distillation* is not — the Rust
searcher gains visit counts and a root value, and both belief models feed its
determinizer. This keeps `search_planner_belief` meaningful in live-eval and
leaves R3 with nothing to build but the loss term itself.

---

## 2. Pipeline shape

```
[1/5] Rust engine     cargo build; search_plan → {indices, visits, value}
[2/5] Collect data    ptcg_mine.mine (download → stats → archetypes → vocab)
[3/5] Build shards    ptcg_il.cli build-shards, belief labels unconditional
[4/5] Train
      4a  per-deck IL specialists (arch 0, arch 2)  → checkpoints_a{0,2}/
      4b  belief heads (--belief), trained in the same runs as 4a
      4c  offline eval                              → data/il_baselines.json
[5/5] RL              budgeted by --rl-steps, skipped by --no-rl
      5a  R1 critic repair
      5b  R2 PPO + KL anchor
```

Stages 1–4 keep the existing script's conventions: quiet by default with per-stage
logs under `python/logs/pipeline-<timestamp>/` and a `logs/latest` symlink,
`--verbose` to tee to the terminal, `run_stage`/`fail_tail` for error reporting,
and all relative paths resolved against `python/`.

### 2.1 New and changed flags

| flag | default | effect |
|---|---|---|
| `--no-rl` | off | skip stage 5 entirely |
| `--rl-steps N` | 50000 | PPO optimizer-step budget for stage 5b |
| `--rl-archetype ID` | 2 | which specialist RL trains; must appear in `--archetypes` |
| `--rl-workers N` | 6 | rollout worker processes (§6.5: base rollout saturates the GPU at ~6) |
| `--belief` | **removed** | belief heads now always train; see §4.2 |

Removing `--belief` follows from §4.2: once labels are unconditional there is no
"belief-less corpus" to guard against, and stage 5's determinizer needs the heads.
The per-term `--belief-<name>` weight flags stay for tuning, and setting them all
to zero remains the way to disable the heads.

`--archetypes "0 2"`, `--live-eval`, `--verbose`, `--skip-download`, `--skip-rust`,
`--no-train`, `--no-eval` and the directory flags keep their present meaning.

### 2.2 Stage 5 always runs

Stage 5 runs by default with a step budget rather than being opt-in, so a plain
`./scripts/run_pipeline.sh` is genuinely end to end. `--rl-steps` bounds it;
`--no-rl` skips it. A stage-5 failure is reported like any other stage failure and
exits non-zero — unlike eval failures, which warn and continue.

---

## 3. Stage 1 — Rust engine

`cargo build --release` as today, plus one API change.

`ptcg_search`'s C-ABI `search_plan` currently returns only chosen indices. Per
`RL_SPEC.md` §8.4 it must also return the root's visit distribution and value:

```json
{"indices": [...], "visits": [[option, n], ...], "value": 0.31, "error": 0}
```

`mcts.rs` already accumulates `visits` and `total_value` per node and already
materialises `(action, visits)` pairs internally, so this is a serialization
change, not a search change. The existing `indices` field keeps its exact current
semantics so `live_eval.py`'s planner baseline is unaffected.

`search_plan_free` and the rest of the C-ABI are untouched. No PyO3 migration —
that belongs to R0e, which is deferred.

---

## 4. Stages 2–4 — corpus and IL training

### 4.1 Stage 2 — collect data

Unchanged: `python -m ptcg_mine.mine` with the existing day-sampling, resumable
download, expert selection, archetype clustering, vocab and static-table build.
Exit code 2 keeps its special meaning (Kaggle rate-limit) with the current
resume guidance.

### 4.2 Stage 3 — build shards, belief labels unconditional

`build-shards` writes belief labels for every decision point, always.

Today the labels are conditional and `--belief` is a *training* flag. The current
help text states that training with `--belief` over shards mined without labels
adds "a fully-masked zero term" — a silent degradation that costs a `[B, V]`
matmul and teaches nothing. Making labels unconditional removes the failure mode;
`belief_weights` then controls only whether the heads are trained.

Labels come from `ptcg_il/belief_labels.py` unchanged: sparse index/count pairs
over `BELIEF_K` slots, deck read from the opponent's step-0 action, hand carried
back from the opponent's next decision.

This stage is also `RL_SPEC.md` **R0c**: shards must be rebuilt because the
`opt_card_id` fix (R0b) changed the model's input distribution, invalidating every
existing shard and checkpoint (§6.4.2).

### 4.3 Stage 4a/4b — specialists and belief heads

One training run per archetype in `--archetypes`, output to
`${CHECKPOINT_DIR}_a<id>/`, each carrying its own `decks.json` / `deck.csv` and the
`ptcg_il.deck` record pinning `vocab_sha1` and `archetypes_sha1`.

Belief heads train in the same runs — they are auxiliary heads on the specialist,
not a separate model, so 4b is not a separate invocation. It is listed separately
because it is a distinct deliverable that stage 5 and MCTS depend on.

Additionally, per `RL_SPEC.md` §10.3, `save_checkpoint` must write the
**architecture config** into the checkpoint. The arch-2 checkpoint's `config` is
empty, which forced §2.3 to infer `V/A/D/heads/layers/ff` from state-dict shapes
and is why the packed `main.py` still needs `strict=False`. Stage 5 loads two
policies (θ and frozen π_IL) and cannot rely on shape inference.

### 4.4 Stage 4c — record IL baselines

After training, run offline eval per specialist and write:

```json
{
  "2": {
    "ckpt_sha1": "…",
    "nontrivial_top1": 0.7538,
    "top1": 0.891,
    "value_corr": 0.045,
    "value_std": 0.0006,
    "recorded_at": "2026-07-26T…"
  }
}
```

to `${DATA_DIR}/il_baselines.json`.

`RL_SPEC.md` §13 lists "stale IL baselines after the feature fix" as a **high,
silent** risk, and §10.2 condition 3 currently quotes hardcoded numbers (arch 0:
0.5394, arch 2: 0.7538) that the `opt_card_id` fix invalidated. Recording the
baselines as a pipeline artifact, sha1-pinned to the checkpoint that produced them,
makes the risk structurally impossible rather than a matter of discipline: stage
5's gate refuses to run when the pinned sha does not match the π_IL checkpoint it
was handed.

`value_corr` and `value_std` are recorded here because they answer `RL_SPEC.md`
§14.2 open question 3 for free — whether the dead value head was partly caused by
the dead card features. R1 reads this to decide its scope.

---

## 5. Stage 5 — RL

New package `python/ptcg_rl/`.

| module | role |
|---|---|
| `config.py` | `RLConfig` — every knob in `RL_SPEC.md` §9.4 |
| `actor.py` | masked AR sampling; `sample_action` / `recompute_logp` — **single source of truth for masking** |
| `vec_env.py` | multiprocess rollout over `libcg` |
| `rollout.py` | trajectory assembly, GAE (fp32), advantage normalisation |
| `ppo.py` | clipped surrogate, value loss, entropy, adaptive-β KL, ratio canary |
| `critic.py` | R1 critic repair and diagnostics |
| `belief.py` | 6-way categorical posterior over 𝒟_opp |
| `gate.py` | paired evaluation, Wilson lower bound, IL-regression check |
| `train.py` | CLI: `python -m ptcg_rl.train --deck-archetype 2 --il-ckpt …` |

Tests live in `python/tests/test_rl_*.py`, matching the existing layout. The engine
is injectable — as `ptcg_mine/download.py` injects the Kaggle API — so unit tests
need no `libcg.so`.

### 5.1 `actor.py` — one masking implementation

An action is a *sequence*, not a token: `a = (a_1, …, a_k, STOP)` over
`minCount ≤ k ≤ maxCount` picks.

```python
def sample_action(policy, batch, *, temperature=1.0, generator=None)
    -> tuple[list[int], Tensor, Tensor]:      # picks, logp_joint, entropy_sum

def recompute_logp(policy, batch, actions) -> tuple[Tensor, Tensor]:
    """Teacher-force stored actions under current θ."""
```

Requirements from `RL_SPEC.md` §5:

- `log π_θ(a|s) = Σ_{t=1..k+1} log softmax(mask(logits_t))[a_t]` — the sum
  **includes the STOP step** wherever a STOP column exists. Dropping it makes
  short selections systematically cheaper and biases toward tiny picks.
- Rows that declined an optional select (`minCount == 0`, no STOP column) contribute
  an empty sum. Handle explicitly; never let it become `-inf`.
- Already-selected options are masked at later steps, exactly as
  `_select_multi_raw` does.
- Entropy is likewise summed over AR steps, masked to legal options.

`recompute_logp` is a generalisation of `multiselect_ce` and must reuse it rather
than reimplement it. Rollout, the PPO update and (later) MCTS all call into this
one module; three copies would drift, and the drift is invisible until the policy
proposes illegal picks.

All `k+1` AR steps run GPU-side with no environment round-trip, because the
multi-select state lives in the model. One decision point costs one obs→batch
transfer.

### 5.2 `vec_env.py` — Python multiprocess rollout

No Rust port (§2, and `RL_SPEC.md` §6.5). Reuses `live_eval.py`'s proven pieces:
`_worker_init`, `_find_libcg_path`, and its multiprocessing pool structure.

- `GameInitialize()` once per worker process at import, per the `libcg` process
  invariants; never `restype=c_char_p` on a pointer the engine frees.
- Auto-reset on terminal to keep batches full; terminal rewards returned
  out-of-band so the learner can close trajectories.
- Forward-pass batch **256–512** (§2.3 measured saturation), kept strictly
  distinct from the 16,384 rollout-buffer size.
- Timeouts record a **loss**, never a discarded game (§4.2) — otherwise training
  silently selects for slow policies.

Reward is terminal-only: `r_T = +1` win / `−1` loss / `0` no winner, `γ = 1.0`,
`λ_GAE = 0.95`. Draws appear not to occur; the branch stays as defensive code but
nothing depends on it.

### 5.3 Stage 5a — R1 critic repair

The value head is collapsed: prediction std 0.0006 against target std 0.994,
correlation 0.045, sign agreement 0.556. With `V ≡ 0`, GAE degenerates to the raw
terminal return for every state — plain REINFORCE over ~150-decision episodes,
which will not learn at usable sample cost. **This blocks R2.**

`critic.py` first re-measures on the correctly loaded stage-4 checkpoint (the §2.3
probe's 0.222 std came from a partially-loaded model and is not evidence of
health), then works the §3 candidate list in order:

1. Freeze the trunk, train only `ValueHead`. If it still collapses, the trunk is
   the problem, not the head.
2. Inspect pre-activation std for tanh saturation.
3. Judge against a per-turn baseline, not against zero loss — every decision in a
   game shares one ±1 label, so early-game states are near-unpredictable by
   construction.

If 37,421 IL decision points prove too few, fit the critic on self-play data from
the frozen IL policy: cheap, unlimited, and on-distribution for RL.

**Exit criterion:** `corr(pred, outcome) ≥ 0.35`, prediction std ≥ 0.3, and
accuracy rising monotonically in turn number. Stage 5b does not start until this
passes; the pipeline fails the stage rather than continuing.

### 5.4 Stage 5b — R2 PPO with an IL anchor

```
L = L_clip + c_v · L_value − c_H · H(π_θ) + β · KL(π_θ ‖ π_IL)
```

`L_search` is absent — that is R3.

- `L_clip` with `ratio` on the **joint** sequence, `ε = 0.2`. Advantages normalised
  per minibatch over decision points, not per episode (episode length varies ~3×).
- `L_value = E[(V_θ(s) − V_target)²]`, `V_target = Â + V_old(s)`, updates clipped
  to ±0.2 around `V_old`.
- `KL(π_θ ‖ π_IL)` — mode-seeking direction, per decision point, summed over AR
  steps, masked to legal options. `π_IL` is the frozen stage-4 `ckpt-best.pt` for
  that deck, loaded once, never updated, kept in bf16 under `no_grad`. Both
  policies must be evaluated on the **same** legal mask and the **same** precision;
  assert the supports match rather than assuming.
- β adapts against a fixed budget `κ = 0.02` nats/decision:
  `β ← clip(β · exp((KL_measured − κ)/κ · η_β), 1e-4, 10)`, `η_β = 0.1`.
  κ stays fixed through R2 — a moving κ would make the go/no-go unattributable.

Hyperparameters from §9.4: rollout buffer 16,384 decision points, PPO minibatch
1,024, 3 epochs per iteration, `lr` 1e-5→3e-5, `c_v` 0.5, `c_H` 0.003, grad clip
1.0, temperature 1.0 train / greedy eval.

**Exit criterion:** ≥60% against frozen π_IL over 400 paired games; IL-regression
under 5 points; epoch-0 ratio canary green.

### 5.5 Floating-point discipline

bf16 rollout inference is endorsed (1.57×, 98.7% top-1 agreement with fp32) but
interacts badly with PPO. Measured, comparing log-probs from the *same weights*:
mean `|Δlogp|` 0.02151, p99 0.691, max 2.078. The tail is ties — p99 ≈ ln 2, max ≈
ln 8, ratio floor exactly 1/8 — because bf16 rounds near-equal logits to exactly
equal. At p99 that consumes 500% of the `ε = 0.2` clip range, so clipping would be
driven by rounding rather than by policy change. Upcasting `log_softmax` does not
help; the error is born in the network's bf16 matmuls.

Rules:

- **Never compare log-probs across precisions.** Recompute `logp_old` at the start
  of the first PPO epoch in the update's precision rather than storing the rollout
  value. One extra forward over the buffer removes the whole class of error.
- **Ratio canary:** at epoch 0 of every update, assert p99 `|ratio − 1| < 1e-3`.
  Fail loudly. This also catches unrelated batching and masking bugs.
- GAE accumulation, the KL term and loss reduction stay **fp32**.
- Master weights and the optimizer stay fp32 (autocast only), as IL already does.

### 5.6 `gate.py`

Paired evaluation for the R2 exit criterion and the eventual R4 promotion gate:
same seed and determinization stream, swapping `State.firstPlayer`, because
first-player advantage is large and unpaired sampling wastes games on that
variance. `score = (wins + 0.5·draws)/n`. Measured with the **greedy policy network
only** — no MCTS, no sampling — which is the deployment-time procedure by decision
(§8.3), so the gate evaluates exactly the artifact that ships.

Reports the Wilson 95% lower bound alongside the point estimate: at n = 400, p̂ =
0.70 has a lower bound of only 0.653.

The IL-regression check reads `data/il_baselines.json` (§4.4) and **refuses to run**
if the recorded `ckpt_sha1` does not match the π_IL checkpoint it was given.

### 5.7 `belief.py` — Bayesian posterior

The opponent's deck is one of six known 𝒟_opp archetypes
(`opp_ids = [0, 2, 1, 3, 5, 9]`), so belief is a 6-way categorical:

```
P(deck_j | observed) ∝ P(deck_j) · Π_{c observed} P(c | deck_j)
```

with `P(deck_j)` the mined frequency. Any observed card absent from `deck_j` zeroes
that hypothesis, so this collapses to near-certainty within a few turns. No
training, negligible cost.

It becomes a third source in `belief_infer.OpponentDeckOracle`'s existing
confidence-threshold chain:

1. learned `arch` head when confident — a `representative` is a deck someone
   actually played, so its energy counts, ratios and evolution lines are
   internally consistent;
2. **Bayesian posterior** when it has collapsed to a single hypothesis;
3. learned `deck` card distribution, rounded into 60 cards.

`ptcg_il/model/belief.py` (learned history encoder feeding the trunk) and
`ptcg_rl/belief.py` (this posterior) are different things. Keep both; do not
conflate them.

---

## 6. Checkpoint format

Extend, never replace, the `"deck"` record from `ptcg_il/deck.py` — the artifact
pinning is as load-bearing under RL, and RL checkpoints must stay loadable by
`scripts/build_submission.py`.

```python
ckpt["rl"] = {
    "generation": 1,
    "parent": "checkpoints_a2/ckpt-best.pt",
    "phase": "R2",
    "gate": {
        "opponent": "checkpoints_a2/ckpt-best.pt",
        "n_paired": 400,
        "score": 0.6325,
        "wilson_lb": 0.584,
        "il_nontrivial_top1": 0.7401,
        "il_baseline": 0.7538,
        "kl_to_il": 0.031,
        "passed": True,
    },
    "kappa": 0.02,
    "steps": 50_000,
}
```

`ckpt["config"]` carries the architecture (§4.3).

---

## 7. Testing

| area | test | why |
|---|---|---|
| `recompute_logp` | reproduces `multiselect_ce` on real IL shard data, including STOP and `minCount == 0` rows | joint-vs-per-step log-probs is high-severity and silent |
| ratio canary | epoch-0 p99 `\|ratio − 1\| < 1e-3` on a real update | precision-path drift, plus incidental masking bugs |
| masking | `sample_action` never proposes an illegal or repeated pick over 1,000 games | one masking implementation, asserted |
| GAE | matches a hand-computed reference on a short synthetic trajectory | fp32 accumulation, `γ = 1`, `λ = 0.95` |
| adaptive β | drives measured KL to κ on a synthetic divergence | β is the knob most likely to be blamed for a failure |
| Wilson bound | matches published values at n ∈ {100, 200, 400, 1000}, p̂ = 0.70 | the gate's anti-noise floor |
| `gate.py` sha pin | refuses a baseline whose `ckpt_sha1` mismatches | stale-baseline risk |
| `belief.py` | a card absent from deck *j* zeroes hypothesis *j*; posterior sums to 1 | zeroing is the whole mechanism |
| `search_plan` JSON | visits sum to the iteration count; `indices` unchanged from current behaviour | keeps the live-eval baseline honest |
| pipeline | stage 5 skipped by `--no-rl`; failure exits non-zero | stage 5 is not an eval-style soft failure |

Tests that examine a property only when a fixture happens to contain the relevant
case pass vacuously. Each test that counts occurrences must **fail on zero
examined**, and the suite should be validated by mutation — disabling a fix must
turn its test red. This is how the R0b belief/coverage tests were validated and it
caught two convincing-looking wrong answers.

---

## 8. Risks

| risk | severity | mitigation |
|---|---|---|
| Dead value head | **blocking** | stage 5a; 5b does not start until its gate passes |
| Per-step instead of joint-sequence log-probs | high, silent | §5.1, one masking module, tested against `multiselect_ce` |
| bf16 log-probs across mismatched precision paths | high, silent | §5.5, recompute in-precision, epoch-0 canary |
| Stale IL baselines | high, silent | §4.4 sha-pinned artifact; gate refuses on mismatch |
| Belief labels silently absent from shards | medium, silent | §4.2, labels unconditional at build time |
| Checkpoint lacks arch config | medium | §4.3; stage 5 loads two policies and cannot infer shapes |
| Timeout losses select for slow policies | low | §5.2, timeouts recorded as losses |
| KL anchor caps achievable strength | medium | κ fixed through R2 by decision; the gate arbitrates |
| Reward hacking | low | reward *is* the objective |

---

## 9. Success criteria

1. `./scripts/run_pipeline.sh --skip-download` runs all five stages to completion.
2. Stage 4c writes `il_baselines.json` with arch 0 and arch 2 non-trivial top-1 at
   or above 0.5394 / 0.7538. These pre-fix numbers are **not** a like-for-like
   comparison — they measure the old features (§4.2) — but they are a usable
   sanity floor, and the `opt_card_id` fix should *improve* IL, since card identity
   is the dominant signal. Treat falling below the floor as evidence the fix broke
   option ordering; treat a jump far beyond expectation as evidence it leaked
   hidden information. Once recorded, the new numbers replace the old ones
   everywhere, including `RL_SPEC.md` §10.2 condition 3.
3. Stage 5a clears `corr(V, outcome) ≥ 0.35`, prediction std ≥ 0.3.
4. Stage 5b reaches ≥60% against frozen π_IL over 400 paired games, with
   IL-regression under 5 points and a green ratio canary.
5. `uv run pytest tests/ python/tests/` passes, including the new
   `python/tests/test_rl_*.py`.
