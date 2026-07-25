# RL_SPEC.md — Self-Play RL on top of the IL policy

**Status:** revision 4. Open questions are **decided** (§14). **R0a, R0b and R0d are
done and measured** (§6.4.3, §6.4.4); current phase is **R0c** — rebuild shards, retrain
both specialists, re-record IL baselines.

**Reading order:** `AGENT_SPEC.md` (engine I/O) → `TRANSFORMER_IL_SPEC.md` (model, tensors) →
this document. Where this document and `TRANSFORMER_IL_SPEC.md` disagree about
tensor layout or the `agent()` contract, **TRANSFORMER_IL_SPEC.md wins** — RL changes
how the policy is *trained*, not what it consumes or emits.

**Revision 2 changes:** all performance claims are now measured, not estimated (§2).
The compute architecture (§6) is rewritten around the measurement that **featurization,
not the engine, is 77% of rollout cost** — which relocates the Rust/rayon work.
Adds §7 on floating-point discipline, because bf16 breaks PPO ratios in a specific
and non-obvious way.

**Revision 3 changes:** the six open questions are decided and folded into the body
(§14 records them). The phase plan (§11) is re-ordered accordingly: **fix
`opt_card_id`, then vectorise the NumPy featurizer, then port to Rust.** §6.4 gains the
root-cause analysis of the `opt_card_id` bug, including the finding that **~20% of the
PAD entries are genuinely hidden information that must stay PAD** — filling them in
would leak. §6.6 records the consequence that fixing features **invalidates the current
IL checkpoints** and forces an IL retrain before R1.

Guiding principle for this revision, per the project decision: **stability and isolated
debugging over throughput.** Where a simpler-but-slower option makes a failure easier to
attribute, take it.

---

## 1. Goal and scope

Take the per-deck IL specialists (archetype 0 and archetype 2) and improve them by
self-play and cross-play, using PPO with an MCTS-derived improvement signal, while
anchoring to the IL policy with a KL penalty so learned behaviour cannot collapse.
Promote a new checkpoint only when it clears a **≥70% score against the current
champion** and passes an IL-regression check.

**In scope:** rollout infrastructure (Rust/rayon), critic repair, PPO, KL anchoring,
determinized MCTS, a two-deck league, and the promotion gate.

**Out of scope:** deck construction (the deck stays fixed and label-pinned per
`ptcg_il/deck.py`), changes to the `agent()` contract, changes to *feature semantics*
(a Rust featurizer must reproduce the NumPy one bit-for-bit — §6.4).

---

## 2. Measured performance baseline

Everything below was measured on this machine (16 cores, RTX 5070 Ti 16.6 GB,
Blackwell sm 12.0, native bf16, torch 2.11+cu128) against the real `libcg.so` and the
real `checkpoints_a2/ckpt-best.pt`. **These numbers replace the estimates in
revision 1 and are the basis for §6.**

### 2.1 Concurrency is safe

8 threads × 3 games, each thread assigned a **distinct deck pair**, asserting that no
battle ever observes a card id outside its own two decks:

- **0 foreign card ids, 0 errors, 0 crashes** across 24 games.
- All 24 games terminated with a winner (`result ∈ {0,1}`).

The Python `class Battle` holds `battle_ptr` as a **class attribute** — a per-process
singleton — but that is a limitation of the Python wrapper only. The C API takes the
pointer explicitly (`GetBattleData(ptr)`, `Select(ptr, …)`, `BattleFinish(ptr)`), and
`BattleStart` *returns* a fresh handle. **Concurrent battles in one process are
independent.** `GameInitialize()` is the one global, called once at import in `sim.py`;
in Rust it must be guarded by `std::sync::Once`.

### 2.2 Where rollout time actually goes

Single-threaded, 40 games, 9,560 decisions:

| stage | µs/decision | share of wall |
|---|---|---|
| engine (`GetBattleData` + `Select`, GIL released) | 35 | 12.9% |
| `json.loads` (Python, holds GIL) | 21 | 7.7% |
| **`featurize()` (NumPy, holds GIL)** | **208** | **77.2%** |
| other | — | 2.2% |

- **3,705 decisions/sec** end-to-end single-threaded.
- **GIL-bound fraction 85% → Amdahl ceiling of 1.18× for threaded Python.** The 1.22×
  measured in §2.1 matches this almost exactly, confirming the poor scaling was the
  GIL and the featurizer, **not** a lock inside libcg.
- Observation JSON is ~4.4 KB mean / 6.5 KB p95; `search_begin_input` ~1.4 KB.

**The single most important consequence:** moving *only the environment* to Rust
reclaims at most ~13% of the loop. The engine is already fast. **The featurizer is
6× more expensive than the engine**, and that is where the Rust work has to land.

### 2.3 GPU inference

Real arch-2 policy (`V=262, A=181, D=256, layers=1, ff=1024, heads=8` — inferred from
state-dict shapes since the checkpoint carries no config; the belief module's 36
tensors were excluded, so a full model is somewhat slower):

| batch | fp32 | bf16 | bf16 speedup |
|---|---|---|---|
| 64 | 27.1k dec/s | 28.1k dec/s | 1.04× |
| 256 | 41.5k | **57.1k** | 1.38× |
| 512 | 39.6k | **61.6k** | 1.56× |
| 1024 | 38.1k | 61.8k | 1.62× |
| 2048 | 39.2k | 61.5k | 1.57× |

- **bf16 gives ~1.57×** at usable batch sizes. Confirmed worthwhile.
- **Throughput saturates at batch 256–512.** Beyond that, larger batches buy nothing
  and only add latency (8.3 ms at 512 → 33.3 ms at 2048). **Use 256–512**, not the
  16,384 figure from revision 1 (which was a *rollout-buffer* size, not a
  forward-pass batch size — keep the two distinct).
- At batch 512 bf16, inference is **16.2 µs/decision** — *cheaper than the engine call
  itself* and 13× cheaper than NumPy featurization.

**The GPU is not, and was never, the bottleneck.** At 61.6k dec/s it can serve roughly
four Rust-featurizing CPU workers (§6.5). This is the crucial planning fact: rayon
parallelism pays for **MCTS**, not for base rollout.

Because the model is tiny with fixed shapes, 16.2 µs/dec is largely kernel-launch
overhead. `torch.compile(mode="reduce-overhead")` or CUDA graphs is a plausible further
~2× and should be tried before adding workers — it is much cheaper than either.

---

## 3. Prerequisite: the value head is dead

**This blocks everything else and must be fixed first.**

`ValueHead` is `tanh(Linear(D,D) → GELU → Linear(D,1))` on `h_CLS`, so its range
`(−1, 1)` already matches a ±1 outcome. But measured on the arch-2 checkpoint:

| statistic | value | healthy |
|---|---|---|
| prediction std | 0.0006 | ~0.5–0.9 |
| target std | 0.994 | — |
| corr(pred, target) | 0.045 | > 0.4 |
| sign agreement | 0.556 | > 0.65 |
| MSE | 0.9988 | < 0.7 |

The head has collapsed to constant ~0 — the MSE-minimising answer when the input
carries no usable signal. PPO's advantage is `A_t = Σ (γλ)^k δ_{t+k}` with
`δ_t = r_t + γV(s_{t+1}) − V(s_t)`. With `V ≡ 0`, GAE degenerates to the raw terminal
return for *every* state, i.e. plain REINFORCE with no variance reduction over
~150-decision episodes. That will not learn at usable sample cost.

> Note the §2.3 probe measured value-head std **0.222** on shard data — non-zero
> because that run used freshly-inferred architecture with 36 tensors unloaded, so it
> is not the trained head. It does not contradict the collapse above; re-measure on a
> correctly loaded model as the first step of R1.

**Phase R1 is a critic-only repair.** Candidate causes, in order:

1. **`h_CLS` carries no outcome information.** IL trains CLS *only* through the value
   loss (the pointer head reads option/state tokens), so `λ_v = 0.5` against a dominant
   CE may never shape it. *Test:* freeze the trunk, train only `ValueHead`. If it still
   collapses, the trunk is the problem, not the head.
2. **tanh saturation / dead gradient at init.** *Test:* inspect pre-activation std.
3. **Label noise.** Every decision in a game shares one ±1 label, so early-game states
   are near-unpredictable by construction. Judge against a per-turn baseline, not
   against zero loss.

**Exit criterion:** `corr(pred, outcome) ≥ 0.35`, prediction std ≥ 0.3, and accuracy
rising monotonically in turn number.

If the IL corpus is too small (37,421 decision points), fit the critic on self-play
data from the frozen IL policy — cheap, unlimited, and on-distribution for RL.

---

## 4. Environment contract

### 4.1 Reward and terminal

`Observation.current.result` is the **winning player index, or −1 if unfinished**. This
is the only reward source.

```
r_t = 0                                for all non-terminal t
r_T = +1  if result == our index
      −1  if result == opponent index
       0  if the game ends with no winner
```

- `γ = 1.0` (finite episodes, no reason to discount a win); `λ_GAE = 0.95`.
- **Draws may not exist.** All 24 probe games produced a winner. Keep the draw branch
  as defensive code, but do not build logic that depends on draws being common.
- **Optional shaping only if potential-based:** `F(s,s') = γΦ(s') − Φ(s)` with
  `Φ = (opp_prizes_taken − our_prizes_taken)/6`. Potential-based shaping provably
  preserves the optimal policy; anything else does not and must not be used.

### 4.2 Timeouts count as losses

`Observation.remainingOverageTime` is real. A worker exceeding budget must record a
**loss**, not discard the game — otherwise training silently selects for slow
policies. MCTS iteration counts must be budgeted against this.

### 4.3 The engine loop

```
BattleStart(int[120] = deck0 ++ deck1) -> StartData { battlePtr, errorPlayer, errorType }
    success  <=>  battlePtr != 0        # errorPlayer/errorType are diagnostic only;
                                        # errorPlayer == -1 means "no player at fault"
GetBattleData(ptr) -> SerialData { json, data, count, selectPlayer }
    json          : observation JSON (~4.4 KB mean)
    data[count]   : ASCII blob -> obs["search_begin_input"], the MCTS seed
    selectPlayer  : whose decision this is
Select(ptr, int* picks, len) -> err     # err 30 = broken ptr, else index error
BattleFinish(ptr)
```

`search_begin_input` must be carried through to MCTS (§8) — it is *not* in the JSON,
`game.py` splices it in from the binary blob.

---

## 5. The action is a *sequence*, not a token

The subtlest correctness requirement here, and the easiest to get silently wrong.

A decision point has `minCount ≤ k ≤ maxCount` picks over ≤ `O_MAX = 64` options,
handled autoregressively with a STOP column (`multiselect_ce` teacher-forced at train
time, `select_multi` greedy at inference). For RL, one **action** is
`a = (a_1, …, a_k, STOP)`, therefore:

```
log π_θ(a | s) = Σ_{t=1..k+1} log softmax( mask( logits_t ) )[a_t]
```

- The sum **includes the STOP step** whenever a STOP column exists. Dropping it makes
  short selections systematically cheaper and biases the policy toward tiny picks.
- Rows that declined an optional select (`minCount == 0`, no STOP column — 98 such rows
  in the IL corpus) contribute an empty sum. Handle explicitly; never let it become
  `-inf`.
- The PPO ratio is on the **joint** sequence:
  `exp(log π_θ(a|s) − log π_θ_old(a|s))`. Per-step ratios optimise a different
  objective.
- Entropy is likewise summed over AR steps, masked to legal options.
- Already-selected options are masked at later steps, as `_select_multi_raw` does.
  **One shared masking function must serve rollout, the PPO update, and MCTS.** Three
  copies will drift, and the drift is invisible until the policy proposes illegal picks.

`ptcg_rl/actor.py` owns this:

```python
def sample_action(policy, batch, *, temperature=1.0, generator=None)
    -> tuple[list[int], Tensor, Tensor]:      # picks, logp_joint, entropy_sum

def recompute_logp(policy, batch, actions) -> tuple[Tensor, Tensor]:
    """Teacher-force stored actions under current θ."""
```

`recompute_logp` is a thin generalisation of `multiselect_ce` — reuse it rather than
writing a parallel implementation. **Gate it with a test** asserting it reproduces
`multiselect_ce` on IL shard data.

One convenient consequence of the AR design: because the multi-select GRU state lives
*in the model*, all `k+1` AR steps happen GPU-side with **no environment round-trip**.
Each decision point costs exactly one obs→batch transfer.

---

## 6. Compute architecture: Rust + rayon, batched bf16 inference

### 6.1 What the measurements dictate

§2.2 says the engine is 13% and featurization 77% of rollout. So:

- Porting **only** the env to Rust caps out at ~1.15× — not worth it alone.
- Porting **env + featurizer** to Rust and escaping the GIL is the real win.
- If Rust envs hand JSON back to Python to featurize, you keep the 77% bottleneck
  *and* add serialization. **That design is worse than today's.** The featurizer must
  cross into Rust or the exercise fails.

**DECIDED: vectorise the NumPy featurizer first, then port to Rust.** 208 µs to fill a
fixed 46-token tensor is slow for NumPy and suggests per-option Python loops.
Vectorising to ~50 µs is far less work than a Rust port, takes single-thread throughput
from 3.7k to ~10k dec/s on its own, and — the actual reason — produces a **mathematically
aligned reference semantics** to port against. Writing Rust against the current
un-vectorised code would mean porting an implementation we have not yet pinned down.

**DECIDED: fix `opt_card_id` before any Rust is written.** Known bugs are not ported
across languages. See §6.4.

### 6.2 Boundary

| side | owns |
|---|---|
| **Rust** | `libcg.so` handles, game loop, JSON→features, rayon pool, MCTS tree |
| **Python** | PyTorch model, PPO update, league/gating, orchestration |

Bind with **PyO3 + `rust-numpy`**, not the existing C-ABI + ctypes. PyO3 returns
NumPy arrays that Python wraps **zero-copy**, and `Python::allow_threads` releases the
GIL around the rayon region — both essential. The existing `ptcg_search` C-ABI stays
for the standalone search-planner baseline used by `live_eval.py`.

### 6.3 `VecEnv`

```rust
pub struct VecEnv { envs: Vec<Env>, /* one battle_ptr each */ }

impl VecEnv {
    /// Advance every env by one decision, auto-resetting finished games.
    /// Returns ONE contiguous, already-batched feature block.
    fn step(&mut self, picks: &[Vec<i32>]) -> BatchedFeatures;
}
```

- `GameInitialize` behind `std::sync::Once`; `dlopen` of one path is refcounted, so all
  workers share a single libcg image — per-§2.1 that is safe.
- `Env` currently holds a raw `*mut c_void`, making it `!Send`, so rayon is
  **compile-time blocked** until an explicit `unsafe impl Send for Env {}` is added.
  That is the right place to document the §2.1 evidence justifying it.
- `par_iter_mut()` over envs inside `py.allow_threads(|| …)`.
- Write features **directly into one pre-allocated batch buffer** at the worker's slot
  index — no per-env allocation, no concatenation. Buffers are double-buffered and
  page-locked so the H2D copy overlaps the next step's CPU work.
- Auto-reset on terminal keeps the batch full; return terminal rewards out-of-band so
  the learner can close out trajectories.

### 6.4 Featurizer parity is a hard gate

Two featurizer implementations that disagree will train RL on different features than
IL was fitted on, **silently invalidating both the IL prior and the KL anchor**.

**Required test, blocking for R0:** for ≥50,000 real observations spanning every
`SelectType`/context, assert the Rust featurizer's every output array is **exactly
equal** to `ptcg_il.featurizer.featurize` — bitwise for integer/bool arrays, and
`atol=0` for float arrays (both should be building identical values, not merely close
ones). Any mismatch fails the build. Run it in CI, not once by hand.

Known traps this must cover:
- `minCount`/`maxCount` are emitted as **`np.int64` scalars**, not arrays
  (`featurizer.py:664, 884`) — the exact discrepancy that caused ~9% agent forfeits in
  the submission bug.
- `stop_column` semantics and the `minCount == 0` no-STOP case.
- Dtypes exactly: `opt_mask` bool, ids int64, feats float32.
- Card ids live under `card["id"]` in state containers, but options use `cardId`.
  Two different keys for the same concept.

### 6.4.1 The `opt_card_id` bug: root cause (measured)

Measured over 80 real episodes / 208,940 options, reading **only**
`$.steps[][].observation` (see the warning below):

- Only **0.01%** of options carry a `cardId` field at all. The current code path
  (`opt.get("cardId")`, handled for option types 0–6 and 14–16) is therefore dead by
  construction. The bug is not "types are unhandled" — it is that **card identity must be
  dereferenced from a location**, which `build_ref_map` was documented to do and never did.
  Its docstring promises a `_card_ids` sub-dict "for … non-tokenized zones (e.g. DECK,
  DISCARD)"; the function returns only `{(area, playerIndex, index): token_row}`.
- `opt_src_idx` / `opt_tgt_idx` resolve correctly. Only `opt_card_id` is affected.

**AreaType values, identified empirically** (`ref_map.py` names only 2/4/5/7):

| area | zone | resolvable? |
|---|---|---|
| 1 | deck | **no** — `deckCount` is a count; there is no card list |
| 2 | hand | yes |
| 3 | discard | yes (public) |
| 4 | active | yes |
| 5 | bench | yes |
| 6 | prize | **only when face-up** — `prize: [Card|None]` |
| 7 | stadium | yes |
| 12 | `looking` (reveal buffer) | yes |

**The 99.6% PAD decomposes into three different things, and only the first is a bug:**

| | share of options | verdict |
|---|---|---|
| resolvable today, currently PAD | **58.9%** | **the bug — fix it** |
| carries no card by nature (End 9.5%, Attack 4.7%, Retreat 3.8%, Yes/No/Number ~1%) | ~19% | correct as PAD |
| genuinely hidden (deck slots, face-down prizes) | ~20% | **must stay PAD** |

> **Do not "fix" the third group.** Writing a card id for a deck slot or a face-down
> prize would hand the policy information the real agent cannot see — an information
> leak that inflates offline metrics and collapses at live-eval. The target is ~59%
> populated (plus discard / `looking` / face-up prizes), **not** 100%.

Type-level fix, all verified resolvable at 100% on real data:

| type | share | how to resolve |
|---|---|---|
| 8 Attach | 22.6% | bare `index` → actor's hand; target already in `opt_tgt_idx` |
| 7 Play | 20.9% | bare `index` → actor's hand |
| 3 Card | 30.5% | `(area, playerIndex, index)`; 28.5% via hand, rest per the area table |
| 10 Ability | 3.4% | `(area, index)` |
| 9 Evolve | 3.1% | bare `index` → actor's hand |
| 6 Energy / 4 Tool / 15 Skill | 0.2% | `(area, index)` |

Note types 7/8/9 carry a **bare `index` with no `area`/`playerIndex`** — implicitly the
acting player's hand. That is precisely why `ref_map` lookups miss them.

> **Trap for anyone re-running these numbers:** episode JSON contains a `visualize`
> block that duplicates each observation with enum names as **strings** (`"Attach"`)
> rather than ints (`8`). A naive recursive search for `select.option` picks up both and
> appears to show two schema versions. There is only one: real observations live at
> `$.steps[][].observation`. Read from there.

### 6.4.2 Consequence: the current IL checkpoints become invalid

Fixing `opt_card_id` changes the model's **input distribution**. Therefore:

- `checkpoints_a0/ckpt-best.pt` and `checkpoints_a2/ckpt-best.pt` are **no longer usable**
  — not as `π_IL` anchors, not as gate baselines, not as league opponents. Their weights
  were fitted with option tokens carrying no card identity.
- The IL baselines this spec quotes (arch 0 non-trivial top-1 **0.5394**, arch 2
  **0.7538**) are measurements of the *old* features and cannot be compared against
  post-fix numbers.
- **Shards must be rebuilt** (`data/shards/*.npz`) and **both specialists retrained**
  before R1 begins. This is unavoidable and is the price of decision 5; it is the right
  price, since every later phase would otherwise inherit the bug.

**Re-measure and re-record the IL baselines after the retrain, then update §10.2
condition 3.** A regression gate against stale numbers is worse than no gate.

Expected direction: the fix should *improve* IL, since card identity is the dominant
signal (it is why per-deck specialists beat the generalist). Treat any *regression* as a
sign the fix leaked hidden information (§6.4.1) or broke option ordering.

### 6.4.3 R0b result: the fix, and why 40% of options are still PAD

Measured over 23,249 observations / 129,278 legal options after the fix:

| metric | before | after |
|---|---|---|
| `opt_card_id` populated | 0.00% | **59.85%** |
| resolving to `UNKNOWN_CARD` | — | **0.00%** |
| deck / face-down-prize refs populated (**leak**) | — | **0** |

The 40.15% still at PAD decomposes into exactly three buckets, with **nothing left over**:

| share | count | reason | correct? |
|---|---|---|---|
| 48.1% | 24,948 | option carries no location reference at all — types 0, 1, 2, 12 (RETREAT), 13 (ATTACK), 14 | ✅ nothing to resolve |
| 37.9% | 19,662 | area 6 prize slot is `None` — face-down | ✅ **must** stay PAD |
| 14.1% | 7,295 | area 1 = deck | ✅ **must** stay PAD |

Per-type coverage is now 100% for every type that references a visible card (4, 6, 7, 8,
9, 10, 15) and 0% for every type that references nothing (0, 1, 2, 12, 13, 14). The one
partial is type 3 (CARD) at 31.6% — the remainder is entirely the deck and face-down-prize
buckets above, i.e. correct.

The residual-bug bucket — *area is a resolvable zone, index in range, card present, yet
`opt_card_id` stayed PAD* — is **empty**. That is the check that makes the coverage number
meaningful; a raw percentage cannot distinguish a fixed bug from a half-fixed one.

Two traps worth recording, both of which produced convincing-looking wrong answers first:

- **`normalize_vocab` is not optional.** `vocab.json` keys are JSON *strings*; engine card
  ids are *ints*. Calling `featurize` with the raw parsed JSON makes every lookup miss, so
  every card silently becomes `UNKNOWN_CARD` — coverage reads 59.85% while carrying no
  information whatsoever. The pipeline does call it (`shard_writer._load_vocab`); ad-hoc
  probes must too. **Any coverage claim must also assert `UNKNOWN_CARD` is ~0%.**
- **Conditional assertions pass vacuously.** Tests that check a property only when a
  fixture happens to contain the relevant option type will go green on an empty loop. Each
  of the three new leak/coverage tests therefore counts what it examined and fails on zero.
  The suite was validated by *mutation*: disabling the fix must turn tests red. Two do.

### 6.4.4 R0d result: the featurizer was never loop-bound

**216.2 → 60.5 µs/decision (3.57×), bitwise identical over 50,000 obs × 36 keys
(101.7M scalar elements).** Target was ≤70 µs.

The premise behind "vectorise the featurizer" was wrong in an instructive way. Profiling
showed the cost was not loop structure, Python-level iteration, or per-token work. It was
**one three-line helper**:

```python
def _clip_norm(value, norm):
    return float(np.clip(value / norm, -1.0, 1.0))   # scalar!
```

Called **125× per decision** (500,095 times per 4,000 obs), it accounted for **69% of all
featurize time** (1.288 s of 1.868 s). Almost none of that is arithmetic — `np.clip` on a
*Python scalar* still pays the full array-dispatch path (`_wrapit` → `_wrapfunc` → `_clip`,
plus an `_array_converter` round trip), ~2.6 µs a call. Replacing it with two branches on a
Python float removed 156 µs/decision and made the code *shorter*.

Lessons that carry into R0e and beyond:

- **Profile before restructuring.** The intuitive fix — rewriting the token loops into
  batched NumPy — would have been a large, semantics-endangering change aimed at the 17%,
  while leaving the 69% untouched.
- **NumPy on scalars is a pessimisation.** Dispatch overhead dwarfs the operation. Anywhere
  a hot path touches `np.<func>(python_scalar)`, plain arithmetic wins by ~20×.
- **After the fix the profile is flat** — the largest remaining entry is 17%
  (`_build_poke_tokens`), and `_clip_norm`'s residue is pure call overhead. There is no
  second big win here, so **stop**: further micro-optimisation would trade the clear
  reference semantics that R0d exists to provide (§6.1) for single-digit percentages.
- Corollary for §6.5: the Rust featurizer's advantage over vectorised NumPy is now 2×,
  not 7×. See §6.5 — R0e is downgraded to optional.

Verification harness (keep it; R0e needs the same gate to prove the Rust port matches):
a frozen `golden_obs.pkl` / `golden_out.npz` pair, compared **bitwise on raw buffers**
rather than with `allclose` — a float tolerance would hide exactly the drift that matters
when two implementations must agree. `_clip_norm`'s replacement was additionally checked
exhaustively over 228 value/norm combinations including NaN, ±inf, −0.0, subnormals, and
mixed float32/float64 inputs.

### 6.5 Throughput projection and the real bottleneck

**Updated after R0d.** The featurizer is now **60.5 µs**, not 208 µs (§6.4.4), which
changes the conclusion of this section. Two scenarios, both measured on the CPU side:

```
today  (vectorised NumPy):  engine 35 µs + featurize 60.5 µs ≈  95 µs/dec/core → ~10.5k dec/s/core
target (Rust featurizer):   engine 35 µs + featurize 30   µs ≈  65 µs/dec/core → ~15.4k dec/s/core
```

| workers | CPU supply (NumPy) | CPU supply (Rust) | GPU supply (bs 512 bf16) | binding constraint |
|---|---|---|---|---|
| 1 | 10.5k | 15k | 61.6k | CPU |
| 4 | 42k | 62k | 61.6k | CPU (NumPy) / **balanced** (Rust) |
| 6 | 63k | 92k | 61.6k | **GPU** (both) |
| 14 | 147k | 215k | 61.6k | **GPU** |

**The Rust featurizer port is no longer required for base rollout.** It buys 1.47× on the
CPU side, but six cores of vectorised NumPy already reach the 61.6k dec/s the GPU can
absorb — and the machine has 16. R0e is therefore **downgraded from prerequisite to
optional**, justified only if MCTS's extra work turns out to exhaust the core budget.
This is the second time measurement has moved this target (§6.1); do not port on
principle. Re-measure at R3, when the MCTS load is real, and decide then.

**Base rollout saturates the GPU at ~6 workers.** Spending 14 cores on plain rollout is
wasted. What consumes the remainder is **MCTS**: at `ρ = 0.05` of states × 128 iterations,
search adds ~6.4× the engine work per decision plus its own network evaluations. Size the
rayon pool from the MCTS budget, and *measure* rather than assuming.

Practical consequences:
- Forward-pass batch **256–512**, decoupled from the PPO rollout-buffer size (16,384).
- Try `torch.compile`/CUDA graphs before adding workers (§2.3) — cheaper than either port.
- Pin workers to cores; leave 2 free for the Python learner and CUDA driver threads.

---

## 7. Floating-point discipline (bf16 breaks PPO ratios)

bf16 for rollout inference is endorsed — 1.57×, and top-1 action agreement with fp32 is
**98.7%**, so behaviour is essentially preserved. But it interacts badly with PPO in a
way that is easy to ship by accident.

**Measured**, comparing log-probs from the *same weights* under bf16 vs fp32:

| path | mean \|Δlogp\| | p99 | max | ratio range |
|---|---|---|---|---|
| bf16 net + bf16 `log_softmax` | 0.02151 | 0.691 | 2.078 | 0.125 – 1.007 |
| bf16 net + **fp32** `log_softmax` | 0.02105 | 0.693 | 2.079 | 0.125 – 1.000 |

Two things to read off this:

1. **Upcasting the reduction does not help** (0.02105 vs 0.02151). The error is born in
   the network's bf16 matmuls, not in `log_softmax`. Revision 1's proposed fix was
   insufficient.
2. The tail is concentrated in **ties**: p99 ≈ 0.693 = ln 2 and max ≈ 2.079 = ln 8, with
   ratio floor exactly 0.125 = 1/8. Those are uniform distributions over 2 and 8 tied
   options — bf16 rounds near-equal logits to *exactly* equal, fp32 does not. Benign for
   argmax, not benign for log-probs.

At p99, a mixed-precision comparison consumes **500% of the ε = 0.2 clip range**. The
clipping would be driven by rounding rather than by policy change.

**Rules:**

- **Never compare log-probs across precisions.** `logp_old` and `logp_new` must come
  from the same dtype *and* the same kernel path. bf16 is self-consistent; *mixing* is
  the bug.
- Prefer **recomputing `logp_old` at the start of the first PPO epoch** in the update's
  precision, rather than storing the rollout value. Costs one extra forward over the
  buffer and removes the whole class of error.
- **Assert it:** at epoch 0 of every PPO update, `ratio` must equal 1 up to
  `p99 |ratio − 1| < 1e-3`. This is a cheap, decisive canary for precision-path drift,
  and it will also catch unrelated batching/masking bugs. Fail loudly.
- Keep **advantage/GAE accumulation, the KL term, and the loss reduction in fp32.**
  Long summations in bf16 lose precision quickly.
- Still cast logits to fp32 before `log_softmax` — standard hygiene, marginally helps
  ties — but understand it is not the fix.
- The **value head is safe in bf16**: max |Δv| 0.004 with std preserved (0.2221 vs
  0.2221).
- Master weights and the optimizer stay fp32 (autocast only), exactly as IL training
  already does.

---

## 8. MCTS: determinized search, for targets not for acting

### 8.1 This is an imperfect-information game

`search_begin` requires the caller to **supply predicted hidden information** (opponent
deck, prizes, hand, active). There is no perfect-information simulator. So MCTS here
means **determinized search**: sample `K` states consistent with the observation,
search each, aggregate.

**The belief problem is unusually tractable and we should exploit it.** The opponent's
deck is one of six known 𝒟_opp archetypes (`opp_ids = [0, 2, 1, 3, 5, 9]`), so belief
over the opponent deck is a **6-way categorical**:

```
P(deck_j | observed) ∝ P(deck_j) · Π_{c observed} P(c | deck_j)
```

with `P(deck_j)` the mined frequency. Any observed card absent from `deck_j` zeroes that
hypothesis, so this collapses to near-certainty within a few turns.

`ptcg_rl/belief.py` owns this posterior. Note `ptcg_il/model/belief.py` is a *learned
history encoder* feeding the trunk — a different thing. Keep both; don't conflate them.

**Named weakness:** determinized search suffers *strategy fusion* (it may assume
different actions in states it cannot distinguish) and *non-locality*. It will
systematically overvalue lines that depend on knowing hidden state. MCTS values are a
training signal, not ground truth.

### 8.2 Why MCTS must not simply pick the action

The naive design — "MCTS chooses the move, PPO learns from it" — is **mathematically
invalid**. PPO's ratio `π_θ(a|s)/π_θ_old(a|s)` presumes `a ~ π_θ_old`. If MCTS chose
`a`, the behaviour policy is the search-improved `π_MCTS`, the ratio is an importance
weight for nothing, and the update is biased in a way no tuning fixes. Worth stating
because the failure is silent: training runs, loss falls, the policy drifts wrongly.

| | acting | learning | verdict |
|---|---|---|---|
| A. AlphaZero | MCTS | CE to visit counts | sound, drops PPO |
| B. PPO-only | `π_θ` sampled | PPO | sound, cheap, no search signal |
| **C. PPO + search distillation** | `π_θ` sampled | PPO **+** distillation to MCTS targets | **adopted** |

### 8.3 Design C

- **Acting (the PPO stream):** always sample from `π_θ` with masking. Ratios stay valid,
  rollouts stay cheap.
- **Search targets (auxiliary):** on a fraction `ρ = 0.05` of decision points, run
  determinized MCTS for an improved distribution `π̃` (visit counts aggregated over `K`
  determinizations) and value `Ṽ`. Add:
  ```
  L_search = −c_π · Σ_o π̃(o) log π_θ(o|s)  +  c_ṽ · (V_θ(s) − Ṽ)²
  ```
  with `c_π = 0.3`, `c_ṽ = 0.3`. This is off-policy *supervised* distillation, so it
  composes with PPO without touching the on-policy ratio.
- **DECIDED — MCTS is training-only.** It is used strictly for offline target
  distillation. **Deployment inference and the promotion gate use the greedy policy
  network alone**, which guarantees Kaggle time-budget compliance and keeps the gate
  measuring exactly the artifact that ships.

  Consequences, all of them simplifying:
  - The gate needs no search budget, so `n = 400` paired games stays cheap.
  - `search_plan`'s latency is irrelevant to submission; only its training throughput
    matters. It never runs inside `agent()`.
  - No MCTS code enters `scripts/build_submission.py`, and the packed `main.py` keeps
    using `select_multi` unchanged.
  - The distillation target may be computed **offline in batch** from stored rollout
    states, decoupled from the acting loop entirely. Search can therefore be slow
    without throttling rollout — run it as a separate pass over the buffer.
  - Determinized search's strategy-fusion bias (§8.1) is now confined to a *training
    signal* and can never mislead the deployed agent directly.

`ρ = 0.05` because search costs ~100× a forward pass. Budget it against R0's measured
throughput; do not pick a number and hope.

### 8.4 Extend the existing Rust searcher

`ptcg_search` already wraps the search API (`Engine::search_begin/search_step/
search_release`, `SearchEnd` on `Drop`) and exposes C-ABI `search_plan`. It currently
returns only chosen indices. For distillation it must also return **visit counts and
the root value**:

```json
{"indices": [...], "visits": [[option, n], ...], "value": 0.31, "error": 0}
```

Reuse it rather than writing a second searcher in Python, and reuse `live_eval.py`'s
`_find_libcg_path()` for library discovery. Note `Engine` already holds a per-instance
`agent_ptr`, so one `Engine` per rayon worker is the natural unit.

---

## 9. PPO objective with an IL anchor

```
L = L_clip  +  c_v · L_value  −  c_H · H(π_θ)  +  β · KL(π_θ ‖ π_IL)  +  L_search
```

### 9.1 Clipped surrogate

Standard, with `ratio` on the joint sequence (§5):

```
L_clip = −E[ min( ratio · Â,  clip(ratio, 1−ε, 1+ε) · Â ) ],   ε = 0.2
```

Normalise `Â` per minibatch over **decision points**, not per episode (episode length
varies ~3×).

### 9.2 Value loss

`L_value = E[(V_θ(s) − V_target)²]`, `V_target = Â + V_old(s)`, with value updates
clipped to ±0.2 around `V_old`. With a freshly repaired critic (§3) this matters more
than usual.

### 9.3 KL anchor — the forgetting guard

`π_IL` is the **frozen** `ckpt-best.pt` for that deck, loaded once, never updated. Per
decision point, summed over AR steps, masked to legal options:

```
KL(π_θ ‖ π_IL) = Σ_t Σ_{o legal} π_θ(o|s,a_<t) · log( π_θ(o|s,a_<t) / π_IL(o|s,a_<t) )
```

- **Direction matters.** `KL(π_θ ‖ π_IL)` (mode-seeking, the RLHF convention) penalises
  θ for mass where IL has none — right for *anchoring*. `KL(π_IL ‖ π_θ)` would force θ
  to cover all of IL's mass including its mistakes, and this IL policy is only
  0.54–0.75 non-trivial top-1.
- **β must be adaptive.** Fixed β is either inert or dominant. Dual update against a
  budget `κ`:
  ```
  β ← clip( β · exp( (KL_measured − κ)/κ · η_β ),  β_min,  β_max )
  ```
  `η_β = 0.1`, `β ∈ [1e-4, 10]`.
- **DECIDED — `κ` is fixed at 0.02 nats/decision through R2.** No curriculum until the
  IL baseline has actually been beaten. Rationale: κ and the learning rate are the two
  knobs most likely to be blamed for a failure, and a moving κ makes R2's go/no-go
  unattributable. β still adapts to *hold* κ — it is the target that stays fixed, not the
  penalty weight. Revisit only in R4+, and record the change in `ckpt["rl"]["kappa"]`.
- Both policies must be evaluated on the **same** legal mask and the **same precision**
  (§7). Assert the supports match rather than assuming.
- Running two networks per update doubles inference cost; `π_IL` needs no grad, so keep
  it in bf16 under `no_grad`.

**State the tension honestly:** the KL anchor is what prevents catastrophic forgetting
and also what caps achievable strength. A tight κ with a weak prior pins the policy
near a weak optimum. κ is the primary knob; §10's gate settles the argument
empirically. Expect to *loosen* κ across generations, with the gate as the safety net.

### 9.4 Hyperparameters (starting points; all need sweeping)

| knob | value | note |
|---|---|---|
| rollout buffer / iter | 16,384 decision points | ~110 games; **not** the forward batch |
| forward batch (rollout) | 256–512 | measured saturation (§2.3) |
| PPO minibatch | 1,024 | |
| epochs / iter | 3 | a KL anchor is already regularising |
| `lr` | 1e-5 → 3e-5 | **far** below IL's 3e-4: fine-tuning, not fitting |
| `ε` | 0.2 | see §7 before trusting it |
| `c_v` | 0.5 | |
| `c_H` | 0.003 | masked-AR entropy exceeds single-token; start low |
| `κ` | 0.02 nats | primary knob |
| `γ`, `λ` | 1.0, 0.95 | |
| grad clip | 1.0 | as IL |
| temperature | 1.0 train / greedy eval | |

---

## 10. The league and the promotion gate

### 10.1 Opponent pool

Each model owns a **fixed, label-pinned deck** (`ptcg_il/deck.py`), so a match is
`deck_A` vs `deck_B`. Both archetype 0 and 2 are in `opp_ids`, so cross-play is
in-distribution for the IL priors — assert this at startup for any deck pair added later.

| opponent | weight | purpose |
|---|---|---|
| current `π_θ`, same deck (mirror) | 0.30 | self-play |
| the other deck's current model | 0.30 | cross-play — the two models fighting |
| uniformly sampled past champion | 0.25 | fictitious self-play; prevents cycling |
| frozen `π_IL` | 0.10 | absolute anchor; regression detector |
| random-legal | 0.05 | sanity floor; must stay ~100% |

The past-champion slot is **not optional**: two models training only against each other
will cycle (A beats B beats A′ beats B′…) and Elo will inflate with no real gain.

**DECIDED — alternating training, not concurrent.** Freeze one agent, train the other,
then swap. Concurrent training is deferred until the pipeline is fully stable (R5).

Rationale: with alternating training the opponent is **stationary within a generation**,
so a policy that stops improving has exactly one explanation instead of three. Concurrent
training makes both sides non-stationary simultaneously, which turns every regression
into a coupled-dynamics question and makes the R2/R4 exit criteria unfalsifiable.

Mechanics:
- A **generation** = one side trains to its next gate attempt while the other is frozen.
- The frozen side's weights are pinned for the whole generation — snapshot them, do not
  read a live handle, or you have reintroduced concurrency by accident.
- Alternate on **gate outcome**, not on a step count: swap when the training side either
  passes (§10.2) or exhausts its step budget.
- **Deck scope is strictly archetypes 0 and 2.** No new decks before R5 (§14.6).

Report **per-pair win-rate matrices**, not just Elo: cross-deck Elo conflates deck
strength with policy strength. Pin each deck's frozen `π_IL` at 0 as the reference.

### 10.2 Gate: a candidate replaces the champion only if all four hold

**Match protocol.**
- Opponent is the **current champion for that deck**, not `π_IL` — gating against IL
  saturates after generation 1.
- **Paired games:** same seed and determinization stream, swapping who goes first
  (`State.firstPlayer`). First-player advantage is large; unpaired sampling wastes games
  on that variance.
- `n = 400` paired games (800 total).
- `score = (wins + 0.5·draws)/n`. Draws appear not to occur (§4.1) but keep the term.
- Measured with the **greedy policy network only** — no MCTS, no sampling. This is the
  deployment-time procedure by decision (§8.3), so the gate evaluates exactly the artifact
  that ships, and gate cost is independent of any search budget.

**Conditions.**

1. **`score ≥ 0.70`** — your bar.
2. **Wilson 95% lower bound ≥ 0.65** — the anti-noise floor. A point estimate of 0.70 is
   compatible with a much weaker policy:

   | n (paired) | p̂ = 0.70 → Wilson 95% LB |
   |---|---|
   | 100 | 0.604 |
   | 200 | 0.633 |
   | **400** | **0.653** |
   | 1000 | 0.671 |

   At `n = 400` you need `p̂ ≈ 0.745` for the *lower bound itself* to clear 0.70. Choose
   deliberately: the default is "point ≥ 0.70 **and** LB ≥ 0.65 at n = 400". For "≥70%
   with 95% confidence" the bar becomes `p̂ ≈ 0.745` at n = 400, or n ≈ 1000 at `p̂ = 0.72`.
3. **IL-regression check — the real forgetting guard.** On the held-out IL **test** split
   for that deck, non-trivial top-1 must not fall more than **5 points** below baseline
   (arch 0: 0.5394; arch 2: 0.7538). A KL penalty bounds *average* divergence; it does
   not protect the rare situations the corpus covers. This does, and the data already
   exists.
4. **`KL(π_θ ‖ π_IL) ≤ 4κ`** over the gate games — catches a policy that met the budget
   in expectation during training but drifted at evaluation temperature.

**On failure:** do **not** discard the candidate; keep training and re-gate. The gate is
a promotion criterion, not a training signal. A candidate passing 1–2 but failing 3 is
the signal to *tighten* κ, not to lower the gate.

### 10.3 Checkpoint format

Extend, never replace, the `"deck"` record from `ptcg_il/deck.py` — the artifact pinning
(`vocab_sha1`, `archetypes_sha1`) is as load-bearing under RL, and RL checkpoints must
stay loadable by `scripts/build_submission.py`.

```python
ckpt["rl"] = {
    "generation": 3,
    "parent": "rl-gen0002-arch2.pt",
    "opponent_pool": [...],
    "gate": {
        "opponent": "rl-gen0002-arch2.pt",
        "n_paired": 400,
        "score": 0.7325,
        "wilson_lb": 0.686,
        "il_nontrivial_top1": 0.7401,   # vs 0.7538 baseline -> -1.4 pts, passes
        "kl_to_il": 0.031,
        "passed": True,
    },
    "elo": 1187.4,
    "kappa": 0.02,
    "steps": 1_200_000,
}
```

Also **write the architecture config into the checkpoint.** The arch-2 checkpoint's
`config` is empty, so §2.3 had to infer `V/A/D/heads/layers/ff` from state-dict shapes.
That is fragile and it is why the packed `main.py` still needs `strict=False`. Fix it
while touching the format.

---

## 11. Phases and exit criteria

Do not start a phase before the previous one passes — RL failures compound and become
unattributable.

**Re-ordered in revision 3.** The Rust/rayon work (formerly R0b) is **halted** until the
featurizer is correct and vectorised. Rationale: a Rust port is a *transcription* task, and
transcribing a known-buggy, un-vectorised reference costs the work twice.

**Revision 4** records the outcome: that halt was the right call for a second reason nobody
predicted. Vectorising first revealed the featurizer's real cost was one scalar `np.clip`
(§6.4.4), so the 3.57× came free — and it moved the Rust port from *prerequisite* to
*optional* (§6.5). Transcribing the original would have carried both a bug and a 156 µs
per-decision pessimisation into a second language.

| phase | work | exit criterion |
|---|---|---|
| **R0a** ✅ | engine concurrency + cost-breakdown probes | **done** — §2. Concurrency safe; featurizer is 77% of rollout |
| **R0b** ✅ | **fix `opt_card_id`** (§6.4.1) | **done** — coverage 0% → **59.85%** (129,278 legal options over 23,249 obs), UNKNOWN_CARD 0.00%; leak check **0** on area-1 and face-down prizes; every remaining PAD accounted for in exactly three benign buckets (§6.4.3); 11 new tests, mutation-verified |
| **R0d** ✅ | **vectorise the NumPy featurizer** | **done** — **216.2 → 60.5 µs/dec (3.57×)**, target was ≤70; bitwise-identical on 50k obs × 36 keys. Root cause was one function, not loop shape (§6.4.4) |
| **R0c** ◀ *current* | rebuild shards, **retrain both specialists**, re-record IL baselines (§6.4.2) | arch 0 and arch 2 non-trivial top-1 ≥ their old values (0.5394 / 0.7538); §10.2 condition 3 updated with the new numbers |
| **R0e** *(optional — see §6.5)* | Rust: `env.rs`, `featurize.rs`, `vecenv.rs`, PyO3 | parity gate (§6.4) exact on ≥50k obs *(reuse R0d's golden corpus)*; ≥30k dec/s aggregate; 1,000 games, **zero** illegal selections. **Do not start until R3 shows MCTS exhausts the core budget** — vectorised NumPy already saturates the GPU at 6 of 16 cores |
| **R1** | critic repair (§3) | `corr(V, outcome) ≥ 0.35`, std ≥ 0.3, accuracy rising in turn number |
| **R2** | PPO + KL anchor (κ fixed 0.02), **one** deck (arch 2), self-play only, no MCTS | ≥60% vs frozen `π_IL` over 400 paired games; IL-regression < 5 pts; epoch-0 ratio canary (§7) green |
| **R3** | MCTS **training-only**: visits+value from the Rust searcher, belief posterior, offline distillation pass | search-distillation ablation beats R2 by ≥3 pts at equal wall-clock |
| **R4** | two-deck league (arch 0 + 2 only), **alternating**, cross-play, Elo, gating | one promoted generation per deck clearing all four §10.2 conditions |
| **R5** | deferred: concurrent training, κ curriculum, additional decks | — |

Order note: **R0d ran before R0c** (revising this document's earlier claim that R0c had to
come first). That claim rested on a mistake: R0d's exit criterion is *bitwise identity with
R0b*, and R0b froze the moment the fix landed and the tests pinned it. Retraining has no
bearing on featurizer semantics, so it was never a prerequisite. Running R0d first is also
strictly cheaper — shards get rebuilt exactly once, by the 3.57×-faster featurizer, instead
of once slowly and then again.

**R2 is the go/no-go.** If PPO with a KL anchor cannot beat frozen IL on one deck with
self-play alone, adding MCTS and a league will not rescue it — it will only make the
failure harder to diagnose.

---

## 12. Module layout

New third package `ptcg_rl/` (PyTorch), plus growth in the existing Rust crate.

| module | role |
|---|---|
| `ptcg_rl/config.py` | `RLConfig` — every knob in §9.4, §8.3, §10 |
| `ptcg_rl/vec_env.py` | thin Python facade over the Rust `VecEnv` |
| `ptcg_rl/actor.py` | masked AR sampling; `sample_action` / `recompute_logp` — **single source of truth for masking** |
| `ptcg_rl/rollout.py` | trajectory assembly, GAE (fp32), advantage normalisation |
| `ptcg_rl/ppo.py` | clipped surrogate, value loss, entropy, adaptive-β KL, ratio canary |
| `ptcg_rl/search.py` | determinized MCTS driver over the Rust searcher; visit/value aggregation |
| `ptcg_rl/belief.py` | 6-way categorical posterior over 𝒟_opp (§8.1) |
| `ptcg_rl/league.py` | opponent pool, matchmaking, Elo, champion archive |
| `ptcg_rl/gate.py` | paired evaluation, Wilson bound, IL-regression check (§10.2) |
| `ptcg_rl/train.py` | CLI: `python -m ptcg_rl.train --deck-archetype 2 --il-ckpt …` |
| `ptcg_search/src/env.rs` | **new** — `BattleStart`/`GetBattleData`/`Select`/`BattleFinish` |
| `ptcg_search/src/featurize.rs` | **new** — port of `ptcg_il/featurizer.py` |
| `ptcg_search/src/vecenv.rs` | **new** — rayon pool, batched output buffers |
| `ptcg_search/src/py.rs` | **new** — PyO3 bindings (`rust-numpy`, `allow_threads`) |

Tests in `python/tests/test_rl_*.py`, matching the existing layout. The engine must be
injectable (as `download.py` injects the Kaggle API) so unit tests need no `libcg.so`.
The featurizer parity test (§6.4) does need it and belongs in CI.

---

## 13. Risks

| risk | severity | mitigation |
|---|---|---|
| **Dead value head** | **blocking** | §3 / R1; do not start R2 without it |
| **Featurizer parity drift between Rust and NumPy** | **high, silent** | §6.4 exact-equality gate on ≥50k obs, in CI |
| **`opt_card_id` fix leaks hidden information** (deck slots, face-down prizes) | **high, silent** | §6.4.1: assert **0%** populated for area 1 and for `prize[i] is None`; an accuracy *jump* beyond expectation is the tell |
| **Stale IL baselines after the feature fix** | **high** | §6.4.2: R0c retrains both specialists and re-records baselines before any gate uses them |
| **bf16 log-probs across mismatched precision paths** | **high, silent** | §7: recompute `logp_old` in-precision; epoch-0 ratio canary |
| **MCTS-acts-and-PPO-learns bias** | **high, silent** | design C (§8.2); act only from `π_θ` |
| Per-step instead of joint-sequence log-probs | high, silent | §5; one masking function; test against `multiselect_ce` |
| Rust port reclaims only 13% (env-only) | high (wasted work) | §6.1: the featurizer is the target, not the engine |
| KL anchor caps strength | medium | κ curriculum, gate as arbiter (§9.3) |
| Determinization bias (strategy fusion) | medium | acknowledged (§8.1); MCTS is a target, not an oracle |
| Two-model league cycles | medium | past-champion slot, weight 0.25 (§10.1) |
| `unsafe impl Send` on a raw engine pointer | medium | justified by §2.1 evidence; re-run the probe in CI |
| Over-provisioning rayon workers | low | §6.5: base rollout saturates the GPU at ~4; size from MCTS |
| Checkpoint lacks arch config | low but recurring | §10.3: write `config`; removes the `strict=False` crutch |
| Timeout losses select for slow policies | low | §4.2: record as losses |
| Reward hacking | low | reward *is* the objective; nothing to hack |

---

## 14. Decisions

All six open questions from revision 2 are settled. The governing principle is
**stability and isolated debugging over throughput.**

| # | question | decision | where |
|---|---|---|---|
| 1 | concurrent or alternating training | **alternating** — freeze one, train the other; concurrent deferred to R5 | §10.1 |
| 2 | does MCTS ship at inference | **no — training-only**, strictly offline target distillation; deployment and the gate use the greedy network alone | §8.3, §10.2 |
| 3 | vectorise NumPy first or go straight to Rust | **vectorise first** — for the aligned reference semantics as much as the 3× | §6.1, R0d |
| 4 | κ curriculum | **fixed at 0.02 through R2**; revisit only after the baseline is beaten | §9.3 |
| 5 | fix `opt_card_id` before or after the Rust port | **before** — known bugs are not ported across languages | §6.4.1, R0b |
| 6 | how many decks | **strictly archetypes 0 and 2**; no new decks before R5 | §10.1 |

### 14.1 Deferred to R5

- **Concurrent training** of both deck models.
- **κ curriculum** (loosening per promoted generation).
- **Additional decks.** Row counts in `data/meta.parquet` (37,421 total):

  | archetype | rows | |
  |---|---|---|
  | 0 | 15,002 | in scope — specialist trained |
  | 2 | 8,481 | in scope — specialist trained |
  | 9 | 5,083 | R5 candidate |
  | 1 | 3,602 | R5 candidate |
  | 5 | 3,189 | R5 candidate |
  | 20 | 2,064 | **zero val rows** — cannot be model-selected |

  A wider league cycles less, but each deck multiplies rollout cost and adds a second
  non-stationary opponent. Not before the two-deck pipeline is stable.

### 14.2 Still genuinely open

1. ~~**Where to stop on `opt_card_id`.**~~ **Resolved by R0b (§6.4.3).** The question is
   moot: discard (area 3), `looking` (area 12), active, bench, hand and *face-up* prizes are
   all resolved, and the remaining 40.15% is provably nothing but options with no reference
   at all, deck slots, and face-down prizes. There is no "long tail" left to trade against
   leak risk — coverage is 100% of everything visible.
2. **Whether `opt_card_id` should encode the *target* as well as the source** for types
   8/9 (attach/evolve). `opt_tgt_idx` already points at the target token, so this may be
   redundant. Ablate in R0c.
3. **Whether the value head's failure (§3) is partly caused by the dead card features.**
   Now testable, and R0c answers it for free: the critic was trained on observations where
   every option's card identity was PAD. If `corr(V, outcome)` improves materially after
   retraining on fixed features, R1's scope shrinks; if it stays at 0.045, the cause is
   architectural and R1 proceeds as written. Read this off R0c before starting R1.
