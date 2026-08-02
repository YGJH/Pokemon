# Foundation Model + Multi-branch Fine-tuning

**Date:** 2026-08-02
**Status:** approved, implementing
**Supersedes for stage 5:** the single-deck `--pretrain-finetune` path in `run_pipeline.sh`

## Problem

Stage-5 MCTS self-play reports an inflated win rate (~0.74 observed). The cause
is in the rollout, not the training loop: `RustVecEnv.collect_batch` routes
*every* pending observation — both seats — through one `act_fn`, so θ pilots the
opponent seat as well as its own. θ is a specialist on one archetype, and seat 1
is dealt a *sampled* archetype from the 179-deck pool. The opponent is therefore
a model playing 60 cards it has never seen, and beating it measures nothing.

`run_self_play_games` already drops the seat-1 decisions from training
(`mcts_train.py`, the `own_idx` filter), so the problem is confined to *who
acts*, not to what is learned.

A league (RL_SPEC §10.1 R4) would fix this, and is deliberately not implemented.
This design gets competent opponents without one.

## Approach

Two phases.

**Phase 1 — Universal Generalist IL, run once.** Train a single IL policy over
the whole corpus. Already supported:

```bash
./scripts/run_pipeline.sh --skip-download --generalist
```

`--generalist` trains `checkpoints_generalist/` and skips stage 5, which is
exactly Phase 1's scope. No change required.

**Phase 2 — Specialist MCTS Branching, 6 branches × R rounds, sequential.**

Define **expert *M*** as the model that owns deck *M*: the latest champion in
`checkpoints_a<M>_mcts/`, falling back to the generalist when branch *M* has not
yet produced one. At round 1 every expert *is* the generalist, so the cold start
is a clean "all six branches init from the same foundation model".

Branch *N* runs `ptcg_rl.mcts_train` with:

- θ_N warm-started from the generalist (`--init-ckpt`)
- seat 0 = θ_N playing deck *N* (`--deck-archetype N`)
- seat 1 = a **sampled** opponent deck, piloted by *that deck's expert* if one is
  registered, else the frozen generalist
- `--il-ckpt` = the generalist, so the KL anchor and the `IL_baseline` gate
  opponent are the same reference for all six branches and the six champions
  stay comparable to each other

Branches run one at a time; only one model trains per round. Between rounds the
experts refresh, so branch *N* in round *r+1* faces rivals that improved in
round *r*.

### The opponent pool, and the rival share

Restricting seat 1 to the six branch decks would guarantee every opponent has a
competent pilot, but the Kaggle opponent is out-of-distribution by construction.
So `--all-archetypes` stays: the rival decks gain a real pilot, and the rest
fall back to the generalist, which is still strictly better than a specialist on
an unknown deck.

That alone, however, does **not** produce the round-robin the branching implies.
Measured on this corpus: 201 archetypes in the pool, 4 rivals per branch — 2.0%
of games, about 4 of every 200. "Branch a1 fights the other experts" would be a
rounding error.

`--opp-expert-share F` (default 0.5 in the wrapper, 0.0 in `mcts_train` so no
existing caller changes behaviour) draws that fraction of games from the rival
decks and the rest from the full pool. Both properties survive: a real
round-robin against the other branches, and continued exposure to the 195
archetypes that no branch owns.

`sp/expert_opp_frac` reports the share actually faced, which is ≥ the share
deliberately drawn because a full-pool draw can land on a rival too. It is
distinct from `sp/opp_piloted_frac` — the default pilot covers *every*
archetype, so "has a pilot" is not "has an expert", and conflating them would
report 100% against an almost entirely generalist field.

## Components

### 1. `python/ptcg_rl/rust_vec_env.py` — seat-aware routing

`collect_batch(act_fn, n_decisions, act_fn_opp=None)`.

- `act_fn_opp is None` → byte-identical to current behaviour. This is the
  compatibility contract; every existing caller passes two arguments.
- Otherwise: split the polled batch by `obs["current"]["yourIndex"]`, route
  `self.our_player`'s observations to `act_fn` and the rest to `act_fn_opp`,
  and reassemble replies **in poll order** — `vec_env_reply` is positional, so a
  reordered list applies each pick to the wrong battle. `_eval_game_batch` in
  `mcts_train.py` already establishes this fill-by-index pattern.
- Only `act_fn` replies are recorded as `Decision`s. Seat-1 replies carry the
  *opponent's* `logp`/`value`, which must never enter θ's replay buffer.

Consequence: `len(traj.decisions)` roughly halves, and `collect_batch`'s budget
is denominated in *recorded* decisions. The caller compensates (below) or it
plays ~2× the games it needs before returning.

### 2. `python/ptcg_rl/mcts_train.py` — opponent registry

Two new optional flags:

- `--opp-default-ckpt PATH` — the seat-1 fallback (the generalist)
- `--opp-expert ID=PATH` — repeatable, per-archetype override

Neither given → `act_fn_opp=None` → today's behaviour, so no existing script or
test changes meaning.

`OpponentExperts` resolves `archetype id → policy`, loading lazily and caching.
`run_self_play_games` already groups games by opponent id (`by_opp`), so the
seat-1 actor is chosen once per group. Only one opponent policy is resident on
the GPU at a time — the previous one is moved back to CPU when a new group
starts, mirroring what `league_evaluate` does for champions.

Self-play budget: `budget = remaining * 150` when a seat-1 actor is active
(≈200 decisions/game across both seats, half of them recorded), `* 300`
otherwise.

### 3. `scripts/run_all_specialists.sh` — the Phase 2 driver

- Derives the six ids from `ptcg_il/archetype_select.py`. **Never hardcoded** —
  they are cluster indices and get reassigned on re-mine. This corpus's
  `self_ids` are `[1, 17, 25, 16, 36, 21]`.
- Outer loop over rounds (`for((;;))`, stop with Ctrl-C); inner sequential loop
  over the six decks.
- Recomputes the `--opp-expert` paths at the start of *each branch*, so a branch
  later in a sweep already sees rivals updated earlier in the same sweep.
- 10000 self-play games per branch per round.
- Resumable: `mcts_train` auto-resumes from the highest-ELO champion in its
  out-dir, and champions are only written after passing the gate, so an
  interrupt never loses passed work.

### 4. `scripts/run_pipeline.sh` — bug fix only

Stage 5 emits `--init-ckpt checkpoints_generalist/ckpt-best.pt` unconditionally
*and* appends `$RL_INIT`, which is a second `--init-ckpt` under
`--pretrain-finetune`. argparse resolves last-wins silently, and outside
`--pretrain-finetune` the hardcoded path need not exist. Remove the hardcoded
copy; keep `$RL_INIT`.

## Testing

In `python/tests/test_rl_mcts_wiring.py`, against the existing `_FakePolledEnv`
(no libcg, no libptcg_search):

1. **Seat split** — a poll carrying both seats sends exactly the seat-1
   observations to `act_fn_opp` and exactly the seat-0 ones to `act_fn`.
2. **Reply order** — the picks list handed to `reply()` is in poll order, not
   grouped-by-actor order. Mutation: return the concatenated groups instead and
   confirm the test goes red.
3. **Recording** — only seat-0 decisions land in `traj.decisions`; seat-1
   `logp`/`value` never appear.
4. **Compatibility** — `act_fn_opp=None` records both seats, as today.
5. **Registry** — `--opp-expert 17=path` resolves for id 17 and falls back to
   `--opp-default-ckpt` for an unregistered id; with neither flag,
   `actor_for` yields `None`.

Tests that count must fail on zero examined, per CLAUDE.md.

## Non-goals

- No league, no cross-play matrix, no Elo across branches beyond what
  `mcts_train` already tracks per out-dir.
- No merge/distill step folding the six champions back into one model.
- No parallel branches. Sequential was chosen; a `-j` cap would need
  `--n-engines`/`--n-workers` retuned per branch and makes games/min
  contention-dependent.
