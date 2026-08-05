# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pokémon TCG AI Battle — an imitation-learning pipeline, with a self-play RL stage on top, for the Kaggle Pokémon TCG competition. The repo has three packages and a four-document spec system.

**Spec documents (read in this order):**
- `AGENT_SPEC.md` — authoritative environment I/O reference (Observation/Option/Log schema, the `agent()` contract, search API). Grounded in the real `libcg.so` engine.
- `TRANSFORMER_IL_SPEC.md` — **the authoritative IL spec** (exact tensors, model architecture, training loop, corpus mining phases). Where other docs differ, this one wins.
- `RL_SPEC.md` — the self-play RL design (PPO, KL anchor, MCTS distillation, league, gates). Defers to TRANSFORMER_IL_SPEC.md on tensor layout and the `agent()` contract: RL changes how the policy is *trained*, not what it consumes or emits.
- `IL_SPEC.md` — design-rationale companion; superseded in detail by TRANSFORMER_IL_SPEC.md.

**Python 3.11**, managed via **uv** (`uv run`, `uv add`). Tests with `pytest`.

## Commands

```bash
# Install / sync dependencies
uv sync

# Run all tests (both test directories)
uv run pytest tests/ python/tests/

# Run a single test file
uv run pytest tests/test_download.py

# Run a specific test
uv run pytest tests/test_download.py::test_something -xvs

# Full 5-stage pipeline: Rust → mine → shards → train(+belief,+baselines) → RL
./scripts/run_pipeline.sh --skip-download

# Stages 1-4 only (RL runs by default and is much slower than the rest)
./scripts/run_pipeline.sh --skip-download --no-rl

# Stages 2 and 3 reuse their artifacts when corpus/config/code are unchanged.
# To recompute anyway:
./scripts/run_pipeline.sh --skip-download --force-mine --force-shards

# See which archetypes have enough data to train a specialist
cd python && uv run python -m ptcg_il.cli archetypes --data-dir data --describe

# --- Individual stages ---
# NOTE: cd python/ first. ptcg_mine/ and ptcg_il/ live under python/, and
# pyproject.toml's `pythonpath` applies to pytest only — `python -m ptcg_mine.mine`
# from the repo root raises ModuleNotFoundError. All corpus paths below are
# therefore relative to python/ (i.e. python/raw, python/data, python/checkpoints).
cd python

# Build the Rust search engine (only used by live-eval's search planner baseline)
cargo build --release --manifest-path ptcg_search/Cargo.toml

# Run the corpus mining pipeline (Phases 0–2)
uv run python -m ptcg_mine.mine --skip-download --raw-dir raw --out-dir data

# With download (needs Kaggle auth at ~/.kaggle/kaggle.json)
uv run python -m ptcg_mine.mine --n-days 5 --target-episodes 500

# Train the IL policy (Phase 3+)
uv run python -m ptcg_il.cli train --data-dir data --out-dir checkpoints

# Train a per-deck specialist (one model per archetype deck).
# Do NOT hardcode the archetype id — see "archetype ids are not stable" below.
uv run python -m ptcg_il.cli train --data-dir data --out-dir checkpoints_a0 \
    --archetype-self 0 --batch-size 512 --total-steps 5000 --val-every 250

# Resume from checkpoint
uv run python -m ptcg_il.cli train --resume checkpoints/ckpt-step-0004000.pt

# Eval-only from a checkpoint
uv run python -m ptcg_il.cli train --eval-only --resume checkpoints/ckpt-best.pt

# Record the IL baseline the RL gate compares against (held-out test split,
# SHA-1-pinned to the checkpoint)
uv run python -m ptcg_il.cli train --eval-only --data-dir data \
    --out-dir checkpoints_a0 --resume checkpoints_a0/ckpt-best.pt \
    --archetype-self 0 --eval-split test --record-baseline

# --- RL (stage 5) ---
# R1 critic repair only — diagnose the value head before spending PPO compute
uv run python -m ptcg_rl.train --data-dir data --il-ckpt checkpoints_a0/ckpt-best.pt \
    --out-dir checkpoints_a0_rl --deck-archetype 0 --phase r1

# R1 then R2 (PPO + KL anchor). R2 does not start unless R1's gate passes.
uv run python -m ptcg_rl.train --data-dir data --il-ckpt checkpoints_a0/ckpt-best.pt \
    --out-dir checkpoints_a0_rl --deck-archetype 0 --total-steps 50000
```

## Architecture

### Three-package structure

**`ptcg_mine/`** — Corpus mining pipeline (Phases 0–2). No PyTorch dependency.

| Module | Role |
|---|---|
| `config.py` | `MineConfig` dataclass — all pipeline parameters (days, experts, vocab, thresholds) |
| `download.py` | Phase 1: resumable, multi-threaded Kaggle episode downloader. Takes an injected `api` adapter (never imports kaggle directly) so tests can supply fakes. |
| `mine.py` | CLI orchestrator (`python -m ptcg_mine.mine`). Wires download → stats → archetypes → vocab → static tables → artifacts. |
| `episode.py` | Parse/validate episode JSON: `load_episode`, `validate_episode`, `deck_of`, `teams`, `rewards` |
| `stats.py` | Team leaderboard aggregation and expert selection (top-K by win_rate, min-games threshold with best-effort fallback) |
| `archetype.py` | Deck canonicalization, Jaccard-multiset clustering, 𝒟_self/𝒟_opp selection |
| `vocab.py` | Build card-id → contiguous-index vocabulary from corpus decks (PAD=0, UNKNOWN=1, real ids 2..V-1) |
| `cards.py` | Load engine static data (`all_card_data`, `all_attack`) and build frozen `.npy` feature tables |
| `sampling.py` | Day selection and deterministic episode picking |
| `artifacts.py` | Write frozen artifacts: `vocab.json`, `archetypes.json`, `mining_report.md` |
| `stamp.py` | Stage-freshness fingerprints. Used by `mine.run` and `cli.cmd_build_shards` to skip an unchanged stage; also a CLI (`check` exits 10 when stale, 0 when cached; `write` records one). |

Pipeline phases: **Phase 0** (day sampling) → **Phase 1** (selective episode download) → **Phase 2** (stats → experts → archetypes → vocab → static tables → frozen artifacts to `data/`).

**`ptcg_il/`** — Transformer IL policy (Phases 3–4). Requires PyTorch.

| Module | Role |
|---|---|
| `cli.py` | Training CLI (`python -m ptcg_il.cli train`). Loads artifacts, builds policy, runs QA gates, dispatches training loop. |
| `featurizer.py` | Pure-NumPy `obs_dict → tensor dict` per TRANSFORMER_IL_SPEC.md Appendix A. Fixed token layout: CLS + P_MAX(12) Pokémon + H_MAX(30) hand + SUM(2) summaries + STAD(1) = L_STATE=46, O_MAX=64 options. |
| `model/policy.py` | `Policy(nn.Module)` — end-to-end: embed → encode → pointer + value. Also `multiselect_ce` (teacher-forced AR training) and `select_multi` (greedy AR inference). |
| `model/embed.py` | `TokenEmbedder` — maps token-type + card-id + static features → D-dim embeddings |
| `model/encoder.py` | `Encoder` — `nn.TransformerEncoder` with pre-norm, GELU, over the L_STATE token sequence |
| `model/pointer.py` | `PointerHead` — cross-attention from option tokens over encoded state tokens, produces per-option logits |
| `model/value.py` | `ValueHead` — MLP off the CLS token, predicts game outcome ∈ (−1, 1) |
| `model/cards.py` | `CardEncoder` — card-id embedding + MLP over static features (52-dim: hp, type, stage, weakness, etc.) |
| `train/loop.py` | Full training loop: `ShardDataset`, AdamW with no-decay param groups, linear-warmup→cosine-decay schedule, EMA, mixed-precision (bf16), early stopping, W&B logging |
| `train/dataset.py` | `ShardDataset` — reads pre-featurized `.pt` shards from `data/shards/` with train/val split |
| `train/eval.py` | Offline eval: top-1/top-3 accuracy per SelectType/context |
| `train/checkpoint.py` | Save/load checkpoints (model, optimizer, scheduler, EMA state) |
| `live_eval.py` | Live-engine evaluation against baseline opponents (random, frozen checkpoint, search planner) |
| `qa.py` | QA gates: coverage, label sanity, deck legality checks before training |
| `deck.py` | Deck identity for a trained policy — builds the record stamped into each `.pt` under the `"deck"` key, the `decks.json` sidecar, and `deck.csv` |
| `shard_writer.py` | Write featurized decision points to sharded `.pt` files |

**`ptcg_rl/`** — Self-play RL on the trained IL specialist (RL_SPEC R1–R2). Requires PyTorch.

| Module | Role |
|---|---|
| `config.py` | `RLConfig` — every knob in RL_SPEC §9.4, validated at construction |
| `actor.py` | **The single source of truth for option masking.** `sample_action` / `recompute_logp`, joint-sequence log-probs including the STOP step |
| `vec_env.py` | `RolloutPool` — worker processes drive `libcg` only; the parent featurizes and batches inference |
| `rollout.py` | GAE (fp32), advantage normalisation, `PolicyActor` (the batched action callback) |
| `ppo.py` | Clipped surrogate, clipped value loss, adaptive-β KL anchor, the epoch-0 ratio canary |
| `critic.py` | R1 critic repair + the §3 diagnostics (`corr`, `pred_std`, per-turn accuracy) |
| `belief.py` | `ArchetypePosterior` — Bayesian categorical over 𝒟_opp, no training |
| `gate.py` | Paired evaluation, Wilson bound, SHA-pinned IL-regression check |
| `train.py` | CLI: `python -m ptcg_rl.train --deck-archetype 0 --il-ckpt …` |
| `logger.py` | W&B for stage 5 — `RLWandbLogger` (subclasses IL's `WandbLogger`), `NullRLLogger`, `build_logger` |

Stage 5 runs R1→R2 once per trained specialist, **sequentially**. That follows RL_SPEC §10.1's alternating-not-concurrent decision, but it is *not* the §10.1 league — there is no cross-play, past-champion pool, or Elo. Each deck does self-play against its own frozen π_IL and is gated independently.

Not implemented (deliberately): `search.py` (R3 MCTS distillation) and `league.py` (R4). RL_SPEC §11 makes R2 the go/no-go — if PPO with a KL anchor cannot beat frozen IL on one deck with self-play alone, adding search and a league only makes the failure harder to diagnose. There is also no Rust `VecEnv`: §6.5 downgraded that port to optional once the NumPy featurizer got 3.57× faster.

**Model flow:** `obs_dict` → `featurize()` → tensor dict → `TokenEmbedder` → `Encoder` (self-attention over ~46 state tokens) → `PointerHead` (cross-attention: option tokens query state tokens) → per-option logits. Multi-select handled autoregressively with teacher-forcing at training time, greedy at inference.

### Engine bindings

`python/pokemon-tcg-ai-battle/` contains the Kaggle-provided `cg/` package (`api.py`, `game.py`, `sim.py`, `utils.py`) wrapping a C++ `libcg.so`. This is the ground-truth game engine used by `live_eval.py`.

### Key design decisions

- **Kaggle API is always injected** (`download.py` takes an `api` parameter) — no real network calls in tests.
- **Vocab is decoupled from the archetype filter**: vocab built from all cards in the corpus (~300–500 ids) to limit live OOV; training games filtered to 𝒟_self/𝒟_opp archetypes only.
- **Training uses both won and lost expert games**, with losses down-weighted (`w_lost=0.6`), not dropped. This preserves even-board-state data and reduces covariate shift.
- **The deck is fixed, not learned** — no deck-generation head. `FIXED_DECK` is the representative of the best 𝒟_self archetype.
- **One model per deck beats one model for all decks.** The six 𝒟_self archetypes are near-disjoint (pairwise multiset-Jaccard ≤ 0.17, *no* card common to all six, union only 89 cards), so a single policy fits several unrelated strategies at once. Training with `--archetype-self <id>` roughly doubles-to-quadruples non-trivial top-1 lift on held-out test (arch 0: +2.7 → +20.5 pts; arch 2: +8.8 → +49.4 pts).
- **Every checkpoint is deck-labelled.** `save_checkpoint` stamps the `ptcg_il.deck` record under the `"deck"` key, and training writes a `decks.json` sidecar plus `deck.csv`. A wrong model/deck pairing otherwise fails *silently* — unseen cards just map to `UNKNOWN_CARD` — so the record also pins `vocab_sha1`/`archetypes_sha1`, since archetype ids are only cluster indices and get reassigned when mining is re-run.
- **Value head is auxiliary** (MSE to game outcome ±1) — intended as a free warm-start critic for RL. **Measured, it is collapsed**: prediction std 0.0005 against a target std of ~1.0, correlation ≈ 0.02 on a correctly-loaded checkpoint. Constant-zero is the MSE-minimising answer when the input carries no signal, so this is invisible in the loss. RL stage 5a exists to repair it and gates 5b on the result.

- **Archetype ids are append-only within a lineage, and only within one.** They are cluster indices assigned in the order the greedy pass opens clusters, which follows the global deck-frequency ordering — so an *unseeded* re-mine renumbers clusters whose membership never changed, and `--archetype-self N` silently becomes a different deck. `ptcg_mine.mine` therefore seeds `cluster_decks` from `<out-dir>/archetypes.json` **by default**: baseline clusters keep their id and their representative, unmatched decks open new clusters past the baseline maximum, and a baseline cluster with no members in the new corpus is retained at frequency 0 rather than freeing its id. `--rebaseline` is the explicit opt-out that renumbers and invalidates every trained checkpoint; `--baseline-archetypes PATH` seeds from elsewhere. `archetypes.json` records which generation it belongs to under `lineage` (`seeded`, `baseline_sha1`, `generation`).

  **Stable ids are only half of it — `opp_ids` order is the other half.** `shard_writer` builds the belief label as `{gid: i for i, gid in enumerate(opp_ids)}`, so the class index is a *position*, not an id. A re-sorted `opp_ids` of unchanged length keeps the head the same width, loads without complaint, and repoints every class. `select_self_opp` therefore appends rather than re-ranks (`archetype._append_only`), which means `n_opp` stops being the head's width and becomes "how many of this corpus's top archetypes may be *added*" — the list only grows. When it does grow, warm-starting an older checkpoint needs `--allow-belief-widening`, which copies its rows into the leading slots; that is sound only because the ordering is append-only, and it raises if the head shrank (which means someone re-baselined).

  Ids being stable does **not** make them guessable: still derive them with `ptcg_il/archetype_select.py` from `archetypes.json` ∩ `meta.parquet`, ranked by training rows and filtered to those with enough held-out data to be model-selected. The pipeline shipped `ARCHETYPES="0 2"` while this corpus's `self_ids` are `[1, 17, 25, 16, 36, 21]`. `run_pipeline.sh` calls it; so should you.

  The cost, which is real: a seeded cluster's **representative is frozen**. A newly-arrived decklist more frequent than the one that opened the cluster does not replace it, so the decklists `ptcg_il.deck` stamps into checkpoints stop tracking the meta. Refreshing them is a deliberate `--rebaseline`, not a side effect of mining.

- **The IL baseline is an artifact, not a constant.** `data/il_baselines.json` is written by pipeline stage 4c from the held-out **test** split and SHA-1-pinned to the checkpoint that produced it. The RL gate refuses to run against a record whose SHA does not match. Hardcoded thresholds went stale once already when the `opt_card_id` fix changed the input distribution, and a stale threshold does not raise — it quietly passes or fails the wrong candidate.

- **Belief heads always train.** `build-shards` writes belief labels for every decision point unconditionally, so the old opt-in `--belief` flag guarded against a corpus that no longer exists while leaving the MCTS determinizer on its mirror-deck guess by default. Use `--no-belief` to disable.

- **Stage 5 runs RL on the best `--rl-top` decks (default 2), not all of them.** R1+R2 costs hours per deck, so with `--n-archetypes all` the old behaviour spread the budget evenly across decks that offline eval already ranks far apart. The ranking comes from `data/il_baselines.json` (held-out **test** top-1, written by stage 4c) intersected with the archetypes that actually produced a `ckpt-best.pt`, so a high-scoring record with no checkpoint on disk cannot be selected. `--rl-archetype` still pins a single deck and overrides the ranking; with no baselines file (e.g. `--no-eval`) it falls back to training order and says so rather than implying it ranked anything. The deck follows the model by construction: `--il-ckpt checkpoints_a<N>/` and `--deck-archetype <N>` are the same `N`, and `ptcg_rl.train` rebuilds the decklist from that id. Offline top-1 is the selector of convenience, not of truth — see the 78%-vs-48% live gap below; re-rank with `--live-eval` before committing a submission.

- **Option suppression goes in the mask, never only in the logits.** `masked_label_smoothed_ce` spreads the label-smoothing mass uniformly over positions the mask calls *valid*, so a column carrying a `-1e9` logit while still masked-valid contributes `log_prob = -1e9` to that average — about `ε·1e9/n_valid` per step. `multiselect_ce` did exactly that when suppressing the STOP column below `minCount`: real-batch multi-select CE was **2.47e7** against a plausible 9.2, and batch loss 5.3e6. It stays *finite*, so every `assert torch.isfinite(ce)` passed and `grad_clip=1.0` hid the magnitude (`grad_norm` read a normal 1.08) while the gradient direction on ~8% of rows was pure artifact. Only training was affected; no val CE is logged, so eval numbers looked fine throughout.

- **IL training logs to W&B `poken/pokemon-tcg-il` by default.** `ptcg_il/cli.py` `DEFAULTS` sets entity/project; `train/logger.py` does the `wandb.init`. Run names are suffixed per model (`pokemon-tcg-il-a0`, `-a1`, `-generalist`) because pipeline stage 4a trains one specialist per archetype into the same project and identically-named runs can only be told apart by opening their config. An explicit `--wandb-name` is honoured verbatim. `--no-wandb` disables it (the flag existed but was never read, so it used to log online anyway); `--wandb-mode offline|disabled` and `--wandb-entity` are the other escapes. Eval-only runs (stage 4c) deliberately create no W&B run. A failed `wandb.init` — wrong team, no auth — only logs a warning and falls back to console metrics, so check the training log if runs stop appearing.

- **RL logs to a *different* W&B project, `poken/pokemon-tcg-rl`.** `ptcg_rl/logger.py` `DEFAULTS`; same flag surface as IL (`--wandb-project/-entity/-name/-mode`, `--no-wandb` winning over `--wandb-mode`), run names `pokemon-tcg-rl-a<N>`. Separate project because the two stages share no axes — IL plots loss and top-1 against optimizer steps, RL plots `kl_to_il`/`beta`/`clip_fraction` against PPO steps on an already-trained policy — so one project's default panels are unreadable with both in it. **R1 and R2 run on different clocks** (critic-repair steps vs PPO optimizer steps) and each is declared against its own `step_metric`, so wandb's global step, which only counts `log` calls, is never the x axis of either. R1's curve is replayed from `critic.repair`'s recorded `history` after each phase — `repair` records rather than calling back — with `r1/step` continuing from phase A into B so the two draw as one curve. R2 logs one point per rollout buffer, matching the console line, with `r2/step` as the cumulative optimizer step and not the iteration index. `NullRLLogger` is the `--no-wandb` stand-in, and `test_rl_logger.py` asserts it implements every public method of `RLWandbLogger` — a logging call added to one and not the other raises only for users who opted out.

- **Shards are decompressed once to `.npy`, then genuinely mmap'd.** `np.load`'s `mmap_mode` is *silently ignored* for `.npz`, so the old `dict(np.load(path, mmap_mode="r"))` decompressed every array into anonymous RAM — 5.6 MB on disk → 528 MB resident per 50k-sample shard (~94×; the padded feature tensors are mostly zeros) — and held it for the process lifetime, per DataLoader worker. Measured on a real 60-step run over the 187k-sample corpus: **17.8 GB → 5.9 GB peak PSS**, and 120 batches went 6.4 s → 2.0 s. `ShardDataset.__init__` now builds `shards/.mmap-cache/<shard>.npz/<key>.npy` (in the parent, before workers fork, one array at a time) and mmaps it, so pages are shared and evictable. Costs ~1.6 GB of disk mirroring `shards/`; the cache self-invalidates on shard size/mtime and is rebuilt if incomplete.

- **Mine and build-shards skip themselves when nothing changed.** Both are pure functions of (raw corpus, config knobs, the code implementing them) yet cost ~15 min and ~35 min over the 9910-episode corpus, so the pipeline re-ran ~50 min of identical work on every invocation. `ptcg_mine/stamp.py` records the inputs in `data/.stamp-<stage>.json`; **the check lives in `ptcg_mine.mine.run` and `cli.cmd_build_shards`, not in the shell**, so calling either module directly gets it too and `run_pipeline.sh` only forwards `--force`. Mine checks *after* Phase 1: the download still runs, and a run that fetched no new episodes skips Phase 2 rather than re-parsing for nothing. It must not degrade to a file-exists check — card ids are vocab indices and archetype ids are cluster indices, so artifacts that merely exist can be silently wrong for the current code. The fingerprint therefore covers the source files that decide the output (a `featurizer.py` edit rebuilds shards), the upstream `vocab.json`/`archetypes.json` content, and the config knobs a caller can vary; numeric params are canonicalised so `0.90` and `0.9` are not a change. Each stage stamps itself only after its outputs are complete, so a failed run leaves nothing to trust. `--force` on either CLI, or `--force-mine` / `--force-shards` / `--force` on the pipeline.

- **Every checkpoint records its own architecture.** `Policy.config` holds `V/A/D/heads/layers/ff/n_opp_arch` and `save_checkpoint` writes it. RL loads two policies (θ and the frozen π_IL) and cannot infer shapes: a wrong `D` fails loudly, but a wrong `n_opp_arch` does not — `load_policy_state` forgives missing belief keys, so the result is a plausible model with randomly-initialised heads.

### Test layout

Two test directories:
- `tests/` — root-level tests for `ptcg_mine` (download, episode, stats, sampling, mine integration)
- `python/tests/` — comprehensive tests covering `ptcg_mine`, `ptcg_il` and `ptcg_rl` (model components, featurizer, training loop, QA, live eval, RL)

`pyproject.toml` sets `pythonpath = ["."]` so both packages are importable without installing.

**Run the two directories separately.** They share test-file basenames, so
`pytest tests/ python/tests/` fails collection with "import file mismatch".

Tests that count occurrences must **fail on zero examined** — a test that checks a
property only when the fixture happens to contain the relevant case passes
vacuously on an empty loop. Validate new guards by **mutation**: break the thing
deliberately and confirm the test goes red. Two of the RL guards were caught this
way (the STOP-step term in the joint log-prob, and a NaN gradient from
`torch.where` in the masked entropy).

## Working style

- Don't explain too much at the end of a task, unless necessary.
- Do not run git commands (commit, push, branch, etc.) unless explicitly asked.
