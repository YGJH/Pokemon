# AGENTS.md — Pokémon TCG AI Battle

Guidance for AI coding agents working in this repository. Assumes no prior
knowledge of the project.

## Project overview

An imitation-learning (IL) pipeline, with a self-play RL stage on top, for the
**Kaggle Pokémon TCG AI Battle competition**. The final artifact is a Kaggle
submission tarball: a fixed 60-card deck plus a transformer policy that plays
the game through the competition's C++ engine (`libcg.so`).

The game is **not** a fixed-action-space environment: the engine pauses at each
decision point, hands the agent an `Observation` and a list of legal `Option`s,
and the agent replies with **indices into that option list** (`agent(obs_dict)
-> list[int]`, see `AGENT_SPEC.md`). The policy is therefore an
entity-transformer encoder with a **pointer head** over the variable-length
option set; multi-select decisions are factorized autoregressively.

The pipeline, end to end:

1. **Mine** a corpus of replay "episodes" from Kaggle (`ptcg_mine`).
2. **Featurize** decision points into `.npz` shards (`ptcg_il build-shards`).
3. **Train** one specialist policy per deck archetype (`ptcg_il train`).
4. **Evaluate** offline (top-1/top-3 accuracy) and live (real engine games).
5. **RL**: AlphaZero-style MCTS self-play distillation (`ptcg_rl.mcts_train`);
   a PPO path (`ptcg_rl.train`, RL_SPEC R1/R2) also exists.
6. **Package** a self-contained submission tarball (`scripts/build_submit.sh`).

## Repository layout

```
main.py                  Kaggle agent entry point (vendored bundle, no-MCTS build)
model/                   Vendored copy of ptcg_il/model + featurizer + ref_map
                         + deck_guard (anti-deck-out guard),
                         imports rewritten to `model.*` — part of the submission
                         bundle; kept in sync by python/tests/test_vendored_sync.py
data/                    Submission data dir: model.pt, vocab.json, deck.csv,
                         engine_card_features.npy, engine_attack_features.npy
scripts/                 Pipeline + packaging shell scripts (see below)
tests/                   Root tests: ptcg_mine, build_submission (268 tests)
python/
  ptcg_mine/             Corpus mining pipeline, Phases 0–2 (no PyTorch dep)
  ptcg_il/               Transformer IL policy, Phases 3–4 (model/, train/)
  ptcg_rl/               Self-play RL: PPO (train.py) + MCTS distillation
                         (mcts_train.py, search.py), arena/tournament/elo tools
  ptcg_search/           Rust cdylib: MCTS/PUCT planner bridging libcg.so
                         (builds libptcg_search.so; cargo)
  pokemon-tcg-ai-battle/ Kaggle-provided engine package. The `cg/` Python
                         bindings (api.py, game.py, sim.py) wrapping libcg.so
                         live at .../sample_submission/sample_submission/cg/
                         and are added to sys.path at runtime by live_eval.py
  tests/                 Comprehensive tests: ptcg_mine + ptcg_il + ptcg_rl
                         (59 files, 1252 tests)
  checkpoints*/          Trained models (gitignored): checkpoints_a<N> per
                         archetype, _s<S> per ensemble seed, _mcts for RL output
  raw/, data/            Corpus + featurized shards (gitignored)
docs/                    Design plans and review notes
wandb/                   Local W&B run logs (gitignored)
```

## Spec documents (read in this order)

- `AGENT_SPEC.md` — authoritative environment I/O reference: Observation /
  Option / Log schema, the `agent()` contract, engine search API, static card
  data. Grounded in the real `libcg.so`.
- `TRANSFORMER_IL_SPEC.md` — authoritative IL spec (exact tensors, model
  architecture, training loop, corpus mining). Where docs differ, this wins.
- `RL_SPEC.md` — self-play RL design (PPO, KL anchor, MCTS distillation,
  league gates). Defers to TRANSFORMER_IL_SPEC on tensor layout.
- `IL_SPEC.md` — design-rationale companion, superseded in detail.
- `PUCT_spec.md`, `arena_spec.md`, `python/MCTS_SPEC.tex` — search/arena designs.
- `CLAUDE.md` — the living engineering log: every load-bearing design decision
  with measured rationale. **Read it before touching training, corpus, or
  packaging code** — many invariants below are expanded there.

Note: the spec still describes the belief module / history GRU, but the
**current implementation retired them** (`history_gru.*`, `belief.*`,
`belief_heads.*` are `_RETIRED_PREFIXES` in `ptcg_il/model/policy.py`, dropped
on checkpoint load). Opponent-deck determination is now the torch-free
Bayesian `ptcg_il/deck_prior.py`. Code wins over the spec on this point.

## Tech stack

- **Python 3.11**, managed by **uv** (`uv sync`, `uv run`; `.python-version`
  pins 3.11, `uv.lock` locks deps).
- Key deps: `torch >= 2.8` (CUDA 12.8 wheels via explicit
  `pytorch-cu128` index — required for Blackwell/RTX 50xx sm_120), `numpy`,
  `pandas`, `pyarrow` (parquet), `orjson`, `rich`, `wandb`, `kaggle`/`kagglehub`.
- **Rust** (edition 2021, `cargo`): `ptcg_search` cdylib — MCTS planner and
  vec-env bridging the engine through a C/C++ bridge (`bridge/bridge.c*`).
- **C++ game engine**: Kaggle-provided `libcg.so` behind the `cg/` Python
  bindings; ground truth for live eval and RL rollouts.
- **pytest** for tests (dev dependency group).
- `pyproject.toml` sets `pythonpath = [".", "python"]` so all packages import
  without installation.

## Setup and build

```bash
uv sync                                          # install deps

# Rust search engine (only needed by live-eval's search-planner baseline
# and the MCTS paths; skipped automatically if cargo is absent)
cargo build --release --manifest-path python/ptcg_search/Cargo.toml
```

**The `cd python/` rule (important):** `ptcg_mine`, `ptcg_il`, `ptcg_rl` live
under `python/`, and `pyproject.toml`'s `pythonpath` applies to pytest only.
`python -m ptcg_mine.mine` from the repo root raises `ModuleNotFoundError`.
All module invocations below assume `cd python` first; corpus paths
(`raw`, `data`, `checkpoints*`) are relative to `python/`. The pipeline script
handles this itself.

## The 5-stage pipeline

`scripts/run_pipeline.sh` runs everything (comments in the script are Chinese;
`--help` prints full usage):

```bash
./scripts/run_pipeline.sh --skip-download              # full run, reuse raw/
./scripts/run_pipeline.sh --skip-download --no-rl      # stages 1–4 only
./scripts/run_pipeline.sh --skip-download --force-mine --force-shards
```

Stages: **[1]** Rust build → **[2]** mine (download → stats → archetypes →
vocab → static tables) → **[3]** build-shards (featurize to `.npz` +
`meta.parquet`) → **[4]** train one specialist per archetype + eval + record IL
baselines → **[5]** MCTS self-play distillation per selected deck
(`ptcg_rl.mcts_train`, sequential, gated; checkpoints only written when the
league gate passes).

Stages 2 and 3 **skip themselves when inputs are unchanged** — fingerprints
(corpus + config knobs + implementing source files + upstream artifact SHA-1s)
live in `ptcg_mine/stamp.py` and are checked inside `ptcg_mine.mine` /
`ptcg_il.cli build-shards`, not in the shell. Each full stage logs to
`python/logs/pipeline-<ts>/` (`logs/latest` symlink).

Individual stages (from `python/`):

```bash
uv run python -m ptcg_mine.mine --skip-download --raw-dir raw --out-dir data
uv run python -m ptcg_il.cli build-shards --raw-dir raw --out-dir data
uv run python -m ptcg_il.cli train --data-dir data --out-dir checkpoints_a1 \
    --archetype-self 1 --batch-size 512 --total-steps 5000
uv run python -m ptcg_il.cli train --eval-only --resume checkpoints_a1/ckpt-best.pt \
    --data-dir data --archetype-self 1 --eval-split test --record-baseline
uv run python -m ptcg_il.cli archetypes --data-dir data --describe   # pick archetypes
uv run python -m ptcg_rl.train --data-dir data --il-ckpt checkpoints_a1/ckpt-best.pt \
    --out-dir checkpoints_a1_rl --deck-archetype 1                   # PPO path (R1→R2)
```

Ensembles: train N specialists with different `--seed` (or
`scripts/run_ensemble_train.sh <ARCH> <N>`), then evaluate/select/package —
see `README.md` for the full workflow (`--ensemble-select` greedy forward
selection fit on **val**, packaged via `--ensemble-top`).

## Testing

Two test directories, **run them in separate pytest invocations** — they share
four test-file basenames (`test_download.py`, `test_episode.py`,
`test_sampling.py`, `test_stats.py`) and neither has an `__init__.py`, so
`pytest tests/ python/tests/` aborts collection with "import file mismatch"
(verified: 4 collection errors).

```bash
uv run pytest tests/          # 268 tests, ~10 s
uv run pytest python/tests/   # 1252 tests
uv run pytest tests/test_download.py::test_something -xvs
```

**Machine-specific gotcha:** this host's `PYTHONPATH` contains ROS Jazzy's
`/opt/ros/jazzy/lib/python3.12/site-packages`, which leaks a broken pytest
plugin entry point (`lark` missing) into startup — bare `uv run pytest` crashes
before collecting anything. Workaround:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/
```

Testing conventions (enforced by the existing suite):

- **The Kaggle API is always injected** (`ptcg_mine/download.py` takes an `api`
  adapter) — no real network calls in tests; supply fakes.
- Tests that count occurrences must **fail on zero examined** — a guard that
  only fires when the fixture happens to contain the case passes vacuously on
  an empty loop.
- Validate new guards by **mutation**: break the thing deliberately, confirm
  the test goes red, revert.
- `python/tests/test_vendored_sync.py` guards drift between `ptcg_il/` and the
  vendored submission copy under root `model/` — update both when changing
  featurizer dims or model modules.

## Package architecture

### `ptcg_mine/` — corpus mining (Phases 0–2, no torch)

| Module | Role |
|---|---|
| `config.py` | `MineConfig` dataclass — all pipeline knobs |
| `download.py` | Phase 1: resumable multi-threaded Kaggle downloader (injected `api`) |
| `mine.py` | CLI orchestrator (`python -m ptcg_mine.mine`) |
| `episode.py` | Parse/validate episode JSON |
| `stats.py` | Team leaderboard, `skill_weight = exp(20·(wilson_lb − 0.5))`, expert ranking |
| `archetype.py` | Deck canonicalization, Jaccard-multiset clustering, 𝒟_self/𝒟_opp selection |
| `vocab.py` | Card-id → contiguous-index vocab (PAD=0, UNKNOWN=1) |
| `cards.py` | Engine static data → frozen `.npy` feature tables |
| `sampling.py` | Day selection, deterministic episode picking |
| `artifacts.py` | Frozen artifacts: `vocab.json`, `archetypes.json`, `mining_report.md` |
| `stamp.py` | Stage-freshness fingerprints (also a CLI: `check` exits 10 when stale) |

### `ptcg_il/` — transformer IL policy (Phases 3–4)

| Module | Role |
|---|---|
| `cli.py` | Training CLI: `train`, `build-shards`, `archetypes`; QA gates; `DEFAULTS` |
| `featurizer.py` | Pure-NumPy `obs_dict → tensor dict`; fixed token layout (L_STATE=46, O_MAX=64) |
| `model/policy.py` | `Policy(nn.Module)`: embed → encode → pointer + value; `multiselect_ce`, `select_multi`; `_RETIRED_PREFIXES` |
| `model/embed.py`, `encoder.py`, `pointer.py`, `value.py`, `cards.py` | TokenEmbedder, pre-norm TransformerEncoder, cross-attention PointerHead, CLS ValueHead, CardEncoder |
| `train/loop.py` | Training loop: ShardDataset, AdamW, warmup→cosine, EMA, bf16, early stop, W&B |
| `train/dataset.py` | `ShardDataset` (mmap'd shards), `compute_sample_weights` |
| `train/eval.py`, `train/checkpoint.py`, `train/logger.py`, `train/muon.py` | Offline eval, checkpoint I/O, W&B logging, Muon optimizer |
| `live_eval.py` | Live-engine eval vs random / frozen ckpt / search planner |
| `deck.py`, `deck_prior.py` | Deck identity stamped into checkpoints; Bayesian opponent-deck determinizer |
| `deck_guard.py` | Inference-time anti-deck-out guard: hard mask for guaranteed deck-out plays + gated near-tie deck-delta tie-break; self-contained, vendored verbatim into the bundle |
| `ensemble.py`, `ensemble_select.py` | `EnsemblePolicy` (log-mean-softmax), greedy forward selection on val |
| `shard_writer.py` | Featurized decision points → `.npz` shards + streamed `meta.parquet` |
| `qa.py`, `diagnose.py`, `archetype_select.py`, `baselines.py` | QA gates, diagnostics, archetype ranking, baseline records |

Model flow: `obs_dict` → `featurize()` → `TokenEmbedder` → `Encoder`
(self-attention over ~46 state tokens) → `PointerHead` (option tokens query
state tokens) → per-option logits. Multi-select: teacher-forced AR at training,
greedy AR at inference.

### `ptcg_rl/` — self-play RL

| Module | Role |
|---|---|
| `train.py` | PPO path CLI (RL_SPEC R1 critic repair → R2 PPO + KL anchor) |
| `mcts_train.py`, `search.py` | AlphaZero-style MCTS self-play distillation (pipeline stage 5) |
| `actor.py` | **Single source of truth for option masking**; joint-sequence log-probs |
| `vec_env.py`, `rust_vec_env.py` | RolloutPool: worker processes drive `libcg`, parent batches inference |
| `rollout.py`, `ppo.py`, `critic.py`, `gate.py` | GAE, clipped PPO + adaptive-β KL anchor, R1 critic repair, paired-eval gate |
| `config.py`, `logger.py` | `RLConfig` (validated at construction); W&B (`poken/pokemon-tcg-rl`) |
| `arena.py`, `tournament.py`, `rating.py`, `elo_calibrate.py` | Head-to-head evaluation and Elo tooling |

### Vendored submission bundle (repo root)

`main.py` + `model/` + `data/` mirror the packaged submission: `main.py` loads
the model **at import time**, featurizes each observation, applies the
`model/deck_guard.py` anti-deck-out guard (mask + tie-break), and returns
argmax (greedy AR for multi-select) option indices, clamped to the engine's
Select contract (distinct, in range, minCount ≤ k ≤ maxCount). The no-MCTS
build dlopens nothing. `scripts/build_submission.py` regenerates `model/` from
`ptcg_il/` with import rewrites (`deck_guard.py` is copied verbatim, like
`ref_map.py`) — edit `ptcg_il/`, not the vendored copy.

### `ptcg_search/` (Rust)

`cdylib` producing `libptcg_search.so`: `src/{mcts,puct}.rs` (search),
`engine.rs`/`ffi.rs`/`bridge.rs` (libcg bridge), `guessing.rs`
(determinization), `vec_env.rs`. Used by live-eval's search-planner baseline
and the MCTS paths; the greedy submission does not need it.

## Load-bearing invariants (details and measurements in CLAUDE.md)

- **Never hardcode archetype ids.** They are cluster indices; only a *seeded
  re-mine* keeps them stable. Derive them from artifacts
  (`ptcg_il.cli archetypes` / `archetype_select.py`). `--rebaseline` renumbers
  and invalidates every trained checkpoint.
- **Every checkpoint is self-describing**: it stamps its `Policy.config`
  (architecture + `feat_dims`) and a `deck` record pinning
  `vocab_sha1`/`archetypes_sha1`. Build policies from the record, never from
  guessed shapes. A wrong model/deck pairing fails *silently* (unseen cards →
  UNKNOWN / zero features).
- **Skill is a weight, not a filter**; only the own-deck (𝒟_self) archetype
  filters the corpus. Losses are down-weighted (`w_lost=0.6`), not dropped.
- **Option suppression goes in the mask, never only in the logits** —
  label-smoothed CE over masked-valid positions turns a `-1e9` logit that is
  still mask-valid into a huge finite loss that `isfinite` checks and grad
  clipping both hide.
- **The IL baseline is an artifact** (`data/il_baselines.json`, held-out test
  split, SHA-1-pinned to its checkpoint) — never a hardcoded threshold.
- **Ensemble subsets come from the recorded greedy ordering**, never from glob
  order or per-member scores; `--ensemble-top` aborts rather than falling back.
- **Shards**: store card ids, re-gather `*_card_feat` on read; decompressed to
  `.npy` and genuinely mmap'd (`np.load(mmap_mode=...)` is silently ignored for
  `.npz`). Memory blowups here present as `systemd-oomd` killing the whole
  terminal scope with no Python traceback.
- **W&B projects**: IL → `poken/pokemon-tcg-il`, RL → `poken/pokemon-tcg-rl`,
  MCTS → `poken/pokemon-tcg-mcts` (separate because the x-axes differ).
  `--no-wandb` / `--wandb-mode` are the escapes; a failed `wandb.init` only
  warns and falls back to console.

## Submission packaging

```bash
./scripts/build_submit.sh a1                          # single ckpt (auto-picks best Elo)
./scripts/build_submit.sh a1 --no-mcts                # pure policy → submission-greedy.tar.gz
./scripts/build_submit.sh --ensemble "python/checkpoints_a1_s*/ckpt-best.pt" --ensemble-top 7
```

`build_submission.py` verifies the checkpoint's pinned vocab/archetype SHA-1s
against the current `python/data` and **aborts on mismatch** — packaging an
off-corpus checkpoint would produce a bundle that runs but loses silently.
Ensemble members must share one decklist (mismatch raises). Output:
`submission.tar.gz` / `submission-greedy[-ensN].tar.gz` in the repo root.

## Security considerations

- Kaggle credentials live at `~/.kaggle/kaggle.json` (needed only for the
  download phase). Never commit them; never make network calls from tests.
- Checkpoints are loaded with `weights_only=True` in the agent; treat `.pt`
  files as untrusted input elsewhere too.
- Data integrity relies on SHA-1 pinning (vocab/archetypes/baselines/ensemble
  members) — don't bypass the mismatch aborts; they exist because the failure
  mode is a silent, plausible-looking wrong model.
- `raw/`, `data/`, `checkpoints*/`, `wandb/`, `*.tar.gz` are gitignored;
  artifacts stay local.
- The RL/vec-env code spawns worker processes and dlopens native libraries —
  review changes there with the same care as network-facing code.

## Development conventions

- **Style**: Python docstrings/comments in English, dense and
  rationale-focused (explain *why*, often with measured numbers — follow the
  existing style). Shell-script comments are largely Traditional Chinese;
  match the surrounding file. Match local naming and structure; minimal,
  scoped diffs.
- **No git mutations** (commit/push/branch) unless explicitly asked.
- Don't over-explain at the end of a task.
- The working tree often carries in-flight changes (e.g. the belief-head
  removal) — check `git status`/`git diff` before assuming file contents match
  the specs.
- **RTK command convention**: this environment rewrites shell commands through
  `rtk` (Rust Token Killer) for token-optimized output. Prefix commands with
  `rtk` where useful — it has dedicated filters for `git`, `pytest`, `cargo`,
  `grep`/`find`/`ls`, `curl`, `docker`, `pnpm`/`npm`, etc., and passes through
  anything it doesn't know unchanged, e.g. `rtk git status`, `rtk pytest`,
  `rtk cargo build`. Meta: `rtk gain` (savings stats), `rtk proxy <cmd>` (raw
  passthrough for debugging). Dedicated file tools (Read/Grep/Glob) remain
  preferable for known paths.
