# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pokémon TCG AI Battle — an imitation-learning pipeline for the Kaggle Pokémon TCG competition. The repo has two main packages and a three-document spec system.

**Spec documents (read in this order):**
- `AGENT_SPEC.md` — authoritative environment I/O reference (Observation/Option/Log schema, the `agent()` contract, search API). Grounded in the real `libcg.so` engine.
- `TRANSFORMER_IL_SPEC.md` — **the authoritative IL spec** (exact tensors, model architecture, training loop, corpus mining phases). Where other docs differ, this one wins.
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

# Full pipeline (Rust engine → mine → shards → train)
./scripts/run_pipeline.sh --skip-download

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

# Train a per-deck specialist (one model per archetype deck)
uv run python -m ptcg_il.cli train --data-dir data --out-dir checkpoints_a2 \
    --archetype-self 2 --batch-size 512 --total-steps 5000 --val-every 250

# Resume from checkpoint
uv run python -m ptcg_il.cli train --resume checkpoints/ckpt-step-0004000.pt

# Eval-only from a checkpoint
uv run python -m ptcg_il.cli train --eval-only --resume checkpoints/ckpt-best.pt
```

## Architecture

### Two-package structure

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
- **Value head is auxiliary** (MSE to game outcome ±1) — free warm-start critic for later RL.

### Test layout

Two test directories:
- `tests/` — root-level tests for `ptcg_mine` (download, episode, stats, sampling, mine integration)
- `python/tests/` — comprehensive tests covering both `ptcg_mine` and `ptcg_il` (model components, featurizer, training loop, QA, live eval)

`pyproject.toml` sets `pythonpath = ["."]` so both packages are importable without installing.

## Working style

- Don't explain too much at the end of a task, unless necessary.
