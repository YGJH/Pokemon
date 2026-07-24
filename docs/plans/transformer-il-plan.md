# Transformer IL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Transformer IL policy (featurizer, PyTorch model, training loop) per `TRANSFORMER_IL_SPEC.md` Appendices A, B, C, and D Phases 3-4.

**Architecture:** A `ptcg_il` package at repo root with three areas: `featurizer` (obs_dict → tensor dict, Appendix A), `model` (nn.Module classes, Appendix B), and `train` (data loading, training loop, checkpointing, eval, Appendix C). The featurizer is pure NumPy (testable without GPU); the model and training use PyTorch + W&B.

**Tech Stack:** Python ≥3.11, PyTorch ≥2.4, wandb, numpy, pyarrow/pandas. The existing `ptcg_mine` package provides vocab.json + archetypes.json + static tables.

## Global Constraints

- **Featurizer output matches Appendix A.4 EXACTLY** — every key, shape, dtype, and fixed position (A.1). `PAD_CARD=0`, `UNKNOWN_CARD=1`, `PAD_ATTACK=0`.
- **Normalizers (A.2):** `HP_N=400`, `RETREAT_N=4`, `ATKDMG_N=350`, `ENERGY_N=12`, `DECK_N=60`, `HAND_N=30`, `TURN_N=50`, `COUNT_N=20`, `PRIZE_N=6`, `BENCH_N=8`, `ATKCOST_N=5`, `DMGCTR_N=20`.
- **Fixed positions (A.1):** `0=CLS`, `1..12=Pokémon-in-play`, `13..42=hand`, `43,44=summary[me,opp]`, `45=stadium`. Pokémon slot order: `1=my active`, `2..6=my bench`, `7=opp active`, `8..12=opp bench`.
- **Tensor shapes:** `L_STATE=46`, `O_MAX=64`, `P_MAX=12`, `H_MAX=30`, `SUM=2`, `D_MAX=60`, `PZ_MAX=6`. `F_CARD=52`, `F_ATK=14`, `F_POKE=26`, `F_HAND=2`, `F_SUM=11`, `F_GLOBAL=93`, `F_OPT=6`.
- **Model dims:** `D_MODEL=256`, `heads=8`, `layers=4`, `ff=1024`, `dropout=0.1`, pre-norm, GELU, `batch_first=True`.
- **Training:** `BATCH=2048`, `PEAK_LR=3e-4`, `WARMUP=1000`, `MIN_LR=3e-5`, `WD=0.01`, `betas=(0.9,0.95)`, `GRAD_CLIP=1.0`, `LABEL_SMOOTH=0.05`, `EMA=0.999`, `LAMBDA_V=0.5`, `W_LOST=0.6`, `EPOCHS=10`.
- **Weighting:** `ALPHA_CTX=0.5`, `ALPHA_ARCH=0.5`. `sample_weight = normalize(w_ctx[sel_ctx] * w_arch[archetype_self] * (1.0 if won else W_LOST))`. Derived at load time, NOT stored in shards.
- **Deck-selection steps (`select is None`) are EXCLUDED** from the featurizer — FIXED_DECK is hardcoded.
- **Split by whole episode** (never within a game). Train/val/test ≈ 96/2/2.
- **Tests:** `uv run pytest` (with `--no-cov`). No network; mock external deps. Use the sample episode fixture at `archive/sample_episodes/80169582.json` and the real engine at `pokemon-tcg-ai-battle/`.

---

## File structure
```
ptcg_il/__init__.py
ptcg_il/featurizer.py        # Core featurize() — obs_dict → tensor dict (Appendix A)
ptcg_il/ref_map.py           # Reference resolution map (A.7)
ptcg_il/shard_writer.py      # Phase 3 shard builder + meta.parquet (D.4)
ptcg_il/qa.py                # Phase 4 QA gates (D.5)
ptcg_il/model/__init__.py
ptcg_il/model/cards.py       # CardEncoder, AttackEncoder (B.1)
ptcg_il/model/embed.py       # TokenEmbedder (B.2)
ptcg_il/model/encoder.py     # Encoder transformer (B.3)
ptcg_il/model/pointer.py     # PointerHead (B.4)
ptcg_il/model/value.py       # ValueHead (B.5)
ptcg_il/model/policy.py      # Policy (B.6) + multi-select (B.7/B.8)
ptcg_il/train/__init__.py
ptcg_il/train/dataset.py     # ShardDataset + collate (C.1–C.4)
ptcg_il/train/loop.py        # Training step, optimizer, schedule (C.5–C.6)
ptcg_il/train/checkpoint.py  # Save/load + submission bundle (C.7)
ptcg_il/train/eval.py        # Offline eval (C.8)
ptcg_il/train/logger.py      # W&B integration (C.9)
ptcg_il/live_eval.py         # Live-engine eval vs baselines (C.8, §5)
ptcg_il/cli.py               # Train CLI entry point
tests/test_featurizer.py
tests/test_shard_writer.py
tests/test_qa.py
tests/test_model_cards.py
tests/test_model_embed.py
tests/test_model_encoder.py
tests/test_model_pointer.py
tests/test_model_policy.py
tests/test_dataset.py
tests/test_train_loop.py
```

### Task 1: Featurizer core — obs_dict → tensor dict (Appendix A)

**Objective.** Implement `featurize(obs_dict, action, vocab, config)` producing the exact Appendix A.4 tensor dictionary for one decision point. This is pure NumPy — no PyTorch dependency.

**Files:**
- Create: `ptcg_il/__init__.py`
- Create: `ptcg_il/featurizer.py`
- Create: `ptcg_il/ref_map.py`
- Create: `tests/test_featurizer.py`

**Interfaces:**
- Consumes: `ptcg_mine.cards` (HP_N, RETREAT_N, etc. normalizers), `ptcg_mine.vocab` (id_to_index remap), engine dataclasses (Observation, SelectData, Option, State, PlayerState, Pokemon, Card)
- Produces: `featurize(obs_dict, action, vocab, config) -> dict[str, np.ndarray]` with ALL Appendix A.4 keys; `build_ref_map(observation) -> dict` for option→token resolution

**Constants in `ptcg_il/featurizer.py`** (A.1–A.2):
```python
# Normalizers
HP_N, RETREAT_N, ATKDMG_N = 400.0, 4.0, 350.0
ENERGY_N, DECK_N, HAND_N = 12.0, 60.0, 30.0
TURN_N, COUNT_N, PRIZE_N, BENCH_N = 50.0, 20.0, 6.0, 8.0
ATKCOST_N, DMGCTR_N = 5.0, 20.0

# Capacities
P_MAX, H_MAX, D_MAX, PZ_MAX = 12, 30, 60, 6
SUM, STAD, CLS = 2, 1, 1
L_STATE = 46   # CLS + P_MAX + H_MAX + SUM + STAD
O_MAX = 64

# Feature dims
F_CARD, F_ATK = 52, 14
F_POKE, F_HAND = 26, 2
F_SUM, F_GLOBAL, F_OPT = 11, 93, 6

# Token-type / owner / zone vocab (int encodings)
TOK_TYPE = {"CLS": 0, "POKE": 1, "HAND": 2, "SUMMARY": 3, "STADIUM": 4}
TOK_OWNER = {"none": 0, "self": 1, "opp": 2}
TOK_ZONE = {"cls": 0, "active": 1, "bench": 2, "hand": 3, "summary": 4, "stadium": 5}

# Pokémon slot layout (rows 1..12)
#  1 = my active, 2..6 = my bench[0..4], 7 = opp active, 8..12 = opp bench[0..4]
```

**Reference map (`ptcg_il/ref_map.py`)** — A.7:
```python
def build_ref_map(observation) -> dict:
    """Returns {(area, playerIndex, index): state_token_row} for all visible entities.
    area: 0=hand, 2=hand(attach src), 4=inPlayArea(active), 5=inPlayArea(bench).
    Covers active/bench for both players + the acting player's hand cards.
    Also stores per-entity card_id for CARD/ENERGY/TOOL_CARD resolution."""
```

- [ ] **Step 1: Write `test_featurizer_pokemon_tokens`** — load the sample episode, find an ACTIVE step with visible Pokémon, call `featurize`. Assert: `poke_card_id.shape == (12,)`, `poke_feat.shape == (12, 26)`, my active at row 0 (index 1), opp active at row 6 (index 7). Verify hp/maxHp/hp_ratio for a known Pokémon. Verify attached-energy histogram. Verify condition flags on active only.

- [ ] **Step 2: Implement Pokémon token featurization** — `_build_poke_tokens(state, yourIndex)` fills `poke_card_id[12]`, `poke_feat[12,26]`. Per A.5: `[hp/HP_N, maxHp/HP_N, hp/maxHp, energies_hist(12), total_energies/ENERGY_N, energy_cards/ENERGY_N, tools/2, evo_depth/2, appearThisTurn, is_active, conditions(5)]`. Face-down opp active → all zeros + pad card id.

- [ ] **Step 3: Write `test_featurizer_hand_and_summary`** — from a later step where hand is visible. Assert `hand_card_id.shape == (30,)`, hand cards at front, pad after. Assert `sum_feat.shape == (2, 11)` with `is_me` flag correct for player 0 vs player 1.

- [ ] **Step 4: Implement hand + summary + stadium** — `_build_hand_tokens(state, yourIndex)`, `_build_summary_tokens(state, yourIndex)`, `_build_stadium_token(state)`. Hand: `card_id + [idx/H_MAX, dup_count/COUNT_N]`. Summary per A.5: `[is_me, deckCount/DECK_N, handCount/HAND_N, bench_len/BENCH_N, benchMax/BENCH_N, prizes/PRIZE_N, discard_len/DECK_N, poisoned, burned, asleep, paralyzed]`. Discard/prize ids are padded id lists, pooled later in the model.

- [ ] **Step 5: Write `test_featurizer_cls_features`** — assert `cls_feat.shape == (93,)`. Verify one-hot encoding of `select.type` and `select.context`. Verify turn, yourIndex, firstPlayer, per-turn flags, min/max counts, condition flags.

- [ ] **Step 6: Implement CLS features** — A.6 layout exactly: `[turn/TURN_N, turn%2, yourIndex, turnActionCount/COUNT_N, firstPlayer_onehot(3), [supporterPlayed,stadiumPlayed,energyAttached,retreated], [my_prizes,opp_prizes]/PRIZE_N, sel_type_onehot(11), sel_context_onehot(49), [minCount,maxCount]/COUNT_N, [remainEnergyCost/ATKCOST_N, remainDamageCounter/DMGCTR_N], my_cond(5), opp_cond(5), [has_contextCard, has_effect], reserved(4)]`.

- [ ] **Step 7: Write `test_featurizer_categorical_attrs`** — assert `tok_type.shape == (46,)`, values correct per fixed position. `tok_owner`: self for rows 1..6+hand, opp for 7..12+summary[1], none for CLS/stadium. `tok_zone`: active/bench/hand/summary/stadium/cls per A.1. `tok_mask` True for filled slots.

- [ ] **Step 8: Implement token categorical attributes** — `_build_tok_attrs()` returns `tok_type[46]`, `tok_owner[46]`, `tok_zone[46]`, `tok_mask[46]`.

- [ ] **Step 9: Write `test_featurizer_options_single_select`** — from a step with single-select (YES_NO context 41 IS_FIRST). Assert `opt_type[64]` with YES/NO types, valid `opt_mask`, `action_idx` has exactly 1 pick. Assert `opt_src_idx`/`opt_tgt_idx` are -1 for constant-type options.

- [ ] **Step 10: Write `test_featurizer_options_multi_select_and_refs`** — test PLAY, ATTACH, EVOLVE, ATTACK, RETREAT option types (find steps with these in the sample). Assert `opt_src_idx` resolves to correct state row, `opt_tgt_idx` set for ATTACH/EVOLVE, `opt_attack_idx` for ATTACK, `opt_card_id` for CARD/ENERGY attachments.

- [ ] **Step 11: Implement option token builder** — `_build_option_tokens(select, ref_map, yourIndex)` per A.4/A.7. For each option, resolve:
  - PLAY(7) → src=hand row, tgt=-1
  - ATTACH(8) → src=ref(area,me,index), tgt=ref(inPlayArea,me,inPlayIndex)
  - EVOLVE(9) → src=ref(area,me,index), tgt=ref(inPlayArea,me,inPlayIndex)
  - ABILITY(10)/DISCARD(11) → src=ref(area,player,index), tgt=-1
  - RETREAT(12) → src=my active row(1), tgt=-1
  - ATTACK(13) → src=my active row(1), tgt=-1, attack_idx set
  - CARD(3)/TOOL_CARD(4)/ENERGY_CARD(5)/ENERGY(6) → src=ref(...) or -1, card_id SET (always, for attachment disambiguation A.7)
  - YES(1)/NO(2)/END(14)/NUMBER(0)/SKILL(15)/SPECIAL_CONDITION(16) → src=-1, tgt=-1, card_id if present
  Fill `opt_scalar[6]` = `[number/COUNT_N, count/COUNT_N, energyIndex/12, toolIndex/2, remainEnergyCost/ATKCOST_N, remainDamageCounter/DMGCTR_N]`.

- [ ] **Step 12: Implement label extraction** — `_build_label(action, select)` returns `action_idx[O_MAX]` (ordered expert picks, -1 padded), `action_len`, `minCount`, `maxCount`. Verify picks are within `[0, len(option))` and distinct.

- [ ] **Step 13: Write `test_featurizer_end_to_end`** — on the sample episode, iterate all ACTIVE decisions with `select is not None`, call `featurize`, verify all keys present with correct shapes/dtypes. Verify deck-selection steps are skipped. No crash on any step.

- [ ] **Step 14: Wire the top-level `featurize()`** — assemble all sub-builders; return the full A.4 dict. Handle `select is None` → raise ValueError (caller must filter).

- [ ] **Step 15: Commit** — `git add ptcg_il/ tests/test_featurizer.py && git commit -m "feat: add featurizer core (obs_dict -> tensor dict per Appendix A)"`

**Done when:** `uv run pytest tests/test_featurizer.py -v` green, all A.4 keys verified on real sample episode.

---

### Task 2: Model modules — nn.Module classes (Appendix B)

**Objective.** Implement all PyTorch modules in `ptcg_il/model/`. Each module is independently testable with synthetic tensors. Add `torch` as a dependency.

**Files:**
- Create: `ptcg_il/model/__init__.py`
- Create: `ptcg_il/model/cards.py` — CardEncoder, AttackEncoder (B.1)
- Create: `ptcg_il/model/embed.py` — TokenEmbedder (B.2)
- Create: `ptcg_il/model/encoder.py` — Encoder (B.3)
- Create: `ptcg_il/model/pointer.py` — PointerHead (B.4)
- Create: `ptcg_il/model/value.py` — ValueHead (B.5)
- Create: `ptcg_il/model/policy.py` — Policy (B.6) + `multiselect_ce` / `select_multi` (B.7/B.8)
- Create: `tests/test_model_cards.py`, `tests/test_model_embed.py`, `tests/test_model_encoder.py`, `tests/test_model_pointer.py`, `tests/test_model_policy.py`

**Interfaces:**
- Consumes: static tables from `ptcg_mine.cards` (card_static_table, attack_static_table), vocab config (V, A), featurizer constants (L_STATE=46, O_MAX=64, etc.)
- Produces:
  - `CardEncoder(V, D=256)` — `forward(ids) -> [...,D]`
  - `AttackEncoder(A, D=256)` — `forward(idx) -> [...,D]`
  - `TokenEmbedder(V, A, D=256)` — `forward(feat_dict) -> [B,L,D]`
  - `Encoder(D=256, heads=8, layers=4)` — `forward(rows, tok_mask) -> [B,L,D]`
  - `PointerHead(D=256, heads=8)` — `forward(h, tok_mask, card_enc, opt_dict, extra_ctx=None) -> (logits[B,O], o[B,O,D])`
  - `ValueHead(D=256)` — `forward(h_cls) -> [B]`
  - `Policy()` — `forward(feat_dict) -> (logits[B,O], value[B])`; `select_multi(policy, feat_dict) -> [B, maxC]`
  - `multiselect_ce(policy, feat_dict) -> scalar loss` (teacher-forced)

**Common MLP helper** (`ptcg_il/model/__init__.py`):
```python
def MLP(in_features, hidden, out_features, dropout=0.0):
    return nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden, out_features))
```

- [ ] **Step 1: `uv add torch wandb`** — add PyTorch and wandb as dependencies (wandb to train group).

- [ ] **Step 2: Write `test_model_cards.py`** — test CardEncoder with synthetic vocab size V=10: (a) verify output shape `[B,*,D]`, (b) PAD id=0 should have non-zero output (id_emb learns it, static is zeros), (c) verify id_emb + static_mlp addition. Same for AttackEncoder. Test with provided static tables from `ptcg_mine.cards.build_static_tables`.

- [ ] **Step 3: Implement `cards.py`** — `CardEncoder(nn.Module)` and `AttackEncoder(nn.Module)` per B.1. `CardEncoder` wraps `nn.Embedding(V,D,padding_idx=0)` + `MLP(52,D,D)` + registered `static` buffer `[V,52]`. `AttackEncoder` is analogous with 14→D MLP and `[A,14]` table.

- [ ] **Step 4: Write `test_model_embed.py`** — build a synthetic featurizer dict (all keys, batch=2) with known values. Test: output shape `[2,46,256]`, CLS at col 0, poke at 1..12, hand at 13..42, etc. Verify that `type_emb`, `owner_emb`, `zone_emb` are added. Verify stadium absent → `no_stadium` param used. Verify context/effect cards added to CLS. Test gradient flows to all params.

- [ ] **Step 5: Implement `embed.py`** — `TokenEmbedder` per B.2. Key details: `masked_sum` for discard pooling (sum / DECK_N); `no_stadium` nn.Parameter; context/effect card embedding added to CLS via `cls_feat[87:89]` flags.

- [ ] **Step 6: Write `test_model_encoder.py`** — test with random `[B,46,256]` input and boolean `tok_mask[B,46]`. Verify output shape `[B,46,256]`, attention respects padding mask (masked positions unchanged by attention?), at minimum verify forward pass completes without error and output is not identical to input.

- [ ] **Step 7: Implement `encoder.py`** — `Encoder` wrapping `nn.TransformerEncoder` with `nn.TransformerEncoderLayer(D, heads, ff, dropout, activation='gelu', batch_first=True, norm_first=True)`.

- [ ] **Step 8: Write `test_model_pointer.py`** — test PointerHead with synthetic encoded state `h[B,46,D]`, token mask, a CardEncoder, and synthetic option dict (opt_type, opt_src_idx, opt_tgt_idx, opt_card_id, opt_attack_idx, opt_scalar, opt_mask). Verify: (a) output logits shape `[B,64]`, (b) masked options have `-1e9` logits, (c) `extra_ctx` adds to all option queries, (d) returned per-option reprs `o` shape `[B,64,D]`, (e) null_token gathered when idx==-1.

- [ ] **Step 9: Implement `pointer.py`** — `PointerHead` per B.4. All sub-modules as fields (no construction in forward). `gather` method handles -1→null row. Cross-attention: queries=options, keys/values=state tokens. Score = Linear(D,1).

- [ ] **Step 10: Write `test_model_policy.py`** — test `Policy.forward()` with synthetic featurizer dict: returns `(logits[B,64], value[B])`. Test `select_multi` with `maxCount>1` options returns `[B, maxC]` indices. Test `multiselect_ce` returns scalar loss with grad. Test label_smoothing=0.05 in single-select CE. Test value head tanh bounds in (-1,1).

- [ ] **Step 11: Implement `policy.py`** — `Policy(nn.Module)` per B.6 wiring embed→encoder→pointer+value. Bind `pointer.card = embed.card`. `select_multi` per B.7 (inference, greedy AR). `multiselect_ce` per B.7 (training, teacher-forced AR, Σ CE over picks). `forward` returns single-select logits + value.

- [ ] **Step 12: Commit** — `git add ptcg_il/model/ tests/test_model_*.py pyproject.toml uv.lock && git commit -m "feat: add model modules (CardEncoder through Policy per Appendix B)"`

**Done when:** `uv run pytest tests/test_model_*.py -v` green, all modules produce correct shapes and gradients flow.

---

### Task 3: Shard writer + meta.parquet (Phase 3 / D.4)

**Objective.** Iterate over kept (episode, player) pairs, call `featurize`, pack into fixed-shape shard `.npz` files, and build `meta.parquet`. This bridges the corpus-mining output (vocab.json, archetypes.json) to the training pipeline.

**Files:**
- Create: `ptcg_il/shard_writer.py`
- Create: `tests/test_shard_writer.py`

**Interfaces:**
- Consumes: `featurize()` from Task 1, `ptcg_mine` (load_episode, validate_episode, teams, rewards, deck_of, canon, assign_archetype), vocab.json / archetypes.json from corpus-mining output
- Produces: `build_shards(config) -> None` writing `data/shards/train-XXXXX.npz`, `val-XXXXX.npz`, `test-XXXXX.npz`, and `data/meta.parquet`

**Key design decisions:**
- Shards hold `SAMPLES_PER_SHARD = 50000` samples each; flush when full.
- Split by `hash(episode_id) % 100`: 0–1 → val, 2–3 → test, 4–99 → train.
- `meta.parquet` columns: `sample_uid, shard, row, episode_id, player, team, archetype_self, archetype_opp, sel_type, sel_ctx, minCount, maxCount, won`.
- `sample_weight` is NOT stored in shards — derived at load time from meta columns.
- Only process kept games: expert team, deck ∈ 𝒟_self, opponent deck ∈ 𝒟_opp, won AND lost games kept.

- [ ] **Step 1: Write `test_shard_writer_kept_filter`** — with a small synthetic episode list (minimal dicts with known teams/decks/archetypes), test that `_is_kept_game(ep, p, experts, self_ids, opp_ids, archetypes)` correctly filters: (a) expert+self+opp → kept, (b) non-expert → dropped, (c) deck not in 𝒟_self → dropped, (d) opponent not in 𝒟_opp → dropped, (e) won game kept, (f) lost game kept.

- [ ] **Step 2: Write `test_shard_writer_active_decisions`** — test `_active_decisions(ep, p)` yields the correct (step_i, obs, action) tuples per the off-by-one rule (IL_SPEC.md §A.3). Verify: drops `select is None` steps, drops final ACTIVE decision, correct pairing.

- [ ] **Step 3: Write `test_shard_writer_split`** — test that `_split_of(episode_id)` assigns train/val/test correctly (hash-based), deterministic, no overlap.

- [ ] **Step 4: Write `test_shard_writer_build`** — with a small synthetic episode set (3 episodes, known teams/decks), run `build_shards` to a temp directory. Assert: shard files created, `meta.parquet` exists, each sample has all expected columns, shapes consistent, can load shard with `np.load` and verify keys. Test that re-running overwrites (idempotent).

- [ ] **Step 5: Implement `shard_writer.py`** — functions:
  - `_is_kept_game(ep, p, experts, self_ids, opp_ids, archetypes) -> bool`
  - `_active_decisions(ep, p) -> Iterator[tuple[int, dict, list[int]]]` (step_i, obs, action)
  - `_split_of(episode_id) -> str` (train/val/test)
  - `build_shards(config) -> None` — orchestrator: load vocab/archetypes, iterate raw episodes, filter, featurize, append to shard buffers, flush, write meta.parquet

- [ ] **Step 6: Commit** — `git add ptcg_il/shard_writer.py tests/test_shard_writer.py && git commit -m "feat: add Phase 3 shard writer and meta.parquet builder"`

**Done when:** `uv run pytest tests/test_shard_writer.py -v` green, shards are valid .npz files with all Appendix A keys, meta.parquet has correct schema.

---

### Task 4: Dataset + training loop (Appendix C)

**Objective.** Implement the PyTorch Dataset, training step, checkpointing, offline eval, and W&B logging. This is the training pipeline that consumes shards + meta.parquet and produces a trained Policy.

**Files:**
- Create: `ptcg_il/train/__init__.py`
- Create: `ptcg_il/train/dataset.py` — ShardDataset + collate_fn (C.1–C.4)
- Create: `ptcg_il/train/loop.py` — Training step, optimizer/schedule setup, main loop (C.5–C.6)
- Create: `ptcg_il/train/checkpoint.py` — Save/load checkpoints + submission bundle (C.7)
- Create: `ptcg_il/train/eval.py` — Offline eval (per-context accuracy, value MSE/AUC) (C.8)
- Create: `ptcg_il/train/logger.py` — W&B init, logging, artifact upload (C.9)
- Create: `tests/test_dataset.py`
- Create: `tests/test_train_loop.py`

**Interfaces:**
- Consumes: `Policy` from Task 2, shards + meta.parquet from Task 3
- Produces: `train(config) -> Path` (best model path)

**Key design:**
- `ShardDataset`: mmap-based, reads shard slices + joins with meta.parquet for weights. Returns single samples (NOT batches).
- `collate_fn`: stacks dict-of-tensors along batch dim, derives `encoder_padding_mask = ~tok_mask`.
- `sample_weight` derived at load time from meta columns per C.3 formula (not from shards).
- Mixed precision: `torch.amp.autocast('cuda', dtype=torch.bfloat16)`.
- Optimizer: `AdamW` with per-param no-decay set (biases, LayerNorm, Embedding, null_token, no_stadium).
- Schedule: linear warmup → cosine decay.
- EMA: `decay=0.999`, evaluate/ship EMA weights.
- Offline eval: top-1/top-3 accuracy per sel_type and per sel_ctx, multi-select exact-set match, value MSE/AUC.
- Checkpoint: save `{step, model_state_dict, ema_state_dict, optimizer_state_dict, scheduler_state_dict, rng_state}`.
- W&B: init with config dict, log scalars every 50 steps, tables every 1000 steps, artifacts for best/last checkpoints.

- [ ] **Step 1: Write `test_shard_dataset`** — create synthetic shards (2 shards × 10 samples) + a mini meta.parquet. Test: (a) `ShardDataset.__getitem__` returns correct sample dict with all keys, (b) `sample_weight` computed correctly from meta columns, (c) `__len__` matches meta rows, (d) works with DataLoader + collate_fn producing batched tensors with correct shapes.

- [ ] **Step 2: Implement `dataset.py`** — `ShardDataset(torch.utils.data.Dataset)`: __init__ takes shard_dir, meta_path, weight_cfg (ALPHA_CTX, ALPHA_ARCH, W_LOST). Pre-computes per-row weights from meta. `__getitem__` reads from mmap'd shard, adds sample_weight, returns dict of torch tensors. `collate_fn` stacks, derives mask.

- [ ] **Step 3: Write `test_train_step`** — using a tiny Policy and a batch of 4 synthetic samples (maxCount==1 and maxCount>1). Test: (a) forward pass produces loss scalar with grad, (b) loss = weighted_CE + LAMBDA_V * value_MSE, (c) single-select and multi-select loss both compute, (d) label_smoothing applied in single-select.

- [ ] **Step 4: Implement `loop.py`** — `create_optimizer(policy, cfg)`, `create_scheduler(opt, cfg)`, `TrainState` dataclass (step, epoch, best_val_metric), `train_step(policy, batch, opt, scaler, cfg) -> dict` (loss scalars), `train_epoch(...)`, `train(config) -> Path` main entry point.

- [ ] **Step 5: Implement `checkpoint.py`** — `save_checkpoint(path, policy, ema, opt, sched, step, rng)`, `load_checkpoint(path, policy, ...)`, `build_submission_bundle(checkpoint_path, out_dir)` → copies main.py, deck.csv, cg/, weights.pt, vocab.json, archetypes.json.

- [ ] **Step 6: Implement `eval.py`** — `offline_eval(policy, val_loader, cfg) -> dict` returning: top-1/top-3 per sel_ctx (table), per sel_type (table), macro/micro aggregates, multi-select exact-set match, value MSE/AUC. Must use `torch.no_grad()` and EMA weights.

- [ ] **Step 7: Implement `logger.py`** — `init_wandb(cfg)`, `log_train_step(step, metrics)`, `log_val_eval(step, eval_results)`, `log_checkpoint_artifact(path, alias)`. All logging uses global `step` as x-axis. W&B config includes full hyperparameters + artifact provenance.

- [ ] **Step 8: Commit** — `git add ptcg_il/train/ tests/test_dataset.py tests/test_train_loop.py && git commit -m "feat: add training loop, dataset, checkpointing, eval, and W&B logging"`

**Done when:** `uv run pytest tests/test_dataset.py tests/test_train_loop.py -v` green, training loop runs end-to-end on synthetic data.

---

### Task 5: QA gates + live eval + CLI (Phase 4 + Section 5 + wiring)

**Objective.** Implement the Phase 4 QA gates (D.5), live-engine eval harness, and the top-level CLI that wires everything together (mine → featurize → train).

**Files:**
- Create: `ptcg_il/qa.py` — QA gates (D.5)
- Create: `ptcg_il/live_eval.py` — Live-engine eval (Section 5, C.8)
- Create: `ptcg_il/cli.py` — Train CLI entry point
- Create: `tests/test_qa.py`

**Interfaces:**
- Consumes: All previous tasks
- Produces: `run_qa_checks(shard_dir, meta_path, vocab, ...) -> dict`; `live_eval(policy, deck, n_games, opponents) -> dict`; CLI `ptcg_il.train` subcommand

- [ ] **Step 1: Write `test_qa_checks`** — using synthetic shards + meta: test each QA gate:
  (a) Coverage: every kept-game card id is in vocab (subset check),
  (b) Variable-length audit: count `minCount < maxCount` samples,
  (c) Attachment-collision audit: count options sharing both opt_src_idx AND opt_card_id,
  (d) Label sanity: action_idx within bounds, distinct, correct length,
  (e) Outcome balance: won/lost counts roughly balanced,
  (f) Deck legality: 60 cards, ≤4 copies per non-basic-energy id, ≤1 ACE SPEC, ≥1 basic Pokémon.

- [ ] **Step 2: Implement `qa.py`** — `run_qa_checks(...) -> dict` with pass/fail per check + detailed counts. `assert` on critical failures (label sanity, coverage). Print warnings for balance/imbalance.

- [ ] **Step 3: Implement `live_eval.py`** — `LiveEvaluator` class: loads EMA policy, wraps as `agent(obs_dict)` per Section 6 reference forward pass, runs games via `cg.game.battle_start/battle_select` in a process pool. Reports win-rate ± Wilson interval vs random agent, frozen checkpoint, and search planner. Implements OOV rate tracking.

- [ ] **Step 4: Implement `cli.py`** — `argparse` CLI: `ptcg-il train` subcommand with all hyperparameters from C.10, `--data-dir`, `--out-dir`, `--resume`, `--eval-only`, `--live-eval`. Wires: load config → build datasets → create model → train → save best checkpoint.

- [ ] **Step 5: Commit** — `git add ptcg_il/qa.py ptcg_il/live_eval.py ptcg_il/cli.py tests/test_qa.py && git commit -m "feat: add QA gates, live eval harness, and train CLI"`

**Done when:** `uv run pytest tests/test_qa.py -v` green; `uv run python -m ptcg_il.cli train --help` prints usage.

---

---

## Execution order

```
Task 1 (Featurizer) ──┬──▶ Task 3 (Shard writer) ──▶ Task 5 (QA + live eval + CLI)
                      │
Task 2 (Model) ───────┼──▶ Task 4 (Training loop) ──▶ Task 5
                      │
                      └── (Tasks 1 and 2 are independent; Tasks 3 and 4 depend on Tasks 1 and 2 respectively;
                           Task 5 depends on Tasks 3 and 4)
```

**Sequential execution** (subagent-driven): Task 1 → Task 2 → Task 3 → Task 4 → Task 5.
Tasks 1 and 2 could run in parallel since they share no code, but sequential is safer for review coherence.

---

## Self-Review

**1. Spec coverage:**
- Appendix A (Featurizer Contract): Task 1 — all A.1–A.9 covered
- Appendix B (Model Modules): Task 2 — all B.0–B.9 covered
- Appendix C (Training Loop): Task 4 — all C.1–C.12 covered
- Appendix D Phase 3 (Featurize pass): Task 3 — D.4 covered
- Appendix D Phase 4 (QA gates): Task 5 — D.5 covered
- Section 5 (Live eval): Task 5 — live_eval.py
- Section 6 (Reference forward pass): Task 5 — live_eval.py wraps agent()
- Section 7 (Decisions): embodied in Global Constraints (W_LOST, LAMBDA_V, vocab decoupling)
- Section 8 (Build order): this plan is the build order

**2. Placeholder scan:** Clean — all steps have concrete code/commands.

**3. Type consistency:**
- `featurize()` returns `dict[str, np.ndarray]` — consumed by `ShardDataset` (loads from .npz) and `TokenEmbedder.forward()` (expects dict with batch dim)
- `Policy.forward(feat_dict)` — the featurizer dict with batch dim added by collate
- `PointerHead.forward(h, tok_mask, card_enc, x, extra_ctx)` — `x` is the option sub-dict from the featurizer output
- `multiselect_ce(policy, feat_dict)` — uses policy.pointer internals for teacher-forcing
- All dim constants match Appendix A values
- Normalizer constants shared between `ptcg_mine.cards` and `ptcg_il.featurizer`
