# Transformer Imitation-Learning Spec — Pokémon TCG AI Battle

Behavior cloning of **top agents** on a **fixed set of meta decks**, with an entity-transformer
encoder + pointer action head + **cross-turn history GRU** on the CLS token + **BeliefModule** over
the logs stream with supervised opponent-card prediction heads. Builds on `AGENT_SPEC.md` (env I/O)
and `IL_SPEC.md` (data format + the verified off-by-one pairing rule).

Design informed by the `entity-transformer-policies` and `micro-step-action-factorization` skills.

---

## 1. Scope & data selection

### 1.1 Top-agent filtering (not all winners)
Replays carry only `rewards = [±1]` and `info.TeamNames` (e.g. `shikisoukan`, `cocoaAI`) — **no
per-agent skill score in the JSON**. So we build our own expert set from the sampled corpus:

1. Sample N episodes across recent, high-`top_avg_score` days (from `archive/manifest.csv`).
2. Aggregate per **team name**: `games`, `wins`, `win_rate = wins/games`.
3. **Expert set = top-K teams** by `win_rate` with `games ≥ G_min` (e.g. K≈10, G_min≈50). Optionally
   cross-check against the public leaderboard names.
4. **Training samples = ALL ACTIVE decisions made by an expert team**, in won *and* lost games, with
   lost-game decisions down-weighted (`W_LOST ≈ 0.6`, folded into `sample_weight`). Rationale below.

> Rationale: we already selected experts by team-level win-rate, so their moves are good *regardless
> of the single game's outcome*. Game outcome is a **noisy proxy for move quality**: a won game can
> contain bad moves (won despite them), a lost game usually contains good moves (lost to prize/coin
> variance). Filtering those experts' decisions *again* by per-game result discards ~half the data for
> marginal signal **and** starves the policy of even/losing board states — exactly the states a
> sub-expert policy drifts into at inference, worsening BC covariate shift. So keep both outcomes and
> down-weight losses instead of dropping them. (Wins-only is retained as an **ablation**, not the
> default — see §7. This also un-blocks the value head: with both outcomes, `value_target ∈ {+1,-1}`
> is informative from v1, giving the free warm-start critic `AGENT_SPEC.md` §8 wants.)

### 1.2 Fixed-deck restriction (bounded card universe) — **M = 3–8**
The deck is the `action` at the `select=None` step (a 60-card id list). Canonicalize each deck to a
**sorted multiset key**, then **cluster near-duplicates into archetypes**: decks within a small edit
distance (e.g. Jaccard ≥ 0.9 on the 60-card multiset, i.e. ≤ ~3 card swaps) are the same archetype.
This stops "most frequent deck" from being fragmented by 1–2-card tech variants. Rank archetypes by
frequency **across all decks in the sampled corpus**.

Separate two roles — critical because a submission pilots **one** fixed deck:
- **𝒟_self = 3–8 candidate archetypes we might submit.** We train the policy to pilot *all* of them,
  which (a) multiplies expert data and (b) lets us later compare candidate decks under the *same*
  trained policy and pick the best. **Submission `deck.csv` = the single 𝒟_self archetype with the
  best measured live win-rate** (a representative decklist of that cluster). No deck head is learned.
- **𝒟_opp = top-M (≈3–8) frequent opponent archetypes.** Do **not** force mirror-only; the policy must
  handle the common matchups it will actually meet.

**Training-game filter (v1):** keep an expert game (won or lost, §1.1) if `expert_deck ∈ 𝒟_self`
**and** `opponent_deck ∈ 𝒟_opp`.

**Decouple the game filter from the vocab.** The archetype filter above defines clean *training
targets*; it must **not** define the id-embedding vocab, or live inference drowns in out-of-vocab
opponent cards (the real meta is far wider than M=3–8 opponent archetypes). Build the vocab from a
**wider card set**: every card appearing in *any* expert game in the sampled corpus (not just kept
games), or the top-`N_VOCAB` cards by corpus frequency. Target **~300–500 ids** (vs ~2,100), which
costs almost nothing in params but sharply cuts live OOV. Track the resulting live OOV rate (next
paragraph) and widen `N_VOCAB` until it is small.

**Out-of-vocab robustness:** at inference the opponent may still play a card outside the vocab. Keep
card features graceful: `card_feat(id) = Embedding(id if in-vocab else UNKNOWN) + MLP(static
features)`, where the static-feature MLP (hp, type, stage, weakness, retreat, …) is computable for
**any** of the ~2,100 ids via `all_card_data()`. So an unseen opponent card still gets meaningful
type/hp/stage features; only its learned id-embedding falls back to a shared `UNKNOWN` slot. **Metric
(D.5/C.8):** measure the **fraction of live opponent cards that are OOV** — this number, not
convenience, drives `N_VOCAB`. The static-feature MLP is a fallback, not the primary opponent
representation.

**Payoff of a compact vocabulary:** the id-embedding table stays small and every in-vocab card is
seen often. Bulk piles (discard, revealed prizes) are handled as **mask-pooled id-embeddings** — a
permutation-invariant bag over the padded id list (A.4/B.2), reusing the shared `CardEncoder` — rather
than long per-card token sequences. (A raw count-vector over `|𝒟vocab|` is a viable alternative, but
does not extend cleanly to the *wider* §1.2 vocab and would not share the card backbone; the pooled-id
form is the contract.)

### 1.3 Sample = one decision point
Per the verified rule (`IL_SPEC.md` §A.3):
```
for i in range(len(steps)-1):              # NB: drops the final step's action (see below)
  for p in (0,1):
    if (steps[i][p].status == "ACTIVE" and team(p) in EXPERTS
        and deck(p) in 𝒟_self and deck(1-p) in 𝒟_opp):     # won/lost both kept (§1.1)
        yield  obs   = steps[i][p].observation          # the obs_dict
               action= steps[i+1][p].action             # expert's chosen indices
               weight_scale = 1.0 if won(p) else W_LOST  # §1.1 loss down-weight
```
`range(len(steps)-1)` intentionally **drops the last ACTIVE decision** (its action would live at
`steps[len]`), which is usually the game-ending attack — an acceptable loss, noted so it is not
mistaken for a bug. Split **by whole episode** into train/val/test (never within a game). Balance
sampling across the 𝒟_self archetypes so the policy learns to pilot each, not just the most common
one. `search_begin_input` is ignored in this IL version (the engine planner is a later-spec concern).

---

## 2. Feature encoding

Build the card & attack feature tables **once** from `all_card_data()` / `all_attack()`. The learned
id-embedding table covers the vocab `𝒟_self ∪ 𝒟_opp` (plus `PAD` and `UNKNOWN`); the static-feature
MLP is defined for all ~2,100 ids so out-of-vocab opponent cards degrade gracefully (§1.2).

`card_feat(id) = Embedding(id) ⊕ MLP([hp, retreatCost, stage(basic/1/2 1hot), cardType 1hot,
energyType 1hot, weakness 1hot, resistance 1hot, isEx, isMegaEx, isTera, isAceSpec, #attacks,
Σattack.damage_norm])`. Attack table: `attackId → [damage_norm, energy-cost multiset(12), #flags]`.

### 2.1 Token layout (state tokens fed to the encoder)
Fixed positions, padded, packed to front (see `entity-transformer-policies`):

| Pos | Token | Count | Notes |
|---|---|---|---|
| 0 | **CLS** | 1 | global read (value + option-head context) |
| 1..12 | **Pokémon in play** | ≤12 | my active(1)+bench(≤5), opp active(1)+bench(≤5) |
| 13..~27 | **My hand** cards | ≤ H_max (pad, e.g. 15) | opp hand is hidden → not tokenized |
| next 2 | **Player summaries** | 2 | one per player (discard pile pooled into this token) |
| next 1 | **Stadium** | 1 | 0/1 card |

Total ≈ 31 tokens → attention is trivially cheap; packing optional.

**Embedding per token** = `type_emb[tok_type] + owner_emb[self/opp/none] + zone_emb[active/bench/hand/
summary/stadium/cls] + feat_proj(features)`. (No RoPE — no spatial board; zone carries structure.)

**Pokémon-in-play features** (attachments are features, **not** tokens):
`card_feat(id), hp, maxHp, hp_ratio, energies count-per-type(12), #energyCards, #tools,
evolution_depth(#preEvolution), appearThisTurn, condition flags(poison/burn/sleep/paralyze/confuse,
active only)`.

**Hand token features**: `card_feat(id)` + owner=self + zone=hand.

**Player-summary features**: scalars `deckCount, handCount, benchMax, prize_remaining, discard_count,
is_me` (exact list in Appendix A.5, `sum_feat`). The **discard pile itself** is not a count-vector on
this token; it is carried as a padded id list (`discard_ids`, A.4) and **pooled into the summary
token** in the model (`B.2`). Revealed prizes are carried the same way (`prize_ids`, A.4). (Earlier
drafts described these as count-vectors over `|𝒟vocab|`; the contract in Appendix A supersedes that —
pooled id-embeddings generalize across the *wider* vocab §1.2 and reuse the shared `CardEncoder`.)

**CLS / global features** (concat, projected in): `turn, turn%2, am_i_first, turnActionCount,
toss_undecided, supporterPlayed, stadiumPlayed, energyAttached, retreated, my_prizes_left,
opp_prizes_left`, **plus the current decision conditioning**: `select.type 1hot(11),
select.context 1hot(49), minCount, maxCount, remainEnergyCost, remainDamageCounter`, and
`card_feat(contextCard)`, `card_feat(effect)` when present. This conditioning is essential — the same
board demands different actions depending on *what is being asked*.

### 2.2 Encoder
Transformer encoder over the state tokens: `d_model=256, layers=4–6, heads=8, GELU, pre-norm`,
padding mask on inactive slots. Output `h ∈ [B, L, d]`. `h[:,0]` = CLS.

---

## 3. Action head — pointer over `select.option`

Options are variable-length, content-addressed, and **already legal** (engine prunes), so no global
action space and no extra masking. Build one **option token** per option by resolving its references
into the encoded state tokens:

- `PLAY{index}` → my-hand token `index`.
- `ATTACH{area,index,inPlayArea,inPlayIndex}` → source card token ⊕ target Pokémon token.
- `EVOLVE{…}` → source(hand) ⊕ target Pokémon token.
- `ATTACK{attackId}` → attack_feat(attackId) ⊕ my active token.
- `ABILITY/DISCARD/RETREAT{…}` → referenced Pokémon token.
- `CARD/ENERGY{area,index,playerIndex}` → resolved token, or `card_feat(id)` if it points into
  `select.deck` / a non-tokenized zone.
- `YES/NO/END/NUMBER{number}` → learned constant embeddings ⊕ scalar.

```
opt_j = MLP( optType_emb[type_j]  ⊕  src_tok_j  ⊕  tgt_tok_j
           ⊕ scalars_j[number,count,energyIndex,toolIndex,remainEnergyCost]  ⊕  h_CLS )
# cross-attention: queries = options, keys/values = state tokens h
opt_ctx_j = CrossAttn(opt_j, h, h)
logit_j   = w · MLP(opt_ctx_j)                       # scalar per option
p(option) = softmax_j(logit_j)                       # over exactly len(option)
```
A reference→token index map is built during featurization (`(area, playerIndex, index) → position`).

### 3.1 Single vs multi-select
- **Single-select** (`minCount=maxCount=1`, the common case): CE of the pointer softmax vs the expert
  index.
- **Multi-select** (`maxCount>1`, e.g. setup bench, discard k): **autoregressive pick-without-
  replacement** — after each pick, mask the chosen option and add a "picked-so-far" pooled embedding
  to the query context; repeat for the expert's length (respect `minCount`). Loss = Σ per-pick CE.
  Teacher-force the expert order (order matters for `SKILL_ORDER`, `EVOLVE` source→target).

### 3.2 Value head (aux)
`value = tanh(Linear(h_CLS))` → predict acting player's episode reward (±1). MSE aux loss, active from
v1: because training keeps both won and lost expert games (§1.1), the target actually varies (`{+1,-1}`)
and the head learns a real critic — free from the data and the warm-start critic for later RL
(`AGENT_SPEC.md` §8). (In the wins-only ablation the target is constant and the head is disabled.)

---

## 4. Losses & optimization
```
L = CE_policy(per-context weighted)  +  λ_v · L_value
```
- **Per-context weighting**: MAIN(0) & YES_NO(9) dominate; up-weight rare contexts (SKILL_ORDER,
  DAMAGE_COUNTER_ANY, …) or class-balanced sample. Track per-context counts.
- Optimizer AdamW, lr ~3e-4 cosine, warmup, weight decay 0.01, grad-clip 1.0, label smoothing 0.05.
- Batch = **decision points** (shuffled across games), e.g. 1–4k per step; ragged option counts →
  pad option dim + option-mask in the softmax.
- Mixed precision; `torch.compile` (keep dense tensors — no NestedTensor / data-dependent control
  flow, per the entity-transformer skill).

---

## 5. Metrics & validation
- **Top-1 / top-3 option accuracy, per `SelectType` and per `SelectContext`** (not just aggregate —
  MAIN accuracy hides rare-context failure).
- **Value AUC** vs game outcome (meaningful from v1 now that both outcomes are trained, §1.1).
- **Live OOV rate**: fraction of opponent cards at live inference that fall to `UNKNOWN` (drives
  `N_VOCAB`, §1.2).
- **The real bar — play the live engine** (`cg.game`): submitted `agent()` vs
  (a) the sample **random** agent — a floor only; beating it proves little,
  (b) a **frozen previous checkpoint** — progress signal,
  (c) the **built-in `search_begin/step` planner** (`AGENT_SPEC.md` §5) as a scripted opponent — the
  honest, non-trivial bar.
  Report win-rate over ≥500 games with 𝒟 decks.
  > ⚠️ Do **not** evaluate "vs a held-out expert *replay*": an interactive game diverges from the
  > recording the moment our policy deviates, so the replay's recorded actions become
  > illegal/meaningless. Playing a policy requires the expert's *policy*, which the replays don't give
  > us — hence the search planner stands in as the strong baseline.
- Sanity: chosen indices always within `[minCount,maxCount]`, distinct, in range (engine err 4/5/6).

---

## 6. Reference forward pass (submission `agent()`)
```python
def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return FIXED_DECK                     # highest-win-rate deck in 𝒟
    h    = encoder(build_state_tokens(obs))   # §2
    opts = build_option_tokens(obs.select.option, h)   # §3
    lg   = pointer_head(opts, h)
    if obs.select.maxCount == 1:
        return [int(lg.argmax())]
    return autoregressive_topk(lg, obs.select.minCount, obs.select.maxCount)  # §3.1
```
Ship `main.py` + `deck.csv` (FIXED_DECK) + `cg/` + weights; loaded from `/kaggle_simulations/agent/`.

---

## 7. Decisions
**Locked:**
- **Expert data = ALL decisions of top agents**, won *and* lost games, lost down-weighted
  `W_LOST≈0.6` (§1.1). Team-level win-rate already selects skill; per-game outcome is a noisy quality
  proxy and dropping losses worsens covariate shift. *Wins-only is the ablation, not the default.*
- **Value head active from v1**: with both outcomes `value_target ∈ {+1,-1}` is informative, so
  `LAMBDA_V=0.5` (warm-start critic for RL, `AGENT_SPEC.md` §8).
- **Vocab decoupled from the archetype filter** (§1.2): games are filtered by archetype for clean
  targets, but the id-embedding vocab is built from a *wider* card set (~300–500 ids) to keep live OOV
  low.
- **Deck sets sized M = 3–8**: `𝒟_self` = 3–8 candidate archetypes (train to pilot all; submit the
  single best-measured one), `𝒟_opp` = top ≈3–8 opponent archetypes (no mirror-only filter).
- **Belief module + cross-turn GRU active from v1**: `BeliefModule` (GRU over log entries) and
  `history_gru` (GRUCell on CLS token) provide history-aware state encoding. `BeliefHeads` with
  supervised opponent-card prediction (archetype, deck, hidden, hand) adds an auxiliary belief loss.
- **Multi-select memory uses GRU cell** (`PointerHead.msgru`): the "already-picked" context is
  maintained by a learned GRUCell rather than an additive running sum, and a **STOP column**
  (option type 17) handles variable-length termination unconditionally.

**Still to confirm (defaults in bold):**
1. **Expert set size / threshold**: **K=10 teams, G_min=50 games, rank by win-rate.**
2. **Archetype clustering threshold**: **Jaccard ≥ 0.9** on the 60-card multiset (≤ ~3 swaps).
3. **Model size**: **d_model=256, 4 layers, 8 heads, card-embed 128** (~10–20M params).
4. **Hand token cap H_max** and in-play cap (12) — bench ≤5 by rules; confirm no format lets bench
   exceed 5, and pick H_max from observed max hand size in the corpus.
5. **Corpus size**: how many episodes / days to sample and download locally (disk + Kaggle limits).

---

## 8. Build order
1. **Corpus tools**: episode sampler/downloader → parser (off-by-one) → team win-rate table →
   expert set; deck-frequency miner → 𝒟 and FIXED_DECK. *(get the data / decks first)*
2. **Featurizer**: obs_dict → (state tokens, option tokens, ref-index map, action label); vocab from
   𝒟. Cache to tensor shards.
3. **Model**: embeddings + encoder + pointer head + value head.
4. **Train loop**: weighted CE + value aux; per-context metrics; **W&B logging from step 1** (train
   scalars, per-context val table, live win-rate — Appendix C.9). Metrics only: checkpoints are
   never uploaded.
5. **Eval harness**: live-engine win-rate vs baselines.
6. Later specs: `logs` belief module (GRU/attention), RL fine-tune (PPO self-play / AlphaZero with the
   engine's `search_begin/step`).
```

---

# Appendix A — Featurizer Contract (exact tensors)

The **exact** output of `featurize(obs_dict) -> dict[str, ndarray]`. One call = one decision point.
The featurizer emits **integer index maps + raw float features + masks**; it never gathers encoded
vectors (option tokens reference state-token *positions*, resolved by the model after the encoder).
All floats are `float32`, all indices `int64`, all masks `bool`. `PAD_CARD = 0`, `UNKNOWN_CARD = 1`;
real card ids are remapped to a contiguous `[2 .. V-1]` via the mined vocab. `PAD_ATTACK = 0`.

## A.1 Constants (fixed at build time)

| Name | Value | Source / reason |
|---|---|---|
| `D_MODEL` | 256 | encoder width |
| `N_ENERGY` | 12 | `EnergyType` 0–11 |
| `N_CARDTYPE` | 7 | `CardType` 0–6 |
| `N_STAGE` | 3 | basic / stage1 / stage2 |
| `N_SELTYPE` | 11 | `SelectType` 0–10 |
| `N_SELCTX` | 49 | `SelectContext` 0–48 |
| `N_OPTTYPE` | 18 | `OptionType` 0–16 + STOP=17 |
| `N_COND` | 5 | poison/burn/sleep/paralyze/confuse |
| `P_MAX` | 12 | ≤(1 active+5 bench)×2 players |
| `H_MAX` | 30 | hand cap (obs max 8; draw effects → pad headroom; confirm vs corpus) |
| `SUM` | 2 | one summary token per player |
| `STAD` | 1 | stadium |
| `CLS` | 1 | global token |
| `L_STATE` | 46 | `CLS+P_MAX+H_MAX+SUM+STAD` = 1+12+30+2+1 |
| `O_MAX` | 64 | option cap (≤60 when picking from a full deck) |
| `D_MAX` | 60 | discard-pile ids per player (pooled, padded) |
| `PZ_MAX` | 6 | prize slots per player |
| `V` | `N_VOCAB`+2 | top-`N_VOCAB` corpus cards (§1.2, ~300–500, **not** just 𝒟) + PAD + UNKNOWN |
| `A` | \|attacks\|+1 | attacks appearing in vocab cards + PAD |

**Fixed token positions** (row index within the `L_STATE` sequence):
`0 = CLS`; `1..12 = Pokémon-in-play`; `13..42 = hand`; `43,44 = player-summary[me,opp]`;
`45 = stadium`. Empty slots are padded and masked.

**Pokémon-in-play slot convention** (deterministic, so option refs resolve): `1 = my active`,
`2..6 = my bench[0..4]`, `7 = opp active`, `8..12 = opp bench[0..4]`.

## A.2 Normalization constants

`HP_N=400, RETREAT_N=4, ATKDMG_N=350, ENERGY_N=12, DECK_N=60, HAND_N=30, TURN_N=50, COUNT_N=20,
PRIZE_N=6, BENCH_N=8, ATKCOST_N=5, DMGCTR_N=20`. Every raw magnitude below is the value divided by its
`*_N`, clipped to `[0,1]` (counts) or `[-1,1]` (signed). Booleans → `{0.,1.}`.

## A.3 Card & attack feature functions (computable for **all** 1267 ids)

`card_static(id) -> float32[F_CARD_STATIC=52]` — concatenation:
| slice | dim | contents |
|---|---|---|
| `[0]` | 1 | `hp/HP_N` |
| `[1]` | 1 | `retreatCost/RETREAT_N` |
| `[2:9]` | 7 | `cardType` one-hot |
| `[9:12]` | 3 | `[basic, stage1, stage2]` |
| `[12:24]` | 12 | `energyType` one-hot |
| `[24:36]` | 12 | `weakness` one-hot (all-zero = none) |
| `[36:48]` | 12 | `resistance` one-hot (all-zero = none) |
| `[48:52]` | 4 | `[ex, megaEx, tera, aceSpec]` |

`attack_static(attackId) -> float32[F_ATK_STATIC=14]`:
`[damage/ATKDMG_N] (1) ⊕ energy-cost histogram over N_ENERGY (12), each `count/ATKCOST_N` ⊕ [len(energies)/ATKCOST_N] (1)`.

> The histogram is normalized, matching `poke_feat[3:15]`. It was originally specified as raw counts
> while `poke_feat` divided by `ENERGY_N`, which put values up to 5.0 in 36 of the 94 card-feature
> dims next to everything else in [0, 1]. `ATKCOST_N` (not `ENERGY_N`) so the histogram and the
> `len(energies)` term that summarises it share one scale.

> ⚠️ **Verify the 12-wide energy space covers every cost symbol, including *colorless*.** The
> `energy-cost histogram` here and the attached-energy histogram in `poke_feat[3:15]` both assume
> `N_ENERGY=12` spans all `EnergyType` values used in costs. Attack costs use colorless heavily; if
> colorless is a distinct `EnergyType` index it must be one of the 12, and if it is a separate symbol
> outside the enum, widen `N_ENERGY`. Assert `max(EnergyType used in any cost) < N_ENERGY` at build time.
> (Also reconcile the id count: this header says 1267, §2 says ~2,100 — pin the true `all_card_data()`
> length in `vocab.json` and use it consistently.)

Model-side (not featurizer): `card_embed(id) = IdEmb[remap(id)] (256) + CardStaticMLP(card_static) (256)`;
`attack_embed = AtkIdEmb[·] (256) + AtkStaticMLP(attack_static) (256)`. Featurizer supplies the ids and
the static rows (or the model looks them up from a prebuilt `[V,52]` / `[A,14]` table — preferred, so
the sample only carries **ids**, not repeated static rows).

## A.4 Per-sample output tensors (the contract)

### State — card identity & structural
| key | shape | dtype | notes |
|---|---|---|---|
| `poke_card_id` | `[P_MAX]` | int64 | remapped id per slot; `PAD_CARD` if slot empty/face-down |
| `hand_card_id` | `[H_MAX]` | int64 | my hand only; pad |
| `stadium_card_id` | `[1]` | int64 | `PAD_CARD` if none |
| `context_card_id` | `[1]` | int64 | `select.contextCard` (else PAD) |
| `effect_card_id` | `[1]` | int64 | `select.effect` (else PAD) |
| `discard_ids` | `[SUM, D_MAX]` | int64 | per player, padded; model pools `Σ card_embed` |
| `discard_mask` | `[SUM, D_MAX]` | bool | valid discard entries |
| `prize_ids` | `[SUM, PZ_MAX]` | int64 | **revealed** prizes only; face-down → PAD |

### State — dense features
| key | shape | dtype | contents |
|---|---|---|---|
| `poke_feat` | `[P_MAX, F_POKE=29]` | float32 | see A.5 |
| `hand_feat` | `[H_MAX, F_HAND=2]` | float32 | `[idx/H_MAX, dup_count_in_hand/COUNT_N]` |
| `sum_feat` | `[SUM, F_SUM=11]` | float32 | see A.5 |
| `cls_feat` | `[F_GLOBAL=95]` | float32 | see A.6 (holds the select conditioning) |
| `stadium_present` | `[1]` | float32 | 1 if a stadium is in play |

### State — categorical token attributes (shared `[L_STATE]` vectors)
| key | shape | dtype | values |
|---|---|---|---|
| `tok_type` | `[L_STATE]` | int64 | 0=CLS,1=POKE,2=HAND,3=SUMMARY,4=STADIUM |
| `tok_owner` | `[L_STATE]` | int64 | 0=none,1=self,2=opp |
| `tok_zone` | `[L_STATE]` | int64 | 0=cls,1=active,2=bench,3=hand,4=summary,5=stadium |
| `tok_mask` | `[L_STATE]` | bool | True = real token (encoder padding mask = ~tok_mask) |

### Action — option references & features
| key | shape | dtype | contents |
|---|---|---|---|
| `opt_type` | `[O_MAX]` | int64 | `OptionType` 0–16; PAD slot = 0 (masked) |
| `opt_src_idx` | `[O_MAX]` | int64 | state-token row this option's *source* points to, else −1 |
| `opt_tgt_idx` | `[O_MAX]` | int64 | *target* token row (ATTACH/EVOLVE), else −1 |
| `opt_card_id` | `[O_MAX]` | int64 | direct card id for refs not in the token set (e.g. `select.deck`), else PAD |
| `opt_attack_idx` | `[O_MAX]` | int64 | remapped attack index for `ATTACK`, else 0 |
| `opt_scalar` | `[O_MAX, F_OPT=6]` | float32 | `[number/COUNT_N, count/COUNT_N, energyIndex/N_ENERGY, toolIndex/2, remainEnergyCost/ATKCOST_N, remainDamageCounter/DMGCTR_N]` |
| `opt_mask` | `[O_MAX]` | bool | True for the real `len(select.option)` options |

### Labels & bookkeeping
| key | shape | dtype | contents |
|---|---|---|---|
| `action_idx` | `[O_MAX]` | int64 | ordered expert picks, `−1`-padded (len ∈ `[minCount,maxCount]`) |
| `action_len` | `[]` | int64 | number of picks |
| `minCount`,`maxCount` | `[]` | int64 | from `select` |
| `sel_type`,`sel_ctx` | `[]` | int64 | for per-context metrics & loss weighting |
| `value_target` | `[]` | float32 | acting player's episode reward: `+1` win / `−1` loss (both present, §1.1) |
| `sample_weight` | `[]` | float32 | per-context / per-archetype balancing weight |

> Deck-selection steps (`select is None`) are **excluded** from this featurizer — the deck is fixed
> (`FIXED_DECK`), not predicted (§1.2).

## A.5 `poke_feat` (F_POKE = 29) and `sum_feat` (F_SUM = 12)

`poke_feat[slot]` (all-zero for empty/face-down slots):
| slice | dim | contents |
|---|---|---|
| `[0]` | 1 | `hp/HP_N` |
| `[1]` | 1 | `maxHp/HP_N` |
| `[2]` | 1 | `hp/maxHp` (ratio; 0 if none) |
| `[3:15]` | 12 | attached-energy histogram over `EnergyType` (`count/ENERGY_N`) |
| `[15]` | 1 | `len(energies)/ENERGY_N` (total attached) |
| `[16]` | 1 | `len(energyCards)/ENERGY_N` |
| `[17]` | 1 | `len(tools)/2` |
| `[18]` | 1 | `len(preEvolution)/2` (evolution depth) |
| `[19]` | 1 | `appearThisTurn` |
| `[20]` | 1 | `is_active` |
| `[21:26]` | 5 | active-only condition flags `[poison,burn,sleep,paralyze,confuse]` (0 on bench) |
| `[26]` | 1 | KO damage ratio, `min(dmg/hp, 2.0)` — see below |
| `[27]` | 1 | KO flag, `1.0` when the ratio reaches `1.0` |
| `[28]` | 1 | `1/(1+ceil(hp/10))` — damage counters still needed to KO |

(21 + 5 + 2 + 1 = 29; conditions read from the owning `PlayerState` flags and applied to that player's
active slot only.)

**The KO block (`[26:28]`, written by `_slot_ko_block`) flips polarity by owner**, matching
`_card_target_preview`: opportunity on their side, danger on mine.

* **my slots 0..5** — their Active's best attack against this slot, over the slot's current HP; the
  flag means *this one dies*. Affordability is **ignored**, matching `_ko_pressure`'s opponent side:
  their energy is theirs to spend next turn. Bench slots are scored as if gusted into the Active
  spot, which is the question a promote or a Boss's Orders read actually asks.
* **their slots 6..11** — my Active's best **affordable** attack against this slot, over its HP; the
  flag means *I can kill it*.

`tok_owner` is in the embedder's input, so the two readings are separable. Every unresolvable case
(no static tables, empty slot, dead attacker, HP <= 0) stays `(0.0, 0.0)` — the same "no information"
signal a PAD row carries. The ratio is on `[0, 2]` like every other damage ratio in the featurizer
(`cls_feat[91]`/`[93]`, `opt_scalar[6]`/`[11]`), not `[0, 1]`: the extra headroom separates "barely
lethal" from "overkill".

`sum_feat[player]`:
`[is_me, deckCount/DECK_N, handCount/HAND_N, len(bench)/BENCH_N, benchMax/BENCH_N, prizes_left/PRIZE_N,
len(discard)/DECK_N, poisoned, burned, asleep, paralyzed, 1/(1+deckCount)]` → 12. (5th condition `confused` is dropped
here to keep 11; it is already in `poke_feat`. Adjust to 12 if you prefer symmetry.)

## A.6 `cls_feat` (F_GLOBAL = 95) — global state + **decision conditioning**

| slice | dim | contents |
|---|---|---|
| `[0]` | 1 | `turn/TURN_N` |
| `[1]` | 1 | `turn % 2` |
| `[2]` | 1 | `am_i_first` — `float(yourIndex == firstPlayer)`, 0 while the toss is open |
| `[3]` | 1 | `turnActionCount/COUNT_N` |
| `[4]` | 1 | `toss_undecided` — `float(firstPlayer == -1)` |
| `[5:9]` | 4 | `[supporterPlayed, stadiumPlayed, energyAttached, retreated]` |
| `[9:11]` | 2 | `[my_prizes_left/PRIZE_N, opp_prizes_left/PRIZE_N]` |
| `[11:22]` | 11 | `select.type` one-hot (`N_SELTYPE`) |
| `[22:71]` | 49 | `select.context` one-hot (`N_SELCTX`) |
| `[71:75]` | 4 | `[minCount/COUNT_N, maxCount/COUNT_N, remainEnergyCost/ATKCOST_N, remainDamageCounter/DMGCTR_N]` |
| `[75:85]` | 10 | my-active conditions (5) ⊕ opp-active conditions (5) |
| `[85:87]` | 2 | `[has_contextCard, has_effect]` |
| `[87:91]` | 4 | reserved (0) for forward-compat with appended enum values |
| `[91:95]` | 4 | KO pressure |

**No absolute seat may appear in this block.** `[2]` and `[4]` replace an earlier
`yourIndex` scalar plus a 3-wide absolute `firstPlayer` one-hot. Both were
absolute-seat quantities and only their XOR carried information: every other
tensor in the dict is already built relative to `yourIndex`, so relabelling seat
0 ↔ seat 1 is a symmetry of the whole input. Keeping them cost accuracy rather
than two floats — seat 0 wins the coin toss in every corpus episode measured
(3000/3000) and elects to go first in 99.1%, leaving `yourIndex` 96.25%
collinear with "am I going second", so the policy used the seat as the proxy.
Measured on a real Kaggle replay, flipping only those columns changed the
agent's chosen action on 11 of 126 decisions. `test_featurizer_seat_invariance`
pins the symmetry; `CLS_AM_I_FIRST`, `CLS_TOSS_UNDECIDED`, `CLS_OUR_PRIZES`,
`CLS_OPP_PRIZES`, `CLS_HAS_CONTEXT_CARD` and `CLS_HAS_EFFECT` are exported from
`featurizer.py` so no other module hardcodes a column that can shift again.

`context_card_id`/`effect_card_id` are embedded via `card_embed` and **added** to the CLS token vector
in the model, alongside `ClsMLP(cls_feat)`.

## A.7 Reference resolution (`opt_src_idx` / `opt_tgt_idx` / `opt_card_id`)

Precompute a map `ref(area, playerIndex, index) -> state-token row` using the fixed layout (A.1):
active/bench of each player → rows 1–12; the acting player's hand → rows 13..13+H_MAX-1. Then per
`Option.type`:

| OptionType | src_idx | tgt_idx | card_id | attack_idx |
|---|---|---|---|---|
| `PLAY(7)` | hand row `index` | −1 | **resolved from hand state** | 0 |
| `ATTACH(8)` | `ref(area,me,index)` (hand/token) | `ref(inPlayArea,me,inPlayIndex)` | **resolved from source area state** | 0 |
| `EVOLVE(9)` | `ref(area,me,index)` | `ref(inPlayArea,me,inPlayIndex)` | **resolved from source area state** | 0 |
| `ABILITY(10)/DISCARD(11)` | `ref(area,player,index)` | −1 | **resolved from referenced area state** | 0 |
| `RETREAT(12)` | row 1 (my active) | −1 | PAD | 0 |
| `ATTACK(13)` | row 1 (my active) | −1 | PAD | `remap(attackId)` |
| `CARD(3)` | `ref(area,playerIndex,index)` or −1 | −1 | **`id` of the referenced card** (always set when known) | 0 |
| `TOOL_CARD(4)/ENERGY_CARD(5)/ENERGY(6)` | `ref(area,playerIndex,index)` | −1 | **`id` of the specific attached/selected card** | 0 |
| `YES(1)/NO(2)/END(14)/NUMBER(0)/SKILL(15)/SPECIAL_CONDITION(16)` | −1 | −1 | `cardId` if present else PAD | 0 |

**Disambiguation via `opt_card_id` (important).** Nearly all option types that reference a
card — `PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, CARD, TOOL_CARD, ENERGY_CARD, ENERGY` — resolve
`opt_card_id` from the referenced card's identity via state dereference. This gives the pointer head a
direct card-embedding signal on top of the gathered source/target token, which helps disambiguate
options that share the same source token (e.g. two different tools on the same Pokémon). For
`ENERGY(6)`/`TOOL_CARD(4)` under `DISCARD`/move contexts in particular, several legal "discard one
energy from active" options may all resolve to the **same** `src_idx` (that Pokémon's row); the
`opt_card_id` distinguishes them. `opt_scalar[energyIndex/toolIndex]` remains a secondary tiebreak.
(If per-`sel_ctx` metrics later show these contexts are still accuracy sinks — options identical in
both card id *and* source, e.g. two copies of the same basic energy — promote attachments to real
tokens; deferred until the data shows it is needed.)

When `src_idx == −1` and `card_id == PAD`, the option token is built from `optType_emb ⊕ opt_scalar`
plus the CLS context only (constant-type options like END/YES/NO). Model gathers rows with a safe
clamp: index `−1` → a learned `NULL_TOKEN` (row appended at position `L_STATE`, always masked-in for
gather only).

## A.8 Batching / collation

Stack per-sample tensors on a leading batch dim `B`. All shapes are already fixed (`L_STATE`, `O_MAX`,
…) so collation is a plain `np.stack` — no ragged handling. Provide:
`encoder_padding_mask = ~tok_mask` `[B, L_STATE]`; option softmax masks with `opt_mask` (`−inf` on pad).
Store shards as compressed `.npz` / webdataset keyed by `(episode_id, step_index)`; keep a sidecar
`meta.parquet` with `episode_id, player, team, archetype_self, archetype_opp, sel_ctx, won` for
stratified sampling, outcome down-weighting (`W_LOST`), and slice-wise metrics.

## A.9 Derived model-input dims (sanity)

`card_static: 52 → CardStaticMLP → 256`; `attack_static: 14 → 256`; `poke_feat: 26 → PokeMLP → 256`;
`hand_feat: 2 → 256`; `sum_feat: 11 → 256`; `cls_feat: 93 → 256`; `opt_scalar: 6` (concatenated with
gathered 256-d refs before the option MLP). Encoder sees `[B, L_STATE=46, 256]`; pointer head scores
`[B, O_MAX=64]` masked to `opt_mask`.

---

# Appendix B — Model Modules (exact `nn.Module` breakdown + forward)

PyTorch reference. Every tensor name/shape matches **Appendix A**. `B` = batch of decision points.
Recap of the dims used below: `D=256`, `L=L_STATE=46`, `O=O_MAX=64`, `P=P_MAX=12`, `H=H_MAX=30`,
`SUM=2`, `V`, `A`, `F_CARD=52`, `F_ATK=14`, `F_POKE=26`, `F_HAND=2`, `F_SUM=11`, `F_GLOBAL=95`,
`F_OPT=6`. Standard blocks are `batch_first=True`, `norm_first=True` (pre-norm), GELU, dropout 0.0.

## B.0 Prebuilt lookup buffers (registered, not learned)
`card_static_table : float32[V, 52]` and `attack_static_table : float32[A, 14]`, built once from
`all_card_data()`/`all_attack()` over the mined vocab (row 0 = PAD zeros, row 1 = UNKNOWN = mean of
in-vocab rows). Registered with `register_buffer` so samples carry only **ids**.

## B.1 `CardEncoder` / `AttackEncoder` — id ⊕ static → D
```python
class CardEncoder(nn.Module):
    def __init__(self, V, D=256):
        self.id_emb = nn.Embedding(V, D, padding_idx=0)
        self.static_mlp = MLP(52, D, D)            # 52→D→D, GELU
        self.register_buffer("static", card_static_table)   # [V,52]
    def forward(self, ids):                        # ids: int64[...]
        return self.id_emb(ids) + self.static_mlp(self.static[ids])   # [...,D]

class AttackEncoder(nn.Module):                    # same shape, 14→D, table [A,14]
    def forward(self, idx): return self.id_emb(idx) + self.static_mlp(self.static[idx])
```
`MLP(i,h,o) = Linear(i,h)→GELU→Linear(h,o)`.

## B.2 `TokenEmbedder` — featurizer dict → `[B, L, D]`
```python
class TokenEmbedder(nn.Module):
    def __init__(self, D=256):
        self.card = CardEncoder(V, D)
        self.type_emb  = nn.Embedding(5, D)        # CLS,POKE,HAND,SUMMARY,STADIUM
        self.owner_emb = nn.Embedding(3, D)        # none,self,opp
        self.zone_emb  = nn.Embedding(6, D)        # cls,active,bench,hand,summary,stadium
        self.poke_mlp  = MLP(F_POKE=26, D, D)
        self.hand_mlp  = MLP(F_HAND=2,  D, D)
        self.sum_mlp   = MLP(F_SUM=11,  D, D)
        self.cls_mlp   = MLP(F_GLOBAL=95, D, D)
        self.no_stadium = nn.Parameter(torch.zeros(D))   # when stadium absent

    def forward(self, x):                          # x = featurizer dict, all with batch dim B
        B = x["tok_type"].shape[0]
        rows = torch.zeros(B, L, D, device=dev)

        # feature contribution per group (placed at fixed positions A.1)
        cls  = self.cls_mlp(x["cls_feat"]) \
             + self.card(x["context_card_id"]).squeeze(1) * x_has_context \
             + self.card(x["effect_card_id"]).squeeze(1)  * x_has_effect          # [B,D]
        poke = self.poke_mlp(x["poke_feat"]) + self.card(x["poke_card_id"])       # [B,P,D]
        hand = self.hand_mlp(x["hand_feat"]) + self.card(x["hand_card_id"])       # [B,H,D]
        # discard bag-of-cards pooled into the summary token.
        # Use mask-aware SUM, not mean: pile size is signal (a 20-card discard ≠ a 2-card one),
        # and mean would erase it. Normalize by a constant (DECK_N=60), not by the live count, so
        # magnitude survives while staying bounded. (sum_feat already carries len(discard) too.)
        disc = masked_sum(self.card(x["discard_ids"]), x["discard_mask"], dim=2) / DECK_N  # [B,SUM,D]
        summ = self.sum_mlp(x["sum_feat"]) + disc                                 # [B,SUM,D]
        stad = torch.where(x["stadium_present"].bool(),
                           self.card(x["stadium_card_id"]).squeeze(1), self.no_stadium)  # [B,D]

        rows[:,0]      = cls
        rows[:,1:13]   = poke
        rows[:,13:43]  = hand
        rows[:,43:45]  = summ
        rows[:,45]     = stad

        # additive categorical embeddings for ALL rows at once
        rows = rows + self.type_emb(x["tok_type"]) \
                    + self.owner_emb(x["tok_owner"]) \
                    + self.zone_emb(x["tok_zone"])
        return rows                                 # [B,L,D]
```
`masked_sum` returns 0 for empty piles (all-`False` mask). `x_has_context/effect` are the `[B,1]`
flags from `cls_feat[87:89]` (broadcast).

## B.3 `Encoder` — self-attention over state tokens
```python
class Encoder(nn.Module):
    def __init__(self, D=256, heads=8, layers=4, ff=1024):
        layer = nn.TransformerEncoderLayer(D, heads, ff, dropout=0.0,
                    activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(D))
    def forward(self, rows, tok_mask):              # rows[B,L,D], tok_mask bool[B,L]
        return self.enc(rows, src_key_padding_mask=~tok_mask)   # h[B,L,D]
```
The trailing `norm` is required: `norm_first=True` normalizes *into* each sublayer but never on the
way out, so without it the residual stream leaves the stack un-normalized and its scale grows with
depth. Measured per-element RMS across the four layers was 0.13 → 0.43 → 0.77 → 1.22 → 1.49. That is
mild at `layers=4` — `PointerHead` is the consumer that cares, since it cross-attends into these rows
as unnormalized keys/values — but it compounds if `layers` is raised.
(Token count is ~46 → packing unnecessary; the `entity-transformer` packing trick is optional and
only helps if `L` grows.)

## B.4 `PointerHead` — options gather encoded rows, cross-attend, score
Option tokens are built **after** the encoder from gathered rows (Appendix A.7 index maps). A learned
`null_token` handles `idx == -1`. The option query is an **additive base** (opt-type + src + tgt +
card + attack, each `D`-dim) concatenated with the `F_OPT` scalars and projected `D+F_OPT → D` by the
single layer `opt_in`. All layers are real module fields — nothing is constructed inside `forward`.
`msgru_h` (default None) is the multi-select GRU hidden state carrying "already-picked" context
into every option query for multi-select (B.7); `forward` returns the per-option reprs `o` so the
caller can update the GRU with the chosen option.
```python
class PointerHead(nn.Module):
    def __init__(self, D=256, heads=8):
        self.opt_type_emb = nn.Embedding(18, D)            # OptionType 0..16 + STOP=17
        self.card   = None                                  # bound to parent CardEncoder (see B.6)
        self.attack = AttackEncoder(A, D)
        self.null_token = nn.Parameter(torch.zeros(D))
        self.msgru = nn.GRUCell(D, D)                       # multi-select memory
        self.opt_in = nn.Linear(D + F_OPT, D)               # base(D) ⊕ opt_scalar(F_OPT) → D
        self.cross = nn.MultiheadAttention(D, heads, batch_first=True)
        self.ln_q, self.ln_o = nn.LayerNorm(D), nn.LayerNorm(D)
        self.ffn   = MLP(D, 4*D, D)
        self.score = nn.Linear(D, 1)

    def gather(self, h_aug, idx):                          # idx[B,O] with -1→last(null) row
        idx = torch.where(idx < 0, torch.full_like(idx, L), idx)
        return torch.gather(h_aug, 1, idx.unsqueeze(-1).expand(-1,-1,D))   # [B,O,D]

    def forward(self, h, tok_mask, card_enc, x, msgru_h=None):
        B = h.shape[0]
        h_aug = torch.cat([h, self.null_token.expand(B,1,D)], 1)          # [B,L+1,D]
        src = self.gather(h_aug, x["opt_src_idx"])                        # [B,O,D]
        tgt = self.gather(h_aug, x["opt_tgt_idx"])
        base = self.opt_type_emb(x["opt_type"]) + src + tgt \
             + card_enc(x["opt_card_id"]) + self.attack(x["opt_attack_idx"])   # [B,O,D]
        if msgru_h is not None:                                           # multi-select: [B,D]→[B,1,D]
            base = base + msgru_h.unsqueeze(1)
        q = self.ln_q(F.gelu(self.opt_in(torch.cat([base, x["opt_scalar"]], -1))))  # [B,O,D]
        # cross-attention: queries=options, keys/values=state tokens
        a,_ = self.cross(q, h, h, key_padding_mask=~tok_mask)            # [B,O,D]
        o = self.ln_o(q + a); o = o + self.ffn(o)                        # [B,O,D] per-option reprs
        logits = self.score(o).squeeze(-1).masked_fill(~x["opt_mask"], -1e9)   # [B,O]
        return logits, o                                                 # o reused by B.7 GRU update
```

## B.5 `ValueHead`
```python
class ValueHead(nn.Module):
    def __init__(self, D=256): self.f = nn.Sequential(nn.Linear(D,D), nn.GELU(), nn.Linear(D,1))
    def forward(self, h_cls): return torch.tanh(self.f(h_cls)).squeeze(-1)   # [B] in (-1,1)
```

## B.6 `Policy` — end-to-end
```python
class Policy(nn.Module):
    def __init__(self, V, A, D=256, heads=8, layers=4, ff=1024, n_opp_arch=0):
        self.embed = TokenEmbedder(); self.encoder = Encoder()
        self.pointer = PointerHead(); self.pointer.card = self.embed.card
        self.value = ValueHead()
        self.belief = BeliefModule(V, D)
        self.belief.card_emb = self.embed.card          # share CardEncoder
        # Supervised opponent predictions (optional: n_opp_arch=0 disables the auxiliary loss).
        self.belief_heads = BeliefHeads(V, D, n_opp_arch, card_emb=self.embed.card)
        self.history_gru = nn.GRUCell(D, D)             # cross-turn memory on CLS token
    def _encode(self, x, history_h=None):
        rows = self.embed(x)                            # [B,L,D]
        h    = self.encoder(rows, x["tok_mask"])        # [B,L,D]
        # Belief: encode logs into a belief vector, added to CLS before history GRU
        belief = self.belief(x["log_feat"], x["log_mask"]) if "log_feat" in x else 0
        cls_token = h[:,0] + belief
        if history_h is None:
            history_h = torch.zeros(B, D)
        cls_out = self.history_gru(cls_token, history_h)
        h = torch.cat([cls_out.unsqueeze(1), h[:,1:]], dim=1)   # replace CLS
        return h, history_h
    def forward(self, x, history_h=None):
        h, history_h = self._encode(x, history_h)
        logits, _ = self.pointer(h, x["tok_mask"], self.embed.card, x)
        value  = self.value(h[:,0])
        return logits, value, history_h.detach()
```
(`forward` returns single-select logits; multi-select re-scoring is driven by B.7, which needs the
per-option reprs `o` that `PointerHead` also returns. The `BeliefModule` encodes the log stream into a
belief vector via a GRU over log entries; `BeliefHeads` outputs 4 auxiliary opponent-card prediction
heads — archetype, deck, hidden cards, hand — used for an auxiliary belief loss during training.)

## B.7 Multi-select (autoregressive pick-without-replacement)
`maxCount==1` (common) → `Policy.forward` logits used directly. For `maxCount>1`, re-score after each
pick using a **GRU cell** to track "already-picked" context, and a **STOP column** for variable-length
termination. The STOP column is inserted as option type 17 at index `stop_column`; when chosen, the
sample is done picking and all subsequent columns are ignored.

`PointerHead.forward` (B.4) accepts `msgru_h` (the GRU hidden state) and returns `(logits, o)` where
`o[B,O,D]` are the per-option reprs — update the GRU from the chosen one:

```python
def select_multi(pointer, h, tok_mask, card_enc, x, minC, maxC, stop_col):
    B = h.shape[0]
    chosen, active = [], torch.ones(B, dtype=torch.bool)
    picked_mask = x["opt_mask"].clone()
    msgru_h = torch.zeros(B, D, device=h.device)
    for t in range(maxC):
        logits, o = pointer(h, tok_mask, card_enc, x, msgru_h=msgru_h)
        logits = logits.masked_fill(~picked_mask, -1e9)
        j = logits.argmax(-1); chosen.append(j)
        # STOP column terminates the sample
        stopped = (j == stop_col) & (t >= minC)
        active = active & ~stopped
        picked_mask[torch.arange(B), j] = False
        # Update GRU with chosen option reprs
        msgru_h[active] = pointer.msgru(o[active, j[active]].float(), msgru_h[active])
    return torch.stack(chosen, 1)                    # [B, maxC], STOP picks = -2
```
**Training** (`multiselect_ce`) teacher-forces the expert order from `action_idx`: at step `t` re-score
with the current `msgru_h`, take masked CE against `action_idx[t]`, mask that true pick, and update the
GRU with the *true* option's `o[·, action_idx[t]]`; loss = Σ_t CE. The STOP column is supervised to
fire after the expert's last pick (for `t ≥ minCount`); STOP picks are encoded as `-2` in `action_idx`.

**Variable length.** The STOP column is always present (not conditional on data), supporting both
fixed-length and variable-length selects. The `minCount` gate ensures STOP is only legal once the
minimum pick count is satisfied.

## B.8 Loss & init
```python
logits, value, _history_h = policy(x)
if x.maxCount == 1:
    ce = F.cross_entropy(logits, x["action_idx"][:,0], label_smoothing=0.05, reduction="none")
else:
    ce = multiselect_ce(policy, x)                 # B.7, teacher-forced
loss = (x["sample_weight"] * ce).mean() + LAMBDA_V * F.mse_loss(value, x["value_target"])
if n_opp_arch > 0:                                 # auxiliary belief loss (opponent card prediction)
    belief_preds = policy.belief_heads(belief_h)
    loss = loss + belief_loss(belief_preds, x)     # deck CE + hidden CE + hand CE + arch CE
```
- **Per-context weighting** folded into `sample_weight` (Appendix A.4).
- **Belief auxiliary loss**: when `n_opp_arch > 0`, the `BeliefHeads` module predicts opponent
  archetype, deck contents, hidden cards, and hand cards from the belief state; the auxiliary loss
  adds four CE terms (deck, hidden, hand, arch) with configurable per-term weights.
- **No RL-style priors here** — BC needs no halt/full-send init biases (those were for the
  `micro-step` *RL* setting). Default init: `nn.init.trunc_normal_(emb, std=0.02)`, LN default,
  `null_token`/`no_stadium` zero-init.
- `LAMBDA_V ≈ 0.5` **from v1**. Because training keeps both won and lost games (§1.1),
  `value_target ∈ {+1,-1}` carries real signal and the value head learns a usable critic. (If you run
  the wins-only *ablation*, `value_target≡1` is constant and teaches nothing — set `LAMBDA_V=0` there.)

## B.9 Parameter budget (D=256, 4 layers)
Embeddings (`id_emb V·256` + type/owner/zone) ≈ 0.1–0.3M; static MLPs ≈ 0.3M; encoder 4×(~0.8M) ≈
3.2M; pointer (opt_in + cross + ffn) ≈ 1M; value ≈ 0.1M. **Total ≈ 5–8M params** — single-GPU, fits
millions of decision points comfortably.

---

# Appendix C — Training Loop & Data Pipeline

Assumes the corpus-mining job (separate spec) has already produced: the vocab remap + normalization
constants (`vocab.json`), the archetype definitions (`𝒟_self`, `𝒟_opp`, `FIXED_DECK`), and the
**pre-featurized** decision-point shards (Appendix A tensors). Training is therefore **GPU-bound**, not
parse-bound — no `obs_dict` parsing happens in the hot loop.

## C.1 On-disk shard format (columnar, mmap-friendly)
Featurize offline once; store **fixed-shape** arrays stacked along a leading sample axis `S`.

```
data/
  vocab.json                 # id remap, attack remap, norm constants, F_* dims, caps  → frozen
  archetypes.json            # 𝒟_self / 𝒟_opp signatures, FIXED_DECK (60 ids)
  shards/
    train-00000.npz          # ~50–100k samples/shard; each key = stacked array [S, …]
    train-00001.npz  …
    val-00000.npz  …
    test-00000.npz  …
  meta.parquet               # one row per sample: sample_uid, shard, row, episode_id, player,
                             #   team, archetype_self, archetype_opp, sel_type, sel_ctx,
                             #   minCount, maxCount, won   (won drives W_LOST + wins-only ablation)
```
Each `.npz` holds exactly the Appendix-A keys (`poke_card_id[S,12]`, `cls_feat[S,93]`,
`opt_src_idx[S,64]`, `action_idx[S,64]`, …). Compression `np.savez_compressed`; open with `mmap_mode`
so the OS page-cache serves random reads. **Split is by whole episode** (mining-time), so no game
leaks across train/val/test.

## C.2 Index & split
`meta.parquet` is the master index. Build three row-index arrays (`train/val/test`) by `split`
column. All stratification/weighting is computed from `meta` **without touching shard bytes**, so the
sampler is cheap to construct and fully reproducible from `meta` + a seed.

## C.3 Sampler & sample weighting
Three factors fold into one scalar `sample_weight` (derived at load time from `meta.parquet`, so
re-weighting never re-featurizes — see D.4):

```
w_ctx[c]   = (N_total / N_ctx[c])   ** ALPHA_CTX      # ALPHA_CTX = 0.5   (rare-context balance)
w_arch[a]  = (N_total / N_arch_self[a]) ** ALPHA_ARCH  # ALPHA_ARCH = 0.5 (balance 𝒟_self piloting)
w_out      = 1.0 if won else W_LOST                    # W_LOST = 0.6      (§1.1 outcome down-weight)
sample_weight = normalize( w_ctx[sel_ctx] * w_arch[archetype_self] * w_out )   # mean ≈ 1 over train
```
(`won` is a `meta` column; the wins-only ablation is simply `keep rows where won`.)
Training uses a **`WeightedRandomSampler`** over train rows with these weights (sampling *with*
replacement, one "epoch" = `len(train)` draws) **or** uniform shuffling + `sample_weight` applied only
in the loss — pick one, not both, to avoid double-counting. **Default: uniform shuffle + loss
weighting** (simpler, exact, no sampler variance). Rare contexts (SKILL_ORDER, DAMAGE_COUNTER_ANY) are
then upweighted in the loss rather than oversampled.

## C.4 DataLoader / batching / collation
Every tensor is already fixed-shape ⇒ **default collate = `torch.stack`**, no ragged logic.
- `batch_size = 2048` decision points (tune to VRAM); `drop_last=True`.
- `num_workers = 8`, `pin_memory=True`, `persistent_workers=True`; workers read shard slices via mmap
  and return dict-of-tensors.
- Move to GPU with `non_blocking=True`; cast float inputs to `bf16` inside the model (keep int index
  maps in `int64`).
- `derive` at collate: `encoder_padding_mask = ~tok_mask`. Nothing else to compute.

## C.5 Optimizer & schedule
| Knob | Value |
|---|---|
| Optimizer | `AdamW(betas=(0.9,0.95), weight_decay=0.01)` |
| No-decay params | biases, `LayerNorm`, all `nn.Embedding`, `null_token`, `no_stadium` |
| Peak LR | `3e-4` |
| Schedule | linear warmup `1000` steps → cosine decay to `3e-5` over total steps |
| Total | `EPOCHS = 10` over train (early-stop on val, see C.8) |
| Grad clip | global-norm `1.0` |
| Precision | `bf16` autocast; fp32 master weights |
| Label smoothing | `0.05` (single-select CE) |
| Weight EMA | decay `0.999`, evaluate & ship the **EMA** weights |
| `LAMBDA_V` | **0.5 from v1** (both outcomes ⇒ informative `value_target ∈ {±1}`); `0.0` only in the wins-only ablation |
| `W_LOST` | **0.6** — sample-weight multiplier on decisions from lost expert games (§1.1) |

## C.6 Training step (pseudocode)
```python
for step, batch in enumerate(loader):
    batch = to_device(batch)
    with autocast("cuda", dtype=torch.bfloat16):
        logits, value, _history_h = policy(batch)                # [B,O], [B]; cross-turn GRU on CLS
        if (batch["maxCount"] == 1).all():
            ce = F.cross_entropy(logits, batch["action_idx"][:,0],
                                 reduction="none", label_smoothing=0.05)
        else:
            ce = multiselect_ce(policy, batch)                   # Appendix B.7, teacher-forced
        loss = (batch["sample_weight"] * ce).mean()
        if LAMBDA_V > 0:
            loss = loss + LAMBDA_V * F.mse_loss(value, batch["value_target"])
        if n_opp_arch > 0:                                      # auxiliary belief loss (B.8)
            loss = loss + belief_loss(policy.belief_heads(belief_h), batch)
    scaler_or_backward(loss); clip_grad_norm_(policy.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True); ema.update(policy)
    if step % LOG_EVERY == 0:   wandb.log(train_signals(loss, ce, value, grad_norm, sched), step=step)  # C.9
    if step % VAL_EVERY == 0:   offline_eval(ema, val_loader)      # C.8 → wandb.log(step)
    if step % CKPT_EVERY == 0:  save_ckpt(step, policy, ema, opt, sched)   # local only (C.7)
```
Live eval is run **after training completes** (not inline on the GPU step), wrapping the EMA policy as
`agent()` and running head-to-head games against baseline opponents in a separate process pool.
All logging goes through **Weights & Biases** (C.9): `train_signals` returns the per-step scalars,
`offline_eval`/`live_eval` return dicts logged at the same global `step` so train/val/live curves share
one x-axis.
Batches are (near-)homogeneous in `maxCount` because most selects are single-pick; to avoid the
`.all()` branch stalling, **bucket samples by `maxCount==1` vs `>1`** at shard-build time (two sub-
streams) and interleave, or just run `multiselect_ce` for all (it reduces to plain CE when len==1).

## C.7 Checkpointing & shippable artifacts
`CKPT_EVERY = 2000` steps + always keep **best-val** and **last**. A checkpoint = `{step, model,
ema, opt, sched, rng}`. The **submission bundle** is assembled from the best-val EMA weights plus the
*frozen* preprocessing artifacts — these must travel together or inference will mis-featurize:
```
submission/  main.py  deck.csv(=FIXED_DECK)  cg/  weights.pt(ema)  vocab.json  archetypes.json
```
`vocab.json` (id/attack remaps + norm constants + caps) is the contract between the featurizer used in
training and the one inside `agent()`. Pin it; never re-mine vocab without retraining. Checkpoints,
`vocab.json` and `archetypes.json` stay **local to `--out-dir`/`--data-dir`** — nothing is uploaded to
W&B (C.9). Every checkpoint already pins `vocab_sha1`/`archetypes_sha1` in its `"deck"` record, and
`decks.json` sidecars the same information, so a run is reconstructible from disk without the tracker.

## C.8 Eval cadence
Two tiers — cheap offline metrics often, expensive live games rarely.

**Offline (`VAL_EVERY = 1000` steps), on held-out decision points:**
- **Top-1 / top-3 option accuracy, sliced per `sel_type` and per `sel_ctx`** (log as a table — an
  aggregate number hides rare-context collapse). Weighted and unweighted.
- Multi-select: exact-set match & per-pick top-1.
- Value MSE/AUC (from v1 — both outcomes trained, §1.1).
- Early-stop / best-ckpt criterion: **non-trivial micro top-1** (`val/top1_nontrivial`, which excludes
  near-degenerate `sel_ctx` buckets with ≤2 forced options). The macro average weights those
  degenerate buckets equally with the buckets holding almost all real decisions, so it tracks noise.
  Falls back to `val/top1_macro` if `top1_nontrivial` is unavailable. Patience ~5 evals.

**Live-engine** — the only metric that really matters, run **after training completes** (not inline
during training, to avoid competing with the GPU for compute):
- Wrap EMA policy as `agent()` piloting `FIXED_DECK`; run a **process pool** (the `cg` lib is one
  battle per process) of `≥500` games each vs: (a) the sample **random** agent (floor), (b) a **frozen
  previous checkpoint** (progress signal), (c) the **built-in `search_begin/step` planner**
  (`AGENT_SPEC.md` §5) as a scripted, non-trivial opponent. **Not** vs a replayed expert — an
  interactive game diverges from any recording once our policy deviates (see §5).
- Fixed seed schedule across evals for comparability; report win-rate ± Wilson interval, plus mean
  game length and illegal-action rate (must be 0 — engine err 4/5/6).

## C.9 Experiment tracking (Weights & Biases)
All training signals stream to **W&B**. One `wandb.run` per training job; **everything is logged
against the global `step`** so train/val/live curves overlay on a shared x-axis.

**Init & config.** `wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name, config=cfg)`
where `cfg` is the full hyperparameter dict. Respect `WANDB_MODE=offline` for air-gapped/Kaggle runs
(sync later); never put credentials in the repo — read `WANDB_API_KEY` from env. **`wandb` is a
training-only dependency (`uv add --group train wandb`)**: it is never imported by the submitted
`agent()` / `main.py`, so the `/kaggle_simulations/agent/` bundle (C.7) stays wandb-free.

**Per-step (every `LOG_EVERY=50`):**
- `train/loss`, `train/ce`, `train/value_mse` (when `LAMBDA_V>0`)
- `train/grad_norm` (pre-clip), `train/lr`
- `perf/samples_per_sec`
- Optional extras from auxiliary losses (e.g. `belief/*` when belief heads are active)

**Per offline eval (`VAL_EVERY=1000`):** log **both** the scalars and the per-context table —
- `val/top1_macro`, `val/top1_micro`, `val/top1_nontrivial` (micro over non-degenerate contexts; primary early-stop criterion, C.8)
- `val/top1@{sel_ctx}` and `val/top1@{sel_type}` as a `wandb.Table` (one row per context) — the
  aggregate hides rare-context collapse, so the table is the primary artifact, not the scalars
- `val/multiselect_exact_set`, `val/multiselect_perpick_top1`
- `val/value_mse`, `val/value_auc`
- `val/best_top1_nontrivial` + mark the best-val step; use `wandb.run.summary` for the best-so-far

**Per live eval (post-training):** for each opponent `o ∈ {random, frozen_ckpt, search}` —
- `live/winrate@{o}` with `live/winrate_ci_lo@{o}` / `_hi@{o}` (Wilson interval)
- `live/game_len_mean@{o}`, `live/illegal_action_rate@{o}` (**must be 0**)
- `live/oov_rate` (fraction of opponent cards hitting `UNKNOWN`, §1.2/§5)

**No artifact uploads.** W&B carries **metrics only**. Checkpoints, `vocab.json` and `archetypes.json`
are written to disk and never mirrored to the tracker — `wandb.Artifact` is not used anywhere in
training. Reproducibility comes from the local artifacts instead: each `.pt` stamps its `"deck"`
record with `vocab_sha1`/`archetypes_sha1` (C.7), and `decks.json`/`deck.csv` sidecar the same
information next to the checkpoints.

**Error handling:** NaN/Inf loss triggers an early-stop break. Illegal actions at live eval are
logged as metrics; a non-zero rate is a hard bug (engine err 4/5/6) to fix before the next run.

## C.10 Hyperparameter summary
```
D_MODEL=256  LAYERS=4  HEADS=8  FF=1024  DROPOUT=0.0
BATCH=2048   EPOCHS=10  PEAK_LR=3e-4  WARMUP=1000  MIN_LR=3e-5
WD=0.01  BETAS=(0.9,0.95)  GRAD_CLIP=1.0  LABEL_SMOOTH=0.05  EMA=0.999
ALPHA_CTX=0.5  ALPHA_ARCH=0.5  LAMBDA_V=0.5  W_LOST=0.6
LOG_EVERY=50  VAL_EVERY=1000  CKPT_EVERY=2000
WANDB_PROJECT="pokemon-tcg-il"  WANDB_ENTITY=<team>  WANDB_MODE=online  # offline on Kaggle, sync later
```

## C.11 Throughput & hardware
- Preprocessing (parse + featurize) is the CPU-heavy one-off; parallelize across episodes with a
  process pool, write shards. Re-run only when vocab/caps change.
- Training: single modern GPU. ~5–8M params + `L=46, O=64` ⇒ attention is tiny; expect input I/O
  (mmap) and Python overhead to dominate, so keep `num_workers` high and shards local (not on network
  FS). **Compile only the static path** — `torch.compile` the `TokenEmbedder+Encoder+single-select
  PointerHead` (all shapes fixed ⇒ no recompiles). Keep the **multi-select AR loop uncompiled**: its
  per-step count is data-dependent (`maxCount` varies) and would trigger graph breaks/recompiles.
  Bucketing by `maxCount==1` vs `>1` (C.6) keeps the compiled path pure.
- Live eval is the wall-clock cost; run it on CPU worker processes **after training** (it only needs
  a weights snapshot), not inline on the GPU step.

## C.12 Failure modes to watch
| Symptom | Likely cause / fix |
|---|---|
| High MAIN acc, low rare-context acc | micro-avg masking imbalance → macro criterion + context loss weights |
| Illegal actions at live eval | featurizer/vocab drift between train and `agent()` → same `vocab.json`; assert index bounds |
| Val acc good, live win-rate flat | BC ceiling / compounding errors → this is expected; the RL fine-tune (later spec) addresses it |
| Loss NaN early | LR too high with bf16 → longer warmup / lower peak; check `masked_fill(-1e9)` isn't hitting all-pad rows |
| Overfit after ~few epochs | small vocab + big model → more dropout, fewer layers, or more episodes/decks |

---

# Appendix D — Corpus Mining (raw episodes → frozen artifacts)

The upstream job. **Outputs** (consumed by Appendix C.1): `vocab.json`, `archetypes.json`,
`shards/`, `meta.parquet`, `mining_report.md`. Runs in three phases over a **downloaded raw cache**,
because vocab must be frozen (Phase 2) before featurization (Phase 3).

Key facts it relies on (all verified against a real episode):
- Each daily dataset `kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD` holds one `<episodeId>.json`
  per game (~0.3–9 MB), ~1.3k–7.8k games/day, ~21 GB/day → **sample, never bulk-download**.
- Per episode: `info.TeamNames=[t0,t1]`, `rewards=[r0,r1]` (`+1/-1/0`), `steps` interleave both
  players; a player's **deck** = the action paired (off-by-one) with its `select==None` step, i.e.
  `steps[1][p].action` (a 60-id list). Both players' decks are present even for the loser.

## D.1 Phase 0 — Sampling plan (bounded, reproducible)
```
budget: TARGET_EPISODES (first pass ≈ 8–15k), DAYS (most recent from manifest)
per_day = TARGET_EPISODES / |DAYS|
for day in DAYS:
    ids = list_all_files(day)                 # paged API, see D.2
    pick = deterministic_pick(ids, per_day, seed)    # hash-sort-take-exactly-k
    enqueue(day, pick)
```
Deterministic hash-sort-take-k makes the corpus reproducible and **resumable** (re-running picks the
same ids). Select the **most recent** days from `archive/manifest.csv` (recency is the primary signal;
`top_avg_score` is available but not used for day filtering — the recency bias implicitly captures
stronger meta as the competition matures).

## D.2 Phase 1 — Download (resumable, rate-limited)
Verified CLI (via the project venv; `KAGGLE_CONFIG_DIR=~/.kaggle`):
```bash
kaggle datasets files    kaggle/...-<day>                 # paginated: capture "Next Page Token"
kaggle datasets download kaggle/...-<day> -f <id>.json -p raw/<day>/   # single file, ~unzipped
```
Use the Python API for paging + parallelism:
```python
api = KaggleApi(); api.authenticate()
tok=None
while True:
    r = api.dataset_list_files("kaggle/...-<day>", page_token=tok); ...; tok=r.nextPageToken
    if not tok: break
# thread-pool downloads with 429 backoff; skip if raw/<day>/<id>.json exists (resume)
```
Cache layout `raw/<day>/<id>.json`; write `downloaded.parquet` (day, id, bytes, ok). **Validate on
read**: `statuses==["DONE","DONE"]`, `len(steps)>=2`, `rewards` length 2, both decks length 60 — drop
malformed episodes and log counts.

## D.3 Phase 2 — Stats pass → freeze EXPERTS / archetypes / vocab
One cheap pass reading only the deck steps + final `rewards` (no full featurization):
```python
for ep in raw_cache:
    t0,t1 = ep.info.TeamNames; r0,r1 = ep.rewards
    d0,d1 = deck_of(ep,0), deck_of(ep,1)        # steps[1][p].action, len 60
    for p,(t,r,d) in [(0,(t0,r0,d0)),(1,(t1,r1,d1))]:
        team_games[t]+=1; team_wins[t]+= (r==1)         # draws (r==0) count as non-win
        key=canon(d)                                     # tuple(sorted(60 ids))
        deck_freq[key]+=1
        if r==1: deck_winfreq[key]+=1
```
Then **freeze the decisions**:
1. **EXPERTS** = top-`K` teams by `win_rate=wins/games` with `games>=G_min` (defaults K=10, G_min=50).
2. **Archetype clustering** (over all decks in the sampled corpus): greedy by descending frequency —
   assign a deck to an existing cluster if `jaccard_multiset(deck, centroid) >= 0.90`, else new
   cluster; centroid/representative = the cluster's single most-frequent exact list. Clustering over
   the full corpus (not just expert wins) captures the complete meta for both 𝒟_self and 𝒟_opp.
   `jaccard_multiset = Σ min(a_c,b_c) / Σ max(a_c,b_c)` over card counts.
3. **𝒟_self** = top 3–8 archetypes by expert frequency *as the expert's own deck* (across all expert
   games, won and lost).
4. **𝒟_opp** = top ~3–8 archetypes by frequency *as the opponent's deck* in expert games
   (may overlap 𝒟_self; no mirror-only constraint).
5. **Game filter** (defines the training corpus): keep an episode's player `p` iff
   `team(p)∈EXPERTS ∧ arch(deck_p)∈𝒟_self ∧ arch(deck_{1-p})∈𝒟_opp`. **Both won and lost expert
   games are kept** (§1.1); the outcome becomes the `won` flag in `meta` and the `value_target`, and
   drives the `W_LOST` down-weight — it is *not* a filter.
6. **vocab** (decoupled from the game filter, §1.2). Do **not** restrict vocab to kept-game cards, or
   live inference drowns in OOV opponent cards. Build it from a **wider** set: rank all card ids by
   frequency across **every episode in the sampled corpus** (both decks, all players, regardless of
   expert status or archetype), take the top `N_VOCAB` (target ~300–500) — plus `PAD=0`, `UNKNOWN=1`
   → contiguous remap `2..V-1`. The vocab is built *before* the expert filter (Phase 2 step 1) so it
   covers the full meta; in-vocab cards cover all kept-game cards *and* a wide margin of live opponent
   cards. `UNKNOWN` catches the long tail only. `A` = attacks referenced by vocab cards. Record the
   frequency-coverage curve so `N_VOCAB` can be tuned against the live-OOV target (D.5).
7. **Caps** — `H_MAX,O_MAX,D_MAX` are set from the mining config defaults (30, 64, 60). These values
   are generous enough to cover all observed game states in the current format (hand ≤8 by rules, options
   ≤60 when picking from a full deck, discard ≤60 by deck size). The caps are written into `vocab.json`
   as part of the frozen artifact and must match between training and inference. If a future format
   exceeds these caps, update the config defaults and re-mine.
8. Build `card_static_table[V,52]`, `attack_static_table[A,14]` from `all_card_data()`/`all_attack()`.
9. **FIXED_DECK** = representative decklist of the 𝒟_self archetype with the highest expert win-rate
   (a *provisional* pick; the final submission deck is chosen among 𝒟_self candidates by **live**
   win-rate in C.8).

Write `vocab.json` (remaps + norm constants + caps + `F_*` dims), `archetypes.json` (𝒟_self/𝒟_opp
signatures + representatives + FIXED_DECK), and a `mining_report.md` (team leaderboard, archetype
table with win-rates, vocab size, kept-game counts, per-context decision counts).

## D.4 Phase 3 — Featurize pass → shards + meta
Re-parse **only kept (episode,player)**; iterate ACTIVE decisions with the off-by-one rule; featurize
against the **frozen** vocab (Appendix A); assign split by **episode hash** (train/val/test ≈
96/2/2 — whole games, never split within); append to open shard buffers, flush at `S` samples.
```python
for ep,p in kept:                           # kept = won AND lost expert games (§1.1)
  split = split_of(hash(ep.id))
  won_p = (ep.rewards[p] == 1)
  for i in active_decisions(ep,p):
     obs = ep.steps[i][p].observation
     act = ep.steps[i+1][p].action
     if obs.select is None: continue        # deck step excluded (fixed deck)
     sample = featurize(obs, act, vocab)     # Appendix A tensors; value_target = +1 if won_p else -1
     shard[split].add(sample)
     meta.append(uid, shard, row, ep.id, p, team, arch_self, arch_opp,
                 sel_type, sel_ctx, minCount, maxCount, won=won_p)   # `won` drives W_LOST at load
flush_all()
```
**`sample_weight` is NOT written into shards** — it is derived at load time from `meta.parquet`
columns (`sel_ctx`, `arch_self`, `won`) via the C.3 formula, so re-weighting (including the wins-only
ablation) never requires re-featurizing. (This supersedes the `sample_weight` row in Appendix A.4,
which becomes optional/derived.)

## D.5 Phase 4 — QA gates (fail loud before training)
- **Coverage**: every card id in every **kept-game** deck ∈ vocab (else vocab build bug). Note the
  vocab is intentionally *wider* than kept-game cards (§1.2), so this is a subset check, not equality.
- **OOV coverage curve**: report the fraction of cards, weighted by appearance frequency across the
  whole sampled corpus, that fall outside vocab at `N_VOCAB` = {200,300,400,500}. This is the
  train-time proxy for **live OOV rate** (§5); pick `N_VOCAB` so proxy OOV is small (target < a few %).
- **Variable-length multi-select audit**: count contexts with `minCount < maxCount` and their sample
  share. The STOP column (B.7) handles variable-length selects unconditionally; this audit is for
  awareness only (reports the fraction of samples that use the STOP mechanism).
- **Attachment-collision audit**: among `ENERGY/TOOL_CARD/CARD` selects, count options that share both
  `opt_src_idx` **and** `opt_card_id` (truly indistinguishable to the pointer head, e.g. two identical
  basic energies on one Pokémon). If this share is non-trivial, promote attachments to tokens (A.7).
- **Label sanity**: for every sample, `0 ≤ action_idx < len(option)`, distinct, `len∈[minCount,maxCount]`.
- **Reference round-trip**: decode a random 1k samples — `opt_src_idx`/`opt_card_id` must resolve to
  the card the option text implies (e.g. a `PLAY` option's hand slot holds a legal-to-play card).
- **Outcome balance**: report won vs lost decision counts (should be roughly balanced now, §1.1); a
  strong skew means the expert filter or the corpus is biased.
- **Deck legality**: `FIXED_DECK` = 60 cards, ≤4 copies of any non-basic-energy id, ≤1 ACE SPEC, ≥1
  basic Pokémon (it is a real winning deck, so this should pass — assert it).
- **Balance report**: samples per `sel_ctx` and per `arch_self`; flag contexts with <N samples (rare
  decisions the policy will underlearn).

## D.6 Provenance, refresh & determinism
- `vocab.json`/`archetypes.json` are **frozen artifacts**; changing `DAYS`, `K`, `G_min`, `N_VOCAB`,
  the Jaccard threshold, or 𝒟 sizes **re-mines vocab ⇒ requires retraining**. Artifacts capture the
  parameter values used to produce them (id remaps, norm constants, caps, archetype definitions).
  Reproducibility is achieved by re-running with the same config and seed; provenance tracking is
  handled externally (e.g. W&B config, git hash of the mining config). (Vocab depends on
  `N_VOCAB` + corpus frequencies, *not* on 𝒟 membership — §1.2 decoupling.)
- **Refresh**: new daily datasets can extend the raw cache and the featurized corpus *without* changing
  vocab (their cards simply map to existing ids or `UNKNOWN`); the archetype **game filter** (D.3.5)
  still decides which games train, so shards stay consistent. Because the vocab is wide and OOV
  degrades gracefully, a modest meta drift needs no retrain; a deliberate meta shift ⇒ new vocab
  version + retrain.
- Everything keys off one `seed` + the config; a full re-run reproduces the same corpus and artifacts.

## D.7 First-pass budget (concrete)
- **Stats/vocab pass**: ~8–15k sampled episodes (recent ~20 days) → enough to rank teams (with
  `G_min=50`) and find the top archetypes. Disk ≈ 25–45 GB raw.
  > **Sizing check before committing K/G_min:** 8–15k episodes ≈ 16–30k team-appearances. If the
  > competition has thousands of *distinct team names* (each submission version is a new name), few
  > teams will reach `G_min=50` and the top-K ranking is noisy. **First print the team-count and
  > games-per-team distribution**; if the tail is thin, either raise `TARGET_EPISODES`, lower `G_min`,
  > or collapse team-name versions before ranking. Don't hard-code K=10/G_min=50 sight-unseen.
- **Featurized corpus**: kept expert games in 𝒟 (won **and** lost, §1.1) → a fraction of the sampled
  set, but each game yields dozens–hundreds of decisions ⇒ easily **millions of `(obs,action)`
  samples**. Keeping losses roughly doubles the corpus vs wins-only. Expand `TARGET_EPISODES` if
  rare-context counts (D.5) are thin.
- Kaggle API rate limits: throttle downloads (small thread pool + backoff on 429); the download is the
  wall-clock bottleneck, fully resumable.

## D.8 Pipeline summary (end to end)
```
manifest.csv ─▶ [P0 sample] ─▶ [P1 download → raw/<day>/<id>.json] ─▶ [P2 stats]
   ─▶ freeze EXPERTS, 𝒟_self, 𝒟_opp, vocab.json, archetypes.json, FIXED_DECK
   ─▶ [P3 featurize kept games] ─▶ shards/ + meta.parquet ─▶ [P4 QA] ─▶ Appendix C training
```


## A.5.1 Deck-out features

Losing by deck-out is **7.8% of corpus games**, and **10.4% of all decision points sit at
`deck <= 5`** — so this is neither an edge case nor a data-scarcity problem.

**`sum_feat[:, 11] = 1/(1+deckCount)`** (`SUM_DECK_OUT_COL`), written for *both* players because
decking the opponent is a win condition. `sum_feat[:, 1]` remains `deckCount/DECK_N` and is *linear*
in a quantity whose decision-utility is *hyperbolic*: deck 50→48 and deck 3→1 are both a 0.033 step
there, and only one of them ends the game. The reciprocal is already in `[0, 1]` with no clipping
and needs no tuned threshold — deck 3→1 moves 0.250→0.500 while 50→48 moves 0.020→0.020, roughly
**250× more gradient where the game is decided**. It is an *additional* fixed divisor, never a
replacement: the all-zero row is still the PAD sentinel.

**`opt_scalar[:, 13] = min(draw_est / deckCount, 1.0)`** (`OPT_DECK_COST_COL`) — how much of what is
left *this* option burns, for **every** option type. `opt_scalar[:, 8]` carries the same quantity but
is filled only inside the `otype == 13` ATTACK branch, and an attack is rarely what decks you out;
the PLAY options are, and they were getting dims 0–5 and nothing else. ATTACK options copy dim 8 so
the two columns cannot disagree. An unresolvable card (PAD, hidden, absent from the table) stays
`0.0` rather than being assigned a guess, which would read as "safe to play" on exactly the cards we
cannot see.

**`card_static_row[83:85]`** (`CARD_DRAW_FIXED_COL`, `CARD_DRAW_TO_HAND_COL`) supplies the counts,
normalised by `DRAW_N` exactly as `attack_static_row[14:16]` is, so one formula reads either source.
`ptcg_mine.keywords.card_draw_counts` parses them from `card.skills[].text` — Trainers store oracle
text there just as Pokémon abilities do — and yields a numeric count for **38 of the 42**
draw-mentioning engine cards (90%); the rest fall back to the binary `draw` keyword already in the
ability row.

The two columns sit **before** the embedded attack blocks, not appended. `CARD_ATTACK_BLOCK_START` is
derived as `F_CARD - 3 * F_ATK` in the featurizer and `52 + K_EFFECT + 2 + 2` in `cards.py`;
appending would leave those disagreeing by exactly 2, and every embedded attack would decode as
garbage while every shape check still passed.


## A.5.2 Counters-to-KO and bench damage

**`poke_feat[:, 28] = 1/(1+ceil(hp/10))`** (`POKE_COUNTERS_TO_KO_COL`). A damage counter is 10 HP, so
this is the unit every counter-placement decision is denominated in. Measured on a real archetype-16
(Dragapult ex) game, **18 of that player's 96 decisions — 19% — were exactly that select**:
`select(type=1, context=14)` with `remainDamageCounter` counting 6 down to 1, choosing among 4–5
benched Pokémon. Reciprocal for the same reason the deck-out curve is: `hp/HP_N` is linear, so 70 HP
and 10 HP sit 0.15 apart while "1 counter away" versus "7 counters away" is the entire decision.

**`attack_static_row[16] = bench_damage/ATKDMG_N`** (`ATK_BENCH_DMG_COL`), pushing the keyword flags
to `17:46` and `F_ATK` to 46, hence `F_CARD` to 223.

**An attack's `damage` field is always the number dealt to the Active, never to a benched Pokémon.**
Measured over the engine's pool, 27 attacks reach the bench and every one states that figure only in
its oracle text, in three forms: *"this attack **also** does N damage to 1 of your opponent's Benched
Pokémon"* (19), *"put N damage counters on your opponent's Benched Pokémon"* (2, worth N×10 on a
single target), and *"also does N damage to each Benched"* (6). One (Pinpoint Dive) is a pure snipe
whose `damage` field is `0`.

`_attack_damage_ratio` previously scored bench targets with col 0, over-reporting all 27 — Phantom
Dive read **200 against a 70 HP benched Pokémon (ratio 2.0, KO flag set)** when the truth is at most
60. It now reads `ATK_BENCH_DMG_COL` for any `tgt_slot != 6`, and applies it **raw**: 25 of the 27
spell out *"Don't apply Weakness and Resistance for Benched Pokémon"*, which is the printed rule for
bench damage generally.
