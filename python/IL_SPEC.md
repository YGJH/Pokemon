# Imitation Learning Spec — Pokémon TCG AI Battle

Grounded by downloading a real episode (`80169582.json`, day 2026-06-16) and inspecting the live
engine. Everything below is verified against actual data, not just the type stubs.

> **Status / authority.** This is a **background / design-rationale** document. The **authoritative,
> implementable spec is `TRANSFORMER_IL_SPEC.md`** (exact tensors, modules, training loop, corpus
> mining). Where the two differ, `TRANSFORMER_IL_SPEC.md` wins. Two deliberate updates supersede
> earlier wording here: (1) training uses **all expert decisions, won *and* lost games** (losses
> down-weighted), not winners-only — see that spec's §1.1; (2) the id-embedding **vocab is decoupled
> from the deck-archetype filter** and built from a wider card set to limit live OOV — §1.2 there. The
> feature list in B.1 below is illustrative; the exact `card_static` layout is Appendix A.3 there.

---

## Part A — The data: what's provided and how to get it

### A.1 Source & access
- **Manifest**: `archive/manifest.csv` — 38 daily rows (2026-06-16 … 2026-07-23), each pointing at a
  Kaggle dataset `kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD`.
- Per day: **~1,300–7,800 episodes**, **~21 GB** each dataset. Columns include `episode_count`,
  `top_avg_score`, `median_avg_score` (agent Elo-like scores → use for quality filtering).
- **Auth**: `~/.kaggle/kaggle.json` present; `kaggle` 2.2.4 + `kagglehub` 1.0.2 in `.venv`.
- **⚠️ Do NOT bulk-download.** 38 × 21 GB ≈ 800 GB. Each episode is a separate file
  (`<episodeId>.json`, ~0.3–9 MB). List then fetch selectively:
  ```bash
  export KAGGLE_CONFIG_DIR=~/.kaggle
  .venv/bin/kaggle datasets files  kaggle/pokemon-tcg-ai-battle-episodes-2026-06-27   # list (paginated)
  .venv/bin/kaggle datasets download kaggle/...-2026-06-27 -f <episodeId>.json -p out # one file
  ```
  Sample a few thousand episodes across days (favor recent high-`top_avg_score` days).

### A.2 Episode file format (`kaggle-environments` replay)
Top-level keys: `configuration, info, rewards, statuses, steps, specification, …`.
- **`rewards`**: `[r0, r1]` final outcome, `+1` win / `-1` loss / `0` draw. (Sample: `[-1, 1]`.)
- **`info.Agents` / `info.TeamNames`**: the two agent names (e.g. `shikisoukan` vs `cocoaAI`).
- **`statuses`**: `["DONE","DONE"]`.
- **`steps`**: `list[ step ]`, each `step = [record_P0, record_P1]`.
  Each **record** = `{observation, action, reward, status, info, visualize}`.
  - `status ∈ {ACTIVE, INACTIVE, DONE}` — `ACTIVE` = this agent was polled this step.
  - `observation` = **the exact `obs_dict` the `agent()` function receives** (see A.4).
  - `action` = `list[int]` the agent returned (option indices, or the 60-card deck).
  - `reward` = running reward (0 until terminal, then ±1).

> One kaggle-step = **one micro-decision** for the acting player. A full match ≈ dozens–hundreds of
> steps (the tiny sample game was 19; real games are longer). Both players' decisions are interleaved
> in the same `steps` list, distinguished by `status`.

### A.3 ⭐ The (observation → action) pairing rule (the off-by-one)
`kaggle-environments` records the action **one step after** the observation it responds to. Verified:
the deck prompt (`select=None`) is at `steps[0][P0]`, and P0's 60-card deck action appears at
`steps[1][P0]`. An action of index `3` at step 5 was invalid for step 5's 1-option select but valid
for step 4's 6-option MAIN — confirming the shift.

**Extraction rule (use this exactly):**
```
for i in range(len(steps) - 1):
    for p in (0, 1):
        if steps[i][p]["status"] == "ACTIVE":
            obs    = steps[i][p]["observation"]      # the obs the agent saw
            action = steps[i+1][p]["action"]         # what it chose (next step's record)
            label  = rewards[p]                      # +1 win / -1 loss / 0 draw  (episode-level)
            yield (obs, action, p, label)
```
Ignore `INACTIVE`/`DONE` records — their `observation.select` is a stale echo and their `action`
field is a carried copy. Only `ACTIVE` records are genuine decision points.

### A.4 What each observation contains (same schema as live engine — verified)
At a mid-game ACTIVE step, `observation` has: `current` (full `State`), `logs`, `select`,
`search_begin_input`, plus harness fields `remainingOverageTime`, `step`.
- **`current.players[yourIndex]`**: your `hand` is fully visible (ids), `active`/`bench` Pokémon with
  `hp/maxHp/energies/energyCards/tools/preEvolution`, `discard`, `prize` (some revealed), counts.
- **`current.players[1-yourIndex]`**: opponent `hand = None`; face-down `active[0]`/`prize[i] = None`
  → imperfect information. (Sample: my active id 721 hp150 visible; opp active id 722 hp90 with 1
  attached energy, but opp hand hidden.)
- **`logs`**: events since your previous decision (sample had 22 logs of types MOVE/DRAW/PLAY/ATTACH/
  TURN_START…) — the belief-update stream, incl. `*_REVERSE` for hidden opponent moves.
- **`select`**: the decision + **legal `option` list** you index into (`type, context, minCount,
  maxCount, option[], remainDamageCounter, remainEnergyCost, deck, contextCard, effect`).
- **`search_begin_input`**: opaque token (~386 chars) to seed the engine's forward-search planner.

### A.5 Labels, filtering & imbalance
- **Quality filter**: select **experts by team-level win-rate** (or high leaderboard/`top_avg_score`),
  then keep **all** their decisions from **both won and lost** games, down-weighting lost-game
  decisions rather than dropping them (`TRANSFORMER_IL_SPEC.md` §1.1). Per-game outcome is a noisy
  proxy for move quality and winners-only worsens covariate shift, so it is *not* the filter. (An
  earlier draft here said "keep only the winning agent"; superseded.)
- **Two heads, two datasets**:
  1. **Deck-building** (the `select=None` step): a fixed 60-card list per agent per game →
     mine frequent high-win-rate decklists; not part of the sequential policy.
  2. **In-game policy**: all other ACTIVE steps → `(obs, action)` pairs.
- **Decision-type imbalance**: most steps are `SelectType.MAIN`(0) & `YES_NO`(9); rarer contexts
  (SKILL_ORDER, DAMAGE_COUNTER_ANY…) are sparse. Track per-`context` counts; consider class-balanced
  sampling or per-context loss weighting.
- **Multi-select** targets (`maxCount>1`) are lists → keep order; train as an ordered sequence.

### A.6 Dataset build pipeline
1. Sample episode ids across days (bias to recent, high-score days); download individually.
2. Parse each with the A.3 rule → stream `(obs, action, player, win)` records.
3. Filter to winners / high-score agents.
4. **Featurize offline** (Part B.2) and shard to a columnar/tensor store (e.g. `.npz`/webdataset/
   parquet of ragged tensors) keyed by `(episode, step)`; store the option list length + legal mask.
5. Build the **card-embedding vocabulary** once from `all_card_data()` + `all_attack()` (ids → dense
   feature rows: type, hp, stage, retreat, weakness, attacks, energy cost, damage).
6. Hold out whole episodes (never split within a game) for val/test to avoid leakage.

---

## Part B — Model architecture for IL

The environment forces two non-negotiable design choices: (1) state is a **variable-length set of
entities**, (2) actions are a **variable-length, content-addressed option list**. So: an
**entity transformer encoder** + a **pointer action head**. (Matches the repo's
`entity-transformer-policies` and `micro-step-action-factorization` guidance.)

### B.1 Card / attack embedding table (shared feature backbone)
Precompute one row per card id from static data:
`embed(card_id) = Embedding(card_id) ⊕ MLP([hp, retreatCost, stage 1hot, cardType 1hot,
energyType 1hot, weakness 1hot, resistance 1hot, isEx, isBasic, #attacks, attack{damage,costlen}…])`.
Attacks get their own small table (`attackId → [damage, energy-cost multiset, flags]`); attack text
can be a frozen text-embed if desired (optional, v2). This makes the policy **generalize across cards**
rather than memorizing ids.

### B.2 State encoding — tokens (variable-length set)
Emit one token per entity, each = `card-embed ⊕ zone-embed ⊕ numeric-features ⊕ owner(me/opp)`:

| Token group | Per-token features |
|---|---|
| My/Opp **active + bench** Pokémon | hp/maxHp, energy counts per type, #tools, #preEvo, appearThisTurn, condition flags (poison/burn/sleep/paralyze/confuse) |
| My **hand** cards | card-embed + "in-hand" zone |
| **Discard** (both) | card-embed, count-pooled or per-card |
| **Prize/deck** | counts + revealed ids; a learned `UNKNOWN` token for face-down active/prizes |
| **Stadium** | card-embed (0/1) |
| **Global** token (CLS) | turn, turn parity, whose-move, per-turn flags (supporter/stadium/energy/retreat used), prize counts remaining, benchMax |

Encode with a **Transformer encoder** (self-attention over all tokens, no positional order; zone/owner
embeddings carry structure). Optionally add a **belief/history module**: a small GRU/attention over the
recent `logs` stream (embed each `Log` by type + card refs) to summarize opponent behavior since last
turn → concat to the CLS token. This is how the net exploits imperfect-info cues.

### B.3 Action head — pointer over options (core)
For each `Option` in `select.option`, build an **option token**:
`opt_embed = OptionType-embed ⊕ (the state token it references)`.
Resolve the reference via the option's fields → the entity token:
- `PLAY{index}` → my hand token `index`; `ATTACH{area,index,inPlayArea,inPlayIndex}` → source card
  token ⊕ target Pokémon token; `ATTACK{attackId}` → attack-embed ⊕ my active; `EVOLVE`, `RETREAT`,
  `ABILITY`, `CARD/ENERGY{area,index,playerIndex}` similarly; `YES/NO/NUMBER` → learned constants (⊕
  `number`/`count`/`remainEnergyCost` scalars).

Then cross-attend option tokens to the encoded state (queries = options, keys/values = state tokens)
and score each: `logit_j = w · MLP(opt_j ‖ pooled_state)`. **Softmax over exactly `len(option)`
logits** — no global action space, nothing to mask (engine already gives only legal options).

- **Single-select** (`min=max=1`, the common case): plain cross-entropy vs the expert index.
- **Multi-select** (`maxCount>1`): autoregressive pick-without-replacement — re-score remaining
  options after each pick, stop at expert length (respect `minCount`). Teacher-force the expert order.
  Loss = sum of per-pick CE.
- Optional **value head** off the CLS token (predict game outcome ±1) — free with the data, and the
  warm-start critic for later RL.

### B.4 Losses & training
- **Policy**: masked cross-entropy over the option pointer (per-context loss weights for rare types).
- **Value** (aux): MSE/BCE to episode `reward` of the acting player (discount by remaining turns
  optional).
- **Deck head** (separate model/step): treat as set-generation or just retrieve the empirically best
  legal 60-card list to start; upgrade later.
- Metrics: **top-1 / top-3 option accuracy per SelectType & context**, value AUC, and the real bar —
  **head-to-head live win-rate** vs (a) the sample random agent, (b) a frozen previous checkpoint, and
  (c) the built-in `search_begin/step` planner (`AGENT_SPEC.md` §5) as a scripted opponent. **Not** vs
  a held-out episode's *replay*: an interactive game diverges from any recording once our policy
  deviates, so recorded actions become illegal — you'd need the expert's policy, which replays don't
  provide.

### B.5 Suggested scale & sequence
- Start small: `d_model≈256`, 4–6 encoder layers, 8 heads; card-embed dim 128. Fits a single GPU;
  millions of `(obs,action)` pairs are available from a few days of episodes.
- **v1**: single-decision BC (no history) → validate the pointer head learns MAIN/attack/attach.
- **v2**: add the `logs` belief module + value head.
- **v3**: this IL policy becomes the **warm-start** for PPO self-play / AlphaZero using the engine's
  `search_begin/step` planner (see `AGENT_SPEC.md` §5, §8).

### B.6 Reference forward pass (what the submitted `agent()` runs)
```python
def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return best_deck                          # from the deck head / mined list
    tokens   = encode_state(obs.current, obs.logs)      # B.2 (+ belief)
    opts     = encode_options(obs.select.option, tokens) # B.3
    logits   = pointer_head(opts, tokens)
    return decode(logits, obs.select.minCount, obs.select.maxCount)  # argmax / AR multi-pick
```

---

## Open questions to resolve before coding
1. **Per-episode agent skill**: `rewards` is only ±1. Is a per-agent leaderboard score available in
   the episode `info`, or must we join names → a scores table? (Affects expert filtering.)
2. **Download budget**: how many episodes/days to pull locally (disk + Kaggle rate limits)?
3. **Deck strategy**: imitate a single dominant meta deck first, or learn deckbuilding jointly?
4. **History depth**: include the `logs` belief module in v1, or defer to v2?
