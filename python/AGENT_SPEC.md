# Pokémon TCG AI Battle — Environment I/O & ML Spec

Grounded in the actual `libcg.so` engine and `cg/api.py` dataclasses. Verified by running real
battles via `cg.game.battle_start` / `battle_select`.

> **Scope / authority.** This document is the **environment I/O reference** (Observation/Option/Log
> schema, static data, the search API, RL sketch) and is the shared foundation for the other two
> specs. For the **imitation-learning implementation**, `TRANSFORMER_IL_SPEC.md` is authoritative
> (exact tensors, modules, training, corpus mining); `IL_SPEC.md` is its design-rationale companion.
> The §6–7 modeling sketch below is intentionally high-level — where it differs from
> `TRANSFORMER_IL_SPEC.md`, that spec wins.

---

## 1. The environment at a glance

This is **not** a fixed-action-space environment. It is a **turn-based, imperfect-information,
two-player game** exposed as a **sequential decision process with a variable, per-step action set**.

- The engine drives the game and *pauses* whenever it needs a decision from a player.
- At each pause it hands the current player an **Observation** and a menu of **legal Options**.
- The agent replies with **indices into that Option list** (not raw actions).
- A single game turn contains **many** such micro-decisions (play → attach → target → attack → end…).

The controller loop each agent sees:

```
obs = battle_start(deck0, deck1)        # env → agent
while game not over:
    choice = agent(obs)                 # agent → env  : list[int] of option indices
    obs = battle_select(choice)         # env → agent  : next Observation (may be the opponent's)
```

> The same `agent()` is called for **both** players; `obs.current.yourIndex` (0/1) tells you who is
> to move. During a match your agent only sees turns where it is the deciding player.

---

## 2. INPUT — the Observation (`obs_dict` → `Observation`)

Top-level keys (`cg/api.py:Observation`):

| Field | Type | Meaning |
|---|---|---|
| `select` | `SelectData \| None` | The decision being asked. **`None` only at initial deck selection.** |
| `logs` | `list[Log]` | Events since the agent's *last* decision (both players' public actions). |
| `current` | `State \| None` | Full current game state. `None` only at deck selection. |
| `search_begin_input` | `str \| None` | Opaque token to seed the built-in search/planner (see §5). |

### 2.1 `State` (`current`) — the world
- `turn`, `turnActionCount`, `yourIndex` (0/1), `firstPlayer`, `result` (winner idx or −1).
- Per-turn flags: `supporterPlayed`, `stadiumPlayed`, `energyAttached`, `retreated`.
- `stadium` (0/1 card), `looking` (cards currently being revealed to you).
- `players: [PlayerState, PlayerState]`.

**`PlayerState`** — for **you** hand cards are visible; for the **opponent** `hand` is `None` and
face-down cards (`active[0]`, `prize[i]`) are `None`. This asymmetric visibility is the core of the
imperfect information.
- `active: [Pokemon|None]` (size 0/1), `bench: [Pokemon]`, `benchMax`.
- `deckCount` (count only), `discard: [Card]` (public), `prize: [Card|None]`, `handCount`,
  `hand: [Card]|None`.
- Active-Pokémon condition flags: `poisoned/burned/asleep/paralyzed/confused`.

**`Pokemon`**: `id, serial, hp, maxHp, appearThisTurn, energies:[EnergyType],
energyCards:[Card], tools:[Card], preEvolution:[Card]`.
**`Card`**: `id` (→ join to `EN_Card_Data.csv`), `serial` (unique within match), `playerIndex`.

### 2.2 `SelectData` (`select`) — the decision + legal action set
| Field | Meaning |
|---|---|
| `type` | `SelectType` (0=MAIN, 1=CARD, 4=ENERGY, 6=ATTACK, 9=YES_NO, …) |
| `context` | `SelectContext` — *why* you're choosing (49 values: SETUP_ACTIVE, DISCARD, ATTACH_TO, IS_FIRST, MULLIGAN, …) |
| `minCount`, `maxCount` | Choose **k** distinct indices, `minCount ≤ k ≤ maxCount` (min can be 0). |
| `option` | `list[Option]` — **the legal action set for this step**. |
| `remainDamageCounter`, `remainEnergyCost` | Budget counters for multi-pick contexts. |
| `deck`, `contextCard`, `effect` | Extra context (cards from deck being chosen; card triggering the effect). |

**`Option`** is a tagged union keyed by `type` (`OptionType`, 0–16). Fields present depend on type:
- `PLAY(7){index}`, `ATTACH(8){area,index,inPlayArea,inPlayIndex}`, `EVOLVE(9){…}`,
  `ABILITY(10)`, `RETREAT(12)`, `ATTACK(13){attackId}`, `END(14)`, `YES(1)/NO(2)`,
  `NUMBER(0){number}`, `CARD(3){area,index,playerIndex}`, `ENERGY(6){…,count}`, …

Real captured MAIN menu (attach hand-card #3 to active vs bench, play card #4, or end turn):
```json
[{"type":8,"area":2,"index":3,"inPlayArea":4,"inPlayIndex":0},
 {"type":8,"area":2,"index":3,"inPlayArea":5,"inPlayIndex":0},
 {"type":7,"index":4},
 {"type":14}]
```

### 2.3 `Log` — event stream since last decision
`LogType` 0–23: SHUFFLE, DRAW/DRAW_REVERSE, MOVE_CARD(_REVERSE), SWITCH, PLAY, ATTACH, EVOLVE,
ATTACK, HP_CHANGE, POISONED/BURNED/…, COIN, **RESULT** (`result`: 0/1/draw, `reason`: 1=no prizes,
2=deck-out, 3=no active, 4=card effect). `*_REVERSE` = opponent did something hidden. This is the
belief-update signal for tracking the opponent.

---

## 3. OUTPUT — the agent's action

`agent(obs_dict) -> list[int]`

1. **Deck selection** (`obs.select is None`, first call): return a **60-card deck** as a `list[int]`
   of card IDs, legal under TCG deckbuilding rules (≤4 copies except basic energy, ≤1 ACE SPEC, ≥1
   basic Pokémon, etc.). Fixed per submission (loaded from `deck.csv`).
2. **Every other step**: return **k distinct indices** into `obs.select.option`, with
   `minCount ≤ k ≤ maxCount`. Order can matter (e.g. `SKILL_ORDER`, `EVOLVE` source→target).

Engine `Select` error codes to respect: 4=count out of range, 5=index OOB, 6=duplicate index.

---

## 4. Static reference data

- `EN_Card_Data.csv` / `JP_Card_Data.csv` (~2,100 cards): id, name, stage, HP, type, weakness,
  resistance, retreat, move name/cost/damage, effect text.
- `all_card_data() -> [CardData]` and `all_attack() -> [Attack]` from the engine give the same,
  structured (types as enums, `attacks:[attackId]`, `energies:[EnergyType]`, `damage`, effect text).
- These are the **card embedding table** for any model.

---

## 5. Built-in search / planner API (model-based option)

`cg/api.py` exposes a determinized forward model for MCTS-style lookahead:
- `search_begin(agent_obs, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand,
  opponent_active, manual_coin)` — you **supply guessed hidden info** (opponent hand/deck/prize/active
  are unknown) → returns a `SearchState`.
- `search_step(search_id, select)` → next `SearchState`; `search_release` / `search_end` for memory.

This lets you roll out hypothetical lines. Useful as: (a) an MCTS baseline, (b) a target generator for
IL, (c) a value/rollout backbone in an AlphaZero-style loop. Hidden-info guesses can come from a learned
belief/opponent model.

---

## 6. Encoding spec for models

### 6.1 State encoding (entity/set-based, **not** a flat vector)
The natural representation is **variable-length sets of entities** → tokenize + attention
(this repo ships `entity-transformer-policies` guidance):

Token types (each = card-embedding(id) ⊕ learned type/zone embed ⊕ numeric features):
- **Your Pokémon** (active + bench): `id, hp/maxHp, energies (per-type counts), #tools, stage,
  appearThisTurn, condition flags`.
- **Opponent Pokémon**: same but no hidden fields; face-down active → a special `UNKNOWN` token.
- **Your hand** cards (ids), **discard** piles (both, multiset/counts), **stadium**, **prizes**
  (counts + revealed ids), **deck counts**.
- **Global scalars**: turn parity, per-turn flags, prize counts remaining, whose move.
- **Option tokens** (§6.2) attend over the state tokens.

### 6.2 Action head — pointer over the option list (critical)
The action set is **variable-length and content-addressed**, so use a **pointer/attention head**:
- Embed each `Option` (its `OptionType` + referenced card/target: look up the entity token it points
  to via `area/index/inPlayIndex/attackId`).
- Score each option token → softmax over exactly `len(option)` logits (mask nothing else — the engine
  already gives only legal options).
- Multi-select (`maxCount>1`): factorize as autoregressive pick-without-replacement, or top-k /
  Plackett-Luce, respecting `minCount`. (See the `micro-step-action-factorization` skill.)

This makes the policy invariant to option ordering/count and generalizes across card pools.

### 6.3 Reward (RL)
- **Sparse, terminal, zero-sum**: from the `RESULT` log — `+1` win / `−1` loss / `0` draw for
  `yourIndex`. Optionally discount by turn count.
- **Shaping candidates** (dense, optional): prize-cards taken/lost (each side races 6→0), KOs,
  damage dealt, board presence. Keep shaping small to avoid degenerate play.
- Episode = one full match; each agent contributes a trajectory of `(obs, chosen_indices)` **only at
  its own decision points**. Credit assignment must skip opponent/env-only steps.

---

## 7. Imitation Learning spec

**Data source**: `archive/manifest.csv` → daily Kaggle "episodes" datasets (replays; thousands/day,
with per-agent scores). Select experts by team-level win-rate / high `top_avg_score`, then keep **all**
their decisions from **won and lost** games (losses down-weighted), not winners-only — see
`TRANSFORMER_IL_SPEC.md` §1.1 for why (game outcome is a noisy quality proxy; winners-only worsens
covariate shift).

**Sample** = one decision point:
```
x = (State, SelectData, logs-since)          # everything visible to the acting player
y = chosen option index/indices              # the expert's action
```
**Pipeline**
1. Parse each episode replay → sequence of `(Observation, action)` for each player.
2. Featurize per §6.1/§6.2. Build the card-embedding table from `all_card_data()`.
3. **Loss**: cross-entropy of the pointer head over `option` (masked to legal set); for multi-pick,
   per-step CE over the autoregressive factorization.
4. **Deck imitation**: separately learn/copy the winning decklists (the deck is a fixed output, not
   part of the sequential policy) — mine frequent high-scoring 60-card lists as a start.
5. Metrics: top-1 action accuracy per `SelectType`/`context`, and **live** win-rate vs (a) the sample
   random agent, (b) a frozen checkpoint, (c) the §5 search planner. (A "teacher-forced" or
   vs-replay win-rate is not well-defined — an interactive game diverges from any recording as soon as
   the policy deviates; use live self-play against the baselines above.)

IL gives a strong warm-start policy; the option-pointer design lets one network cover all
`SelectType`/`SelectContext` decision kinds.

---

## 8. Reinforcement Learning spec

**Env wrapper** (build over `cg.game`): a Gym-style single-agent view.
```
reset()  -> battle_start(my_deck, opp_deck); return obs at my first decision
step(a)  -> battle_select(a); fast-forward through opponent/auto steps
            (opponent = frozen policy / self-play snapshot / search agent) until it is
            my decision again or terminal; return (obs, reward, done)
```
Key design points:
- **Action space**: `Discrete(len(option))` *dynamic per step* → pointer head, no global action id.
- **Opponent**: self-play with a league / snapshot pool (avoids overfitting one opponent);
  imperfect-info ⇒ mix opponents to keep belief modeling honest.
- **Algorithm**: PPO/IMPALA on the sparse terminal reward, **initialized from the IL policy**.
  Or AlphaZero/MuZero-style using the §5 search as the planner (determinize hidden info per rollout).
- **Hidden info**: either (a) train a belief/opponent model to fill `search_begin` guesses, or
  (b) treat unknowns as `UNKNOWN` tokens and let the recurrent/attention policy infer from `logs`.
- **Throughput**: the engine is a C++ lib called per decision. For scale, batch many concurrent
  matches across processes; a GPU-batched re-implementation of hot paths is the ceiling
  (`gpu-batched-env-simulation`, `pytorch-cuda-extension` skills) but not required to start.

**Suggested progression**: IL warm-start → PPO self-play fine-tune → optional search-augmented
(AlphaZero) policy/value with learned belief model.

---

## 9. Minimal interface contract (what any agent module must implement)

```python
def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return build_deck()                      # 60 card IDs, rules-legal
    logits = policy(encode_state(obs), encode_options(obs.select.option))
    return decode_selection(logits, obs.select.minCount, obs.select.maxCount)  # distinct indices
```
Submission ships `main.py` + `deck.csv` + `cg/` (engine bindings) + model weights; loaded from
`/kaggle_simulations/agent/` at run time.
```
```
