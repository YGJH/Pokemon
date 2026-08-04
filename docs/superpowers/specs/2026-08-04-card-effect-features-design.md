# Card-effect features for the IL featurizer

**Date:** 2026-08-04
**Status:** approved design, not yet implemented
**Affects:** `ptcg_mine`, `ptcg_il`, `ptcg_rl`; invalidates every existing checkpoint and shard.

---

## 1. Problem

The policy has **no learned card-id embedding**. `ptcg_il/model/cards.py` states the
contract explicitly:

> No learned id embeddings. Every card is represented purely by its static features.
> Two cards with identical features are truly identical to the model.

Card identity therefore reaches the model only through the 94-dim
`card_static_row` built in `ptcg_mine/cards.py`, which is entirely numeric: HP,
retreat, cardType, stage, energy type, weakness, resistance, four rule-box flags,
and three attacks' (damage, energy-cost histogram, cost count).

Nothing in that row encodes what a card *does*. Measured over all 1267 engine
cards:

| cardType | cards | distinct 94-dim rows |
|---|---:|---:|
| POKEMON | 1056 | 1004 |
| ITEM | 77 | **3** |
| SUPPORTER | 61 | **1** |
| STADIUM | 26 | **2** |
| TOOL | 27 | **3** |
| BASIC_ENERGY | 8 | 8 |
| SPECIAL_ENERGY | 12 | 6 |

164 Item + Supporter + Stadium cards collapse to **6** distinct rows. `Boss's
Orders` (the single most decisive card in the format — it drags a benched
Pokémon into the Active Spot) is **bit-identical** to `Cheren` ("Draw 3 cards.").
23.1% of all cards share a row with at least one other card, and 23.0% share it
with a card of a *different name*.

This is not weak generalisation. Without an id embedding there is no fallback
channel, so it is a hard information bottleneck: the model cannot learn to
distinguish these cards even by memorisation.

It bites the trained specialists directly. Each archetype's 60-card decklist:

| archetype | distinct card ids | distinct feature rows |
|---:|---:|---:|
| 1 | 19 | 11 |
| 17 | 23 | 17 |
| 25 | 21 | 13 |
| 16 | 23 | 14 |
| 36 | 19 | 14 |
| 21 | 24 | 12 |
| 0 | 22 | 13 |
| 95 | 18 | **7** |
| 27 | 17 | 9 |

A specialist playing a fixed deck sees roughly 60% of the cards that deck
actually contains.

### The data exists and is unused

`CardData.skills: list[Skill]` (`Skill.name`, `Skill.text`) and `Attack.text`
carry the full English oracle text and are read by nothing in the repo:

- All 61 Supporters, 77 Items, 27 Tools and 26 Stadiums have non-empty
  `skills[].text`.
- 421 of 1267 cards have at least one skill; all 421 have non-empty text.
- 1023 of 1556 attacks (65.7%) have non-empty text.

Examples: `Boss's Orders` → `"Switch in 1 of your opponent's Benched Pokémon to
the Active Spot."`; `Cheren` → `"Draw 3 cards."`.

### Feasibility is measured, not assumed

A throwaway 25-pattern regex prototype over `skills[].text` + `attacks[].text`,
appended to the existing row:

| cardType | distinct rows now | with keyword block |
|---|---:|---:|
| SUPPORTER | 1 | **18** |
| ITEM | 3 | **23** |
| STADIUM | 2 | **11** |
| TOOL | 3 | **16** |
| POKEMON | 1004 | 1029 |

27 Supporters still collided, caused by case-sensitivity bugs in the throwaway
patterns (`Search your deck` vs `search your deck`), not by missing data. The
curated list in §3 is expected to do better; §8 gates on it.

---

## 2. Scope

In scope, in priority order:

1. **Effect-keyword features** on cards and attacks (§3).
2. **Hand playability flags** (§4).
3. **KO-pressure scalars** on the CLS token (§5).
4. **Option damage preview** (§6).
5. **fp16 card features in shards** — pays for 1–4 (§7).
6. **Collapse the four duplicate `F_CARD = 94` literals** to one source of truth (§9).

Explicitly **out** of scope, with reasons:

- **Per-Pokémon `damage_taken`, `has_tool`, `num_attacks`.** Already present or
  linearly derivable: `poke_feat[0]` is `hp/HP_N` and `poke_feat[1]` is
  `maxHp/HP_N` against the *same* divisor, so `damage_taken` is one subtraction
  the first MLP layer performs for free; `poke_feat[2]` is already the hp ratio;
  `poke_feat[17]` is already `len(tools)/2`; unused attack slots in
  `card_static_row` are already zero-padded, which encodes the attack count.
  Only `num_abilities` is genuinely absent — folded into §3.
- **Hand keyword aggregation onto CLS.** The encoder self-attends over all 30
  hand tokens; a sum-pool of a per-token feature is exactly what attention can
  already compute. YAGNI.
- **`card_type` one-hot on option tokens.** Already present:
  `opt_card_feat` is the full card row, whose `[2:9]` slice is the cardType
  one-hot.
- **Backward compatibility with existing checkpoints.** Deliberately dropped;
  see §10.

---

## 3. Effect keywords

### 3.1 New module `ptcg_mine/keywords.py`

```python
KEYWORDS: tuple[tuple[str, re.Pattern], ...]   # ordered; index == feature column
K_EFFECT: int                                   # == len(KEYWORDS)

def effect_keyword_row(texts: Iterable[str]) -> np.ndarray:   # float32[K_EFFECT]
def ability_keyword_row(card) -> np.ndarray:                  # over card.skills[].text
def attack_keyword_row(attack) -> np.ndarray:                 # over attack.text
```

All patterns compile with `re.IGNORECASE`. The row is a binary multihot: 1.0 if
the pattern matches anywhere in the concatenated text, else 0.0. No counts — a
card that draws twice is not twice the draw card, and counts would need a
normalizer that nothing else in the row uses.

**`KEYWORDS` order is frozen and append-only.** The tuple index *is* the feature
column, so inserting a keyword in the middle silently repoints every column after
it — the same failure mode `archetypes.json` cluster ids have, documented in
CLAUDE.md. New keywords are appended; a retired keyword keeps its slot as a
dead always-zero column rather than freeing it. A module-level
`assert len(KEYWORDS) == K_EFFECT` and a test pinning the first N names guard
this.

### 3.2 The list (K_EFFECT = 29)

One list, applied to both text sources. The vocabulary of effects is the same
whether an effect is delivered by an ability or by an attack; what differs is the
delivery, and *position in the row* encodes that (ability block vs attack block).
A few columns will be near-always-zero on one side — that costs a handful of dims
and the MLP learns a zero weight, which is cheaper than curating two lists that
drift apart.

All patterns compile with `re.IGNORECASE`. This is load-bearing, not cosmetic:
oracle text capitalises the leading verb, so a case-sensitive `draw \d+ card`
misses `"Draw 3 cards."` — i.e. it misses `Cheren` entirely. A prototype without
it left 33 of 61 Supporters uncovered.

Card counts are measured over all 1267 engine cards (ability OR any attack).

| # | name | what it captures | cards |
|---:|---|---|---:|
| 0 | `draw` | draw cards | 75 |
| 1 | `search_deck` | search your deck | 144 |
| 2 | `deck_look` | look at the top N cards (dig without search) | 20 |
| 3 | `discard_own` | discard from your own hand / deck / board | 187 |
| 4 | `discard_opp` | force the opponent to discard | 42 |
| 5 | `hand_disrupt` | shuffle/reveal a hand back into a deck | 14 |
| 6 | `gust` | pull an opponent's benched Pokémon into the Active Spot | 21 |
| 7 | `switch_own` | switch your own Active | 23 |
| 8 | `heal` | heal damage / remove damage counters | 55 |
| 9 | `place_damage` | put damage counters | 38 |
| 10 | `bench_damage` | damage to benched Pokémon | 140 |
| 11 | `bench_accel` | put a Pokémon onto the Bench | 38 |
| 12 | `evolve_effect` | evolve out of turn / devolve | 20 |
| 13 | `status_offensive` | Poisoned, Burned (damage over time) | 53 |
| 14 | `status_lock` | Asleep, Paralyzed, Confused (cannot act) | 66 |
| 15 | `energy_accel` | attach Energy beyond the once-per-turn attachment | 72 |
| 16 | `energy_deny` | discard / move the opponent's Energy | 35 |
| 17 | `prevent_damage` | prevent or reduce damage / effects | 56 |
| 18 | `damage_scaling` | "does N more damage", "damage for each …" | 315 |
| 19 | `coin_flip` | outcome depends on a coin flip | 151 |
| 20 | `ability_lock` | Pokémon have no Abilities | 7 |
| 21 | `prize_effect` | take extra / fewer Prize cards | 36 |
| 22 | `retreat_effect` | modifies Retreat Cost | 11 |
| 23 | `once_per_turn` | "Once during your turn" ability economy | 81 |
| 24 | `target_pokemon` | the effect targets Pokémon | — |
| 25 | `target_energy` | the effect targets Energy cards | — |
| 26 | `target_trainer` | the effect targets Trainer/Supporter/Item/Stadium/Tool | — |
| 27 | `to_deck` | moves cards into / on top of / under a deck | — |
| 28 | `recover_discard` | return cards from the discard pile | 36 |

`status_offensive` / `status_lock` are deliberately two columns rather than the
five individual conditions: the decision-relevant split is
"chips damage each turn" vs "the Defending Pokémon cannot attack", and five
columns spend three dims to encode a distinction the model can also read off
`poke_feat[21:26]`, which already carries the *actual* condition flags per turn.

Columns 24–27 (`target_*`, `to_deck`) exist because measurement demanded them,
not by prior reasoning. Without them the residual in-deck collisions were all the
same shape — searches identical in mechanism but different in *what they fetch*:
`Energy Search` vs `Tera Orb`, `Team Rocket's Petrel` vs `Dawn`, `Pokégear 3.0`
vs `Bug Catching Set`, `Night Stretcher` vs `Sacred Ash`. Adding the four took
worst-case per-deck separation from 88.9% to 94.1%, and seven of nine decks to
100%.

### 3.3 Where the block lands

```
attack_static_row  : 14 → 14 + K_EFFECT            = 43     (F_ATK)
card_static_row[0] : 52 base
                   + K_EFFECT  ability keywords              (29)
                   + 1         n_abilities / 3.0             (1)
                   + 1         n_attacks   / 3.0             (1)
                   = 83
card_static_row    : 83 + 3 × F_ATK                = 212    (F_CARD)
```

Putting attack keywords in `attack_static_row` rather than only in the card row
is the load-bearing choice: `opt_attack_feat` is built from `attack_static_row`,
so **ATTACK options gain their effect text for free** — precisely the decision
where text matters most. `card_static_row` embeds three `attack_static_row`s
verbatim, so the block propagates to every card view at no extra code, preserving
the existing invariant that "the two views of an attack cannot drift apart"
(`ptcg_mine/cards.py` docstring).

`n_abilities` and `n_attacks` are divided by 3.0, matching the fixed-divisor
scheme the rest of the row uses (never z-scoring — see the
`il-input-normalization-scheme` note).

### 3.4 Propagation

`ptcg_mine/mine.py:419` already builds `engine_card_features.npy` /
`engine_attack_features.npy` as mining artifacts, and every consumer
(`shard_writer`, `rollout`, `arena`, `elo_calibrate`, `search`, `search_infer`,
the submission bundle) loads those `.npy` files. Regexes therefore run **once
during mining** and never at inference — confirming the zero-runtime-cost
requirement. Nine `*_card_feat` keys pick the new dims up with no further change.

`ptcg_mine/stamp.py` must add `keywords.py` to the mine stage's fingerprint
source list, so editing a pattern invalidates the artifact instead of silently
serving a stale one. `featurizer.py` is already in the shards fingerprint, so
shards rebuild automatically.

---

## 4. Hand playability flags — `F_HAND` 2 → 7

`hand_feat` is currently `[idx / H_MAX, dup_count / COUNT_N]`. Append five
deterministic legality flags:

| col | flag | derivation |
|---:|---|---|
| 2 | `can_bench` | card is a Basic Pokémon (`card_row[CARD_FEAT_BASIC_COL] > 0.5`) **and** `len(bench) < benchMax` |
| 3 | `can_evolve` | some in-play Pokémon of mine is this card's pre-evolution **and** that Pokémon has `appearThisTurn == False` |
| 4 | `can_attach_energy` | card is BASIC_ENERGY or SPECIAL_ENERGY **and** `state["energyAttached"] == False` |
| 5 | `can_play_supporter` | card is SUPPORTER **and** `state["supporterPlayed"] == False` |
| 6 | `can_play_stadium` | card is STADIUM **and** `state["stadiumPlayed"] == False` |

`_build_hand_tokens` currently takes only `(state, your_index)` and must also
receive the card-feature table (for cardType / basic) and the pre-evolution map.

### 4.1 `can_evolve` needs a new artifact

`CardData.evolvesFrom` is a **name string**, and names are not in the 94-dim row.
Mining must emit `data/evolution_map.npy`: `{card_id: [pre_evolution_card_ids]}`,
built by resolving `evolvesFrom` against `{card.name: [card_ids]}`. Several
distinct card ids share a name (reprints), hence the list value.

The tempting shortcut — "this card is Stage 1 and some in-play Pokémon is Basic"
— is wrong: it fires across unrelated evolution lines and would teach the model
that an illegal evolution is available. Do the name resolution.

### 4.2 Why this is not redundant with the option list

At a `MAIN` select the engine already enumerates the legal plays, so legality is
in principle readable from the options. The value here is at *non-MAIN*
decisions and for lookahead: while choosing which card to discard, the model can
see that one of the candidates is a Supporter it has not yet played this turn.

---

## 5. KO-pressure scalars — `F_GLOBAL` 93 → 97

Appended at index 93; `cls_feat[87:88]` and `[88:89]` are read by
`ptcg_il/model/embed.py`, so appending past 93 disturbs nothing.

| col | feature |
|---:|---|
| 93 | my best **affordable** attack damage × weakness ÷ opp active current HP, clipped to [0, 2] |
| 94 | `can_ko_opp` = 1.0 if col 93 ≥ 1.0 |
| 95 | opp best attack damage (affordability ignored) × weakness ÷ my active current HP, clipped to [0, 2] |
| 96 | `opp_can_ko_me` = 1.0 if col 95 ≥ 1.0 |

The asymmetry is deliberate: on my turn I can only use an attack I can pay for
right now, whereas the opponent gets a full turn to attach Energy first.

**Affordability is exactly computable.** The attack's energy-cost histogram is
`attack_static_row[1:13]` and the Pokémon's attached energy is `poke_feat[3:15]`,
both 12-wide over `EnergyType` with COLORLESS at index 0. An attack is
affordable when, for every non-colorless type `t`, `attached[t] >= cost[t]`, and
`sum(attached) >= sum(cost)`. This is exact except for the 12 special Energy
cards that provide multiple or arbitrary types (Prism, Legacy, Neo Upper, …),
which the engine resolves and this heuristic does not.

### 5.1 Two caveats that must be stated, not buried

1. **Base damage is not real damage.** 284 attacks carry `does N more damage` or
   `damage for each …` text. A KO check computed from `attack.damage` alone is
   systematically wrong on those. This is why cols 93/95 are continuous ratios
   and the booleans are derived from them, rather than a bare boolean being the
   only signal: a ratio of 0.85 that is really lethal is a much smaller error
   than a `can_ko = 0` that is really 1. The `damage_scaling` keyword (§3.2 #14)
   is in the same row, so the model can learn to distrust the ratio when it fires.
2. **The weakness multiplier is unverified.** Weakness resolution lives in
   `libcg.so`; nothing in the Python bindings states whether it is ×2 (standard
   TCG) or a flat bonus. **Implementation must verify empirically** — run one
   attack into a known weakness and read the resulting `Log.value` — before
   hardcoding a multiplier. Assuming ×2 and being wrong poisons four CLS dims
   and both option dims of §6 with no error anywhere.

---

## 6. Option damage preview — `F_OPT` 6 → 8

| col | feature |
|---:|---|
| 6 | for `OptionType.ATTACK`: damage × weakness ÷ target's current HP, clipped to [0, 2]; 0.0 otherwise |
| 7 | `is_lethal` = 1.0 if col 6 ≥ 1.0; 0.0 otherwise |

Target is the Pokémon at `opt_tgt_idx` when the option sets one (snipe attacks),
otherwise the opponent's Active. Same two caveats as §5.1 apply.

`_build_option_tokens` does not currently receive the engine feature tables or
`poke_feat`, and must.

---

## 7. fp16 card features in shards

§3 takes `F_CARD` from 94 to 212 (2.26×) and `F_ATK` from 14 to 43 (3.07×).
Card-feature keys are **89.8%** of an uncompressed shard, measured on
`data/shards/test-00000.npz` (18879 samples, 2158.85 MB uncompressed):

| key | MB |
|---|---:|
| `discard_card_feat` | 851.82 |
| `opt_card_feat` | 454.30 |
| `log_card_feat` | 227.15 |
| `hand_card_feat` | 212.96 |
| `poke_card_feat` | 85.18 |
| `prize_card_feat` | 85.18 |
| `context`/`effect`/`stadium` | 21.30 |
| `opt_attack_feat` | 67.66 |
| everything else | 153.30 |

Left alone this would take the shard to ~4.9 GB, giving back — and then some —
the 17.8 GB → 5.9 GB peak-PSS win recorded in CLAUDE.md. Storing the ten float
card/attack feature keys as **float16** instead absorbs almost all of it:

| | now | after |
|---|---:|---:|
| card feature keys | 1937.9 MB | 2184.9 MB |
| `opt_attack_feat` | 67.7 MB | 103.9 MB |
| everything else | 153.3 MB | 153.3 MB |
| **total** | **2158.9 MB** | **~2442 MB** |

2.26× the card features for **+13%** memory; projected peak PSS 5.9 → ~6.7 GB.

Without fp16 the same change costs +127%. The fp16 store is therefore not an
optional optimisation attached to this work — it is what makes the feature
expansion affordable, and §8.4 gates on it.

**Fallback if +13% proves unacceptable:** drop keyword columns 24–27
(`target_*`, `to_deck`), giving K_EFFECT = 25, F_CARD = 196, ~2267 MB (+5%).
The cost is worst-case per-deck separation falling from 94.1% to 88.9% and two
decks dropping off 100%. Taking that trade is a decision to re-open, not a
default to apply silently.

### 7.1 The read path already handles it

`ShardDataset.__getitem__` ends every non-int, non-bool key with
`torch.from_numpy(...).float()` regardless of stored dtype, and
`build_mmap_cache` copies arrays through `np.load`/`np.save` without touching
dtype. So the change is confined to the **write** side: cast the ten keys in
`ptcg_il/shard_writer.py::_write_shard` before `np.savez_compressed`.

### 7.2 Precision is not a concern, and one test must prove it

Every value in these rows is a one-hot, a small integer count, or a fixed-divisor
ratio. The tightest case is `attack.damage / 350.0`: adjacent real damage values
differ by 10, i.e. 2.9e-2 in normalized units, against fp16 relative precision of
about 4.9e-4 near 0.37. Two orders of margin.

This matters because `ptcg_il/qa.py::_attachment_collision` keys feature rows by
`tobytes()`. fp16 rounding two genuinely distinct cards onto one row would make
that gate over-report collisions. A test must assert that the fp32 → fp16 → fp32
round trip preserves the number of distinct rows across all 1267 engine cards.

---

## 8. Success criteria

Ordered from cheapest to most expensive; each gates the next.

1. **Separation, offline, no training.** After §3, the distinct-row count over
   all 1267 engine cards must reach the floors below. The "measured" column is
   what the §3.2 list actually achieves — the floors sit under it so that a
   later pattern edit that regresses separation goes red without the gate being
   so tight that any refactor trips it.

   | cardType | before | floor | measured |
   |---|---:|---:|---:|
   | SUPPORTER | 1 | 25 | **43** |
   | ITEM | 3 | 25 | **57** |
   | STADIUM | 2 | 12 | **16** |
   | TOOL | 3 | 12 | **18** |

   Perfect separation is not the target and would be the wrong target: a binary
   multihot genuinely cannot split `Cheren` ("Draw 3 cards") from `Carmine`
   ("Discard your hand and draw 5 cards"), which differ only in magnitude. That
   is the accepted cost of an interpretable list. The **reviewed coverage
   report** remains the substantive gate: it lists every card whose keyword row
   is all-zero and every surviving collision group, and no whole cardType may be
   uncovered. Measured all-zero rows: 267 of 1267, of which 238 are Pokémon with
   no ability and no attack text and 8 are the Basic Energies — correctly zero,
   because those cards genuinely have no effects.

2. **Per-deck separation.** For each of the nine `self_ids` archetypes, distinct
   feature rows must cover ≥ 90% of distinct card ids, and every residual
   collision must be listed and individually justified as a genuine effect twin
   rather than a missing keyword.

   | archetype | ids | before | measured | coverage |
   |---:|---:|---:|---:|---:|
   | 1 | 19 | 11 | 19 | 100% |
   | 17 | 23 | 17 | 23 | 100% |
   | 25 | 21 | 13 | 20 | 95.2% |
   | 16 | 23 | 14 | 23 | 100% |
   | 36 | 19 | 14 | 19 | 100% |
   | 21 | 24 | 12 | 24 | 100% |
   | 0 | 22 | 13 | 22 | 100% |
   | 95 | 18 | 7 | 18 | 100% |
   | 27 | 17 | 9 | 16 | 94.1% |

   The two residuals are the same pair both times — `Buneary` and `Dunsparce`,
   which have **empty** skill text and no attack text. They are irreducible: no
   keyword list can separate two cards that carry no text, and separating them
   would require reinstating an id embedding. Accepted.
3. **Shard size.** The rebuilt `test-00000.npz` uncompressed total must be
   ≤ 2158.85 MB (today's figure).
4. **Peak PSS.** A 60-step training run over the full corpus must stay at or
   below today's 5.9 GB.
5. **Held-out top-1.** Retrain the archetype-0 and archetype-2 specialists and
   compare against `data/il_baselines.json` on the **test** split. This is the
   real gate; §1's bottleneck argument predicts the largest gains on decisions
   whose options are Trainer cards.

Criteria 1–4 are mechanical and must all pass before spending training compute
on 5.

---

## 9. Files touched

| file | change |
|---|---|
| `ptcg_mine/keywords.py` | **new** — `KEYWORDS`, `K_EFFECT`, row builders |
| `ptcg_mine/cards.py` | import `F_CARD`/`F_ATK` from `ptcg_il.featurizer`; add the §9.1 asserts; extend both row builders |
| `ptcg_mine/artifacts.py` | import `F_CARD` instead of redefining `94` |
| `ptcg_mine/mine.py` | emit `evolution_map.npy` |
| `ptcg_mine/stamp.py` | add `keywords.py` to the mine fingerprint sources |
| `ptcg_il/featurizer.py` | **owns** `F_CARD` 94→212 and `F_ATK` 14→43; `F_HAND` 2→7, `F_GLOBAL` 93→97, `F_OPT` 6→8; extend `_build_hand_tokens`, `_build_cls_features`, `_build_option_tokens` and their call sites |
| `ptcg_il/model/cards.py` | import `F_CARD`/`F_ATK` instead of redefining `94`/`14` |
| `ptcg_il/model/embed.py` | `F_HAND` 2→7, `F_GLOBAL` 93→97 |
| `ptcg_il/model/pointer.py` | `F_OPT` 6→8 |
| `ptcg_il/model/belief.py` | none — already imports `F_CARD` from `model/cards.py` |
| `ptcg_il/shard_writer.py` | cast the ten card/attack float keys to fp16 in `_write_shard` |
| `tests/`, `python/tests/` | §11 |

`CARD_FEAT_BASIC_COL = 9` stays valid: the base block `[0:52]` is unchanged and
all new dims are appended. It is the only hardcoded index into a card row
(`ptcg_il/search_infer.py:51`, `ptcg_rl/mcts_train.py:1251`).

### 9.1 Single source of truth for `F_CARD`

`F_CARD = 94` is currently written as a literal in four files
(`ptcg_mine/cards.py:31`, `ptcg_mine/artifacts.py:12`, `ptcg_il/featurizer.py:52`,
`ptcg_il/model/cards.py:20`). Changing it in three of four produces a silent
shape mismatch far from the edit.

**`ptcg_il/featurizer.py` owns the definition**; `ptcg_mine/cards.py`,
`ptcg_mine/artifacts.py` and `ptcg_il/model/cards.py` import it.

The direction is forced by the submission bundle, and the obvious choice is the
wrong one. `scripts/build_submission.py:642` copies `ptcg_il/featurizer.py` into
`submission/model/featurizer.py`, rewriting only its `ptcg_il.ref_map` import
(line 627). **`ptcg_mine` is not shipped** — `build_submission.py:873` imports it
on the *host* at build time to regenerate the static tables. So a
`from ptcg_mine.cards import F_CARD` inside `featurizer.py` would import fine in
the repo and `ImportError` inside the Kaggle bundle, which is exactly the class
of failure that only shows up after submission. Making the featurizer the owner
keeps it self-contained and needs no change to the bundle builder.

Inverting the package direction (`ptcg_mine` → `ptcg_il`) is safe here:
`ptcg_il/featurizer.py` is pure NumPy, so it does not violate `ptcg_mine`'s
no-PyTorch rule. It is also the honest layering — the featurizer *defines* the
tensor contract and mining produces tables that must conform to it.

**`K_EFFECT` cannot flow the other way.** `featurizer.py` must not import
`ptcg_mine.keywords` (same bundle problem), so it carries `F_CARD = 212` /
`F_ATK = 43` as literals, and `ptcg_mine/cards.py` asserts agreement at import:

```python
assert F_ATK == 14 + K_EFFECT, (
    f"F_ATK={F_ATK} in ptcg_il.featurizer disagrees with "
    f"K_EFFECT={K_EFFECT} in ptcg_mine.keywords (expected {14 + K_EFFECT})"
)
assert F_CARD == 52 + K_EFFECT + 2 + 3 * F_ATK, (
    f"F_CARD={F_CARD} in ptcg_il.featurizer disagrees with "
    f"K_EFFECT={K_EFFECT} (expected {52 + K_EFFECT + 2 + 3 * F_ATK})"
)
```

Appending a keyword without updating the featurizer then fails loudly at mining
time with the two numbers named, instead of writing a short row that every
downstream shape check accepts.

---

## 10. What this breaks

Accepted deliberately (decided 2026-08-04); no compatibility shim is built.

- **Every checkpoint becomes unloadable.** `CardFeaturizer.mlp`'s first layer
  goes `[94→256]` to `[212→256]`, and `BeliefHead.in_proj` (`ptcg_il/model/belief.py:57`)
  changes width with `F_CARD`. This covers `checkpoints/`, `checkpoints_a0`,
  `_a1`, `_a17`, `_a27`, `_generalist`, and all `*_mcts` variants.
- **`data/il_baselines.json` must be re-recorded** — it is SHA-1-pinned to the
  checkpoints that produced it and the RL gate refuses a mismatched record.
- **`arena_ratings.json` is void** — the rated roster no longer loads.
- **Full pipeline re-run.** Mine re-runs Phase 2 (`keywords.py` enters the
  fingerprint); shards rebuild (`featurizer.py` is already in theirs).
  The `.mmap-cache/` is size/mtime-invalidated and rebuilds itself.
- **Old shards are unusable.** They carry 94-wide rows; the model expects 212.
  This fails loudly at the first forward pass, which is the desired behaviour.

---

## 11. Testing

Following the repo's rule that tests counting occurrences must **fail on zero
examined**, and that new guards are validated by deliberately breaking the thing.

- `keywords.py`: per-keyword unit tests with a real card text for a positive and
  a near-miss negative each; a test pinning `KEYWORDS[:5]` names and `K_EFFECT`
  so a mid-list insertion goes red; a coverage test asserting every cardType has
  at least one non-zero keyword row.
- `cards.py`: `card_static_row` length is 212; `card_static_row[83:126]` equals
  `attack_static_row(first_attack)` — the no-drift invariant; base block `[0:52]`
  is bit-identical to the pre-change implementation on a fixture card.
- Separation: the §8.1 and §8.2 thresholds, asserted over the real engine tables.
  Must fail on zero cards examined.
- fp16: the §7.2 round-trip distinct-row test.
- `F_CARD` single-source test (§9.1).
- Featurizer: `can_evolve` fires for a real evolution pair and does **not** fire
  across two unrelated lines of matching stage; `can_attach_energy` is 0 when
  `energyAttached` is True; affordability is 0 for an attack whose colored cost
  exceeds attached energy of that colour but whose total is met.
- Weakness multiplier: an engine-backed test that reads `Log.value` and asserts
  the constant used in §5/§6 matches it. This is the guard for §5.1 caveat 2.
- Shape propagation: one end-to-end `featurize()` call asserting every
  `*_card_feat` key is `[..., 212]` and `opt_attack_feat` is `[..., 43]`.

---

## 12. Open items for implementation

1. Verify the weakness multiplier against the engine before writing §5/§6 (§5.1).
   This is the only remaining unknown in the design.
2. Confirm `evolvesFrom` resolves for every evolution card in the nine `self_ids`
   decklists; log any unresolved name rather than silently emitting `can_evolve = 0`.

The 29 regexes are authored and measured (§3.2, §8.1, §8.2) — they are inputs to
the plan, not open work.
