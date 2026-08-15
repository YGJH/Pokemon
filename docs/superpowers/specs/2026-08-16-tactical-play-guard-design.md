# Tactical play guard — inference-time rules for attack, retreat and energy

**Date:** 2026-08-16
**Status:** approved, pending implementation plan

A second inference-time guard, `ptcg_il/play_guard.py`, in the shape of the
existing `ptcg_il/deck_guard.py`. Five rules covering the decisions two live
losses turned on: taking an attack that is already paid for, not throwing the
attacker's energy away to a retreat, attaching energy toward an attack cost
that is actually unmet, and not spending damage counters on a corpse.

Greedy inference only. The MCTS path is obsolete and is not modified.

---

## 1. Motivation

Two live losses, both analysed decision-by-decision from the Kaggle replays.

### 1.1 Episode 93320914 — lost 6–3 on prizes (`Result{reason: 1}`, turn 14)

POKEN (Dragapult ex) vs pepe (Archaludon ex). Total damage 750 vs 1150.

**The decisive decision, entry `[157]`.** POKEN had just played Jamming Tower
over Full Metal Lab, dropping Archaludon ex from 230/400 to **130/300**. Its own
Dragapult ex was at 320/320 with Phantom Dive (attack 154, 200 damage) legal in
the option list. Lethal, for 2 prizes, against the only Pokémon scoring prizes
for the opponent all game.

It chose `Play hand#8` = Boss's Orders, then at `[158]` gusted the KO-able
Archaludon ex **off** the active slot and replaced it with a fresh Duraludon
130/130. It attacked the Duraludon for 1 prize and left Archaludon alive at 70
HP. On turn 14 the opponent replayed Full Metal Lab (70 → 170), played Jumbo Ice
Cream (+80 → 250), gusted POKEN's benched Fezandipiti ex up and took the last 2
prizes.

**Wasted damage counters, entries `[140]`–`[144]`.** Phantom Dive places 6
counters on benched Pokémon. Relicanth reached 0 HP at `[139]`; POKEN then
selected it **five more times**, its displayed HP running 0 → −10 → −20 → −30 →
−40, while two Duraludon (130 HP each, the pre-evolutions of more Archaludon ex)
sat as legal targets. 50 damage discarded.

### 1.2 Episode 93323813 — lost 6–1 on prizes (`Result{reason: 1}`, turn 24)

POKEN vs GauravKr23 (Mega Abomasnow ex, 35 basic Water energy). Total damage
**340 vs 1640**.

Attack options offered to POKEN, by episode:

| episode | offered |
|---|---|
| 93320914 | `{153: 16, 154: 14}` |
| 93323813 | `{153: 16, 323: 1}` |

Phantom Dive was offered **zero times in 24 turns**. Its cost is `[2, 5]` —
one Fire and one Psychic. POKEN never had both on the same Pokémon, so every
attack it made was Jet Headbutt (cost `[0]`, 70 base, 40 after resistance),
three times, against a 350 HP attacker hitting for 200 every turn.

Energy in play, per turn, never exceeded **2** across the whole game, against
the opponent's Mega Abomasnow ex going 3 → 4 → 5 → 6 → 7 → 8. POKEN attached
roughly five energy all game and discarded five: four `DiscardEnergy` retreat
costs (t7, t9, t13, t21) and one Basic {D} Energy pitched to Ultra Ball.

**The turn-21 chain**, three consecutive decisions:

```
[178] attach Basic {R} Energy → Dragapult ex 150/320       correct
[179] options: Attack153 | Retreat | End  → chose Retreat  discards that energy
[182] active is now Budew 30/30, facing a 200-damage attacker; attacks for 0
t22   Budew is KO'd → a free prize
```

The same retreat-with-an-attack-available pattern fired at t7.

### 1.3 What the two losses have in common

| | 93320914 | 93323813 |
|---|---|---|
| POKEN turns with a Main select | 7 | 12 |
| turns where an Attack option appeared | 4 | 4 |
| turns POKEN attacked | 3 | 3 |
| retreats | 1 | 3 |
| **ended a turn with an attack available** | **0** | **0** |

The policy never flatly declines an attack. The failures are adjacent to it: it
does not price the energy a retreat discards, it does not play toward an attack
requirement, and it does not stop spending a resource once the target is dead.

---

## 2. Architecture

### 2.1 Module

`python/ptcg_il/play_guard.py`. NumPy and stdlib only, no torch at module
scope, copied **verbatim** into the submission bundle exactly as
`deck_guard.py` and `ref_map.py` are.

### 2.2 No new bundled artifact

Everything the guard needs is in `engine_card_features.npy`, which
`build_submission._build_engine_features_from_engine` already rebuilds from the
competition engine at packaging time and `main.py` already loads.

`ptcg_mine.cards.card_static_row` embeds up to three `attack_static_row` blocks
at cols `85:223` (`CARD_ATTACK_BLOCK_START = 85`, `F_ATK = 46`, `85 + 3·46 =
223 = F_CARD`). Within block *i*, the 12-slot `EnergyType` cost histogram sits
at `85 + 46·i + 1 : 85 + 46·i + 13`, divided by `ATKCOST_N = 5.0`. Recovery is
exact:

```python
off = CARD_ATTACK_BLOCK_START + i * F_ATK
cost_hist = np.rint(row[off + 1: off + 13] * ATKCOST_N).astype(int)
```

Verified two ways: against `all_attack()` for **all 1556 engine attacks** via
`attack_static_row`, and against **all 1555 card-row attack blocks** via
`card_static_row`. 0 mismatches in both. Use `np.rint`, never `int()` — `int()`
truncation on a float32 round-trip is the same class of bug as the KO-flag
`1.0 - 1e-6` guard in `deck_guard`.

The three-block cap is never reached: no Pokémon in this engine has more than
two attacks (557 with one, 499 with two). The implementation asserts this rather
than assuming it, because a future set adding a third attack would silently drop
it from the affordability math.

`engine_attack_features.npy` is therefore **not** a dependency of this guard —
the only thing it would have supplied is cost keyed by `attackId`, and no rule
needs that (see §3.3).

Prize value likewise needs no new data: `card_static_row` writes
`[ex, megaEx, tera, aceSpec]` at cols `48:52`, so "this KO is worth 2 prizes" is
`row[48] or row[49]`.

### 2.3 Energy types are read from `energyCards`, never from `energies`

An in-play Pokémon carries both an `energies` list and an `energyCards` list.
Whether `energies` holds `EnergyType` values or card ids **cannot be determined
from available data**: basic energy card ids 1–8 coincide exactly with their
`EnergyType` (id 2 = Basic {R} = FIRE = 2, id 5 = Basic {P} = PSYCHIC = 5), and
across **2864** Pokémon-with-energy observations in the two replays `energies`
is byte-identical to `energyCards[].id`, order included. Neither deck in either
game runs special energy, which is the only thing that would separate them —
Boomerang Energy is card id 9 with `energyType` 0 (COLORLESS), Legacy Energy is
card id 12 with `energyType` 10 (RAINBOW).

The guard therefore resolves each attached energy through
`energyCards[i].id → engine_card_features[id][12:24]`, the `energyType` one-hot.
This is correct under either interpretation and is the only correct reading for
special energy. `energies` is used for nothing.

One consequence, recorded deliberately: `OptionType.ENERGY` carries a `count`
field ("how many energy units does it correspond to"), so a card providing two
units would be counted as one. No such card appears in the observed data. The
error direction is conservative — under-counting available energy makes R2 and
R3 fire *less* often, never more aggressively.

### 2.4 Interface

```python
@dataclass(frozen=True)
class TacticsConfig:
    enable_lethal_end: bool = True      # R1a
    enable_lethal_gust: bool = True     # R1c
    enable_retreat: bool = True         # R2
    enable_attach: bool = True          # R3
    enable_dead_target: bool = True     # R4
    low_hp_frac: float = ...            # §4.1, derived
    margin: float = ...                 # §4.2, derived

@dataclass
class TacticsStats:
    r1a_fired: int = 0
    r1c_fired: int = 0
    r2_gated: int = 0
    r2_changed: int = 0
    r3_gated: int = 0
    r3_changed: int = 0
    r4_masked: int = 0
    decisions: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=50))

class PlayGuard:
    def __init__(self, config, engine_card_features=None, stats=None): ...
    def apply_mask(self, batch, obs_dict, *, sel_type: int, sel_ctx: int) -> None
    def rerank(self, logits, batch, obs_dict, *, sel_type: int, sel_ctx: int,
               max_count: int) -> "np.ndarray"
```

`obs_dict` is new relative to `DeckGuard` and is unavoidable: R2, R3 and R4 need
live per-Pokémon `hp`/`maxHp`, the attached `energyCards`, and the hand contents,
none of which survive into the batch. All three call sites already hold it.

`sel_type` and `sel_ctx` are passed explicitly rather than read from the batch,
for the same reason `DeckGuard` does it: `live_eval._sample_to_batch` drops 0-d
scalars, so a missing key would silently disable the guard.

### 2.5 Precedence — one owner of the final index

Two guards must not fight over the answer. The rule:

1. **Every hard rule is an `opt_mask` edit.** Masks compose with `&`, so
   ordering between `DeckGuard.apply_mask` and `PlayGuard.apply_mask` is
   irrelevant, and an in-place mask edit is automatically honoured by
   `select_multi`'s cloned picked-mask as well as the single-select
   `masked_fill`.
2. **`PlayGuard.rerank` returns adjusted logits, not an index.** Soft rules
   apply before the pick.
3. **`DeckGuard.pick` remains the sole owner of the returned index.** It is
   called last, on the reranked logits.

Call order at every site:

```
DeckGuard.apply_mask(batch, ...)          # deck-out mechanism A
PlayGuard.apply_mask(batch, obs, ...)     # R1a, R1c, R4
logits = model(batch)
logits = PlayGuard.rerank(logits, batch, obs, ...)   # R2, R3
index  = DeckGuard.pick(logits, batch, ...)          # deck-out mechanism B + argmax
```

Every hard rule carries a **never-empty floor**: if a mask would remove every
legal option, the rule is not applied. An empty mask turns argmax into an
arbitrary engine-legal pick, which is worse than the thing being prevented.

---

## 3. The rules

Engine constants used, from `cg/api.py`:

```
SelectType.MAIN = 0
SelectContext:  MAIN=0  SWITCH=3  TO_ACTIVE=4
                DAMAGE_COUNTER=13  DAMAGE_COUNTER_ANY=14  DAMAGE=15
OptionType:     CARD=3  ATTACH=8  ABILITY=10  RETREAT=12  ATTACK=13  END=14
AreaType:       HAND=2  ACTIVE=4  BENCH=5
EnergyType:     COLORLESS=0 ... RAINBOW=10  TEAM_ROCKET=11
```

`opt_scalar[:, 7]` is the existing KO flag (attack damage ≥ target current HP),
compared as `>= 1.0 - 1e-6`. `DeckGuard._arrays` already reads it; `PlayGuard`
reuses the same accessor rather than recomputing damage.

### 3.1 R1a — never end or retreat away a lethal attack (hard)

**Context:** `sel_type == MAIN`.
**Predicate:** some legal `ATTACK` option has the KO flag set.
**Action:** mask `END` and `RETREAT`.

Provable: both alternatives forfeit the attack outright, so replacing them with
a KO cannot be worse. This is the entire justification, and it is why the naive
form of this rule — "if lethal is legal, force the attack" — is **rejected**.
MAIN is re-presented after every sub-action (10–30 times per turn in the
replays) and attacking ends the turn, so forcing it on the first MAIN where
lethal appears forfeits every remaining attachment, ability and Supporter.

Honest scope note: the measured table in §1.3 shows POKEN ended a turn with an
attack available **zero** times in either game. R1a would not have changed
either replay. It is a cheap invariant, not the fix.

### 3.2 R1c — do not gust a KO-able target off the active slot (hard)

This is the actual `[157]` bug. The error was not playing Boss's Orders; it was
that doing so *moved the lethal target away*.

**Context:** `sel_type == MAIN`.
**Predicate:** an `ATTACK` option is legal with the KO flag set against the
opponent's current active, **and** that active is worth 2 prizes
(`engine_card_features[cid][48] or [49]`).
**Action:** mask `PLAY` options whose `opt_card_id` is in `GUST_CARDS`.

**This rule requires a small explicit card table**, and that is a deviation from
"generic rules only" that needs recording. The engine gives no signal at MAIN
that a `PLAY` option will change the opponent's active — the option is a bare
`{"index": 8, "type": "Play"}`. The card id is recoverable (`opt_card_id`), so
the guard only needs to know *which ids gust*. `GUST_CARDS` is exactly parallel
to `deck_guard.TRAINER_BURN_TABLE`, which exists for the same reason: the static
features do not parse the relevant text.

The table is derived, not guessed. Scanning every card's `skills[].text` (the
text lives on `skills`, **not** on `card.text`, which is empty for trainers) for
"opponent" plus "switch in" yields exactly eight cards:

| id | type | card | option that gusts |
|---|---|---|---|
| 1088 | ITEM | Prime Catcher (ACE SPEC) | `PLAY` |
| 1124 | ITEM | Pokémon Catcher (coin flip) | `PLAY` |
| 1182 | SUPPORTER | Boss's Orders | `PLAY` |
| 1204 | SUPPORTER | Lisia's Appeal (Basic only) | `PLAY` |
| 1218 | SUPPORTER | Team Rocket's Giovanni | `PLAY` |
| 221 | POKEMON | Meowstic (Ability) | `ABILITY` |
| 310 | POKEMON | Hop's Dubwool (on evolve) | — |
| 674 | POKEMON | Hariyama (on evolve) | — |

`GUST_CARDS` masks the five `PLAY` ids and Meowstic's `ABILITY`, whose only
effect *is* the gust. The two on-evolve gusts are deliberately **not** masked:
their gust is a "you may" rider on an evolution valuable in its own right, so
blocking the evolve would cost more than the gust does. Both present a `YES_NO`,
and the guard declines there instead.

Implementing this at the `SWITCH` select instead does not work: once the gust
card is played the switch is forced (`minCount = 1`) and *every* option moves
the active away. Verified at `[158]`, where all three options were opponent
bench slots.

The 2-prize condition keeps the rule narrow. Gusting away a 1-prize target you
could KO is often correct — you take the same one prize either way and choose a
better next-turn board. Gusting away a 2-prize target you could KO is not.

### 3.3 R2 — do not retreat a loaded attacker (soft)

**Context:** `sel_type == MAIN`, option type `RETREAT`.
**Gate, all of:**
- the active has at least one attached energy card;
- at least one `ATTACK` option is legal **in this same select**;
- `active.hp / active.maxHp > low_hp_frac`.

The second condition deliberately does **not** recompute affordability from the
cost histogram. The engine only offers an `ATTACK` option when its cost is
already paid, so the option list is a more accurate answer than anything the
guard could derive, and it needs no cost math at all. R3 is the only rule that
does cost arithmetic, because it reasons about a Pokémon that is *not yet* able
to attack.

**Action:** within `margin` logits of the top, demote `RETREAT` below any
non-retreat candidate. Ties keep the model's pick.

Soft, not hard: retreating a doomed attacker is genuinely right sometimes, which
is what the HP gate encodes and why the rule must not be absolute.

### 3.4 R3 — attach energy toward an unmet attack cost (soft)

**Context:** `sel_type == MAIN`, option type `ATTACH`.

For each attach option, resolve the hand card
(`obs["current"]["players"][me]["hand"][index]`) and the in-play target
(`inPlayArea`/`inPlayIndex`). Then:

- **energy type supplied** = `engine_card_features[hand_cid][12:24]` argmax,
  with `RAINBOW` treated as supplying any type and `COLORLESS` as generic;
- **target's cheapest unmet cost** = over the target's up-to-3 attack blocks
  (`engine_card_features[cid][85:223]`, each `F_ATK` wide, cols `1:13` the cost
  histogram), the attack minimising total remaining need, where need per colored
  type `t` is `max(0, cost[t] - attached[t])` plus the colorless requirement
  `cost[0]` absorbed by surplus.

**Score:** an option scores higher when the supplied type strictly reduces that
unmet need, and higher again when the target is the active.
**Gate:** `target.hp / target.maxHp > low_hp_frac` — your "if its hp is not
low". Investing energy in a Pokémon about to be knocked out is what turned
`[178]`–`[179]` into a two-card loss.
**Action:** re-rank within `margin`, as R2.

### 3.5 R4 — no damage counters on a dead target (hard)

**Context:** `sel_ctx in (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE)`.
**Predicate:** the option's referenced Pokémon has `hp <= 0`.
**Action:** mask it, subject to the never-empty floor.

Provable, and directly the `[140]`–`[144]` failure. Note the floor matters here:
when a spread effect must place more counters than there are live targets, every
remaining option *is* a dead target, and masking all of them would be worse than
placing them.

---

## 4. Thresholds

Neither number is invented. Both follow how `deck_guard`'s `deck_low = 10` and
`margin = 0.25` were established — measured, with the measurement recorded.

### 4.1 `low_hp_frac`

Definition: the HP fraction at which
`P(my active is knocked out on the opponent's next turn | hp / maxHp)` crosses
0.5, measured over the corpus's turn transitions — the same 1.75M-transition
scan that produced `deck_low`.

That reading is what makes both consumers true at once: below it, "don't invest
energy here" (R3) and "retreating is fine here" (R2) are both correct, because
the Pokémon is empirically about to die.

### 4.2 `margin`

Definition: p25 of the masked top-2 logit gap over exactly the decisions R2 and
R3 would gate, measured on the `val-repack-a16` shard.

**Fitted against the deployed 3-member ensemble, not a single seed.**
`data/il_baselines.json` records `ens-3-16` at 0.7056 `nontrivial_top1`, so the
arch-16 artifact that ships is an ensemble. `EnsemblePolicy` combines members as
`log(mean(softmax(masked logits)))`, which compresses the logit scale relative
to any one member, so a margin fitted on `checkpoints_a16_s7` alone would be the
wrong width for what actually runs.

R2 and R3 get **separate** margins if their gated gap distributions differ
materially; one shared value only if they do not.

**Out of scope, recorded:** the same argument implies `deck_guard`'s
`margin = 0.25`, fitted on a single checkpoint, is mis-scaled for the deployed
ensemble. Not changed here.

---

## 5. Call-site changes

| file | change |
|---|---|
| `python/ptcg_il/play_guard.py` | new |
| `python/ptcg_il/live_eval.py` | build `PlayGuard` beside `_ensure_guard`; insert `apply_mask` / `rerank` around the existing `DeckGuard` calls at `:234–255`; add `use_play_guard: bool = True` next to `use_deck_guard` so it can be switched off for the A/B |
| `scripts/build_submission.py` | copy `play_guard.py` verbatim beside `deck_guard.py` in `build_model_package`; add the import and construction to `MAIN_PY_TEMPLATE_GREEDY` |
| `main.py` (repo root) | the same two lines as the greedy template |

`MAIN_PY_TEMPLATE` (MCTS) is not modified. `play_guard.py` is still *copied*
into both bundles the way `deck_guard.py` is, so the sibling of
`test_deck_guard_shipped_in_both_packages` stays trivial and costs nothing.

---

## 6. What is explicitly not changing

- No retraining, no featurizer change, no shard rebuild, no new checkpoint. The
  featurizer is untouched, so `Policy.config["feat_dims"]` and every existing
  shard stay valid.
- `search_infer.py`, `MAIN_PY_TEMPLATE`, and the Rust PUCT tree. MCTS is
  obsolete per the owner's decision.
- `deck_guard.py`, including its `margin`.
- No archetype gate. The rules read engine data, not card ids — with the single
  exception of `GUST_CARDS` in R1c (§8).

---

## 7. Testing

`python/tests/test_play_guard.py`. Per the repo convention, every test that
counts occurrences **fails on zero examined**, so no test can pass vacuously.

**Unit, per rule:**
- R1a: `END` and `RETREAT` masked iff a lethal attack is legal; untouched when
  no attack is legal and when the attack is legal but not lethal.
- R1c: gust `PLAY` masked only when the opponent's active is both KO-able and
  worth 2 prizes; untouched at 1 prize; never empties the option set.
- R2: gate closed above `low_hp_frac` and open below; retreat untouched when the
  active has no affordable attack; ties keep the model's pick.
- R3: `COLORLESS` absorbed by surplus, `RAINBOW` supplies any type, unmet-cost
  targeting selects the right Pokémon, low-HP target excluded.
- R4: dead targets masked; floor holds when *every* target is dead.

**Data-contract tests:**
- Cost recovery equals `Counter(attack.energies)` for all 1556 engine attacks
  *and* all 1555 card-row attack blocks — catches a silent `int()` truncation
  regression in either layout.
- No Pokémon has more than 2 attacks, so the 3-block cap never truncates. The
  test fails loudly if a future set breaks this.
- The guard reads no `energies` key anywhere (§2.3). Asserted structurally, so
  a future edit reintroducing it fails.

**Mutation:** break each rule deliberately and confirm the test goes red — the
convention that caught the STOP-step term and the `torch.where` NaN in the RL
guards.

**Replay regression:** load the two episodes and assert the guard changes the
specific decisions — `93320914[158]` (R1c must prevent the gust) and
`93320914[140]` (R4 must refuse the 0-HP Relicanth). This is the only test that
shows the rules fix the observed failures rather than merely firing somewhere.

**Live A/B:** `live_eval` with `use_play_guard` on and off, same seeds and
opponents, 200 games each — the protocol the prior determinizer shipped under.
Ship only on better-or-equal across win rate, illegal-action count, and OOV.

---

## 8. Open items

1. ~~**`GUST_CARDS` (R1c).**~~ **Resolved 2026-08-16: approved.** The table is
   enumerated in §3.2 — five `PLAY` ids, one `ABILITY`, two on-evolve gusts
   handled at their `YES_NO` instead.
2. **Shipping on directional evidence.** 200 games/opponent will very likely
   leave overlapping CIs, as it did for the determinizer, which shipped on
   "better or equal on every measured dimension" rather than significance.
   Confirm the same bar applies.
3. **Which arch-16 members the `margin` fit runs against**, if `ens-3-16`'s
   `selection` record has gone stale relative to the checkpoints on disk.
4. **One margin or two** for R2 and R3 — decided by the measurement in §4.2.
