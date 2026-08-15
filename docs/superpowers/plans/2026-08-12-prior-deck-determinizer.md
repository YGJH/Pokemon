# Prior-Based Opponent Deck Determinization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the learned belief heads' inference-time role with a pure Bayesian posterior over all 362 mined archetypes (Phase 1), then remove the belief machinery from the codebase entirely — heads, auxiliary loss, labels, and the game-log history encoder (Phase 2).

**Architecture:** Phase 1 adds one torch-free module `python/ptcg_il/deck_prior.py` owning the posterior, the observation accumulator, and the caller-facing `OpponentDeckPredictor`. `ArchetypePosterior` relocates here from `ptcg_rl/belief.py` (fixing IL→RL backwards layering). Three call sites — the Kaggle bundle, `live_eval`, and `ptcg_rl.train` — switch to it, and `ptcg_il/belief_infer.py` plus `search_infer.predict_opponent_deck` are deleted. Phase 2 then deletes both belief objects: `BeliefHeads` (auxiliary prediction heads + `belief_loss` + `bel_*` labels) and `BeliefModule` (the game-log encoder whose output is added to the CLS token).

**Phase ordering is not optional.** Phase 1 removes the only *consumers* of the belief heads; Phase 2 removes the heads. Doing Phase 2 first would leave the determinizer with no opponent model at all.

**Tech Stack:** Python 3.11, uv, pytest, numpy (lazily, for sampling only), ctypes FFI to `libptcg_search.so`.

**Spec:** `docs/superpowers/specs/2026-08-12-prior-deck-determinizer-design.md`

## Global Constraints

- **`python/ptcg_il/deck_prior.py` must not import torch**, and must import numpy only lazily inside `sample_templates`. It is copied flat into the Kaggle bundle as `model/deck_prior.py` and is imported at agent load time.
- **`epsilon` stays `DEFAULT_EPSILON = 1e-3`.** Measured optimal across all tested variant regimes; not a tuning knob in this change.
- **Default hypothesis set is all 362 archetype ids**, not `archetypes.json:opp_ids`.
- **Default prior is mined `frequency`.**
- **`template()` never abstains and never returns `[]`.** No confidence threshold, no mirror fallback.
- **Accumulate observed cards by per-card-id maximum, never by sum.**
- Run tests from the repo root: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest <path>`. The env var is required (ROS steals plugin autoload). `pyproject.toml` sets `pythonpath = [".", "python"]`.
- **`tests/` and `python/tests/` must be run separately** — they share basenames and collection fails if combined.
- Tests that count occurrences must fail on zero examined. Validate each new guard by **mutation**: break it deliberately, confirm the test goes red, restore.
- **NO GIT COMMANDS AT ALL in this run.** The repository owner has explicitly declined commits and branching. Every task's "Commit" step is **skipped** — finish at the passing test. Do not run `git commit`, `git add`, `git mv`, `git rm`, `git checkout`, `git stash`, or any other git subcommand that writes. Use plain `mv` and `rm` for file moves and deletions. Read-only inspection (`git diff`, `git log`, `git show`) is fine. Changes accumulate in the working tree and the owner commits when they choose.
- **A training run may be in flight.** At the time of writing, `ptcg_il.cli train --archetype-self 1 --seed 2` had been running for 5+ hours against the existing shards. Phase 2 must not disturb it: no task rebuilds shards until Task 15, and every code change must keep reading the *existing* shards (which still carry `bel_*` and `log_*` keys) without error. Check `nvidia-smi` before starting Task 15.
- **Checkpoints trained with belief weights must keep loading.** Task 10 establishes this and every later task preserves it. A checkpoint from the in-flight run carries `belief.*` and `belief_heads.*` parameters that the post-removal `Policy` does not define.

---

### Task 1: `deck_prior.py` foundation — posterior + prior helpers

Relocate `ArchetypePosterior` and add the two helpers that configure it for the 362-way frequency-prior case.

**Files:**
- Create: `python/ptcg_il/deck_prior.py`
- Delete: `python/ptcg_rl/belief.py`
- Modify: `python/ptcg_il/belief_infer.py:252` (import repoint, temporary — file is deleted in Task 8)
- Create: `python/tests/test_deck_prior.py` (from `python/tests/test_rl_belief.py`)
- Delete: `python/tests/test_rl_belief.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `DEFAULT_EPSILON: float = 1e-3`
  - `CONFIDENT: float = 0.80`
  - `DECK_SIZE: int = 60`
  - `ArchetypePosterior(archetypes: dict, opp_ids: Sequence[int] | None = None, priors: Sequence[float] | None = None, epsilon: float = DEFAULT_EPSILON)` with methods `.posterior(observed: Iterable[int]) -> dict[int, float]`, `.most_likely(observed) -> tuple[int, float]`, `.is_confident(observed, threshold=CONFIDENT) -> bool`, `.deck_template(observed, threshold=CONFIDENT) -> list[int] | None`, and attribute `.opp_ids: list[int]`
  - `_representative_of(archetypes: dict, archetype_id: int) -> list[int]`
  - `all_archetype_ids(archetypes: dict) -> list[int]`
  - `frequency_prior(archetypes: dict, ids: Sequence[int]) -> list[float]`

- [ ] **Step 1: Create the new module by moving the old one**

```bash
cd /home/charles/Documents/Pokemon
mv python/ptcg_rl/belief.py python/ptcg_il/deck_prior.py
mv python/tests/test_rl_belief.py python/tests/test_deck_prior.py
```

If the worker lacks git permission, use `mv` instead — the file content matters, not the git history.

- [ ] **Step 2: Repoint the moved module's docstring and the two importers**

In `python/ptcg_il/deck_prior.py`, replace the module docstring's first line and the cross-reference paragraph. The old docstring names `ptcg_rl.belief` and `ptcg_il.belief_infer.OpponentDeckOracle`, both of which stop existing.

Replace the whole docstring (lines 1–28 of the moved file) with:

```python
"""Opponent deck determinization from the mined archetype prior — no learned model.

``ptcg_search``'s determinizer (``guessing.rs``) needs one plausible
``opponent_deck_template`` in **engine card ids**: it subtracts the cards it can
already see and deals the remainder into the opponent's deck, hand and prizes.

The template comes from a categorical posterior over every archetype in
``archetypes.json``:

```
P(deck_j | observed) ∝ P(deck_j) · Π_{c observed} P(c | deck_j)
```

Two things make this work, and neither is a neural network.

**The prior carries the early turns.** Mined ``frequency`` is heavily skewed —
the top cluster is 37% of 153,702 player-slots — so before any card is seen the
argmax already shares 44.6 of 60 cards with the opponent's real deck, against
19.4 for the mirror heuristic this replaces.

**Elimination carries the rest.** Any card observed that archetype *j* does not
contain is decisive evidence against *j*, so the posterior collapses within a
few turns: 57.1/60 by four cards seen, 59.6/60 by twenty.

The hypothesis set is deliberately **all** archetypes, not the ``opp_ids``
subset the retired belief heads classified over. Those 9 ids cover only 68.7%
of games by frequency, so a *perfect* classifier over them caps at ~52/60 —
the ceiling was in the label space, not in the model.

Counting is **multiset-aware**. Seeing a third copy of a card an archetype runs
two of is evidence against that archetype, and a set-based check would miss it.
Basic Energy is exempt from the 4-copy rule and real decks in this corpus run up
to 22 copies of one card, so the counts have to come from the decklist itself
rather than from an assumed maximum.
"""
```

Then in `python/ptcg_il/belief_infer.py`, line 252, change:

```python
                from ptcg_rl.belief import ArchetypePosterior
```

to:

```python
                from ptcg_il.deck_prior import ArchetypePosterior
```

And in `python/tests/test_deck_prior.py`, change line 16:

```python
from ptcg_rl.belief import CONFIDENT, ArchetypePosterior
```

to:

```python
from ptcg_il.deck_prior import CONFIDENT, ArchetypePosterior
```

Also update that test file's module docstring first line from ``` ``ptcg_rl.belief`` — the categorical posterior over 𝒟_opp archetypes.``` to ``` ``ptcg_il.deck_prior`` — the categorical posterior over archetypes.```

- [ ] **Step 3: Run the moved tests to confirm the move is clean**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_deck_prior.py -q`
Expected: PASS, 19 passed. The class body is unchanged, so any failure here is a bad move, not a logic change.

- [ ] **Step 4: Confirm nothing still imports the old path**

Run: `cd /home/charles/Documents/Pokemon && grep -rn "ptcg_rl.belief\|ptcg_rl import belief" --include="*.py" . | grep -v worktrees`
Expected: no output. If `scripts/build_submission.py:746` shows up, leave it — Task 7 owns that line.

- [ ] **Step 5: Write the failing tests for the two new helpers**

Append to `python/tests/test_deck_prior.py`:

```python
from ptcg_il.deck_prior import all_archetype_ids, frequency_prior

# `frequency` is appearances (player-slots), not distinct decklists — that is
# `n_members`.  Getting these two confused would prior the posterior on how
# many *variants* a cluster has rather than how often you face it.
FREQ_ARCHETYPES = {
    "opp_ids": [0, 1],
    "archetypes": [
        {"id": 0, "representative": [100] * 60, "frequency": 30, "n_members": 9},
        {"id": 1, "representative": [200] * 60, "frequency": 10, "n_members": 1},
        {"id": 7, "representative": [300] * 60, "frequency": 60, "n_members": 3},
    ],
}


class TestAllArchetypeIds:
    def test_returns_every_id_not_just_opp_ids(self):
        # The whole point of the change: the hypothesis set is the file, not
        # its `opp_ids` subset.
        assert all_archetype_ids(FREQ_ARCHETYPES) == [0, 1, 7]

    def test_handles_the_id_keyed_dict_container_form(self):
        as_dict = {"archetypes": {"5": {"representative": [1] * 60},
                                  "2": {"representative": [2] * 60}}}
        assert all_archetype_ids(as_dict) == [2, 5]

    def test_empty_file_yields_no_ids(self):
        assert all_archetype_ids({"archetypes": []}) == []


class TestFrequencyPrior:
    def test_reads_frequency_not_n_members(self):
        p = frequency_prior(FREQ_ARCHETYPES, [0, 1, 7])
        assert p == [30.0, 10.0, 60.0]

    def test_follows_the_requested_id_order(self):
        assert frequency_prior(FREQ_ARCHETYPES, [7, 0]) == [60.0, 30.0]

    def test_zero_frequency_is_floored_not_made_impossible(self):
        """A re-mine retains baseline clusters at frequency 0 (see CLAUDE.md).

        ArchetypePosterior maps a non-positive prior to -inf, which is a
        permanent, silent elimination: if the opponent actually plays that
        deck, no amount of evidence can ever recover it.
        """
        archetypes = {
            "archetypes": [
                {"id": 0, "representative": [100] * 60, "frequency": 1000},
                {"id": 1, "representative": [200] * 60, "frequency": 0},
            ],
        }
        p = frequency_prior(archetypes, [0, 1])
        assert p[1] > 0.0, "a retained baseline cluster must stay reachable"
        assert p[1] < p[0]

        # And it must actually be recoverable through the posterior.
        post = ArchetypePosterior(archetypes, opp_ids=[0, 1], priors=p)
        assert post.most_likely([200] * 4)[0] == 1

    def test_all_zero_frequencies_degrade_to_uniform(self):
        archetypes = {
            "archetypes": [
                {"id": 0, "representative": [100] * 60, "frequency": 0},
                {"id": 1, "representative": [200] * 60, "frequency": 0},
            ],
        }
        assert frequency_prior(archetypes, [0, 1]) == [1.0, 1.0]

    def test_missing_frequency_key_is_treated_as_zero(self):
        archetypes = {"archetypes": [
            {"id": 0, "representative": [100] * 60, "frequency": 5},
            {"id": 1, "representative": [200] * 60},
        ]}
        p = frequency_prior(archetypes, [0, 1])
        assert p[0] == 5.0
        assert 0.0 < p[1] < 5.0
```

- [ ] **Step 6: Run to verify they fail**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_deck_prior.py -q`
Expected: FAIL — `ImportError: cannot import name 'all_archetype_ids' from 'ptcg_il.deck_prior'`

- [ ] **Step 7: Implement the two helpers**

Append to `python/ptcg_il/deck_prior.py`, after `_representative_of`:

```python
def _archetype_entries(archetypes: dict[str, Any]) -> dict[int, Any]:
    """``{archetype_id: entry}``, accepting both container forms.

    ``archetypes.json``'s ``archetypes`` container has been both a list and an
    id-keyed object across revisions of the mining code, and JSON object keys
    are strings while archetype ids are ints.
    """
    container = archetypes.get("archetypes")
    if isinstance(container, dict):
        return {int(k): v for k, v in container.items()}
    return {int(item["id"]): item
            for item in (container or [])
            if isinstance(item, dict) and "id" in item}


def all_archetype_ids(archetypes: dict[str, Any]) -> list[int]:
    """Every archetype id in the file, ascending.

    This — not ``opp_ids`` — is the hypothesis set.  ``opp_ids`` is the belief
    heads' 9-way label space, which covers only 68.7% of games by frequency and
    caps template quality at ~52/60 even with a perfect classifier.
    """
    return sorted(_archetype_entries(archetypes))


def frequency_prior(
    archetypes: dict[str, Any], ids: Sequence[int]
) -> list[float]:
    """Mined ``frequency`` per id, in *ids* order, floored away from zero.

    ``frequency`` counts appearances (player-slots); ``n_members`` counts
    distinct decklists in the cluster.  The prior wants the former — how often
    you face the deck, not how many variants of it were mined.

    Two degenerate cases matter, both of which arise from real artifacts.  A
    cluster retained from a baseline lineage with no members in the current
    corpus carries ``frequency: 0`` (see CLAUDE.md on append-only ids), and
    :class:`ArchetypePosterior` turns a non-positive prior into a ``-inf``
    log-prior — a permanent, silent elimination no evidence can undo.  Such
    entries are floored to a thousandth of the smallest positive frequency:
    strongly disfavoured, still reachable.  A file where *every* frequency is
    absent or zero degrades to uniform rather than to all-impossible.
    """
    entries = _archetype_entries(archetypes)
    raw: list[float] = []
    for aid in ids:
        entry = entries.get(int(aid))
        value = entry.get("frequency", 0) if isinstance(entry, dict) else 0
        try:
            raw.append(max(0.0, float(value)))
        except (TypeError, ValueError):
            raw.append(0.0)

    positive = [f for f in raw if f > 0.0]
    if not positive:
        return [1.0] * len(raw)
    floor = min(positive) * 1e-3
    return [f if f > 0.0 else floor for f in raw]
```

- [ ] **Step 8: Run to verify they pass**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_deck_prior.py -q`
Expected: PASS, 27 passed.

- [ ] **Step 9: Mutation-check the zero-frequency floor**

Temporarily change the last line of `frequency_prior` to `return raw`. Run the tests.
Expected: `test_zero_frequency_is_floored_not_made_impossible` FAILS. Restore the line and confirm green again. If it passes with the mutation, the test is not testing the floor.

- [ ] **Step 10: Confirm the RL package still imports**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/ -q -x --co 2>&1 | tail -5`
Expected: collection succeeds with no `ImportError`. This catches any straggling `ptcg_rl.belief` import that the grep in Step 4 missed inside a function body.

- [ ] **Step 11: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 2: Observation extraction and accumulation

There are currently **three** implementations of "which opponent cards can I see": `search_infer.extract_opp_visible_cards`, `vec_env._extract_opp_visible_card_ids` (a byte-identical copy), and `belief_labels.opp_visible_counts`. Only the third is correct. This task consolidates onto the correct rules.

**Files:**
- Modify: `python/ptcg_il/deck_prior.py` (append)
- Modify: `python/tests/test_deck_prior.py` (append)

**Interfaces:**
- Consumes: Task 1's module
- Produces:
  - `observed_cards_from_state(state: dict, your_index: int) -> Counter[int]`
  - `observed_cards_from_obs(obs_dict: dict) -> Counter[int]`
  - `ObservedOpponentCards` with `.reset() -> None`, `.observe(obs_dict: dict) -> None`, `.add_cards(card_ids: Iterable[int]) -> None`, `.counts() -> Counter[int]`, `.multiset() -> list[int]`

`add_cards` is a first-class entry point, not a test affordance: `ptcg_rl` records `Decision.opp_visible_card_ids` as a flat id list (`vec_env.py:81`) and has no observation dict to hand over at determinization time. It takes the **same per-card-maximum** combining rule as `observe`, because that id list is itself one snapshot of the visible zones — summing it into a running total would double-count exactly as summing observations would.

**Background the implementer needs.** Per `AGENT_SPEC.md:57`, face-down cards (`active[0]`, `prize[i]`) arrive as `None`, so an `isinstance(card, dict)` guard is what excludes hidden information — there is no leak here, and there must not be one introduced. Per `belief_labels.py:117-134`, cards carry a `playerIndex` naming their owner, and the Stadium sits at `state["stadium"]` as a **bare dict, not a list**, owned by whoever played it.

The two divergences from `search_infer.extract_opp_visible_cards` being fixed:

1. **It never filters on `playerIndex`.** The Stadium is a shared zone; crediting one we played to the opponent is false evidence, and each unexplained copy costs the true archetype a factor of `epsilon = 1e-3` — a manufactured elimination of the right answer.
2. **It iterates the Stadium as a list.** `for card in (current.get("stadium") or [])` over a dict iterates its *keys* (strings), and `isinstance(card, dict)` is then False, so the Stadium currently contributes nothing at all. Fixing the shape without also adding the owner filter would turn a silent no-op into active false evidence — do both or neither.

It also picks up evidence the old function discarded: `energyCards` and `tools` attached to the opponent's Pokémon, which come from their deck and are real signal.

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_deck_prior.py`:

```python
from collections import Counter

from ptcg_il.deck_prior import (
    ObservedOpponentCards,
    observed_cards_from_obs,
    observed_cards_from_state,
)


def _obs(state: dict, your_index: int = 0) -> dict:
    state = dict(state)
    state["yourIndex"] = your_index
    return {"current": state, "select": {}}


class TestObservedCardsFromState:
    def test_reads_the_opponents_zones_not_ours(self):
        state = {
            "players": [
                {"active": [{"id": 11}], "bench": [], "discard": [{"id": 12}]},
                {"active": [{"id": 21}], "bench": [{"id": 22}],
                 "discard": [{"id": 23}], "prize": [{"id": 24}]},
            ],
        }
        assert observed_cards_from_state(state, your_index=0) == Counter(
            {21: 1, 22: 1, 23: 1, 24: 1})

    def test_face_down_cards_are_none_and_are_skipped(self):
        """AGENT_SPEC.md:57 — face-down active and prizes arrive as None.

        Counting them would be reading hidden information.
        """
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [None], "bench": [],
                 "discard": [], "prize": [None, {"id": 24}, None]},
            ],
        }
        assert observed_cards_from_state(state, your_index=0) == Counter({24: 1})

    def test_counts_copies_not_distinct_ids(self):
        """Multiplicity is the evidence a set-based check throws away."""
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [],
                 "discard": [{"id": 30}, {"id": 30}, {"id": 30}]},
            ],
        }
        assert observed_cards_from_state(state, your_index=0)[30] == 3

    def test_attached_energy_and_tools_are_counted(self):
        """They come out of the opponent's own deck, so they are evidence."""
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [{"id": 21, "playerIndex": 1,
                             "energyCards": [{"id": 90, "playerIndex": 1}],
                             "tools": [{"id": 91, "playerIndex": 1}],
                             "preEvolution": [{"id": 92, "playerIndex": 1}]}],
                 "bench": [], "discard": []},
            ],
        }
        counts = observed_cards_from_state(state, your_index=0)
        assert counts == Counter({21: 1, 90: 1, 91: 1, 92: 1})

    def test_energy_we_attached_to_their_pokemon_is_not_their_card(self):
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [{"id": 21, "playerIndex": 1,
                             "energyCards": [{"id": 90, "playerIndex": 0}]}],
                 "bench": [], "discard": []},
            ],
        }
        counts = observed_cards_from_state(state, your_index=0)
        assert counts[21] == 1
        assert counts[90] == 0, "our Energy on their Pokemon is ours"

    def test_stadium_is_a_bare_dict_and_counts_when_they_played_it(self):
        """`state["stadium"]` is one card, not a list — iterating it as a list
        walks its string keys and silently contributes nothing."""
        state = {
            "players": [{"active": [], "bench": [], "discard": []}] * 2,
            "stadium": {"id": 102, "playerIndex": 1},
        }
        assert observed_cards_from_state(state, your_index=0) == Counter({102: 1})

    def test_our_own_stadium_is_not_evidence_about_them(self):
        """The decisive case: a Stadium we played, mis-credited, is a card the
        true archetype cannot explain — an epsilon-weighted false elimination
        of the correct answer."""
        state = {
            "players": [{"active": [], "bench": [], "discard": []}] * 2,
            "stadium": {"id": 102, "playerIndex": 0},
        }
        assert observed_cards_from_state(state, your_index=0) == Counter()

    def test_unowned_cards_are_credited_to_the_zone_they_sit_in(self):
        """No `playerIndex` means the engine did not say; a card in their
        discard is theirs regardless."""
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [], "discard": [{"id": 55}]},
            ],
        }
        assert observed_cards_from_state(state, your_index=0) == Counter({55: 1})

    def test_seat_one_reads_seat_zero(self):
        state = {
            "players": [
                {"active": [], "bench": [], "discard": [{"id": 12}]},
                {"active": [], "bench": [], "discard": [{"id": 23}]},
            ],
        }
        assert observed_cards_from_state(state, your_index=1) == Counter({12: 1})

    def test_missing_opponent_yields_nothing(self):
        assert observed_cards_from_state({"players": []}, your_index=0) == Counter()
        assert observed_cards_from_state({}, your_index=0) == Counter()


class TestObservedCardsFromObs:
    def test_reads_your_index_off_current(self):
        state = {
            "players": [
                {"active": [], "bench": [], "discard": [{"id": 12}]},
                {"active": [], "bench": [], "discard": [{"id": 23}]},
            ],
        }
        assert observed_cards_from_obs(_obs(state, 0)) == Counter({23: 1})
        assert observed_cards_from_obs(_obs(state, 1)) == Counter({12: 1})

    def test_empty_observation_yields_nothing(self):
        assert observed_cards_from_obs({}) == Counter()


class TestObservedOpponentCards:
    def _state(self, discard_ids: list[int]) -> dict:
        return {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [],
                 "discard": [{"id": i} for i in discard_ids]},
            ],
        }

    def test_accumulates_cards_that_leave_view(self):
        """A card seen on the bench that later returns to hand is still
        evidence — the snapshot loses it, the accumulator keeps it."""
        seen = ObservedOpponentCards()
        seen.observe(_obs(self._state([40])))
        seen.observe(_obs(self._state([])))
        assert seen.counts() == Counter({40: 1})

    def test_a_card_moving_zone_is_not_double_counted(self):
        """The load-bearing guard.  Summing snapshots manufactures a second
        copy, and a phantom copy the true archetype cannot explain costs it a
        factor of epsilon — eliminating the right answer with invented
        evidence.  Per-card maximum, never sum.
        """
        active = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [{"id": 41}], "bench": [], "discard": []},
            ],
        }
        discarded = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [], "discard": [{"id": 41}]},
            ],
        }
        seen = ObservedOpponentCards()
        seen.observe(_obs(active))
        seen.observe(_obs(discarded))
        assert seen.counts()[41] == 1, "same card, two zones, one copy"

    def test_a_genuine_second_copy_is_counted(self):
        """The max must still rise when the opponent really shows two."""
        seen = ObservedOpponentCards()
        seen.observe(_obs(self._state([42])))
        seen.observe(_obs(self._state([42, 42])))
        assert seen.counts()[42] == 2

    def test_reset_clears_between_games(self):
        seen = ObservedOpponentCards()
        seen.observe(_obs(self._state([43])))
        seen.reset()
        assert seen.counts() == Counter()

    def test_multiset_expands_every_copy(self):
        seen = ObservedOpponentCards()
        seen.observe(_obs(self._state([44, 44, 45])))
        assert sorted(seen.multiset()) == [44, 44, 45]

    def test_counts_returns_a_copy_not_the_internal_state(self):
        seen = ObservedOpponentCards()
        seen.observe(_obs(self._state([46])))
        seen.counts()[46] = 99
        assert seen.counts()[46] == 1


class TestAddCards:
    """`ptcg_rl` records visible cards as a flat id list, not an observation."""

    def test_adds_a_raw_id_list(self):
        seen = ObservedOpponentCards()
        seen.add_cards([50, 50, 51])
        assert seen.counts() == Counter({50: 2, 51: 1})

    def test_combines_by_maximum_like_observe(self):
        """Same rule as observe: an id list is one snapshot of the visible
        zones, so re-adding it must not double the counts."""
        seen = ObservedOpponentCards()
        seen.add_cards([52, 52])
        seen.add_cards([52, 52])
        assert seen.counts()[52] == 2

    def test_a_larger_later_snapshot_raises_the_count(self):
        seen = ObservedOpponentCards()
        seen.add_cards([53])
        seen.add_cards([53, 53, 53])
        assert seen.counts()[53] == 3

    def test_interoperates_with_observe(self):
        seen = ObservedOpponentCards()
        seen.observe(_obs({
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [], "discard": [{"id": 54}]},
            ],
        }))
        seen.add_cards([55])
        assert seen.counts() == Counter({54: 1, 55: 1})

    def test_reset_clears_added_cards(self):
        seen = ObservedOpponentCards()
        seen.add_cards([56])
        seen.reset()
        assert seen.counts() == Counter()

    def test_empty_list_is_a_no_op(self):
        seen = ObservedOpponentCards()
        seen.add_cards([57])
        seen.add_cards([])
        assert seen.counts() == Counter({57: 1})
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_deck_prior.py -q`
Expected: FAIL — `ImportError: cannot import name 'ObservedOpponentCards'`

- [ ] **Step 3: Implement**

Append to `python/ptcg_il/deck_prior.py`:

```python
# ── Observing the opponent ──────────────────────────────────────────────────


def observed_cards_from_state(state: dict, your_index: int) -> Counter[int]:
    """Engine card ids of opponent cards visible in *state*, with multiplicity.

    Mirrors :func:`ptcg_il.belief_labels.opp_visible_counts`, which is the
    validated definition of this concept in this codebase — it agrees with the
    observation's own ``deckCount + handCount + face-down prizes`` on 98.4% of
    decision points.  Three of its rules are load-bearing and easy to lose:

    **Ownership.** Cards carry a ``playerIndex``.  Energy *we* attached to
    *their* Pokémon, and a Stadium *we* played, are our cards.  Crediting them
    to the opponent is not a harmless over-count: an observed card the true
    archetype cannot explain costs it a factor of ``epsilon``, so false
    evidence eliminates the correct hypothesis.

    **The Stadium is a bare dict**, hanging off the state rather than off a
    player.  Iterating it as if it were a list walks its string keys and
    contributes nothing.

    **Face-down cards are ``None``** (``AGENT_SPEC.md``: ``active[0]``,
    ``prize[i]``).  The ``isinstance`` guard is what keeps hidden information
    out; do not replace it with a ``.get("id")`` on a defaulted dict.

    Attached ``energyCards`` and ``tools`` *are* counted when the opponent owns
    them — they came out of their deck and are as much evidence as anything in
    their discard.
    """
    players = state.get("players") or []
    opp_index = 1 - int(your_index)
    if not (0 <= opp_index < len(players)) or players[opp_index] is None:
        return Counter()
    opp = players[opp_index]

    counts: Counter[int] = Counter()

    def add_card(card: Any) -> None:
        if not isinstance(card, dict):
            return  # face-down (None), or a malformed entry
        owner = card.get("playerIndex")
        if owner is not None and int(owner) != opp_index:
            return
        cid = card.get("id")
        if cid is None:
            return
        counts[int(cid)] += 1

    def add_pokemon(poke: Any) -> None:
        if not isinstance(poke, dict):
            return
        add_card(poke)
        for key in ("preEvolution", "energyCards", "tools"):
            for sub in poke.get(key) or []:
                add_card(sub)

    for poke in opp.get("active") or []:
        add_pokemon(poke)
    for poke in opp.get("bench") or []:
        add_pokemon(poke)
    for card in opp.get("discard") or []:
        add_card(card)
    for card in opp.get("prize") or []:
        add_card(card)

    stadium = state.get("stadium")
    if isinstance(stadium, dict):
        stadium = [stadium]
    for card in stadium or []:
        add_card(card)

    return counts


def observed_cards_from_obs(obs_dict: dict) -> Counter[int]:
    """:func:`observed_cards_from_state` against an observation's ``current``."""
    state = obs_dict.get("current") or {}
    return observed_cards_from_state(state, int(state.get("yourIndex", 0) or 0))


class ObservedOpponentCards:
    """Running record of what the opponent has shown, across a whole game.

    Combines snapshots by **per-card-id maximum**, not by sum.  A card that
    moves from bench to discard appears in two consecutive observations; summing
    would invent a second copy, and an invented copy the true archetype cannot
    explain is an ``epsilon``-weighted elimination of the correct answer.  The
    maximum keeps a card that has left view while never over-counting one that
    merely moved.

    Accumulating is worth roughly +8 cards of template accuracy over a
    per-turn snapshot (k=2 → 47.2, k=8 → 55.1 in the measured variant regime).
    """

    def __init__(self) -> None:
        self._counts: Counter[int] = Counter()

    def reset(self) -> None:
        """Forget everything.  Call at the start of each new game."""
        self._counts.clear()

    def observe(self, obs_dict: dict) -> None:
        """Fold one observation's visible cards into the record."""
        self._merge(observed_cards_from_obs(obs_dict))

    def add_cards(self, card_ids: Iterable[int]) -> None:
        """Fold a raw multiset of engine card ids into the record.

        ``ptcg_rl`` records the opponent's visible cards as a flat id list
        (``Decision.opp_visible_card_ids``) and has no observation dict to hand
        over at determinization time.  The combining rule is the same maximum
        ``observe`` uses: that list is one snapshot of the visible zones, so
        adding it twice must not double the counts.
        """
        self._merge(Counter(int(c) for c in card_ids))

    def _merge(self, snapshot: Counter[int]) -> None:
        for cid, n in snapshot.items():
            if n > self._counts[cid]:
                self._counts[cid] = n

    def counts(self) -> Counter[int]:
        """A copy of the accumulated per-card counts."""
        return Counter(self._counts)

    def multiset(self) -> list[int]:
        """Every observed copy, expanded — what the posterior consumes."""
        return list(self._counts.elements())
```

`Counter` is already imported at the top of the moved file (`from collections import Counter`) — verify rather than duplicate. `Iterable` is already imported too (`from typing import Any, Iterable, Sequence`).

- [ ] **Step 4: Run to verify they pass**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_deck_prior.py -q`
Expected: PASS.

- [ ] **Step 5: Mutation-check the two guards that matter**

Mutation A — turn the max into a sum. In `observe`, replace the loop body with `self._counts[cid] += n`.
Expected: `test_a_card_moving_zone_is_not_double_counted` FAILS. Restore.

Mutation B — drop the owner filter. In `add_card`, delete the two `owner` lines.
Expected: `test_our_own_stadium_is_not_evidence_about_them` and `test_energy_we_attached_to_their_pokemon_is_not_their_card` FAIL. Restore.

Confirm green after restoring both.

- [ ] **Step 6: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 3: `OpponentDeckPredictor`

The caller-facing object. Everything downstream talks to this and nothing else.

**Files:**
- Modify: `python/ptcg_il/deck_prior.py` (append)
- Modify: `python/tests/test_deck_prior.py` (append)

**Interfaces:**
- Consumes: Tasks 1–2
- Produces: `OpponentDeckPredictor(archetypes: dict, *, ids: Sequence[int] | None = None, use_frequency_prior: bool = True, epsilon: float = DEFAULT_EPSILON)` with `.reset() -> None`, `.observe(obs_dict: dict) -> None`, `.observe_cards(card_ids: Iterable[int]) -> None`, `.posterior() -> dict[int, float]`, `.template() -> list[int]`, `.sample_templates(k: int, rng) -> list[list[int]]`, and attribute `.ids: list[int]`

`observe_cards` forwards to `ObservedOpponentCards.add_cards` and is what `ptcg_rl` calls in Task 4. Nothing outside the class touches `_seen`.

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_deck_prior.py`:

```python
import numpy as np

from ptcg_il.deck_prior import OpponentDeckPredictor

# Three archetypes over a tiny card space, with a lopsided prior so the
# frequency/uniform distinction is observable.
PRED_ARCHETYPES = {
    "opp_ids": [0],
    "archetypes": [
        {"id": 0, "representative": [100] * 60, "frequency": 900},
        {"id": 1, "representative": [200] * 60, "frequency": 50},
        {"id": 2, "representative": [300] * 60, "frequency": 50},
    ],
}


class TestPredictorDefaults:
    def test_hypothesis_set_is_every_archetype_not_opp_ids(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        assert pred.ids == [0, 1, 2]

    def test_frequency_prior_is_the_default(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        p = pred.posterior()
        assert p[0] > 0.85, f"frequency prior should dominate before evidence: {p}"

    def test_uniform_prior_can_be_asked_for(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES, use_frequency_prior=False)
        p = pred.posterior()
        assert all(abs(v - 1 / 3) < 1e-9 for v in p.values())


class TestTemplateNeverAbstains:
    def test_commits_with_zero_observations(self):
        """The never-abstain guarantee.  The alternative the old code fell back
        to — an empty list, i.e. the Rust mirror heuristic — is worth 19.4/60
        against this argmax's 44.6/60, so there is no k at which abstaining
        pays."""
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        deck = pred.template()
        assert len(deck) == 60
        assert set(deck) == {100}, "the frequency argmax before any evidence"

    def test_commits_when_the_posterior_is_flat(self):
        flat = {"archetypes": [
            {"id": 0, "representative": [100] * 60, "frequency": 10},
            {"id": 1, "representative": [200] * 60, "frequency": 10},
        ]}
        deck = OpponentDeckPredictor(flat).template()
        assert len(deck) == 60

    def test_never_returns_an_empty_list(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        for ids in ([], [999], [100] * 4, [200] * 4, [100, 200, 300]):
            pred.reset()
            pred.observe_cards(ids)
            assert pred.template(), f"abstained on {ids}"

    def test_evidence_overrides_the_prior(self):
        """Elimination is the mechanism: four copies of card 200 rule out the
        900-frequency favourite outright."""
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        pred.observe_cards([200] * 4)
        assert set(pred.template()) == {200}

    def test_an_unknown_card_does_not_abstain(self):
        """A card no archetype contains leaves every hypothesis equally
        penalised; the predictor must still commit."""
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        pred.observe_cards([999])
        assert len(pred.template()) == 60


class TestPredictorObservation:
    def _obs_with(self, card_id: int, n: int = 1) -> dict:
        return {"current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [],
                 "discard": [{"id": card_id} for _ in range(n)]},
            ],
        }}

    def test_observe_then_template_tracks_the_evidence(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        assert set(pred.template()) == {100}
        pred.observe(self._obs_with(300, 4))
        assert set(pred.template()) == {300}

    def test_reset_returns_to_the_prior(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        pred.observe(self._obs_with(300, 4))
        pred.reset()
        assert set(pred.template()) == {100}

    def test_template_is_a_pure_function_of_the_observation_sequence(self):
        a, b = OpponentDeckPredictor(PRED_ARCHETYPES), OpponentDeckPredictor(PRED_ARCHETYPES)
        for obs in (self._obs_with(200, 1), self._obs_with(200, 2)):
            a.observe(obs)
            b.observe(obs)
        assert a.template() == b.template()

    def test_a_returned_template_cannot_mutate_the_predictor(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        pred.template().append(999)
        assert len(pred.template()) == 60


class TestSampleTemplates:
    def test_returns_k_decks_of_sixty(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        decks = pred.sample_templates(8, np.random.default_rng(0))
        assert len(decks) == 8
        assert all(len(d) == 60 for d in decks)

    def test_is_reproducible_under_a_seeded_rng(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        a = pred.sample_templates(8, np.random.default_rng(7))
        b = pred.sample_templates(8, np.random.default_rng(7))
        assert a == b

    def test_a_diffuse_posterior_yields_varied_worlds(self):
        """The reason this call site does not use argmax: K identical
        determinizations collapse `mcts_k_determinizations` to 1 while leaving
        the config knob looking effective."""
        flat = {"archetypes": [
            {"id": 0, "representative": [100] * 60, "frequency": 10},
            {"id": 1, "representative": [200] * 60, "frequency": 10},
            {"id": 2, "representative": [300] * 60, "frequency": 10},
        ]}
        decks = OpponentDeckPredictor(flat).sample_templates(
            50, np.random.default_rng(0))
        assert len({tuple(d) for d in decks}) > 1

    def test_a_collapsed_posterior_yields_the_same_world(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        pred.observe_cards([200] * 4)
        decks = pred.sample_templates(20, np.random.default_rng(0))
        assert {tuple(d) for d in decks} == {tuple([200] * 60)}

    def test_zero_k_is_no_worlds(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        assert pred.sample_templates(0, np.random.default_rng(0)) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_deck_prior.py -q`
Expected: FAIL — `ImportError: cannot import name 'OpponentDeckPredictor'`

- [ ] **Step 3: Implement**

Append to `python/ptcg_il/deck_prior.py`:

```python
# ── The caller-facing predictor ─────────────────────────────────────────────


class OpponentDeckPredictor:
    """A 60-card opponent decklist for the determinizer, from the prior alone.

    Stateful across a game: :meth:`observe` folds each observation into the
    running record, :meth:`template` reads out the current best guess, and
    :meth:`reset` clears between games.

    Two read-out modes, because the two consumers want different things.
    :meth:`template` returns the argmax and **always commits** — the submission
    and ``live_eval`` want one world, and the thing abstention falls back to
    (the Rust mirror heuristic) is worth 19.4/60 against an unconfident
    argmax's 44.6/60.  :meth:`sample_templates` draws *k* worlds from the
    posterior instead, because ``ptcg_rl``'s ``mcts_k_determinizations`` exists
    to cover uncertainty across *different* worlds; handing it k copies of the
    argmax would silently collapse the knob to 1.

    Parameters
    ----------
    archetypes : dict
        Parsed ``archetypes.json``.
    ids : sequence of int, optional
        Hypothesis set.  Defaults to **every** archetype in the file — not
        ``opp_ids``, which covers only 68.7% of games by frequency.
    use_frequency_prior : bool
        Prior from mined ``frequency`` (default) or uniform.  The prior is what
        carries the early turns: 44.6/60 against 39.0/60 at one card seen, and
        top-1 of 0.56 against 0.30.
    epsilon : float
        Mass kept per unexplained observed copy.  The default is measured
        optimal across variant regimes; see the module docstring.
    """

    def __init__(
        self,
        archetypes: dict[str, Any],
        *,
        ids: Sequence[int] | None = None,
        use_frequency_prior: bool = True,
        epsilon: float = DEFAULT_EPSILON,
    ):
        self.ids: list[int] = (
            [int(i) for i in ids] if ids is not None
            else all_archetype_ids(archetypes)
        )
        priors = frequency_prior(archetypes, self.ids) if use_frequency_prior else None
        self._post = ArchetypePosterior(
            archetypes, opp_ids=self.ids, priors=priors, epsilon=epsilon
        )
        self._reps: dict[int, list[int]] = {
            aid: _representative_of(archetypes, aid) for aid in self.ids
        }
        self._seen = ObservedOpponentCards()

    # ── Game lifecycle ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget this game's observations.  Call when a new game starts."""
        self._seen.reset()

    def observe(self, obs_dict: dict) -> None:
        """Fold one observation into the running record."""
        self._seen.observe(obs_dict)

    def observe_cards(self, card_ids: Iterable[int]) -> None:
        """Fold a raw multiset of engine card ids into the running record.

        For callers holding an id list rather than an observation — ``ptcg_rl``
        records ``Decision.opp_visible_card_ids`` that way.
        """
        self._seen.add_cards(card_ids)

    # ── Read-out ────────────────────────────────────────────────────────

    def posterior(self) -> dict[int, float]:
        """``{archetype_id: probability}`` given everything observed so far.

        Exposed for logging and tests.  Do not branch on it in production code:
        the never-abstain rule means there is no confidence threshold to check.
        """
        return self._post.posterior(self._seen.multiset())

    def template(self) -> list[int]:
        """The most likely archetype's 60-card representative.  Never empty."""
        post = self.posterior()
        best = max(post, key=lambda aid: post[aid])
        return list(self._reps[best])

    def sample_templates(self, k: int, rng: Any) -> list[list[int]]:
        """*k* decklists drawn from the posterior, for *k* determinized worlds.

        *rng* is a ``numpy.random.Generator``.  numpy is imported here rather
        than at module scope so the argmax path stays importable without it.
        """
        if k <= 0:
            return []
        import numpy as np

        post = self.posterior()
        ids = list(post)
        weights = np.asarray([post[aid] for aid in ids], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            weights = np.full(len(ids), 1.0 / len(ids))
        else:
            weights = weights / total
        picks = rng.choice(len(ids), size=k, replace=True, p=weights)
        return [list(self._reps[ids[int(i)]]) for i in picks]
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_deck_prior.py -q`
Expected: PASS.

- [ ] **Step 5: Verify against the real artifact, not just fixtures**

Create `python/tests/test_deck_prior_real.py`:

```python
"""The predictor against the real `data/archetypes.json`.

Fixtures cannot catch a shape assumption that only the shipped artifact
violates, and this artifact is now the entire opponent model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ptcg_il.deck_prior import OpponentDeckPredictor, all_archetype_ids

ARCHETYPES_PATH = Path(__file__).resolve().parents[1] / "data" / "archetypes.json"


@pytest.fixture(scope="module")
def archetypes() -> dict:
    if not ARCHETYPES_PATH.exists():
        pytest.skip(f"no mined artifact at {ARCHETYPES_PATH}")
    with open(ARCHETYPES_PATH) as f:
        return json.load(f)


def test_every_archetype_has_a_full_representative(archetypes):
    ids = all_archetype_ids(archetypes)
    assert ids, "artifact has no archetypes"
    entries = {int(a["id"]): a for a in archetypes["archetypes"]}
    short = [i for i in ids if len(entries[i].get("representative") or []) != 60]
    assert not short, f"{len(short)} archetypes lack a 60-card representative: {short[:5]}"


def test_predictor_commits_to_sixty_cards_before_any_evidence(archetypes):
    deck = OpponentDeckPredictor(archetypes).template()
    assert len(deck) == 60


def test_hypothesis_set_is_much_wider_than_opp_ids(archetypes):
    pred = OpponentDeckPredictor(archetypes)
    n_opp = len(archetypes.get("opp_ids") or [])
    assert n_opp > 0, "artifact has no opp_ids to compare against"
    assert len(pred.ids) > n_opp * 5, (
        f"expected the full archetype set, got {len(pred.ids)} against "
        f"{n_opp} opp_ids")


def test_evidence_from_a_real_decklist_recovers_that_decklist(archetypes):
    """Reveal 20 cards of a real archetype; the argmax should be that deck.

    Fails on zero examined: the loop asserts it saw every id it planned to.
    """
    pred = OpponentDeckPredictor(archetypes)
    entries = {int(a["id"]): a for a in archetypes["archetypes"]}
    rng = np.random.default_rng(0)
    # The ten most frequent archetypes — the ones actually worth recovering.
    top = sorted(entries, key=lambda i: -entries[i].get("frequency", 0))[:10]

    examined = 0
    hits = 0
    for aid in top:
        deck = [int(c) for c in entries[aid]["representative"]]
        revealed = [int(c) for c in rng.choice(deck, size=20, replace=False)]
        pred.reset()
        pred.observe_cards(revealed)
        if sorted(pred.template()) == sorted(deck):
            hits += 1
        examined += 1

    assert examined == 10, f"examined {examined}, expected 10"
    assert hits >= 8, f"recovered only {hits}/10 real decklists from 20 cards"
```

- [ ] **Step 6: Run the real-artifact tests**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_deck_prior_real.py -v`
Expected: PASS (or `skip` if `python/data/archetypes.json` is absent — check that it is present before accepting a skip).

- [ ] **Step 7: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 4: Rewire `ptcg_rl.train` determinization

This call site is currently **dead**: `OpponentDeckOracle(policy=None, ...)` raises `AttributeError` inside `__init__`, absorbed by the function's own bare `except Exception: return []`. RL's K determinizations have all been running on the Rust mirror heuristic.

**Files:**
- Modify: `python/ptcg_rl/train.py:697-711` (call site), `779-810` (`_sample_opp_deck`)
- Create: `python/tests/test_rl_opp_deck.py`

**Interfaces:**
- Consumes: `OpponentDeckPredictor` from Task 3
- Produces: `_opp_deck_templates(args, obs_dict: dict, observed_card_ids: list[int], k: int, seed: int) -> list[list[int]]`, replacing `_sample_opp_deck`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_rl_opp_deck.py`:

```python
"""`ptcg_rl.train`'s determinization templates.

The function this replaces constructed `OpponentDeckOracle(policy=None)`,
which raises inside `__init__`, and swallowed it in a bare `except Exception:
return []`.  Every one of K determinizations therefore ran on the Rust mirror
heuristic, silently, for the lifetime of the code.  These tests exist so that
cannot recur: a failure must raise, and the templates must actually differ.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ptcg_rl.train import _opp_deck_templates

ARCHETYPES = {
    "opp_ids": [0],
    "fixed_deck": [100] * 60,
    "archetypes": [
        {"id": 0, "representative": [100] * 60, "frequency": 40},
        {"id": 1, "representative": [200] * 60, "frequency": 30},
        {"id": 2, "representative": [300] * 60, "frequency": 30},
    ],
}


@pytest.fixture
def args(tmp_path):
    with open(tmp_path / "archetypes.json", "w") as f:
        json.dump(ARCHETYPES, f)
    return SimpleNamespace(data_dir=str(tmp_path))


def test_returns_k_full_decklists(args):
    decks = _opp_deck_templates(args, {}, [], k=8, seed=0)
    assert len(decks) == 8
    assert all(len(d) == 60 for d in decks)


def test_no_template_is_empty(args):
    """An empty list is the Rust mirror fallback — the thing being removed."""
    assert all(_opp_deck_templates(args, {}, [], k=8, seed=0))


def test_templates_vary_when_the_posterior_is_diffuse(args):
    """K identical worlds would collapse `mcts_k_determinizations` to 1 while
    leaving the config knob looking effective."""
    decks = _opp_deck_templates(args, {}, [], k=32, seed=0)
    assert len({tuple(d) for d in decks}) > 1


def test_observed_cards_steer_the_templates(args):
    decks = _opp_deck_templates(args, {}, [200] * 4, k=8, seed=0)
    assert {tuple(d) for d in decks} == {tuple([200] * 60)}


def test_the_same_seed_gives_the_same_worlds(args):
    a = _opp_deck_templates(args, {}, [], k=8, seed=3)
    b = _opp_deck_templates(args, {}, [], k=8, seed=3)
    assert a == b


def test_different_seeds_give_different_worlds(args):
    a = _opp_deck_templates(args, {}, [], k=16, seed=1)
    b = _opp_deck_templates(args, {}, [], k=16, seed=2)
    assert a != b


def test_a_missing_artifact_raises_rather_than_returning_mirrors(tmp_path):
    """The regression guard.  The predecessor's bare `except Exception` is
    exactly what hid a permanently-broken path behind plausible behaviour.
    """
    args = SimpleNamespace(data_dir=str(tmp_path / "nonexistent"))
    with pytest.raises(Exception):
        _opp_deck_templates(args, {}, [], k=8, seed=0)


def test_the_predictor_is_built_once_per_data_dir(args, monkeypatch):
    """It used to reload vocab.json and archetypes.json on every one of
    K x batch calls."""
    import ptcg_rl.train as train_mod

    train_mod._deck_predictor.cache_clear()   # the lru_cache is on this one
    calls = []
    real_open = open

    def counting_open(path, *a, **k):
        if "archetypes.json" in str(path):
            calls.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", counting_open)
    for seed in range(5):
        _opp_deck_templates(args, {}, [], k=2, seed=seed)
    assert len(calls) == 1, f"artifact re-read {len(calls)} times"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_rl_opp_deck.py -q`
Expected: FAIL — `ImportError: cannot import name '_opp_deck_templates' from 'ptcg_rl.train'`

- [ ] **Step 3: Replace `_sample_opp_deck`**

In `python/ptcg_rl/train.py`, delete the whole `_sample_opp_deck` function (lines 779–810) and put this in its place:

```python
@functools.lru_cache(maxsize=4)
def _deck_predictor(data_dir: str):
    """The archetype-prior predictor for *data_dir*, built once.

    Cached because the predecessor reloaded ``vocab.json`` and
    ``archetypes.json`` on every one of K x batch calls.  Keyed on the path
    string so ``lru_cache`` can hash it.
    """
    from ptcg_il.deck_prior import OpponentDeckPredictor

    with open(Path(data_dir) / "archetypes.json") as f:
        archetypes = json.load(f)
    return OpponentDeckPredictor(archetypes)


def _opp_deck_templates(
    args, obs_dict: dict, observed_card_ids: list[int], k: int, seed: int
) -> list[list[int]]:
    """*k* opponent decklists for *k* determinized worlds.

    Draws from the archetype posterior rather than taking its argmax: K worlds
    exist to cover uncertainty, and k copies of the same deck would collapse
    ``mcts_k_determinizations`` to 1 while leaving the knob looking effective.

    **No exception handling.**  The version this replaces wrapped everything in
    ``except Exception: return []`` and therefore hid an ``AttributeError`` in
    its own constructor for the lifetime of the code — every determinization
    silently ran on the Rust mirror heuristic instead.  A broken artifact must
    stop the run.
    """
    import numpy as np

    predictor = _deck_predictor(str(Path(args.data_dir)))
    predictor.reset()
    if obs_dict:
        predictor.observe(obs_dict)
    if observed_card_ids:
        predictor.observe_cards(observed_card_ids)
    return predictor.sample_templates(k, np.random.default_rng(seed))
```

Add `import functools` to the module's imports if it is not already there. Verify `Path` and `json` are already imported at module scope (they are — `_fixed_deck` and `_load_archetypes` use both).

`observe_cards` rather than `observe` for the id list: `observed_card_ids` arrives pre-extracted from `Decision.opp_visible_card_ids` (`vec_env.py:81`), so it is a flat list of engine ids, not an observation. Both go through the same per-card-maximum merge, so passing both — as this function does — cannot double-count a card that appears in each.

- [ ] **Step 4: Update the call site**

In `python/ptcg_rl/train.py`, replace lines 697–711 (the `for k in range(cfg.mcts_k_determinizations):` block) with:

```python
            # ── K weighted determinizations per state ──────────────────
            templates = _opp_deck_templates(
                args, obs_dict, opp_visible,
                k=cfg.mcts_k_determinizations, seed=cfg.seed + i * 1000,
            )
            for k, opp_template in enumerate(templates):
                tree_id = forest.add_root(
                    obs_dict,
                    fixed_deck,
                    opp_deck_template=opp_template,
                    iterations=cfg.mcts_iterations,
                    c_puct=cfg.mcts_c_puct,
                    seed=cfg.seed + i * 1000 + k,
                )
                if tree_id >= 0:
                    tree_map[tree_id] = (i, k)
```

Also update `_run_mcts_distillation`'s docstring: "sampled from the belief posterior" → "sampled from the archetype prior posterior".

- [ ] **Step 5: Run to verify the tests pass**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_rl_opp_deck.py -v`
Expected: PASS, 8 passed.

- [ ] **Step 6: Mutation-check the no-swallow guard**

Wrap the body of `_opp_deck_templates` in `try: ... except Exception: return []`.
Expected: `test_a_missing_artifact_raises_rather_than_returning_mirrors` and `test_no_template_is_empty` FAIL. Restore, confirm green.

- [ ] **Step 7: Run the RL suite for regressions**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/ -q -k "rl or mcts"`
Expected: PASS. `test_rl_mcts_wiring.py` exercises this path; if it stubs `_sample_opp_deck` by name, repoint the stub to `_opp_deck_templates` and adjust for the list return.

- [ ] **Step 8: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 5: Rewire `live_eval` and the `cli` opponent

**Files:**
- Modify: `python/ptcg_il/live_eval.py:323-357` (`SearchPlannerAgent`)
- Modify: `python/ptcg_il/cli.py:1108-1128` (opponent construction)
- Create: `python/tests/test_live_eval_planner.py`

**Interfaces:**
- Consumes: `OpponentDeckPredictor` from Task 3
- Produces: `SearchPlannerAgent(predictor: OpponentDeckPredictor | None = None, iterations: int = 200, seed: int = 42)`; the `opponents` dict key `"search_planner_prior"` replacing `"search_planner_belief"`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_live_eval_planner.py`:

```python
"""`SearchPlannerAgent`'s opponent model, now prior-driven.

The agent must reset its predictor between games: `live_eval` reuses one agent
object across a whole match series, and carrying game 1's observations into
game 2 is evidence about a deck that is no longer in play.
"""
from __future__ import annotations

import pytest

from ptcg_il.deck_prior import OpponentDeckPredictor
from ptcg_il.live_eval import SearchPlannerAgent

ARCHETYPES = {
    "opp_ids": [0],
    "archetypes": [
        {"id": 0, "representative": [100] * 60, "frequency": 90},
        {"id": 1, "representative": [200] * 60, "frequency": 10},
    ],
}


def _obs(discard_ids: list[int]) -> dict:
    return {
        "select": {"selectType": 0},
        "current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [],
                 "discard": [{"id": i} for i in discard_ids]},
            ],
        },
    }


@pytest.fixture
def captured(monkeypatch):
    """Capture the `opp_deck` handed to the Rust planner."""
    seen: list = []

    def fake_rust(obs_dict, opp_deck=None, iterations=200, seed=42):
        seen.append(opp_deck)
        return [0]

    monkeypatch.setattr("ptcg_il.live_eval._call_rust_search_planner", fake_rust)
    return seen


def test_passes_a_real_decklist_not_none(captured):
    agent = SearchPlannerAgent(predictor=OpponentDeckPredictor(ARCHETYPES))
    agent(_obs([]))
    assert captured[0] is not None, "None is the mirror fallback being removed"
    assert len(captured[0]) == 60


def test_observations_steer_the_template(captured):
    agent = SearchPlannerAgent(predictor=OpponentDeckPredictor(ARCHETYPES))
    agent(_obs([]))
    assert set(captured[0]) == {100}
    agent(_obs([200] * 4))
    assert set(captured[1]) == {200}


def test_a_new_game_resets_the_predictor(captured):
    """`select is None` is the deck-selection step: a new game.  Without a
    reset, the previous game's opponent is still being inferred."""
    agent = SearchPlannerAgent(predictor=OpponentDeckPredictor(ARCHETYPES))
    agent(_obs([200] * 4))
    assert set(captured[0]) == {200}

    agent({"select": None})          # new game
    agent(_obs([]))
    assert set(captured[-1]) == {100}, "stale evidence carried across games"


def test_no_predictor_still_plays(captured):
    """The plain mirror-determinized planner stays available as the baseline
    the prior-driven one is measured against."""
    agent = SearchPlannerAgent(predictor=None)
    assert agent(_obs([])) == [0]
    assert captured[0] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_live_eval_planner.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'predictor'`

- [ ] **Step 3: Rewrite `SearchPlannerAgent`**

In `python/ptcg_il/live_eval.py`, replace lines 323–357 entirely with:

```python
class SearchPlannerAgent:
    """:func:`search_planner_agent` with a prior-driven opponent model.

    The plain function leaves the Rust determinizer to assume the opponent
    mirrors our own deck, which is worth 19.4 of their 60 cards.  Given an
    :class:`~ptcg_il.deck_prior.OpponentDeckPredictor`, this variant infers
    their archetype from the mined frequency prior plus whatever they have
    shown — 44.6/60 before a single card is seen, 59.1/60 by eight — so the
    worlds MCTS searches are drawn from a deck they might actually be playing.

    Both forms are kept: ``predictor=None`` is the mirror baseline the
    prior-driven one is measured against.

    A class rather than a closure because live eval hands agents to a process
    pool: ``make_agent_from_policy``'s inner function could not be pickled, and
    a bound ``__call__`` can.
    """

    def __init__(self, predictor: Any = None, iterations: int = 200, seed: int = 42):
        self.predictor = predictor
        self.iterations = iterations
        self.seed = seed

    def __call__(self, obs_dict: dict) -> list[int]:
        select = obs_dict.get("select")
        if select is None:
            # Deck selection — a new game.  Stale evidence would otherwise be
            # inference about an opponent who is no longer at the table.
            if self.predictor is not None:
                self.predictor.reset()
            return _search_planner_deck()

        opp_deck = None
        if self.predictor is not None:
            self.predictor.observe(obs_dict)
            opp_deck = self.predictor.template()

        result = _call_rust_search_planner(
            obs_dict, opp_deck=opp_deck,
            iterations=self.iterations, seed=self.seed,
        )
        if result is not None:
            return result
        return _search_planner_fallback_python(obs_dict)
```

Note the removed `or None`: `template()` never returns empty, so coalescing would only mask a bug.

- [ ] **Step 4: Run to verify it passes**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_live_eval_planner.py -v`
Expected: PASS, 4 passed.

- [ ] **Step 5: Mutation-check the reset**

Delete the `self.predictor.reset()` line.
Expected: `test_a_new_game_resets_the_predictor` FAILS. Restore.

- [ ] **Step 6: Rewire the `cli` opponent**

In `python/ptcg_il/cli.py`, replace lines 1108–1128 with:

```python
    # A second planner whose determinization uses the archetype prior instead
    # of the mirror assumption.  Kept alongside the plain planner rather than
    # replacing it, so the two numbers measure what the prior is actually
    # worth.  No NN in this path any more, so it needs no policy, no CPU copy
    # for fork-safety, and no --belief gate.
    try:
        from ptcg_il.deck_prior import OpponentDeckPredictor
        from ptcg_il.live_eval import SearchPlannerAgent

        opponents["search_planner_prior"] = SearchPlannerAgent(
            predictor=OpponentDeckPredictor(artifacts["archetypes"]),
        )
    except Exception as e:
        logger.warning("Could not build prior-backed search planner: %s", e)
```

- [ ] **Step 7: Confirm the old key is gone everywhere**

Run: `cd /home/charles/Documents/Pokemon && grep -rn "search_planner_belief" --include="*.py" --include="*.sh" --include="*.md" . | grep -v worktrees | grep -v docs/superpowers`
Expected: no output. If a shell script or spec names the old key, update it — it appears in eval output and any downstream comparison.

- [ ] **Step 8: Run the eval-related suite**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/ -q -k "live_eval or cli or planner"`
Expected: PASS.

- [ ] **Step 9: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 6: Prune `search_infer.py`

**Files:**
- Modify: `python/ptcg_il/search_infer.py` — delete `predict_opponent_deck` (152–232), `extract_opp_visible_cards` (235–272), `_representatives` (555–575), `_deck_from_distribution` (577–610)
- Modify: `python/tests/test_search_infer.py` — delete lines 183–233 (three tests of the deleted helpers)

**Interfaces:**
- Consumes: nothing new
- Produces: `search_infer` with no belief dependency; `mcts_search` unchanged

- [ ] **Step 1: Find every remaining reference**

Run:
```bash
cd /home/charles/Documents/Pokemon
grep -rn "predict_opponent_deck\|extract_opp_visible_cards\|_deck_from_distribution\|_representatives" \
  --include="*.py" . | grep -v worktrees
```
Record the list. Everything outside `search_infer.py`, `tests/test_search_infer.py`, and `scripts/build_submission.py` must already have been rewired by Tasks 4–5; if anything else appears, stop and report it rather than deleting.

**Do not delete `_dict_to_batch` or `_no_grad`.** They look like belief helpers but `mcts_search`'s own leaf evaluation uses them at lines 441/448 and 482/485. Verified before writing this plan.

- [ ] **Step 2: Delete the functions**

Remove from `python/ptcg_il/search_infer.py`:
- the `# ── Belief-based opponent deck prediction ───` section header and `predict_opponent_deck`
- `extract_opp_visible_cards`
- `_representatives`
- `_deck_from_distribution`

Keep `_dict_to_batch` and `_no_grad` (see Step 1).

Update the module docstring's first paragraph to drop the belief mention. It currently reads "Inference-only PUCT MCTS search for Kaggle submission agent." — append:

```
The opponent's decklist is supplied by the caller (see
``ptcg_il.deck_prior.OpponentDeckPredictor``); this module only forwards it to
the Rust determinizer.
```

- [ ] **Step 3: Delete the orphaned tests**

Remove from `python/tests/test_search_infer.py`:
- `test_deck_from_distribution_treats_none_as_engine_ids` (183–190)
- `test_deck_from_distribution_still_maps_when_given_a_table` (192–198)
- `test_tier3_keeps_full_deck_for_high_engine_ids` (201–233)

These test tiers of a fallback chain that no longer exists. The `torch` import at line 13 may become unused — check and remove if so.

- [ ] **Step 4: Run the search_infer suite**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_search_infer.py -v`
Expected: PASS, 9 passed. The file has 12 tests before this task; the three deleted above leave 9. (An earlier draft of this plan said 11 — that was a miscount, corrected after Task 6 was implemented.) The MCTS-registration and fallback-warning tests are untouched and must all still run — a drop below 9 means something was over-deleted.

- [ ] **Step 5: Confirm the module imports without torch**

Run:
```bash
cd /home/charles/Documents/Pokemon/python && uv run python -c "
import sys, ptcg_il.search_infer
assert 'torch' not in sys.modules, 'search_infer pulled in torch'
print('clean')
"
```
Expected: `clean`. The bundle imports this at agent load; a torch dependency creeping in here is a load-time cost on every Kaggle game.

- [ ] **Step 6: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 7: Rewire the Kaggle bundle

**Files:**
- Modify: `scripts/build_submission.py` — `REWRITE_RULES` (29–53), `EXTRA_FILES` (67–71), `MAIN_PY_TEMPLATE` (128–134, 287–339), `build_model_package` (742–748), docstring (701–715)
- Modify: `tests/test_build_submission.py:222` and add new tests

**Interfaces:**
- Consumes: `ptcg_il/deck_prior.py` from Tasks 1–3
- Produces: a bundle whose `model/deck_prior.py` is importable as a flat module and whose `main.py` builds one `OpponentDeckPredictor` at import

**Note:** `tests/` (repo root) and `python/tests/` must be run separately.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_build_submission.py`. The existing `test_greedy_main_py_calls_no_search` walks the AST inline; hoist that walk into a module-level helper first so both tests share one implementation:

```python
def _imports_and_calls(source: str) -> tuple[set[str], set[str]]:
    """(imported names, called names) for a template, parsed rather than grepped.

    Both main.py templates are checked for what they do and do not reference,
    and their prose mentions the very symbols being asserted absent — so a
    substring search over the template text reports false positives.
    """
    import ast

    tree = ast.parse(source)
    imported, called = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Call):
            fn = node.func
            called.add(fn.id if isinstance(fn, ast.Name)
                       else fn.attr if isinstance(fn, ast.Attribute) else "")
    return imported, called


def test_mcts_main_py_uses_the_prior_predictor(bs):
    imported, called = _imports_and_calls(bs.MAIN_PY_TEMPLATE)

    assert "OpponentDeckPredictor" in imported
    assert "OpponentDeckPredictor" in called
    assert "mcts_search" in called
    assert "observe" in called, "the predictor must be fed each observation"
    assert "template" in called
    assert "reset" in called, "a new game must clear the previous opponent"
    # The retired path
    assert "predict_opponent_deck" not in called
    assert "extract_opp_visible_cards" not in called
    assert "belief_logits" not in called


def test_mcts_package_bundles_deck_prior(bs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = Path(bs.__file__).resolve().parents[1]
    bs.build_model_package(src, tmp_path / "sub", mcts=True)
    assert (tmp_path / "sub" / "model" / "deck_prior.py").exists()
    assert not (tmp_path / "sub" / "model" / "belief_posterior.py").exists()


def test_bundled_deck_prior_has_no_unrewritten_imports(bs, tmp_path, monkeypatch):
    """A missed rewrite rule fails only at agent runtime, on Kaggle."""
    monkeypatch.chdir(tmp_path)
    src = Path(bs.__file__).resolve().parents[1]
    bs.build_model_package(src, tmp_path / "sub", mcts=True)
    for name in ("deck_prior.py", "search_infer.py"):
        text = (tmp_path / "sub" / "model" / name).read_text()
        assert "ptcg_il" not in text, f"{name} has an unrewritten ptcg_il import"
        assert "ptcg_rl" not in text, f"{name} has an unrewritten ptcg_rl import"


def test_greedy_main_py_calls_no_search(bs):
    imported, called = _imports_and_calls(bs.MAIN_PY_TEMPLATE_GREEDY)

    assert not any("search_infer" in m for m in imported), imported
    assert not any("deck_prior" in m for m in imported), imported
    assert "mcts_search" not in called
    assert "OpponentDeckPredictor" not in called
    assert "select_multi" in called, "multi-select decisions still need the AR path"
    assert "featurize" in called
```

Replace the existing `test_greedy_main_py_calls_no_search` (lines 203–224) with the version above: its inline AST walk moves to `_imports_and_calls`, it drops the `predict_opponent_deck` assertion for a symbol that no longer exists, and it gains the `deck_prior` equivalents.

Ensure `from pathlib import Path` is imported in the test module; add it if absent.

- [ ] **Step 2: Run to verify they fail**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_build_submission.py -q`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Update `EXTRA_FILES` and `REWRITE_RULES`**

In `scripts/build_submission.py`, change `EXTRA_FILES` (lines 67–71) to:

```python
# Additional Python files to bundle (from ptcg_il/)
EXTRA_FILES: list[tuple[str, str]] = [
    # (source_rel, dest_name_in_model)
    ("ptcg_il/search_infer.py", "search_infer.py"),
    ("ptcg_il/deck_prior.py", "deck_prior.py"),
]
```

Append to `REWRITE_RULES`, before the closing `]`:

```python
    # deck_prior (copied into model/ so modules import it as model.deck_prior).
    # Both forms are listed deliberately: the module-object form has been
    # missed before, and a missed rule fails only at agent runtime on Kaggle.
    (r"from ptcg_il\.deck_prior import", r"from model.deck_prior import"),
    (r"from ptcg_il import deck_prior", r"from model import deck_prior"),
```

- [ ] **Step 4: Simplify `build_model_package`**

In `scripts/build_submission.py`, replace lines 741–748 with:

```python
        text = src.read_text()
        text = rewrite_imports(text)
        (model_dst / dest_name).write_text(text)
        print(f"  Copied + rewrote {dest_name}")
```

The hand-rolled `.replace()` chain and its `from ptcg_rl.belief import` special case go: every source is now under `ptcg_il`, so the shared `rewrite_imports` covers them, and one rewrite path means one place to keep correct.

Update `build_model_package`'s docstring (lines 701–715): replace both mentions of `belief_posterior.py` with `deck_prior.py`.

- [ ] **Step 5: Update `MAIN_PY_TEMPLATE`**

Change the imports (lines 130–134) to:

```python
from model.search_infer import mcts_search
from model.deck_prior import OpponentDeckPredictor
```

Change the header docstring (lines 105–112) first paragraph to:

```
Uses the mined archetype prior to predict the opponent's deck, then PUCT MCTS
(via bundled libptcg_search.so) to search for the best action.  Falls back to
greedy policy if MCTS is unavailable.

The opponent model is a Bayesian posterior over every archetype in
archetypes.json, seeded by mined frequency and sharpened by elimination.  No
learned belief head is involved: the packaged weights still carry them, they
are simply never called.
```

Replace lines 287–291 with:

```python
# Load archetypes and build the opponent model once.  This is the entire
# opponent model now, so archetypes.json is load-bearing: a bundle whose
# archetypes.json does not match the checkpoint's archetypes_sha1 is refused
# at packaging time, not here.
_archetypes = _load_json(os.path.join(DATA_DIR, "archetypes.json"))
_opp_predictor = OpponentDeckPredictor(_archetypes)
```

Replace the `agent` function (lines 303–339) with:

```python
def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)

    # Deck selection step — a new game starts here.
    if obs.select is None:
        _opp_predictor.reset()
        return list(_fixed_deck)

    # Fold this observation into the opponent model, then read out one
    # decklist for the determinizer.  `template()` never abstains: the mirror
    # heuristic it would otherwise fall back to shares 19.4 of their 60 cards,
    # against 44.6 for an argmax with no evidence at all.
    _opp_predictor.observe(obs_dict)
    opp_deck = _opp_predictor.template()

    # Run MCTS via bundled libptcg_search.so
    result = mcts_search(
        obs_dict,
        fixed_deck=_fixed_deck,
        opp_deck=opp_deck,
        policy=_model,
        vocab=_vocab,
        device=_device,
        libcg_path=_find_libcg(),
        iterations=_MCTS_ITERATIONS,
        c_puct=_MCTS_C_PUCT,
        seed=_MCTS_SEED,
    )

    return result.get("indices", [])
```

Delete the now-unused `global _opp_visible_cards` and the module-level `_opp_visible_cards: list[int] = []`.

- [ ] **Step 6: Run the bundle tests**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_build_submission.py -v`
Expected: PASS.

- [ ] **Step 7: Build a real bundle and exec it with no `__file__`**

The Kaggle runner execs `main.py` into a namespace with no `__file__`, so an `import main` smoke test binds it and hides this whole class of failure.

Run:
```bash
cd /home/charles/Documents/Pokemon
./scripts/build_submit.sh a0 2>&1 | tail -20
```

Then, against the built bundle directory (adjust the path to whatever the script reports):
```bash
cd /home/charles/Documents/Pokemon/submission && uv run python -c "
import sys; sys.path.insert(0, '.')
ns = {'__name__': '__main__'}   # deliberately no __file__
exec(open('main.py').read(), ns)
print('agent loaded:', callable(ns['agent']))
print('predictor ids:', len(ns['_opp_predictor'].ids))
print('template len:', len(ns['_opp_predictor'].template()))
"
```
Expected: `agent loaded: True`, an id count in the hundreds (not 9), and `template len: 60`. A `NameError` on `__file__` here is the failure this step exists to catch.

- [ ] **Step 8: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 8: Delete `belief_infer.py`

Deletion comes last so the tree stays green throughout. By now every caller is rewired.

**Files:**
- Delete: `python/ptcg_il/belief_infer.py`
- Modify: `python/tests/test_belief.py` — delete `TestDeckFromDistribution` (369–405), the `oracle` fixture (407–424), `TestOpponentDeckOracle` (426–489), and the line-17 import

**Interfaces:** none produced; this task only removes.

- [ ] **Step 1: Confirm there are no callers left**

Run:
```bash
cd /home/charles/Documents/Pokemon
grep -rn "belief_infer\|OpponentDeckOracle\|deck_from_distribution" \
  --include="*.py" --include="*.sh" --include="*.md" . \
  | grep -v worktrees | grep -v docs/superpowers
```
Expected: only `python/tests/test_belief.py`. Anything else means a task was skipped — stop and report rather than deleting.

- [ ] **Step 2: Delete the module**

```bash
cd /home/charles/Documents/Pokemon
rm python/ptcg_il/belief_infer.py
```

- [ ] **Step 3: Prune the orphaned tests**

In `python/tests/test_belief.py`:
- delete line 17: `from ptcg_il.belief_infer import OpponentDeckOracle, deck_from_distribution`
- delete `class TestDeckFromDistribution` (369–405)
- delete the `oracle` fixture (407–424)
- delete `class TestOpponentDeckOracle` (426–489)

**Keep everything else.** `TestDeckCounts`, `TestVisibleCounts`, `TestHandTimeline`, `TestSparseRoundTrip`, `TestBeliefAreaEncoding`, `TestBeliefHeads`, `TestBeliefLoss`, `TestDatasetDensification` and the two module-level forward tests all cover the belief heads' **training** path, which is unchanged and still runs. Deleting them would drop coverage of a live code path.

Check whether `numpy`, `json` or `tmp_path` imports are now unused at the top of the file and remove only those that are.

- [ ] **Step 4: Run the belief suite**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_belief.py -v`
Expected: PASS. The count should drop by roughly 16 (the deleted classes) and no more.

- [ ] **Step 5: Full suite, both directories, separately**

Run:
```bash
cd /home/charles/Documents/Pokemon
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/ -q 2>&1 | tail -15
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q 2>&1 | tail -15
```
Expected: both PASS. Report any failure with its output rather than adjusting the test to pass.

- [ ] **Step 6: Confirm training is genuinely untouched**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/ -q -k "train_loop or belief or policy"`
Expected: PASS. The belief heads still train; only their inference consumers are gone.

- [ ] **Step 7: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 9: Live A/B — the ship gate

Every number motivating this change is a simulation against `archetypes.json`. None is a played game. This task produces the only evidence that decides whether it ships.

**Files:**
- Create: `docs/superpowers/specs/2026-08-12-prior-determinizer-results.md`
- No source changes unless the result demands them.

**Interfaces:** none.

- [ ] **Step 1: Pick the checkpoint**

Run: `cd /home/charles/Documents/Pokemon/python && uv run python -m ptcg_il.cli archetypes --data-dir data --describe`

Use an archetype that has a trained `checkpoints_a<N>/ckpt-best.pt`. Do **not** hardcode an id — `run_pipeline.sh` derives them via `ptcg_il/archetype_select.py`, and this corpus's `self_ids` are `[1, 17, 25, 16, 36, 21, 0, 95, 27, 204, 49]`, not `0 1 2`.

- [ ] **Step 2: Run the paired evaluation**

Both planners are registered as separate opponent keys, so one run measures both:

```bash
cd /home/charles/Documents/Pokemon/python
uv run python -m ptcg_il.cli train --eval-only --live-eval \
    --live-eval-games 1000 \
    --data-dir data --archetype-self <N> \
    --resume checkpoints_a<N>/ckpt-best.pt \
    2>&1 | tee /tmp/prior-determinizer-ab.log
```

`--live-eval-games` is per opponent per run (default 500; 1000 above). At 1000 games a win-rate difference has a 95% half-width of roughly ±3 points, so an effect smaller than that will not be resolvable — raise the count if the first run lands inside the noise. Throughput is roughly 6.2k games/h per the arena cost profile, and this run covers three opponents (`random`, `search_planner`, `search_planner_prior`), so budget ~30 min at 1000.

- [ ] **Step 3: Read out the comparison**

The two numbers that matter are the win rates for `search_planner` (mirror determinization) and `search_planner_prior` (this change). Record both with their game counts.

- [ ] **Step 4: Write up the result**

Create `docs/superpowers/specs/2026-08-12-prior-determinizer-results.md` with: the checkpoint and its archetype id, game counts, both win rates with confidence intervals, the command used, and a one-line verdict. Record the result whichever way it goes — a null result here is worth as much as a positive one, because the simulated overlap gain is large enough that its absence in real games would mean the determinizer is not the bottleneck.

- [ ] **Step 5: Decide**

- **Prior planner wins:** ship. Rebuild the submission (`./scripts/build_submit.sh`) and note in the results doc which bundle carries it.
- **No significant difference:** the change still stands on its own — it deletes a dead RL path, removes a forward pass per decision, and consolidates three copies of the visible-cards logic — but say so plainly in the writeup rather than implying a win.
- **Prior planner loses:** stop and report. Do not tune `epsilon` or the prior to chase it; a loss against the mirror despite a 3× overlap advantage means something about the determinizer's use of the template is not what this design assumed, and that needs diagnosis, not knob-turning.

- [ ] **Step 6: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

# Phase 2 — Remove the belief machinery

**Do not start Phase 2 until Task 9 is complete.** Phase 1 removes the belief heads' only consumers; these tasks remove the heads.

## What is being deleted, and why it is two things

`ptcg_il/model/belief.py` defines two unrelated objects that share a name. Conflating them is the main hazard in this phase.

**Half A — `BeliefHeads`** (`belief.py:126`): four auxiliary prediction heads (`arch`/`deck`/`hidden`/`hand`) reading the CLS token, plus `soft_cross_entropy` and `belief_loss`. Supervised by the `bel_*` shard labels. Nothing reads its output after Phase 1. Gated today by `--no-belief`.

**Half B — `BeliefModule`** (`belief.py:33`): encodes `log_feat` (the game log) into a `[B, D]` vector which `policy.py:299` **adds to the CLS token**: `cls_token = h[:, 0, :] + belief`. The value head and the pointer's CLS key/value read that token, so this is a **policy input, not a head**. `--no-belief` does not gate it. Since `history_gru` was removed (`policy.py:268-278`), it is the model's only remaining cross-turn mechanism — removing it makes the policy purely Markov on the current observation.

Measured cost of each, on a real 26,586-row shard:

| | shard bytes | loader, per batch @ bs=1024 | model fwd+bwd |
|---|---|---|---|
| Half A (`bel_*`) | 9.89 MiB / 2.9% | 14.9 MiB densify | 0.5% |
| Half B (`log_*`) | 20.49 MiB / 6.0% | 26.5 MiB `log_card_feat` rebuild | included above |
| both | 30.37 MiB / **8.9%** | **41.4 MiB** of ~437 MB/batch (~9.5%) | ~0.5% |

Plus ~7% of build-shards pass B for the label construction.

**Half B is an unmeasured ablation, and this plan treats it as one.** Task 15 measures it against the current baseline rather than assuming it is free. If it regresses, the fix is to restore `BeliefModule` alone — Half A's removal is independent and stands either way.

## Sequencing

Model-side removal comes before producer-side removal, and the shard rebuild comes last. This is deliberate: the dataset reads an explicit key list, so **existing shards keep working with the new code** — their `bel_*` and `log_*` arrays are simply never read. That decouples the code change from the ~23-minute rebuild and keeps any in-flight training safe.

---

### Task 10: Make belief-carrying checkpoints loadable

Do this first. Every later task depends on old checkpoints still loading, and doing it last would mean a window where the in-flight run's output is unloadable.

**Files:**
- Modify: `python/ptcg_il/model/policy.py:365-380` (prefix constants), `440-460` (stale drop)
- Modify: `python/tests/test_model_policy.py`

**Interfaces:**
- Consumes: nothing
- Produces: `load_policy_state` drops `belief.*` and `belief_heads.*` from a checkpoint's state dict the same way it already drops `history_gru.*`

- [ ] **Step 1: Write the failing test**

Append to `python/tests/test_model_policy.py`:

```python
def test_belief_carrying_checkpoint_still_loads():
    """A checkpoint written before the belief removal carries `belief.*` and
    `belief_heads.*` parameters the current Policy does not define.  Those must
    be dropped like the retired `history_gru.*` ones, not raised on: the run
    that produced them cannot be repeated.
    """
    from ptcg_il.model.policy import Policy, load_policy_state

    policy = _tiny_policy()          # existing helper in this module
    state = dict(policy.state_dict())
    # Simulate the pre-removal checkpoint.
    state["belief.gru.weight_ih"] = torch.zeros(3, 4)
    state["belief_heads.arch_head.weight"] = torch.zeros(9, 8)
    state["belief_heads.arch_head.bias"] = torch.zeros(9)

    load_policy_state(policy, state)   # must not raise


def test_a_genuinely_unknown_key_still_raises():
    """The drop must be prefix-scoped.  Swallowing every unexpected key would
    turn a real load mismatch into a silently half-initialised model."""
    from ptcg_il.model.policy import Policy, load_policy_state

    policy = _tiny_policy()
    state = dict(policy.state_dict())
    state["some_module_that_never_existed.weight"] = torch.zeros(2, 2)

    with pytest.raises(Exception):
        load_policy_state(policy, state)
```

If `_tiny_policy()` does not exist in that module, build a `Policy(D=8, heads=2, layers=1, ff=16, n_all_cards=4, all_card_feat=torch.zeros(4, F_CARD), all_attack_feat=torch.zeros(4, F_ATK))` inline — match whatever construction the file's existing tests use.

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_model_policy.py -q -k "belief_carrying or unknown_key"`
Expected: FAIL on unexpected keys. (It may pass today because the heads still exist — that is fine, it becomes the regression guard the moment Task 11 removes them. Re-run it after Task 11 and confirm it still passes.)

- [ ] **Step 3: Extend the stale-prefix drop**

In `python/ptcg_il/model/policy.py`, next to `_HISTORY_GRU_PREFIX` (line 369), add:

```python
# Retired alongside `history_gru`: the auxiliary belief heads (`belief_heads.*`)
# and the game-log encoder (`belief.*`).  Checkpoints written before their
# removal carry both, and they would land in `unexpected` and raise.  Dropping
# them is safe in the same narrow way: the modules are gone and nothing reads
# their weights.  Anything else unexpected still raises.
_RETIRED_PREFIXES = ("history_gru.", "belief.", "belief_heads.")
```

Replace the `stale = [...]` block (lines 450–458) with:

```python
    stale = [k for k in state_dict if k.startswith(_RETIRED_PREFIXES)]
    if stale:
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith(_RETIRED_PREFIXES)}
        logger.info(
            "dropped %d parameter(s) from retired modules (%s); their weights "
            "are no longer read by any code path",
            len(stale), ", ".join(sorted({k.split(".")[0] for k in stale})),
        )
```

Keep `_HISTORY_GRU_PREFIX` defined if anything else references it; otherwise remove it.

`str.startswith` accepts a tuple, so no loop is needed. Note that `"belief."` must not be written as `"belief"` — the latter would also match `belief_heads.` (harmless here) *and* any future attribute beginning with those letters (not harmless).

- [ ] **Step 4: Run to verify it passes**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_model_policy.py -q`
Expected: PASS.

- [ ] **Step 5: Mutation-check the scoping**

Change `_RETIRED_PREFIXES` to `("",)` (drop everything).
Expected: `test_a_genuinely_unknown_key_still_raises` FAILS. Restore.

- [ ] **Step 6: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 11: Remove Half A — the auxiliary heads, loss, and metrics

**Files:**
- Modify: `python/ptcg_il/model/policy.py` — drop `self.belief_heads` (140), `n_opp_arch` (123/142/149/572), `belief_logits` (333-345), `forward_with_belief` (347-359), `_ARCH_HEAD_PREFIX` (365), `widen_belief_arch_head`, `allow_belief_widening` plumbing
- Modify: `python/ptcg_il/model/belief.py` — delete `BeliefHeads` (126-232), `soft_cross_entropy` (234-256), `belief_loss` (258+). **Keep `BeliefModule`** — Task 12 removes it.
- Modify: `python/ptcg_il/train/loop.py` — `belief_weights` params (507/629/702), the `want_belief` branch (525-532), the loss block (608-616), `BELIEF_WEIGHTS` re-export (28), `belief_loss` import (29), `allow_belief_widening` (726/926), logger `belief=` (1081)
- Modify: `python/ptcg_il/train/eval.py` — `belief` param (32), the accumulator (109-132), `_accum_belief` (359-398), `_belief_metrics` (400-405), the `metrics.update` (353-354)
- Modify: `python/ptcg_il/ensemble.py` — delete `belief_logits` (238-243) and `forward_with_belief` (245-251)
- Modify: `python/tests/test_belief.py`, `test_train_loop.py`, `test_model_policy.py`, `test_ensemble.py`, `test_muon_partition.py`

**Interfaces:**
- Consumes: Task 10
- Produces: `Policy` with no `belief_heads` and no `n_opp_arch`; `Policy.forward` unchanged in signature; `train_step` and `offline_eval` with no `belief_weights` / `belief` parameter

- [ ] **Step 1: Delete the model pieces**

Work top-down so each edit's compile error points at the next site. After each file, run `python -c "import ptcg_il.model.policy"` to catch syntax damage early.

Delete from `model/belief.py`: `BeliefHeads`, `soft_cross_entropy`, `belief_loss`. Leave `BeliefModule` and the module docstring's description of it; delete the docstring paragraphs describing the heads.

Delete from `model/policy.py`: the `belief_heads` construction and its `set_all_card_feat` wiring (187), `n_opp_arch` from `__init__` and from the `config` dict (149), `belief_logits`, `forward_with_belief`, `_ARCH_HEAD_PREFIX`, `widen_belief_arch_head`, and the `allow_belief_widening` parameter of `load_policy_state` together with its `belief_missing` bookkeeping (471-472).

**Keep** `self.belief = BeliefModule(D)` (138), `self.belief.card_emb = self.embed.card` (139), and the `_encode` belief residual (287-300). Those are Half B.

- [ ] **Step 2: Delete the training-loop pieces**

In `train/loop.py`, remove the `belief_weights` parameter from `_compute_loss`, `train_step` and `train` and every call site; delete the `want_belief` branch so `_encode`-based `policy(batch)` is the only path; delete the `belief_parts` block and its `**belief_parts` spreads (684, 1049); drop the `BELIEF_WEIGHTS` re-export and `belief_loss` import; drop `allow_belief_widening` and the `belief=` kwarg to the logger.

In `train/eval.py`, remove the `belief` parameter, the `belief_acc` dict, the `forward_with_belief` branch (130-132) leaving only the plain forward, `_accum_belief`, `_belief_metrics`, and the `metrics.update` call.

In `ensemble.py`, delete `belief_logits` and `forward_with_belief`.

- [ ] **Step 3: Update the tests**

In `python/tests/test_belief.py`, delete `TestBeliefHeads` (278) and `TestBeliefLoss` (313), and the `_heads()` helper (205) they share.

**Leave these alone in this task:**
- `TestBeliefAreaEncoding` (213) builds `L_LOG_MAX × LOG_FEAT_DIM` tensors — it tests `BeliefModule`, which is Half B. Task 12 deletes it.
- `TestDeckCounts`, `TestVisibleCounts`, `TestHandTimeline`, `TestSparseRoundTrip`, `TestDatasetDensification` cover label construction and densification, which still run until Task 13.

In `test_train_loop.py`, `test_model_policy.py`, `test_ensemble.py`, `test_muon_partition.py`: remove `belief_weights=` kwargs, `n_opp_arch=` kwargs, and any assertion naming `belief_heads`. `test_muon_partition.py` asserts the optimizer's parameter partition — if it enumerates belief head params, the expected set shrinks.

- [ ] **Step 4: Run the suites**

Run:
```bash
cd /home/charles/Documents/Pokemon
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/ -q 2>&1 | tail -15
```
Expected: PASS. Fix real failures; do not delete a test to make it pass without confirming what it covered is genuinely gone.

- [ ] **Step 5: Confirm training still runs end to end on existing shards**

Run:
```bash
cd /home/charles/Documents/Pokemon/python
uv run python -m ptcg_il.cli train --data-dir data --out-dir /tmp/belief-smoke \
    --archetype-self 1 --total-steps 20 --val-every 10 --batch-size 32 \
    --no-wandb 2>&1 | tail -20
```
Expected: 20 steps complete and a val line prints. The shards still contain `bel_*` arrays; nothing should read them. If this OOMs or contends with the in-flight run, add `CUDA_VISIBLE_DEVICES=""` to force CPU.

- [ ] **Step 6: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 12: Remove Half B — the game-log encoder

This is the ablation. It changes what the policy sees.

**Files:**
- Modify: `python/ptcg_il/model/policy.py:138-139` (construction), `248-253` (`log_card_feat` gather), `286-300` (the CLS residual)
- Delete: `python/ptcg_il/model/belief.py` (only `BeliefModule` remains after Task 11)
- Modify: `python/ptcg_il/diagnose.py:70,186,208-210,423-426`
- Modify: `python/tests/test_belief.py` — delete `TestBeliefAreaEncoding` (213-276), which tests this module
- Modify: `python/tests/test_model_encoder.py`

**Interfaces:**
- Consumes: Tasks 10-11
- Produces: `Policy._encode` returning the encoder's CLS token unmodified

- [ ] **Step 1: Write the characterisation test first**

Before deleting, pin what changes. Add to `python/tests/test_model_policy.py`:

```python
def test_cls_token_is_the_encoders_own_output():
    """After the log encoder's removal the CLS row passes through untouched.

    `policy.py` used to do `cls_token = h[:, 0, :] + belief`.  This asserts the
    residual is gone, so a future re-introduction cannot happen silently.
    """
    from ptcg_il.model.policy import Policy

    policy = _tiny_policy().eval()
    assert not hasattr(policy, "belief"), "the log encoder is still constructed"

    batch = _tiny_batch()          # existing helper; must NOT contain log_feat
    import torch
    with torch.no_grad():
        _x, h, _hist = policy._encode(batch)
        rows = policy.embed(policy._gather_card_feats(dict(batch)))
        expected = policy.encoder(rows, batch["tok_mask"])
    assert torch.allclose(h, expected, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/test_model_policy.py -q -k cls_token`
Expected: FAIL — `policy.belief` still exists.

- [ ] **Step 3: Delete the encoder**

In `model/policy.py`:
- delete `self.belief = BeliefModule(D)` and `self.belief.card_emb = self.embed.card`
- delete the `log_card_feat` gather in `_gather_card_feats` (248-253)
- replace the `_encode` block at 286-300 with nothing: `h` from `self.encoder(...)` is returned as-is. Keep the `history_h` zero-fill and the `(x, h, history_h)` return shape — callers still unpack three values, and changing that signature is a separate, larger edit.
- rewrite the `_encode` docstring: it currently explains the belief-augmented CLS and the retired `history_gru`. State that the CLS row is now the encoder's own output, and keep the `history_gru` paragraph as the record of why `history_h` is inert.

Delete `python/ptcg_il/model/belief.py`.

In `diagnose.py`, remove the `"game log (belief)"` entry (70), `log_feat` from the key loop (186), the `log_mask` occupancy block (208-210) and its report line (423-426).

- [ ] **Step 4: Run to verify it passes**

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/ -q 2>&1 | tail -15`
Expected: PASS. `test_belief.py` should now be empty or nearly so — if every remaining class in it covered `BeliefModule`, delete the file.

- [ ] **Step 5: Confirm the bundle's model package no longer needs belief.py**

In `scripts/build_submission.py`, remove `"belief.py"` from `MODEL_FILES` (line 61).

Run: `cd /home/charles/Documents/Pokemon && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_build_submission.py -q`
Expected: PASS.

- [ ] **Step 6: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 13: Stop producing labels and log tensors

Only now does anything stop being *written*. Until this task the shards are unchanged.

**Files:**
- Delete: `python/ptcg_il/belief_labels.py`
- Modify: `python/ptcg_il/shard_writer.py:24-27` (imports), `730-745` (label construction), `607` (`_DERIVED_KEYS`)
- Modify: `python/ptcg_il/featurizer.py:70-71` (`L_LOG_MAX`, `LOG_FEAT_DIM`), `1373-1400` (log tensor builder), `1489`, `1507` (`LOG_CARD_ID_COLUMN`), `1715`, `1793`, and the `log_*` entries in the returned dict
- Modify: `python/ptcg_il/train/dataset.py:93,100,105,111,118-120,514,588,796,799-840`
- Modify: `python/ptcg_mine/stamp.py:94` (drop `ptcg_il/belief_labels.py` from the fingerprint)

**Interfaces:**
- Consumes: Tasks 10-12
- Produces: `featurize()` emitting no `log_*` keys; `build_shards` writing no `bel_*` or `log_*` arrays; `ShardDataset` reading neither

- [ ] **Step 1: Remove the label producer**

Delete `python/ptcg_il/belief_labels.py`. In `shard_writer.py`, delete the import block (24-27), the `build_belief_labels(...)` call and the `empty_belief_labels()` fallback (730-745), and `"log_card_feat"` from `_DERIVED_KEYS` (607).

In `ptcg_mine/stamp.py`, remove `"ptcg_il/belief_labels.py"` from the fingerprint source list (94). **This is load-bearing**: the stamp hashes source files by path, and a missing file must not silently become an empty hash that makes a stale corpus look fresh. Verify the stamp still computes after the edit.

- [ ] **Step 2: Remove the log tensors from the featurizer**

In `featurizer.py`, delete `L_LOG_MAX` and `LOG_FEAT_DIM` (70-71), the log tensor builder (~1373-1400), `LOG_CARD_ID_COLUMN` (1507), and the `log_feat` / `log_mask` / `log_len` entries from `featurize()`'s output dict (~1715, ~1793).

Do **not** touch `current_feature_dims()` beyond removing the log entries if present. `policy.py:530` compares only keys present in both the checkpoint's record and the live dims (`if k in live`), so a *removed* key is tolerated and an old checkpoint will not raise on it. Changing a surviving key's value would raise — so remove, never repurpose.

- [ ] **Step 3: Remove the dataset plumbing**

In `train/dataset.py`, delete `_BELIEF_SPARSE` (118-120), `_densify_belief` (799-840) and its call (796), the `bel_*` and `log_*` entries from the key lists (93, 100, 105, 111), `"log_card_feat"` from `_DERIVED_FEAT_KEYS` (588), and the `n_all_cards` plumbing at 514 **only if** nothing else uses it — the belief head's output width is gone, but check before deleting.

- [ ] **Step 4: Confirm existing shards still load**

This is the compatibility check that protects the in-flight run's corpus.

```bash
cd /home/charles/Documents/Pokemon/python && uv run python -c "
from ptcg_il.train.dataset import ShardDataset
ds = ShardDataset('data/shards', split='val')
s = ds[0]
bad = [k for k in s if k.startswith('bel_') or k.startswith('log_')]
print(f'{len(ds)} samples, {len(s)} keys')
assert not bad, f'belief/log keys still reaching the batch: {bad}'
print('clean')
"
```
Expected: `clean`. The `.npz` files still contain those arrays; the dataset must simply not read them.

- [ ] **Step 5: Run the suites**

Run:
```bash
cd /home/charles/Documents/Pokemon
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/ -q 2>&1 | tail -15
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q 2>&1 | tail -15
```
Expected: both PASS. `test_featurizer.py` and `test_shard_writer.py` will need their `log_*` / `bel_*` assertions removed.

- [ ] **Step 6: Confirm a fresh shard is smaller**

Build a small corpus slice and compare against the old shard's key set:

```bash
cd /home/charles/Documents/Pokemon/python
uv run python -m ptcg_il.cli build-shards --raw-dir raw --out-dir /tmp/shards-nobelief \
    --samples-per-shard 3000 --force 2>&1 | tail -5
uv run python -c "
import numpy as np, glob
old = np.load(sorted(glob.glob('data/shards/train-*.npz'))[0])
new = np.load(sorted(glob.glob('/tmp/shards-nobelief/shards/train-*.npz'))[0])
go = sum(old[k].nbytes for k in old.files) / old[old.files[0]].shape[0]
gn = sum(new[k].nbytes for k in new.files) / new[new.files[0]].shape[0]
print(f'per-row bytes: {go/1024:.2f} KiB -> {gn/1024:.2f} KiB  ({1-gn/go:.1%} smaller)')
print('removed keys:', sorted(set(old.files) - set(new.files)))
"
```
Expected: ~8.9% smaller, and the removed keys are exactly the `bel_*` and `log_*` set. A smaller reduction means something is still being written.

- [ ] **Step 7: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 14: Retire the CLI surface and the RL plumbing

**Files:**
- Modify: `python/ptcg_il/cli.py` — `_belief_weights` (398-410), the `--belief-*` / `--no-belief` flags (~261-270), `--allow-belief-widening`, `n_opp_arch` in `_build_policy` (~33-45), the `belief=` kwarg (733)
- Modify: `python/ptcg_rl/train.py:160,192,229-235`; `python/ptcg_rl/mcts_train.py:294,338-347`
- Modify: `python/ptcg_mine/mine.py:108,118,379-385`; `python/ptcg_mine/archetype.py` docstrings; `python/ptcg_mine/config.py:34`
- Modify: `scripts/run_pipeline.sh` and any script passing `--belief`/`--no-belief`

**Interfaces:**
- Consumes: Tasks 10-13
- Produces: no belief flags anywhere; `Policy` constructed without `n_opp_arch`

- [ ] **Step 1: Find every flag reference**

Run:
```bash
cd /home/charles/Documents/Pokemon
grep -rn -- "--belief\|--no-belief\|allow.belief.widening\|belief_weights\|n_opp_arch" \
  --include="*.py" --include="*.sh" . | grep -v worktrees
```
Record the list; every line must be gone or justified by the end of this task.

- [ ] **Step 2: Remove the CLI surface**

Delete `_belief_weights`, the `--belief-arch/-deck/-hidden/-hand` weight flags, `--no-belief`, and `--allow-belief-widening` from `cli.py`, along with `belief_weights=` and `belief=` at the call sites (632, 733) and `n_opp_arch` from `_build_policy`.

`--allow-belief-widening` deserves a note in the commit message: it existed because `opp_ids` order was the belief head's class order, and the whole class of bug it guarded against (a re-sorted `opp_ids` silently repointing every class) disappears with the head.

- [ ] **Step 3: Remove the RL plumbing**

In `ptcg_rl/train.py` and `ptcg_rl/mcts_train.py`, delete the `if key.startswith("belief_heads."):` filters in the partial-load helpers and the `n_opp_arch` inference. Task 10's `_RETIRED_PREFIXES` now covers what those filters were doing, in one place.

- [ ] **Step 4: Update the mining docs**

`ptcg_mine` has **no belief code** — every hit is prose about `opp_ids` being append-only "because the belief head indexes by position". That rationale is now historical. Update `mine.py:108,118`, `archetype.py:67,122,191,198,226` and `config.py:34` to say ids are append-only because **trained checkpoints and `--archetype-self N` reference them**, which remains true. Leave `opp_ids` in `archetypes.json`: it is now unused metadata, and removing it from the artifact is a separate change with its own lineage consequences.

Delete the `--allow-belief-widening` guidance at `mine.py:379-385`.

- [ ] **Step 5: Update CLAUDE.md**

Several bullets now describe removed machinery: "Belief heads always train", the `--belief` flag note, the `𝒟_opp`/`bel_arch` discussion in the corpus-filter bullet, and the `allow_belief_widening` paragraph in the archetype-ids bullet. Rewrite them to describe what the code does now, keeping the *measurements* (the 92.3%/7.7% filter numbers, the sharpness-20 derivation) which are still true and still load-bearing.

- [ ] **Step 6: Run everything**

Run:
```bash
cd /home/charles/Documents/Pokemon
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest python/tests/ -q 2>&1 | tail -15
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/ -q 2>&1 | tail -15
grep -rn "belief" --include="*.py" . | grep -v worktrees | grep -v "label" | wc -l
```
Expected: both suites PASS, and the grep count is 0 (or only comments you deliberately kept).

- [ ] **Step 7: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

### Task 15: Rebuild the corpus and measure the ablation

**Check `nvidia-smi` first.** Do not start until the in-flight training run has finished — this task rebuilds the shards it is reading.

**Files:**
- Create: `docs/superpowers/specs/2026-08-12-belief-removal-results.md`

**Interfaces:** none.

- [ ] **Step 1: Confirm nothing is training**

Run: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
Expected: no `ptcg_il.cli train` process. If one is running, stop and wait.

- [ ] **Step 2: Record the baseline before rebuilding**

The comparison needs a number from *before* the change. Take it from the existing checkpoint rather than retraining:

```bash
cd /home/charles/Documents/Pokemon/python
uv run python -m ptcg_il.cli train --eval-only --data-dir data \
    --archetype-self 1 --resume checkpoints_a1_s2/ckpt-best.pt \
    --eval-split test 2>&1 | tee /tmp/baseline-with-belief.log
```

Record `nontrivial_top1`. If this checkpoint predates the removal it carries belief weights — Task 10 makes it load, and the eval path no longer calls the heads, so the number measures the trunk as trained *with* the auxiliary loss and *with* the log encoder.

- [ ] **Step 3: Rebuild the shards**

```bash
cd /home/charles/Documents/Pokemon/python
uv run python -m ptcg_il.cli build-shards --raw-dir raw --out-dir data --force \
    2>&1 | tail -20
```
Expected: ~23 minutes. Record the wall time against the pre-change figure — this is where the ~7% pass-B saving shows up.

Record the corpus size before and after: `du -sh data/shards`.

- [ ] **Step 4: Retrain the same archetype and seed**

```bash
cd /home/charles/Documents/Pokemon/python
uv run python -m ptcg_il.cli train --data-dir data --out-dir checkpoints_a1_s2_nobelief \
    --archetype-self 1 --seed 2 --total-steps 15000 --batch-size 1024 \
    2>&1 | tee /tmp/train-nobelief.log
```

Same archetype, same seed, same steps, same batch size as the baseline run — otherwise the comparison measures the difference in hyperparameters.

- [ ] **Step 5: Compare**

```bash
cd /home/charles/Documents/Pokemon/python
uv run python -m ptcg_il.cli train --eval-only --data-dir data \
    --archetype-self 1 --resume checkpoints_a1_s2_nobelief/ckpt-best.pt \
    --eval-split test 2>&1 | tee /tmp/nobelief-test.log
```

Record: test `nontrivial_top1` before vs after, steps/sec from both training logs, corpus size before vs after, build-shards wall time before vs after.

- [ ] **Step 6: Write it up and decide**

Create `docs/superpowers/specs/2026-08-12-belief-removal-results.md` with all six numbers and a verdict.

- **Accuracy holds or improves:** done. The speed numbers are the payoff and the codebase is materially simpler.
- **Accuracy regresses:** the likely culprit is Half B, not Half A — the log encoder was a real policy input, and Half A was measured at 0.5% of compute with no consumer. Restoring `BeliefModule` alone is the targeted fix: revert Task 12 and Task 13's featurizer/log portions, keep everything else. Do not revert wholesale, and do not conclude the auxiliary heads mattered without testing that separately.
- Record the result either way. A regression here is the measurement that Phase 2 was explicitly taken without.

- [ ] **Step 7: Do NOT commit**

This run makes no commits (see Global Constraints). Leave the changes in the working tree and report the files you touched. Do not run any git subcommand that writes.

---

## Appendix: measurements behind the constants

Reproduced from the spec so an implementer need not switch documents.

**Template quality** (mean of the opponent's 60 cards the determinizer gets right):

| k cards seen | mirror | posterior over 9 `opp_ids` | oracle best of those 9 | posterior over all 362 |
|---|---|---|---|---|
| 1 | 19.4 | 42.7 | 51.8 | 44.6 |
| 4 | 19.4 | 50.0 | 51.5 | 57.1 |
| 8 | 19.5 | 50.7 | 51.3 | 59.1 |
| 20 | 19.4 | 51.6 | 51.8 | 59.6 |

`opp_ids` covers 68.7% of games by frequency, which is what caps the oracle column at ~52.

**Prior choice** (top-1 archetype identification): uniform 0.295 / 0.255 / 0.343 / 0.590 at k=1/4/8/20 against frequency 0.560 / 0.818 / 0.885 / 0.943.

**Abstention** (mean overlap, always-argmax vs abstain-to-mirror below p≥0.80): 44.6 vs 27.3 at k=1; 57.1 vs 42.5 at k=4; 59.6 vs 55.8 at k=20.

**Epsilon**, under opponents playing a d-card variant of their cluster:

| d | k | ε=1e-6 | ε=1e-3 | ε=1e-2 | ε=0.05 | ε=0.2 |
|---|---|---|---|---|---|---|
| 0 | 4 | 57.6 | 56.8 | 53.5 | 31.9 | 13.0 |
| 4 | 4 | 53.5 | 53.8 | 50.8 | 30.1 | 13.1 |
| 10 | 16 | 50.5 | 50.6 | 49.9 | 36.1 | 6.9 |

**Cost:** 3.34 ms to construct the 362-hypothesis posterior, 0.55 ms per `posterior()` call with 15 cards observed.
