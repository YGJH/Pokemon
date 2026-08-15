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

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence

# Probability mass reserved for "this archetype could contain a card I have not
# accounted for".  Without it a single mis-parsed observation, or an opponent
# playing a one-card variant of a known list, permanently zeroes the correct
# hypothesis and the posterior never recovers.
DEFAULT_EPSILON = 1e-3

# Above this posterior mass on the top archetype, `argmax` is worth acting on.
# Below it, the caller should fall back rather than determinize on a guess.
CONFIDENT = 0.80

# Every mined decklist is exactly 60 cards (Pokémon TCG deck-size rule).
DECK_SIZE = 60


class ArchetypePosterior:
    """Bayesian belief over which 𝒟_opp archetype the opponent is playing.

    Parameters
    ----------
    archetypes : dict
        The parsed ``archetypes.json``.
    opp_ids : sequence of int, optional
        Which archetype ids form the hypothesis set.  Defaults to the file's
        ``opp_ids``.
    priors : sequence of float, optional
        Prior over *opp_ids*.  Defaults to uniform.  The spec suggests mined
        frequencies; uniform is the honest default when they are absent, since a
        made-up prior would bias every determinization in the same direction.
    epsilon : float
        Mass reserved for unseen cards, per :data:`DEFAULT_EPSILON`.
    """

    def __init__(
        self,
        archetypes: dict[str, Any],
        opp_ids: Sequence[int] | None = None,
        priors: Sequence[float] | None = None,
        epsilon: float = DEFAULT_EPSILON,
    ):
        if not 0.0 <= epsilon < 1.0:
            raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
        self.epsilon = epsilon

        ids = list(opp_ids) if opp_ids is not None else [
            int(i) for i in archetypes.get("opp_ids", [])
        ]
        if not ids:
            raise ValueError("no opponent archetype ids to form a hypothesis set")
        self.opp_ids: list[int] = [int(i) for i in ids]

        self._counts: list[Counter[int]] = [
            Counter(_representative_of(archetypes, aid)) for aid in self.opp_ids
        ]
        missing = [aid for aid, c in zip(self.opp_ids, self._counts) if not c]
        if missing:
            raise ValueError(
                f"archetypes {missing} have no representative decklist; the "
                f"posterior cannot score a hypothesis it cannot enumerate"
            )

        if priors is None:
            self._log_prior = [-math.log(len(self.opp_ids))] * len(self.opp_ids)
        else:
            if len(priors) != len(self.opp_ids):
                raise ValueError(
                    f"priors has {len(priors)} entries for {len(self.opp_ids)} archetypes"
                )
            total = float(sum(priors))
            if total <= 0:
                raise ValueError("priors must sum to something positive")
            self._log_prior = [
                math.log(p / total) if p > 0 else -math.inf for p in priors
            ]

    # ── Core ────────────────────────────────────────────────────────────────

    def posterior(self, observed: Iterable[int]) -> dict[int, float]:
        """``{archetype_id: probability}`` given the opponent cards seen so far.

        *observed* is a multiset of engine card ids — pass every copy seen, not
        the distinct set, or the multiplicity evidence is thrown away.

        A hypothesis that cannot explain the observation keeps only ``epsilon``
        per unexplained copy rather than dropping to exactly zero, so the belief
        degrades instead of dying.  When *every* hypothesis is ruled out the
        result falls back to the prior, which is the right answer to "the
        opponent is playing something I have never seen".
        """
        seen = Counter(int(c) for c in observed)

        log_post = []
        for log_prior, deck in zip(self._log_prior, self._counts):
            lp = log_prior
            deck_size = sum(deck.values())
            for card, n_seen in seen.items():
                available = deck.get(card, 0)
                explained = min(n_seen, available)
                unexplained = n_seen - explained
                if explained:
                    lp += explained * math.log(available / deck_size)
                if unexplained:
                    lp += unexplained * math.log(self.epsilon) if self.epsilon > 0 else -math.inf
            log_post.append(lp)

        return dict(zip(self.opp_ids, _softmax(log_post) or _softmax(self._log_prior)))

    def most_likely(self, observed: Iterable[int]) -> tuple[int, float]:
        """``(archetype_id, probability)`` of the top hypothesis."""
        post = self.posterior(observed)
        aid = max(post, key=lambda k: post[k])
        return aid, post[aid]

    def is_confident(self, observed: Iterable[int], threshold: float = CONFIDENT) -> bool:
        """Whether the posterior has collapsed enough to act on its argmax."""
        return self.most_likely(observed)[1] >= threshold

    def deck_template(
        self, observed: Iterable[int], threshold: float = CONFIDENT
    ) -> list[int] | None:
        """The top archetype's 60-card representative, or ``None`` if unsure.

        ``None`` rather than a best guess: RL_SPEC §8.1 notes that a confidently
        wrong decklist is worse for the determinizer than a vague one, because
        every rollout through an impossible world is wasted work.  The caller
        decides what to do with the uncertainty.
        """
        aid, p = self.most_likely(observed)
        if p < threshold:
            return None
        return list(self._counts[self.opp_ids.index(aid)].elements())


# ── Helpers ─────────────────────────────────────────────────────────────────


def _representative_of(archetypes: dict[str, Any], archetype_id: int) -> list[int]:
    """The 60-card representative decklist for *archetype_id*.

    ``archetypes.json``'s ``archetypes`` container has been both a list and an
    id-keyed object across revisions of the mining code, and JSON object keys
    are strings while archetype ids are ints.  Handle all of it here so a
    lookup miss cannot masquerade as an empty deck.
    """
    container = archetypes.get("archetypes")

    entry: Any = None
    if isinstance(container, dict):
        entry = container.get(str(archetype_id), container.get(archetype_id))
    elif isinstance(container, list):
        for item in container:
            if isinstance(item, dict) and int(item.get("id", -1)) == archetype_id:
                entry = item
                break

    if entry is None:
        return []
    if isinstance(entry, dict):
        return [int(c) for c in entry.get("representative", [])]
    return [int(c) for c in entry]


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


def _softmax(log_weights: Sequence[float]) -> list[float]:
    """Normalise log-weights, or ``[]`` when every hypothesis is impossible."""
    finite = [w for w in log_weights if w != -math.inf and not math.isnan(w)]
    if not finite:
        return []
    hi = max(finite)
    exps = [math.exp(w - hi) if w != -math.inf else 0.0 for w in log_weights]
    total = sum(exps)
    if total <= 0:
        return []
    return [e / total for e in exps]


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
