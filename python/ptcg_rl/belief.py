"""Categorical posterior over the opponent's 𝒟_opp archetype (RL_SPEC §8.1).

The opponent's deck is one of a handful of known archetypes, so belief over it is
a small categorical, not a learned distribution:

```
P(deck_j | observed) ∝ P(deck_j) · Π_{c observed} P(c | deck_j)
```

The mechanism that makes this sharp is **elimination**, not the likelihood
ratios: any card observed that archetype *j* does not contain zeros hypothesis
*j* outright, so the posterior collapses to near-certainty within a few turns.
It needs no training, costs nothing to evaluate, and — unlike the learned heads
in ``ptcg_il/model/belief.py`` — cannot be wrong about a deck it has seen the
cards of.

This is a *different object* from ``ptcg_il.model.belief``, which is a learned
history encoder feeding the policy trunk. Keep both; do not conflate them.
:class:`~ptcg_il.belief_infer.OpponentDeckOracle` consumes this one as the middle
rung of its fallback chain: learned archetype head when confident, this posterior
when it has collapsed, learned card distribution otherwise.

Counting is **multiset-aware**.  Seeing a third copy of a card an archetype runs
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
