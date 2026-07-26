"""Turn the belief heads' output into a concrete 60-card opponent decklist.

``ptcg_search``'s determinizer (``guessing.rs``) needs one plausible
``opponent_deck_template`` in **engine card ids**: it subtracts the cards it can
already see and deals the remainder into the opponent's deck, hand and prizes.
Until now ``live_eval`` passed an empty list, and the Rust side fell back to
assuming the opponent runs a mirror of our own deck -- which is wrong in every
game where they do not.

Two ways to produce the template, in order of preference:

1. **Archetype lookup.**  The ``arch`` head classifies over the retained 𝒟_opp
   set, and every archetype in ``archetypes.json`` carries a real 60-card
   ``representative``.  When the head is confident this is strictly better than
   anything assembled card-by-card, because a representative is a deck someone
   actually played: its energy counts, its ratios and its evolution lines are
   all internally consistent.
2. **Card distribution.**  When the head is unsure -- an opponent outside the
   retained set, or early in the game before much is visible -- fall back to
   the ``deck`` distribution and round it into 60 cards.  The result is a
   plausible *bag* of cards rather than a coherent list, which is still a large
   improvement on the mirror assumption.

Falling back at low confidence rather than always taking the argmax matters
because a confidently wrong decklist is worse for the search than a vague one:
the determinizer will happily deal the opponent four copies of a card they
cannot possibly hold, and every rollout through that world is wasted.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

DECK_SIZE = 60

# Below this probability on the top archetype, prefer the card distribution.
# 1/6 is chance on the current 𝒟_opp set, so 0.5 is "clearly better than
# guessing" without demanding certainty the head will not have on turn one.
ARCH_CONFIDENCE = 0.5

logger = logging.getLogger(__name__)


def _as_card_id(value: Any) -> int:
    """Coerce a vocab entry to an engine card id, or -1 if it is not one.

    ``vocab.json`` stores the two reserved slots as the literal strings
    ``"PAD"`` and ``"UNKNOWN"``, so a blanket ``int()`` over ``index_to_id``
    raises ``ValueError`` on a perfectly valid artifact.  Anything unparseable
    becomes -1, which :func:`deck_from_distribution` drops.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def deck_from_distribution(
    probs: np.ndarray,
    index_to_id: list[int],
    deck_size: int = DECK_SIZE,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Round a per-card probability vector into a ``deck_size`` multiset of ids.

    With *rng* the counts are a multinomial draw (use this when the caller
    wants several distinct determinizations); without it they are the largest
    remainder apportionment of ``deck_size * p``, which is deterministic and
    reproduces the distribution's mode rather than a sample from it.

    PAD (index 0) and UNKNOWN (index 1) are dropped before rounding: neither
    names a card the engine can deal.
    """
    probs = np.asarray(probs, dtype=np.float64).copy()
    probs[:2] = 0.0  # PAD, UNKNOWN
    total = probs.sum()
    if total <= 0:
        return []
    probs /= total

    if rng is not None:
        counts = rng.multinomial(deck_size, probs)
    else:
        exact = probs * deck_size
        counts = np.floor(exact).astype(np.int64)
        short = deck_size - int(counts.sum())
        if short > 0:
            # Largest remainder: hand the leftover slots to the cards the
            # rounding shortchanged most.
            order = np.argsort(-(exact - counts))
            counts[order[:short]] += 1

    deck: list[int] = []
    for idx in np.nonzero(counts)[0]:
        if idx >= len(index_to_id):
            continue
        card_id = int(index_to_id[idx])
        if card_id < 0:  # reserved slot, see _as_card_id
            continue
        deck.extend([card_id] * int(counts[idx]))
    return deck


class OpponentDeckOracle:
    """Predicts the opponent's decklist from a trained policy's belief heads.

    Parameters
    ----------
    policy : Policy
        A policy trained with ``--belief``.  Untrained heads still produce a
        deck, just an uninformative one -- see :meth:`predict`.
    vocab : dict
        Normalized vocab (``ptcg_il.featurizer.normalize_vocab``).
    archetypes : dict
        Parsed ``archetypes.json``; supplies ``opp_ids`` and the per-archetype
        ``representative`` decklists.
    arch_confidence : float
        Minimum probability on the top archetype before its representative is
        used instead of the card distribution.
    """

    def __init__(
        self,
        policy: Any,
        vocab: dict,
        archetypes: dict,
        device: Any = None,
        arch_confidence: float = ARCH_CONFIDENCE,
    ):
        self.policy = policy
        self.vocab = vocab
        self.device = device or str(next(policy.parameters()).device)
        self.arch_confidence = arch_confidence

        # index -> engine card id.  Derived from id_to_index when the artifact
        # does not carry the reverse map, because getting this wrong produces a
        # 60-card deck of the wrong cards rather than any kind of error.
        index_to_id = vocab.get("index_to_id")
        if not index_to_id:
            size = int(vocab.get("size") or 0)
            index_to_id = [-1] * size
            for card_id, idx in vocab.get("id_to_index", {}).items():
                if 0 <= int(idx) < size:
                    index_to_id[int(idx)] = _as_card_id(card_id)
        # vocab.json spells the reserved slots as the strings "PAD"/"UNKNOWN",
        # so a blanket int() raises.  They map to -1 and are skipped when the
        # deck is assembled.
        self.index_to_id = [_as_card_id(c) for c in index_to_id]

        # opp_ids order *is* the head's class order -- shard_writer assigned
        # bel_arch by position in this list, so any reordering here would
        # silently pair each class with the wrong decklist.
        by_id = {int(a["id"]): a for a in archetypes.get("archetypes", [])}
        self.representatives: list[list[int]] = []
        for gid in archetypes.get("opp_ids", []):
            rep = by_id.get(int(gid), {}).get("representative") or []
            self.representatives.append([int(c) for c in rep])

    def belief(self, obs_dict: dict) -> dict[str, np.ndarray]:
        """Raw belief logits for one observation, as numpy arrays."""
        import torch

        from ptcg_il.featurizer import featurize

        feats = featurize(obs_dict, self.vocab)
        # Same dtype coercion as live_eval._sample_to_batch: the embedding
        # tables index with int64 and the masks must stay bool, so letting
        # numpy's dtype through unchanged is not enough.
        batch = {}
        for k, v in feats.items():
            if not isinstance(v, np.ndarray):
                continue
            t = torch.from_numpy(v).unsqueeze(0)
            if v.dtype == np.int64:
                t = t.long()
            elif v.dtype == np.bool_:
                t = t.bool()
            else:
                t = t.float()
            batch[k] = t.to(self.device)
        with torch.no_grad():
            out = self.policy.belief_logits(batch)
        return {k: v[0].float().cpu().numpy() for k, v in out.items()}

    def predict(
        self, obs_dict: dict, rng: np.random.Generator | None = None
    ) -> list[int]:
        """A 60-card opponent decklist in engine card ids (empty on failure).

        An empty list is a valid answer and means "no opinion" -- the Rust
        determinizer then falls back to the mirror heuristic, which is exactly
        the behaviour we want when the belief heads have nothing to say.
        """
        try:
            logits = self.belief(obs_dict)
        except Exception as exc:  # noqa: BLE001 - never break the planner
            logger.debug("Belief forward failed: %s", exc)
            return []

        arch = logits.get("arch")
        if arch is not None and arch.size > 1 and self.representatives:
            p = _softmax(arch[: len(self.representatives)])
            best = int(p.argmax())
            rep = self.representatives[best]
            if p[best] >= self.arch_confidence and len(rep) == DECK_SIZE:
                return list(rep)

        return deck_from_distribution(
            _softmax(logits["deck"]), self.index_to_id, rng=rng
        )
