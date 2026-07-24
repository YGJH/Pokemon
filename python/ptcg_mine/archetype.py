"""Phase 2 deck archetype clustering and 𝒟_self/𝒟_opp/FIXED_DECK selection.

See docs/plans/corpus-mining-plan.md "Global Constraints" (Archetype and
𝒟_self/𝒟_opp bullets) for the binding definitions implemented here.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from ptcg_mine.episode import deck_of, rewards, teams

DEFAULT_JACCARD_THRESH = 0.90


def canon(deck: Iterable[int]) -> tuple[int, ...]:
    """Canonical form of a decklist: sorted tuple of card ids."""
    return tuple(sorted(deck))


def multiset_counts(deck: Iterable[int]) -> Counter:
    """Per-card-id multiplicities of a decklist."""
    return Counter(deck)


def _as_counter(deck) -> Counter:
    return deck if isinstance(deck, Counter) else Counter(deck)


def jaccard_multiset(a: Iterable[int], b: Iterable[int]) -> float:
    """Multiset Jaccard similarity: sum(min) / sum(max) over the union of ids.

    Returns 0.0 for a pair whose union is empty (both decks empty).
    """
    ca, cb = _as_counter(a), _as_counter(b)
    ids = set(ca) | set(cb)
    if not ids:
        return 0.0
    numerator = sum(min(ca[i], cb[i]) for i in ids)
    denominator = sum(max(ca[i], cb[i]) for i in ids)
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass
class Archetype:
    """A cluster of near-identical decklists.

    `representative` is the single most-frequent exact decklist in the
    cluster (canon form); `members` are the canon decklists assigned to it;
    `frequency` is their total exact-deck frequency.
    """

    id: int
    representative: tuple[int, ...]
    members: list[tuple[int, ...]] = field(default_factory=list)
    frequency: int = 0


def cluster_decks(
    deck_freq: dict[tuple[int, ...], int], thresh: float = DEFAULT_JACCARD_THRESH
) -> list[Archetype]:
    """Greedily cluster canon decklists in descending exact-deck frequency.

    A deck joins the best existing cluster (highest jaccard_multiset against
    the cluster's representative) if that score is >= thresh; otherwise it
    starts a new cluster. Because decks are processed in descending
    frequency order, the deck that opens a cluster always has the highest
    frequency of any deck later assigned to it, so it remains the
    representative permanently.
    """
    items = sorted(deck_freq.items(), key=lambda kv: (-kv[1], kv[0]))
    archetypes: list[Archetype] = []
    next_id = 0
    for deck, freq in items:
        deck_c = canon(deck)
        best_arch = None
        best_score = -1.0
        for arch in archetypes:
            score = jaccard_multiset(deck_c, arch.representative)
            if score >= thresh and score > best_score:
                best_arch = arch
                best_score = score
        if best_arch is not None:
            best_arch.members.append(deck_c)
            best_arch.frequency += freq
        else:
            archetypes.append(
                Archetype(id=next_id, representative=deck_c, members=[deck_c], frequency=freq)
            )
            next_id += 1
    return archetypes


def assign_archetype(
    deck: Iterable[int], archetypes: list[Archetype], thresh: float = DEFAULT_JACCARD_THRESH
) -> int | None:
    """Best-matching archetype id for `deck` (jaccard >= thresh vs its
    representative), or None if no archetype qualifies."""
    deck_c = canon(deck)
    best_id = None
    best_score = -1.0
    for arch in archetypes:
        score = jaccard_multiset(deck_c, arch.representative)
        if score >= thresh and score > best_score:
            best_id = arch.id
            best_score = score
    return best_id


def select_self_opp(
    episodes: list[dict],
    experts: list[str],
    archetypes: list[Archetype],
    n_self: int,
    n_opp: int,
) -> tuple[list[int], list[int]]:
    """𝒟_self / 𝒟_opp archetype-id lists.

    𝒟_self = top n_self archetypes by frequency as an expert's own deck
    across all expert games (won and lost).
    𝒟_opp = top n_opp archetypes by frequency as the opponent's deck in
    those same expert games (may overlap 𝒟_self).
    """
    expert_set = set(experts)
    self_counts: Counter = Counter()
    opp_counts: Counter = Counter()

    for ep in episodes:
        t0, t1 = teams(ep)
        for me, p_me, p_opp in ((t0, 0, 1), (t1, 1, 0)):
            if me not in expert_set:
                continue
            self_id = assign_archetype(deck_of(ep, p_me), archetypes)
            if self_id is not None:
                self_counts[self_id] += 1
            opp_id = assign_archetype(deck_of(ep, p_opp), archetypes)
            if opp_id is not None:
                opp_counts[opp_id] += 1

    self_ids = [aid for aid, _ in sorted(self_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    opp_ids = [aid for aid, _ in sorted(opp_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return self_ids[:n_self], opp_ids[:n_opp]


def pick_fixed_deck(
    episodes: list[dict],
    experts: list[str],
    self_ids: list[int],
    archetypes: list[Archetype],
) -> list[int]:
    """Representative decklist (60 ids) of the 𝒟_self archetype with the
    highest expert win-rate (wins/games as an expert's own deck)."""
    expert_set = set(experts)
    self_id_set = set(self_ids)
    arch_by_id = {arch.id: arch for arch in archetypes}

    wins: Counter = Counter()
    games: Counter = Counter()
    for ep in episodes:
        t0, t1 = teams(ep)
        r0, r1 = rewards(ep)
        for me, r, p_me in ((t0, r0, 0), (t1, r1, 1)):
            if me not in expert_set:
                continue
            aid = assign_archetype(deck_of(ep, p_me), archetypes)
            if aid not in self_id_set:
                continue
            games[aid] += 1
            if r == 1:
                wins[aid] += 1

    best_id = None
    best_rate = -1.0
    for aid in self_ids:
        rate = wins[aid] / games[aid] if games[aid] > 0 else -1.0
        if rate > best_rate:
            best_rate = rate
            best_id = aid

    if best_id is None:
        raise ValueError("pick_fixed_deck: no self_id archetype has any expert games")
    return list(arch_by_id[best_id].representative)
