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


def load_archetypes_json(path) -> tuple[list[Archetype], list[int], list[int]]:
    """Read a previous generation's ``archetypes.json`` back into objects.

    Returns ``(archetypes, self_ids, opp_ids)``.  ``members`` is not stored in
    the artifact and comes back empty; only ``id`` and ``representative`` are
    needed to seed :func:`cluster_decks`, and only the id *lists* are needed to
    keep the belief slots pinned.

    A file with **no** archetypes returns an empty list rather than raising —
    it holds no ids, so there is nothing a fresh run could destroy, and the
    caller decides whether that is acceptable.  A file whose entries are
    *malformed* does raise: a baseline that quietly lost half its clusters
    would leave their ids free to be handed to unrelated decks, which is the
    precise failure this whole path exists to prevent.
    """
    import json
    from pathlib import Path

    path = Path(path)
    with open(path) as f:
        doc = json.load(f)
    entries = doc.get("archetypes") or []
    archetypes = []
    for d in entries:
        rep = d.get("representative")
        if d.get("id") is None or not rep:
            raise ValueError(
                f"{path} has an archetype entry with no id or no "
                f"representative: {d!r}. Seeding from it would leave a hole in "
                "the id space for a later run to reuse."
            )
        archetypes.append(Archetype(id=int(d["id"]), representative=canon(rep),
                                    members=[], frequency=int(d.get("frequency", 0))))
    ids = [a.id for a in archetypes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} has duplicate archetype ids")
    return (archetypes,
            [int(i) for i in doc.get("self_ids", [])],
            [int(i) for i in doc.get("opp_ids", [])])


def cluster_decks(
    deck_freq: dict[tuple[int, ...], int],
    thresh: float = DEFAULT_JACCARD_THRESH,
    baseline: list[Archetype] | None = None,
) -> list[Archetype]:
    """Greedily cluster canon decklists in descending exact-deck frequency.

    A deck joins the best existing cluster (highest jaccard_multiset against
    the cluster's representative) if that score is >= thresh; otherwise it
    starts a new cluster.

    Without *baseline*, ids are assigned in the order clusters are opened, and
    because decks are processed in descending frequency order, the deck that
    opens a cluster has the highest frequency of any deck later assigned to it
    and stays the representative for that run.

    **Ids from an unseeded run are not stable across corpora.**  They are
    positions in the open order, which follows the global frequency ordering,
    so adding episodes renumbers clusters whose membership did not change at
    all.  Everything downstream keys off those ids — ``--archetype-self N``,
    the ``𝒟_opp`` belief slots, every checkpoint's deck record — so the
    renumbering silently repoints a trained model at somebody else's deck.

    Pass *baseline* (a previous run's archetypes, e.g. from
    :func:`load_archetypes_json`) to make ids **append-only**: every baseline
    cluster keeps its id and its representative, and only decks that match no
    baseline representative open new clusters, numbered from
    ``max(baseline id) + 1``.  A baseline cluster with no members in this
    corpus is retained at frequency 0 rather than dropped — reusing its id
    later would be exactly the renumbering this exists to prevent.

    The cost, stated rather than buried: a seeded cluster's representative is
    **frozen**.  A newly-arrived decklist more frequent than the one that
    originally opened the cluster does not replace it, so representatives (and
    therefore the decklists ``ptcg_il.deck`` stamps into checkpoints) stop
    tracking the meta.  That is the price of an id meaning one thing forever;
    refreshing them is a deliberate re-baseline, not a side effect of mining.
    """
    items = sorted(deck_freq.items(), key=lambda kv: (-kv[1], kv[0]))
    archetypes: list[Archetype] = []
    next_id = 0
    if baseline:
        # Members and frequency are re-accumulated from *this* corpus; id and
        # representative are the parts that must not move.
        archetypes = [
            Archetype(id=a.id, representative=canon(a.representative),
                      members=[], frequency=0)
            for a in baseline
        ]
        next_id = max(a.id for a in archetypes) + 1
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


def _append_only(baseline: list[int] | None, ranked: list[int], n: int) -> list[int]:
    """``baseline`` in its original order, then this corpus's top-*n* newcomers.

    Position in these lists is not cosmetic.  ``shard_writer`` builds the belief
    head's class index as ``{gid: i for i, gid in enumerate(opp_ids)}``, so slot
    *i* means "whatever archetype sits at position *i*".  Re-sorting by this
    corpus's frequencies would keep the list six long and the checkpoint would
    load without complaint, while slot 3 quietly came to mean a different deck.

    Consequence to be aware of: the list only ever grows.  ``n_opp`` stops being
    the width of the belief head and becomes "how many of this corpus's top
    archetypes may be *added*".  Re-baselining is the only way to shrink it.
    """
    if not baseline:
        return ranked[:n]
    seen = set(baseline)
    return list(baseline) + [aid for aid in ranked[:n] if aid not in seen]


def select_self_opp(
    episodes: list[dict],
    experts: list[str],
    archetypes: list[Archetype],
    n_self: int,
    n_opp: int,
    baseline_self_ids: list[int] | None = None,
    baseline_opp_ids: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """𝒟_self / 𝒟_opp archetype-id lists.

    𝒟_self = top n_self archetypes by frequency as an expert's own deck
    across all expert games (won and lost).
    𝒟_opp = top n_opp archetypes by frequency as the opponent's deck in
    those same expert games (may overlap 𝒟_self).

    With a baseline, both lists are **append-only** — see :func:`_append_only`
    for why the order, not just the membership, is load-bearing.  Stable
    archetype ids alone do not make an old checkpoint valid against a newer
    corpus: the belief head indexes by *position in these lists*, so a
    re-sorted ``opp_ids`` invalidates it just as thoroughly as renumbering.
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

    self_ranked = [aid for aid, _ in sorted(self_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    opp_ranked = [aid for aid, _ in sorted(opp_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return (_append_only(baseline_self_ids, self_ranked, n_self),
            _append_only(baseline_opp_ids, opp_ranked, n_opp))


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
