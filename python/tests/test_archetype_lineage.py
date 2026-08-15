"""Append-only archetype ids and 𝒟_opp belief slots.

Cluster ids are positions in the greedy open order, which follows the global
deck-frequency ordering, so an unseeded re-mine renumbers clusters whose
membership never changed.  Everything downstream keys off those ids —
``--archetype-self N``, the belief head's class index, every checkpoint's deck
record — and every one of those failures is silent: the model loads, plays, and
reports a win rate for the wrong deck.

Two separate properties are needed for an old checkpoint to stay valid, and
having only the first is the trap:

1. **ids are append-only** — cluster 17 is the same decklist forever;
2. **the 𝒟_opp list is append-only** — slot *i* is the same archetype forever.

``shard_writer`` builds the belief label as ``{gid: i for i, gid in
enumerate(opp_ids)}``, so a re-sorted ``opp_ids`` of unchanged length keeps the
head the same width and silently repoints every class.  These tests pin both,
and each guard is checked by mutation: shuffle the baseline, drop it, shrink it.
"""

from __future__ import annotations

import json

import pytest

from ptcg_mine.archetype import (
    Archetype,
    _append_only,
    canon,
    cluster_decks,
    load_archetypes_json,
)
from tests.test_model_policy import make_policy

# Four mutually dissimilar 60-card decks (multiset Jaccard 0 pairwise).
DECK_A = tuple([1] * 30 + [2] * 30)
DECK_B = tuple([3] * 30 + [4] * 30)
DECK_C = tuple([5] * 30 + [6] * 30)
DECK_D = tuple([7] * 30 + [8] * 30)
# A near-copy of A: 59/60 shared → multiset Jaccard ≈ 0.967, above the 0.90 bar.
DECK_A2 = tuple(sorted([1] * 29 + [2] * 30 + [9]))


def _ids_by_deck(archetypes: list[Archetype]) -> dict[tuple, int]:
    return {a.representative: a.id for a in archetypes}


# ── The failure this exists to prevent ──────────────────────────────────────


def test_unseeded_reclustering_renumbers_an_unchanged_deck():
    """Establishes the bug is real before testing the fix.

    Same decks, only their frequencies changed — and the ids move.
    """
    gen1 = cluster_decks({DECK_A: 100, DECK_B: 50, DECK_C: 10})
    gen2 = cluster_decks({DECK_A: 100, DECK_B: 50, DECK_C: 500})

    ids1, ids2 = _ids_by_deck(gen1), _ids_by_deck(gen2)
    assert ids1.keys() == ids2.keys(), "fixture changed the deck set"
    moved = [d for d in ids1 if ids1[d] != ids2[d]]
    assert moved, (
        "no id moved between generations — the fixture no longer reproduces "
        "the renumbering these tests are about, so everything below is vacuous"
    )


# ── Append-only ids ─────────────────────────────────────────────────────────


def test_seeding_keeps_every_baseline_id_on_its_own_deck():
    gen1 = cluster_decks({DECK_A: 100, DECK_B: 50, DECK_C: 10})
    gen2 = cluster_decks({DECK_A: 100, DECK_B: 50, DECK_C: 500}, baseline=gen1)

    ids1, ids2 = _ids_by_deck(gen1), _ids_by_deck(gen2)
    checked = 0
    for deck, old_id in ids1.items():
        assert ids2[deck] == old_id, (
            f"deck {deck[:3]}... was id {old_id} and is now {ids2[deck]}"
        )
        checked += 1
    assert checked == 3, f"expected 3 baseline decks, checked {checked}"


def test_a_new_deck_gets_a_fresh_id_past_the_baseline_maximum():
    gen1 = cluster_decks({DECK_A: 100, DECK_B: 50})
    gen2 = cluster_decks({DECK_A: 100, DECK_B: 50, DECK_D: 900}, baseline=gen1)

    assert len(gen2) == 3
    new = _ids_by_deck(gen2)[DECK_D]
    assert new == max(a.id for a in gen1) + 1, (
        f"new cluster took id {new}; ids must continue past the baseline "
        "maximum or a later generation can reissue one"
    )


def test_a_baseline_cluster_absent_from_the_new_corpus_is_retained_at_zero():
    """Dropping it would free its id for an unrelated deck later."""
    gen1 = cluster_decks({DECK_A: 100, DECK_B: 50, DECK_C: 10})
    gen2 = cluster_decks({DECK_A: 100}, baseline=gen1)

    assert len(gen2) == 3, "a baseline cluster was dropped"
    by_id = {a.id: a for a in gen2}
    gone = _ids_by_deck(gen1)[DECK_C]
    assert by_id[gone].representative == canon(DECK_C)
    assert by_id[gone].frequency == 0, "frequency must reflect *this* corpus"


def test_the_representative_is_frozen_even_when_a_member_overtakes_it():
    """The stated cost of stable ids, pinned so it cannot regress silently.

    DECK_A2 is within the Jaccard threshold of DECK_A, so it joins A's cluster
    rather than opening its own — even though it is now far more frequent.
    """
    gen1 = cluster_decks({DECK_A: 100, DECK_B: 50})
    gen2 = cluster_decks({DECK_A: 1, DECK_A2: 5000, DECK_B: 50}, baseline=gen1)

    by_id = {a.id: a for a in gen2}
    a_id = _ids_by_deck(gen1)[DECK_A]
    assert by_id[a_id].representative == canon(DECK_A), (
        "the representative moved to the newly-frequent member — ids would be "
        "stable but the decklist they name would not, which is the same "
        "silent repointing wearing different clothes"
    )
    assert canon(DECK_A2) in by_id[a_id].members, "DECK_A2 did not join A"


def test_seeding_also_lifts_the_processing_order_constraint():
    """Seeding is **not** a no-op on an unseeded file, and the stamp design
    depends on knowing that.

    Unseeded, a deck can only join a cluster that a *higher-frequency* deck has
    already opened.  Seeded, every baseline representative exists before the
    first deck is placed, so a deck reaches its true best match even when that
    cluster's representative is rarer than itself.  Measured over the real
    11.9k-episode corpus this moved ~102 decklists across 7 of 201 clusters,
    with ids and representatives untouched.

    ``MID`` matches both ``COMMON`` (0.905) and ``RARE`` (0.967), and prefers
    ``RARE``.  Unseeded, ``RARE`` is rarer than ``MID`` and so has not opened a
    cluster yet when ``MID`` is placed — it lands on ``COMMON`` instead.  Seeded
    from a generation where ``RARE`` was its own cluster, it reaches the better
    match.
    """
    # Multiset Jaccard is sum(min)/sum(max) over the *union* of card ids, so a
    # swapped card costs in both terms: 57/63 = 0.905, not 57/60.
    common = tuple([1] * 60)
    mid = canon([1] * 57 + [2] * 3)          # J vs common = 57/63 = 0.905
    rare = canon([1] * 58 + [2] * 2)         # J vs mid    = 59/61 = 0.967

    unseeded = cluster_decks({common: 100, mid: 50})
    assert len(unseeded) == 1, "fixture must collapse to one cluster unseeded"
    assert mid in unseeded[0].members and unseeded[0].representative == common

    baseline = [Archetype(id=0, representative=common),
                Archetype(id=1, representative=rare)]
    seeded = cluster_decks({common: 100, mid: 50}, baseline=baseline)
    by_id = {a.id: a for a in seeded}
    assert mid in by_id[1].members, (
        "mid stayed on the cluster the processing order forced it onto; "
        "seeding is not letting it reach the better-matching baseline cluster"
    )
    assert mid not in by_id[0].members
    assert by_id[0].representative == common and by_id[1].representative == rare, (
        "a representative moved — ids would be stable but the decklists they "
        "name would not"
    )


def test_a_seeded_run_reproduces_its_own_output_exactly():
    """The fixpoint the ``"auto"`` stamp key relies on.

    Once every representative is frozen, assignment no longer depends on
    processing order at all, so seeding a seeded generation over the same
    corpus is exact — including the frequencies that shifted on the first pass.
    """
    common = tuple([1] * 60)
    mid = canon([1] * 57 + [2] * 3)
    rare = canon([1] * 58 + [2] * 2)
    freq = {common: 100, mid: 50, DECK_B: 30, DECK_C: 7}

    baseline = [Archetype(id=0, representative=common),
                Archetype(id=1, representative=rare)]
    gen2 = cluster_decks(freq, baseline=baseline)
    gen3 = cluster_decks(freq, baseline=gen2)
    assert [(a.id, a.representative, a.frequency, sorted(a.members)) for a in gen2] == \
           [(a.id, a.representative, a.frequency, sorted(a.members)) for a in gen3], (
        "a seeded run did not reproduce itself; the stage stamp would cache a "
        "generation that differs from what re-running would produce"
    )


# ── Append-only 𝒟_opp slots ─────────────────────────────────────────────────


def test_append_only_preserves_baseline_order_not_just_membership():
    """A re-sort keeps the width and repoints every belief class."""
    baseline = [7, 3, 11]
    ranked = [11, 7, 3]          # same three, this corpus ranks them differently
    assert _append_only(baseline, ranked, 3) == [7, 3, 11], (
        "opp_ids was re-sorted by the new corpus's frequencies; the belief "
        "head keeps its width so nothing raises, and slot 0 now means a "
        "different archetype than the checkpoint was trained on"
    )


def test_append_only_appends_newcomers_after_the_baseline():
    assert _append_only([7, 3], [3, 9, 7, 4], 3) == [7, 3, 9]


def test_append_only_never_drops_a_baseline_slot():
    """Even one that has vanished from this corpus's ranking entirely."""
    out = _append_only([7, 3, 11], [4], 2)
    assert out[:3] == [7, 3, 11]
    assert 4 in out


def test_without_a_baseline_it_is_the_old_top_n():
    assert _append_only(None, [5, 2, 8, 1], 2) == [5, 2]
    assert _append_only([], [5, 2, 8, 1], 3) == [5, 2, 8]


def test_the_slot_map_shard_writer_builds_is_stable_under_append():
    """Mirrors ``shard_writer``'s ``{gid: i for i, gid in enumerate(opp_ids)}``
    so this test fails if that construction ever diverges from the assumption."""
    base = [7, 3, 11]
    grown = _append_only(base, [9, 7, 3], 3)
    old_map = {gid: i for i, gid in enumerate(base)}
    new_map = {gid: i for i, gid in enumerate(grown)}
    for gid, slot in old_map.items():
        assert new_map[gid] == slot, f"archetype {gid} moved slot {slot} → {new_map[gid]}"


# ── The artifact round-trip ─────────────────────────────────────────────────


def _write_artifact(path, archetypes, self_ids, opp_ids):
    from ptcg_mine.artifacts import write_archetypes_json

    write_archetypes_json(path, self_ids, opp_ids, archetypes,
                          list(archetypes[0].representative))


def test_a_written_artifact_seeds_the_next_generation(tmp_path):
    gen1 = cluster_decks({DECK_A: 100, DECK_B: 50})
    path = tmp_path / "archetypes.json"
    _write_artifact(path, gen1, [0], [0, 1])

    loaded, self_ids, opp_ids = load_archetypes_json(path)
    assert [(a.id, a.representative) for a in loaded] == \
           [(a.id, a.representative) for a in gen1]
    assert self_ids == [0] and opp_ids == [0, 1]

    gen2 = cluster_decks({DECK_A: 1, DECK_D: 900}, baseline=loaded)
    assert _ids_by_deck(gen2)[DECK_A] == _ids_by_deck(gen1)[DECK_A]


def test_lineage_is_recorded_so_a_generation_is_identifiable(tmp_path):
    from ptcg_mine.artifacts import write_archetypes_json

    gen1 = cluster_decks({DECK_A: 100})
    path = tmp_path / "a.json"
    write_archetypes_json(path, [0], [0], gen1, list(DECK_A),
                          lineage={"seeded": True, "baseline_sha1": "abc123",
                                   "generation": 2})
    doc = json.loads(path.read_text())
    assert doc["lineage"]["seeded"] is True
    assert doc["lineage"]["generation"] == 2


def test_an_unlabelled_artifact_still_loads_as_generation_zero(tmp_path):
    """Artifacts written before lineage existed must remain readable."""
    from ptcg_mine.artifacts import write_archetypes_json

    gen1 = cluster_decks({DECK_A: 100})
    path = tmp_path / "a.json"
    write_archetypes_json(path, [0], [0], gen1, list(DECK_A))
    doc = json.loads(path.read_text())
    assert doc["lineage"] == {"seeded": False, "baseline_sha1": None,
                              "generation": 0}


def test_a_malformed_entry_raises_rather_than_losing_its_id(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({
        "self_ids": [], "opp_ids": [],
        "archetypes": [{"id": 0, "representative": list(DECK_A)},
                       {"id": 1}],  # no representative
    }))
    with pytest.raises(ValueError, match="representative"):
        load_archetypes_json(path)


def test_duplicate_ids_raise(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({
        "archetypes": [{"id": 0, "representative": list(DECK_A)},
                       {"id": 0, "representative": list(DECK_B)}],
    }))
    with pytest.raises(ValueError, match="duplicate"):
        load_archetypes_json(path)


def test_an_empty_artifact_holds_no_ids_and_is_not_an_error(tmp_path):
    """Nothing to preserve means starting fresh destroys nothing."""
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"self_ids": [0]}))
    archetypes, _s, _o = load_archetypes_json(path)
    assert archetypes == []


# ── Baseline resolution ─────────────────────────────────────────────────────


def _config(tmp_path, **kw):
    from ptcg_mine.config import MineConfig

    return MineConfig(raw_dir=tmp_path / "raw", out_dir=tmp_path / "data", **kw)


def test_the_out_dir_artifact_is_the_implicit_baseline(tmp_path):
    """The case that matters: re-mining in place must not need a flag."""
    from ptcg_mine.mine import resolve_baseline

    data = tmp_path / "data"
    data.mkdir(parents=True)
    gen1 = cluster_decks({DECK_A: 100, DECK_B: 50})
    _write_artifact(data / "archetypes.json", gen1, [0], [0, 1])

    baseline, self_ids, opp_ids, path = resolve_baseline(_config(tmp_path))
    assert baseline is not None and len(baseline) == 2
    assert opp_ids == [0, 1]
    assert path == data / "archetypes.json"


def test_rebaseline_beats_the_implicit_and_the_explicit_baseline(tmp_path):
    from ptcg_mine.mine import resolve_baseline

    data = tmp_path / "data"
    data.mkdir(parents=True)
    gen1 = cluster_decks({DECK_A: 100})
    _write_artifact(data / "archetypes.json", gen1, [0], [0])

    baseline, _s, _o, path = resolve_baseline(_config(tmp_path, rebaseline=True))
    assert baseline is None and path is None


def test_a_named_baseline_that_is_missing_raises(tmp_path):
    """Silently falling through to fresh ids is the one unacceptable answer."""
    from ptcg_mine.mine import resolve_baseline

    cfg = _config(tmp_path, baseline_archetypes=tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="rebaseline"):
        resolve_baseline(cfg)


def test_a_fresh_out_dir_has_no_baseline(tmp_path):
    from ptcg_mine.mine import resolve_baseline

    baseline, _s, _o, path = resolve_baseline(_config(tmp_path))
    assert baseline is None and path is None


# ── The stage stamp must track the id generation ────────────────────────────


def test_rebaseline_invalidates_the_stamp_but_the_default_does_not(tmp_path):
    """The default baseline is the run's own previous output, so hashing it
    would make every fresh pipeline pay a second Phase 2 for nothing."""
    from ptcg_mine import stamp

    plain = stamp.params_from_config("mine", _config(tmp_path))
    rebase = stamp.params_from_config("mine", _config(tmp_path, rebaseline=True))
    assert plain["baseline"] == "auto"
    assert rebase["baseline"] == "rebaseline"
    assert plain != rebase, "--rebaseline must force a recompute"

    # And it stays "auto" once the directory has artifacts in it.
    data = tmp_path / "data"
    data.mkdir(parents=True)
    _write_artifact(data / "archetypes.json", cluster_decks({DECK_A: 1}), [0], [0])
    assert stamp.params_from_config("mine", _config(tmp_path))["baseline"] == "auto"


def test_an_explicit_baseline_is_fingerprinted_by_content(tmp_path):
    """Two different generations behind the same flag must not share a stamp."""
    from ptcg_mine import stamp

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_artifact(a, cluster_decks({DECK_A: 100}), [0], [0])
    _write_artifact(b, cluster_decks({DECK_B: 100}), [0], [0])

    pa = stamp.params_from_config("mine", _config(tmp_path, baseline_archetypes=a))
    pb = stamp.params_from_config("mine", _config(tmp_path, baseline_archetypes=b))
    assert pa["baseline"].startswith("file:")
    assert pa["baseline"] != pb["baseline"]

