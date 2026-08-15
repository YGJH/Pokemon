"""Tests for ptcg_mine.deck_meta_trend: the per-day archetype share series.

The numbers this module produces are read as "is this deck getting stronger",
so the two things that must not slip are the *denominator* (a share measured
against only the decks we could cluster is not a share of the meta) and the
day key (a mislabelled day shifts every trend by one).
"""

import json

import pandas as pd
import pytest

from ptcg_mine.archetype import Archetype, canon
from ptcg_mine import deck_meta_trend as dmt


def _deck(card, n=60):
    """A 60-card decklist made of a single card id — trivially self-similar."""
    return [card] * n


def _write_episode(day_dir, name, deck0, deck1, *, statuses=("DONE", "DONE")):
    ep = {
        "info": {"TeamNames": ["alpha", "beta"]},
        "statuses": list(statuses),
        "rewards": [1, -1],
        "steps": [
            [{"action": None}, {"action": None}],
            [{"action": list(deck0)}, {"action": list(deck1)}],
        ],
    }
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / name).write_text(json.dumps(ep))


@pytest.fixture
def archetypes():
    return [
        Archetype(id=0, representative=canon(_deck(100))),
        Archetype(id=1, representative=canon(_deck(200))),
        Archetype(id=2, representative=canon(_deck(300))),
    ]


# ============================================================
# Scanning
# ============================================================

def test_day_key_comes_from_the_directory_name(tmp_path, archetypes):
    """raw/<day>/ is the only record of when an episode was played."""
    _write_episode(tmp_path / "2026-07-04", "a.json", _deck(100), _deck(200))
    _write_episode(tmp_path / "2026-07-05", "b.json", _deck(100), _deck(100))

    scan = dmt.scan_day_slots(tmp_path, archetypes, jobs=1)

    assert sorted(scan.slots) == ["2026-07-04", "2026-07-05"]
    assert scan.slots["2026-07-04"] == {0: 1, 1: 1}
    assert scan.slots["2026-07-05"] == {0: 2}


def test_unclustered_decks_are_counted_as_other(tmp_path, archetypes):
    """A deck matching no archetype is still a game that was played.

    Dropping it would inflate every clustered deck's share by the unclustered
    fraction, which is exactly the quantity the chart is trying to measure.
    """
    _write_episode(tmp_path / "2026-07-04", "a.json", _deck(100), _deck(999))

    scan = dmt.scan_day_slots(tmp_path, archetypes, jobs=1)

    assert scan.slots["2026-07-04"] == {0: 1, dmt.OTHER_ID: 1}


def test_invalid_episodes_are_excluded_from_the_denominator(tmp_path, archetypes):
    """An episode that never finished is not evidence about the meta."""
    _write_episode(tmp_path / "2026-07-04", "ok.json", _deck(100), _deck(100))
    _write_episode(tmp_path / "2026-07-04", "bad.json", _deck(200), _deck(200),
                   statuses=("DONE", "ERROR"))

    scan = dmt.scan_day_slots(tmp_path, archetypes, jobs=1)

    assert scan.slots["2026-07-04"] == {0: 2}
    assert scan.episodes["2026-07-04"] == 1
    assert scan.skipped["2026-07-04"] == 1


def test_parallel_scan_matches_serial(tmp_path, archetypes):
    """The pool must ship its context, not inherit it from a forked parent.

    Under `fork` a module global set by an earlier serial run is visible in
    every worker, so a missing `initializer=` stays hidden until the pool runs
    first.  Comparing the two runs pins it.
    """
    cards = [100, 200, 300, 999]
    for i in range(400):
        day = f"2026-07-{4 + i % 3:02d}"
        _write_episode(tmp_path / day, f"e{i}.json",
                       _deck(cards[i % 4]), _deck(cards[(i + 1) % 4]))

    serial = dmt.scan_day_slots(tmp_path, archetypes, jobs=1)
    parallel = dmt.scan_day_slots(tmp_path, archetypes, jobs=4)

    assert parallel.slots == serial.slots
    assert parallel.episodes == serial.episodes
    assert sum(sum(c.values()) for c in serial.slots.values()) == 800


# ============================================================
# Shares
# ============================================================

def test_share_is_measured_against_every_slot_including_other(tmp_path, archetypes):
    _write_episode(tmp_path / "2026-07-04", "a.json", _deck(100), _deck(100))
    _write_episode(tmp_path / "2026-07-04", "b.json", _deck(999), _deck(999))

    df = dmt.daily_shares(dmt.scan_day_slots(tmp_path, archetypes, jobs=1))

    row = df.set_index(["day", "archetype_id"])
    assert row.loc[("2026-07-04", 0), "share"] == pytest.approx(0.5)
    assert row.loc[("2026-07-04", dmt.OTHER_ID), "share"] == pytest.approx(0.5)
    assert row.loc[("2026-07-04", 0), "day_total_slots"] == 4


def test_a_deck_absent_on_a_day_gets_an_explicit_zero(tmp_path, archetypes):
    """Gaps and zeroes plot differently; an absent deck really was at 0%."""
    _write_episode(tmp_path / "2026-07-04", "a.json", _deck(100), _deck(100))
    _write_episode(tmp_path / "2026-07-05", "b.json", _deck(200), _deck(200))

    df = dmt.daily_shares(dmt.scan_day_slots(tmp_path, archetypes, jobs=1))
    row = df.set_index(["day", "archetype_id"])

    assert row.loc[("2026-07-05", 0), "slots"] == 0
    assert row.loc[("2026-07-05", 0), "share"] == pytest.approx(0.0)


# ============================================================
# Series selection
# ============================================================

def _shares(rows):
    """rows: (day, archetype_id, slots) -> the DataFrame `union_top_n` reads."""
    df = pd.DataFrame(rows, columns=["day", "archetype_id", "slots"])
    totals = df.groupby("day")["slots"].transform("sum")
    df["day_total_slots"] = totals
    df["share"] = df["slots"] / totals
    return df


def test_union_top_n_keeps_a_deck_that_peaks_on_one_day_only():
    """The whole point of the union: catch a deck that surges mid-corpus.

    Deck 2 is nowhere near the overall top 2, but owns day two — which is the
    "this deck just became strong" signal the chart exists to show.
    """
    df = _shares([
        ("d1", 0, 100), ("d1", 1, 50), ("d1", 2, 1),
        ("d2", 0, 100), ("d2", 1, 1), ("d2", 2, 50),
    ])

    assert dmt.union_top_n(df, n=2) == [0, 1, 2]


def test_union_top_n_never_returns_other():
    """`other` is a bucket of unrelated decks, not a deck."""
    df = _shares([
        ("d1", dmt.OTHER_ID, 900), ("d1", 0, 50), ("d1", 1, 10),
    ])

    assert dmt.union_top_n(df, n=2) == [0, 1]


def test_union_top_n_since_restricts_selection_not_the_data():
    """Thin days must not nominate series, but must still be plotted.

    On a 210-episode day the 10th-ranked deck holds ~2% — eight games — so a
    naive union over every day nominates decks that were never really there.
    `since` moves the *selection* onto the well-sampled days; the returned ids
    are still charted over their whole history.
    """
    df = _shares([
        ("d1", 0, 5), ("d1", 9, 4),      # thin day: 9 rides in on 4 slots
        ("d2", 0, 5000), ("d2", 1, 4000), ("d2", 9, 1),
    ])

    assert dmt.union_top_n(df, n=2, since="d2") == [0, 1]
    assert dmt.union_top_n(df, n=2) == [0, 1, 9]


def test_union_top_n_since_after_every_day_selects_nothing():
    """Silence beats a quiet fallback to the unfiltered union."""
    df = _shares([("d1", 0, 5), ("d1", 1, 4)])

    assert dmt.union_top_n(df, n=2, since="d9") == []


def test_union_top_n_is_ordered_by_total_slots():
    """Legend order should be stable and readable, not day-one order."""
    df = _shares([
        ("d1", 0, 1), ("d1", 1, 10),
        ("d2", 0, 100), ("d2", 1, 10),
    ])

    assert dmt.union_top_n(df, n=2) == [0, 1]


# ============================================================
# Labels
# ============================================================

def test_load_card_table_reads_id_name_and_kind(tmp_path):
    csv_path = tmp_path / "cards.csv"
    csv_path.write_text(
        "Card ID,Card Name,Stage (Pokémon)/Type (Energy and Trainer)\n"
        "1,Basic {G} Energy,Basic Energy\n"
        "42,Pikachu ex,Basic Pokémon\n",
        encoding="utf-8",
    )

    table = dmt.load_card_table(csv_path)

    assert table[42] == ("Pikachu ex", "Basic Pokémon")
    assert table[1][0] == "Basic {G} Energy"


def test_deck_label_prefers_pokemon_over_the_most_copied_card():
    """Every deck runs a pile of basic energy; none of them is its name.

    The representative's *most-copied* card is almost always energy, so a
    naive top-count label reads "Basic {G} Energy" for half the meta.
    """
    arch = Archetype(id=7, representative=canon([1] * 40 + [42] * 20))
    table = {1: ("Basic {G} Energy", "Basic Energy"), 42: ("Pikachu ex", "Basic Pokémon")}

    labels = dmt.deck_labels([arch], table, top_k=1)

    assert labels[7] == "#7 Pikachu ex"


def test_deck_label_prefers_the_card_that_distinguishes_the_cluster():
    """A staple every deck runs identifies none of them.

    Both decks run 20 Professor's Research; only one runs Charizard.  The
    label must lead with the card that tells them apart.
    """
    common = [50] * 40
    a = Archetype(id=0, representative=canon(common + [60] * 20))
    b = Archetype(id=1, representative=canon(common + [61] * 20))
    table = {
        50: ("Professor's Research", "Basic Pokémon"),
        60: ("Charizard ex", "Basic Pokémon"),
        61: ("Gardevoir ex", "Basic Pokémon"),
    }

    labels = dmt.deck_labels([a, b], table, top_k=1)

    assert labels[0] == "#0 Charizard ex"
    assert labels[1] == "#1 Gardevoir ex"


def test_deck_label_names_the_payoff_card_not_the_basic_it_evolves_from():
    """Players call it Alakazam, never "Abra / Kadabra".

    Evolution basics are 4-ofs and the payoff is a 2- or 3-of, so ranking by
    copies alone names every deck after a card nobody attacks with.
    """
    arch = Archetype(id=0, representative=canon([10] * 4 + [11] * 4 + [12] * 3 + [13] * 49))
    table = {
        10: ("Abra", "Basic Pokémon"),
        11: ("Kadabra", "Stage 1 Pokémon"),
        12: ("Alakazam", "Stage 2 Pokémon"),
        13: ("Basic {P} Energy", "Basic Energy"),
    }

    assert dmt.deck_labels([arch], table, top_k=1)[0] == "#0 Alakazam"


def test_deck_label_leads_with_the_ex_over_a_higher_stage():
    """"Mega Kangaskhan ex" outranks Crustle even though Crustle is Stage 1
    with more copies — the ex is what the deck is called."""
    arch = Archetype(id=5, representative=canon([20] * 4 + [21] * 4 + [22] * 4 + [23] * 48))
    table = {
        20: ("Dwebble", "Basic Pokémon"),
        21: ("Crustle", "Stage 1 Pokémon"),
        22: ("Mega Kangaskhan ex", "Basic Pokémon"),
        23: ("Basic {F} Energy", "Basic Energy"),
    }

    assert dmt.deck_labels([arch], table, top_k=1)[5] == "#5 Mega Kangaskhan ex"


def test_deck_label_prefers_the_ex_it_actually_runs_four_of():
    """A 1-of tech ex must not outrank the 3-of the deck is built around."""
    arch = Archetype(id=6, representative=canon([30] * 3 + [31] + [32] * 56))
    table = {
        30: ("Marnie's Grimmsnarl ex", "Stage 2 Pokémon"),
        31: ("Fezandipiti ex", "Basic Pokémon"),
        32: ("Basic {D} Energy", "Basic Energy"),
    }

    assert dmt.deck_labels([arch], table, top_k=1)[6] == "#6 Marnie's Grimmsnarl ex"


def test_deck_label_rejects_a_splashable_tech_ex_for_the_real_payoff():
    """Fezandipiti ex is a 1-of in half the format; it names nothing.

    Preferring `ex` must stay a *weight*, not a veto — otherwise every deck
    running the generic tech attacker gets labelled after it, and the Stage 2
    the deck is actually built around never appears.
    """
    tech, abra, kadabra, zam, energy = 31, 30, 33, 32, 39
    target = Archetype(id=0, representative=canon(
        [abra] * 4 + [kadabra] * 4 + [zam] * 3 + [tech] + [energy] * 48))
    # the same tech ex, one copy, in four unrelated clusters
    others = [Archetype(id=i, representative=canon([tech] + [50 + i] * 59))
              for i in range(1, 5)]
    table = {
        abra: ("Abra", "Basic Pokémon"),
        kadabra: ("Kadabra", "Stage 1 Pokémon"),
        zam: ("Alakazam", "Stage 2 Pokémon"),
        tech: ("Fezandipiti ex", "Basic Pokémon"),
        energy: ("Basic {P} Energy", "Basic Energy"),
        **{50 + i: (f"Filler {i}", "Basic Pokémon") for i in range(1, 5)},
    }

    assert dmt.deck_labels([target, *others], table, top_k=1)[0] == "#0 Alakazam"


def test_deck_label_falls_back_to_the_bare_id_without_a_card_table():
    arch = Archetype(id=3, representative=canon([1] * 60))

    assert dmt.deck_labels([arch], {})[3] == "#3"


def test_other_is_labelled():
    assert dmt.deck_labels([], {})[dmt.OTHER_ID] == "other"
