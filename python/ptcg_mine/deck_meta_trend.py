"""Per-day archetype share of the downloaded corpus — the meta timeline.

The corpus is a sample of *top-ranked* games, so the fraction of a day's games
played with a given decklist is a usable proxy for how strong that deck is
believed to be.  A deck whose share climbs is a deck the best players are
switching to.

This is a read-only analysis over ``raw/<YYYY-MM-DD>/*.json`` and the frozen
``data/archetypes.json``.  It writes no artifact the pipeline consumes and
changes no stage's fingerprint.

    uv run python -m ptcg_mine.deck_meta_trend --raw-dir raw --data-dir data

Two things it deliberately does *not* do.  It does not re-cluster: archetype
ids are cluster indices, and a fresh clustering would produce a series whose
ids mean nothing to any trained checkpoint (see CLAUDE.md, "archetype ids are
append-only").  And it does not drop decks that match no cluster — they are
tallied under `OTHER_ID` and stay in every day's denominator, because a share
measured only over clustered decks is not a share of the meta.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .archetype import (
    DEFAULT_JACCARD_THRESH,
    Archetype,
    canon,
    jaccard_multiset,
    load_archetypes_json,
)
from .episode import OK, deck_of, scan_episode

#: Tally slot for a decklist that matched no archetype at or above the
#: threshold.  Negative so it can never collide with a cluster index, and the
#: same sentinel `meta.parquet`'s `archetype_opp` already uses for "matched no
#: cluster at all".
OTHER_ID = -1

#: A pool costs ~1 s to stand up against ~1 ms per projection, so below this
#: many files it is pure overhead.  Matches `episode._PARALLEL_SCAN_MIN_FILES`
#: in intent; kept separate because this scan's per-file work is different.
_PARALLEL_MIN_FILES = 256
_CHUNKSIZE = 32


@dataclass
class DayScan:
    """What one pass over the raw corpus found, keyed by day.

    `slots` counts **player slots**, not episodes: each valid episode
    contributes two, one per seat, because the question is "what fraction of
    decks piloted that day were this one".
    """

    slots: dict[str, Counter] = field(default_factory=dict)
    #: valid episodes per day (the ones that contributed two slots each)
    episodes: dict[str, int] = field(default_factory=dict)
    #: episodes that failed to parse or failed `validate_episode`
    skipped: dict[str, int] = field(default_factory=dict)


# ============================================================
# Assignment
# ============================================================

def assign_with_memo(deck, archetypes: list[Archetype], thresh: float,
                     memo: dict) -> int:
    """Archetype id for *deck*, or `OTHER_ID`, memoised on the canon decklist.

    Deliberately not `archetype.assign_archetype`: that is a linear scan over
    every cluster (409 in the current artifact) of multiset Jaccard, and this
    module runs it once per player slot — ~183k times over the 41-day corpus,
    which measures in the tens of minutes single-threaded.  Exact decklists
    repeat heavily among top players, so memoising on the canon form collapses
    nearly all of it.

    The memo is per worker process and never shared back, so it cannot make the
    result depend on the order files were visited.
    """
    key = canon(deck)
    hit = memo.get(key)
    if hit is not None:
        return hit
    best_id, best_score = OTHER_ID, -1.0
    for arch in archetypes:
        score = jaccard_multiset(key, arch.representative)
        if score >= thresh and score > best_score:
            best_id, best_score = arch.id, score
    memo[key] = best_id
    return best_id


# ============================================================
# The scan
# ============================================================

#: Read-only context for `_scan_one`, published per worker by a pool
#: initializer.  See `shard_writer._SCAN_CTX` for why the serial path must take
#: it as an argument instead: under `fork`, a global set in the parent is
#: inherited by every worker, which hides a missing `initializer=`.
_CTX: dict | None = None


def _init_ctx(ctx: dict) -> None:
    """Pool initializer: publish *ctx* (and a fresh memo) for this worker."""
    global _CTX
    _CTX = dict(ctx, memo={})


def _scan_one_ctx(path_str: str, ctx: dict) -> tuple[str, tuple[int, int] | None]:
    """``(day, (arch_p0, arch_p1) | None)`` for one episode file.

    The day is the *parent directory* name.  Nothing inside the episode records
    when it was played, and `data/downloaded.parquet` only covers the most
    recent mining run, so the directory layout is the sole source.
    """
    day = Path(path_str).parent.name
    _eid, proj, status = scan_episode(path_str)
    if status is not OK:
        return day, None
    archetypes, thresh, memo = ctx["archetypes"], ctx["thresh"], ctx["memo"]
    return day, (
        assign_with_memo(deck_of(proj, 0), archetypes, thresh, memo),
        assign_with_memo(deck_of(proj, 1), archetypes, thresh, memo),
    )


def _scan_one(path_str: str) -> tuple[str, tuple[int, int] | None]:
    """Pool entry point: `_scan_one_ctx` against this worker's `_CTX`."""
    return _scan_one_ctx(path_str, _CTX)


def scan_day_slots(raw_dir, archetypes: list[Archetype], *,
                   thresh: float = DEFAULT_JACCARD_THRESH,
                   jobs: int | None = None,
                   progress=None) -> DayScan:
    """Tally archetype slots per day over every episode under *raw_dir*.

    *progress*, if given, is called with the number of files completed so far.

    Result order is irrelevant here — every consumer is a counter — so nothing
    downstream can shift with *jobs*.
    """
    raw_dir = Path(raw_dir)
    paths = sorted(str(p) for p in raw_dir.rglob("*.json"))
    ctx = {"archetypes": archetypes, "thresh": thresh}
    n_jobs = (os.cpu_count() or 1) if jobs is None else max(1, jobs)

    scan = DayScan()
    for i, (day, pair) in enumerate(_iter_scanned(paths, n_jobs, ctx), start=1):
        if pair is None:
            scan.skipped[day] = scan.skipped.get(day, 0) + 1
        else:
            scan.episodes[day] = scan.episodes.get(day, 0) + 1
            counter = scan.slots.setdefault(day, Counter())
            counter[pair[0]] += 1
            counter[pair[1]] += 1
        if progress is not None and i % 2000 == 0:
            progress(i)
    if progress is not None:
        progress(len(paths))
    # Days whose every episode was unusable still exist; give them a bucket so
    # the timeline shows a real gap rather than skipping the date silently.
    for day in scan.skipped:
        scan.slots.setdefault(day, Counter())
        scan.episodes.setdefault(day, 0)
    return scan


def _iter_scanned(paths: list[str], n_jobs: int, ctx: dict):
    if n_jobs <= 1 or len(paths) < _PARALLEL_MIN_FILES:
        serial_ctx = dict(ctx, memo={})
        for path in paths:
            yield _scan_one_ctx(path, serial_ctx)
        return

    import multiprocessing as mp

    # fork for the same reason `episode.scan_projections` does: this scan is
    # read-only and predates any engine handle or CUDA context, while spawn
    # would re-import pandas in every worker and oblige callers to carry a
    # __main__ guard.
    try:
        mp_ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover - non-fork platforms
        mp_ctx = mp.get_context()
    with mp_ctx.Pool(processes=n_jobs, initializer=_init_ctx,
                     initargs=(ctx,)) as pool:
        yield from pool.imap_unordered(_scan_one, paths, chunksize=_CHUNKSIZE)


# ============================================================
# Shares
# ============================================================

def daily_shares(scan: DayScan) -> "pd.DataFrame":
    """Long-form ``day × archetype_id`` table with an explicit zero per gap.

    Columns: ``day, archetype_id, slots, day_total_slots, share``.

    Every archetype seen anywhere in the corpus gets a row on every day, so a
    deck that vanished plots as a line to zero rather than a line that stops.
    """
    import pandas as pd

    days = sorted(scan.slots)
    ids = sorted({aid for c in scan.slots.values() for aid in c})
    rows = []
    for day in days:
        counter = scan.slots[day]
        total = sum(counter.values())
        for aid in ids:
            slots = counter.get(aid, 0)
            rows.append({
                "day": day,
                "archetype_id": aid,
                "slots": slots,
                "day_total_slots": total,
                # A day with no usable episode has no share to report; 0.0
                # would claim every deck was absent, which is a measurement we
                # did not make.
                "share": (slots / total) if total else float("nan"),
            })
    return pd.DataFrame(rows, columns=["day", "archetype_id", "slots",
                                       "day_total_slots", "share"])


def union_top_n(df, n: int = 10, since: str | None = None) -> list[int]:
    """Archetype ids that entered **any** day's top *n* by slots.

    The union rather than the overall top *n* on purpose: a deck that surges
    late is exactly the signal this chart exists to show, and it can own a week
    of the corpus while still ranking 30th overall.

    *since* restricts which days may **nominate** a series — not which days are
    measured.  Corpus density is wildly uneven (this corpus ranges 210 to 4812
    episodes a day), and on a thin day the *n*-th rank is a handful of games,
    so an unrestricted union nominates decks that were never really there:
    measured, every-day selection returns 70 series against 27 from the dense
    era alone, and a 1%-share floor removes none of the difference.  Days
    before *since* still contribute to every returned series' history.

    Returned in descending total-slot order so the legend is stable across
    re-runs and the biggest decks read first.  `OTHER_ID` is excluded — it is a
    bucket of unrelated decklists, not a deck.
    """
    real = df[df["archetype_id"] != OTHER_ID]
    if real.empty:
        return []
    window = real if since is None else real[real["day"] >= since]
    picked = set()
    for _day, group in window.groupby("day"):
        picked.update(group.nlargest(n, "slots")["archetype_id"].tolist())
    if not picked:
        return []
    totals = (real[real["archetype_id"].isin(picked)]
              .groupby("archetype_id")["slots"].sum()
              .sort_values(ascending=False))
    return [int(i) for i in totals.index]


# ============================================================
# Deck labels
# ============================================================

#: The Kaggle bundle's card list, read for names only.  Deliberately not
#: `cards.load_engine()`: that imports `cg.api`, and *importing* it is the
#: `GameInitialize` call, which is a process-wide latch — far too much to take
#: on for a legend label.
DEFAULT_CARD_CSV = (Path(__file__).resolve().parents[1]
                    / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv")

_NAME_COL = "Card Name"
_ID_COL = "Card ID"
_KIND_COL = "Stage (Pokémon)/Type (Energy and Trainer)"


def load_card_table(csv_path=None) -> dict[int, tuple[str, str]]:
    """``card_id -> (name, kind)`` from the bundled card CSV.

    *kind* is the raw "Basic Pokémon" / "Item" / "Supporter" / "Basic Energy"
    column; `deck_labels` only asks whether it names a Pokémon.

    A missing or unreadable file returns ``{}`` rather than raising — labels
    degrade to bare ids, and the id is what actually identifies the deck.
    """
    import csv

    path = Path(csv_path) if csv_path is not None else DEFAULT_CARD_CSV
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, UnicodeDecodeError):
        return {}
    table = {}
    for row in rows:
        try:
            cid = int(row[_ID_COL])
        except (KeyError, TypeError, ValueError):
            continue
        table[cid] = (row.get(_NAME_COL) or "", row.get(_KIND_COL) or "")
    return table


def deck_labels(archetypes: list[Archetype], card_table: dict | None = None,
                top_k: int = 3) -> dict[int, str]:
    """``archetype_id -> human label`` built from each cluster's representative.

    "#17" is unreadable in a legend, so the label carries the cards that
    identify the deck.  Two corrections to the obvious "most-copied card"
    heuristic, both of which otherwise make every label the same:

    * **Pokémon first.** The most-copied card in nearly every 60-card list is
      basic energy, which names no deck.  Non-Pokémon are considered only when
      the representative holds no Pokémon at all.
    * **The payoff card, not the basic it evolves from.** Evolution basics are
      4-ofs and the card the deck is named after is a 2- or 3-of, so ranking by
      copies alone labels a Gardevoir deck "Abra / Kadabra" and a Dragapult
      deck "Dreepy / Drakloak".  ex/Mega, then Stage 2, then Stage 1 carry a
      multiplier — a *weight*, never a veto.  As a hard tier it fails the other
      way: a 1-of splashable tech ex (Fezandipiti ex sits in a large fraction of
      the format's clusters) would outrank every deck's real payoff.
    * **Weight by how few clusters run it.** Staples — the draw Supporter, the
      search Item, the format's ubiquitous attacker — appear in most clusters
      and distinguish none, so a card's copy count is scaled by
      ``log((N + 1) / (1 + df))`` over the *archetypes* that run it.  The
      ``N + 1`` keeps the factor non-negative: with a plain ``N`` the term flips
      sign for a card most clusters run, which silently *inverts* the ranking
      rather than flattening it.

    Ties fall back to the stage order and then to copy count, which is what
    decides a single-archetype call — every ``df`` equals ``N`` there, so the
    inverse-frequency term carries no information at all.

    Falls back to a bare ``#id`` when no name is known.
    """
    import math
    import re

    card_table = card_table or {}
    n_arch = len(archetypes)
    df: Counter = Counter()
    for arch in archetypes:
        df.update(set(arch.representative))

    #: Multipliers, not ranks.  Gentle on purpose: they reorder cards of
    #: comparable distinctiveness without letting a card nobody associates with
    #: the deck jump a well-separated payoff.
    _STAGE_WEIGHT = {"ex": 2.0, "s2": 1.7, "s1": 1.2, "basic": 1.0}

    def stage(kind: str, name: str) -> str:
        if re.search(r"\bex\b|^Mega\b|\bVSTAR\b|\bVMAX\b", name):
            return "ex"
        if "Stage 2" in kind:
            return "s2"
        if "Stage 1" in kind:
            return "s1"
        return "basic"

    _ORDER = list(_STAGE_WEIGHT)

    labels = {}
    for arch in archetypes:
        counts = Counter(arch.representative)
        named = {cid: card_table[cid][0] for cid in counts
                 if cid in card_table and card_table[cid][0]}
        pokemon = {cid: name for cid, name in named.items()
                   if "Pokémon" in card_table[cid][1]}
        pool = pokemon or named

        def key(cid):
            st = stage(card_table[cid][1], pool[cid])
            idf = math.log((n_arch + 1) / (1 + df[cid]))
            return (-counts[cid] * idf * _STAGE_WEIGHT[st],
                    _ORDER.index(st), -counts[cid], cid)

        parts = [pool[cid] for cid in sorted(pool, key=key)[:top_k]]
        labels[arch.id] = (f"#{arch.id} " + " / ".join(parts)) if parts else f"#{arch.id}"
    labels[OTHER_ID] = "other"
    return labels


# ============================================================
# CLI
# ============================================================

def run(raw_dir, data_dir, out_csv, *, top_n: int = 10,
        select_since: str | None = None,
        thresh: float = DEFAULT_JACCARD_THRESH, jobs: int | None = None,
        emit_page=None, from_csv: bool = False) -> "pd.DataFrame":
    """Scan the corpus and write the day × archetype share table to *out_csv*.

    With *from_csv* the scan is skipped and *out_csv* is read back instead —
    the measurement takes ~25 minutes over a 90k-episode corpus while the page
    takes under a second, so re-rendering must not require re-measuring.

    *emit_page*, when given, is the path of a standalone HTML report to write
    beside the CSV.
    """
    import pandas as pd

    from .deck_meta_trend_page import (
        archetypes_sha1, check_source_stamp, write_source_stamp,
    )

    arch_path = Path(data_dir) / "archetypes.json"
    archetypes, _self_ids, _opp_ids = load_archetypes_json(arch_path)
    arch_sha = archetypes_sha1(arch_path)
    out_csv = Path(out_csv)
    stamp_path = out_csv.with_name(out_csv.stem + "_source.json")

    if from_csv:
        if not out_csv.exists():
            raise SystemExit(
                f"--from-csv needs an existing {out_csv}; run without it first "
                "to scan the corpus")
        check_source_stamp(stamp_path, archetypes_sha1=arch_sha)
        df = pd.read_csv(out_csv)
        print(f"deck-meta-trend: reusing {out_csv} "
              f"({df['day'].nunique()} days, no scan)")
    else:
        n_files = sum(1 for _ in Path(raw_dir).rglob("*.json"))
        print(f"deck-meta-trend: {n_files} episodes, {len(archetypes)} archetypes")

        def _tick(done):
            print(f"  scanned {done}/{n_files}", flush=True)

        scan = scan_day_slots(raw_dir, archetypes, thresh=thresh, jobs=jobs,
                              progress=_tick)
        df = daily_shares(scan)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        labels = deck_labels(archetypes, load_card_table())
        out_csv.with_name(out_csv.stem + "_labels.json").write_text(
            json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False,
                       indent=1))
        # Stamped only after the CSV is complete, so a failed run leaves
        # nothing that a later --from-csv would trust.
        write_source_stamp(stamp_path, archetypes_sha1=arch_sha,
                           n_archetypes=len(archetypes),
                           n_days=len(scan.slots),
                           n_episodes=sum(scan.episodes.values()))
        print(f"deck-meta-trend: {len(scan.slots)} days, "
              f"{sum(scan.episodes.values())} valid episodes -> {out_csv}")

    series = union_top_n(df, n=top_n, since=select_since)
    print(f"deck-meta-trend: {len(series)} series in the top-{top_n} union"
          + (f" (nominated from {select_since} on)" if select_since else ""))

    if emit_page is not None:
        from .deck_meta_trend_page import write_page

        path = write_page(df, archetypes, emit_page, top_n=top_n,
                          select_since=select_since)
        print(f"deck-meta-trend: report -> file://{path.resolve()}")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None,
                    help="CSV path (default: <data-dir>/deck_meta_trend.csv)")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--select-since", default=None, metavar="YYYY-MM-DD",
                    help="only days from this date on may nominate a series; "
                         "earlier days are still measured and plotted")
    ap.add_argument("--emit-page", nargs="?", const=True, default=None,
                    metavar="PATH",
                    help="also write a standalone HTML report (default: "
                         "<data-dir>/deck_meta_trend.html); open it off disk, "
                         "it needs no server")
    ap.add_argument("--from-csv", action="store_true",
                    help="re-render from the existing CSV instead of "
                         "re-scanning the corpus (~25 min saved)")
    ap.add_argument("--jaccard-thresh", type=float, default=DEFAULT_JACCARD_THRESH)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)
    out = Path(args.out or (Path(args.data_dir) / "deck_meta_trend.csv"))
    page = args.emit_page
    if page is True:  # bare --emit-page
        page = out.with_suffix(".html")
    run(args.raw_dir, args.data_dir, out, top_n=args.top_n,
        select_since=args.select_since, thresh=args.jaccard_thresh,
        jobs=args.jobs, emit_page=page, from_csv=args.from_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
