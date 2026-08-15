"""Render `deck_meta_trend`'s CSV into a standalone local HTML report.

The output is one self-contained file — inline CSS, inline JS, the data
embedded as JSON, charts hand-built as SVG.  No server, no build step, no
external request of any kind, so it opens straight off disk:

    uv run python -m ptcg_mine.deck_meta_trend --emit-page
    xdg-open data/deck_meta_trend.html

Split from `deck_meta_trend` because the two answer different questions.  That
module measures the corpus and its output (the CSV) is the durable artifact;
this one makes presentation choices — a 7-day window, which series to draw,
what to call each deck — that a later reader may well want to make differently
without re-running a 25-minute scan.
"""

from __future__ import annotations

import json
from pathlib import Path

from .archetype import Archetype
from .deck_meta_trend import OTHER_ID, deck_labels, load_card_table, union_top_n

#: The token the template reserves for the JSON payload.
PLACEHOLDER = "__DECK_META_TREND_DATA__"

TEMPLATE_PATH = Path(__file__).with_name("deck_meta_trend_page.html")

#: Days in the trailing mean.  The corpus swings from 210 to 4812 episodes a
#: day, so the raw daily line is dominated by sampling noise on its left half;
#: a week also absorbs any day-of-week effect in ladder play.
DEFAULT_MA_WINDOW = 7

#: Days per arm of the "last window vs the one before it" comparison.  Two
#: 7-day arms both sit inside the dense era of this corpus, so the difference
#: is a change in play and not a change in how much was sampled.
DEFAULT_MOMENTUM_WINDOW = 7


def build_payload(df, archetypes: list[Archetype], *, top_n: int = 10,
                  select_since: str | None = None,
                  window: int = DEFAULT_MA_WINDOW,
                  momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
                  card_table: dict | None = None) -> dict:
    """The JSON the page renders, from `deck_meta_trend`'s long-form table.

    *df* carries every archetype; the payload keeps only the tracked series
    plus two aggregates — `other` (decklists that matched no cluster at all)
    and `rest` (everything clustered but untracked), which together make each
    day's numbers sum to 1.
    """
    days = sorted(df["day"].unique())
    series_ids = union_top_n(df, n=top_n, since=select_since)
    labels = deck_labels(archetypes,
                         load_card_table() if card_table is None else card_table,
                         top_k=2)

    wide = df.pivot(index="day", columns="archetype_id", values="share").reindex(days)
    slots = df.pivot(index="day", columns="archetype_id", values="slots").reindex(days)
    totals = df.groupby("day")["day_total_slots"].first().reindex(days)

    def ma(s):
        return s.rolling(window, min_periods=1).mean()

    def rounded(s):
        return [round(float(v), 6) for v in s]

    out_series = []
    for aid in series_ids:
        share = wide[aid]
        recent = share.iloc[-momentum_window:].mean()
        prior = share.iloc[-2 * momentum_window:-momentum_window].mean()
        # Fewer than 2*window days leaves the prior arm empty, and NaN would
        # sort to the middle of the momentum chart rather than reading as
        # "not measured".
        prior = 0.0 if prior != prior else prior
        out_series.append({
            "id": int(aid),
            "label": labels.get(aid, f"#{aid}"),
            "share": rounded(share),
            "ma": rounded(ma(share)),
            "slots": [int(v) for v in slots[aid].fillna(0)],
            "total": int(slots[aid].fillna(0).sum()),
            "recent": round(float(recent), 6),
            "prior": round(float(prior), 6),
            "momentum": round(float(recent - prior), 6),
        })

    other = wide[OTHER_ID] if OTHER_ID in wide else None
    rest = 1.0 - wide[series_ids].sum(axis=1) if series_ids else 1.0 - wide.sum(axis=1)

    return {
        "days": list(days),
        "totals": [int(v) for v in totals],
        "episodes": [int(v) // 2 for v in totals],
        "series": out_series,
        "other": {"label": "未分群牌表", "share": rounded(other),
                  "ma": rounded(ma(other))} if other is not None else None,
        "rest": {"label": "其餘牌組", "share": rounded(rest)},
        "meta": {
            "n_days": len(days),
            "n_episodes": int(totals.sum() // 2),
            "n_slots": int(totals.sum()),
            "n_archetypes": len(archetypes),
            "n_series": len(series_ids),
            "ma_window": window,
            "momentum_window": momentum_window,
            "top_n": top_n,
            "select_since": select_since,
            "dense_days": (len([d for d in days if d >= select_since])
                           if select_since else len(days)),
            # No OTHER column at all means every decklist matched a cluster —
            # that is 100% coverage, not "unknown".
            "coverage": round(float(1 - other.mean()), 4) if other is not None else 1.0,
        },
    }


# ============================================================
# Source pinning
# ============================================================

def archetypes_sha1(path) -> str:
    """The 12-hex fingerprint `mine` already stamps into ``lineage``."""
    import hashlib

    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def write_source_stamp(path, *, archetypes_sha1: str, n_archetypes: int,
                       **extra) -> Path:
    """Record which `archetypes.json` a CSV was measured against."""
    path = Path(path)
    path.write_text(json.dumps(
        {"archetypes_sha1": archetypes_sha1, "n_archetypes": n_archetypes, **extra},
        indent=1))
    return path


def check_source_stamp(path, *, archetypes_sha1: str) -> None:
    """Refuse to re-render a CSV measured against different clusters.

    Archetype ids are cluster indices, not names.  A ``--rebaseline`` renumbers
    them, so pairing a stale CSV with a fresh `archetypes.json` relabels every
    series — silently, and into a chart that still looks entirely plausible.

    A *missing* stamp is not an error: a CSV written before this existed is
    unverifiable rather than known-wrong, and refusing it would strand data the
    caller can still legitimately re-render.
    """
    path = Path(path)
    if not path.exists():
        return
    try:
        stamp = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    recorded = stamp.get("archetypes_sha1")
    if recorded and recorded != archetypes_sha1:
        raise SystemExit(
            f"{path} says the CSV was measured against archetypes.json "
            f"{recorded}, but the one on disk is {archetypes_sha1}. Archetype "
            "ids are cluster indices, so re-rendering would relabel every "
            "series. Re-run without --from-csv to re-measure."
        )


def render(payload: dict, template: str | None = None) -> str:
    """Substitute *payload* into the page template.

    Raises rather than returning the template untouched when the slot is
    missing: a page whose data never landed renders as an empty chart, which
    is indistinguishable from a corpus that genuinely had nothing in it.
    """
    if template is None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(
            f"template has no {PLACEHOLDER} slot to substitute the payload into")
    return template.replace(PLACEHOLDER,
                            json.dumps(payload, ensure_ascii=False), 1)


def write_page(df, archetypes: list[Archetype], out_path, **kwargs) -> Path:
    """Build the payload from *df* and write the finished page to *out_path*."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(build_payload(df, archetypes, **kwargs)),
                        encoding="utf-8")
    return out_path
