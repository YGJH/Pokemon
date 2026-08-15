"""Tests for ptcg_mine.deck_meta_trend_page — the local HTML report.

The page is a single self-contained file opened straight off disk, so the two
things that must hold are that the payload's derived numbers (moving average,
momentum, the untracked remainder) match their definitions, and that the
template really got its data substituted — a page that renders empty looks
exactly like a page with no data in it.
"""

import json

import pandas as pd
import pytest

from ptcg_mine import deck_meta_trend as dmt
from ptcg_mine import deck_meta_trend_page as page
from ptcg_mine.archetype import Archetype, canon


def _frame(rows):
    """rows: (day, archetype_id, slots) -> the CSV's long-form schema."""
    df = pd.DataFrame(rows, columns=["day", "archetype_id", "slots"])
    totals = df.groupby("day")["slots"].transform("sum")
    df["day_total_slots"] = totals
    df["share"] = df["slots"] / totals
    return df


def _by_id(payload, aid):
    """The series entry for *aid* — `series` is ordered by total slots, not id."""
    return next(s for s in payload["series"] if s["id"] == aid)


@pytest.fixture
def archetypes():
    return [Archetype(id=0, representative=canon([1] * 60)),
            Archetype(id=999, representative=canon([2] * 60))]


# ============================================================
# Derived numbers
# ============================================================

def test_moving_average_trails_the_window_and_starts_partial():
    """Day 1 has no seven days behind it; the curve must still start at day 1.

    `min_periods=1` rather than a leading gap: the chart's whole left edge is
    the sparse-corpus era, and dropping it would hide the days the sample-size
    strip exists to warn about.
    """
    days = [f"2026-08-{d:02d}" for d in range(1, 9)]
    df = _frame([(d, 0, s) for d, s in zip(days, [0, 0, 0, 0, 0, 0, 0, 80])]
                + [(d, 999, 20) for d in days])

    payload = page.build_payload(df, [], window=7, momentum_window=2)
    ma = _by_id(payload, 0)["ma"]

    assert ma[0] == pytest.approx(0.0)
    # day 8 is the first non-zero; a 7-day trailing mean holds 1/7 of it
    assert ma[7] == pytest.approx((80 / 100) / 7, abs=1e-6)


def test_momentum_is_the_last_window_minus_the_one_before_it():
    days = [f"2026-08-{d:02d}" for d in range(1, 5)]
    # 100 slots every day, so shares are 10%, 10%, 30%, 30%
    df = _frame([(days[0], 0, 10), (days[1], 0, 10), (days[2], 0, 30), (days[3], 0, 30),
                 (days[0], 999, 90), (days[1], 999, 90),
                 (days[2], 999, 70), (days[3], 999, 70)])

    payload = page.build_payload(df, [], window=2, momentum_window=2)
    s = _by_id(payload, 0)

    assert s["recent"] == pytest.approx(0.30)
    assert s["prior"] == pytest.approx(0.10)
    assert s["momentum"] == pytest.approx(0.20)


def test_rest_is_everything_outside_the_tracked_set():
    """The table's rows have to sum to 100%.

    Only ~27 of 409 clusters are charted, so without this the reader has no
    idea whether the untracked remainder is 2% or 40% — and on the sparse days
    it really is closer to 40%.
    """
    df = _frame([("d1", 0, 30), ("d1", 999, 20), ("d1", 7, 50)])

    payload = page.build_payload(df, [], top_n=1, window=1, momentum_window=1)

    assert [s["id"] for s in payload["series"]] == [7]
    assert payload["rest"]["share"][0] == pytest.approx(0.5)


def test_coverage_is_one_when_every_deck_matched_a_cluster():
    """No OTHER bucket means 100% coverage, not unknown."""
    df = _frame([("d1", 0, 30), ("d1", 7, 70)])

    assert page.build_payload(df, [], window=1, momentum_window=1)["meta"]["coverage"] == 1.0


def test_coverage_counts_the_unclustered_bucket():
    df = _frame([("d1", dmt.OTHER_ID, 25), ("d1", 7, 75)])

    payload = page.build_payload(df, [], window=1, momentum_window=1)

    assert payload["meta"]["coverage"] == pytest.approx(0.75)
    assert payload["other"] is not None


def test_select_since_reaches_the_payload(archetypes):
    """The window that chose the series has to be stated on the page itself."""
    df = _frame([("2026-07-01", 0, 90), ("2026-07-01", 7, 10),
                 ("2026-08-01", 0, 10), ("2026-08-01", 7, 90)])

    payload = page.build_payload(df, archetypes, top_n=1, select_since="2026-08-01",
                                 window=1, momentum_window=1)

    assert payload["meta"]["select_since"] == "2026-08-01"
    assert [s["id"] for s in payload["series"]] == [7]


# ============================================================
# Rendering
# ============================================================

def test_render_substitutes_the_payload_and_leaves_no_placeholder():
    html = page.render({"days": ["d1"], "series": []},
                       template="<p>x</p><script>" + page.PLACEHOLDER + "</script>")

    assert page.PLACEHOLDER not in html
    assert '"days"' in html


def test_render_rejects_a_template_with_no_placeholder():
    """Silently returning the template would publish an empty-looking chart."""
    with pytest.raises(ValueError, match=page.PLACEHOLDER):
        page.render({}, template="<p>no slot here</p>")


def test_the_shipped_template_carries_exactly_one_placeholder():
    """Two slots would inject the payload twice and double the file size."""
    assert page.TEMPLATE_PATH.exists()
    assert page.TEMPLATE_PATH.read_text(encoding="utf-8").count(page.PLACEHOLDER) == 1


def test_source_stamp_pins_the_archetypes_the_csv_was_measured_against(tmp_path):
    """`--from-csv` must refuse a CSV measured against different clusters.

    Archetype ids are cluster indices.  A `--rebaseline` renumbers them, so
    re-rendering a stale CSV against a fresh `archetypes.json` would relabel
    every series on the page — with no error, and a chart that looks entirely
    plausible.
    """
    stamp = tmp_path / "trend_source.json"
    page.write_source_stamp(stamp, archetypes_sha1="aaaaaaaaaaaa", n_archetypes=409)

    page.check_source_stamp(stamp, archetypes_sha1="aaaaaaaaaaaa")  # matching: quiet
    with pytest.raises(SystemExit, match="archetypes.json"):
        page.check_source_stamp(stamp, archetypes_sha1="bbbbbbbbbbbb")


def test_a_missing_source_stamp_is_not_an_error(tmp_path):
    """A CSV from before the stamp existed is unverifiable, not wrong."""
    page.check_source_stamp(tmp_path / "absent.json", archetypes_sha1="aaaaaaaaaaaa")


def test_write_page_produces_a_file_a_browser_can_open(tmp_path, archetypes):
    df = _frame([("2026-08-01", 0, 40), ("2026-08-01", 7, 60),
                 ("2026-08-02", 0, 30), ("2026-08-02", 7, 70)])
    out = tmp_path / "report.html"

    page.write_page(df, archetypes, out, top_n=1)

    html = out.read_text(encoding="utf-8")
    assert html.lstrip().startswith("<title>")
    assert page.PLACEHOLDER not in html
    # the payload survived the round trip as parseable JSON
    body = html.split('<script id="payload" type="application/json">')[1]
    assert json.loads(body.split("</script>")[0])["days"] == ["2026-08-01", "2026-08-02"]
