"""Tests for the partial-parse projection reader in ptcg_mine.episode.

`load_projection` must be observationally identical to the full parse it
replaces -- ``project_for_selection(json.load(f))`` -- while reading only the
head of the document.  The equivalence is the whole contract: expert selection,
archetype clustering and the kept-game filter all run off this projection, and a
subtly different result would silently change which games reach the model.
"""

import json
from pathlib import Path

import pytest

from ptcg_mine import episode as episode_mod
from ptcg_mine.episode import (
    INVALID,
    OK,
    UNREADABLE,
    ProjectionIncomplete,
    load_projection,
    parse_projection,
    project_for_selection,
    scan_projections,
    validate_episode,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "archive/sample_episodes/80169582.json"


def _full_projection(path):
    with open(path) as f:
        return project_for_selection(json.load(f))


def _synthetic(**overrides) -> dict:
    ep = {
        "info": {"TeamNames": ["alice", "bob"]},
        "rewards": [1, -1],
        "statuses": ["DONE", "DONE"],
        "steps": [
            [{"action": None}, {"action": None}],
            [
                {"action": list(range(60)), "observation": {"junk": "x" * 100}},
                {"action": list(range(100, 160)), "observation": {"junk": "y" * 100}},
            ],
            [{"action": [1]}, {"action": [2]}],
        ],
    }
    ep.update(overrides)
    return ep


# ============================================================
# Equivalence with the full parse
# ============================================================


def test_matches_full_parse_on_real_fixture():
    assert FIXTURE_PATH.exists(), f"fixture missing: {FIXTURE_PATH}"
    assert load_projection(FIXTURE_PATH) == _full_projection(FIXTURE_PATH)


def test_real_fixture_projection_still_validates():
    assert validate_episode(load_projection(FIXTURE_PATH)) is True


def test_matches_full_parse_on_synthetic(tmp_path):
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(_synthetic()))
    assert load_projection(p) == _full_projection(p)


@pytest.mark.parametrize(
    "key_order",
    [
        ["info", "rewards", "statuses", "steps"],
        ["steps", "info", "rewards", "statuses"],
        ["rewards", "steps", "statuses", "info"],
        ["statuses", "info", "steps", "rewards"],
    ],
)
def test_key_order_independent(tmp_path, key_order):
    """Every ordering must agree with the full parse.

    Orders that put `steps` last are the fast path; orders that put a needed
    key after it must fall back rather than return a partial projection.
    """
    ep = _synthetic()
    ordered = {k: ep[k] for k in key_order}
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(ordered))
    assert load_projection(p) == _full_projection(p)


def test_ignores_unknown_top_level_keys(tmp_path):
    ep = _synthetic()
    ep["configuration"] = {"episodeSteps": 300, "nested": [1, 2, {"a": "b"}]}
    ep["schema_version"] = "2"
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(ep))
    assert load_projection(p) == _full_projection(p)


def test_survives_pretty_printed_whitespace(tmp_path):
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(_synthetic(), indent=4))
    assert load_projection(p) == _full_projection(p)


def test_strings_containing_structural_chars(tmp_path):
    """A team name full of braces and escaped quotes must not desync the scan."""
    ep = _synthetic()
    ep["info"]["TeamNames"] = ['a{"steps": [', 'b],}\\"']
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(ep))
    proj = load_projection(p)
    assert proj == _full_projection(p)
    assert proj["info"]["TeamNames"] == ['a{"steps": [', 'b],}\\"']


# ============================================================
# It really is partial
# ============================================================


def test_does_not_parse_the_game_log(tmp_path):
    """The tail of `steps` is never decoded, so garbage there is not seen.

    This is the property that makes the reader fast; asserting it directly
    stops a future refactor from silently reverting to a full parse.
    """
    ep = _synthetic()
    text = json.dumps(ep)
    marker = '[{"action": [1]}, {"action": [2]}]'
    compact = json.dumps(ep, separators=(", ", ": "))
    assert marker in compact, "test fixture no longer contains the third step verbatim"
    corrupted = compact.replace(marker, '[{"action": @@@BROKEN@@@}]')
    p = tmp_path / "ep.json"
    p.write_text(corrupted)

    with pytest.raises(json.JSONDecodeError):
        json.loads(corrupted)
    assert load_projection(p) == _full_projection_of_dict(ep)


def _full_projection_of_dict(ep):
    return project_for_selection(ep)


# ============================================================
# Malformed / short input
# ============================================================


def test_truncated_document_raises(tmp_path):
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(_synthetic())[:200])
    with pytest.raises(json.JSONDecodeError):
        load_projection(p)


def test_not_an_object_raises(tmp_path):
    p = tmp_path / "ep.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(json.JSONDecodeError):
        load_projection(p)


def test_short_steps_projection_fails_validation(tmp_path):
    ep = _synthetic()
    ep["steps"] = [[{"action": None}, {"action": None}]]
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(ep))
    proj = load_projection(p)
    assert validate_episode(proj) is False


def test_empty_steps_projection_fails_validation(tmp_path):
    ep = _synthetic()
    ep["steps"] = []
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(ep))
    assert validate_episode(load_projection(p)) is False


def test_missing_action_key_fails_validation(tmp_path):
    ep = _synthetic()
    del ep["steps"][1][0]["action"]
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(ep))
    assert validate_episode(load_projection(p)) is False


def test_missing_required_key_falls_back_and_still_projects(tmp_path):
    """A document with no `rewards` at all must not be mistaken for a fast-path
    hit; it falls back to the full parse and projects `rewards` as None."""
    ep = _synthetic()
    del ep["rewards"]
    p = tmp_path / "ep.json"
    p.write_text(json.dumps(ep))
    proj = load_projection(p)
    assert proj["rewards"] is None
    assert validate_episode(proj) is False


def test_parse_projection_signals_incomplete_directly():
    """`parse_projection` reports a missing key rather than guessing."""
    ep = _synthetic()
    del ep["statuses"]
    with pytest.raises(ProjectionIncomplete):
        parse_projection(json.dumps(ep))


# ============================================================
# Parallel scan
# ============================================================


@pytest.fixture
def mixed_corpus(tmp_path):
    """A raw dir with one of each outcome, several times over."""
    raw = tmp_path / "raw"
    raw.mkdir()
    expected = {}
    for i in range(12):
        good = _synthetic()
        good["info"]["TeamNames"] = [f"alice{i}", f"bob{i}"]
        (raw / f"ok_{i:03d}.json").write_text(json.dumps(good))
        expected[f"ok_{i:03d}"] = OK

        bad = _synthetic()
        bad["statuses"] = ["DONE", "ACTIVE"]
        (raw / f"invalid_{i:03d}.json").write_text(json.dumps(bad))
        expected[f"invalid_{i:03d}"] = INVALID

        (raw / f"broken_{i:03d}.json").write_text("{not json at all")
        expected[f"broken_{i:03d}"] = UNREADABLE
    return raw, expected


def test_parallel_scan_matches_serial(mixed_corpus, monkeypatch):
    """Workers and in-process must agree on every field.

    Regression guard: results cross a `pickle` boundary, so a status compared
    by identity reads False for every parallel row while the serial path stays
    green.  That failure mode empties the corpus silently -- Phase 2 simply
    finds no episodes -- so parity is asserted rather than assumed.
    """
    raw, expected = mixed_corpus
    paths = sorted(raw.glob("*.json"))
    monkeypatch.setattr(episode_mod, "_PARALLEL_SCAN_MIN_FILES", 2)
    monkeypatch.setattr(episode_mod, "_SCAN_CHUNKSIZE", 3)

    serial = list(scan_projections(paths, jobs=1))
    parallel = list(scan_projections(paths, jobs=4))

    assert len(serial) == len(paths) > 0
    assert serial == parallel, "parallel scan disagrees with serial"


def test_parallel_scan_statuses_are_usable_by_identity(mixed_corpus, monkeypatch):
    """`status is OK` must hold for pool results, not just serial ones."""
    raw, expected = mixed_corpus
    paths = sorted(raw.glob("*.json"))
    monkeypatch.setattr(episode_mod, "_PARALLEL_SCAN_MIN_FILES", 2)

    rows = list(scan_projections(paths, jobs=4))
    assert rows, "no rows scanned; test is vacuous"

    n_ok = 0
    for eid, proj, status in rows:
        assert status is expected[eid], f"{eid}: {status} is not {expected[eid]}"
        if status is OK:
            n_ok += 1
            assert validate_episode(proj)
        else:
            assert proj is None
    assert n_ok == 12, f"expected 12 usable episodes, got {n_ok}"


def test_parallel_scan_preserves_path_order(mixed_corpus, monkeypatch):
    """Cluster ids are assigned in corpus order, so order cannot vary with jobs."""
    raw, _ = mixed_corpus
    paths = sorted(raw.glob("*.json"))
    monkeypatch.setattr(episode_mod, "_PARALLEL_SCAN_MIN_FILES", 2)

    for jobs in (1, 2, 4, 8):
        got = [eid for eid, _, _ in scan_projections(paths, jobs=jobs)]
        assert got == [p.stem for p in paths], f"order changed at jobs={jobs}"
