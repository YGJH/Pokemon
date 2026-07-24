"""Tests for ptcg_mine.mine orchestration: graceful handling of a corpus too
thin to select any experts (must not crash deep inside archetype selection),
and memory-bounded corpus loading."""

import json

import pytest

from ptcg_mine import mine
from ptcg_mine.config import MineConfig
from ptcg_mine.episode import deck_of, rewards, teams, validate_episode


def _write_episode(raw_dir, name, n_steps=140, step_payload_cards=400):
    """Write an episode whose step log is far larger than its Phase-2 payload.

    Real episodes are ~4 MB on disk, nearly all of it the turn-by-turn `steps`
    log; Phase 2 only ever reads TeamNames / rewards / statuses / steps[1][p].
    """
    deck0 = [f"c{i % 30}" for i in range(60)]
    deck1 = [f"d{i % 30}" for i in range(60)]
    fat_step = [
        {"action": [f"x{i}" for i in range(step_payload_cards)], "observation": {"junk": "y" * 200}},
        {"action": [f"z{i}" for i in range(step_payload_cards)], "observation": {"junk": "y" * 200}},
    ]
    ep = {
        "info": {"TeamNames": ["alpha", "beta"]},
        "statuses": ["DONE", "DONE"],
        "rewards": [1, 0],
        "steps": [[{"action": None}, {"action": None}],
                  [{"action": deck0}, {"action": deck1}]]
        + [fat_step for _ in range(n_steps)],
    }
    path = raw_dir / name
    path.write_text(json.dumps(ep))
    return path


def test_load_raw_episodes_does_not_retain_step_log(tmp_path):
    """Loading must be bounded by episode *count*, not corpus bytes.

    Holding whole parsed episodes costs ~14.5 MB each (3.5x the on-disk 4 MB),
    so a 10k-episode corpus needs ~145 GB of RAM and gets OOM-killed. Only the
    Phase-2 projection may be retained.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(5):
        _write_episode(raw, f"ep{i}.json")

    episodes, n_loaded = mine.load_raw_episodes(raw)
    assert n_loaded == 5
    assert len(episodes) == 5

    on_disk = (raw / "ep0.json").stat().st_size
    retained = len(json.dumps(episodes[0]).encode())
    assert retained * 10 < on_disk, (
        f"retained {retained} B per episode vs {on_disk} B on disk — "
        "the step log is still being held in memory"
    )


def test_load_raw_episodes_preserves_phase2_accessors(tmp_path):
    """Pruning must not break anything Phase 2 reads off an episode."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_episode(raw, "ep0.json")

    episodes, _ = mine.load_raw_episodes(raw)
    ep = episodes[0]

    assert validate_episode(ep)
    assert tuple(teams(ep)) == ("alpha", "beta")
    assert list(rewards(ep)) == [1, 0]
    assert len(deck_of(ep, 0)) == 60
    assert len(deck_of(ep, 1)) == 60
    assert deck_of(ep, 0)[0] == "c0"
    assert deck_of(ep, 1)[0] == "d0"


def test_load_raw_episodes_still_drops_invalid_episodes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_episode(raw, "good.json")
    bad = json.loads((raw / "good.json").read_text())
    bad["statuses"] = ["DONE", "ERROR"]
    (raw / "bad.json").write_text(json.dumps(bad))

    episodes, n_loaded = mine.load_raw_episodes(raw)
    assert n_loaded == 2
    assert len(episodes) == 1
    assert tuple(teams(episodes[0])) == ("alpha", "beta")


def test_run_raises_insufficient_data_on_empty_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    config = MineConfig(raw_dir=raw, out_dir=tmp_path / "data")
    with pytest.raises(mine.InsufficientDataError):
        mine.run(config, skip_download=True)


def test_main_reports_insufficient_data_cleanly(monkeypatch, capsys, tmp_path):
    def boom(config, skip_download):
        raise mine.InsufficientDataError("corpus too thin")

    monkeypatch.setattr(mine, "run", boom)
    rc = mine.main(["--skip-download", "--raw-dir", str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "corpus too thin" in (captured.out + captured.err)
