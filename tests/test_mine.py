"""Tests for ptcg_mine.mine orchestration: graceful handling of a corpus too
thin to select any experts (must not crash deep inside archetype selection)."""

import pytest

from ptcg_mine import mine
from ptcg_mine.config import MineConfig


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
