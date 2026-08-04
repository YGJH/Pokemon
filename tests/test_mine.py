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
    def boom(config, skip_download, force=False, jobs=None):
        raise mine.InsufficientDataError("corpus too thin")

    monkeypatch.setattr(mine, "run", boom)
    rc = mine.main(["--skip-download", "--raw-dir", str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "corpus too thin" in (captured.out + captured.err)


# ============================================================
# Phase 2 skip-when-unchanged (ptcg_mine.stamp)
# ============================================================
#
# Phase 2 re-parses the whole corpus (~15 min over the real 9910 episodes) to
# produce artifacts that are a pure function of (corpus, config, code).  These
# tests pin *when* it is allowed to skip that work — a skip that fires on a
# changed corpus would train the next stage on artifacts describing a corpus
# that no longer exists.


def _stamped_data_dir(tmp_path, config, summary=None):
    """A data dir holding every mine output plus a matching fresh stamp."""
    from ptcg_mine import stamp

    data = config.out_dir
    data.mkdir(parents=True, exist_ok=True)
    (data / "vocab.json").write_text('{"size": 296}')
    (data / "archetypes.json").write_text('{"self_ids": [0]}')
    for rel in stamp.STAGES["mine"].outputs:
        p = data / rel
        if not p.exists():
            p.write_bytes(rel.encode())
    stamp.write("mine", raw_dir=config.raw_dir, data_dir=data,
                params=stamp.params_from_config("mine", config),
                summary=summary or {"vocab_size": 296, "n_attacks": 216,
                                    "n_experts": 10, "n_archetypes": 179,
                                    "self_ids": [0, 1], "opp_ids": [0, 1],
                                    "out_dir": str(data), "episodes_loaded": 9910,
                                    "episodes_valid": 9900, "expert_games": 500})
    return data


def _config_with_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_episode(raw, "a.json", n_steps=2, step_payload_cards=2)
    return MineConfig(raw_dir=raw, out_dir=tmp_path / "data")


def test_run_skips_phase2_when_nothing_changed(tmp_path, monkeypatch):
    config = _config_with_corpus(tmp_path)
    _stamped_data_dir(tmp_path, config)

    def boom(*a, **k):
        raise AssertionError("Phase 2 re-parsed the corpus despite a fresh stamp")

    monkeypatch.setattr(mine, "load_raw_episodes", boom)
    summary = mine.run(config, skip_download=True)
    assert summary["cached"] is True
    assert summary["vocab_size"] == 296
    assert summary["self_ids"] == [0, 1]


def _spy_phase2(monkeypatch):
    """Record whether Phase 2 re-parsed the corpus, without stubbing it out.

    The thin fixture corpus completes Phase 2 successfully (select_experts
    falls back to a 1-game threshold on thin data), so "did it raise?" cannot
    tell a skip from a run — the call itself has to be observed.
    """
    calls = []
    real = mine.load_raw_episodes

    def spy(raw_dir, jobs=None):
        calls.append(raw_dir)
        return real(raw_dir, jobs)

    monkeypatch.setattr(mine, "load_raw_episodes", spy)
    return calls


def test_run_recomputes_when_the_corpus_grew(tmp_path, monkeypatch):
    """The skip must key on corpus content, not merely on artifacts existing."""
    config = _config_with_corpus(tmp_path)
    _stamped_data_dir(tmp_path, config)
    _write_episode(config.raw_dir, "b.json", n_steps=2, step_payload_cards=2)
    calls = _spy_phase2(monkeypatch)

    summary = mine.run(config, skip_download=True)
    assert calls, "a grown corpus was served from the stamp"
    assert summary["cached"] is False


def test_run_recomputes_when_forced(tmp_path, monkeypatch):
    config = _config_with_corpus(tmp_path)
    _stamped_data_dir(tmp_path, config)
    calls = _spy_phase2(monkeypatch)

    mine.run(config, skip_download=True, force=True)
    assert calls, "--force did not recompute"


def test_run_recomputes_when_an_artifact_was_deleted(tmp_path, monkeypatch):
    config = _config_with_corpus(tmp_path)
    data = _stamped_data_dir(tmp_path, config)
    (data / "vocab.json").unlink()
    calls = _spy_phase2(monkeypatch)

    mine.run(config, skip_download=True)
    assert calls, "a missing artifact was served from the stamp"


def test_a_completed_run_stamps_itself(tmp_path, monkeypatch):
    """Second run is free only if the first one recorded its fingerprint."""
    config = _config_with_corpus(tmp_path)
    mine.run(config, skip_download=True)

    calls = _spy_phase2(monkeypatch)
    summary = mine.run(config, skip_download=True)
    assert not calls, "the second identical run re-parsed the corpus"
    assert summary["cached"] is True


def test_download_still_runs_before_the_freshness_check(tmp_path, monkeypatch):
    """The check is deliberately *after* Phase 1.  Skipping the whole stage
    when artifacts look fresh would silently stop fetching new episodes."""
    import sys
    import types

    from ptcg_mine import download as download_mod

    config = _config_with_corpus(tmp_path)
    _stamped_data_dir(tmp_path, config)

    fake_kaggle = types.ModuleType("kaggle.api.kaggle_api_extended")
    fake_kaggle.KaggleApi = lambda: types.SimpleNamespace(authenticate=lambda: None)
    monkeypatch.setitem(sys.modules, "kaggle", types.ModuleType("kaggle"))
    monkeypatch.setitem(sys.modules, "kaggle.api", types.ModuleType("kaggle.api"))
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", fake_kaggle)

    calls = []
    monkeypatch.setattr(download_mod, "download_corpus",
                        lambda cfg, api: calls.append(cfg))

    summary = mine.run(config, skip_download=False)
    assert calls, "Phase 1 was skipped along with Phase 2"
    assert summary["cached"] is True
