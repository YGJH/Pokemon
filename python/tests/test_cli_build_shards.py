"""Tests for the build-shards CLI's skip-when-unchanged behaviour.

Featurizing the corpus takes ~35 min, and the result is a pure function of
(corpus, config, featurizer/selection code, vocab.json + archetypes.json).  The
skip therefore has to fire on an unchanged run and *not* fire when any of those
moved — stale shards are silently wrong rather than loud, because card ids are
vocab indices.
"""

import argparse

import pytest

from ptcg_il import cli
from ptcg_mine import stamp


def _args(tmp_path, force=False, samples_per_shard=50000):
    return argparse.Namespace(
        raw_dir=str(tmp_path / "raw"),
        out_dir=str(tmp_path / "data"),
        k_experts=10,
        g_min=50,
        jaccard_thresh=0.90,
        samples_per_shard=samples_per_shard,
        force=force,
    )


@pytest.fixture
def corpus(tmp_path):
    """raw/ + data/ populated as they are *after* a successful shard build."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ep.json").write_text('{"x": 1}')

    data = tmp_path / "data"
    (data / "shards").mkdir(parents=True)
    (data / "shards" / "train-00000.npz").write_bytes(b"shard")
    (data / "meta.parquet").write_bytes(b"meta")
    (data / "vocab.json").write_text('{"size": 296}')
    (data / "archetypes.json").write_text('{"self_ids": [0]}')
    return tmp_path


def _stamp_it(tmp_path, samples_per_shard=50000, summary=None):
    from ptcg_mine.config import MineConfig

    config = MineConfig(
        raw_dir=tmp_path / "raw",
        out_dir=tmp_path / "data",
        k_experts=10,
        g_min=50,
        jaccard_thresh=0.90,
    )
    stamp.write(
        "shards",
        raw_dir=config.raw_dir,
        data_dir=config.out_dir,
        params=stamp.params_from_config(
            "shards", config, **{"samples-per-shard": samples_per_shard}
        ),
        summary=summary or {"total_samples": 187068, "n_shards": 6,
                            "n_kept_games": 900, "n_loaded": 9910,
                            "n_invalid": 10, "split_counts": {"train": 152780},
                            "meta_path": str(config.out_dir / "meta.parquet")},
    )


@pytest.fixture
def spy(monkeypatch):
    """Record calls into the real builder without running it."""
    from ptcg_il import shard_writer

    calls = []

    def fake(config, samples_per_shard=50000, jobs=None):
        calls.append(samples_per_shard)
        return {"total_samples": 10, "n_shards": 1, "n_kept_games": 2,
                "n_loaded": 3, "n_invalid": 0, "split_counts": {"train": 10},
                "meta_path": "meta.parquet"}

    monkeypatch.setattr(shard_writer, "build_shards", fake)
    return calls


def test_skips_when_nothing_changed(corpus, spy):
    _stamp_it(corpus)
    assert cli.cmd_build_shards(_args(corpus)) == 0
    assert not spy, "rebuilt despite a fresh stamp"


def test_builds_when_no_stamp(corpus, spy):
    assert cli.cmd_build_shards(_args(corpus)) == 0
    assert spy


def test_force_rebuilds(corpus, spy):
    _stamp_it(corpus)
    assert cli.cmd_build_shards(_args(corpus, force=True)) == 0
    assert spy


def test_rebuilds_when_vocab_changed(corpus, spy):
    """Mine producing a new vocab must invalidate the shards built on the old
    one: the ids inside them are indices into that vocab."""
    _stamp_it(corpus)
    (corpus / "data" / "vocab.json").write_text('{"size": 401}')
    assert cli.cmd_build_shards(_args(corpus)) == 0
    assert spy


def test_rebuilds_when_corpus_changed(corpus, spy):
    _stamp_it(corpus)
    (corpus / "raw" / "ep2.json").write_text('{"x": 2}')
    assert cli.cmd_build_shards(_args(corpus)) == 0
    assert spy


def test_rebuilds_when_samples_per_shard_changed(corpus, spy):
    """It changes the shard layout meta.parquet rows point into."""
    _stamp_it(corpus, samples_per_shard=50000)
    assert cli.cmd_build_shards(_args(corpus, samples_per_shard=10000)) == 0
    assert spy


def test_rebuilds_when_shards_deleted(corpus, spy):
    _stamp_it(corpus)
    (corpus / "data" / "shards" / "train-00000.npz").unlink()
    assert cli.cmd_build_shards(_args(corpus)) == 0
    assert spy


def test_successful_build_stamps_itself(corpus, spy):
    assert cli.cmd_build_shards(_args(corpus)) == 0
    assert spy
    spy.clear()
    assert cli.cmd_build_shards(_args(corpus)) == 0
    assert not spy, "the second identical build did not reuse the first"


def test_empty_build_is_not_stamped(corpus, monkeypatch):
    """A build that produced nothing must not bless the directory — the next
    run has to try again rather than report 0 samples forever."""
    from ptcg_il import shard_writer

    monkeypatch.setattr(
        shard_writer, "build_shards",
        lambda config, samples_per_shard=50000, jobs=None: {
            "total_samples": 0, "n_shards": 0, "n_kept_games": 0, "n_loaded": 0,
            "n_invalid": 0, "split_counts": {}, "meta_path": "meta.parquet"},
    )
    assert cli.cmd_build_shards(_args(corpus)) == 1
    assert stamp.read("shards", corpus / "data") is None
