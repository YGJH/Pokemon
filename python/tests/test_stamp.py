"""Tests for ptcg_mine.stamp — the stage-freshness fingerprint.

The whole point of the stamp is to *not* re-run a 15-minute mine or a
35-minute shard build when nothing changed, while still catching every input
that would make the cached artifacts wrong.  So each test here breaks exactly
one input and asserts the stage goes stale; a test that only checks the happy
path would pass just as well against a `return True` implementation.
"""

import json

import pytest

from ptcg_mine import stamp


@pytest.fixture
def env(tmp_path):
    """A minimal (raw_dir, data_dir, source_root) triple with a fake source
    tree, so code-change detection can be tested without editing the repo."""
    raw = tmp_path / "raw"
    (raw / "2026-01-01").mkdir(parents=True)
    (raw / "2026-01-01" / "a.json").write_text('{"x": 1}')
    (raw / "2026-01-01" / "b.json").write_text('{"x": 22}')

    data = tmp_path / "data"
    data.mkdir()
    (data / "vocab.json").write_text('{"size": 3}')
    (data / "archetypes.json").write_text('{"self_ids": [0]}')
    (data / "card_static_table.npy").write_bytes(b"card")
    (data / "attack_static_table.npy").write_bytes(b"atk")
    (data / "engine_card_features.npy").write_bytes(b"ecf")
    (data / "engine_attack_features.npy").write_bytes(b"eaf")
    (data / "meta.parquet").write_bytes(b"meta")
    shards = data / "shards"
    shards.mkdir()
    (shards / "train-00000.npz").write_bytes(b"shard")

    src = tmp_path / "src"
    for rel in stamp.all_code_files():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {rel}\n")

    return raw, data, src


def fresh(stage, env, params=None):
    raw, data, src = env
    return stamp.check(stage, raw_dir=raw, data_dir=data,
                       params=params or {"g-min": "50"}, source_root=src)


def stamp_it(stage, env, params=None):
    raw, data, src = env
    stamp.write(stage, raw_dir=raw, data_dir=data,
                params=params or {"g-min": "50"}, source_root=src)


@pytest.mark.parametrize("stage", ["mine", "shards"])
def test_no_stamp_is_stale(stage, env):
    ok, reason = fresh(stage, env)
    assert not ok
    assert "no stamp" in reason


@pytest.mark.parametrize("stage", ["mine", "shards"])
def test_fresh_after_write(stage, env):
    stamp_it(stage, env)
    ok, reason = fresh(stage, env)
    assert ok, reason


@pytest.mark.parametrize("stage", ["mine", "shards"])
def test_new_raw_episode_is_stale(stage, env):
    raw, _, _ = env
    stamp_it(stage, env)
    (raw / "2026-01-01" / "c.json").write_text('{"x": 3}')
    ok, reason = fresh(stage, env)
    assert not ok
    assert "raw corpus" in reason


@pytest.mark.parametrize("stage", ["mine", "shards"])
def test_changed_raw_episode_size_is_stale(stage, env):
    raw, _, _ = env
    stamp_it(stage, env)
    (raw / "2026-01-01" / "a.json").write_text('{"x": 1234567}')
    ok, reason = fresh(stage, env)
    assert not ok
    assert "raw corpus" in reason


@pytest.mark.parametrize("stage", ["mine", "shards"])
def test_changed_param_is_stale(stage, env):
    stamp_it(stage, env, params={"g-min": "50"})
    ok, reason = fresh(stage, env, params={"g-min": "20"})
    assert not ok
    assert "config" in reason


@pytest.mark.parametrize("stage,code_file", [
    ("mine", "ptcg_mine/archetype.py"),
    ("shards", "ptcg_il/featurizer.py"),
])
def test_changed_code_is_stale(stage, code_file, env):
    """The failure this guards against: editing the featurizer and silently
    training on shards built by the old one (see CLAUDE.md on opt_card_id)."""
    _, _, src = env
    stamp_it(stage, env)
    (src / code_file).write_text("# edited\n")
    ok, reason = fresh(stage, env)
    assert not ok
    assert "code" in reason


def test_unrelated_code_change_does_not_invalidate_shards(env):
    """ptcg_il/cli.py grows train-only flags constantly; those must not cost a
    35-minute rebuild."""
    _, _, src = env
    stamp_it("shards", env)
    (src / "ptcg_il" / "cli.py").parent.mkdir(parents=True, exist_ok=True)
    (src / "ptcg_il" / "cli.py").write_text("# a train flag was added\n")
    ok, reason = fresh("shards", env)
    assert ok, reason


@pytest.mark.parametrize("stage,missing", [
    ("mine", "vocab.json"),
    ("mine", "card_static_table.npy"),
    ("shards", "meta.parquet"),
])
def test_missing_output_is_stale(stage, missing, env):
    _, data, _ = env
    stamp_it(stage, env)
    (data / missing).unlink()
    ok, reason = fresh(stage, env)
    assert not ok
    assert "missing output" in reason


def test_empty_shards_dir_is_stale(env):
    _, data, _ = env
    stamp_it("shards", env)
    (data / "shards" / "train-00000.npz").unlink()
    ok, reason = fresh("shards", env)
    assert not ok
    assert "missing output" in reason


def test_shards_stale_when_vocab_changes(env):
    """Mine re-running with a different vocab must force a shard rebuild:
    card ids are vocab indices, so stale shards are silently mislabelled."""
    _, data, _ = env
    stamp_it("shards", env)
    (data / "vocab.json").write_text('{"size": 999}')
    ok, reason = fresh("shards", env)
    assert not ok
    assert "upstream" in reason


@pytest.mark.parametrize(
    "table", ["engine_card_features.npy", "engine_attack_features.npy"]
)
def test_shards_stale_when_engine_feature_table_changes(env, table):
    """A rebuilt static table must force a shard rebuild.

    `shard_writer` loads these and the featurizer bakes their contents into
    every `*_card_feat` tensor.  They are not in the shards stage's `code`
    list, so without the upstream entry a `ptcg_mine/cards.py` edit invalidates
    mine, rewrites the table, and leaves 20 GB of shards holding features from
    the old one while this stage reports itself cached.
    """
    _, data, _ = env
    stamp_it("shards", env)
    (data / table).write_bytes(b"rebuilt-with-normalized-histogram")
    ok, reason = fresh("shards", env)
    assert not ok
    assert "upstream" in reason


def test_mine_requires_the_tables_it_feeds_downstream(env):
    """Mine must not stamp itself complete without the engine feature tables."""
    _, data, _ = env
    stamp_it("mine", env)
    assert fresh("mine", env)[0]
    (data / "engine_card_features.npy").unlink()
    ok, reason = fresh("mine", env)
    assert not ok
    assert "engine_card_features.npy" in reason


def test_mine_ignores_vocab_since_it_produces_it(env):
    _, data, _ = env
    stamp_it("mine", env)
    (data / "vocab.json").write_text('{"size": 999}')
    ok, _ = fresh("mine", env)
    assert ok


def test_stamp_file_is_readable_json(env):
    _, data, _ = env
    stamp_it("mine", env)
    rec = json.loads((data / ".stamp-mine.json").read_text())
    assert rec["stage"] == "mine"
    assert rec["raw"]["n_files"] == 2
    assert "written_at" in rec


def test_corrupt_stamp_is_stale_not_fatal(env):
    _, data, _ = env
    stamp_it("mine", env)
    (data / ".stamp-mine.json").write_text("{not json")
    ok, reason = fresh("mine", env)
    assert not ok
    assert "unreadable" in reason


def test_unknown_stage_raises(env):
    raw, data, src = env
    with pytest.raises(KeyError):
        stamp.check("nope", raw_dir=raw, data_dir=data, params={}, source_root=src)


# ---- CLI ----------------------------------------------------------------

def cli(stage, env, action, params=("g-min=50",)):
    raw, data, src = env
    argv = [action, "--stage", stage, "--raw-dir", str(raw),
            "--data-dir", str(data), "--source-root", str(src)]
    for p in params:
        argv += ["--param", p]
    return stamp.main(argv)


def test_cli_check_returns_stale_code_then_fresh(env, capsys):
    assert cli("mine", env, "check") == stamp.EXIT_STALE
    assert cli("mine", env, "write") == 0
    capsys.readouterr()
    assert cli("mine", env, "check") == 0
    assert "cached" in capsys.readouterr().out


def test_cli_check_prints_reason_when_stale(env, capsys):
    cli("mine", env, "write")
    capsys.readouterr()
    cli("mine", env, "check", params=("g-min=20",))
    assert "config" in capsys.readouterr().out


def test_cli_missing_raw_dir_is_stale_not_crash(env, tmp_path):
    _, data, src = env
    rc = stamp.main(["check", "--stage", "mine", "--raw-dir", str(tmp_path / "nope"),
                     "--data-dir", str(data), "--source-root", str(src)])
    assert rc == stamp.EXIT_STALE


# ---- params / canonicalisation -----------------------------------------

def test_params_from_config_reads_the_config():
    from ptcg_mine.config import MineConfig
    cfg = MineConfig(k_experts=7, g_min=33, jaccard_thresh=0.85)
    p = stamp.params_from_config("mine", cfg)
    assert p["k-experts"] == "7"
    assert p["g-min"] == "33"
    assert p["jaccard-thresh"] == "0.85"
    # Knobs that only affect the download must stay out: they change raw/, and
    # raw/ is already fingerprinted.
    assert "n-days" not in p and "target-episodes" not in p


def test_params_from_config_takes_extras():
    from ptcg_mine.config import MineConfig
    p = stamp.params_from_config("shards", MineConfig(), **{"samples-per-shard": 100})
    assert p["samples-per-shard"] == "100"


def test_numerically_equal_params_are_not_a_change(env):
    """`--jaccard-thresh 0.90` and `0.9` are the same run; re-formatting the
    invocation must not cost a 35-minute rebuild."""
    stamp_it("mine", env, params={"jaccard-thresh": "0.90", "g-min": "50"})
    ok, reason = fresh("mine", env, params={"jaccard-thresh": "0.9", "g-min": "50.0"})
    assert ok, reason


def test_different_numeric_params_still_stale(env):
    stamp_it("mine", env, params={"jaccard-thresh": "0.90"})
    ok, _ = fresh("mine", env, params={"jaccard-thresh": "0.91"})
    assert not ok


# ---- summary round-trip -------------------------------------------------

def test_summary_round_trips(env):
    raw, data, src = env
    stamp.write("mine", raw_dir=raw, data_dir=data, params={}, source_root=src,
                summary={"vocab_size": 296, "self_ids": [0, 1]})
    rec = stamp.read("mine", data)
    assert rec["summary"] == {"vocab_size": 296, "self_ids": [0, 1]}


def test_require_summary_rejects_a_stamp_without_one(env):
    raw, data, src = env
    stamp.write("mine", raw_dir=raw, data_dir=data, params={}, source_root=src)
    ok, reason = stamp.check("mine", raw_dir=raw, data_dir=data, params={},
                             source_root=src, require_summary=True)
    assert not ok
    assert "summary" in reason
    # Without the flag the same stamp is perfectly usable.
    ok, _ = stamp.check("mine", raw_dir=raw, data_dir=data, params={}, source_root=src)
    assert ok


def test_read_missing_stamp_is_none(env):
    _, data, _ = env
    assert stamp.read("mine", data) is None


def test_source_root_defaults_to_the_package(env):
    """Callers inside the package should not have to locate their own source."""
    raw, data, _ = env
    stamp.write("mine", raw_dir=raw, data_dir=data, params={})
    ok, reason = stamp.check("mine", raw_dir=raw, data_dir=data, params={})
    assert ok, reason
