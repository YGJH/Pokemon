"""Pipeline-level pieces: baselines, archetype selection, checkpoint config.

These guard the seams between stages, which is where the pipeline's silent
failures live: a baseline recorded from one model and compared against another, a
hardcoded archetype id that no longer exists, a checkpoint whose architecture has
to be guessed.
"""

from __future__ import annotations

import json

import pytest

from ptcg_il.archetype_select import MIN_TEST_ROWS, MIN_VAL_ROWS, pick_archetypes, split_of_shard
from ptcg_il.baselines import (
    BaselineMismatch,
    checkpoint_sha1,
    get_baseline,
    load_baselines,
    record_baseline,
)

pd = pytest.importorskip("pandas")


# ── Baselines ───────────────────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path):
    ckpt = tmp_path / "ckpt-best.pt"
    ckpt.write_bytes(b"pretend-these-are-weights")
    return tmp_path


class TestBaselines:
    def test_roundtrip(self, data_dir):
        ckpt = data_dir / "ckpt-best.pt"
        record_baseline(data_dir, 2, ckpt, {
            "val/top1_nontrivial": 0.7538,
            "val/top1_macro": 0.83,
            "val/value_corr": 0.045,
            "val/value_std": 0.0006,
        })
        rec = get_baseline(data_dir, 2, ckpt)
        assert rec["nontrivial_top1"] == pytest.approx(0.7538)
        assert rec["value_corr"] == pytest.approx(0.045)

    def test_a_changed_checkpoint_is_rejected(self, data_dir):
        """The whole point: a baseline must describe the model it is compared to.

        RL_SPEC §13 lists stale baselines as high-severity *and silent* — a stale
        threshold does not raise on its own, it quietly passes or fails the wrong
        candidate.
        """
        ckpt = data_dir / "ckpt-best.pt"
        record_baseline(data_dir, 2, ckpt, {"val/top1_nontrivial": 0.75})
        ckpt.write_bytes(b"different-weights-entirely")
        with pytest.raises(BaselineMismatch, match="does not describe this checkpoint"):
            get_baseline(data_dir, 2, ckpt)

    def test_missing_archetype_raises(self, data_dir):
        with pytest.raises(FileNotFoundError, match="no IL baseline"):
            get_baseline(data_dir, 99, None)

    def test_recording_one_archetype_preserves_the_others(self, data_dir):
        """Training a second specialist must not erase the first's baseline."""
        a = data_dir / "a.pt"
        a.write_bytes(b"model-a")
        b = data_dir / "b.pt"
        b.write_bytes(b"model-b")

        record_baseline(data_dir, 0, a, {"val/top1_nontrivial": 0.53})
        record_baseline(data_dir, 1, b, {"val/top1_nontrivial": 0.75})

        all_records = load_baselines(data_dir)
        assert set(all_records) == {"0", "1"}
        assert get_baseline(data_dir, 0, a)["nontrivial_top1"] == pytest.approx(0.53)

    def test_keys_are_strings_so_json_roundtrips(self, data_dir):
        """JSON object keys are strings; an int lookup would miss after reload."""
        ckpt = data_dir / "ckpt-best.pt"
        record_baseline(data_dir, 11, ckpt, {"val/top1_nontrivial": 0.6})
        raw = json.loads((data_dir / "il_baselines.json").read_text())
        assert "11" in raw
        # Both spellings must resolve through the accessor.
        assert get_baseline(data_dir, 11, ckpt) == get_baseline(data_dir, "11", ckpt)

    def test_generalist_key(self, data_dir):
        ckpt = data_dir / "ckpt-best.pt"
        record_baseline(data_dir, None, ckpt, {"val/top1_nontrivial": 0.4})
        assert "generalist" in load_baselines(data_dir)

    def test_sha1_is_stable_and_content_dependent(self, tmp_path):
        f = tmp_path / "x.pt"
        f.write_bytes(b"abc" * 1000)
        first = checkpoint_sha1(f)
        assert first == checkpoint_sha1(f)
        f.write_bytes(b"abd" * 1000)
        assert checkpoint_sha1(f) != first

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_baselines(tmp_path) == {}


# ── Archetype selection ─────────────────────────────────────────────────────


def _write_corpus(tmp_path, self_ids, counts):
    """A minimal archetypes.json + meta.parquet pair.

    *counts* maps ``archetype_id -> (train, val, test)``.
    """
    (tmp_path / "archetypes.json").write_text(json.dumps({
        "self_ids": self_ids,
        "opp_ids": self_ids,
        "fixed_deck": list(range(60)),
        "archetypes": [{"id": i, "representative": list(range(60))} for i in self_ids],
    }))

    rows = []
    for aid, (n_train, n_val, n_test) in counts.items():
        for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
            rows.extend({"shard": f"{split}-00000", "archetype_self": aid} for _ in range(n))
    pd.DataFrame(rows).to_parquet(tmp_path / "meta.parquet", index=False)
    return tmp_path


class TestArchetypeSelection:
    def test_ranks_by_training_rows(self, tmp_path):
        d = _write_corpus(tmp_path, [0, 1, 5], {
            0: (1000, 200, 200), 1: (5000, 200, 200), 5: (300, 200, 200),
        })
        assert pick_archetypes(d, top=2) == [1, 0]

    def test_excludes_archetypes_that_cannot_be_held_out(self, tmp_path):
        """RL_SPEC §14.1 records archetype 20 having zero val rows.

        Such an archetype cannot be model-selected or gated, so training a
        specialist for it produces a checkpoint nobody can evaluate.
        """
        d = _write_corpus(tmp_path, [0, 7], {
            0: (1000, 200, 200),
            7: (99999, MIN_VAL_ROWS - 1, MIN_TEST_ROWS - 1),
        })
        picked = pick_archetypes(d, top=2)
        assert picked == [0], f"an un-evaluable archetype was selected: {picked}"

    def test_returns_fewer_than_requested_rather_than_padding(self, tmp_path):
        d = _write_corpus(tmp_path, [0, 3], {0: (1000, 200, 200), 3: (10, 1, 1)})
        assert pick_archetypes(d, top=4) == [0]

    def test_ignores_ids_absent_from_the_corpus(self, tmp_path):
        """self_ids may list an archetype with no surviving rows after filtering."""
        d = _write_corpus(tmp_path, [0, 2], {0: (500, 200, 200)})
        assert pick_archetypes(d, top=2) == [0]

    def test_missing_artifacts_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="archetypes.json"):
            pick_archetypes(tmp_path, top=2)

    @pytest.mark.parametrize("shard,expected", [
        ("train-00000", "train"), ("val-00012", "val"), ("test-00003", "test"),
    ])
    def test_split_is_parsed_from_the_shard_name(self, shard, expected):
        """meta.parquet has no split column; the split lives in the shard name."""
        assert split_of_shard(shard) == expected

    def test_real_corpus_has_no_archetype_2(self, tmp_path):
        """A regression pin for the stale hardcoded default.

        ``run_pipeline.sh`` shipped ``ARCHETYPES="0 2"`` while this corpus's
        self_ids are [0, 1, 11, 3, 4, 5].  Training archetype 2 fails outright
        with "No samples for archetype_self=2" — which is the *good* outcome;
        the bad one is a re-mine that reassigns 2 to an unrelated cluster.
        """
        from pathlib import Path

        data = Path(__file__).resolve().parents[1] / "data"
        if not (data / "archetypes.json").exists():
            pytest.skip("no mined corpus in python/data/")
        picked = pick_archetypes(data, top=2)
        assert picked, "no usable archetype in the real corpus"
        assert all(isinstance(a, int) for a in picked)


# ── Checkpoint architecture record ──────────────────────────────────────────


class TestCheckpointConfig:
    def test_policy_records_its_own_config(self):
        torch = pytest.importorskip("torch")
        from ptcg_il.model.policy import Policy

        from ptcg_il.model.policy import current_feature_dims

        p = Policy(D=32, heads=4, layers=1, ff=64, n_opp_arch=6)
        # No V/A: cards are static features, so there is no vocab width to record.
        arch = {k: v for k, v in p.config.items() if k != "feat_dims"}
        assert arch == {"D": 32, "heads": 4, "layers": 1, "ff": 64,
                        "n_opp_arch": 6, "n_all_cards": 0, "seed": 42}
        # ...plus the featurizer widths the weights were shaped by, so a
        # featurizer edit cannot silently redefine the checkpoint.
        assert p.config["feat_dims"] == current_feature_dims()
        del torch

    def test_config_roundtrips_through_policy_from_config(self):
        pytest.importorskip("torch")
        from ptcg_il.model.policy import Policy, load_policy_state, policy_from_config

        original = Policy(D=32, heads=4, layers=1, ff=64, n_opp_arch=6)
        rebuilt = policy_from_config(original.config)
        # A strict-enough load: only belief keys may be missing, and here none are.
        assert load_policy_state(rebuilt, original.state_dict()) == []

    def test_missing_config_raises_rather_than_guessing(self):
        """A guessed n_opp_arch would not raise — it would just be wrong.

        ``load_policy_state`` forgives missing ``belief_heads.*`` keys, so a
        wrong belief width yields a plausible model with randomly-initialised
        heads instead of an error.
        """
        pytest.importorskip("torch")
        from ptcg_il.model.policy import policy_from_config

        with pytest.raises(KeyError, match="predates Policy.config"):
            policy_from_config({"V": 40})
