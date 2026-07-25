"""Tests for deck identity metadata (ptcg_il.deck)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ptcg_il.deck import (
    build_deck_metadata,
    describe,
    read_deck_metadata,
    update_sidecar,
    write_deck_csv,
)


@pytest.fixture
def data_dir(tmp_path):
    """A minimal artifact directory with two distinct archetype decks."""
    d = tmp_path / "data"
    d.mkdir()

    deck_a = [10] * 30 + [11] * 30      # archetype 0
    deck_b = [20] * 30 + [21] * 30      # archetype 2 — disjoint from deck_a
    (d / "archetypes.json").write_text(
        json.dumps(
            {
                "self_ids": [0, 2],
                "opp_ids": [0, 2],
                "fixed_deck": deck_b,
                "archetypes": [
                    {"id": 0, "representative": deck_a, "frequency": 100, "n_members": 3},
                    {"id": 2, "representative": deck_b, "frequency": 50, "n_members": 2},
                ],
            }
        )
    )
    (d / "vocab.json").write_text(
        json.dumps({"size": 5, "id_to_index": {"10": 2, "11": 3, "20": 4, "21": 5}})
    )
    pd.DataFrame(
        {
            "shard": ["train-00000.npz"] * 5 + ["val-00000.npz"] * 2,
            "archetype_self": [0, 0, 0, 2, 2, 0, 2],
        }
    ).to_parquet(d / "meta.parquet")
    return d


class TestBuildDeckMetadata:
    def test_specialist_records_its_own_deck(self, data_dir):
        m = build_deck_metadata(data_dir, archetype_self=0)
        assert m["archetype_self"] == 0
        assert m["specialist"] is True
        assert m["deck"] == [10] * 30 + [11] * 30
        assert m["deck_size"] == 60
        assert m["n_distinct_cards"] == 2
        assert m["deck_counts"] == {"10": 30, "11": 30}
        assert m["archetype_frequency"] == 100
        assert m["archetype_n_members"] == 3

    def test_generalist_falls_back_to_fixed_deck(self, data_dir):
        m = build_deck_metadata(data_dir, archetype_self=None)
        assert m["archetype_self"] is None
        assert m["specialist"] is False
        assert m["deck"] == [20] * 30 + [21] * 30
        assert m["is_fixed_deck"] is True

    def test_is_fixed_deck_flags_the_shipped_deck(self, data_dir):
        # archetype 2 *is* the fixed deck here; archetype 0 is not.
        assert build_deck_metadata(data_dir, 2)["is_fixed_deck"] is True
        assert build_deck_metadata(data_dir, 0)["is_fixed_deck"] is False

    def test_records_per_split_sample_counts(self, data_dir):
        assert build_deck_metadata(data_dir, 0)["samples"] == {"train": 3, "val": 1}
        assert build_deck_metadata(data_dir, 2)["samples"] == {"train": 2, "val": 1}

    def test_pins_artifact_hashes(self, data_dir):
        m = build_deck_metadata(data_dir, 0)
        assert len(m["vocab_sha1"]) == 12
        assert len(m["archetypes_sha1"]) == 12
        assert m["vocab_size"] == 5

    def test_vocab_hash_changes_when_vocab_changes(self, data_dir):
        before = build_deck_metadata(data_dir, 0)["vocab_sha1"]
        (data_dir / "vocab.json").write_text(json.dumps({"size": 6, "id_to_index": {"10": 2}}))
        assert build_deck_metadata(data_dir, 0)["vocab_sha1"] != before

    def test_unknown_archetype_raises(self, data_dir):
        with pytest.raises(KeyError, match="not in"):
            build_deck_metadata(data_dir, archetype_self=999)

    def test_json_serialisable(self, data_dir):
        # Must survive both torch.save and json.dump — no numpy scalars.
        json.dumps(build_deck_metadata(data_dir, 0))


class TestSidecar:
    def test_writes_keyed_by_archetype(self, data_dir, tmp_path):
        out = tmp_path / "ckpts"
        update_sidecar(out, build_deck_metadata(data_dir, 0))
        update_sidecar(out, build_deck_metadata(data_dir, 2))
        doc = json.loads((out / "decks.json").read_text())
        assert set(doc["decks"]) == {"0", "2"}
        assert doc["decks"]["0"]["deck"][0] == 10
        assert doc["decks"]["2"]["deck"][0] == 20

    def test_generalist_keyed_as_all(self, data_dir, tmp_path):
        update_sidecar(tmp_path, build_deck_metadata(data_dir, None))
        assert "all" in json.loads((tmp_path / "decks.json").read_text())["decks"]

    def test_rerun_overwrites_rather_than_accumulates(self, data_dir, tmp_path):
        update_sidecar(tmp_path, build_deck_metadata(data_dir, 0), checkpoints=["a.pt"])
        update_sidecar(tmp_path, build_deck_metadata(data_dir, 0), checkpoints=["b.pt"])
        doc = json.loads((tmp_path / "decks.json").read_text())
        assert list(doc["decks"]) == ["0"]
        assert doc["decks"]["0"]["checkpoints"] == ["b.pt"]

    def test_corrupt_sidecar_does_not_fail_the_run(self, data_dir, tmp_path):
        (tmp_path / "decks.json").write_text("{not json")
        update_sidecar(tmp_path, build_deck_metadata(data_dir, 0))
        assert "0" in json.loads((tmp_path / "decks.json").read_text())["decks"]


class TestDeckCsv:
    def test_matches_engine_format(self, data_dir, tmp_path):
        p = write_deck_csv(build_deck_metadata(data_dir, 0), tmp_path / "deck.csv")
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 60
        assert all(line.isdigit() for line in lines)
        assert lines[0] == "10"


class TestReadDeckMetadata:
    def test_roundtrips_through_torch_save(self, data_dir, tmp_path):
        torch = pytest.importorskip("torch")
        meta = build_deck_metadata(data_dir, 0)
        p = tmp_path / "ckpt.pt"
        torch.save({"step": 1, "deck": meta}, p)
        assert read_deck_metadata(p) == meta

    def test_unlabelled_checkpoint_returns_none(self, tmp_path):
        torch = pytest.importorskip("torch")
        p = tmp_path / "old.pt"
        torch.save({"step": 1}, p)
        assert read_deck_metadata(p) is None


class TestDescribe:
    def test_handles_missing_metadata(self):
        assert describe(None) == "deck=<unlabelled>"

    def test_names_the_archetype(self, data_dir):
        assert "archetype 0" in describe(build_deck_metadata(data_dir, 0))

    def test_names_generalist(self, data_dir):
        assert "all-decks" in describe(build_deck_metadata(data_dir, None))
