"""Loader locality: the per-archetype dense repack.

Specialist training reads a few percent of the rows of every shard it touches,
so its working set is the whole split -- ~70 GB of decompressed shards traversed
to consume 3.5 GB of rows, on 30 GiB of RAM.  The repack copies just those rows
into dense shards once, which is what lets the working set be cached at all and
what makes the generalist mmap cache unnecessary for a specialist run.

Its correctness requirement is absolute and its failure mode is silent: the
repack is a pure rearrange, so an index bug does not raise, it just trains the
model on the wrong labels.  Hence a byte-for-byte equality test against the
originals, and a runtime check inside the builder itself.

(``MADV_RANDOM`` was measured here too and rejected -- slower in every paired
comparison.  See the note in ``dataset.py``.)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptcg_il.train import dataset as ds_mod
from ptcg_il.train.dataset import ShardDataset, repack_dirname
from tests.test_dataset import _make_synthetic_sample

# ============================================================
# Fixture — a corpus with a sparse archetype
# ============================================================

#: The deck under test.  Deliberately not 0/1/2: an off-by-one that fell back to
#: "keep everything" would still produce plausible counts for a small id.
ARCH = 7


def _build_sparse_corpus(
    n_shards: int = 3,
    rows_per_shard: int = 100,
    density: float = 0.1,
    n_val_shards: int = 1,
    seed: int = 0,
) -> tuple[Path, int]:
    """Write ``n_shards`` shards where only ``density`` of rows are ``ARCH``.

    Returns ``(data_dir, n_arch_rows_in_train)``.
    """
    rng = np.random.default_rng(seed)
    data_dir = Path(tempfile.mkdtemp())
    shards_dir = data_dir / "shards"
    shards_dir.mkdir(parents=True)

    meta_rows: list[dict] = []
    n_arch_train = 0
    for split, count in (("train", n_shards), ("val", n_val_shards)):
        for shard_i in range(count):
            shard_name = f"{split}-{shard_i:05d}.npz"
            samples = []
            for j in range(rows_per_shard):
                is_arch = rng.random() < density
                arch = ARCH if is_arch else int(rng.integers(0, 4))
                if is_arch and split == "train":
                    n_arch_train += 1
                won = (shard_i + j) % 2 == 0
                samples.append(
                    _make_synthetic_sample(
                        sel_ctx=j % 5,
                        archetype_self=arch,
                        won=won,
                        value_target=1.0 if won else -1.0,
                    )
                )
                meta_rows.append({
                    "sample_uid": f"{shard_name}:{j}",
                    "shard": shard_name,
                    "row": j,
                    "episode_id": f"ep_{shard_i}_{j}",
                    "player": 0,
                    "team": "t",
                    "archetype_self": arch,
                    "archetype_opp": 0,
                    "sel_type": int(samples[-1]["sel_type"]),
                    "sel_ctx": int(samples[-1]["sel_ctx"]),
                    "minCount": 1,
                    "maxCount": 1,
                    "won": won,
                })
            keys = sorted(samples[0].keys())
            np.savez_compressed(
                shards_dir / shard_name,
                **{k: np.stack([s[k] for s in samples], axis=0) for k in keys},
            )

    pd.DataFrame(meta_rows).to_parquet(data_dir / "meta.parquet", index=False)
    return data_dir, n_arch_train


def _original_rows(data_dir: Path, uids: list[str]) -> dict[str, list[np.ndarray]]:
    """Read the named samples straight from the original .npz shards."""
    meta = pd.read_parquet(data_dir / "meta.parquet").set_index("sample_uid")
    out: dict[str, list[np.ndarray]] = {}
    for uid in uids:
        row = meta.loc[uid]
        with np.load(data_dir / "shards" / str(row["shard"])) as z:
            for key in z.files:
                out.setdefault(key, []).append(z[key][int(row["row"])].copy())
    return out


# ============================================================
# Repack correctness
# ============================================================


class TestRepackCorrectness:
    def test_every_repacked_row_equals_the_original(self):
        """The repack is a pure rearrange; any value change is a silent relabel."""
        data_dir, n_arch = _build_sparse_corpus()
        ds = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        assert len(ds) == n_arch > 0

        uids = [str(u) for u in ds.meta["sample_uid"]]
        expected = _original_rows(data_dir, uids)
        n_checked = 0
        for i in range(len(ds)):
            raw = ds.get_raw(i)
            pos = uids.index(str(ds.meta.iloc[int(ds._indices[i])]["sample_uid"]))
            for key, got in raw.items():
                np.testing.assert_array_equal(got, expected[key][pos], err_msg=key)
                assert got.dtype == expected[key][pos].dtype, key
                n_checked += 1
        assert n_checked > 0

    def test_gather_block_follows_meta_order_not_shard_order(self):
        """Output position i is ``names[i]``/``rows[i]``, whatever the order.

        ``_adopt_repack`` recovers the dense ``(shard, row)`` pair by arithmetic
        on the meta index, which is only valid if the block preserves that
        order.  A corpus's meta happens to be shard-major and row-ascending, so
        a gather that quietly assumed sorted input would pass every end-to-end
        test here and break on the first meta that is not.
        """
        data_dir, _ = _build_sparse_corpus(n_shards=3, rows_per_shard=20, density=1.0)
        shards_dir = Path(data_dir) / "shards"
        with np.load(shards_dir / "train-00000.npz") as z:
            keys = sorted(z.files)

        names = np.array(
            ["train-00002.npz", "train-00000.npz", "train-00002.npz",
             "train-00001.npz", "train-00000.npz"]
        )
        rows = np.array([17, 3, 4, 11, 19], dtype=np.int64)
        block = ds_mod._gather_block(shards_dir, keys, names, rows)

        for i, (name, row) in enumerate(zip(names, rows)):
            with np.load(shards_dir / name) as z:
                for key in keys:
                    np.testing.assert_array_equal(
                        block[key][i], z[key][int(row)], err_msg=f"{key}@{i}"
                    )

    def test_repack_spans_multiple_dense_shards(self):
        """Exercise the shard/row arithmetic, not just a single-shard corpus."""
        data_dir, n_arch = _build_sparse_corpus(
            n_shards=6, rows_per_shard=20, density=0.35
        )
        ds = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        assert n_arch > 20, "fixture must overflow one dense shard"
        assert ds.meta["shard"].nunique() > 1
        # Dense: rows are consecutive from 0 within each shard.
        for _, grp in ds.meta.groupby("shard"):
            assert sorted(grp["row"]) == list(range(len(grp)))

    def test_dense_shards_hold_only_this_archetype(self):
        data_dir, n_arch = _build_sparse_corpus()
        ds = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        total = 0
        for shard_name in sorted(set(ds.meta["shard"].astype(str))):
            d = Path(data_dir) / "shards" / shard_name
            total += int(np.load(d / "tok_mask.npy", mmap_mode="r").shape[0])
        assert total == n_arch

    def test_samples_per_shard_inherited_from_the_source_shards(self):
        data_dir, _ = _build_sparse_corpus(
            n_shards=6, rows_per_shard=20, density=0.35
        )
        ds = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        stamp = json.loads((ds._repack_dir / "_source.json").read_text())
        assert stamp["samples_per_shard"] == 20

    def test_meta_points_at_the_repack_directory(self):
        data_dir, _ = _build_sparse_corpus()
        ds = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        prefix = repack_dirname(ARCH, "train")
        assert ds.meta["shard"].str.startswith(prefix + "/").all()

    def test_samples_are_still_well_formed(self):
        data_dir, _ = _build_sparse_corpus()
        ds = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        sample = ds[0]
        assert sample["opt_mask"].shape[0] > 0
        assert sample["sample_weight"].ndim == 0
        assert "value_target" in sample

    def test_generalist_does_not_repack(self):
        data_dir, _ = _build_sparse_corpus()
        ds = ShardDataset(data_dir, split="train", archetype_self=None)
        assert ds._repack_dir is None
        assert ds.meta["shard"].str.endswith(".npz").all()
        assert not list((Path(data_dir) / "shards").glob("*repack*"))

    def test_train_and_val_repack_independently(self):
        data_dir, _ = _build_sparse_corpus()
        train = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        val = ShardDataset(data_dir, split="val", archetype_self=ARCH)
        assert train._repack_dir != val._repack_dir
        assert len(val) > 0
        assert val.meta["sample_uid"].str.startswith("val-").all()
        assert train.meta["sample_uid"].str.startswith("train-").all()


# ============================================================
# Reuse, invalidation, fallback
# ============================================================


class TestRepackReuse:
    def _probe(self, ds: ShardDataset) -> Path:
        return next(ds._repack_dir.glob("*/tok_mask.npy"))

    def test_reused_when_nothing_changed(self):
        data_dir, _ = _build_sparse_corpus()
        first = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        before = self._probe(first).stat().st_mtime_ns
        second = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        assert self._probe(second).stat().st_mtime_ns == before

    def test_reuse_yields_identical_samples(self):
        data_dir, _ = _build_sparse_corpus()
        first = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        rows_a = [first.get_raw(i) for i in range(len(first))]
        second = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        assert [str(u) for u in second.meta["sample_uid"]] == [
            str(u) for u in first.meta["sample_uid"]
        ]
        for a, b in zip(rows_a, (second.get_raw(i) for i in range(len(second)))):
            assert a.keys() == b.keys()
            for k in a:
                np.testing.assert_array_equal(a[k], b[k], err_msg=k)

    def test_rebuilt_when_a_source_shard_changes(self):
        """Card ids are vocab indices; a stale dense copy is silently wrong."""
        data_dir, _ = _build_sparse_corpus()
        first = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        before = first.get_raw(0)["poke_card_id"].copy()
        uid = str(first.meta.iloc[int(first._indices[0])]["sample_uid"])
        src_shard = uid.split(":")[0]

        path = Path(data_dir) / "shards" / src_shard
        with np.load(path) as z:
            arrays = {k: z[k].copy() for k in z.files}
        arrays["poke_card_id"] = arrays["poke_card_id"] + 1
        np.savez_compressed(path, **arrays)

        second = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        assert str(second.meta.iloc[int(second._indices[0])]["sample_uid"]) == uid
        np.testing.assert_array_equal(second.get_raw(0)["poke_card_id"], before + 1)

    def test_partial_repack_is_rebuilt(self):
        """An interrupted build leaves a dir that exists but is missing keys."""
        data_dir, _ = _build_sparse_corpus()
        first = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        self._probe(first).unlink()
        second = ShardDataset(data_dir, split="train", archetype_self=ARCH)
        assert self._probe(second).exists()
        assert second[0]["tok_mask"].shape[0] > 0

    def test_repack_does_not_build_the_source_mmap_cache(self):
        """The point of reading .npz directly: never materialise the full split."""
        data_dir, _ = _build_sparse_corpus()
        ShardDataset(data_dir, split="train", archetype_self=ARCH)
        assert not (Path(data_dir) / "shards" / ".mmap-cache").exists()

    def test_falls_back_when_the_repack_cannot_be_written(self):
        """A read-only corpus must still train, just without the locality win."""
        data_dir, n_arch = _build_sparse_corpus()
        shards_dir = Path(data_dir) / "shards"
        mode = shards_dir.stat().st_mode
        shards_dir.chmod(0o555)
        try:
            ds = ShardDataset(data_dir, split="train", archetype_self=ARCH)
            assert ds._repack_dir is None
            assert len(ds) == n_arch
            assert ds[0]["tok_mask"].shape[0] > 0
        finally:
            shards_dir.chmod(mode)

    def test_absent_deck_still_raises_before_repacking(self):
        data_dir, _ = _build_sparse_corpus()
        with pytest.raises(ValueError, match="No samples for archetype_self=99"):
            ShardDataset(data_dir, split="train", archetype_self=99)
        assert not list((Path(data_dir) / "shards").glob("*repack*"))
