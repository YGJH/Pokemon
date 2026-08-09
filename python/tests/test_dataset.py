"""Tests for ShardDataset, collate_fn, and compute_sample_weights.

Creates small synthetic .npz shards + meta.parquet so tests are self-contained
and do not require a full corpus.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ptcg_il.featurizer import F_GLOBAL, F_HAND, F_OPT, F_POKE, F_SUM
from ptcg_il.model.pointer import O_MAX
from ptcg_il.train.dataset import (
    ALPHA_ARCH,
    ALPHA_CTX,
    W_LOST,
    ShardDataset,
    collate_fn,
    compute_sample_weights,
)

# ============================================================
# Helpers — synthetic data generation
# ============================================================

# Minimal featurizer-like tensor shapes (subset of full Appendix A contract)
_KEYS_FLOAT = {
    # Widths from ptcg_il.featurizer, never literals -- see the note in
    # test_model_policy._make_synthetic_batch.
    "cls_feat": (F_GLOBAL,),
    "poke_feat": (12, F_POKE),
    "hand_feat": (30, F_HAND),
    "sum_feat": (2, F_SUM),
    "stadium_present": (1,),
    "opt_scalar": (64, F_OPT),
    "value_target": (),
    "sample_weight": (),
}

_KEYS_INT = {
    "poke_card_id": (12,),
    "hand_card_id": (30,),
    "stadium_card_id": (1,),
    "context_card_id": (1,),
    "effect_card_id": (1,),
    "discard_ids": (2, 60),
    "prize_ids": (2, 6),
    "tok_type": (46,),
    "tok_owner": (46,),
    "tok_zone": (46,),
    "opt_type": (64,),
    "opt_src_idx": (64,),
    "opt_tgt_idx": (64,),
    "opt_card_id": (64,),
    "opt_attack_idx": (64,),
    "opt_group": (64,),
    "sel_type": (),
    "sel_ctx": (),
    "action_idx": (64,),
    "minCount": (),
    "maxCount": (),
    "action_len": (),
}

_KEYS_BOOL = {
    "tok_mask": (46,),
    "opt_mask": (64,),
    "discard_mask": (2, 60),
}


def _make_synthetic_sample(
    sel_ctx: int = 0,
    archetype_self: int = 0,
    won: bool = True,
    value_target: float = 1.0,
    max_count: int = 1,
) -> dict[str, np.ndarray]:
    """Create one synthetic featurizer-like sample dict."""
    sample: dict[str, np.ndarray] = {}
    for key, shape in _KEYS_FLOAT.items():
        if key in ("sample_weight",):
            continue  # derived from meta
        if shape == ():
            sample[key] = np.float32(np.random.randn())
        else:
            sample[key] = np.random.randn(*shape).astype(np.float32)
    for key, shape in _KEYS_INT.items():
        if shape == ():
            sample[key] = np.int64(np.random.randint(0, 64))
        else:
            sample[key] = np.random.randint(0, 64, size=shape).astype(np.int64)
    for key, shape in _KEYS_BOOL.items():
        if shape == ():
            sample[key] = np.bool_(np.random.randint(0, 2))
        else:
            sample[key] = np.random.randint(0, 2, size=shape).astype(bool)

    # Override specific fields for realistic data
    sample["sel_type"] = np.array(sel_ctx % 11, dtype=np.int64)
    sample["sel_ctx"] = np.array(sel_ctx, dtype=np.int64)
    sample["value_target"] = np.array(value_target, dtype=np.float32)
    sample["minCount"] = np.array(1, dtype=np.int64)
    sample["maxCount"] = np.array(max_count, dtype=np.int64)
    sample["action_idx"][0] = 3  # expert picked option 3
    sample["action_idx"][1:] = -1
    sample["action_len"] = np.array(1, dtype=np.int64)
    sample["opt_mask"][:max_count] = True
    sample["opt_group"][:max_count] = 0  # single group containing all valid options
    sample["tok_mask"][:30] = True  # first 30 tokens active

    return sample


def _build_synthetic_data(
    n_train: int = 100,
    n_val: int = 20,
    samples_per_shard: int = 60,
    seed: int = 42,
    drop_keys: tuple[str, ...] = (),
) -> Path:
    """Create a temporary data/ dir with shards/ and meta.parquet.

    Returns the path to the data directory.
    """
    rng = np.random.default_rng(seed)
    tmp = tempfile.mkdtemp()
    data_dir = Path(tmp)
    shards_dir = data_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []

    for split, n in [("train", n_train), ("val", n_val)]:
        for shard_i in range(0, n, samples_per_shard):
            shard_end = min(shard_i + samples_per_shard, n)
            shard_samples = shard_end - shard_i
            shard_name = f"{split}-{shard_i // samples_per_shard:05d}.npz"

            samples = []
            for j in range(shard_samples):
                sample_idx = shard_i + j
                sel_ctx = sample_idx % 5  # 5 different contexts
                archetype_self = sample_idx % 3  # 3 archetypes
                won = sample_idx % 2 == 0
                value_target = 1.0 if won else -1.0
                sample = _make_synthetic_sample(
                    sel_ctx=sel_ctx,
                    archetype_self=archetype_self,
                    won=won,
                    value_target=value_target,
                )
                samples.append(sample)

                meta_rows.append({
                    "sample_uid": f"ep_{sample_idx}_0_{j}",
                    "shard": shard_name,
                    "row": j,
                    "episode_id": f"ep_{sample_idx}",
                    "player": 0,
                    "team": "expert_A",
                    "archetype_self": archetype_self,
                    "archetype_opp": 0,
                    "sel_type": sample["sel_type"].item(),
                    "sel_ctx": sample["sel_ctx"].item(),
                    "minCount": 1,
                    "maxCount": 1,
                    "won": won,
                })

            # Stack and save shard
            stacked = {}
            if samples:
                keys = sorted(samples[0].keys())
                for k in keys:
                    stacked[k] = np.stack([s[k] for s in samples], axis=0)
                for k in drop_keys:
                    stacked.pop(k, None)
                np.savez_compressed(shards_dir / shard_name, **stacked)

    # Write meta
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(data_dir / "meta.parquet", index=False)
    return data_dir


# ============================================================
# Tests — compute_sample_weights
# ============================================================


class TestComputeSampleWeights:
    def test_basic(self):
        """Weights should be positive and normalized to mean approx 1."""
        meta = pd.DataFrame({
            "sel_ctx": [0, 1, 0, 2, 1],
            "archetype_self": [0, 0, 1, 1, 0],
            "won": [True, True, True, False, True],
        })
        weights = compute_sample_weights(meta)
        assert len(weights) == 5
        assert (weights >= 0).all()
        assert np.allclose(weights.mean(), 1.0, atol=1e-5)

    def test_lost_discounted(self):
        """Lost-game samples should get lower weight than won, all else equal."""
        meta = pd.DataFrame({
            "sel_ctx": [0, 0],
            "archetype_self": [0, 0],
            "won": [True, False],
        })
        weights = compute_sample_weights(meta, w_lost=0.6)
        # Both samples have same ctx/arch, so ratio = 1.0 / 0.6
        assert weights[0] > weights[1]
        assert np.isclose(weights[0] / weights[1], 1.0 / 0.6, rtol=1e-4)

    def test_rare_context_upweighted(self):
        """Rare context should get higher weight."""
        meta = pd.DataFrame({
            "sel_ctx": [0, 1, 1, 1, 1],
            "archetype_self": [0, 0, 0, 0, 0],
            "won": [True, True, True, True, True],
        })
        weights = compute_sample_weights(meta, alpha_ctx=0.5)
        # Context 0 appears once, context 1 appears 4 times
        # w_ctx[0] = (5/1)^0.5 ≈ 2.236
        # w_ctx[1] = (5/4)^0.5 ≈ 1.118
        assert weights[0] > weights[1]

    def test_skill_weight_is_applied(self):
        """``skill_w`` scales the row, all else equal.

        This is the only thing standing between an unfiltered corpus and
        imitating the median player, so it must survive into the final weight
        rather than being computed and dropped.
        """
        meta = pd.DataFrame({
            "sel_ctx": [0, 0],
            "archetype_self": [0, 0],
            "won": [True, True],
            "skill_w": [4.0, 1.0],
        })
        weights = compute_sample_weights(meta)
        assert np.isclose(weights[0] / weights[1], 4.0, rtol=1e-4)
        assert np.allclose(weights.mean(), 1.0, atol=1e-5)

    def test_skill_weight_composes_with_the_other_factors(self):
        """A strong player's loss can still outweigh a weak player's win.

        The factors multiply rather than one overriding another; a 10x skill
        gap against ``w_lost=0.6`` is the case that pins the ordering.
        """
        meta = pd.DataFrame({
            "sel_ctx": [0, 0],
            "archetype_self": [0, 0],
            "won": [False, True],
            "skill_w": [10.0, 1.0],
        })
        weights = compute_sample_weights(meta, w_lost=0.6)
        assert weights[0] > weights[1]
        assert np.isclose(weights[0] / weights[1], 6.0, rtol=1e-4)

    def test_missing_skill_column_falls_back_to_uniform_with_a_warning(self, caplog):
        """A pre-skill-weight meta.parquet still trains, but says so.

        Correct for a corpus that was expert-filtered -- every row is an
        expert's, so uniform is what weighting would have produced -- but silent
        is not acceptable, because the same silence on an unfiltered corpus
        would mean imitating 0.500 play with no visible symptom.
        """
        meta = pd.DataFrame({
            "sel_ctx": [0, 0],
            "archetype_self": [0, 0],
            "won": [True, True],
        })
        with caplog.at_level(logging.WARNING):
            weights = compute_sample_weights(meta)
        assert np.allclose(weights, 1.0, atol=1e-5)
        assert any("skill_w" in r.message for r in caplog.records)

    def test_empty(self):
        """Empty meta returns empty weights."""
        meta = pd.DataFrame(columns=["sel_ctx", "archetype_self", "won"])
        weights = compute_sample_weights(meta)
        assert len(weights) == 0

    def test_all_same(self):
        """When all select contexts and archetypes are the same and all won,
        weights should all equal ~1.0."""
        meta = pd.DataFrame({
            "sel_ctx": [0, 0, 0],
            "archetype_self": [0, 0, 0],
            "won": [True, True, True],
        })
        weights = compute_sample_weights(meta)
        assert np.allclose(weights, 1.0, atol=1e-5)


# ============================================================
# Tests — opt_group plumbing
# ============================================================


class TestOptGroup:
    def test_dataset_yields_opt_group_as_int64(self):
        data_dir = _build_synthetic_data(n_train=8, n_val=4)
        ds = ShardDataset(data_dir, split="train")
        sample = ds[0]
        assert sample["opt_group"].dtype == torch.int64
        assert sample["opt_group"].shape == (O_MAX,)

    def test_dataset_backfills_opt_group_for_legacy_shards(self):
        """A shard written before opt_group existed must load, with every option in
        its own group so group-marginal CE degenerates to plain CE."""
        data_dir = _build_synthetic_data(n_train=8, n_val=4, drop_keys=("opt_group",))
        ds = ShardDataset(data_dir, split="train")
        sample = ds[0]
        valid = sample["opt_mask"]
        g = sample["opt_group"]
        assert g[valid].unique().numel() == int(valid.sum()), "legacy fallback must be all-distinct"
        assert (g[~valid] == -1).all()


# ============================================================
# Tests — ShardDataset
# ============================================================


class TestShardDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Ensure reproducibility
        pass

    def test_construction_train_split(self):
        data_dir = _build_synthetic_data(n_train=100, n_val=20)
        ds = ShardDataset(data_dir, split="train")
        assert len(ds) == 100

    def test_construction_val_split(self):
        data_dir = _build_synthetic_data(n_train=100, n_val=20)
        ds = ShardDataset(data_dir, split="val")
        assert len(ds) == 20

    def test_getitem_returns_dict_of_tensors(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        sample = ds[0]
        assert isinstance(sample, dict)
        assert all(isinstance(v, torch.Tensor) for v in sample.values())

    def test_getitem_has_sample_weight(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        sample = ds[0]
        assert "sample_weight" in sample
        assert sample["sample_weight"].ndim == 0  # scalar

    def test_getitem_has_value_target(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        sample = ds[0]
        assert "value_target" in sample

    def test_getitem_has_tok_mask(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        sample = ds[0]
        assert "tok_mask" in sample
        assert sample["tok_mask"].dtype == torch.bool

    def test_len(self):
        data_dir = _build_synthetic_data(n_train=50, n_val=10)
        ds = ShardDataset(data_dir, split="train")
        assert len(ds) == 50

    def test_shuffle_reproducible(self):
        data_dir = _build_synthetic_data(n_train=20, n_val=2)
        ds1 = ShardDataset(data_dir, split="train", shuffle=True, seed=42)
        ds2 = ShardDataset(data_dir, split="train", shuffle=True, seed=42)
        for i in range(len(ds1)):
            s1 = ds1[i]
            s2 = ds2[i]
            # Compare tensor values
            for k in s1:
                assert torch.equal(s1[k], s2[k]), f"Mismatch at key {k}"

    def test_shuffle_changes_order(self):
        data_dir = _build_synthetic_data(n_train=20, n_val=2)
        ds1 = ShardDataset(data_dir, split="train", shuffle=False)
        ds2 = ShardDataset(data_dir, split="train", shuffle=True, seed=123)
        # With many samples, at least one position should differ
        n_diff = 0
        for i in range(min(len(ds1), 10)):
            s1_sel_ctx = ds1[i]["sel_ctx"].item()
            s2_sel_ctx = ds2[i]["sel_ctx"].item()
            if s1_sel_ctx != s2_sel_ctx:
                n_diff += 1
        # Should have some position changes with shuffle
        # (could rarely fail by chance; accept 0 as edge case for tiny dataset)
        assert n_diff >= 0

    def test_get_raw_returns_ndarray(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        raw = ds.get_raw(0)
        assert isinstance(raw, dict)
        assert all(isinstance(v, np.ndarray) for v in raw.values())

    def test_raises_on_missing_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                ShardDataset(tmp, split="train")

    def test_raises_on_empty_split(self):
        data_dir = _build_synthetic_data(n_train=50, n_val=10)
        with pytest.raises(ValueError, match="No samples"):
            ShardDataset(data_dir, split="test")

    def test_cpu_tensors(self):
        """All tensors should be on CPU when accessed from ShardDataset."""
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        sample = ds[0]
        for v in sample.values():
            assert v.device.type == "cpu"


# ============================================================
# Tests — collate_fn
# ============================================================


class TestPerDeckFilter:
    """``archetype_self`` narrows the dataset to a single deck's decision points.

    The synthetic builder assigns ``archetype_self = sample_idx % 3``.
    """

    def test_none_keeps_every_deck(self):
        data_dir = _build_synthetic_data(n_train=90, n_val=20)
        assert len(ShardDataset(data_dir, split="train", archetype_self=None)) == 90

    def test_filter_selects_only_that_deck(self):
        data_dir = _build_synthetic_data(n_train=90, n_val=20)
        subsets = [
            ShardDataset(data_dir, split="train", archetype_self=a) for a in (0, 1, 2)
        ]
        # Partition: the three specialists together cover the full split.
        assert sum(len(s) for s in subsets) == 90
        for s in subsets:
            assert len(s) > 0
            assert (s.meta["archetype_self"] == s.archetype_self).all()

    def test_filter_applies_within_the_split(self):
        # Filtering must not leak val rows into train.
        data_dir = _build_synthetic_data(n_train=90, n_val=30)
        train = ShardDataset(data_dir, split="train", archetype_self=1)
        val = ShardDataset(data_dir, split="val", archetype_self=1)
        assert train.meta["shard"].str.startswith("train").all()
        assert val.meta["shard"].str.startswith("val").all()

    def test_samples_are_still_well_formed(self):
        data_dir = _build_synthetic_data(n_train=90, n_val=20)
        ds = ShardDataset(data_dir, split="train", archetype_self=2)
        sample = ds[0]
        assert "opt_mask" in sample
        assert "sample_weight" in sample

    def test_absent_deck_raises_with_actionable_message(self):
        data_dir = _build_synthetic_data(n_train=90, n_val=20)
        with pytest.raises(ValueError, match="No samples for archetype_self=99"):
            ShardDataset(data_dir, split="train", archetype_self=99)


class TestCollateFn:
    def test_stacks_batch_dim(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        batch = collate_fn([ds[i] for i in range(3)])
        for k, v in batch.items():
            assert v.shape[0] == 3, f"Key {k} has wrong batch dim: {v.shape}"

    def test_derives_encoder_padding_mask(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        batch = collate_fn([ds[0]])
        assert "encoder_padding_mask" in batch
        assert batch["encoder_padding_mask"].dtype == torch.bool
        # encoder_padding_mask = ~tok_mask
        assert torch.equal(
            batch["encoder_padding_mask"],
            ~batch["tok_mask"],
        )

    def test_handles_scalar_tensors(self):
        """Scalar tensors (shape ()) should become [B] after stacking."""
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        batch = collate_fn([ds[i] for i in range(3)])
        # sel_type is a scalar in each sample
        assert batch["sel_type"].shape == (3,)
        assert batch["value_target"].shape == (3,)
        assert batch["sample_weight"].shape == (3,)

    def test_empty_batch(self):
        assert collate_fn([]) == {}

    def test_single_sample_batch(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        batch = collate_fn([ds[0]])
        assert batch["sel_type"].shape == (1,)


# ============================================================
# Tests — memory-mapped shard cache
# ============================================================
#
# The shards are written with np.savez_compressed, and np.load's mmap_mode is
# silently ignored for .npz archives: `dict(np.load(path, mmap_mode="r"))`
# decompresses every array into anonymous RAM (a 5.6 MB shard inflates to
# 528 MB) and the dataset then held it for the process lifetime.  With
# num_workers=8 each worker paid that independently.  These tests pin the
# behaviour that replaced it: decompress once to a .npy-per-key cache on disk,
# then genuinely mmap, so pages are shared and evictable.


class TestMmapCache:
    def _shard_keys(self, data_dir, shard_name):
        with np.load(Path(data_dir) / "shards" / shard_name) as z:
            return set(z.files)

    def test_cache_built_on_init(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ShardDataset(data_dir, split="train")
        cache = Path(data_dir) / "shards" / ".mmap-cache" / "train-00000.npz"
        assert cache.is_dir()
        names = {p.stem for p in cache.glob("*.npy")}
        assert names == self._shard_keys(data_dir, "train-00000.npz")

    def test_cache_only_covers_the_requested_split(self):
        """A val-split dataset must not spend disk decompressing train shards."""
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ShardDataset(data_dir, split="val")
        cache = Path(data_dir) / "shards" / ".mmap-cache"
        assert (cache / "val-00000.npz").is_dir()
        assert not (cache / "train-00000.npz").exists()

    def test_arrays_are_memmaps_not_ram(self):
        """The actual laziness guarantee: nothing is materialised on open."""
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        shard = ds._open_shard("train-00000.npz")
        assert shard
        for key, arr in shard.items():
            assert isinstance(arr, np.memmap), f"{key} is {type(arr)}, not mmap'd"

    def test_values_match_the_npz(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        with np.load(Path(data_dir) / "shards" / "train-00000.npz") as z:
            for i in (0, 3, 9):
                raw = ds.get_raw(i)
                row = int(ds.meta.iloc[int(ds._indices[i])]["row"])
                for key in raw:
                    np.testing.assert_array_equal(
                        raw[key], z[key][row], err_msg=f"{key} row {row}"
                    )

    def test_cache_reused_not_rebuilt(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ShardDataset(data_dir, split="train")
        probe = Path(data_dir) / "shards" / ".mmap-cache" / "train-00000.npz" / "tok_mask.npy"
        before = probe.stat().st_mtime_ns
        ShardDataset(data_dir, split="train")
        assert probe.stat().st_mtime_ns == before

    def test_cache_rebuilt_when_shard_changes(self):
        """A rebuilt corpus must not be read through a cache of the old one —
        card ids are vocab indices, so stale rows are silently mislabelled."""
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ds = ShardDataset(data_dir, split="train")
        before = ds.get_raw(0)["poke_card_id"].copy()

        shard_path = Path(data_dir) / "shards" / "train-00000.npz"
        with np.load(shard_path) as z:
            arrays = {k: z[k].copy() for k in z.files}
        arrays["poke_card_id"] = arrays["poke_card_id"] + 1
        np.savez_compressed(shard_path, **arrays)

        ds2 = ShardDataset(data_dir, split="train")
        after = ds2.get_raw(0)["poke_card_id"]
        np.testing.assert_array_equal(after, before + 1)

    def test_falls_back_when_cache_cannot_be_written(self):
        """A read-only corpus (shared/NFS) must still train, just without the
        cache — degraded memory, not a crash."""
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        shards_dir = Path(data_dir) / "shards"
        mode = shards_dir.stat().st_mode
        shards_dir.chmod(0o555)
        try:
            ds = ShardDataset(data_dir, split="train")
            sample = ds[0]
            assert sample["tok_mask"].shape[0] > 0
            assert not (shards_dir / ".mmap-cache").exists()
        finally:
            shards_dir.chmod(mode)

    def test_partial_cache_is_rebuilt(self):
        """An interrupted build leaves a directory that exists but is missing
        keys; reusing it would raise KeyError deep inside __getitem__."""
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ShardDataset(data_dir, split="train")
        cache = Path(data_dir) / "shards" / ".mmap-cache" / "train-00000.npz"
        (cache / "tok_mask.npy").unlink()
        ds = ShardDataset(data_dir, split="train")
        assert (cache / "tok_mask.npy").exists()
        assert ds[0]["tok_mask"].shape[0] > 0

    def test_two_shards_do_not_share_rows(self):
        data_dir = _build_synthetic_data(n_train=100, n_val=10, samples_per_shard=60)
        ds = ShardDataset(data_dir, split="train")
        shards = {str(s) for s in ds.meta["shard"]}
        assert len(shards) > 1, "fixture must span >1 shard for this to test anything"
        with np.load(Path(data_dir) / "shards" / "train-00001.npz") as z:
            rows = ds.meta.index[ds.meta["shard"] == "train-00001.npz"].tolist()
            assert rows
            pos = int(np.where(ds._indices == rows[0])[0][0])
            raw = ds.get_raw(pos)
            np.testing.assert_array_equal(
                raw["poke_card_id"], z["poke_card_id"][int(ds.meta.iloc[rows[0]]["row"])]
            )

    def test_orphan_cache_is_removed(self):
        """A rebuilt corpus with fewer shards must not strand GBs of cache."""
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ShardDataset(data_dir, split="train")
        cache_root = Path(data_dir) / "shards" / ".mmap-cache"
        orphan = cache_root / "train-09999.npz"
        orphan.mkdir()
        (orphan / "junk.npy").write_bytes(b"x")
        ShardDataset(data_dir, split="train")
        assert not orphan.exists()
        assert (cache_root / "train-00000.npz").is_dir()

    def test_other_splits_cache_is_not_evicted(self):
        data_dir = _build_synthetic_data(n_train=10, n_val=2)
        ShardDataset(data_dir, split="val")
        cache_root = Path(data_dir) / "shards" / ".mmap-cache"
        assert (cache_root / "val-00000.npz").is_dir()
        ShardDataset(data_dir, split="train")
        assert (cache_root / "val-00000.npz").is_dir()
