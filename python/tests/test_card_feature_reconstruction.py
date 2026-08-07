"""Shards store card *ids*; the `*_card_feat` tensors are rebuilt on read.

Every ``*_card_feat`` key is a gather from the frozen static table in
``data/engine_card_features.npy`` (1.1 MB), keyed by an id the featurizer
already computes.  Storing the gathered result per row made a 50k-sample shard
6.2 GB decompressed -- ``discard_card_feat`` alone is [50000, 2, 60, 212] fp16
at 0.5% nonzero -- so the train split's mmap cache reached 47 GB against 30 GB
of RAM.  A shuffled epoch then touches random pages across a working set that
cannot be cached, the kernel never leaves reclaim, and systemd-oomd kills the
whole terminal scope on memory *pressure* (not exhaustion).

These tests pin the three halves of the fix: the featurizer emits the ids, the
writer stores ids and not features, and the dataset rebuilds features that are
bit-identical to what the featurizer produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ptcg_il.featurizer import (
    CARD_FEAT_SOURCES,
    E_MAX,
    F_ATK,
    F_CARD,
    P_MAX,
    T_MAX,
    _ids_to_feat,
    build_static_table,
    gather_static_feats,
)
from ptcg_il.train.dataset import ShardDataset


# ============================================================
# The table gather must be exactly the loop it replaces
# ============================================================


class TestGatherEquivalence:
    """``gather_static_feats`` replaces ``_ids_to_feat``'s per-element loop.

    They must agree everywhere, including on the three cases the loop treats
    specially: PAD (0), negative ids, and ids the engine has no features for.
    """

    @pytest.fixture(scope="class")
    def features(self) -> dict[int, np.ndarray]:
        rng = np.random.default_rng(0)
        # Deliberately sparse ids with a gap at 5 and 7 -- ids in range but
        # absent from the dict must gather zeros, same as the loop's
        # ``features.get(id) is None`` branch.
        return {i: rng.standard_normal(F_CARD).astype(np.float32)
                for i in (1, 2, 3, 4, 6, 8, 9)}

    def test_matches_loop_on_ordinary_ids(self, features):
        ids = np.array([[1, 2, 3], [4, 6, 8]], dtype=np.int64)
        table = build_static_table(features, F_CARD)
        assert np.array_equal(
            gather_static_feats(ids, table), _ids_to_feat(ids, features, F_CARD, None)
        )

    def test_matches_loop_on_pad_negative_missing_and_out_of_range(self, features):
        # 0 = PAD, -1 = negative sentinel, 5/7 = in range but no features,
        # 999 = past the end of the table.
        ids = np.array([0, -1, 5, 7, 999, 1], dtype=np.int64)
        table = build_static_table(features, F_CARD)
        got = gather_static_feats(ids, table)

        assert np.array_equal(got, _ids_to_feat(ids, features, F_CARD, None))
        # Not vacuous: the one real id must be nonzero, the rest exactly zero.
        assert got[5].any(), "id 1 has features and must not gather zeros"
        assert not got[:5].any(), "PAD/negative/missing/out-of-range must be zero"

    def test_invalid_ids_zero_even_when_row_0_is_populated(self):
        """The re-zero must not lean on row 0 happening to be zeros.

        ``np.where(valid, ids, 0)`` steers every invalid id at row 0, which is
        zeros only because no real engine card has id 0.  Give the table a
        populated row 0 and the clamp alone would hand PAD, negative and
        out-of-range ids that card's features instead of an empty slot.
        """
        rng = np.random.default_rng(1)
        features = {i: rng.standard_normal(F_CARD).astype(np.float32)
                    for i in (0, 1, 2)}
        table = build_static_table(features, F_CARD)
        assert table[0].any(), "fixture must populate row 0 for this to bite"

        ids = np.array([0, -1, 99, 2], dtype=np.int64)
        got = gather_static_feats(ids, table)

        assert not got[:3].any(), "PAD/negative/out-of-range leaked row 0"
        assert np.array_equal(got[3], features[2]), "valid id must still gather"

    def test_empty_feature_dict_is_all_zeros(self):
        ids = np.array([1, 2, 3], dtype=np.int64)
        table = build_static_table({}, F_CARD)
        assert np.array_equal(gather_static_feats(ids, table), np.zeros((3, F_CARD)))

    def test_gather_preserves_id_shape(self, features):
        ids = np.zeros((2, 60), dtype=np.int64)
        table = build_static_table(features, F_CARD)
        assert gather_static_feats(ids, table).shape == (2, 60, F_CARD)


# ============================================================
# The featurizer emits every id the writer needs to store
# ============================================================


class TestFeaturizerEmitsIds:
    def test_every_stored_feature_has_a_matching_id_key(self):
        """Each ``*_card_feat`` must be reconstructible from a key in the dict."""
        from tests.test_featurizer import (
            _build_test_vocab,
            _engine_attack_features,
            _engine_card_features,
            _get_active_step,
            _load_episode,
        )
        from ptcg_il.featurizer import featurize

        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)
        result = featurize(
            obs, vocab, action,
            engine_card_features=_engine_card_features(),
            engine_attack_features=_engine_attack_features(),
        )

        tables = {
            "card": build_static_table(_engine_card_features(), F_CARD),
            "attack": build_static_table(_engine_attack_features(), F_ATK),
        }

        assert CARD_FEAT_SOURCES, "CARD_FEAT_SOURCES must not be empty"
        n_checked = 0
        for feat_key, (id_key, kind) in CARD_FEAT_SOURCES.items():
            assert feat_key in result, f"featurize dropped {feat_key}"
            assert id_key in result, (
                f"featurize must also return {id_key} -- the writer stores it "
                f"in place of {feat_key}"
            )
            rebuilt = gather_static_feats(result[id_key], tables[kind])
            assert np.array_equal(rebuilt, result[feat_key]), (
                f"{feat_key} is not the table gather of {id_key}"
            )
            n_checked += 1
        assert n_checked >= 8, f"only checked {n_checked} feature keys"

    def test_log_card_feat_comes_from_log_feat_column_2(self):
        """``log_card_feat`` needs no id key -- log_feat already carries it."""
        from tests.test_featurizer import (
            _build_test_vocab,
            _engine_card_features,
            _get_active_step,
            _load_episode,
        )
        from ptcg_il.featurizer import featurize

        ep = _load_episode()
        vocab = _build_test_vocab(ep)
        obs, action = _get_active_step(ep, 8, 0)
        result = featurize(obs, vocab, action,
                           engine_card_features=_engine_card_features())

        table = build_static_table(_engine_card_features(), F_CARD)
        rebuilt = gather_static_feats(
            result["log_feat"][:, 2].astype(np.int64), table
        )
        assert np.array_equal(rebuilt, result["log_card_feat"])


# ============================================================
# The writer stores ids, not features
# ============================================================


def _write_corpus(
    tmp_path: Path,
    samples: list[dict[str, np.ndarray]],
    card_features: dict[int, np.ndarray],
    attack_features: dict[int, np.ndarray],
) -> Path:
    """Minimal data/ dir: one train shard plus the static tables and meta."""
    data_dir = tmp_path / "data"
    (data_dir / "shards").mkdir(parents=True)

    stacked = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
    np.savez_compressed(data_dir / "shards" / "train-00000.npz", **stacked)

    np.save(data_dir / "engine_card_features.npy", card_features)  # dict, pickled
    np.save(data_dir / "engine_attack_features.npy", attack_features)

    pd.DataFrame([
        {
            "sample_uid": f"ep_{i}", "shard": "train-00000.npz", "row": i,
            "episode_id": f"ep_{i}", "player": 0, "team": "t",
            "archetype_self": 0, "archetype_opp": 0,
            "sel_type": 0, "sel_ctx": 0, "minCount": 1, "maxCount": 1,
            "won": True,
        }
        for i in range(len(samples))
    ]).to_parquet(data_dir / "meta.parquet", index=False)
    return data_dir


@pytest.fixture
def engine_tables():
    rng = np.random.default_rng(7)
    cards = {i: rng.standard_normal(F_CARD).astype(np.float32) for i in range(1, 40)}
    attacks = {i: rng.standard_normal(F_ATK).astype(np.float32) for i in range(1, 12)}
    return cards, attacks


def _id_sample(rng, feat_shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """One id-only sample: the id keys plus the minimum the dataset reads."""
    sample: dict[str, np.ndarray] = {}
    for id_key, shape in feat_shapes.items():
        # 0 exercises PAD, 60+ exercises ids with no entry in the tables.
        sample[id_key] = rng.integers(0, 45, size=shape).astype(np.int32)
    sample["log_feat"] = rng.integers(0, 45, size=(32, 8)).astype(np.float32)
    sample["tok_mask"] = np.ones(46, dtype=bool)
    sample["opt_mask"] = np.ones(64, dtype=bool)
    sample["maxCount"] = np.int64(1)
    return sample


_ID_SHAPES = {
    "poke_card_id": (12,),
    # Cards attached to each Pokémon slot — stored as ids like everything else
    # in CARD_FEAT_SOURCES, so the reader re-gathers them the same way.
    "poke_tool_ids": (P_MAX, T_MAX),
    "poke_energy_ids": (P_MAX, E_MAX),
    "hand_card_id": (30,),
    "stadium_card_id": (1,),
    "context_card_id": (1,),
    "effect_card_id": (1,),
    "discard_ids": (2, 60),
    "prize_ids": (2, 6),
    "opt_card_id": (64,),
    "opt_attack_idx": (64,),
}


class TestWriterStoresIds:
    def test_shards_carry_ids_and_no_materialised_features(self, tmp_path):
        """build_shards must not write any *_card_feat key."""
        from ptcg_il.shard_writer import _DERIVED_KEYS, _INT32_KEYS

        # The writer's contract, asserted against the featurizer's own map so
        # the two cannot drift apart silently.
        assert _DERIVED_KEYS == set(CARD_FEAT_SOURCES) | {"log_card_feat"}
        assert _INT32_KEYS == {id_key for id_key, _ in CARD_FEAT_SOURCES.values()}

    def test_write_shard_downcasts_ids_to_int32(self, tmp_path):
        from ptcg_il.shard_writer import _write_shard

        rng = np.random.default_rng(1)
        buf = [_id_sample(rng, _ID_SHAPES) for _ in range(4)]
        for s in buf:  # writer receives int64 from featurize
            for k in _ID_SHAPES:
                s[k] = s[k].astype(np.int64)

        path = _write_shard("train", 0, buf, tmp_path)
        with np.load(path) as z:
            for id_key in _ID_SHAPES:
                assert z[id_key].dtype == np.int32, f"{id_key} not downcast"
            assert z["poke_card_id"][0].tolist() == buf[0]["poke_card_id"].tolist()


# ============================================================
# The reader rebuilds what the writer dropped
# ============================================================


class TestDatasetRebuildsFeatures:
    def test_features_match_the_table_gather(self, tmp_path, engine_tables):
        cards, attacks = engine_tables
        rng = np.random.default_rng(3)
        samples = [_id_sample(rng, _ID_SHAPES) for _ in range(6)]
        data_dir = _write_corpus(tmp_path, samples, cards, attacks)

        ds = ShardDataset(data_dir, split="train")
        tables = {
            "card": build_static_table(cards, F_CARD),
            "attack": build_static_table(attacks, F_ATK),
        }

        n_nonzero = 0
        for i in range(len(ds)):
            item = ds[i]
            raw = ds.get_raw(i)
            for feat_key, (id_key, kind) in CARD_FEAT_SOURCES.items():
                assert feat_key in item, f"{feat_key} was not rebuilt"
                want = gather_static_feats(raw[id_key].astype(np.int64), tables[kind])
                assert np.array_equal(item[feat_key].numpy(), want), feat_key
                n_nonzero += int(np.count_nonzero(want) > 0)

        # Guards against the whole assertion passing on all-zero tensors.
        assert n_nonzero > 0, "every rebuilt feature was zero — test is vacuous"

    def test_log_card_feat_rebuilt_from_log_feat(self, tmp_path, engine_tables):
        cards, attacks = engine_tables
        rng = np.random.default_rng(4)
        samples = [_id_sample(rng, _ID_SHAPES) for _ in range(3)]
        data_dir = _write_corpus(tmp_path, samples, cards, attacks)

        ds = ShardDataset(data_dir, split="train")
        table = build_static_table(cards, F_CARD)
        item, raw = ds[0], ds.get_raw(0)
        want = gather_static_feats(raw["log_feat"][:, 2].astype(np.int64), table)

        assert np.array_equal(item["log_card_feat"].numpy(), want)
        assert want.any(), "log_card_feat rebuilt to all zeros — test is vacuous"

    def test_rebuilt_features_are_float32(self, tmp_path, engine_tables):
        cards, attacks = engine_tables
        rng = np.random.default_rng(5)
        data_dir = _write_corpus(
            tmp_path, [_id_sample(rng, _ID_SHAPES)], cards, attacks
        )
        item = ShardDataset(data_dir, split="train")[0]
        for feat_key in CARD_FEAT_SOURCES:
            assert item[feat_key].dtype == torch.float32, feat_key

    def test_legacy_shards_with_stored_features_are_left_alone(
        self, tmp_path, engine_tables
    ):
        """A shard predating the change carries features and no ids."""
        cards, attacks = engine_tables
        rng = np.random.default_rng(6)
        sample = _id_sample(rng, _ID_SHAPES)
        stored = {
            feat_key: rng.standard_normal(
                _ID_SHAPES[id_key] + (F_CARD if kind == "card" else F_ATK,)
            ).astype(np.float16)
            for feat_key, (id_key, kind) in CARD_FEAT_SOURCES.items()
        }
        # log_card_feat has no id key, so it is the one feature a legacy shard
        # can still be clobbered on: log_feat is present either way.
        stored["log_card_feat"] = rng.standard_normal((32, F_CARD)).astype(np.float16)
        for id_key in _ID_SHAPES:  # legacy shards have no id keys at all
            sample.pop(id_key)
        sample.update(stored)

        data_dir = _write_corpus(tmp_path, [sample], cards, attacks)
        item = ShardDataset(data_dir, split="train")[0]

        for feat_key in stored:
            assert np.allclose(
                item[feat_key].numpy(), stored[feat_key].astype(np.float32)
            ), f"{feat_key} was overwritten instead of used as stored"
