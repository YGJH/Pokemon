"""The ``*_card_feat`` tensors are gathered on the model's device, not in the loader.

Every one of them is a lookup from a frozen ~1 MB table keyed by an id the
featurizer already produces, so materialising them per sample in a DataLoader
worker was pure duplication -- and the expensive kind: it inflated a sample from
29.6 KiB to 416 KiB (93% `*_card_feat`), putting 437 MB per batch through the
worker->parent queue and costing 128 of 174 ms/step.

Moving the gather into ``Policy`` makes the tables the single producer.  The
tests here pin the two things that then have to hold:

*Equivalence.*  The device gather must reproduce ``gather_static_feats`` exactly
-- including its unusual out-of-range rule.  Ids <= 0, negative, or past the end
of the table gather **zeros**, not row 0 and not a wrapped row.  A clamp without
the paired re-zero is the bug this rule exists to prevent, and it is invisible:
it silently gives an unknown card some other card's stats.

*Loudness.*  A model with no tables must raise, and a checkpoint whose recorded
table shape disagrees with the live one must raise.  The alternative in both
cases is a forward pass over all-zero features, which trains and evaluates
perfectly happily while the model sees no card identity at all.
"""

import numpy as np
import pytest
import torch

from ptcg_il.featurizer import (
    CARD_FEAT_SOURCES,
    F_ATK,
    F_CARD,
    build_static_table,
    gather_static_feats,
)
from ptcg_il.model.policy import Policy, load_static_tables, policy_from_config
from tests.test_model_policy import _make_synthetic_batch

N_CARDS, N_ATK = 40, 12


def _tables(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense tables whose every row is distinct, so a wrong row is detectable."""
    rng = np.random.default_rng(seed)
    card = torch.from_numpy(rng.normal(size=(N_CARDS, F_CARD)).astype(np.float32))
    atk = torch.from_numpy(rng.normal(size=(N_ATK, F_ATK)).astype(np.float32))
    card[0] = 0.0
    atk[0] = 0.0
    return card, atk


def _policy(**kw) -> Policy:
    card, atk = _tables()
    return Policy(D=32, heads=2, layers=1, ff=64, n_all_cards=N_CARDS,
                  all_card_feat=card, all_attack_feat=atk, **kw)


class TestGatherEquivalence:
    @pytest.mark.parametrize("kind,dim,n", [("card", F_CARD, N_CARDS),
                                            ("attack", F_ATK, N_ATK)])
    def test_matches_gather_static_feats_including_out_of_range(self, kind, dim, n):
        """Same answer as the numpy gather for in-range, PAD and out-of-range ids."""
        card, atk = _tables()
        table = card if kind == "card" else atk
        ids = np.array(
            [[0, 1, n - 1, n, n + 5000, -1, -7, 2]], dtype=np.int64
        )
        expected = gather_static_feats(ids, table.numpy())

        p = _policy()
        got = p._gather_rows(torch.from_numpy(ids), table)

        assert got.shape == (1, ids.shape[1], dim)
        np.testing.assert_array_equal(got.numpy(), expected)
        # The rule that a clamp alone would break.
        for col, i in enumerate(ids[0]):
            if i <= 0 or i >= n:
                assert torch.all(got[0, col] == 0), f"id {i} must gather zeros"

    def test_out_of_range_does_not_borrow_another_card(self):
        """A clamp without the re-zero returns row n-1 for every id past the end."""
        card, _ = _tables()
        p = _policy()
        ids = torch.tensor([[N_CARDS + 3]])
        got = p._gather_rows(ids, card)
        assert not torch.allclose(got[0, 0], card[N_CARDS - 1])
        assert torch.all(got == 0)

    def test_preserves_arbitrary_leading_shape(self):
        """discard_ids is [2, 60] and prize_ids [2, 6]; rank must survive."""
        card, _ = _tables()
        p = _policy()
        ids = torch.randint(0, N_CARDS, (3, 2, 60))
        got = p._gather_rows(ids, card)
        assert got.shape == (3, 2, 60, F_CARD)
        np.testing.assert_array_equal(
            got.numpy(), gather_static_feats(ids.numpy(), card.numpy())
        )


class TestGatherIntoBatch:
    def test_every_card_feat_source_is_produced_from_its_id(self):
        card, atk = _tables()
        p = _policy()
        x = _make_synthetic_batch(B=2, n_cards=N_CARDS, n_attacks=N_ATK)
        for feat_key in CARD_FEAT_SOURCES:
            x.pop(feat_key, None)

        out = p._gather_card_feats(x)

        n_checked = 0
        for feat_key, (id_key, kind) in CARD_FEAT_SOURCES.items():
            assert feat_key in out, f"{feat_key} not gathered"
            table = card if kind == "card" else atk
            expected = gather_static_feats(x[id_key].numpy(), table.numpy())
            np.testing.assert_allclose(out[feat_key].numpy(), expected, rtol=0, atol=0)
            n_checked += 1
        assert n_checked == len(CARD_FEAT_SOURCES) > 0

    def test_log_card_feat_is_no_longer_gathered(self):
        """After the log encoder's removal, ``log_card_feat`` is not gathered."""
        p = _policy()
        x = _make_synthetic_batch(B=2, n_cards=N_CARDS, n_attacks=N_ATK)
        x.pop("log_card_feat", None)

        out = p._gather_card_feats(x)

        assert "log_card_feat" not in out, (
            "log_card_feat should not be gathered after log encoder removal"
        )

    def test_ids_win_over_a_stale_feat_key_in_the_batch(self):
        """One producer: a feature tensor riding along in the batch is ignored.

        Nothing should be putting ``*_card_feat`` into a batch any more -- the
        featurizer stopped emitting them and the shards never stored them -- but
        a gather that deferred to a key already present would make any leftover
        producer authoritative again, silently and only for that key.
        """
        card, atk = _tables()
        p = _policy()
        x = _make_synthetic_batch(B=2, n_cards=N_CARDS, n_attacks=N_ATK)
        x["poke_card_feat"] = torch.full(
            (2, x["poke_card_id"].shape[1], F_CARD), 99.0
        )

        out = p._gather_card_feats(x)

        np.testing.assert_array_equal(
            out["poke_card_feat"].numpy(),
            gather_static_feats(x["poke_card_id"].numpy(), card.numpy()),
        )
        assert not torch.allclose(out["poke_card_feat"], x["poke_card_feat"])

    def test_does_not_mutate_the_caller_dict(self):
        """The batch is shared with the training loop; gathering must not leak."""
        p = _policy()
        x = _make_synthetic_batch(B=2, n_cards=N_CARDS, n_attacks=N_ATK)
        for feat_key in CARD_FEAT_SOURCES:
            x.pop(feat_key, None)
        before = set(x)
        p._gather_card_feats(x)
        assert set(x) == before

    def test_forward_runs_without_any_feat_key_in_the_batch(self):
        p = _policy()
        x = _make_synthetic_batch(B=2, n_cards=N_CARDS, n_attacks=N_ATK)
        for feat_key in list(CARD_FEAT_SOURCES) + ["log_card_feat"]:
            x.pop(feat_key, None)
        logits, value, _ = p(x)
        assert logits.shape[0] == 2
        assert torch.isfinite(logits).all()
        assert torch.isfinite(value).all()

    def test_gathered_features_actually_reach_the_logits(self):
        """A gather wired to nothing would still return finite logits.

        Changing a card table row must change the output, or the model is
        running on zeros and every test above passes vacuously.
        """
        x = _make_synthetic_batch(B=2, n_cards=N_CARDS, n_attacks=N_ATK)
        for feat_key in list(CARD_FEAT_SOURCES) + ["log_card_feat"]:
            x.pop(feat_key, None)
        x["poke_card_id"] = torch.full_like(x["poke_card_id"], 3)

        card, atk = _tables()
        p = Policy(D=32, heads=2, layers=1, ff=64, n_all_cards=N_CARDS,
                   all_card_feat=card, all_attack_feat=atk)
        p.eval()
        with torch.no_grad():
            before = p(x)[0].clone()
            card2 = card.clone()
            card2[3] += 5.0
            p.set_static_tables(card2, atk)
            after = p(x)[0]
        assert not torch.allclose(before, after), "card table does not reach the logits"


class TestLoudFailures:
    def test_forward_without_tables_raises(self):
        p = Policy(D=32, heads=2, layers=1, ff=64, n_all_cards=N_CARDS)
        x = _make_synthetic_batch(B=2, n_cards=N_CARDS, n_attacks=N_ATK)
        for feat_key in list(CARD_FEAT_SOURCES) + ["log_card_feat"]:
            x.pop(feat_key, None)
        with pytest.raises(RuntimeError, match="static.*table"):
            p(x)

    def test_config_records_the_table_shapes(self):
        p = _policy()
        assert p.config["static_table_shapes"] == {
            "card": [N_CARDS, F_CARD],
            "attack": [N_ATK, F_ATK],
        }

    def test_policy_from_config_rejects_a_different_card_table(self):
        """The tables define what the weights mean; a resized one is a new model."""
        p = _policy()
        card, atk = _tables()
        wider = torch.zeros(N_CARDS + 9, F_CARD)
        with pytest.raises(ValueError, match="static table"):
            policy_from_config(p.config, all_card_feat=wider, all_attack_feat=atk)

    def test_policy_from_config_accepts_the_matching_tables(self):
        p = _policy()
        card, atk = _tables()
        rebuilt = policy_from_config(p.config, all_card_feat=card,
                                     all_attack_feat=atk)
        assert rebuilt.config["static_table_shapes"] == p.config["static_table_shapes"]

    def test_old_config_without_the_record_is_let_through(self):
        """Pre-record checkpoints load as they always did."""
        p = _policy()
        cfg = dict(p.config)
        cfg.pop("static_table_shapes")
        card, atk = _tables()
        assert policy_from_config(cfg, all_card_feat=card, all_attack_feat=atk)


class TestLoadStaticTables:
    def test_reads_both_artifacts(self, tmp_path):
        ecf = {3: np.arange(F_CARD, dtype=np.float32),
               7: np.full(F_CARD, 2.0, dtype=np.float32)}
        eaf = {1: np.arange(F_ATK, dtype=np.float32)}
        np.save(tmp_path / "engine_card_features.npy", ecf, allow_pickle=True)
        np.save(tmp_path / "engine_attack_features.npy", eaf, allow_pickle=True)

        card, atk = load_static_tables(tmp_path)

        np.testing.assert_array_equal(card.numpy(), build_static_table(ecf, F_CARD))
        np.testing.assert_array_equal(atk.numpy(), build_static_table(eaf, F_ATK))
        assert card.dtype == torch.float32 and atk.dtype == torch.float32

    def test_missing_artifacts_yield_none(self, tmp_path):
        assert load_static_tables(tmp_path) == (None, None)
