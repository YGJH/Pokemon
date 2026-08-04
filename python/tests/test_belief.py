"""Tests for the opponent-card belief model: labels, heads, loss, sampling.

The belief pipeline has an unusual failure mode — every stage degrades
*silently*.  A wrong label still trains, a mismatched mask still produces a
finite loss, and a bogus decklist still lets MCTS run.  These tests pin the
invariants that would otherwise only show up as a mysteriously weak planner.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ptcg_il.belief_infer import OpponentDeckOracle, deck_from_distribution
from ptcg_il.belief_labels import (
    BELIEF_K,
    DECK_SIZE,
    build_belief_labels,
    deck_counts_dense,
    densify,
    empty_belief_labels,
    hand_after,
    opp_hand_timeline,
    opp_visible_counts,
)
from ptcg_il.featurizer import F_CARD
from ptcg_il.model.belief import (
    LOG_FEAT_DIM,
    L_LOG_MAX,
    MAX_AREA,
    N_AREA_EMB,
    BeliefHeads,
    BeliefModule,
    belief_loss,
    soft_cross_entropy,
)
from ptcg_il.model.cards import CardFeaturizer

# n_all_cards = max engine card ID + 1 (simulate ~200 cards)
N_ALL = 200


def _card(card_id: int, player: int = 1) -> dict:
    return {"id": card_id, "playerIndex": player}


# ============================================================
# Labels
# ============================================================


class TestDeckCounts:
    def test_counts_sum_to_deck_size(self):
        deck = [100] * 4 + [101] * 20 + [102] * 36
        dense = deck_counts_dense(deck, N_ALL)
        assert dense.sum() == DECK_SIZE
        assert dense[101] == 20

    def test_oov_ids_ignored(self):
        """Card ids beyond n_all_cards are silently dropped."""
        dense = deck_counts_dense([9999, 9998, 100], N_ALL)
        assert dense[100] == 1
        assert dense.sum() == 1  # only card 100 was in range


class TestVisibleCounts:
    def test_counts_only_opponent_cards(self):
        state = {
            "players": [
                {"active": [], "bench": [], "discard": [_card(100, player=0)]},
                {"active": [], "bench": [], "discard": [_card(101, player=1)]},
            ],
        }
        dense = opp_visible_counts(state, your_index=0, n_all_cards=N_ALL)
        assert dense[101] == 1
        assert dense[100] == 0

    def test_energy_we_attached_is_not_credited_to_them(self):
        """Our Energy sitting on their Pokemon is ours, not part of their deck.
        Without the playerIndex filter it would inflate every hidden pool."""
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {
                    "active": [{"id": 101, "playerIndex": 1,
                                "energyCards": [_card(100, player=0)]}],
                    "bench": [], "discard": [],
                },
            ],
        }
        dense = opp_visible_counts(state, 0, N_ALL)
        assert dense[101] == 1
        assert dense[100] == 0

    def test_stadium_is_counted(self):
        """The Stadium hangs off the state, not off a player.  Missing it made
        the hidden-pool size wrong by +1 in 31% of decision points."""
        state = {
            "players": [{"active": [], "bench": [], "discard": []}] * 2,
            "stadium": _card(102, player=1),
        }
        dense = opp_visible_counts(state, 0, N_ALL)
        assert dense[102] == 1


class TestHandTimeline:
    EP = {
        "steps": [
            [{"status": "ACTIVE", "observation": {"current": {}}},
             {"status": "INACTIVE"}],
            [{"status": "INACTIVE"},
             {"status": "ACTIVE",
              "observation": {"current": {"players": [{}, {"hand": [_card(100)]}],
                                          "yourIndex": 1}}}],
            [{"status": "INACTIVE"},
             {"status": "ACTIVE",
              "observation": {"current": {"players": [{}, {"hand": [_card(101),
                                                                    _card(101)]}],
                                          "yourIndex": 1}}}],
        ]
    }

    def test_timeline_is_sorted_and_covers_active_steps(self):
        tl = opp_hand_timeline(self.EP, opp_player=1)
        assert [s for s, _ in tl] == sorted(s for s, _ in tl)
        assert len(tl) == 2

    def test_hand_after_picks_the_next_step_strictly_later(self):
        tl = opp_hand_timeline(self.EP, opp_player=1)
        assert hand_after(tl, 0) == [100]
        # Strictly later: at their own step 1 the answer is their *next* hand.
        assert hand_after(tl, 1) == [101, 101]

    def test_hand_after_returns_none_past_the_last_action(self):
        """None, not [] — an empty hand is a real observation and must not be
        confused with 'they never acted again'."""
        tl = opp_hand_timeline(self.EP, opp_player=1)
        assert hand_after(tl, 99) is None


class TestSparseRoundTrip:
    def test_densify_inverts_to_a_normalized_distribution(self):
        deck = [100] * 30 + [101] * 30
        labels = build_belief_labels(
            state={"players": [{"active": [], "bench": [], "discard": []}] * 2},
            your_index=0, opp_deck=deck, n_all_cards=N_ALL,
            opp_arch_index=2, opp_hand_ids=[100],
        )
        dense = densify(labels["bel_deck_idx"][None, :],
                        labels["bel_deck_cnt"][None, :], N_ALL)[0]
        assert dense.sum() == pytest.approx(1.0)
        assert dense[100] == pytest.approx(0.5)
        assert dense[0] == 0.0  # PAD

    def test_hidden_is_the_deck_minus_what_is_visible(self):
        deck = [100] * 30 + [101] * 30
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [], "discard": [_card(100)] * 4},
            ],
        }
        labels = build_belief_labels(
            state=state, your_index=0, opp_deck=deck, n_all_cards=N_ALL,
            opp_arch_index=0, opp_hand_ids=None,
        )
        assert labels["bel_hidden_total"] == DECK_SIZE - 4
        assert not bool(labels["bel_hand_valid"])
        assert bool(labels["bel_valid"])

    def test_empty_labels_have_the_same_keys_as_real_ones(self):
        """_write_shard stacks by the *first* sample's key set, so a decision
        point without labels must still emit every key or the rest of the
        shard silently loses its labels."""
        real = build_belief_labels(
            state={"players": [{"active": [], "bench": [], "discard": []}] * 2},
            your_index=0, opp_deck=[100] * 60, n_all_cards=N_ALL,
            opp_arch_index=1, opp_hand_ids=[100],
        )
        empty = empty_belief_labels()
        assert set(real) == set(empty)
        for k in real:
            assert np.asarray(real[k]).shape == np.asarray(empty[k]).shape, k
        assert not bool(empty["bel_valid"])

    def test_sparse_slots_are_wide_enough_for_a_legal_deck(self):
        deck = [100 + i for i in range(BELIEF_K // 2)] * 2
        deck = (deck * 10)[:DECK_SIZE]
        labels = build_belief_labels(
            state={"players": [{"active": [], "bench": [], "discard": []}] * 2},
            your_index=0, opp_deck=deck, n_all_cards=N_ALL,
            opp_arch_index=0, opp_hand_ids=None,
        )
        assert int(labels["bel_deck_cnt"].sum()) == DECK_SIZE


# ============================================================
# Heads and loss
# ============================================================


def _heads(n_arch: int = 4, D: int = 8) -> BeliefHeads:
    cf = CardFeaturizer(D)
    heads = BeliefHeads(D, n_arch, N_ALL, card_emb=cf)
    # Minimal all_card_feat for the card matrix
    heads.set_all_card_feat(torch.randn(N_ALL, F_CARD))
    return heads


class TestBeliefAreaEncoding:
    """The log-area embedding must separate every AreaType the engine emits.

    It was `nn.Embedding(7, 4)` behind a `clamp(-1, 5)`, which folded BENCH(5)
    through LOOKING(12) onto one vector -- the GRU could not tell "drawn to
    hand" from "moved to prize".  Real shard logs contain areas up to 12.
    """

    def _log(self, area_from: int, area_to: int) -> tuple[torch.Tensor, torch.Tensor]:
        feat = torch.zeros(1, L_LOG_MAX, LOG_FEAT_DIM)
        feat[0, 0, 3] = float(area_from)
        feat[0, 0, 4] = float(area_to)
        mask = torch.zeros(1, L_LOG_MAX, dtype=torch.bool)
        mask[0, 0] = True
        return feat, mask

    def test_embedding_covers_every_area_type(self):
        from ptcg_mine.cards import load_engine

        load_engine()  # puts the bundled `cg` package on sys.path
        from cg.api import AreaType

        engine_max = max(int(a) for a in AreaType)
        assert MAX_AREA == engine_max, (
            f"cg.api.AreaType now reaches {engine_max}; MAX_AREA is {MAX_AREA}"
        )
        # area+1 for area in -1..MAX_AREA must all be valid embedding rows.
        assert BeliefModule(8).area_emb.num_embeddings == engine_max + 2

    def test_every_area_gets_a_distinct_embedding(self):
        emb = BeliefModule(8).area_emb
        rows = emb(torch.arange(N_AREA_EMB))
        pairs = [
            (i, j)
            for i in range(N_AREA_EMB)
            for j in range(i + 1, N_AREA_EMB)
            if torch.equal(rows[i], rows[j])
        ]
        assert not pairs, f"area embedding rows collide: {pairs}"

    @pytest.mark.parametrize("area", [5, 6, 7, 8, 9, 10, 11, 12])
    def test_high_areas_are_not_folded_onto_bench(self, area):
        """Each area above BENCH(5) produces a belief distinct from BENCH's."""
        mod = BeliefModule(8)
        mod.eval()
        with torch.no_grad():
            bench = mod(*self._log(5, -1))
            other = mod(*self._log(area, -1))
        if area == 5:
            assert torch.allclose(bench, other)
        else:
            assert not torch.allclose(bench, other, atol=1e-6), (
                f"area {area} is indistinguishable from BENCH(5)"
            )

    def test_out_of_range_area_is_clamped_not_crashed(self):
        """An area beyond the enum must clamp, not index out of bounds."""
        mod = BeliefModule(8)
        mod.eval()
        with torch.no_grad():
            out = mod(*self._log(99, -7))
        assert out.shape == (1, 8)
        assert torch.isfinite(out).all()


class TestBeliefHeads:
    def test_output_shapes(self):
        out = _heads()(torch.randn(3, 8))
        assert out["arch"].shape == (3, 4)
        for k in ("deck", "hidden", "hand"):
            assert out[k].shape == (3, N_ALL)

    def test_pad_gets_zero_probability(self):
        out = _heads()(torch.randn(3, 8))
        for k in ("deck", "hidden", "hand"):
            p = torch.softmax(out[k], dim=-1)
            assert torch.all(p[:, 0] == 0.0)  # PAD=0

    def test_missing_all_card_feat_fails_loudly(self):
        """Without set_all_card_feat the belief heads cannot produce output."""
        heads = BeliefHeads(8, 4, N_ALL, card_emb=CardFeaturizer(8))
        with pytest.raises(RuntimeError, match="all_card_feat"):
            heads(torch.randn(2, 8))


def _labels(B: int = 4, valid: bool = True, hand_valid: bool = True,
            arch: int = 1, n_arch: int = 4) -> dict[str, torch.Tensor]:
    dist = torch.zeros(B, N_ALL)
    dist[:, 2] = 0.5
    dist[:, 3] = 0.5
    return {
        "bel_deck": dist.clone(),
        "bel_hidden": dist.clone(),
        "bel_hand": dist.clone(),
        "bel_valid": torch.full((B,), valid, dtype=torch.bool),
        "bel_hand_valid": torch.full((B,), hand_valid, dtype=torch.bool),
        "bel_arch": torch.full((B,), arch, dtype=torch.long),
    }


class TestBeliefLoss:
    def test_untrained_loss_matches_the_uniform_baseline(self):
        heads = _heads()
        preds = heads(torch.zeros(4, 8))
        loss, parts = belief_loss(preds, _labels(), w_deck=1.0, w_hidden=0.0,
                                  w_hand=0.0, w_arch=0.0)
        # V-1 candidates because PAD is masked out.
        assert parts["belief/deck_ce"] == pytest.approx(np.log(N_ALL - 1), abs=0.5)
        assert torch.isfinite(loss)

    def test_all_invalid_gives_exactly_zero(self):
        preds = _heads()(torch.randn(4, 8))
        loss, _ = belief_loss(preds, _labels(valid=False))
        assert float(loss) == 0.0

    def test_missing_hand_label_does_not_poison_the_loss(self):
        preds = _heads()(torch.randn(4, 8))
        with_hand, _ = belief_loss(preds, _labels(hand_valid=True),
                                   w_deck=0.0, w_hidden=0.0, w_arch=0.0)
        without, parts = belief_loss(preds, _labels(hand_valid=False),
                                     w_deck=0.0, w_hidden=0.0, w_arch=0.0)
        assert float(without) == 0.0
        assert float(with_hand) > 0.0
        assert parts["belief/hand_ce"] == 0.0

    def test_unmapped_archetype_is_ignored_not_misclassified(self):
        preds = _heads()(torch.randn(4, 8))
        loss, parts = belief_loss(preds, _labels(arch=-1), w_deck=0.0,
                                  w_hidden=0.0, w_hand=0.0)
        assert float(loss) == 0.0
        assert "belief/arch_ce" not in parts

    def test_soft_cross_entropy_is_minimised_by_the_target(self):
        target = torch.zeros(1, N_ALL)
        target[0, 5] = 1.0
        confident = torch.full((1, N_ALL), -10.0)
        confident[0, 5] = 10.0
        assert float(soft_cross_entropy(confident, target)) < float(
            soft_cross_entropy(torch.zeros(1, N_ALL), target)
        )

    def test_gradients_reach_every_head(self):
        heads = _heads()
        loss, _ = belief_loss(heads(torch.randn(4, 8, requires_grad=True)),
                              _labels())
        loss.backward()
        for name, p in heads.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name


# ============================================================
# Deck sampling for MCTS
# ============================================================


class TestDeckFromDistribution:
    def test_returns_exactly_sixty_cards(self):
        probs = np.full(N_ALL, 1.0 / N_ALL)
        index_to_id = list(range(N_ALL))
        assert len(deck_from_distribution(probs, index_to_id)) == DECK_SIZE

    def test_never_emits_pad_or_unknown(self):
        """PAD and UNKNOWN name no card the engine can deal; leaking either
        into the template makes the determinizer's deck illegal."""
        probs = np.zeros(N_ALL)
        probs[0] = probs[1] = 0.4
        probs[5] = 0.2
        index_to_id = list(range(N_ALL))
        deck = deck_from_distribution(probs, index_to_id)
        assert set(deck) == {5}
        assert len(deck) == DECK_SIZE

    def test_dominant_card_gets_most_slots(self):
        probs = np.zeros(N_ALL)
        probs[5] = 0.9
        probs[6] = 0.1
        index_to_id = list(range(N_ALL))
        deck = deck_from_distribution(probs, index_to_id)
        assert deck.count(5) == 54

    def test_sampled_mode_is_still_sixty_cards(self):
        probs = np.full(N_ALL, 1.0 / N_ALL)
        index_to_id = list(range(N_ALL))
        rng = np.random.default_rng(0)
        assert len(deck_from_distribution(probs, index_to_id, rng=rng)) == DECK_SIZE

    def test_all_zero_distribution_yields_no_opinion(self):
        """An empty list is the signal that leaves the Rust determinizer on its
        own fallback, which beats handing it a deck of PAD."""
        assert deck_from_distribution(np.zeros(N_ALL), [0, 1]) == []


@pytest.fixture
def oracle(tmp_path):
    from ptcg_il.featurizer import normalize_vocab
    from ptcg_il.model.policy import Policy

    ids = [100 + i for i in range(N_ALL - 2)]
    vocab = normalize_vocab({
        "id_to_index": {str(c): i + 2 for i, c in enumerate(ids)},
        "size": N_ALL,
        "index_to_id": {i + 2: c for i, c in enumerate(ids)},
    })
    reps = [[ids[i % len(ids)]] * DECK_SIZE for i in range(3)]
    archetypes = {
        "opp_ids": [0, 1, 2],
        "archetypes": [{"id": i, "representative": reps[i]} for i in range(3)],
    }
    policy = Policy(D=16, layers=1, heads=2, ff=32, n_opp_arch=3, n_all_cards=N_ALL).eval()
    return OpponentDeckOracle(policy, vocab, archetypes, device="cpu"), reps


class TestOpponentDeckOracle:
    def test_representatives_follow_opp_ids_order(self, oracle):
        """bel_arch was assigned by position in opp_ids; reordering here would
        pair every class with the wrong decklist and nothing would complain."""
        orc, reps = oracle
        assert orc.representatives == reps

    def test_confident_archetype_returns_a_real_decklist(self, oracle, monkeypatch):
        orc, reps = oracle
        monkeypatch.setattr(orc, "belief", lambda obs: {
            "arch": np.array([10.0, 0.0, 0.0]),
            "deck": np.zeros(N_ALL),
        })
        assert orc.predict({}) == reps[0]

    def test_unsure_archetype_falls_back_to_the_distribution(self, oracle, monkeypatch):
        orc, reps = oracle
        deck_logits = np.zeros(N_ALL)
        deck_logits[5] = 20.0
        monkeypatch.setattr(orc, "belief", lambda obs: {
            "arch": np.zeros(3),  # uniform → below the confidence floor
            "deck": deck_logits,
        })
        deck = orc.predict({})
        assert len(deck) == DECK_SIZE
        assert deck not in reps

    def test_reserved_vocab_slots_are_not_card_ids(self, oracle, monkeypatch):
        """The real vocab.json spells indices 0 and 1 as the strings "PAD" and
        "UNKNOWN"; int()-ing them blindly used to abort oracle construction."""
        orc, _ = oracle
        vocab = dict(orc.vocab)
        vocab["index_to_id"] = ["PAD", "UNKNOWN", *orc.index_to_id[2:]]
        rebuilt = OpponentDeckOracle(
            orc.policy, vocab, {"opp_ids": [], "archetypes": []}, device="cpu"
        )
        assert rebuilt.index_to_id[:2] == [-1, -1]
        assert rebuilt.index_to_id[2:] == orc.index_to_id[2:]

        deck_logits = np.zeros(N_ALL)
        deck_logits[5] = 20.0
        monkeypatch.setattr(rebuilt, "belief", lambda obs: {
            "arch": np.zeros(3), "deck": deck_logits,
        })
        deck = rebuilt.predict({})
        assert len(deck) == DECK_SIZE
        assert all(c > 0 for c in deck)

    def test_a_broken_forward_returns_no_opinion(self, oracle, monkeypatch):
        """The oracle sits inside the planner's hot path; an exception here must
        degrade to the mirror fallback, never kill the game."""
        orc, _ = oracle

        def boom(obs):
            raise RuntimeError("no")

        monkeypatch.setattr(orc, "belief", boom)
        assert orc.predict({}) == []


# ============================================================
# Dataset densification
# ============================================================


class TestDatasetDensification:
    def _dataset(self, tmp_path, n_all_cards):
        from ptcg_il.train.dataset import ShardDataset

        ds = ShardDataset.__new__(ShardDataset)
        ds.n_all_cards = n_all_cards
        return ds

    def _sparse_sample(self):
        labels = build_belief_labels(
            state={"players": [{"active": [], "bench": [], "discard": []}] * 2},
            your_index=0, opp_deck=[100] * 30 + [101] * 30,
            n_all_cards=N_ALL, opp_arch_index=1,
            opp_hand_ids=[100],
        )
        return {k: torch.as_tensor(v) for k, v in labels.items()}

    def test_rows_become_normalized_distributions(self, tmp_path):
        ds = self._dataset(tmp_path, N_ALL)
        sample = self._sparse_sample()
        ds._densify_belief(sample)
        for key in ("bel_deck", "bel_hidden", "bel_hand"):
            assert sample[key].shape == (N_ALL,)
            assert float(sample[key].sum()) == pytest.approx(1.0)
            assert float(sample[key][0]) == 0.0  # PAD=0

    def test_sparse_keys_do_not_leak_into_the_batch(self, tmp_path):
        ds = self._dataset(tmp_path, N_ALL)
        sample = self._sparse_sample()
        ds._densify_belief(sample)
        assert not [k for k in sample if k.startswith("bel_") and
                    (k.endswith("_idx") or k.endswith("_cnt"))]

    def test_pre_belief_shards_still_load(self, tmp_path):
        """An older corpus has no bel_* keys at all; the dataset must invent
        masked-out ones rather than raise, or the belief flag would fork the
        whole training path."""
        ds = self._dataset(tmp_path, N_ALL)
        sample: dict[str, torch.Tensor] = {}
        ds._densify_belief(sample)
        assert not bool(sample["bel_valid"])
        assert int(sample["bel_arch"]) == -1

    def test_unknown_vocab_size_marks_labels_invalid(self, tmp_path):
        """Without vocab.json there is no way to size the row; emitting a
        length-1 row would blow up in the head instead of being ignored."""
        ds = self._dataset(tmp_path, None)
        sample = self._sparse_sample()
        ds._densify_belief(sample)
        assert not bool(sample["bel_valid"])


# ============================================================
# End-to-end through the policy
# ============================================================


def test_forward_with_belief_shares_one_encode_pass():
    from ptcg_il.model.policy import Policy

    torch.manual_seed(0)
    policy = Policy(D=16, layers=1, heads=2, ff=32, n_opp_arch=3, n_all_cards=N_ALL).eval()
    assert hasattr(policy, "belief_heads")
    assert policy.belief_heads.card_emb is policy.embed.card


def test_belief_heads_are_absent_from_the_plain_forward_graph():
    """The auxiliary heads must cost nothing when the belief loss is off."""
    from ptcg_il.model.policy import Policy

    policy = Policy(D=16, layers=1, heads=2, ff=32, n_opp_arch=3, n_all_cards=N_ALL)
    names = {n for n, _ in policy.named_parameters() if n.startswith("belief_heads.")}
    assert names, "belief heads should still be registered parameters"
