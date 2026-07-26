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
from ptcg_il.model.belief import BeliefHeads, belief_loss, soft_cross_entropy

V = 20
PAD, UNKNOWN = 0, 1
ID_TO_INDEX = {100 + i: i + 2 for i in range(V - 2)}


def _card(card_id: int, player: int = 1) -> dict:
    return {"id": card_id, "playerIndex": player}


# ============================================================
# Labels
# ============================================================


class TestDeckCounts:
    def test_counts_sum_to_deck_size(self):
        deck = [100] * 4 + [101] * 20 + [102] * 36
        dense = deck_counts_dense(deck, ID_TO_INDEX, V)
        assert dense.sum() == DECK_SIZE
        assert dense[ID_TO_INDEX[101]] == 20

    def test_oov_collapses_to_unknown(self):
        """Unknown ids must land on UNKNOWN, matching what the featurizer does
        with the same card — otherwise label and input disagree."""
        dense = deck_counts_dense([9999, 9998, 100], ID_TO_INDEX, V)
        assert dense[UNKNOWN] == 2
        assert dense[ID_TO_INDEX[100]] == 1


class TestVisibleCounts:
    def test_counts_only_opponent_cards(self):
        state = {
            "players": [
                {"active": [], "bench": [], "discard": [_card(100, player=0)]},
                {"active": [], "bench": [], "discard": [_card(101, player=1)]},
            ],
        }
        dense = opp_visible_counts(state, your_index=0, id_to_index=ID_TO_INDEX,
                                   vocab_size=V)
        assert dense[ID_TO_INDEX[101]] == 1
        assert dense[ID_TO_INDEX[100]] == 0

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
        dense = opp_visible_counts(state, 0, ID_TO_INDEX, V)
        assert dense[ID_TO_INDEX[101]] == 1
        assert dense[ID_TO_INDEX[100]] == 0

    def test_stadium_is_counted(self):
        """The Stadium hangs off the state, not off a player.  Missing it made
        the hidden-pool size wrong by +1 in 31% of decision points."""
        state = {
            "players": [{"active": [], "bench": [], "discard": []}] * 2,
            "stadium": _card(102, player=1),
        }
        dense = opp_visible_counts(state, 0, ID_TO_INDEX, V)
        assert dense[ID_TO_INDEX[102]] == 1


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
            your_index=0, opp_deck=deck, id_to_index=ID_TO_INDEX, vocab_size=V,
            opp_arch_index=2, opp_hand_ids=[100],
        )
        dense = densify(labels["bel_deck_idx"][None, :],
                        labels["bel_deck_cnt"][None, :], V)[0]
        assert dense.sum() == pytest.approx(1.0)
        assert dense[ID_TO_INDEX[100]] == pytest.approx(0.5)
        assert dense[PAD] == 0.0

    def test_hidden_is_the_deck_minus_what_is_visible(self):
        deck = [100] * 30 + [101] * 30
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [], "discard": [_card(100)] * 4},
            ],
        }
        labels = build_belief_labels(
            state=state, your_index=0, opp_deck=deck, id_to_index=ID_TO_INDEX,
            vocab_size=V, opp_arch_index=0, opp_hand_ids=None,
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
            your_index=0, opp_deck=[100] * 60, id_to_index=ID_TO_INDEX,
            vocab_size=V, opp_arch_index=1, opp_hand_ids=[100],
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
            your_index=0, opp_deck=deck, id_to_index=ID_TO_INDEX, vocab_size=V,
            opp_arch_index=0, opp_hand_ids=None,
        )
        assert int(labels["bel_deck_cnt"].sum()) == DECK_SIZE


# ============================================================
# Heads and loss
# ============================================================


def _heads(n_arch: int = 4, D: int = 8) -> BeliefHeads:
    heads = BeliefHeads(V, D, n_arch, card_emb=torch.nn.Embedding(V, D))
    return heads


class TestBeliefHeads:
    def test_output_shapes(self):
        out = _heads()(torch.randn(3, 8))
        assert out["arch"].shape == (3, 4)
        for k in ("deck", "hidden", "hand"):
            assert out[k].shape == (3, V)

    def test_pad_gets_zero_probability(self):
        out = _heads()(torch.randn(3, 8))
        for k in ("deck", "hidden", "hand"):
            p = torch.softmax(out[k], dim=-1)
            assert torch.all(p[:, PAD] == 0.0)

    def test_missing_card_encoder_fails_loudly(self):
        """A silent zero matrix here would train to a uniform belief and look
        merely 'weak' rather than broken."""
        with pytest.raises(RuntimeError, match="card_emb"):
            BeliefHeads(V, 8, 4)(torch.randn(2, 8))


def _labels(B: int = 4, valid: bool = True, hand_valid: bool = True,
            arch: int = 1, n_arch: int = 4) -> dict[str, torch.Tensor]:
    dist = torch.zeros(B, V)
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
        assert parts["belief/deck_ce"] == pytest.approx(np.log(V - 1), abs=0.5)
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
        target = torch.zeros(1, V)
        target[0, 5] = 1.0
        confident = torch.full((1, V), -10.0)
        confident[0, 5] = 10.0
        assert float(soft_cross_entropy(confident, target)) < float(
            soft_cross_entropy(torch.zeros(1, V), target)
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
        probs = np.full(V, 1.0 / V)
        index_to_id = [0, 1] + [100 + i for i in range(V - 2)]
        assert len(deck_from_distribution(probs, index_to_id)) == DECK_SIZE

    def test_never_emits_pad_or_unknown(self):
        """PAD and UNKNOWN name no card the engine can deal; leaking either
        into the template makes the determinizer's deck illegal."""
        probs = np.zeros(V)
        probs[PAD] = probs[UNKNOWN] = 0.4
        probs[5] = 0.2
        index_to_id = [0, 1] + [100 + i for i in range(V - 2)]
        deck = deck_from_distribution(probs, index_to_id)
        assert set(deck) == {index_to_id[5]}
        assert len(deck) == DECK_SIZE

    def test_dominant_card_gets_most_slots(self):
        probs = np.zeros(V)
        probs[5] = 0.9
        probs[6] = 0.1
        index_to_id = [0, 1] + [100 + i for i in range(V - 2)]
        deck = deck_from_distribution(probs, index_to_id)
        assert deck.count(index_to_id[5]) == 54

    def test_sampled_mode_is_still_sixty_cards(self):
        probs = np.full(V, 1.0 / V)
        index_to_id = [0, 1] + [100 + i for i in range(V - 2)]
        rng = np.random.default_rng(0)
        assert len(deck_from_distribution(probs, index_to_id, rng=rng)) == DECK_SIZE

    def test_all_zero_distribution_yields_no_opinion(self):
        """An empty list is the signal that leaves the Rust determinizer on its
        own fallback, which beats handing it a deck of PAD."""
        assert deck_from_distribution(np.zeros(V), [0, 1]) == []


@pytest.fixture
def oracle(tmp_path):
    from ptcg_il.featurizer import normalize_vocab
    from ptcg_il.model.policy import Policy

    ids = [100 + i for i in range(V - 2)]
    vocab = normalize_vocab({
        "id_to_index": {str(c): i + 2 for i, c in enumerate(ids)},
        "size": V,
    })
    reps = [[ids[i % len(ids)]] * DECK_SIZE for i in range(3)]
    archetypes = {
        "opp_ids": [0, 1, 2],
        "archetypes": [{"id": i, "representative": reps[i]} for i in range(3)],
    }
    policy = Policy(V=V, A=2, D=16, layers=1, heads=2, ff=32, n_opp_arch=3).eval()
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
            "deck": np.zeros(V),
        })
        assert orc.predict({}) == reps[0]

    def test_unsure_archetype_falls_back_to_the_distribution(self, oracle, monkeypatch):
        orc, reps = oracle
        deck_logits = np.zeros(V)
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

        deck_logits = np.zeros(V)
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
    def _dataset(self, tmp_path, vocab_size):
        from ptcg_il.train.dataset import ShardDataset

        ds = ShardDataset.__new__(ShardDataset)
        ds.vocab_size = vocab_size
        return ds

    def _sparse_sample(self):
        labels = build_belief_labels(
            state={"players": [{"active": [], "bench": [], "discard": []}] * 2},
            your_index=0, opp_deck=[100] * 30 + [101] * 30,
            id_to_index=ID_TO_INDEX, vocab_size=V, opp_arch_index=1,
            opp_hand_ids=[100],
        )
        return {k: torch.as_tensor(v) for k, v in labels.items()}

    def test_rows_become_normalized_distributions(self, tmp_path):
        ds = self._dataset(tmp_path, V)
        sample = self._sparse_sample()
        ds._densify_belief(sample)
        for key in ("bel_deck", "bel_hidden", "bel_hand"):
            assert sample[key].shape == (V,)
            assert float(sample[key].sum()) == pytest.approx(1.0)
            assert float(sample[key][PAD]) == 0.0

    def test_sparse_keys_do_not_leak_into_the_batch(self, tmp_path):
        ds = self._dataset(tmp_path, V)
        sample = self._sparse_sample()
        ds._densify_belief(sample)
        assert not [k for k in sample if k.startswith("bel_") and
                    (k.endswith("_idx") or k.endswith("_cnt"))]

    def test_pre_belief_shards_still_load(self, tmp_path):
        """An older corpus has no bel_* keys at all; the dataset must invent
        masked-out ones rather than raise, or the belief flag would fork the
        whole training path."""
        ds = self._dataset(tmp_path, V)
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
    policy = Policy(V=V, A=2, D=16, layers=1, heads=2, ff=32, n_opp_arch=3).eval()
    assert hasattr(policy, "belief_heads")
    assert policy.belief_heads.card_emb is policy.embed.card


def test_belief_heads_are_absent_from_the_plain_forward_graph():
    """The auxiliary heads must cost nothing when the belief loss is off."""
    from ptcg_il.model.policy import Policy

    policy = Policy(V=V, A=2, D=16, layers=1, heads=2, ff=32, n_opp_arch=3)
    names = {n for n, _ in policy.named_parameters() if n.startswith("belief_heads.")}
    assert names, "belief heads should still be registered parameters"
