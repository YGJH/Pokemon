"""Identity gaps in the observation/action space that the featurizer used to drop.

Three independent defects, one file because they share fixtures:

* **Deck-search options carried no card identity.**  ``select["deck"]`` is in
  every observation and holds the acting player's own deck contents during a
  search; the featurizer never read it, so every option collapsed to
  ``type=3, src=-1, tgt=-1, card=PAD, scalar=0``.  Group-marginal CE then covers
  every valid option and the NLL is exactly ``-log(1) = 0``: those rows trained
  nothing at all.
* **The policy could not decline.**  ``minCount == 0, maxCount == 1`` means
  "you may take nothing", but a STOP column was only built for ``maxCount > 1``.
* **Attached Tools / Energy cards were reduced to a count.**  ``poke_feat[17]``
  and ``poke_feat[16]`` kept ``len(...)`` and threw the card identity away.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ptcg_il.featurizer import (
    E_MAX,
    O_MAX,
    PAD_CARD,
    P_MAX,
    STOP_OPT_TYPE,
    T_MAX,
    F_CARD,
    option_groups,
)
from ptcg_il.ref_map import card_id_at

# ``featurize`` emits card *ids*; ``Policy`` gathers the ``*_card_feat`` rows on
# device.  These tests are about the identity those features carry, so they use
# the same wrapper ``test_featurizer`` does: real featurizer, plus the model's
# gather.  The featurizer's own key set is pinned in
# ``test_card_feature_reconstruction.py``.
from tests.test_featurizer import featurize

SAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "archive"
    / "sample_episodes"
    / "80169582.json"
)

_ECF_CACHE: dict[int, np.ndarray] | None = None


def _engine_card_features() -> dict[int, np.ndarray]:
    global _ECF_CACHE
    if _ECF_CACHE is None:
        from ptcg_mine.cards import build_engine_card_features, load_engine

        card_data, attack_data = load_engine()
        _ECF_CACHE = build_engine_card_features(card_data, attack_data)
    return _ECF_CACHE


def _vocab(ep: dict) -> dict:
    from ptcg_mine.vocab import build_vocab

    return build_vocab([ep], mode="all_corpus")


def _episode() -> dict:
    with open(SAMPLE_PATH) as f:
        return json.load(f)


def _deck_search_steps(ep: dict):
    """Yield ``(obs, action)`` for every select that indexes into ``select['deck']``."""
    steps = ep["steps"]
    for i in range(len(steps) - 1):
        for p, rec in enumerate(steps[i]):
            if rec.get("status") != "ACTIVE":
                continue
            obs = rec.get("observation")
            if not obs or obs.get("select") is None:
                continue
            sel = obs["select"]
            if sel.get("deck") and any(o.get("area") == 1 for o in sel["option"]):
                yield obs, steps[i + 1][p].get("action", [])


# ---------------------------------------------------------------------------
# Defect 1 — deck-search options carry card identity
# ---------------------------------------------------------------------------


class TestDeckOptionIdentity:
    def test_card_id_at_resolves_a_deck_slot_when_given_select_deck(self):
        """``card_id_at`` refuses area=DECK on state alone, but resolves it from
        ``select['deck']`` — the engine reveals the deck during a search."""
        state = {"players": [{"deckCount": 3}, {"deckCount": 3}], "yourIndex": 0}
        deck = [{"id": 111, "playerIndex": 0}, {"id": 222, "playerIndex": 0}]

        # Without the payload the answer must stay None: the state never carries
        # deck contents and guessing would leak.
        assert card_id_at(state, 1, 0, 1) is None
        assert card_id_at(state, 1, 0, 1, select_deck=deck) == 222

    def test_card_id_at_deck_ignores_out_of_range_and_foreign_entries(self):
        state = {"players": [{"deckCount": 3}, {"deckCount": 3}], "yourIndex": 0}
        deck = [{"id": 111, "playerIndex": 0}]
        assert card_id_at(state, 1, 0, 5, select_deck=deck) is None
        assert card_id_at(state, 1, 0, -1, select_deck=deck) is None
        # An entry belonging to the other player is not ours to read.
        assert card_id_at(state, 1, 1, 0, select_deck=deck) is None

    def test_deck_search_options_get_their_real_card_ids(self):
        """Every area=DECK option resolves to the id at that index of select['deck']."""
        ep = _episode()
        vocab = _vocab(ep)
        ecf = _engine_card_features()

        n_checked = 0
        for obs, action in _deck_search_steps(ep):
            sel = obs["select"]
            out = featurize(obs, vocab, action, engine_card_features=ecf)
            for j, opt in enumerate(sel["option"]):
                if opt.get("area") != 1:
                    continue
                expected = sel["deck"][opt["index"]]["id"]
                assert out["opt_card_id"][j] == expected, (
                    f"option {j} should resolve to deck[{opt['index']}] = {expected}"
                )
                n_checked += 1
        assert n_checked > 0, "fixture contains no deck-search options to check"

    def test_deck_search_options_are_no_longer_byte_identical(self):
        """Distinct deck cards must land in distinct option groups.

        This is the property that matters: while every option was PAD they formed
        a single equivalence class, the group-marginal NLL was exactly zero, and
        the decision produced no gradient.
        """
        ep = _episode()
        vocab = _vocab(ep)
        ecf = _engine_card_features()

        obs, action = next(iter(_deck_search_steps(ep)))
        obs = copy.deepcopy(obs)
        sel = obs["select"]
        deck = sel["deck"]
        me = obs["current"]["yourIndex"]

        # Widen the real select to four options naming four *different* cards.
        seen: dict[int, int] = {}
        for idx, card in enumerate(deck):
            if card["id"] not in seen:
                seen[card["id"]] = idx
            if len(seen) == 4:
                break
        assert len(seen) == 4, "fixture deck has fewer than 4 distinct cards"
        sel["option"] = [
            {"type": 3, "area": 1, "index": idx, "playerIndex": me}
            for idx in seen.values()
        ]

        out = featurize(obs, vocab, [0], engine_card_features=ecf)
        # Slots 0..3 are the four deck options; this select is minCount=0 so a
        # STOP column follows them, and it is legitimately its own group.
        groups = out["opt_group"][:4]
        assert (out["opt_mask"][:4]).all()
        assert len(set(groups.tolist())) == 4, (
            f"four different deck cards collapsed into {len(set(groups.tolist()))} "
            "group(s) — the pointer head cannot tell them apart"
        )

    def test_deck_option_card_features_match_the_engine_row(self):
        ep = _episode()
        vocab = _vocab(ep)
        ecf = _engine_card_features()
        obs, action = next(iter(_deck_search_steps(ep)))
        sel = obs["select"]
        out = featurize(obs, vocab, action, engine_card_features=ecf)

        n_checked = 0
        for j, opt in enumerate(sel["option"]):
            if opt.get("area") != 1:
                continue
            cid = sel["deck"][opt["index"]]["id"]
            np.testing.assert_array_equal(
                out["opt_card_feat"][j], np.asarray(ecf[cid], dtype=np.float32)
            )
            n_checked += 1
        assert n_checked > 0, "no deck options examined"

    def test_face_down_prizes_stay_hidden(self):
        """The deck fix must not become a licence to reveal genuinely hidden zones."""
        state = {"players": [{"prize": [None, {"id": 42}]}], "yourIndex": 0}
        assert card_id_at(state, 6, 0, 0) is None          # face-down
        assert card_id_at(state, 6, 0, 1) == 42            # revealed


# ---------------------------------------------------------------------------
# Defect 5 — the policy must be able to decline
# ---------------------------------------------------------------------------


class TestDeclineIsRepresentable:
    def _min0_max1_step(self, ep: dict):
        steps = ep["steps"]
        for i in range(len(steps) - 1):
            for p, rec in enumerate(steps[i]):
                if rec.get("status") != "ACTIVE":
                    continue
                obs = rec.get("observation")
                if not obs or obs.get("select") is None:
                    continue
                sel = obs["select"]
                if sel.get("minCount") == 0 and sel.get("maxCount") == 1:
                    return obs, steps[i + 1][p].get("action", [])
        pytest.fail("fixture has no minCount=0, maxCount=1 select")

    def test_optional_single_select_gets_a_stop_column(self):
        ep = _episode()
        obs, action = self._min0_max1_step(ep)
        out = featurize(obs, _vocab(ep), action,
                        engine_card_features=_engine_card_features())

        stop = int(out["stop_column"])
        assert stop >= 0, "minCount=0 means declining is legal and must be expressible"
        assert out["opt_mask"][stop]
        assert out["opt_type"][stop] == STOP_OPT_TYPE
        assert out["opt_card_id"][stop] == PAD_CARD

    def test_declining_expert_is_labelled_as_stop(self):
        ep = _episode()
        obs, _ = self._min0_max1_step(ep)
        out = featurize(obs, _vocab(ep), [],
                        engine_card_features=_engine_card_features())

        stop = int(out["stop_column"])
        assert out["action_idx"][0] == stop
        assert int(out["action_len"]) == 1

    def test_taking_an_option_does_not_append_stop_on_single_select(self):
        """A single-select pick is one step; appending STOP would make it two."""
        ep = _episode()
        obs, _ = self._min0_max1_step(ep)
        out = featurize(obs, _vocab(ep), [0],
                        engine_card_features=_engine_card_features())

        assert out["action_idx"][0] == 0
        assert int(out["action_len"]) == 1
        assert out["action_idx"][1] == -1

    def test_mandatory_single_select_has_no_stop_column(self):
        """minCount >= 1 with maxCount == 1 must stay exactly as it was."""
        ep = _episode()
        obs, action = self._min0_max1_step(ep)
        obs = copy.deepcopy(obs)
        obs["select"]["minCount"] = 1

        out = featurize(obs, _vocab(ep), action or [0],
                        engine_card_features=_engine_card_features())
        assert int(out["stop_column"]) == -1

    def test_multi_select_labels_are_unchanged(self):
        """The multi-select contract (STOP appended after the last pick) still holds."""
        ep = _episode()
        obs, _ = self._min0_max1_step(ep)
        obs = copy.deepcopy(obs)
        sel = obs["select"]
        me = obs["current"]["yourIndex"]
        sel["minCount"], sel["maxCount"] = 0, 3
        sel["option"] = [{"type": 1}, {"type": 2}, {"type": 14}]

        out = featurize(obs, _vocab(ep), [0, 1],
                        engine_card_features=_engine_card_features())
        stop = int(out["stop_column"])
        assert stop == 3
        assert out["action_idx"][0] == 0
        assert out["action_idx"][1] == 1
        assert out["action_idx"][2] == stop
        assert int(out["action_len"]) == 3


class TestSingleSelectDecode:
    """A STOP pick at inference must become an empty action, not an OOB index."""

    def test_stop_argmax_decodes_to_no_selection(self):
        import torch

        from ptcg_il.model.policy import decode_single_select

        opt_mask = torch.tensor([[True, True, True, False]])
        logits = torch.tensor([[0.1, 0.2, 5.0, -1e9]])
        # column 2 is STOP
        assert decode_single_select(logits, opt_mask, torch.tensor([2])) == []

    def test_regular_argmax_decodes_to_that_index(self):
        import torch

        from ptcg_il.model.policy import decode_single_select

        opt_mask = torch.tensor([[True, True, True, False]])
        logits = torch.tensor([[0.1, 5.0, 0.2, -1e9]])
        assert decode_single_select(logits, opt_mask, torch.tensor([2])) == [1]

    def test_masked_options_are_never_chosen(self):
        import torch

        from ptcg_il.model.policy import decode_single_select

        opt_mask = torch.tensor([[True, False, False, False]])
        logits = torch.tensor([[0.0, 9.0, 9.0, 9.0]])
        assert decode_single_select(logits, opt_mask, torch.tensor([-1])) == [0]


# ---------------------------------------------------------------------------
# Defects 3 & 4 — attached Tool / Energy card identity
# ---------------------------------------------------------------------------


class TestAttachedCardIdentity:
    def _obs_with_attachments(self, ep: dict):
        """First ACTIVE obs whose own active Pokémon carries energy cards."""
        steps = ep["steps"]
        for i in range(len(steps) - 1):
            for p, rec in enumerate(steps[i]):
                if rec.get("status") != "ACTIVE":
                    continue
                obs = rec.get("observation")
                if not obs or obs.get("select") is None:
                    continue
                st = obs["current"]
                act = st["players"][st["yourIndex"]].get("active") or []
                if act and isinstance(act[0], dict) and act[0].get("energyCards"):
                    return obs, steps[i + 1][p].get("action", [])
        pytest.fail("fixture has no active Pokemon carrying energy cards")

    def test_energy_card_ids_are_kept_per_slot(self):
        ep = _episode()
        obs, action = self._obs_with_attachments(ep)
        out = featurize(obs, _vocab(ep), action,
                        engine_card_features=_engine_card_features())

        st = obs["current"]
        active = st["players"][st["yourIndex"]]["active"][0]
        expected = [c["id"] for c in active["energyCards"]][:E_MAX]

        assert out["poke_energy_ids"].shape == (P_MAX, E_MAX)
        got = [int(v) for v in out["poke_energy_ids"][0] if v != PAD_CARD]
        assert got == expected

    def test_energy_card_features_are_gathered(self):
        ep = _episode()
        obs, action = self._obs_with_attachments(ep)
        ecf = _engine_card_features()
        out = featurize(obs, _vocab(ep), action, engine_card_features=ecf)

        st = obs["current"]
        active = st["players"][st["yourIndex"]]["active"][0]
        first = active["energyCards"][0]["id"]

        assert out["poke_energy_feat"].shape == (P_MAX, E_MAX, F_CARD)
        np.testing.assert_array_equal(
            out["poke_energy_feat"][0, 0], np.asarray(ecf[first], dtype=np.float32)
        )

    def test_tool_ids_are_kept_per_slot(self):
        """No fixture Pokemon carries a Tool, so attach one and check it survives."""
        ep = _episode()
        obs, action = self._obs_with_attachments(ep)
        obs = copy.deepcopy(obs)
        ecf = _engine_card_features()
        tool_id = next(iter(sorted(ecf)))

        st = obs["current"]
        st["players"][st["yourIndex"]]["active"][0]["tools"] = [{"id": tool_id}]

        out = featurize(obs, _vocab(ep), action, engine_card_features=ecf)
        assert out["poke_tool_ids"].shape == (P_MAX, T_MAX)
        assert out["poke_tool_ids"][0, 0] == tool_id
        np.testing.assert_array_equal(
            out["poke_tool_feat"][0, 0], np.asarray(ecf[tool_id], dtype=np.float32)
        )

    def test_empty_slots_stay_pad(self):
        ep = _episode()
        obs, action = self._obs_with_attachments(ep)
        out = featurize(obs, _vocab(ep), action,
                        engine_card_features=_engine_card_features())

        # A Pokemon slot with no Pokemon in it carries no attachments.
        empty = np.flatnonzero(out["poke_card_id"] == PAD_CARD)
        assert empty.size > 0, "fixture has no empty Pokemon slot"
        for slot in empty:
            assert (out["poke_tool_ids"][slot] == PAD_CARD).all()
            assert (out["poke_energy_ids"][slot] == PAD_CARD).all()

    def test_two_pokemon_differing_only_in_tool_embed_differently(self):
        """The whole point: the model must be able to tell the Tools apart."""
        import torch

        from ptcg_il.model.embed import TokenEmbedder

        ep = _episode()
        obs, action = self._obs_with_attachments(ep)
        ecf = _engine_card_features()
        ids = sorted(ecf)
        tool_a, tool_b = ids[0], ids[len(ids) // 2]

        def _embed(tool_id):
            o = copy.deepcopy(obs)
            st = o["current"]
            st["players"][st["yourIndex"]]["active"][0]["tools"] = [{"id": tool_id}]
            s = featurize(o, _vocab(ep), action, engine_card_features=ecf)
            batch = {
                k: torch.from_numpy(np.asarray(v)[None])
                for k, v in s.items()
                if isinstance(v, np.ndarray)
            }
            for b in ("tok_mask", "opt_mask", "discard_mask"):
                batch[b] = batch[b].bool()
            for i in ("tok_type", "tok_owner", "tok_zone"):
                batch[i] = batch[i].long()
            torch.manual_seed(0)
            return TokenEmbedder(D=32)(batch)[:, 1]  # my active token

        torch.manual_seed(0)
        a = _embed(tool_a)
        torch.manual_seed(0)
        b = _embed(tool_b)
        assert not torch.allclose(a, b), (
            "two different Tool cards produced the same Pokemon token"
        )
