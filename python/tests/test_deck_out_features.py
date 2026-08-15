"""Tests for the deck-out feature set (curve + per-option draw cost)."""
import numpy as np
import pytest


class TestCardDrawCounts:
    """``card_draw_counts`` must read a *card's* skills the way attacks read text.

    Trainers store their oracle text in ``card.skills[].text``, exactly like
    Pokemon abilities -- so the same two regex parsers that feed
    ``attack_static_row[14:16]`` apply, and 38 of the 42 draw-mentioning cards
    in the engine yield a numeric count.  Without this the per-option deck cost
    can only ever be a binary keyword flag.
    """

    def test_parses_a_fixed_draw_count(self):
        from ptcg_mine.keywords import card_draw_counts

        class _Skill:
            text = "Once during your turn, you may draw 3 cards."

        class _Card:
            skills = [_Skill()]

        assert card_draw_counts(_Card()) == (3, 0)

    def test_parses_a_draw_to_hand_size(self):
        from ptcg_mine.keywords import card_draw_counts

        class _Skill:
            text = ("Once during your turn, you may draw cards until you have "
                    "5 cards in your hand.")

        class _Card:
            skills = [_Skill()]

        fixed, to_hand = card_draw_counts(_Card())
        assert to_hand == 5

    def test_a_card_with_no_skills_is_zero(self):
        from ptcg_mine.keywords import card_draw_counts

        class _Card:
            skills = []

        assert card_draw_counts(_Card()) == (0, 0)

    def test_a_card_with_no_skills_attribute_is_zero(self):
        """Engine objects are duck-typed; a missing attribute must not raise."""
        from ptcg_mine.keywords import card_draw_counts

        class _Card:
            pass

        assert card_draw_counts(_Card()) == (0, 0)

    def test_real_engine_cards_yield_counts(self):
        """Fails on zero examined -- the parse must actually hit the corpus."""
        from ptcg_mine.cards import load_engine
        from ptcg_mine.keywords import card_draw_counts

        cd, _ = load_engine()
        cards = list(cd.values()) if hasattr(cd, "values") else list(cd)
        n = sum(1 for c in cards if any(card_draw_counts(c)))
        assert n >= 30, f"only {n} engine cards parsed a draw count (expected ~38)"


class TestCardTableDrawColumns:
    """``card_static_row`` carries the draw counts, and the attack blocks move.

    They are inserted at 83:85, *before* the attack blocks, not appended.
    ``CARD_ATTACK_BLOCK_START`` is defined as ``F_CARD - 3 * F_ATK`` in the
    featurizer and ``52 + K_EFFECT + 2`` in cards.py; appending would make those
    two disagree by exactly 2 and silently reinterpret every attack block as
    starting 2 columns early.
    """

    def test_dims_and_offsets_agree(self):
        from ptcg_il.featurizer import CARD_ATTACK_BLOCK_START, F_ATK, F_CARD
        from ptcg_mine.cards import CARD_ATTACK_BLOCK_START as MINE_START

        assert F_CARD == 223
        assert CARD_ATTACK_BLOCK_START == 85
        assert MINE_START == CARD_ATTACK_BLOCK_START, (
            "cards.py and featurizer.py disagree on where attack blocks start"
        )
        assert CARD_ATTACK_BLOCK_START == F_CARD - 3 * F_ATK

    def test_draw_columns_are_populated_and_normalised(self):
        from ptcg_il.featurizer import (CARD_DRAW_FIXED_COL,
                                        CARD_DRAW_TO_HAND_COL, DRAW_N)
        from ptcg_mine.cards import build_engine_card_features, load_engine

        cd, ad = load_engine()
        ecf = build_engine_card_features(cd, ad)
        rows = np.stack(list(ecf.values()))

        assert rows.shape[1] == 223
        nz = int((rows[:, CARD_DRAW_FIXED_COL] > 0).sum())
        assert nz >= 25, f"only {nz} cards carry a fixed draw count"
        assert rows[:, CARD_DRAW_FIXED_COL].max() <= 1.0, "draw count exceeds 1.0"
        assert rows[:, CARD_DRAW_TO_HAND_COL].max() <= 1.0
        # DRAW_N = 10; a 'draw 3' card must read 0.3, not 3.
        assert 0.0 < rows[:, CARD_DRAW_FIXED_COL].max() <= 1.0
        assert np.isclose(rows[:, CARD_DRAW_FIXED_COL].max() * DRAW_N,
                          round(rows[:, CARD_DRAW_FIXED_COL].max() * DRAW_N)), (
            "draw column is not an integer count over DRAW_N"
        )

    def test_attack_blocks_still_decode_after_the_shift(self):
        """The inserted columns must not corrupt the embedded attack damage."""
        from ptcg_il.featurizer import (ATKDMG_N, CARD_ATTACK_BLOCK_START,
                                        F_ATK)
        from ptcg_mine.cards import (build_engine_attack_features,
                                     build_engine_card_features, load_engine)

        cd, ad = load_engine()
        ecf = build_engine_card_features(cd, ad)
        eaf = build_engine_attack_features(ad)
        cards = list(cd.values()) if hasattr(cd, "values") else list(cd)

        n_checked = 0
        for card in cards:
            aids = list(getattr(card, "attacks", []) or [])[:3]
            if not aids:
                continue
            row = ecf.get(int(card.cardId))
            if row is None:
                continue
            for ai, aid in enumerate(aids):
                atk_row = eaf.get(int(aid))
                if atk_row is None:
                    continue
                off = CARD_ATTACK_BLOCK_START + ai * F_ATK
                assert np.isclose(row[off], atk_row[0]), (
                    f"card {card.cardId} attack {ai}: embedded damage "
                    f"{row[off] * ATKDMG_N} != {atk_row[0] * ATKDMG_N}"
                )
                n_checked += 1
            if n_checked > 400:
                break
        assert n_checked > 0, "no card with attacks examined -- test is vacuous"


def _deck_obs(my_deck: int, opp_deck: int, options=None, my_hand=0):
    """Observation with explicit deck counts on both sides."""
    def player(deck):
        return {"active": [], "bench": [], "benchMax": 5, "deckCount": deck,
                "discard": [], "prize": [None] * 6, "handCount": my_hand,
                "hand": [], "poisoned": False, "burned": False, "asleep": False,
                "paralyzed": False, "confused": False}
    return {
        "select": {"type": 1, "context": 3, "minCount": 1, "maxCount": 1,
                   "option": list(options or [])},
        "logs": [],
        "current": {"turn": 5, "turnActionCount": 0, "yourIndex": 0,
                    "firstPlayer": 0, "supporterPlayed": False,
                    "stadiumPlayed": False, "energyAttached": False,
                    "retreated": False, "result": -1, "stadium": [],
                    "looking": None,
                    "players": [player(my_deck), player(opp_deck)]},
    }


class TestDeckOutCurve:
    """``sum_feat[:, 11] = 1/(1+deckCount)`` -- the hyperbolic companion to f[1].

    ``f[1] = deckCount/DECK_N`` is linear, so deck 50->48 and deck 3->1 are the
    same 0.033 step in feature space and only one of them ends the game.  This
    column is *additional*, not a replacement: the all-zero row is still the PAD
    sentinel, so the fixed-divisor scheme stays intact.
    """

    def test_exported_column_is_an_independent_claim(self):
        from ptcg_il.featurizer import F_SUM, SUM_DECK_OUT_COL

        assert SUM_DECK_OUT_COL == 11
        assert F_SUM == 12

    def test_curve_values(self):
        from ptcg_il.featurizer import SUM_DECK_OUT_COL, _build_summary_tokens

        n_checked = 0
        for deck, expect in ((0, 1.0), (1, 0.5), (2, 1 / 3), (5, 1 / 6),
                             (10, 1 / 11), (40, 1 / 41), (60, 1 / 61)):
            sf = _build_summary_tokens(_deck_obs(deck, 60)["current"], 0)
            assert np.isclose(sf[0][SUM_DECK_OUT_COL], expect), (
                f"deck={deck}: {sf[0][SUM_DECK_OUT_COL]} != {expect}"
            )
            n_checked += 1
        assert n_checked == 7

    def test_it_is_in_range_and_monotone_decreasing(self):
        from ptcg_il.featurizer import SUM_DECK_OUT_COL, _build_summary_tokens

        vals = [float(_build_summary_tokens(_deck_obs(d, 60)["current"], 0)[0][SUM_DECK_OUT_COL])
                for d in range(0, 61)]
        assert all(0.0 <= v <= 1.0 for v in vals), "curve escaped [0, 1]"
        assert all(a > b for a, b in zip(vals, vals[1:])), "curve is not strictly decreasing"

    def test_it_resolves_the_endgame_far_better_than_the_linear_column(self):
        """The whole point: the same 2-card step must move it much more."""
        from ptcg_il.featurizer import SUM_DECK_OUT_COL, _build_summary_tokens

        def pair(d):
            sf = _build_summary_tokens(_deck_obs(d, 60)["current"], 0)
            return float(sf[0][1]), float(sf[0][SUM_DECK_OUT_COL])

        lin_hi, cur_hi = pair(50)
        lin_hi2, cur_hi2 = pair(48)
        lin_lo, cur_lo = pair(3)
        lin_lo2, cur_lo2 = pair(1)

        assert np.isclose(abs(lin_hi - lin_hi2), abs(lin_lo - lin_lo2)), (
            "the linear column should treat both steps identically"
        )
        assert abs(cur_lo - cur_lo2) > 100 * abs(cur_hi - cur_hi2), (
            "the curve does not concentrate resolution near deck-out"
        )

    def test_both_players_get_it(self):
        """Decking the *opponent* is a win condition, so row 1 matters too."""
        from ptcg_il.featurizer import SUM_DECK_OUT_COL, _build_summary_tokens

        sf = _build_summary_tokens(_deck_obs(40, 2)["current"], 0)
        assert np.isclose(sf[0][SUM_DECK_OUT_COL], 1 / 41)
        assert np.isclose(sf[1][SUM_DECK_OUT_COL], 1 / 3)


class TestPerOptionDeckCost:
    """``opt_scalar[:, 13]`` -- how much of my deck *this* option burns.

    dim 8 already carried this for ATTACK options only.  PLAY options are the
    ones that actually deck you out, and they were getting dims 0-5 and nothing
    else.
    """

    def test_exported_column_is_an_independent_claim(self):
        from ptcg_il.featurizer import F_OPT, OPT_DECK_COST_COL

        assert OPT_DECK_COST_COL == 13
        assert F_OPT == 14

    def _draw_card_id(self):
        """An engine card with a non-zero fixed draw count."""
        from ptcg_il.featurizer import CARD_DRAW_FIXED_COL
        from ptcg_mine.cards import build_engine_card_features, load_engine

        cd, ad = load_engine()
        ecf = build_engine_card_features(cd, ad)
        for cid, row in ecf.items():
            if row[CARD_DRAW_FIXED_COL] > 0:
                return cid, float(row[CARD_DRAW_FIXED_COL]), ecf
        pytest.fail("no engine card carries a fixed draw count")

    def test_a_draw_card_costs_more_as_the_deck_empties(self):
        from ptcg_il.featurizer import (DRAW_N, OPT_DECK_COST_COL,
                                        _build_option_tokens)
        from ptcg_il.ref_map import build_ref_map

        cid, dnorm, ecf = self._draw_card_id()
        draw = dnorm * DRAW_N

        vals = {}
        for deck in (40, 10, 4):
            obs = _deck_obs(deck, 60, options=[{"type": 3, "area": 2, "playerIndex": 0, "index": 0, "cardId": cid}])
            state = obs["current"]
            out = _build_option_tokens(
                obs["select"], build_ref_map(obs), 0, None, state=state,
                engine_card_features=ecf,
            )
            vals[deck] = float(out[6][0, OPT_DECK_COST_COL])

        assert np.isclose(vals[40], min(draw / 40, 1.0)), vals
        assert vals[4] > vals[10] > vals[40] > 0.0, (
            f"deck cost did not rise as the deck emptied: {vals}"
        )
        assert vals[4] <= 1.0, "cost escaped [0, 1]"

    def test_a_non_drawing_card_costs_nothing(self):
        from ptcg_il.featurizer import (CARD_DRAW_FIXED_COL,
                                        CARD_DRAW_TO_HAND_COL,
                                        OPT_DECK_COST_COL, _build_option_tokens)
        from ptcg_il.ref_map import build_ref_map
        from ptcg_mine.cards import build_engine_card_features, load_engine

        cd, ad = load_engine()
        ecf = build_engine_card_features(cd, ad)
        quiet = next(c for c, r in ecf.items()
                     if r[CARD_DRAW_FIXED_COL] == 0 and r[CARD_DRAW_TO_HAND_COL] == 0)

        obs = _deck_obs(5, 60, options=[{"type": 3, "area": 2, "playerIndex": 0, "index": 0, "cardId": quiet}])
        state = obs["current"]
        out = _build_option_tokens(
            obs["select"], build_ref_map(obs), 0, None, state=state,
            engine_card_features=ecf,
        )
        assert out[6][0, OPT_DECK_COST_COL] == 0.0

    def test_unresolvable_cards_stay_zero_rather_than_guessing(self):
        """A card we cannot see must not be assigned an invented cost.

        Three distinct routes reach "unresolvable", and each returns before a
        different guard, so all three are exercised: no table at all, an option
        that names no card (PAD), and a card id absent from the table.
        """
        from ptcg_il.featurizer import OPT_DECK_COST_COL, _build_option_tokens
        from ptcg_il.ref_map import build_ref_map
        from ptcg_mine.cards import build_engine_card_features, load_engine

        cd, ad = load_engine()
        ecf = build_engine_card_features(cd, ad)
        missing_id = max(int(c) for c in ecf) + 1000
        assert missing_id not in ecf

        cases = {
            "no table": (None, [{"type": 3, "area": 2, "playerIndex": 0, "index": 0, "cardId": 1121}]),
            "no card named (PAD)": (ecf, [{"type": 3, "area": 9, "index": 0}]),
            "card id absent from table": (ecf, [{"type": 3, "area": 2, "playerIndex": 0, "index": 0, "cardId": missing_id}]),
        }
        for name, (table, options) in cases.items():
            obs = _deck_obs(3, 60, options=options)
            state = obs["current"]
            out = _build_option_tokens(
                obs["select"], build_ref_map(obs), 0, None, state=state,
                engine_card_features=table,
            )
            assert out[6][0, OPT_DECK_COST_COL] == 0.0, (
                f"{name}: invented a deck cost for a card we cannot resolve"
            )


class TestDeckCostOnlyCountsMyOwnPlays:
    """An option that *targets* a Pokémon must never be charged a deck cost.

    ``opt_type == 3`` covers both "play this card from my hand" and "pick this
    Pokémon in play", distinguished only by ``area``.  Both leave the named
    card's id in ``opt_card_id``, so an unguarded lookup reads the *opponent's*
    ability text and divides it by *my* deck size.

    Regression fixture is the real case that exposed it: at a Phantom Dive
    counter-placement select, the option naming the opponent's Teal Mask
    Ogerpon ex scored 0.05 purely because that card's ability says "draw a
    card".  It was the *only* column separating four otherwise-identical
    options, so the pointer would have keyed on it.
    """

    @staticmethod
    def _cost(options, table):
        from ptcg_il.featurizer import OPT_DECK_COST_COL, _build_option_tokens
        from ptcg_il.ref_map import build_ref_map

        obs = _deck_obs(20, 30, options=options)
        out = _build_option_tokens(
            obs["select"], build_ref_map(obs), 0, None, state=obs["current"],
            engine_card_features=table,
        )
        return out[6][:, OPT_DECK_COST_COL]

    def _drawing_card(self):
        from ptcg_il.featurizer import CARD_DRAW_FIXED_COL
        from ptcg_mine.cards import build_engine_card_features, load_engine

        cd, ad = load_engine()
        ecf = build_engine_card_features(cd, ad)
        cid = next(c for c, r in ecf.items() if r[CARD_DRAW_FIXED_COL] > 0)
        return cid, ecf

    def test_targeting_an_opponent_pokemon_costs_nothing(self):
        from ptcg_il.featurizer import AREA_BENCH

        cid, ecf = self._drawing_card()
        cost = self._cost(
            [{"type": 3, "area": AREA_BENCH, "playerIndex": 1, "index": 0,
              "cardId": cid}], ecf)
        assert cost[0] == 0.0, (
            "charged my deck for naming an opponent's Pokémon in play"
        )

    def test_targeting_my_own_pokemon_in_play_costs_nothing_either(self):
        """A promote/switch target is not a play, even though the card is mine."""
        from ptcg_il.featurizer import AREA_BENCH

        cid, ecf = self._drawing_card()
        cost = self._cost(
            [{"type": 3, "area": AREA_BENCH, "playerIndex": 0, "index": 0,
              "cardId": cid}], ecf)
        assert cost[0] == 0.0

    def test_the_same_card_played_from_my_hand_does_cost(self):
        """The guard must not silence the real signal -- fails on zero examined."""
        from ptcg_il.featurizer import AREA_HAND

        cid, ecf = self._drawing_card()
        cost = self._cost(
            [{"type": 3, "area": AREA_HAND, "playerIndex": 0, "index": 0,
              "cardId": cid}], ecf)
        assert cost[0] > 0.0, "a real hand play lost its deck cost"

    def test_my_own_ability_still_costs(self):
        """SKILL options use my Pokémon's ability, and those genuinely draw."""
        cid, ecf = self._drawing_card()
        cost = self._cost([{"type": 15, "cardId": cid, "serial": 10}], ecf)
        assert cost[0] > 0.0, "a SKILL that draws lost its deck cost"
