"""``ptcg_il.deck_prior`` — the categorical posterior over archetypes.

The mechanism under test is **elimination**: a card the opponent played that
archetype *j* does not contain is decisive evidence against *j*.  If that
zeroing does not happen the posterior stays near its prior forever, which looks
like a working-but-cautious belief rather than a broken one — so the tests below
assert the collapse, not merely that a distribution comes out.
"""

from __future__ import annotations

import math

import pytest

from ptcg_il.deck_prior import CONFIDENT, ArchetypePosterior

# Four hypothesis decks over a tiny card space.  Decks 0 and 3 differ *only* in
# how many copies of card 100 they run, which is what makes multiplicity
# testable: a set-based posterior cannot tell them apart at all.
#   deck 0: 4× card 100
#   deck 1: 4× card 200
#   deck 2: card 300 only
#   deck 3: 2× card 100
ARCHETYPES = {
    "opp_ids": [0, 1, 2, 3],
    "archetypes": [
        {"id": 0, "representative": [100] * 4 + [300] * 56},
        {"id": 1, "representative": [200] * 4 + [300] * 56},
        {"id": 2, "representative": [300] * 60},
        {"id": 3, "representative": [100] * 2 + [300] * 58},
    ],
}


@pytest.fixture
def post() -> ArchetypePosterior:
    return ArchetypePosterior(ARCHETYPES)


class TestPosterior:
    def test_no_observations_returns_the_prior(self, post):
        p = post.posterior([])
        assert set(p) == {0, 1, 2, 3}
        assert all(abs(v - 1 / 4) < 1e-9 for v in p.values())
        assert abs(sum(p.values()) - 1.0) < 1e-9

    def test_a_card_absent_from_a_deck_all_but_zeroes_it(self, post):
        """Card 100 is in decks 0 and 3 only, so 1 and 2 must collapse."""
        p = post.posterior([100])
        assert p[1] < 0.02 and p[2] < 0.02, f"decks without card 100 survived: {p}"
        assert p[0] + p[3] > 0.95

    def test_posterior_sharpens_monotonically_with_evidence(self, post):
        """More copies of a discriminating card must not make belief vaguer."""
        probs = [post.posterior([100] * n)[0] for n in (1, 2, 3, 4)]
        assert probs == sorted(probs), f"belief in deck 0 went down with evidence: {probs}"
        assert probs[-1] > probs[0]

    def test_multiplicity_discriminates_between_2_of_and_4_of(self, post):
        """Decks 0 and 3 differ only in copy count, so only counts separate them.

        A set-based posterior scores these two identically at every observation,
        which is the specific failure this asserts against.  Seeing a third copy
        is decisive: deck 3 cannot explain it at all.
        """
        one = post.posterior([100])
        assert one[0] > one[3], (
            f"the 4-of deck should already lead on one copy, got {one}"
        )

        three = post.posterior([100] * 3)
        assert three[3] < 0.01, f"a 2-of deck survived a third copy: {three}"
        assert three[0] / three[3] > one[0] / one[3], (
            "the odds ratio did not grow with copies — counts are being ignored"
        )

    def test_more_copies_than_any_deck_holds_is_uninformative(self, post):
        """A 5th copy of a card no deck runs 5 of must not shift belief.

        It is unexplainable by *every* hypothesis, so it penalises them all
        equally and cancels in the normalisation.  Worth pinning: the intuitive
        expectation is that it should count against the 4-of deck, and a
        posterior that did that would be double-counting evidence it has
        already used.
        """
        four = post.posterior([100] * 4)
        five = post.posterior([100] * 5)
        assert all(abs(four[k] - five[k]) < 1e-9 for k in four), (
            f"an unexplainable copy shifted the posterior: {four} → {five}"
        )

    def test_shared_card_is_uninformative_between_decks_that_both_run_it(self, post):
        """Card 300 is in all three decks, so it should barely discriminate."""
        p = post.posterior([300])
        spread = max(p.values()) - min(p.values())
        assert spread < 0.2, f"a common card discriminated too strongly: {p}"

    def test_result_is_always_a_normalised_distribution(self, post):
        for observed in ([], [100], [999], [100, 200], [300] * 20, [999] * 5):
            p = post.posterior(observed)
            assert abs(sum(p.values()) - 1.0) < 1e-9, f"unnormalised for {observed}: {p}"
            assert all(v >= 0.0 for v in p.values())

    def test_impossible_observation_falls_back_to_the_prior(self, post):
        """A card in no decklist must not produce NaNs or an empty posterior."""
        p = post.posterior([999] * 3)
        assert abs(sum(p.values()) - 1.0) < 1e-9
        assert all(math.isfinite(v) for v in p.values())

    def test_contradictory_evidence_does_not_crash(self, post):
        """Cards exclusive to two different decks at once."""
        p = post.posterior([100, 200])
        assert abs(sum(p.values()) - 1.0) < 1e-9
        assert p[2] < max(p[0], p[1]), "the deck containing neither card should not lead"


class TestDeckTemplate:
    def test_returns_sixty_cards_when_confident(self, post):
        template = post.deck_template([100] * 4)
        assert template is not None
        assert len(template) == 60
        assert template.count(100) == 4

    def test_returns_none_when_unsure(self, post):
        """A vague belief must yield no template at all.

        RL_SPEC §8.1: a confidently wrong decklist is worse for the determinizer
        than no decklist, because every rollout through an impossible world is
        wasted.
        """
        assert post.deck_template([]) is None
        assert not post.is_confident([])

    def test_confidence_threshold_is_respected(self, post):
        """One copy of card 100 is not enough; three is.

        Decks 0 and 3 both run card 100, so a single copy leaves real ambiguity
        and the oracle must decline to hand over a template.
        """
        aid_one, p_one = post.most_likely([100])
        assert p_one < CONFIDENT, f"one ambiguous card should not be conclusive ({p_one})"
        assert post.deck_template([100]) is None

        aid_three, p_three = post.most_likely([100] * 3)
        assert aid_three == 0 and p_three >= CONFIDENT
        assert post.deck_template([100] * 3) is not None


class TestConstruction:
    def test_rejects_an_archetype_with_no_representative(self):
        bad = {"opp_ids": [0, 7], "archetypes": [{"id": 0, "representative": [1] * 60}]}
        with pytest.raises(ValueError, match="representative"):
            ArchetypePosterior(bad)

    def test_rejects_an_empty_hypothesis_set(self):
        with pytest.raises(ValueError, match="hypothesis set"):
            ArchetypePosterior({"opp_ids": [], "archetypes": []})

    def test_rejects_mismatched_priors(self):
        with pytest.raises(ValueError, match="priors"):
            ArchetypePosterior(ARCHETYPES, priors=[0.5, 0.5, 0.0])

    def test_non_uniform_priors_shift_the_no_evidence_posterior(self):
        p = ArchetypePosterior(ARCHETYPES, priors=[0.7, 0.1, 0.1, 0.1]).posterior([])
        assert abs(p[0] - 0.7) < 1e-9

    def test_evidence_can_overturn_a_prior(self):
        """A strong prior must not survive a decisive observation."""
        p = ArchetypePosterior(ARCHETYPES, priors=[0.85, 0.05, 0.05, 0.05]).posterior([200] * 4)
        assert p[1] > p[0], f"prior on deck 0 survived decisive evidence for deck 1: {p}"

    def test_accepts_the_dict_shaped_archetypes_container(self):
        """archetypes.json has carried both a list and an id-keyed object."""
        as_dict = {
            "opp_ids": [0, 1],
            "archetypes": {
                "0": {"representative": [100] * 60},
                "1": {"representative": [200] * 60},
            },
        }
        p = ArchetypePosterior(as_dict).posterior([100])
        assert p[0] > 0.99


class TestRealArtifacts:
    """The posterior must work on this corpus's actual archetypes.json."""

    def test_collapses_on_the_real_opponent_set(self):
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "data" / "archetypes.json"
        if not path.exists():
            pytest.skip("no mined archetypes.json in python/data/")

        with open(path) as f:
            archetypes = json.load(f)
        post = ArchetypePosterior(archetypes)
        assert len(post.opp_ids) >= 2

        # Feed a hypothesis its own cards; belief in it must reach certainty.
        n_checked = 0
        for i, aid in enumerate(post.opp_ids):
            deck = list(post._counts[i].elements())
            p = post.posterior(deck[:20])
            assert p[aid] == max(p.values()), (
                f"archetype {aid} did not lead after seeing 20 of its own cards"
            )
            n_checked += 1
        assert n_checked >= 2, "examined fewer than two archetypes — test proves little"


from ptcg_il.deck_prior import all_archetype_ids, frequency_prior

# `frequency` is appearances (player-slots), not distinct decklists — that is
# `n_members`.  Getting these two confused would prior the posterior on how
# many *variants* a cluster has rather than how often you face it.
FREQ_ARCHETYPES = {
    "opp_ids": [0, 1],
    "archetypes": [
        {"id": 0, "representative": [100] * 60, "frequency": 30, "n_members": 9},
        {"id": 1, "representative": [200] * 60, "frequency": 10, "n_members": 1},
        {"id": 7, "representative": [300] * 60, "frequency": 60, "n_members": 3},
    ],
}


class TestAllArchetypeIds:
    def test_returns_every_id_not_just_opp_ids(self):
        # The whole point of the change: the hypothesis set is the file, not
        # its `opp_ids` subset.
        assert all_archetype_ids(FREQ_ARCHETYPES) == [0, 1, 7]

    def test_handles_the_id_keyed_dict_container_form(self):
        as_dict = {"archetypes": {"5": {"representative": [1] * 60},
                                  "2": {"representative": [2] * 60}}}
        assert all_archetype_ids(as_dict) == [2, 5]

    def test_empty_file_yields_no_ids(self):
        assert all_archetype_ids({"archetypes": []}) == []


class TestFrequencyPrior:
    def test_reads_frequency_not_n_members(self):
        p = frequency_prior(FREQ_ARCHETYPES, [0, 1, 7])
        assert p == [30.0, 10.0, 60.0]

    def test_follows_the_requested_id_order(self):
        assert frequency_prior(FREQ_ARCHETYPES, [7, 0]) == [60.0, 30.0]

    def test_zero_frequency_is_floored_not_made_impossible(self):
        """A re-mine retains baseline clusters at frequency 0 (see CLAUDE.md).

        ArchetypePosterior maps a non-positive prior to -inf, which is a
        permanent, silent elimination: if the opponent actually plays that
        deck, no amount of evidence can ever recover it.
        """
        archetypes = {
            "archetypes": [
                {"id": 0, "representative": [100] * 60, "frequency": 1000},
                {"id": 1, "representative": [200] * 60, "frequency": 0},
            ],
        }
        p = frequency_prior(archetypes, [0, 1])
        assert p[1] > 0.0, "a retained baseline cluster must stay reachable"
        assert p[1] < p[0]

        # And it must actually be recoverable through the posterior.
        post = ArchetypePosterior(archetypes, opp_ids=[0, 1], priors=p)
        assert post.most_likely([200] * 4)[0] == 1

    def test_all_zero_frequencies_degrade_to_uniform(self):
        archetypes = {
            "archetypes": [
                {"id": 0, "representative": [100] * 60, "frequency": 0},
                {"id": 1, "representative": [200] * 60, "frequency": 0},
            ],
        }
        assert frequency_prior(archetypes, [0, 1]) == [1.0, 1.0]

    def test_missing_frequency_key_is_treated_as_zero(self):
        archetypes = {"archetypes": [
            {"id": 0, "representative": [100] * 60, "frequency": 5},
            {"id": 1, "representative": [200] * 60},
        ]}
        p = frequency_prior(archetypes, [0, 1])
        assert p[0] == 5.0
        assert 0.0 < p[1] < 5.0


from collections import Counter

from ptcg_il.deck_prior import (
    ObservedOpponentCards,
    observed_cards_from_obs,
    observed_cards_from_state,
)


def _obs(state: dict, your_index: int = 0) -> dict:
    state = dict(state)
    state["yourIndex"] = your_index
    return {"current": state, "select": {}}


class TestObservedCardsFromState:
    def test_reads_the_opponents_zones_not_ours(self):
        state = {
            "players": [
                {"active": [{"id": 11}], "bench": [], "discard": [{"id": 12}]},
                {"active": [{"id": 21}], "bench": [{"id": 22}],
                 "discard": [{"id": 23}], "prize": [{"id": 24}]},
            ],
        }
        assert observed_cards_from_state(state, your_index=0) == Counter(
            {21: 1, 22: 1, 23: 1, 24: 1})

    def test_face_down_cards_are_none_and_are_skipped(self):
        """AGENT_SPEC.md:57 — face-down active and prizes arrive as None.

        Counting them would be reading hidden information.
        """
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [None], "bench": [],
                 "discard": [], "prize": [None, {"id": 24}, None]},
            ],
        }
        assert observed_cards_from_state(state, your_index=0) == Counter({24: 1})

    def test_counts_copies_not_distinct_ids(self):
        """Multiplicity is the evidence a set-based check throws away."""
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [],
                 "discard": [{"id": 30}, {"id": 30}, {"id": 30}]},
            ],
        }
        assert observed_cards_from_state(state, your_index=0)[30] == 3

    def test_attached_energy_and_tools_are_counted(self):
        """They come out of the opponent's own deck, so they are evidence."""
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [{"id": 21, "playerIndex": 1,
                             "energyCards": [{"id": 90, "playerIndex": 1}],
                             "tools": [{"id": 91, "playerIndex": 1}],
                             "preEvolution": [{"id": 92, "playerIndex": 1}]}],
                 "bench": [], "discard": []},
            ],
        }
        counts = observed_cards_from_state(state, your_index=0)
        assert counts == Counter({21: 1, 90: 1, 91: 1, 92: 1})

    def test_energy_we_attached_to_their_pokemon_is_not_their_card(self):
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [{"id": 21, "playerIndex": 1,
                             "energyCards": [{"id": 90, "playerIndex": 0}]}],
                 "bench": [], "discard": []},
            ],
        }
        counts = observed_cards_from_state(state, your_index=0)
        assert counts[21] == 1
        assert counts[90] == 0, "our Energy on their Pokemon is ours"

    def test_stadium_is_a_bare_dict_and_counts_when_they_played_it(self):
        """`state["stadium"]` is one card, not a list — iterating it as a list
        walks its string keys and silently contributes nothing."""
        state = {
            "players": [{"active": [], "bench": [], "discard": []}] * 2,
            "stadium": {"id": 102, "playerIndex": 1},
        }
        assert observed_cards_from_state(state, your_index=0) == Counter({102: 1})

    def test_our_own_stadium_is_not_evidence_about_them(self):
        """The decisive case: a Stadium we played, mis-credited, is a card the
        true archetype cannot explain — an epsilon-weighted false elimination
        of the correct answer."""
        state = {
            "players": [{"active": [], "bench": [], "discard": []}] * 2,
            "stadium": {"id": 102, "playerIndex": 0},
        }
        assert observed_cards_from_state(state, your_index=0) == Counter()

    def test_unowned_cards_are_credited_to_the_zone_they_sit_in(self):
        """No `playerIndex` means the engine did not say; a card in their
        discard is theirs regardless."""
        state = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [], "discard": [{"id": 55}]},
            ],
        }
        assert observed_cards_from_state(state, your_index=0) == Counter({55: 1})

    def test_seat_one_reads_seat_zero(self):
        state = {
            "players": [
                {"active": [], "bench": [], "discard": [{"id": 12}]},
                {"active": [], "bench": [], "discard": [{"id": 23}]},
            ],
        }
        assert observed_cards_from_state(state, your_index=1) == Counter({12: 1})

    def test_missing_opponent_yields_nothing(self):
        assert observed_cards_from_state({"players": []}, your_index=0) == Counter()
        assert observed_cards_from_state({}, your_index=0) == Counter()


class TestObservedCardsFromObs:
    def test_reads_your_index_off_current(self):
        state = {
            "players": [
                {"active": [], "bench": [], "discard": [{"id": 12}]},
                {"active": [], "bench": [], "discard": [{"id": 23}]},
            ],
        }
        assert observed_cards_from_obs(_obs(state, 0)) == Counter({23: 1})
        assert observed_cards_from_obs(_obs(state, 1)) == Counter({12: 1})

    def test_empty_observation_yields_nothing(self):
        assert observed_cards_from_obs({}) == Counter()


class TestObservedOpponentCards:
    def _state(self, discard_ids: list[int]) -> dict:
        return {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [],
                 "discard": [{"id": i} for i in discard_ids]},
            ],
        }

    def test_accumulates_cards_that_leave_view(self):
        """A card seen on the bench that later returns to hand is still
        evidence — the snapshot loses it, the accumulator keeps it."""
        seen = ObservedOpponentCards()
        seen.observe(_obs(self._state([40])))
        seen.observe(_obs(self._state([])))
        assert seen.counts() == Counter({40: 1})

    def test_a_card_moving_zone_is_not_double_counted(self):
        """The load-bearing guard.  Summing snapshots manufactures a second
        copy, and a phantom copy the true archetype cannot explain costs it a
        factor of epsilon — eliminating the right answer with invented
        evidence.  Per-card maximum, never sum.
        """
        active = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [{"id": 41}], "bench": [], "discard": []},
            ],
        }
        discarded = {
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [], "discard": [{"id": 41}]},
            ],
        }
        seen = ObservedOpponentCards()
        seen.observe(_obs(active))
        seen.observe(_obs(discarded))
        assert seen.counts()[41] == 1, "same card, two zones, one copy"

    def test_a_genuine_second_copy_is_counted(self):
        """The max must still rise when the opponent really shows two."""
        seen = ObservedOpponentCards()
        seen.observe(_obs(self._state([42])))
        seen.observe(_obs(self._state([42, 42])))
        assert seen.counts()[42] == 2

    def test_reset_clears_between_games(self):
        seen = ObservedOpponentCards()
        seen.observe(_obs(self._state([43])))
        seen.reset()
        assert seen.counts() == Counter()

    def test_multiset_expands_every_copy(self):
        seen = ObservedOpponentCards()
        seen.observe(_obs(self._state([44, 44, 45])))
        assert sorted(seen.multiset()) == [44, 44, 45]

    def test_counts_returns_a_copy_not_the_internal_state(self):
        seen = ObservedOpponentCards()
        seen.observe(_obs(self._state([46])))
        seen.counts()[46] = 99
        assert seen.counts()[46] == 1


class TestAddCards:
    """`ptcg_rl` records visible cards as a flat id list, not an observation."""

    def test_adds_a_raw_id_list(self):
        seen = ObservedOpponentCards()
        seen.add_cards([50, 50, 51])
        assert seen.counts() == Counter({50: 2, 51: 1})

    def test_combines_by_maximum_like_observe(self):
        """Same rule as observe: an id list is one snapshot of the visible
        zones, so re-adding it must not double the counts."""
        seen = ObservedOpponentCards()
        seen.add_cards([52, 52])
        seen.add_cards([52, 52])
        assert seen.counts()[52] == 2

    def test_a_larger_later_snapshot_raises_the_count(self):
        seen = ObservedOpponentCards()
        seen.add_cards([53])
        seen.add_cards([53, 53, 53])
        assert seen.counts()[53] == 3

    def test_interoperates_with_observe(self):
        seen = ObservedOpponentCards()
        seen.observe(_obs({
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [], "discard": [{"id": 54}]},
            ],
        }))
        seen.add_cards([55])
        assert seen.counts() == Counter({54: 1, 55: 1})

    def test_reset_clears_added_cards(self):
        seen = ObservedOpponentCards()
        seen.add_cards([56])
        seen.reset()
        assert seen.counts() == Counter()

    def test_empty_list_is_a_no_op(self):
        seen = ObservedOpponentCards()
        seen.add_cards([57])
        seen.add_cards([])
        assert seen.counts() == Counter({57: 1})


import numpy as np

from ptcg_il.deck_prior import OpponentDeckPredictor

# Three archetypes over a tiny card space, with a lopsided prior so the
# frequency/uniform distinction is observable.
PRED_ARCHETYPES = {
    "opp_ids": [0],
    "archetypes": [
        {"id": 0, "representative": [100] * 60, "frequency": 900},
        {"id": 1, "representative": [200] * 60, "frequency": 50},
        {"id": 2, "representative": [300] * 60, "frequency": 50},
    ],
}


class TestPredictorDefaults:
    def test_hypothesis_set_is_every_archetype_not_opp_ids(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        assert pred.ids == [0, 1, 2]

    def test_frequency_prior_is_the_default(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        p = pred.posterior()
        assert p[0] > 0.85, f"frequency prior should dominate before evidence: {p}"

    def test_uniform_prior_can_be_asked_for(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES, use_frequency_prior=False)
        p = pred.posterior()
        assert all(abs(v - 1 / 3) < 1e-9 for v in p.values())


class TestTemplateNeverAbstains:
    def test_commits_with_zero_observations(self):
        """The never-abstain guarantee.  The alternative the old code fell back
        to — an empty list, i.e. the Rust mirror heuristic — is worth 19.4/60
        against this argmax's 44.6/60, so there is no k at which abstaining
        pays."""
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        deck = pred.template()
        assert len(deck) == 60
        assert set(deck) == {100}, "the frequency argmax before any evidence"

    def test_commits_when_the_posterior_is_flat(self):
        flat = {"archetypes": [
            {"id": 0, "representative": [100] * 60, "frequency": 10},
            {"id": 1, "representative": [200] * 60, "frequency": 10},
        ]}
        deck = OpponentDeckPredictor(flat).template()
        assert len(deck) == 60

    def test_never_returns_an_empty_list(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        for ids in ([], [999], [100] * 4, [200] * 4, [100, 200, 300]):
            pred.reset()
            pred.observe_cards(ids)
            assert pred.template(), f"abstained on {ids}"

    def test_evidence_overrides_the_prior(self):
        """Elimination is the mechanism: four copies of card 200 rule out the
        900-frequency favourite outright."""
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        pred.observe_cards([200] * 4)
        assert set(pred.template()) == {200}

    def test_an_unknown_card_does_not_abstain(self):
        """A card no archetype contains leaves every hypothesis equally
        penalised; the predictor must still commit."""
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        pred.observe_cards([999])
        assert len(pred.template()) == 60


class TestPredictorObservation:
    def _obs_with(self, card_id: int, n: int = 1) -> dict:
        return {"current": {
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [],
                 "discard": [{"id": card_id} for _ in range(n)]},
            ],
        }}

    def test_observe_then_template_tracks_the_evidence(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        assert set(pred.template()) == {100}
        pred.observe(self._obs_with(300, 4))
        assert set(pred.template()) == {300}

    def test_reset_returns_to_the_prior(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        pred.observe(self._obs_with(300, 4))
        pred.reset()
        assert set(pred.template()) == {100}

    def test_template_is_a_pure_function_of_the_observation_sequence(self):
        a, b = OpponentDeckPredictor(PRED_ARCHETYPES), OpponentDeckPredictor(PRED_ARCHETYPES)
        for obs in (self._obs_with(200, 1), self._obs_with(200, 2)):
            a.observe(obs)
            b.observe(obs)
        assert a.template() == b.template()

    def test_a_returned_template_cannot_mutate_the_predictor(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        pred.template().append(999)
        assert len(pred.template()) == 60


class TestSampleTemplates:
    def test_returns_k_decks_of_sixty(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        decks = pred.sample_templates(8, np.random.default_rng(0))
        assert len(decks) == 8
        assert all(len(d) == 60 for d in decks)

    def test_is_reproducible_under_a_seeded_rng(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        a = pred.sample_templates(8, np.random.default_rng(7))
        b = pred.sample_templates(8, np.random.default_rng(7))
        assert a == b

    def test_a_diffuse_posterior_yields_varied_worlds(self):
        """The reason this call site does not use argmax: K identical
        determinizations collapse `mcts_k_determinizations` to 1 while leaving
        the config knob looking effective."""
        flat = {"archetypes": [
            {"id": 0, "representative": [100] * 60, "frequency": 10},
            {"id": 1, "representative": [200] * 60, "frequency": 10},
            {"id": 2, "representative": [300] * 60, "frequency": 10},
        ]}
        decks = OpponentDeckPredictor(flat).sample_templates(
            50, np.random.default_rng(0))
        assert len({tuple(d) for d in decks}) > 1

    def test_a_collapsed_posterior_yields_the_same_world(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        pred.observe_cards([200] * 4)
        decks = pred.sample_templates(20, np.random.default_rng(0))
        assert {tuple(d) for d in decks} == {tuple([200] * 60)}

    def test_zero_k_is_no_worlds(self):
        pred = OpponentDeckPredictor(PRED_ARCHETYPES)
        assert pred.sample_templates(0, np.random.default_rng(0)) == []
