"""``ptcg_rl.belief`` — the categorical posterior over 𝒟_opp archetypes.

The mechanism under test is **elimination**: a card the opponent played that
archetype *j* does not contain is decisive evidence against *j*.  If that
zeroing does not happen the posterior stays near its prior forever, which looks
like a working-but-cautious belief rather than a broken one — so the tests below
assert the collapse, not merely that a distribution comes out.
"""

from __future__ import annotations

import math

import pytest

from ptcg_rl.belief import CONFIDENT, ArchetypePosterior

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
