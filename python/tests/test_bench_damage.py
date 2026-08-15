"""Bench damage is stated in text, never in the attack's ``damage`` field."""
import numpy as np
import pytest


class _A:
    def __init__(self, text, damage=0):
        self.text = text
        self.damage = damage


class TestAttackBenchDamage:
    """Measured over the engine: 27 attacks deal damage to a benched Pokemon,
    and in every one of them ``damage`` is the *Active* number.  The bench
    figure appears only in the oracle text, in three forms.
    """

    def test_also_does_n_to_one_benched(self):
        from ptcg_mine.keywords import attack_bench_damage
        # Dirty Beam: active 160, bench 30
        assert attack_bench_damage(_A(
            "This attack also does 30 damage to 1 of your opponent’s Benched "
            "Pokémon. (Don’t apply Weakness and Resistance.)", 160)) == 30

    def test_pure_snipe_with_a_zero_damage_field(self):
        from ptcg_mine.keywords import attack_bench_damage
        # Pinpoint Dive: damage field is 0; all of it lands on the bench.
        assert attack_bench_damage(_A(
            "This attack does 60 damage to 1 of your opponent’s Benched "
            "Pokémon {ex} or Benched Pokémon V.", 0)) == 60

    def test_damage_to_each_benched(self):
        from ptcg_mine.keywords import attack_bench_damage
        assert attack_bench_damage(_A(
            "This attack does 50 damage to each of your opponent’s Benched "
            "Pokémon. (Don’t apply Weakness and Resistance.)", 0)) == 50

    def test_put_n_damage_counters_is_ten_hp_each(self):
        from ptcg_mine.keywords import attack_bench_damage
        # Phantom Dive: 6 counters = 60 HP if all are aimed at one target.
        assert attack_bench_damage(_A(
            "Put 6 damage counters on your opponent’s Benched Pokémon in any "
            "way you like.", 200)) == 60

    def test_worded_counter_counts(self):
        from ptcg_mine.keywords import attack_bench_damage
        assert attack_bench_damage(_A(
            "Put 2 damage counters on your opponent’s Benched Pokémon.", 90)) == 20

    def test_an_attack_with_no_bench_clause_is_zero(self):
        from ptcg_mine.keywords import attack_bench_damage
        assert attack_bench_damage(_A("This attack does 30 damage.", 30)) == 0
        assert attack_bench_damage(_A("", 100)) == 0

    def test_moving_energy_to_the_bench_is_not_bench_damage(self):
        """'Move an Energy to 1 of your Benched Pokemon' must not match."""
        from ptcg_mine.keywords import attack_bench_damage
        assert attack_bench_damage(_A(
            "Move an Energy from this Pokémon to 1 of your Benched Pokémon.",
            140)) == 0

    def test_benching_a_pokemon_is_not_bench_damage(self):
        from ptcg_mine.keywords import attack_bench_damage
        assert attack_bench_damage(_A(
            "Search your deck for up to 2 Froakie and put them onto your Bench. "
            "Then, shuffle your deck.", 0)) == 0

    def test_real_engine_population(self):
        """Fails on zero examined; pins the measured count."""
        from ptcg_mine.cards import load_engine
        from ptcg_mine.keywords import attack_bench_damage

        _cd, ad = load_engine()
        atks = list(ad.values()) if hasattr(ad, "values") else list(ad)
        hits = [a for a in atks if attack_bench_damage(a) > 0]
        assert len(hits) >= 20, f"only {len(hits)} bench-damage attacks parsed"
        vals = [attack_bench_damage(a) for a in hits]
        # Thunder Raid really does 210 to a benched ex after discarding all Energy.
        assert min(vals) >= 10 and max(vals) <= 250, f"implausible range {min(vals)}..{max(vals)}"

    def test_phantom_dive_specifically(self):
        """Kh0a's primary attack -- the case that motivated this."""
        from ptcg_mine.cards import load_engine
        from ptcg_mine.keywords import attack_bench_damage

        _cd, ad = load_engine()
        atks = {a.attackId: a for a in (ad.values() if hasattr(ad, "values") else ad)}
        assert attack_bench_damage(atks[154]) == 60
        assert atks[154].damage == 200, "damage field must stay the Active number"


class TestAttackTableBenchColumn:
    def test_dims_shifted_consistently(self):
        from ptcg_il.featurizer import ATK_BENCH_DMG_COL, CARD_ATTACK_BLOCK_START, F_ATK, F_CARD
        from ptcg_mine.keywords import K_EFFECT

        assert ATK_BENCH_DMG_COL == 16
        assert F_ATK == 17 + K_EFFECT == 46
        assert F_CARD == 223
        assert CARD_ATTACK_BLOCK_START == F_CARD - 3 * F_ATK == 85

    def test_column_is_populated_and_normalised(self):
        from ptcg_il.featurizer import ATKDMG_N, ATK_BENCH_DMG_COL
        from ptcg_mine.cards import build_engine_attack_features, load_engine

        _cd, ad = load_engine()
        eaf = build_engine_attack_features(ad)
        rows = np.stack(list(eaf.values()))
        col = rows[:, ATK_BENCH_DMG_COL]
        assert (col > 0).sum() >= 20, "bench damage column is empty"
        assert col.max() <= 1.0, "bench damage escaped [0, 1]"
        assert np.isclose(col.max() * ATKDMG_N, round(col.max() * ATKDMG_N))

    def test_keywords_still_decode_after_the_shift(self):
        """Keyword flags moved 16->17; a stale offset reads the wrong effect."""
        from ptcg_il.featurizer import F_ATK
        from ptcg_mine.cards import build_engine_attack_features, load_engine
        from ptcg_mine.keywords import K_EFFECT, KEYWORD_NAMES, attack_keyword_row

        _cd, ad = load_engine()
        eaf = build_engine_attack_features(ad)
        atks = {a.attackId: a for a in (ad.values() if hasattr(ad, "values") else ad)}

        n = 0
        for aid, row in eaf.items():
            a = atks.get(int(aid))
            if a is None:
                continue
            np.testing.assert_allclose(row[17:17 + K_EFFECT], attack_keyword_row(a))
            n += 1
            if n > 300:
                break
        assert n > 0, "no attack examined -- test is vacuous"


def _boards(my_active, opp_active, opp_bench):
    from ptcg_il.featurizer import F_CARD, _build_poke_tokens, _feat_gatherer
    from ptcg_mine.cards import build_engine_card_features, load_engine
    cd, ad = load_engine()
    ecf = build_engine_card_features(cd, ad)

    def poke(cid, s, hp, en=()):
        en = list(en)
        return {"id": cid, "serial": s, "hp": hp, "maxHp": hp, "appearThisTurn": False,
                "energies": en,
                "energyCards": [{"id": 1, "serial": 900 + i, "playerIndex": 0}
                                for i in range(len(en))],
                "tools": [], "preEvolution": []}

    def pl(a, b):
        return {"active": a, "bench": b, "benchMax": 5, "deckCount": 40, "discard": [],
                "prize": [None] * 6, "handCount": 0, "hand": [], "poisoned": False,
                "burned": False, "asleep": False, "paralyzed": False, "confused": False}

    state = {"turn": 9, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
             "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
             "retreated": False, "result": -1, "stadium": [], "looking": None,
             "players": [pl([poke(*my_active)], []),
                         pl([poke(*opp_active)], [poke(*m) for m in opp_bench])]}
    pid, pf, _, _ = _build_poke_tokens(state, 0)
    return _feat_gatherer(ecf, F_CARD)(pid), pf


class TestAttackDamageRatioUsesBenchDamage:
    """``_attack_damage_ratio`` scored bench targets with the *Active* number.

    For Phantom Dive into a 70 HP benched Pokémon that returned ratio 2.0 and
    set the KO flag; the truth is at most 60 damage, ratio 0.857, no KO.  This
    is wrong information rather than missing information, and it fired on the
    primary attack of archetype 16.
    """

    def test_bench_target_uses_bench_damage_not_active_damage(self):
        from ptcg_il.featurizer import _attack_damage_ratio
        from ptcg_mine.cards import build_engine_attack_features, load_engine

        _cd, ad = load_engine()
        eaf = build_engine_attack_features(ad)
        # Dragapult ex active; opponent Active 340 HP; opponent bench Dreepy 70 HP.
        pcf, pf = _boards((121, 10, 320, [2, 5]), (756, 20, 340), [(119, 21, 70)])

        r_active = _attack_damage_ratio(154, 6, pcf, pf, eaf)
        r_bench = _attack_damage_ratio(154, 7, pcf, pf, eaf)

        assert np.isclose(r_active, 200.0 / 340.0, atol=1e-4), (
            f"Active reading changed: {r_active}"
        )
        assert np.isclose(r_bench, 60.0 / 70.0, atol=1e-4), (
            f"bench target should read 60/70, got {r_bench}"
        )
        assert r_bench < 1.0, "bench target must no longer read as a KO"

    def test_an_attack_that_cannot_reach_the_bench_scores_zero_there(self):
        from ptcg_il.featurizer import _attack_damage_ratio
        from ptcg_mine.cards import build_engine_attack_features, load_engine
        from ptcg_mine.keywords import attack_bench_damage

        _cd, ad = load_engine()
        eaf = build_engine_attack_features(ad)
        atks = {a.attackId: a for a in (ad.values() if hasattr(ad, "values") else ad)}
        plain = next(aid for aid, a in atks.items()
                     if getattr(a, "damage", 0) > 0 and attack_bench_damage(a) == 0)

        pcf, pf = _boards((121, 10, 320, [2, 5]), (756, 20, 340), [(119, 21, 70)])
        assert _attack_damage_ratio(plain, 7, pcf, pf, eaf) == 0.0, (
            "an attack with no bench clause must not reach the bench"
        )
        assert _attack_damage_ratio(plain, 6, pcf, pf, eaf) > 0.0, (
            "the Active reading must be unaffected"
        )

    def test_bench_damage_ignores_weakness(self):
        """25 of the 27 spell out "Don't apply Weakness and Resistance".

        The attacker/target pair here is chosen so weakness *would* fire: Dirty
        Beam's Farigiraf ex (83) is energy type 7, and Smoochum (183) is weak to
        type 7.  A weakness-adjusted reading would double the 30 to 60, so this
        fixture is only meaningful because the pairing actually matches -- with
        Dragapult as attacker nothing in the pool is weak to its type and the
        assertion would hold vacuously.
        """
        from ptcg_il.featurizer import (_CARD_ENERGYTYPE, _CARD_WEAKNESS,
                                        _attack_damage_ratio, _onehot_index)
        from ptcg_mine.cards import build_engine_attack_features, load_engine

        _cd, ad = load_engine()
        eaf = build_engine_attack_features(ad)
        pcf, pf = _boards((83, 10, 300, [2, 5]), (756, 20, 340), [(183, 21, 60)])

        assert _onehot_index(pcf[0][_CARD_ENERGYTYPE]) == \
            _onehot_index(pcf[7][_CARD_WEAKNESS]), (
            "fixture no longer pairs an attacker type with the target's weakness"
        )
        r = _attack_damage_ratio(99, 7, pcf, pf, eaf)
        assert np.isclose(r, 30.0 / 60.0, atol=1e-4), (
            f"bench damage was weakness-adjusted: {r} (expected {30/60})"
        )


class TestCountersToKO:
    """``poke_feat[28]`` = 1/(1+ceil(hp/10)) -- counters needed to KO this slot.

    A damage counter is 10 HP, so this is the natural unit for every
    counter-placement decision.  Measured on a real Dragapult game, **18 of
    Kh0a's 96 decisions (19%)** are exactly that: `select(type=1, context=14)`
    with `remainDamageCounter` counting 6 down to 1, choosing among 4-5 benched
    Pokemon.

    Reciprocal for the same reason the deck-out curve is: `hp/HP_N` is linear,
    so 70 HP and 10 HP differ by 0.15 while "1 counter" and "7 counters" is the
    whole decision.
    """

    def test_exported_column(self):
        from ptcg_il.featurizer import F_POKE, POKE_COUNTERS_TO_KO_COL

        assert POKE_COUNTERS_TO_KO_COL == 28
        assert F_POKE == 29

    def test_values(self):
        from ptcg_il.featurizer import POKE_COUNTERS_TO_KO_COL, _build_poke_tokens

        n = 0
        for hp, counters in ((10, 1), (20, 2), (60, 6), (70, 7), (110, 11), (340, 34)):
            pcf, pf = _boards((121, 10, 320, [2, 5]), (756, 20, 340), [(119, 21, hp)])
            got = float(pf[7][POKE_COUNTERS_TO_KO_COL])
            assert np.isclose(got, 1.0 / (1.0 + counters), atol=1e-6), (
                f"hp={hp}: expected {counters} counters -> {1/(1+counters)}, got {got}"
            )
            n += 1
        assert n == 6

    def test_partial_counters_round_up(self):
        """55 HP needs 6 counters, not 5 -- ceil, not floor."""
        from ptcg_il.featurizer import POKE_COUNTERS_TO_KO_COL, _build_poke_tokens

        pcf, pf = _boards((121, 10, 320, [2, 5]), (756, 20, 340), [(119, 21, 55)])
        assert np.isclose(float(pf[7][POKE_COUNTERS_TO_KO_COL]), 1.0 / 7.0, atol=1e-6)

    def test_empty_slots_stay_zero(self):
        from ptcg_il.featurizer import POKE_COUNTERS_TO_KO_COL

        pcf, pf = _boards((121, 10, 320, [2, 5]), (756, 20, 340), [])
        n = 0
        for s in list(range(1, 6)) + list(range(7, 12)):
            assert float(pf[s][POKE_COUNTERS_TO_KO_COL]) == 0.0
            n += 1
        assert n == 10

    def test_it_resolves_the_finishable_range(self):
        """The point: 1-vs-2 counters must separate far more than 30-vs-31."""
        from ptcg_il.featurizer import HP_N, POKE_COUNTERS_TO_KO_COL

        def val(hp):
            _p, pf = _boards((121, 10, 320, [2, 5]), (756, 20, 340), [(119, 21, hp)])
            return float(pf[7][POKE_COUNTERS_TO_KO_COL]), float(pf[7][0])

        near_a, lin_a = val(10)
        near_b, lin_b = val(20)
        far_a, lin_c = val(300)
        far_b, lin_d = val(310)
        assert np.isclose(abs(lin_a - lin_b), abs(lin_c - lin_d), atol=1e-6), (
            "hp/HP_N should treat both 10-HP steps identically"
        )
        assert abs(near_a - near_b) > 50 * abs(far_a - far_b)
