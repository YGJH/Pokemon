"""``ptcg_rl.rating`` and ``ptcg_rl.arena`` — the cross-deck rating fit.

The fit is the part that can be wrong *silently*: a Bradley-Terry solve over an
unidentified design converges, prints two confident columns, and both are
whatever the optimiser's starting point put there.  So it gets the real tests —
matches are generated from known model strengths and deck effects and the fit
has to recover them inside its own confidence interval, and each guard is
validated by mutation: delete the bridge, break the anchor, disconnect the
graph, and confirm the fit refuses instead of returning plausible numbers.

``arena`` gets a fake-engine test for the one thing the module exists to fix:
each entry must carry its own decklist into both seats.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ptcg_rl.rating import (
    ANCHOR_RATING,
    SCALE,
    EntrySpec,
    Match,
    RatingError,
    check_bridge,
    expected_score,
    fit_bradley_terry,
    format_table,
)

# ── A synthetic world with known answers ────────────────────────────────────
#
# Five models over three decks.  ``bridge`` plays all three; every other model
# plays exactly one.  That is the repo's real shape in miniature: without
# ``bridge`` the S/D split has no data behind it at all.

TRUE_S = {
    "bridge": 1500.0,
    "spec_a": 1620.0,
    "spec_b": 1450.0,
    "spec_c": 1560.0,
    "weak": 1300.0,
}
TRUE_D = {"da": 60.0, "db": -90.0, "dc": 30.0}

ROSTER = [
    EntrySpec("bridge@da", "bridge", "da"),
    EntrySpec("bridge@db", "bridge", "db"),
    EntrySpec("bridge@dc", "bridge", "dc"),
    EntrySpec("spec_a", "spec_a", "da"),
    EntrySpec("spec_b", "spec_b", "db"),
    EntrySpec("spec_c", "spec_c", "dc"),
    EntrySpec("weak", "weak", "da"),
]


def _centred_truth() -> tuple[dict[str, float], dict[str, float]]:
    """``TRUE_S``/``TRUE_D`` moved into the fit's gauge.

    The likelihood only sees ``R``, so the truth has to be expressed under the
    same two constraints the fit imposes (``sum D = 0`` and the anchor at 1500)
    before the numbers are comparable at all.
    """
    shift = float(np.mean(list(TRUE_D.values())))
    d = {k: v - shift for k, v in TRUE_D.items()}
    s = {k: v + shift for k, v in TRUE_S.items()}
    anchor = next(e for e in ROSTER if e.name == ANCHOR_NAME)
    offset = ANCHOR_RATING - (s[anchor.model] + d[anchor.deck])
    return {k: v + offset for k, v in s.items()}, d


ANCHOR_NAME = "bridge@da"


def _true_rating(entry: EntrySpec) -> float:
    s, d = _centred_truth()
    return s[entry.model] + d[entry.deck]


def simulate(roster, n_games: int, seed: int = 0) -> list[Match]:
    """Round-robin sampled from the true Bradley-Terry probabilities."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(len(roster)):
        for j in range(i + 1, len(roster)):
            a, b = roster[i], roster[j]
            p = expected_score(_true_rating(a), _true_rating(b))
            wins_a = int(rng.binomial(n_games, p))
            out.append(Match(a=a.name, b=b.name,
                             wins_a=wins_a, wins_b=n_games - wins_a))
    return out


# ── Recovery ────────────────────────────────────────────────────────────────


def test_fit_recovers_known_ratings_within_ci():
    matches = simulate(ROSTER, n_games=4000, seed=7)
    assert matches, "simulate produced no matches — nothing was examined"

    table = fit_bradley_terry(matches, ROSTER, ANCHOR_NAME)
    assert table.converged

    checked = 0
    for entry in ROSTER:
        est = table.rating[entry.name]
        truth = _true_rating(entry)
        assert est.lo <= truth <= est.hi, (
            f"{entry.name}: true rating {truth:.0f} outside the fitted 95% CI "
            f"[{est.lo:.0f}, {est.hi:.0f}] (point {est.value:.0f})"
        )
        checked += 1
    assert checked == len(ROSTER), "not every entry was checked"


def test_fit_recovers_the_deck_effects_the_bridge_reveals():
    """The whole point: deck strength, separated from model strength."""
    matches = simulate(ROSTER, n_games=4000, seed=11)
    table = fit_bradley_terry(matches, ROSTER, ANCHOR_NAME)

    _s, true_d = _centred_truth()
    assert set(table.deck_effect) == set(true_d)
    for deck, truth in true_d.items():
        est = table.deck_effect[deck]
        assert est.lo <= truth <= est.hi, (
            f"deck {deck}: true effect {truth:+.0f} outside CI "
            f"[{est.lo:+.0f}, {est.hi:+.0f}]"
        )


def test_model_strengths_are_recovered_not_just_their_sums():
    matches = simulate(ROSTER, n_games=4000, seed=3)
    table = fit_bradley_terry(matches, ROSTER, ANCHOR_NAME)
    true_s, _d = _centred_truth()
    for model, truth in true_s.items():
        est = table.model_strength[model]
        assert est.lo <= truth <= est.hi, (
            f"model {model}: true strength {truth:.0f} outside CI "
            f"[{est.lo:.0f}, {est.hi:.0f}]"
        )


def test_anchor_sits_exactly_at_1500():
    matches = simulate(ROSTER, n_games=200, seed=1)
    table = fit_bradley_terry(matches, ROSTER, ANCHOR_NAME)
    assert table.rating[ANCHOR_NAME].value == pytest.approx(ANCHOR_RATING, abs=1e-6)
    assert table.rating[ANCHOR_NAME].stderr == pytest.approx(0.0, abs=1e-6), (
        "the anchor is a constraint, not an estimate — a non-zero standard "
        "error means the constraint is not being applied"
    )


def test_deck_effects_sum_to_zero():
    matches = simulate(ROSTER, n_games=200, seed=2)
    table = fit_bradley_terry(matches, ROSTER, ANCHOR_NAME)
    total = sum(e.value for e in table.deck_effect.values())
    assert total == pytest.approx(0.0, abs=1e-6)


def test_rating_reproduces_the_observed_win_rates():
    """A fit that recovers the parameters must also predict the data."""
    matches = simulate(ROSTER, n_games=4000, seed=5)
    table = fit_bradley_terry(matches, ROSTER, ANCHOR_NAME)
    checked = 0
    for m in matches:
        p_hat = expected_score(table.rating[m.a].value, table.rating[m.b].value)
        observed = m.score_a / m.n
        assert abs(p_hat - observed) < 0.05, (
            f"{m.a} vs {m.b}: fit predicts {p_hat:.3f}, observed {observed:.3f}"
        )
        checked += 1
    assert checked == len(matches) > 0


# ── Mutation: each guard must actually go red ───────────────────────────────


def test_deleting_the_bridge_raises_instead_of_returning_numbers():
    """The mutation the whole design exists to survive.

    With every model on exactly one deck, ``S_model`` and ``D_deck`` appear only
    as a sum.  The optimiser still converges; the two columns it prints are the
    starting point, not a measurement.
    """
    roster = [e for e in ROSTER if e.model != "bridge"]
    matches = simulate(roster, n_games=200, seed=4)
    assert matches, "nothing examined"

    with pytest.raises(RatingError, match="rank-deficient"):
        fit_bradley_terry(matches, roster, "spec_a")


def test_one_bridge_entry_is_not_enough_to_reach_a_third_deck():
    """A partial bridge leaves the decks it does not touch unidentified."""
    roster = [e for e in ROSTER if e.name != "bridge@dc"]
    with pytest.raises(RatingError, match="rank-deficient"):
        check_bridge(roster)


def test_full_bridge_passes_the_same_check():
    """The negative test above is only meaningful if the positive one holds."""
    check_bridge(ROSTER)  # must not raise


def test_disconnected_comparison_graph_raises():
    """Two islands that never played have no common scale."""
    matches = [
        m for m in simulate(ROSTER, n_games=100, seed=6)
        if not ({m.a, m.b} & {"weak"}) or {m.a, m.b} == {"weak", "spec_a"}
    ]
    # Isolate `weak` entirely.
    matches = [m for m in matches if "weak" not in (m.a, m.b)]
    with pytest.raises(RatingError, match="disconnected"):
        fit_bradley_terry(matches, ROSTER, ANCHOR_NAME)


def test_a_perfect_record_raises_rather_than_diverging():
    matches = simulate(ROSTER, n_games=100, seed=8)
    matches = [
        Match(a=m.a, b=m.b, wins_a=m.n if m.a == "spec_a" else 0,
              wins_b=0 if m.a == "spec_a" else m.n)
        if "spec_a" in (m.a, m.b) else m
        for m in matches
    ]
    with pytest.raises(RatingError, match="every game"):
        fit_bradley_terry(matches, ROSTER, ANCHOR_NAME)


def test_unknown_anchor_raises():
    matches = simulate(ROSTER, n_games=50, seed=9)
    with pytest.raises(RatingError, match="anchor"):
        fit_bradley_terry(matches, ROSTER, "no-such-entry")


def test_zero_games_raises_rather_than_fitting_nothing():
    empty = [Match(a=m.a, b=m.b, wins_a=0, wins_b=0)
             for m in simulate(ROSTER, n_games=10, seed=0)]
    with pytest.raises(RatingError, match="no games"):
        fit_bradley_terry(empty, ROSTER, ANCHOR_NAME)


def test_duplicate_entry_names_raise():
    roster = list(ROSTER) + [EntrySpec("spec_a", "spec_a", "da")]
    matches = simulate(ROSTER, n_games=50, seed=0)
    with pytest.raises(RatingError, match="duplicate"):
        fit_bradley_terry(matches, roster, ANCHOR_NAME)


def test_draws_count_as_half_a_win_each():
    m = Match(a="x", b="y", wins_a=3, wins_b=1, draws=2)
    assert m.n == 6
    assert m.score_a == 4.0


def test_scale_is_the_elo_scale():
    """400 points is 10:1 odds; anything else silently rescales every number."""
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)
    assert SCALE == pytest.approx(math.log(10) / 400)


def test_format_table_names_every_entry():
    matches = simulate(ROSTER, n_games=100, seed=12)
    text = format_table(fit_bradley_terry(matches, ROSTER, ANCHOR_NAME))
    for entry in ROSTER:
        assert entry.name in text


# ── arena: the deck must follow the model ───────────────────────────────────


class _FakeEnv:
    """Records the decks it was constructed with, then ends every game at once.

    Deliberately a real (if tiny) state machine rather than a mock: the property
    under test is about the *sequence* — which seat's observations reach which
    actor, and which decklist each seat was dealt.
    """

    created: list[dict] = []

    def __init__(self, deck_self, deck_opp, n_envs, our_player, seed):
        self.deck_self = list(deck_self)
        self.deck_opp = list(deck_opp)
        self.our_player = our_player
        self.n_envs = n_envs
        type(self).created.append(
            {"seat0": self.deck_self, "seat1": self.deck_opp,
             "our_player": our_player}
        )
        self._served = 0
        self._finished = 0
        self.n_games = 4

    def poll(self):
        if self._served >= self.n_games:
            return []
        self._served += 1
        obs = {"current": {"yourIndex": self._served % 2}, "select": None}
        return [{"battle_idx": 0, "obs_json": json.dumps(obs), "sbi": "",
                 "select_player": self._served % 2, "opp_id": -1}]

    def reply(self, picks_list):
        return 0

    def drain(self):
        if self._finished >= self._served:
            return []
        out = []
        while self._finished < self._served:
            out.append(_FakeTraj(reward=1.0))
            self._finished += 1
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTraj:
    def __init__(self, reward):
        self.reward = reward
        self.battle_idx = 0
        self.opp_id = -1
        self.error = None


def _seat_recording_actor(log: list, tag: str):
    def actor(requests):
        for r in requests:
            log.append((tag, r["obs"]["current"]["yourIndex"]))
        return [{"picks": [0]} for _ in requests]
    return actor


@pytest.fixture(autouse=True)
def _reset_fake_env():
    _FakeEnv.created = []
    yield
    _FakeEnv.created = []


def test_each_entry_carries_its_own_deck_into_both_seats():
    """The bug this module exists to fix.

    ``elo_calibrate`` hands both halves the same ``(fixed_deck, opp_decks[0])``
    pair regardless of who is playing, so a specialist spends half its games
    piloting somebody else's archetype.  Here the seat swap must move the decks
    with the models.
    """
    from ptcg_rl.arena import ArenaEntry, play_pair

    a = ArenaEntry("a", "a", "da", tuple([11] * 60), path=None)  # type: ignore[arg-type]
    b = ArenaEntry("b", "b", "db", tuple([22] * 60), path=None)  # type: ignore[arg-type]
    noop = lambda reqs: [{"picks": [0]} for _ in reqs]  # noqa: E731

    play_pair(a, b, noop, noop, games=8, n_envs=2, env_factory=_FakeEnv)

    assert len(_FakeEnv.created) == 2, "expected exactly two halves"
    first, second = _FakeEnv.created
    assert first["seat0"] == [11] * 60 and first["seat1"] == [22] * 60, (
        "first half did not deal each entry its own deck"
    )
    assert second["seat0"] == [22] * 60 and second["seat1"] == [11] * 60, (
        "the seat swap moved the models but not the decks — that is a deck "
        "swap wearing a seat swap's clothes, and it is the exact bug in "
        "elo_calibrate._eval_with_seats"
    )
    assert all(c["our_player"] == 0 for c in _FakeEnv.created), (
        "our_player must stay 0: the engine deals deck_self to seat 0 no "
        "matter what, and only the reward frame follows our_player"
    )


def test_wins_are_attributed_to_the_entry_not_to_the_seat():
    """Seat 0 wins every game in the fake env, so a correct tally is a split."""
    from ptcg_rl.arena import ArenaEntry, play_pair

    a = ArenaEntry("a", "a", "da", tuple([1] * 60), path=None)  # type: ignore[arg-type]
    b = ArenaEntry("b", "b", "db", tuple([2] * 60), path=None)  # type: ignore[arg-type]
    noop = lambda reqs: [{"picks": [0]} for _ in reqs]  # noqa: E731

    m = play_pair(a, b, noop, noop, games=8, n_envs=2, env_factory=_FakeEnv)
    assert m.wins_a == m.wins_b > 0, (
        f"seat-0-always-wins gave {m.wins_a}-{m.wins_b}; wins are being "
        "credited to the seat rather than to the entry sitting in it"
    )
    assert m.n == m.wins_a + m.wins_b


def test_observations_are_routed_by_select_player():
    """The featurizer is egocentric — a misrouted seat plays the wrong side and
    nothing about it is visible downstream."""
    from ptcg_rl.arena import play_half

    log: list = []
    w0, w1, draws = play_half(
        _seat_recording_actor(log, "seat0"),
        _seat_recording_actor(log, "seat1"),
        [1] * 60, [2] * 60, n_games=4, n_envs=2, env_factory=_FakeEnv,
    )
    assert log, "no decisions were routed — nothing was examined"
    for tag, your_index in log:
        assert tag == f"seat{your_index}", (
            f"an observation with yourIndex={your_index} was handed to {tag}"
        )
    assert w0 + w1 + draws == 4


class _BurstEnv(_FakeEnv):
    """Finishes games three at a time, so the target is always overshot."""

    def drain(self):
        if self._finished >= self._served:
            return []
        out = []
        for _ in range(3):
            out.append(_FakeTraj(reward=1.0))
            self._finished += 1
        return out


def test_the_tally_stops_at_the_requested_game_count():
    """Battles already in flight when the target is reached must not be counted.

    The two halves are the only thing cancelling first-player advantage; letting
    one of them absorb more overspill than the other feeds that advantage
    straight into the result.
    """
    from ptcg_rl.arena import play_half

    w0, w1, draws = play_half(
        lambda reqs: [{"picks": [0]} for _ in reqs],
        lambda reqs: [{"picks": [0]} for _ in reqs],
        [1] * 60, [2] * 60, n_games=2, n_envs=4, env_factory=_BurstEnv,
    )
    assert w0 + w1 + draws == 2, (
        f"asked for 2 games, tallied {w0 + w1 + draws} — in-flight battles are "
        "being counted past the target"
    )


def test_a_short_actor_reply_raises_instead_of_desynchronising():
    from ptcg_rl.arena import play_half

    with pytest.raises(RuntimeError, match="replies"):
        play_half(
            lambda reqs: [],          # seat 0 answers nothing
            lambda reqs: [{"picks": [0]} for _ in reqs],
            [1] * 60, [2] * 60, n_games=4, n_envs=2, env_factory=_FakeEnv,
        )


# ── arena: roster construction ──────────────────────────────────────────────


def _candidate(name, deck_key, archetype, deck_card=1, vsha="v", asha="a"):
    from pathlib import Path

    from ptcg_rl.arena import Candidate

    return Candidate(
        name=name, path=Path(f"/tmp/{name}.pt"), deck=tuple([deck_card] * 60),
        deck_key=deck_key, archetype_self=archetype,
        vocab_sha1=vsha, archetypes_sha1=asha, file_sha1=name,
    )


def test_roster_enters_the_bridge_once_per_deck():
    from ptcg_rl.arena import build_roster

    cands = [
        _candidate("generalist/best", "FIXED", None, 0),
        _candidate("a1/best", "a1", 1, 1),
        _candidate("a17/best", "a17", 17, 17),
    ]
    roster = build_roster(cands, bridge="generalist/best")
    names = [e.name for e in roster]
    assert "generalist/best@a1" in names
    assert "generalist/best@a17" in names
    assert "generalist/best@FIXED" not in names, (
        "the bridge already plays its own deck — a second entry on it is the "
        "same model against itself"
    )
    # And the bridge entries carry the deck they are named for.
    bridge_a1 = next(e for e in roster if e.name == "generalist/best@a1")
    assert bridge_a1.deck == tuple([1] * 60)
    assert bridge_a1.path == cands[0].path, "bridge entries must share weights"
    check_bridge([e.spec() for e in roster])  # identifiable


def test_a_specialist_cannot_be_the_bridge():
    from ptcg_rl.arena import build_roster

    cands = [
        _candidate("generalist/best", "FIXED", None, 0),
        _candidate("a1/best", "a1", 1, 1),
    ]
    with pytest.raises(ValueError, match="specialist"):
        build_roster(cands, bridge="a1/best")


def test_mixed_artifact_generations_are_refused():
    from ptcg_rl.arena import check_artifacts

    cands = [
        _candidate("generalist/best", "FIXED", None, 0, vsha="v1"),
        _candidate("a1/best", "a1", 1, 1, vsha="v2"),
    ]
    with pytest.raises(ValueError, match="artifact generation"):
        check_artifacts(cands)


def test_matching_artifact_generations_pass():
    from ptcg_rl.arena import check_artifacts

    cands = [_candidate("generalist/best", "FIXED", None, 0),
             _candidate("a1/best", "a1", 1, 1)]
    assert check_artifacts(cands) == ("v", "a")


@pytest.mark.parametrize("filename,expected", [
    ("ckpt-best.pt", "best"),
    ("ckpt-last.pt", "last"),
    ("ckpt-step-0002000.pt", "step-0002000"),
    ("ckpt-mcts-champion-001600.pt", "champ-001600"),
    ("ckpt-mcts-last.pt", "mcts-last"),
])
def test_checkpoint_labels(filename, expected):
    from pathlib import Path

    from ptcg_rl.arena import ckpt_label

    assert ckpt_label(Path(filename)) == expected


@pytest.mark.parametrize("dirname,expected", [
    ("checkpoints", "default"),
    ("checkpoints_generalist", "generalist"),
    ("checkpoints_a1", "a1"),
    ("checkpoints_a16_mcts", "a16_mcts"),
])
def test_directory_labels(dirname, expected):
    from pathlib import Path

    from ptcg_rl.arena import dir_label

    assert dir_label(Path(dirname)) == expected


def test_round_trip_through_the_saved_document():
    """The result matrix is written so the fit can be re-run without games."""
    from ptcg_rl.arena import fit_from_document

    matches = simulate(ROSTER, n_games=500, seed=13)
    table = fit_bradley_terry(matches, ROSTER, ANCHOR_NAME)
    doc = {
        "anchor": ANCHOR_NAME,
        "entries": [{"name": e.name, "model": e.model, "deck": e.deck}
                    for e in ROSTER],
        "matches": [{"a": m.a, "b": m.b, "wins_a": m.wins_a,
                     "wins_b": m.wins_b, "draws": m.draws} for m in matches],
    }
    again = fit_from_document(doc)
    for name, est in table.rating.items():
        assert again.rating[name].value == pytest.approx(est.value, abs=1e-6)
