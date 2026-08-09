"""Unit tests for live_eval OOV tracking and game-end detection.

Tests use real-shaped observation dicts (Pokemon/Card dicts with ``id`` keys,
not ``cardId`` — matching the live engine's cg.api dataclass schemas).
"""

import sys
from pathlib import Path

import pytest

# Ensure cg is importable (for enum references if needed)
_ENGINE_DIR = (
    Path(__file__).resolve().parent.parent
    / "pokemon-tcg-ai-battle"
    / "sample_submission"
    / "sample_submission"
)
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

import ctypes
import inspect
import pickle
from concurrent.futures import ProcessPoolExecutor

import torch

from ptcg_il import live_eval
from ptcg_il.live_eval import (
    PolicyAgent,
    _count_oov_opponent_cards,
    make_agent_from_policy,
    random_agent,
    search_planner_agent,
    wilson_interval,
)
from tests.test_model_policy import make_policy


def _call_deck_step(agent):
    """Invoke an agent in a worker process (must be module-level to be picklable)."""
    return agent({"select": None})


class _TinyPolicy(torch.nn.Module):
    """Stand-in for Policy. Module-level for the same reason PolicyAgent is a class:
    a class defined inside a function is a local object and will not pickle."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


# ---------------------------------------------------------------------------
# Helpers: build real-shaped observation dicts
# ---------------------------------------------------------------------------

def _make_opp_active(*card_ids: int | None) -> list:
    """Build an active list: ``[{"id": cid, ...}]`` or ``[None]``."""
    if not card_ids or card_ids[0] is None:
        return [None]
    return [{"id": cid, "serial": 100 + i, "playerIndex": 1} for i, cid in enumerate(card_ids)]


def _make_pokemon(*card_ids: int) -> list[dict]:
    """Build a bench list of Pokemon dicts."""
    return [
        {"id": cid, "serial": 200 + i, "hp": 100, "maxHp": 150,
         "appearThisTurn": False, "energies": [], "energyCards": [],
         "tools": [], "preEvolution": []}
        for i, cid in enumerate(card_ids)
    ]


def _make_discard(*card_ids: int) -> list[dict]:
    """Build a discard list of Card dicts (cg.api.Card: id, serial, playerIndex)."""
    return [
        {"id": cid, "serial": 300 + i, "playerIndex": 1}
        for i, cid in enumerate(card_ids)
    ]


def _make_obs(your_index: int, opp_active, opp_bench, opp_discard) -> dict:
    """Build a minimal observation dict with opponent card data."""
    my_player = {
        "active": _make_opp_active(None),  # irrelevant for OOV counting
        "bench": [],
        "discard": [],
        "deckCount": 40,
        "handCount": 5,
        "prize": [None] * 6,
        "hand": None,
        "benchMax": 5,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }
    opp_player = {
        "active": opp_active,
        "bench": opp_bench,
        "discard": opp_discard,
        "deckCount": 40,
        "handCount": 5,
        "prize": [None] * 6,
        "hand": None,
        "benchMax": 5,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }
    players = [my_player, opp_player] if your_index == 0 else [opp_player, my_player]
    return {"players": players}


# ---------------------------------------------------------------------------
# _count_oov_opponent_cards tests
# ---------------------------------------------------------------------------

class TestOovTracking:
    """Verify _count_oov_opponent_cards reads ``id`` (not ``cardId``) from
    real-shaped engine observation dicts."""

    def test_all_in_vocab(self):
        """Active (1), bench (2), discard (3 cards): all in vocab → oov=0, total=6."""
        id_to_index = {10: 2, 20: 3, 30: 4, 40: 5, 50: 6, 60: 7}
        obs = _make_obs(
            your_index=0,
            opp_active=_make_opp_active(10),
            opp_bench=_make_pokemon(20, 30),
            opp_discard=_make_discard(40, 50, 60),
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=0, id_to_index=id_to_index)
        assert oov == 0
        assert total == 6  # 1 active + 2 bench + 3 discard

    def test_some_oov(self):
        """Cards 99 and 88 not in vocab → oov=2, total=4."""
        id_to_index = {10: 2, 20: 3}
        obs = _make_obs(
            your_index=0,
            opp_active=_make_opp_active(10),
            opp_bench=_make_pokemon(99),
            opp_discard=_make_discard(20, 88),
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=0, id_to_index=id_to_index)
        assert oov == 2  # 99 and 88
        assert total == 4

    def test_face_down_active_skipped(self):
        """Face-down active (None) adds nothing to counts."""
        id_to_index = {10: 2, 20: 3}
        obs = _make_obs(
            your_index=0,
            opp_active=_make_opp_active(None),  # face-down
            opp_bench=_make_pokemon(10, 20),
            opp_discard=[],
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=0, id_to_index=id_to_index)
        assert oov == 0
        assert total == 2  # bench only

    def test_empty_opponent(self):
        """Opponent has no visible cards → oov=0, total=0."""
        id_to_index = {10: 2}
        obs = _make_obs(
            your_index=0,
            opp_active=[],
            opp_bench=[],
            opp_discard=[],
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=0, id_to_index=id_to_index)
        assert oov == 0
        assert total == 0

    def test_none_id_to_index_skips(self):
        """id_to_index=None → return (0,0) without iterating (fast path)."""
        obs = _make_obs(
            your_index=0,
            opp_active=_make_opp_active(10),
            opp_bench=_make_pokemon(20),
            opp_discard=_make_discard(30),
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=0, id_to_index=None)
        assert oov == 0
        assert total == 0

    def test_your_index_1(self):
        """Works when the opponent is player 0 (yourIndex=1)."""
        id_to_index = {10: 2}
        obs = _make_obs(
            your_index=1,  # we are player 1, opponent is player 0
            opp_active=_make_opp_active(10),
            opp_bench=[],
            opp_discard=[],
        )
        oov, total = _count_oov_opponent_cards(obs, your_index=1, id_to_index=id_to_index)
        assert oov == 0
        assert total == 1


# ---------------------------------------------------------------------------
# Wilson interval tests (existing behavior, kept for coverage)
# ---------------------------------------------------------------------------

class TestWilsonInterval:
    def test_perfect_winrate(self):
        center, lo, hi = wilson_interval(500, 500)
        assert 0.99 < center < 1.0
        assert lo > 0.99

    def test_zero_games(self):
        center, lo, hi = wilson_interval(0, 0)
        assert center == 0.0 and lo == 0.0 and hi == 0.0

    def test_even_split(self):
        center, lo, hi = wilson_interval(250, 500)
        assert 0.45 < center < 0.55
        assert lo < center < hi


class TestYourIndexResolution:
    """``yourIndex`` lives on ``obs["current"]``, never at the top level.

    Reading it from the top level yielded 0 for every decision, so ``_run_one_game``
    gave every turn to ``agent0`` and the opponent never played — measured over a
    real game: 0/103 decisions had a top-level ``yourIndex`` while
    ``current.yourIndex`` alternated 55/48.  A win rate measured that way is one
    agent playing itself, and nothing about it looks wrong from the outside.
    """

    def test_reads_from_current(self):
        assert live_eval._your_index({"current": {"yourIndex": 1}}) == 1
        assert live_eval._your_index({"current": {"yourIndex": 0}}) == 0

    def test_current_wins_over_absent_top_level(self):
        """The real engine shape: current.yourIndex set, no top-level key."""
        obs = {"current": {"yourIndex": 1, "result": -1}, "select": {"option": []}}
        assert live_eval._your_index(obs) == 1, (
            "returned 0 for a player-1 turn; agent0 would play both sides")

    def test_top_level_fallback(self):
        assert live_eval._your_index({"yourIndex": 1}) == 1

    def test_defaults_to_zero_when_absent(self):
        assert live_eval._your_index({}) == 0
        assert live_eval._your_index({"current": {}}) == 0

    def test_dispatch_alternates_between_agents(self):
        """Both seats must get turns when current.yourIndex alternates."""
        seen = []
        for yi in (0, 1, 0, 1, 1):
            seen.append(live_eval._your_index({"current": {"yourIndex": yi}}))
        assert set(seen) == {0, 1}, (
            "dispatch never selects player 1; the opponent agent would never act")


class TestRustSearchPlannerFfiContract:
    """The Rust ``search_plan`` FFI contract has two process-fatal footguns.

    Neither one raises a catchable Python exception — both kill the process — so
    they cannot be covered by a "does it work" test.  These tests pin the two
    declarations that keep them from coming back.
    """

    @staticmethod
    def _load_or_skip():
        lib = live_eval._load_rust_search_lib()
        if lib is None:
            pytest.skip("libptcg_search.so not built")
        return lib

    def test_host_initialized_arg_is_declared(self):
        """search_plan must take the trailing host_initialized flag.

        libcg.so registers into a fixed-capacity global table; a second
        ``GameInitialize()`` throws C++ ``std::runtime_error("buffer full.
        capacity:7")``.  Rust cannot catch a foreign exception, so the worker dies
        with SIGABRT and every game in it is lost.  Python callers have already
        initialized the engine via ``cg.sim``'s import and must pass 1.
        """
        lib = self._load_or_skip()
        assert len(lib.search_plan.argtypes) == 7, (
            "search_plan lost its host_initialized argument; Rust will call "
            "GameInitialize() a second time and abort the process")
        assert lib.search_plan.argtypes[-1] is ctypes.c_int

    def test_return_pointer_is_not_c_char_p(self):
        """restype must keep the raw address so the right pointer can be freed.

        With ``restype=c_char_p`` ctypes copies the string into Python bytes and
        discards the pointer; passing that back to ``search_plan_free`` hands
        ``CString::from_raw`` a pointer into Python's heap and glibc aborts with
        "munmap_chunk(): invalid pointer".
        """
        lib = self._load_or_skip()
        assert lib.search_plan.restype is not ctypes.c_char_p, (
            "restype=c_char_p loses the Rust pointer; freeing it corrupts the heap")
        assert lib.search_plan.restype is ctypes.c_void_p
        assert lib.search_plan_free.argtypes == [ctypes.c_void_p]


class TestFixedDeckLookupIsCwdIndependent:
    """The search planner's deck must resolve without depending on cwd.

    The old lookup used a bare ``Path("data")``, so running the CLI from anywhere
    but python/ silently substituted an illegal ``range(1, 61)`` deck and every
    search_planner win rate became meaningless without any error.
    """

    def test_resolves_from_an_unrelated_cwd(self, tmp_path, monkeypatch):
        data_dir = (Path(live_eval.__file__).resolve().parent.parent / "data")
        if not (data_dir / "archetypes.json").exists():
            pytest.skip("data/archetypes.json not present")

        monkeypatch.delenv("PTCG_DATA_DIR", raising=False)
        monkeypatch.chdir(tmp_path)  # nothing named data/ here
        deck = live_eval._get_fixed_deck_for_search()

        assert deck != list(range(1, 61)), (
            "fell back to the illegal dummy deck; the lookup is still cwd-relative")
        assert len(deck) == 60


class TestAgentsArePicklable:
    """Every agent must survive pickling.

    `eval_vs_opponent` dispatches each game to a ProcessPoolExecutor, so agents
    travel to the worker as pickled job arguments.  When `make_agent_from_policy`
    returned a closure, *every* game died with

        Can't pickle local object 'make_agent_from_policy.<locals>.agent'

    and — because the failure was caught and logged per game rather than raised —
    live eval reported a completed run with zero games played.  These tests exist
    so that regression cannot be silent again.
    """

    @staticmethod
    def _tiny_policy():
        return _TinyPolicy()

    @staticmethod
    def _vocab():
        return {"id_to_index": {"7": 2, "1152": 3}, "attack_id_to_index": {"1": 2}}

    def test_policy_agent_pickles(self):
        agent = make_agent_from_policy(
            self._tiny_policy(), self._vocab(), list(range(60)))
        restored = pickle.loads(pickle.dumps(agent))
        assert isinstance(restored, PolicyAgent)

    def test_factory_returns_picklable_not_closure(self):
        """The factory must not hand back a local function."""
        agent = make_agent_from_policy(
            self._tiny_policy(), self._vocab(), list(range(60)))
        assert not inspect.isfunction(agent), (
            "make_agent_from_policy returned a plain function; if it is a closure "
            "every live-eval game will fail to pickle")
        assert callable(agent)

    def test_baseline_agents_pickle(self):
        """The top-level baseline agents must stay top-level."""
        for fn in (random_agent, search_planner_agent):
            assert pickle.loads(pickle.dumps(fn)) is fn

    def test_unpickled_agent_is_eval_mode_and_usable(self):
        """A worker-side agent must be in eval mode and answer the deck step."""
        deck = list(range(60))
        agent = make_agent_from_policy(self._tiny_policy(), self._vocab(), deck)
        restored = pickle.loads(pickle.dumps(agent))

        assert restored.policy.training is False
        # select is None -> the deck-submission step, which needs no forward pass.
        assert restored({"select": None}) == deck

    def test_pickles_through_a_real_process_pool(self):
        """End-to-end: the agent survives a genuine ProcessPoolExecutor dispatch.

        A direct `pickle.dumps` round trip can pass while spawn-based dispatch
        fails, so exercise the mechanism live eval actually uses.
        """
        agent = make_agent_from_policy(
            self._tiny_policy(), self._vocab(), list(range(60)))
        with ProcessPoolExecutor(max_workers=1) as ex:
            got = ex.submit(_call_deck_step, agent).result()
        assert got == list(range(60))


class TestPolicyAgentLoadsEngineFeatures:
    """A1 — ``PolicyAgent`` used to featurize with no engine tables at all.

    ``featurize`` treats a missing table as "no information", so every
    ``*_card_feat`` and ``opt_attack_feat`` tensor came out zeros: the agent
    played with no card identity, no attack identity, no KO-pressure block and
    no hand legality flags, against a model trained with all of them.  Nothing
    raised and nothing looked wrong from outside.
    """

    @staticmethod
    def _tiny_policy():
        from ptcg_il.model.policy import Policy
        return make_policy(D=32, heads=2, layers=1, ff=64).eval()

    @staticmethod
    def _vocab():
        return {"id_to_index": {"7": 2}, "attack_id_to_index": {"1": 2}}

    def _agent(self, data_dir):
        return make_agent_from_policy(
            self._tiny_policy(), self._vocab(), list(range(60)), data_dir=data_dir)

    def test_tables_are_loaded_from_data_dir(self, tmp_path):
        import numpy as np

        from ptcg_il.featurizer import F_ATK, F_CARD

        np.save(tmp_path / "engine_card_features.npy",
                {7: np.ones(F_CARD, dtype=np.float32)})
        np.save(tmp_path / "engine_attack_features.npy",
                {1: np.ones(F_ATK, dtype=np.float32)})
        np.save(tmp_path / "evolution_map.npy", {7: [3]})

        tables = self._agent(tmp_path)._tables()
        assert tables["engine_card_features"] is not None
        assert tables["engine_attack_features"] is not None
        assert tables["evolution_map"] == {7: [3]}

    def test_missing_data_dir_still_runs_but_yields_nothing(self):
        """The old behaviour stays reachable — it just no longer happens silently."""
        tables = self._agent(None)._tables()
        assert set(tables) == {
            "engine_card_features", "engine_attack_features", "evolution_map"}
        assert all(v is None for v in tables.values())

    def test_absent_files_do_not_raise(self, tmp_path):
        tables = self._agent(tmp_path)._tables()
        assert all(v is None for v in tables.values())

    def test_tables_are_not_pickled_to_workers(self, tmp_path):
        """Only the path travels; 1.1 MB of card table per job would not."""
        import numpy as np

        from ptcg_il.featurizer import F_CARD

        np.save(tmp_path / "engine_card_features.npy",
                {7: np.ones(F_CARD, dtype=np.float32)})
        agent = self._agent(tmp_path)
        agent._tables()                       # force the load
        assert agent._engine_tables is not None

        restored = pickle.loads(pickle.dumps(agent))
        assert restored._engine_tables is None, "loaded tables were pickled"
        assert restored.data_dir == str(tmp_path)
        # ...and the worker rebuilds them on demand.
        assert restored._tables()["engine_card_features"] is not None
