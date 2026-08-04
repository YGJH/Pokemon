"""Regression tests for the two silent breaks that made MCTS distillation
a no-op: decisions carried no ``obs_json``, and the CLI's MCTS flags never
reached the ``config.mcts_*`` names the search paths read.

Both failed by *omission* — a ``dict.get`` and a ``getattr`` default — so
neither raised and the only visible symptom was ``0 MCTS, 0%``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ptcg_rl.mcts_train import _basic_pokemon_ids, parse_args
from ptcg_rl.rust_vec_env import RustVecEnv


# ── obs_json must survive poll() → Decision ──────────────────────────────

class _FakePolledEnv(RustVecEnv):
    """RustVecEnv with the FFI replaced — exercises collect_batch's
    bookkeeping without loading libcg or libptcg_search."""

    def __init__(self, polls):
        self._polls = list(polls)
        self._pending = {}
        self.n_envs = 1
        self.our_player = 0
        self.total_decisions = 0
        self.replies = []
        self._drained = False

    def poll(self):
        return self._polls.pop(0) if self._polls else []

    def drain(self):
        """Finish the battle only once every queued poll is consumed."""
        from ptcg_rl.vec_env import Trajectory
        if self._polls or self._drained:
            return []
        self._drained = True
        t = Trajectory()
        t.battle_idx = 0
        t.reward = 1.0
        return [t]

    def reply(self, picks_list):
        self.replies.append(picks_list)
        return 0

    def close(self):  # no handle to free
        pass


def _obs(turn: int) -> str:
    return json.dumps({"turn": turn, "current": {"yourIndex": 0}})


def _act_fn(requests):
    return [
        {"picks": [0], "features": {"dummy": 1}, "action_idx": [0],
         "action_len": 1, "logp": -0.5, "value": 0.1}
        for _ in requests
    ]


def test_collect_batch_records_obs_json():
    """Decision.obs_json must carry the raw observation from poll().

    MCTS distillation skips any decision whose obs_json is falsy
    (``mcts_train._run_mcts_on_decisions_pooled``), so a None here silently
    zeroes the MCTS target count.  Mutation check: reading a key that poll()
    does not emit — as ``pending[i].get("obs_json_raw")`` did — turns this
    red.
    """
    polls = [
        [{"battle_idx": 0, "obs_json": _obs(1), "sbi": "", "select_player": 0}],
        [{"battle_idx": 0, "obs_json": _obs(2), "sbi": "", "select_player": 0}],
        [],  # engine has nothing pending → battle drains
    ]
    env = _FakePolledEnv(polls)
    done = env.collect_batch(_act_fn, n_decisions=1)

    assert len(done) == 1
    decisions = done[0].decisions
    assert len(decisions) == 2, "fixture must produce decisions to examine"
    for dec, expected_turn in zip(decisions, (1, 2)):
        assert dec.obs_json, "Decision.obs_json is empty — MCTS would skip it"
        assert json.loads(dec.obs_json)["turn"] == expected_turn


# ── CLI flags must land on the mcts_* attribute names ────────────────────

_BASE_ARGV = ["--data-dir", "d", "--il-ckpt", "c"]

# (CLI flag, value, the attribute the MCTS code paths read)
_FLAG_TO_ATTR = [
    ("--rho", "1.0", "mcts_rho", 1.0),
    ("--iterations", "128", "mcts_iterations", 128),
    ("--c-puct", "3.5", "mcts_c_puct", 3.5),
    ("--k-determinizations", "6", "mcts_k_determinizations", 6),
    ("--leaf-batch", "4096", "mcts_leaf_batch", 4096),
    ("--n-engines", "20", "mcts_n_engines", 20),
]


@pytest.mark.parametrize("flag,value,attr,expected", _FLAG_TO_ATTR)
def test_mcts_flag_reaches_config_attribute(flag, value, attr, expected):
    """Every MCTS knob is read as ``getattr(config, "mcts_*", <default>)``,
    so an unmapped flag is not an error — it is silently the default."""
    args = parse_args(_BASE_ARGV + [flag, value])
    assert getattr(args, attr) == expected


def test_mcts_distill_and_enabled_are_unified():
    """mcts_train gates on ``mcts_distill``; ptcg_rl.search gates on
    ``mcts_enabled``.  Either flag must switch on both paths."""
    for flag in ("--mcts-distill", "--mcts-enabled"):
        args = parse_args(_BASE_ARGV + [flag])
        assert args.mcts_distill is True, flag
        assert args.mcts_enabled is True, flag

    off = parse_args(_BASE_ARGV)
    assert off.mcts_distill is False
    assert off.mcts_enabled is False


def test_pipeline_flag_set_is_fully_mapped():
    """The exact flag set scripts/run_pipeline.sh passes must survive
    normalization — no knob may fall back to a hardcoded default."""
    argv = _BASE_ARGV + [
        "--iterations", "64", "--c-puct", "2.0", "--k-determinizations", "4",
        "--leaf-batch", "4096", "--rho", "1.0", "--mcts-distill",
        "--n-engines", "20",
    ]
    args = parse_args(argv)
    assert (args.mcts_iterations, args.mcts_c_puct) == (64, 2.0)
    assert (args.mcts_k_determinizations, args.mcts_leaf_batch) == (4, 4096)
    assert (args.mcts_rho, args.mcts_n_engines) == (1.0, 20)
    assert args.mcts_distill is True


# ── A champion is labelled with the deck it plays ────────────────────────
#
# The label used to be copied off --il-ckpt.  That is the same file only
# while θ_init is a specialist on this very deck.  Under pre-train +
# fine-tune it is a generalist, whose deck record carries `archetypes.json`'s
# `fixed_deck` — archetype 21's list in this corpus, the *smallest* deck —
# while `_deck_for(--deck-archetype)` hands the run archetype 1's.  Every
# champion then shipped a deck it never played, into a Kaggle submission,
# with no error anywhere: `_check_ckpt_deck` reads a generalist's
# `archetype_self=None` as "nothing to verify".


def test_deck_label_follows_deck_archetype_not_the_il_checkpoint(monkeypatch):
    """The stamped record must come from --deck-archetype."""
    from ptcg_rl import mcts_train

    seen = {}

    def fake_build(data_dir, archetype_self=None):
        seen["archetype_self"] = archetype_self
        return {"archetype_self": archetype_self, "deck": [7] * 60,
                "deck_size": 60}

    import ptcg_il.deck as deck_mod
    monkeypatch.setattr(deck_mod, "build_deck_metadata", fake_build)

    record = mcts_train._deck_record_for(Path("data"), 1)

    assert seen["archetype_self"] == 1, (
        "the label was built for a different archetype than the one played"
    )
    assert record["archetype_self"] == 1


def test_generalist_ckpt_is_allowed_but_a_wrong_specialist_is_not(tmp_path):
    """A generalist warm-start is legitimate; a foreign specialist is not."""
    import torch

    from ptcg_rl.mcts_train import _check_ckpt_deck

    generalist = tmp_path / "generalist.pt"
    torch.save({"deck": {"archetype_self": None, "archetypes_sha1": "a" * 40}},
               generalist)
    # Must not raise: pre-training across every archetype is the point.
    _check_ckpt_deck(str(generalist), 1)

    foreign = tmp_path / "a17.pt"
    torch.save({"deck": {"archetype_self": 17, "archetypes_sha1": "a" * 40}},
               foreign)
    with pytest.raises(SystemExit, match="archetype 17"):
        _check_ckpt_deck(str(foreign), 1)


def test_init_ckpt_separates_theta_init_from_the_anchor():
    """--init-ckpt must not disturb the anchor/baseline role of --il-ckpt.

    Pointing both at the generalist would make the league gate ask whether
    the fine-tuned model beats its own starting point, which it cannot
    fail informatively.
    """
    args = parse_args(["--data-dir", "d", "--il-ckpt", "a1.pt",
                       "--init-ckpt", "gen.pt", "--deck-archetype", "1"])
    assert args.init_ckpt == "gen.pt"
    assert args.il_ckpt == "a1.pt", "the anchor must stay the specialist"

    # Omitted, θ_init falls back to the anchor — the historical behaviour.
    assert parse_args(["--data-dir", "d", "--il-ckpt", "a1.pt"]).init_ckpt is None


# ── The determinizer's Basic-Pokémon id set ──────────────────────────────

def _feat_row(basic: float, stage1: float = 0.0, stage2: float = 0.0):
    """A 94-dim engine_card_features row; cols 9:12 are (basic, stage1, stage2)."""
    row = np.zeros(94, dtype=np.float32)
    row[9:12] = (basic, stage1, stage2)
    return row


def test_basic_pokemon_ids_selects_only_basics():
    """Only a Basic can sit face-down as the opponent's active.  Feeding the
    engine anything else gets the root refused with SearchBegin error 2, so
    this set is what keeps MCTS from silently losing determinizations."""
    feats = {
        11: _feat_row(1.0),                    # Basic
        12: _feat_row(0.0, stage1=1.0),        # Stage 1 — accepted by the
                                               # engine but never a legal
                                               # face-down active
        13: _feat_row(0.0, stage2=1.0),        # Stage 2
        14: _feat_row(0.0),                    # Trainer/Energy
        15: _feat_row(1.0),                    # Basic
    }
    assert _basic_pokemon_ids(feats) == [11, 15]


def test_basic_pokemon_ids_reads_the_basic_column_not_a_neighbour():
    """Cols 10 and 11 are stage1/stage2.  Reading either instead of 9 yields a
    plausible non-empty set that the engine rejects one root at a time."""
    feats = {21: _feat_row(0.0, stage1=1.0, stage2=1.0)}
    assert _basic_pokemon_ids(feats) == []


def test_basic_pokemon_ids_without_features_is_empty_not_wrong():
    """No feature table → no claim.  An empty list leaves Rust on its old
    whole-template guess rather than asserting a set we cannot justify."""
    assert _basic_pokemon_ids(None) == []
    assert _basic_pokemon_ids({}) == []


# ── priors must be one per tree child ────────────────────────────────────

def _leaf_obs(n_options: int, max_count: int) -> str:
    """A minimal observation with `n_options` options at one decision point."""
    return json.dumps({
        "search_begin_input": "sbi",
        "select": {
            "type": 3, "context": 0, "minCount": 1, "maxCount": max_count,
            "option": [{"type": 1} for _ in range(n_options)],
        },
        "current": {
            "turn": 3, "turnActionCount": 1, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False,
            "energyAttached": False, "retreated": False, "result": -1,
            "stadium": [], "looking": None,
            "players": [
                {"active": [], "bench": [], "benchMax": 5, "deckCount": 40,
                 "discard": [], "prize": [None] * 6, "handCount": 3, "hand": [],
                 "poisoned": False, "burned": False, "asleep": False,
                 "paralyzed": False, "confused": False},
                {"active": [], "bench": [], "benchMax": 5, "deckCount": 40,
                 "discard": [], "prize": [None] * 6, "handCount": 3, "hand": [],
                 "poisoned": False, "burned": False, "asleep": False,
                 "paralyzed": False, "confused": False},
            ],
        },
        "log": [],
    })


class _StubPolicy:
    """Returns uniform logits — this test is about lengths, not values."""

    def __call__(self, batch):
        import torch
        b = batch["opt_mask"].shape[0]
        o = batch["opt_mask"].shape[1]
        return (torch.zeros(b, o), torch.zeros(b), None)


@pytest.mark.parametrize("n_options,max_count", [(3, 4), (7, 2), (5, 1), (1, 1)])
def test_priors_length_matches_the_engine_option_count(n_options, max_count):
    """`select_leaf` walks children and reads `priors` at the same index, so
    one extra prior is an unused child slot and one too few is an out-of-bounds
    panic on a rayon worker.  Multi-select (max_count > 1) is where the
    featurizer appends a STOP column that has no child behind it.
    """
    import torch
    from ptcg_rl.search import batch_evaluate_leaves

    leaves = [{
        "tree_id": 0, "obs_json": _leaf_obs(n_options, max_count),
        "player_role": 0, "is_terminal": False, "n_options": n_options,
    }]
    out = batch_evaluate_leaves(
        leaves, _StubPolicy(), {}, torch.device("cpu"), bf16=False,
    )
    assert len(out) == 1, "fixture examined nothing"
    assert len(out[0]["priors"]) == n_options
    assert abs(sum(out[0]["priors"]) - 1.0) < 1e-5, (
        "priors must still be a distribution after STOP is removed"
    )


# ── The engine pool must outlive the search batch ────────────────────────
#
# libcg exports no `AgentEnd`, and `SearchEnd` only returns the arena to
# the same agent for reuse (see cg/api.py's `search_end` docstring), so a
# dropped Engine strands everything it allocated.  Building a forest per
# game leaked ~50 MB/game at --n-engines 20 and put self-play into swap
# around game 2000.  Both tests below fail if a caller goes back to
# constructing-and-closing a forest per batch.

class _FakeForest:
    """Counts the lifecycle calls that decide whether agents leak."""

    instances: list = []

    def __init__(self, n_engines=4, libcg_path=None):
        self.n_engines = n_engines
        self.n_reset = 0
        self.n_close = 0
        self.roots = 0
        _FakeForest.instances.append(self)

    def set_basic_pokemon(self, ids):
        return len(ids)

    def add_root(self, *a, **kw):
        self.roots += 1
        return self.roots - 1

    def select_batch(self, batch_size):
        return []

    def expand_batch(self, expansions):
        return 0

    def results(self):
        return []

    def reset(self):
        self.n_reset += 1
        return 0

    def close(self):
        self.n_close += 1


@pytest.fixture
def fake_forest(monkeypatch):
    from ptcg_rl import search as search_mod

    _FakeForest.instances = []
    monkeypatch.setattr(search_mod, "MctsForest", _FakeForest)
    monkeypatch.setattr(search_mod, "_SHARED_FOREST", None)
    monkeypatch.setattr(search_mod, "_SHARED_FOREST_ENGINES", 0)
    yield _FakeForest
    monkeypatch.setattr(search_mod, "_SHARED_FOREST", None)


def test_shared_forest_builds_one_pool_and_resets_it(fake_forest):
    """Every call after the first must reuse the pool, not build one."""
    from ptcg_rl.search import shared_forest

    first = shared_forest(n_engines=3)
    for _ in range(5):
        again = shared_forest(n_engines=3)
        assert again is first, "a second pool means n_engines more stranded agents"

    assert len(fake_forest.instances) == 1, (
        f"{len(fake_forest.instances)} pools built; libcg can free none of them"
    )
    assert first.n_reset == 5, "reuse must clear the trees between batches"
    assert first.n_close == 0, "close() frees the engines — they never come back"


def test_pooled_mcts_never_closes_the_forest(fake_forest):
    """The per-game search path is where the leak actually ran."""
    import argparse
    from types import SimpleNamespace

    from ptcg_rl.mcts_train import _run_mcts_on_decisions_pooled

    cfg = argparse.Namespace(
        mcts_n_engines=20, mcts_k_determinizations=2, mcts_iterations=8,
        mcts_c_puct=2.0, mcts_leaf_batch=64,
    )
    obs = json.dumps({
        "search_begin_input": "x",
        "select": {"type": 0},
        "current": {"yourIndex": 0},
    })
    decisions = [SimpleNamespace(obs_json=obs) for _ in range(3)]

    n_games = 4
    for _ in range(n_games):
        _run_mcts_on_decisions_pooled(
            decisions, {0, 1, 2}, policy=None, vocab={},
            fixed_deck=[1], opp_deck=[1], config=cfg, device="cpu", seed=0,
            basic_pokemon_ids=[1],
        )

    assert len(fake_forest.instances) == 1, (
        f"{len(fake_forest.instances)} pools for {n_games} games — "
        "20 libcg agents stranded per extra pool"
    )
    forest = fake_forest.instances[0]
    assert forest.n_close == 0, "close() strands the whole engine arena"
    assert forest.roots > 0, "fixture added no roots — the test proved nothing"
    # One reset in the finally per game, plus one per shared_forest() reuse.
    assert forest.n_reset >= n_games, "trees must be dropped between games"


# ── Every search_id must live on the tree's own engine ───────────────────
#
# A `search_id` is meaningful only to the agent that minted it: SearchStep
# and SearchRelease both take the agent pointer and look the id up in that
# agent's table.  `puct_forest_add_root` used to mint every root on engine
# 0 while `realise_leaves`/`all_results`/`reset` addressed engine
# `tree_id % n_engines`, so with --n-engines 20 nineteen trees in twenty
# (a) could not step past their root, and (b) never had their root state
# released — ~22 KB per root, permanently, because libcg has no AgentEnd.
# At --rho 1.0 --k-determinizations 4 that is ~8 MB/game.
#
# This needs the real engine: a fake forest cannot see a Rust-side engine
# index.  It is the only test here that loads libcg.


def test_init_engine_defers_to_cg_sim(monkeypatch):
    """``GameInitialize`` must not be called when ``cg.sim`` already did.

    Importing ``cg.sim`` *is* the call, and libcg only tolerates one per
    process — the second throws a C++ exception that crosses the FFI
    boundary as SIGABRT, so no test can assert on an exception here.  This
    asserts on the call not being made at all.
    """
    import sys as _sys

    from ptcg_rl import rust_vec_env as rve

    called = []
    monkeypatch.setattr(rve, "_engine_initialized", False)
    monkeypatch.setattr(
        rve.ctypes, "CDLL",
        lambda *a, **kw: pytest.fail(
            "CDLL(...).GameInitialize() called while cg.sim was imported — "
            "that is the second call, and it aborts the process"
        ),
    )
    monkeypatch.setitem(_sys.modules, "cg.sim", object())
    rve._init_engine("/nonexistent/libcg.so")
    assert rve._engine_initialized is True, (
        "the engine is initialized; a later call must still be suppressed"
    )
    assert not called

    # Without cg.sim, this module still owns the one call.
    monkeypatch.setattr(rve, "_engine_initialized", False)
    monkeypatch.delitem(_sys.modules, "cg.sim", raising=False)

    class _Lib:
        def GameInitialize(self):  # noqa: N802 — mirrors the C symbol
            called.append(True)

    monkeypatch.setattr(rve.ctypes, "CDLL", lambda *a, **kw: _Lib())
    rve._init_engine("/nonexistent/libcg.so")
    assert called == [True], "nobody else had initialized it — we must"


def _real_forest_available() -> bool:
    try:
        from ptcg_rl.rust_vec_env import _find_libcg, _find_libptcg_search
        _find_libcg()
        _find_libptcg_search()
    except Exception:
        return False
    return True


def _harvest_roots(n_wanted: int = 4):
    """Real observations carrying a `search_begin_input`.

    SearchBegin dereferences that pointer unconditionally, so a root built
    from a synthetic obs takes the process down instead of failing.
    """
    from pathlib import Path

    from ptcg_rl.rust_vec_env import RustVecEnv

    data = Path(__file__).resolve().parent.parent / "data"
    arch = json.loads((data / "archetypes.json").read_text())
    decks: list[list[int]] = []

    def walk(o):
        if isinstance(o, dict):
            for key in ("deck", "decklist", "cards", "representative"):
                v = o.get(key)
                if isinstance(v, list) and len(v) == 60:
                    decks.append([int(x) for x in v])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(arch)
    if len(decks) < 2:
        return [], [], []

    out = []
    with RustVecEnv(deck_self=decks[0], deck_opp=decks[1],
                    n_envs=4, our_player=0, seed=1) as pool:
        for _ in range(400):
            pending = pool.poll()
            if not pending:
                pool.drain()
                continue
            for p in pending:
                obs = json.loads(p["obs_json"])
                obs["search_begin_input"] = p.get("sbi", "")
                if obs.get("search_begin_input") and obs.get("select"):
                    out.append(obs)
            pool.reply([[0] for _ in pending])
            if len(out) >= n_wanted:
                break
    return out, decks[0], decks[1]


@pytest.mark.skipif(not _real_forest_available(),
                    reason="needs libcg.so and a built libptcg_search.so")
def test_every_tree_searches_not_just_the_ones_on_engine_zero():
    """Trees must expand whatever their index modulo the pool size."""
    from ptcg_rl.search import MctsForest

    roots, deck_self, deck_opp = _harvest_roots(4)
    if not roots:
        pytest.skip("no observation with a search_begin_input to root from")

    # Two engines, four trees: engine_idx cycles 0,1,0,1, so trees 1 and 3 sit
    # off engine 0 and that is all the bug needs to show.  Asking for more is
    # not free — libcg registers agents in a fixed-capacity global table
    # (capacity 7), and overflowing it throws a C++ exception across the FFI
    # boundary that no `except` can catch: the whole pytest process SIGABRTs.
    # This test shares its process with every other test that starts an agent,
    # so it takes the smallest pool that can still fail.
    n_engines = 2
    n_trees = 4
    # Its own pool, not `shared_forest`: closing the shared one would strand
    # the arenas of whatever built it and force fresh AgentStart calls here.
    try:
        forest = MctsForest(n_engines=n_engines)
    except Exception as e:  # noqa: BLE001 — agent table may already be full
        pytest.skip(f"could not build a {n_engines}-engine forest: {e}")
    try:
        tree_ids = []
        for r in range(n_trees):
            tid = forest.add_root(
                roots[r % len(roots)], deck_self,
                opp_deck_template=deck_opp,
                iterations=16, c_puct=2.0, seed=r,
            )
            if tid >= 0:
                tree_ids.append(tid)

        # A vacuous pass is the failure mode this whole test exists to
        # avoid: with no trees off engine 0 there is nothing to check.
        off_engine_zero = [t for t in tree_ids if t % n_engines != 0]
        assert off_engine_zero, (
            f"engine refused every root but {tree_ids} — nothing was examined"
        )

        # Uniform priors stand in for the policy: this is about the engine
        # plumbing, not about what the network thinks.
        for _ in range(64):
            leaves = forest.select_batch(256)
            if not leaves:
                break
            forest.expand_batch([
                {"tree_id": lf["tree_id"],
                 "priors": [1.0 / max(int(lf.get("n_options", 1)), 1)]
                           * max(int(lf.get("n_options", 1)), 1),
                 "value": 0.0}
                for lf in leaves
            ])

        by_id = {r["tree_id"]: r for r in forest.results()}
        # A root plus its children and nothing deeper is what a tree looks
        # like when search_step is failing on a foreign agent.
        floor = 4
        starved = {t: by_id[t]["nodes_created"]
                   for t in off_engine_zero
                   if by_id.get(t, {}).get("nodes_created", 0) <= floor}
        assert not starved, (
            f"trees {sorted(starved)} grew to {starved} nodes against "
            f"tree 0's {by_id.get(0, {}).get('nodes_created')} — their "
            f"search_ids are being stepped on the wrong agent"
        )
    finally:
        forest.close()


# ── W&B payloads ──────────────────────────────────────────────────────
#
# The MCTS logger builds keys from run-time names (opponent checkpoints,
# ELO model ids), so a payload can go wrong in ways `wandb.init` never
# sees: a non-numeric value, a missing step metric, or a name carrying a
# "/" that silently reparents the series under another panel group.


class _StubWandbRun:
    def __init__(self):
        self.payloads: list[dict] = []
        self.metrics: list[tuple] = []

    def define_metric(self, name, step_metric=None):
        self.metrics.append((name, step_metric))

    def log(self, payload):
        self.payloads.append(payload)

    def finish(self):
        pass


def _live_logger():
    from ptcg_rl.mcts_train import _MctsWandbLogger
    wb = _MctsWandbLogger.__new__(_MctsWandbLogger)
    wb.active = True
    wb._run = _StubWandbRun()
    return wb


_SP_STATS = {
    "sp_games": 8, "sp_sec": 12.5, "sp_games_per_min": 38.4,
    "sp_decisions": 400, "sp_mcts_decisions": 120, "sp_dec_per_game": 50.0,
    "sp_win_rate": 0.625, "sp_wins": 5, "sp_losses": 2, "sp_draws": 1,
    "sp_opp_distribution": {0: 3, 11: 5},
    "perf": {"t_featurize": 1.0, "t_forward": 2.0, "n_actor_calls": 300,
             "t_sweep": 0.5, "t_act": 0.25, "n_act_calls": 40, "t_other": 8.75},
}

_LEAGUE = {"passed": False, "results": {
    "IL_baseline": {"win_rate": 0.55, "wins": 11, "losses": 9, "draws": 0,
                    "total": 20, "passed": True},
    "a0/ckpt-best": {"win_rate": 0.60, "wins": 12, "losses": 8, "draws": 0,
                     "total": 20, "passed": True},
}}


def _elo_tracker():
    from ptcg_rl.mcts_train import EloTracker
    elo = EloTracker()
    elo.add_model("ckpt-best", fixed=True)
    elo.add_model("mcts-000008")
    elo.update("mcts-000008", "ckpt-best", 12, 8)
    return elo


def test_every_payload_is_numeric_and_carries_the_step():
    wb = _live_logger()
    elo = _elo_tracker()
    wb.log_sp(_SP_STATS, 8)
    wb.log_train({"pi_ce": 1.2, "ev": None, "kl": 0.04, "note": "skip me"}, 8)
    wb.log_eval(_LEAGUE, 1, 8)
    wb.log_elo(elo, "mcts-000008", elo.top_pct(0.70), 8)

    assert len(wb._run.payloads) == 4
    for payload in wb._run.payloads:
        assert payload["mcts/step"] == 8, f"no step metric in {sorted(payload)}"
        for k, v in payload.items():
            assert isinstance(v, (int, float)), f"{k}={v!r} is not a scalar"
    train = wb._run.payloads[1]
    assert "train/ev" not in train, "NaN-guarded ev must be dropped, not logged"
    assert "train/note" not in train, "non-numeric value leaked into the payload"


def test_opponent_names_with_a_slash_do_not_reparent_the_series():
    wb = _live_logger()
    wb.log_eval(_LEAGUE, 1, 8)
    keys = [k for k in wb._run.payloads[0] if k.startswith("eval/wr/")]
    assert "eval/wr/a0_ckpt-best" in keys, keys
    assert all(k.count("/") == 2 for k in keys), (
        f"a '/' in a checkpoint stem split the metric: {keys}")


def test_signals_that_used_to_be_console_only_now_reach_wandb():
    wb = _live_logger()
    elo = _elo_tracker()
    wb.log_sp(_SP_STATS, 8)
    wb.log_eval(_LEAGUE, 1, 8)
    wb.log_elo(elo, "mcts-000008", elo.top_pct(0.70), 8)
    keys = {k for p in wb._run.payloads for k in p}
    for expected in ("sp/wins", "sp/losses", "sp/draws",
                     "sp/opp_share/11", "sp/perf/t_featurize",
                     "eval/wr/IL_baseline", "eval/wr_vs_il",
                     "elo/rating", "elo/threshold", "elo/margin",
                     "elo/rating/ckpt-best"):
        assert expected in keys, f"{expected} is still console-only"


def test_model_dump_is_one_log_call():
    import torch.nn as nn
    wb = _live_logger()
    policy = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    policy(__import__("torch").randn(2, 4)).sum().backward()
    wb.log_model(policy, 8)
    assert len(wb._run.payloads) == 1, (
        f"{len(wb._run.payloads)} calls for 4 params — one per parameter "
        "floods the wandb step counter")
    keys = wb._run.payloads[0]
    assert "model/0.weight_mean" in keys and "grad/0.weight_std" in keys


def test_inactive_logger_is_a_noop_on_every_method():
    from ptcg_rl.mcts_train import _MctsWandbLogger
    import torch.nn as nn
    wb = _MctsWandbLogger.__new__(_MctsWandbLogger)
    wb.active = False   # --no-wandb, or a failed wandb.init
    wb.log_sp(_SP_STATS, 0)
    wb.log_train({"pi_ce": 1.0}, 0)
    wb.log_eval(_LEAGUE, 0, 0)
    wb.log_elo(_elo_tracker(), "mcts-000008", 1500.0, 0)
    wb.log_model(nn.Linear(2, 2), 0)
    wb.finish()


# ── Seat orientation and the deck/seat contract ───────────────────────
#
# The engine deals deck_self to seat 0 and deck_opp to seat 1 regardless of
# `our_player` (ptcg_search/src/vec_env.rs Battle::start), while the reward
# arrives in one fixed seat's frame.  Every bug in this block is invisible in
# the loss: a sign-flipped value target is still finite, and a deck swap in
# eval still produces a plausible win rate.


def test_outcome_is_expressed_in_the_moving_players_frame():
    from ptcg_rl.mcts_train import SELF_SEAT, _orient_outcome
    # A win for SELF_SEAT is a loss for the player sitting opposite.
    assert _orient_outcome(1.0, SELF_SEAT) == 1.0
    assert _orient_outcome(1.0, 1 - SELF_SEAT) == -1.0
    assert _orient_outcome(-1.0, 1 - SELF_SEAT) == 1.0
    # Draws have no frame to flip.
    assert _orient_outcome(0.0, 1 - SELF_SEAT) == 0.0


def test_orientation_matches_the_puct_expand_leaf_contract():
    """puct.rs negates the NN value when player_role == 1.

    The critic is therefore contracted to speak in the perspective of the
    player to move.  Training labels must use the same convention or the
    search double-counts the flip on every opponent node.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "ptcg_search" / "src" / "puct.rs"
    text = src.read_text()
    assert "if player_role == 1 { -value } else { value }" in text, (
        "expand_leaf no longer negates by player_role — the training-side "
        "orientation in _orient_outcome must be revisited to match")


def test_eval_keeps_theta_on_its_own_deck_in_both_seats():
    """The seat loop must swap turn order, not decklists."""
    from ptcg_rl.mcts_train import _decks_by_seat
    fixed, opp = [1, 2, 3], [7, 8, 9]
    for theta_seat in (0, 1):
        seats = _decks_by_seat(fixed, opp, theta_seat)
        assert seats[theta_seat] == fixed, (
            f"theta_seat={theta_seat} gave theta {seats[theta_seat]}, not its "
            "own deck — that is a deck swap wearing a seat swap's clothes")
        assert seats[1 - theta_seat] == opp


def test_eval_routes_its_decks_through_the_seat_mapping():
    import inspect
    from ptcg_rl import mcts_train
    src = inspect.getsource(mcts_train._eval_game_batch)
    assert "_decks_by_seat(fixed_deck, opp_deck, our_player)" in src, (
        "eval builds its deck pair directly again — the seat mapping is the "
        "only thing keeping theta on its own archetype")


# ── League opponents bring their own deck ────────────────────────────────
#
# Evaluation is best-deck vs best-deck.  It used to draw the opponent's deck
# uniformly from the whole archetype pool, once per seat per round, so the
# opponent played a deck it was not trained on and the round-to-round delta
# tracked which two decks came up rather than anything about theta.


def test_specialist_opponent_plays_its_own_deck():
    from ptcg_rl.mcts_train import _opponent_deck
    theta_deck = [17] * 60
    record = {"specialist": True, "archetype_self": 1, "deck": [7] * 60}
    assert _opponent_deck(record, theta_deck) == [7] * 60, (
        "a specialist opponent was not given the decklist its own checkpoint "
        "records — it is being evaluated on a deck it never trained on")


def test_generalist_opponent_mirrors_theta():
    """A generalist has no best deck; its record carries archetypes.json's
    global fixed_deck, which it is no better at than any other."""
    from ptcg_rl.mcts_train import _opponent_deck
    theta_deck = [17] * 60
    record = {"specialist": False, "archetype_self": None,
              "is_fixed_deck": True, "deck": [21] * 60}
    assert _opponent_deck(record, theta_deck) == theta_deck, (
        "the generalist was handed its stored fixed_deck instead of theta's — "
        "eval/wr_vs_il then measures deck strength, not policy strength")


def test_opponent_without_a_deck_record_is_dropped():
    """Returning None is the signal to skip; anything else is a guess."""
    from ptcg_rl.mcts_train import _opponent_deck
    theta_deck = [17] * 60
    for record in (None, {}, "not-a-dict",
                   {"specialist": True, "deck": []},
                   {"specialist": True}):
        assert _opponent_deck(record, theta_deck) is None, (
            f"record {record!r} produced a deck instead of None — a league "
            "opponent would be evaluated holding a deck nobody chose")


def test_league_evaluate_takes_decks_from_its_caller():
    """No pool, no rng: an opponent deck re-drawn each round makes successive
    eval rounds incomparable, which is what the ELO gate reads."""
    import inspect
    from ptcg_rl import mcts_train

    sig = inspect.signature(mcts_train.league_evaluate)
    assert "opp_decks" not in sig.parameters, (
        "league_evaluate still takes the archetype pool — it can only be "
        "sampling opponent decks from it")
    assert "rng" not in sig.parameters, (
        "league_evaluate still takes an rng; opponent decks are decided by "
        "the caller and nothing in there should be random")

    src = inspect.getsource(mcts_train.league_evaluate)
    assert "for name, opp, opp_deck in opponents:" in src, (
        "opponents no longer carry their own deck")


def test_deck_record_failure_stops_the_run_at_startup_not_at_save():
    """A champion is written after hours of self-play.  Discovering the deck
    label is unbuildable at that point throws the run away, so `main` builds
    and validates it before the first game."""
    import inspect
    from ptcg_rl import mcts_train

    src = inspect.getsource(mcts_train.main)
    assert "raise SystemExit(" in src.split("Deck label")[0][-1200:] or \
        "Refusing to start" in src, (
        "main no longer refuses to start on an unbuildable deck record")
    assert "MCTS champions will lack it" not in src, (
        "main still warns-and-continues with _deck_record = None; champions "
        "written by that run cannot be evaluated or shipped")

    # And the writer is required to have one, so no other caller can regress.
    sig = inspect.signature(mcts_train._save_ckpt)
    assert sig.parameters["deck_record"].default is inspect.Parameter.empty, (
        "_save_ckpt's deck_record is optional again")
    sig = inspect.signature(mcts_train._manage_champions)
    assert sig.parameters["deck_record"].default is inspect.Parameter.empty, (
        "_manage_champions' deck_record is optional again")


def test_rl_refuses_a_parent_checkpoint_with_no_deck_record():
    """ptcg_rl.train copies the record forward from --il-ckpt, so an unlabelled
    parent silently produces an unlabelled child, generation after generation."""
    import inspect
    from ptcg_rl import train as rl_train

    src = inspect.getsource(rl_train._save_checkpoint)
    assert "require_deck_record(" in src, (
        "_save_checkpoint takes the parent's record on faith again")
    assert "if il_ckpt.get(DECK_KEY):" not in src, (
        "the silent `if parent has one` branch is back — a missing record is "
        "inherited as a missing record")


def test_require_deck_record_is_one_shared_rule():
    """Three packages write checkpoints; a per-writer rule would let one drift
    into leniency without any test noticing."""
    from ptcg_il.deck import require_deck_record

    good = {"specialist": True, "deck": [7] * 60}
    assert require_deck_record(good, "test") is good
    for bad in (None, {}, "not-a-dict", {"deck": []}, {"deck": "60 cards"}):
        with pytest.raises(ValueError, match="deck record"):
            require_deck_record(bad, "test")


def test_empty_league_does_not_pass_its_gate():
    """all_passed starts True, so a league with no opponents reports PASS on
    zero games — and the caller promotes a champion on that."""
    from ptcg_rl import mcts_train

    class _Args:
        eval_games = 4
        gate_score = 0.55
        data_dir = "data"

    result = mcts_train.league_evaluate(
        theta=None, opponents=[], vocab={}, fixed_deck=[17] * 60,
        config=_Args(), device=None,
    )
    assert result["results"] == {}
    # The guard lives in the caller, which must refuse to act on this.
    import inspect
    src = inspect.getsource(mcts_train.main)
    assert "if not opponents:" in src, (
        "the eval phase no longer checks for an empty league — league_evaluate "
        "returns passed=True on zero opponents and a champion is promoted")


def _decision(seat: int, turn: int = 0):
    from ptcg_rl.vec_env import Decision
    return Decision(
        features={}, picks=[0], action_idx=np.zeros(1, dtype=np.int64),
        action_len=1, logp=-0.5, value=0.0, turn=turn, your_index=seat,
    )


def test_decision_carries_the_seat_that_moved():
    """Without this field the seat is unrecoverable: the featurizer indexes
    every zone as [your_index, 1 - your_index], so both seats produce the
    same egocentric tensor."""
    assert _decision(1).your_index == 1
    assert _decision(0).your_index == 0


def test_opponent_seat_decisions_are_dropped_from_training():
    """theta drives both seats but is a specialist on one deck."""
    import inspect
    from ptcg_rl import mcts_train
    from ptcg_rl.mcts_train import SELF_SEAT

    src = inspect.getsource(mcts_train.run_self_play_games)
    assert "d.your_index == SELF_SEAT" in src, (
        "self-play no longer filters by seat — theta would train on the "
        "opponent archetype it plays badly")
    assert "for i in own_idx:" in src, (
        "samples are built from all decisions again, not the own-seat subset")
    assert "_orient_outcome(" in src, (
        "game_outcome is stored unoriented — seat-1 labels would be inverted")

    # The filter the source applies, exercised on real Decisions.
    decisions = [_decision(SELF_SEAT, 0), _decision(1 - SELF_SEAT, 1),
                 _decision(SELF_SEAT, 2), _decision(1 - SELF_SEAT, 3)]
    own_idx = [i for i, d in enumerate(decisions)
               if d.your_index == SELF_SEAT]
    assert own_idx == [0, 2], own_idx
    assert len(decisions) - len(own_idx) == 2, "drop counter would be wrong"


def test_advantage_weighting_pushes_losing_actions_down():
    """Unweighted cloning reinforces a loss as hard as a win.

    normalize_advantages is zero-mean, so in a batch with one win and one
    loss the losing sample must carry a negative weight — the gradient on its
    log-prob then points away from the action taken.
    """
    import torch
    from ptcg_rl.rollout import normalize_advantages
    rets = torch.tensor([1.0, -1.0])
    values = torch.zeros(2)
    adv = normalize_advantages(rets - values)
    assert adv[0] > 0 and adv[1] < 0, adv
    # And the loss term inherits that sign: -adv * logp.
    logp = torch.tensor([-0.7, -0.7])
    per_sample = -adv * logp
    assert per_sample[1] < 0 < per_sample[0], (
        f"losing sample did not flip sign: {per_sample}")


def test_train_step_weights_the_clone_term_by_advantage():
    """The clone term must carry the advantage, however it is spelled.

    This used to pin the exact expression ``loss_pi - adv[i] * logp[0]``.
    That broke the moment the branch was batched into
    ``loss_pi - (adv[rows] * logp_bc).sum()`` — the same quantity, verified
    equal to 1e-7, but a different string.  Assert the two things that
    actually matter: the advantage reaches the clone term, and it is
    normalized before it does.
    """
    import inspect
    from ptcg_rl import mcts_train
    src = inspect.getsource(mcts_train.train_step)
    assert "normalize_advantages" in src, (
        "advantages are no longer normalized before weighting the clone term")
    assert ("adv[rows]" in src or "adv[i]" in src), (
        "behaviour cloning is unweighted again — won and lost games would be "
        "reinforced identically")
    # The sign must be a subtraction: -adv * logp, so a negative advantage
    # pushes probability *away* from the action taken.
    assert "loss_pi - (adv[" in src or "loss_pi - adv[" in src, (
        "the clone term stopped being subtracted; a losing action would be "
        "reinforced rather than suppressed")


# ── KL anchor and search-value targets ────────────────────────────────


def _tiny_policy(seed: int = 0, D: int = 64):
    import torch
    from ptcg_il.model.policy import Policy
    torch.manual_seed(seed)
    p = Policy(D=D, heads=4, layers=1, ff=128)
    p.train()
    return p


def _tiny_batch(n: int = 4, with_mcts: bool = False, seed: int = 0):
    """Samples shaped like ``run_self_play_games`` output.

    Built by splitting the IL suite's ``_make_synthetic_batch`` back into
    per-sample numpy dicts, so the key set and shapes cannot drift from what
    the embedder and pointer actually read — ``train_step`` re-stacks them
    through ``_collate_feat_list``.
    """
    import torch
    from tests.test_model_policy import _make_synthetic_batch

    torch.manual_seed(seed)
    x = _make_synthetic_batch(n, max_count=1)
    out = []
    for i in range(n):
        feats = {k: v[i].numpy() for k, v in x.items()}
        # Match what featurize() emits during self-play: the observation is
        # featurized before an action is chosen, so the label keys carry the
        # empty sentinel and the real picks live only on the sample.
        feats["action_idx"] = np.full_like(feats["action_idx"], -1)
        feats["action_len"] = np.int64(0)
        out.append({
            "features": feats,
            "action_idx": np.array([0, -1], dtype=np.int64),
            "action_len": 1,
            "game_outcome": 1.0 if i % 2 == 0 else -1.0,
            "mcts_pi": None,
            "mcts_value": 0.25 if with_mcts else None,
        })
    return out


@pytest.fixture
def train_cfg():
    from types import SimpleNamespace
    return SimpleNamespace(c_pi=1.0, c_value=0.5, beta=1.0, grad_clip=1.0)


@pytest.fixture
def unclipped_cfg():
    """`train_cfg` with the gradient clip effectively off.

    `clip_grad_norm_(params, 1.0)` rewrites `p.grad` in place to norm 1.0, so
    any test that reads gradients after `train_step` under `train_cfg` is
    reading a renormalised copy and cannot see a scale error at all.
    """
    from types import SimpleNamespace
    return SimpleNamespace(c_pi=1.0, c_value=0.5, beta=1.0, grad_clip=1e9)


def test_kl_anchor_produces_gradient_not_just_a_log_line(train_cfg):
    """The whole point of an anchor is that it pulls.

    Under `with torch.no_grad()` the term is a constant added to total_loss:
    finite, plausible, logged — and worth exactly zero to the optimizer.
    """
    import torch
    from ptcg_rl.mcts_train import train_step

    policy = _tiny_policy(0)
    frozen = _tiny_policy(1)          # a *different* policy, so KL > 0
    for p in frozen.parameters():
        p.requires_grad_(False)
    batch = _tiny_batch()
    device = torch.device("cpu")

    # Isolate the anchor: no policy term, no value term.
    cfg_kl_only = type(train_cfg)(c_pi=0.0, c_value=0.0, beta=1.0,
                                  grad_clip=1e9)
    opt = torch.optim.SGD(policy.parameters(), lr=0.0)
    stats = train_step(policy, opt, batch, cfg_kl_only, device,
                       frozen_il=frozen)

    assert stats["grad_norm"] > 0.0, (
        "KL is the only active loss term and produced no gradient — the "
        "anchor is inert")
    assert stats["kl_k3"] >= 0.0, "k3 must be non-negative by construction"


def test_kl_is_zero_against_itself(train_cfg):
    import torch
    from ptcg_rl.mcts_train import train_step

    policy = _tiny_policy(0)
    frozen = _tiny_policy(0)          # identical weights
    opt = torch.optim.SGD(policy.parameters(), lr=0.0)
    stats = train_step(policy, opt, _tiny_batch(), train_cfg,
                       torch.device("cpu"), frozen_il=frozen)
    assert abs(stats["kl"]) < 1e-5, stats["kl"]
    assert stats["kl_k3"] < 1e-5, stats["kl_k3"]


def test_critic_target_is_the_game_outcome_not_the_search_value(train_cfg):
    """AlphaZero: l = (z - v)^2 - pi_MCTS . log p.

    The policy target comes from search; the value target is the final
    result.  Training v on the root value of a search whose leaves are this
    same critic closes a loop with no external signal, and every constant is
    a fixed point of it — measured once as target std 1.0 -> 0.022 and
    ev 0.669 -> -0.199.
    """
    import torch
    from ptcg_rl.mcts_train import train_step

    device = torch.device("cpu")

    policy = _tiny_policy(0)
    opt = torch.optim.SGD(policy.parameters(), lr=0.0)
    with_search = train_step(policy, opt, _tiny_batch(with_mcts=True),
                             train_cfg, device)
    # Search values are present and reported...
    assert with_search["search_target_frac"] == 1.0
    # ...but the critic still regresses to the ±1 outcomes, which average to
    # 0 here.  A 0.25 would mean the search value became the target again.
    assert abs(with_search["ret_mean"] - 0.0) < 1e-6, with_search["ret_mean"]
    assert with_search["target_std"] > 0.9, (
        f"target std {with_search['target_std']:.4f} — the critic is fitting "
        "something far flatter than a ±1 outcome")

    policy2 = _tiny_policy(0)
    opt2 = torch.optim.SGD(policy2.parameters(), lr=0.0)
    without = train_step(policy2, opt2, _tiny_batch(with_mcts=False),
                         train_cfg, device)
    assert without["search_target_frac"] == 0.0
    assert abs(without["ret_mean"] - 0.0) < 1e-6, without["ret_mean"]


# ── Gradient accumulation must not change the update ─────────────────────
#
# Activations are ~99% of train_step's memory (the model is 6.1M params —
# 0.12 GB with grads and Adam states), so the only lever on a 16 GiB card is
# how many rows' activations are alive at once.  --grad-accum splits the
# batch and steps once, which is worth nothing if it also changes the update.


def test_grad_accum_matches_a_single_step(unclipped_cfg):
    """Same effective batch → the same gradient, in size as well as direction.

    Not bit-identical: `normalize_advantages` runs per microbatch, so the
    advantage *scale* shifts slightly.  Magnitude must still land close.

    Note `unclipped_cfg`, not `train_cfg`: the latter sets grad_clip=1.0, and
    `clip_grad_norm_` then rescales every gradient to norm exactly 1.0 before
    this test can read it.  Dropping the `/ n_micro` — an n_micro-fold
    over-scaling — was completely invisible under that fixture.
    """
    import torch
    from ptcg_rl.mcts_train import train_step_accum

    device = torch.device("cpu")
    batch = _tiny_batch(n=16)

    def grads(accum):
        policy = _tiny_policy(0)
        # lr=0: the weights never move, so every run differentiates the same
        # point and the comparison is of gradients, not of trajectories.
        opt = torch.optim.SGD(policy.parameters(), lr=0.0)
        stats = train_step_accum(policy, opt, batch, unclipped_cfg, device,
                                 grad_accum=accum)
        g = torch.cat([p.grad.flatten() for p in policy.parameters()
                       if p.grad is not None])
        return g, stats

    g1, s1 = grads(1)
    assert g1.norm() > 0, "fixture produced no gradient — nothing was tested"
    for accum in (2, 4):
        gn, sn = grads(accum)
        cos = torch.nn.functional.cosine_similarity(g1, gn, dim=0).item()
        assert cos > 0.99, (
            f"grad_accum={accum} points somewhere else (cos={cos:.4f}) — "
            "the accumulated gradient is not the batch gradient"
        )
        # Magnitude is the half that cosine cannot see, and it is exactly
        # what a missing `/ n_micro` breaks.
        ratio = gn.norm().item() / g1.norm().item()
        assert 0.85 < ratio < 1.18, (
            f"grad_accum={accum} scaled the gradient by {ratio:.3f} — the "
            "microbatch losses are not being divided by n_micro"
        )
        rel = (g1 - gn).norm().item() / max(g1.norm().item(), 1e-12)
        assert rel < 0.15, f"grad_accum={accum} rescaled the update by {rel:.3f}"
        assert sn["n_pi"] == s1["n_pi"], (
            "microbatching dropped rows from the policy term"
        )


def test_grad_accum_keeps_every_stat_key(train_cfg):
    """The merge must not drop keys, finite or not.

    `main` reads stats["loss"], stats["grad_norm"] and friends
    unconditionally.  Skipping a non-finite microbatch value instead of
    propagating it made the key vanish, turning a NaN loss — which the run
    needs to see — into a KeyError three frames away.
    """
    import torch
    from ptcg_rl.mcts_train import train_step, train_step_accum

    device = torch.device("cpu")
    batch = _tiny_batch(n=8)

    single = train_step(_tiny_policy(0),
                        torch.optim.SGD(_tiny_policy(0).parameters(), lr=0.0),
                        batch, train_cfg, device)
    policy = _tiny_policy(0)
    merged = train_step_accum(policy,
                              torch.optim.SGD(policy.parameters(), lr=0.0),
                              batch, train_cfg, device, grad_accum=4)

    missing = set(single) - set(merged)
    assert not missing, f"grad_accum dropped stat keys: {sorted(missing)}"


def test_grad_accum_reports_the_whole_gradient_norm(train_cfg):
    """grad_norm is measured on the last microbatch only.

    It is the norm of the *accumulated* gradient there.  Averaging it with
    the zeros the earlier microbatches report would divide it by n_micro and
    make --grad-clip look like it never fires.
    """
    import torch
    from ptcg_rl.mcts_train import train_step_accum

    device = torch.device("cpu")
    batch = _tiny_batch(n=16)

    policy = _tiny_policy(0)
    one = train_step_accum(policy, torch.optim.SGD(policy.parameters(), lr=0.0),
                           batch, train_cfg, device, grad_accum=1)
    policy4 = _tiny_policy(0)
    four = train_step_accum(policy4,
                            torch.optim.SGD(policy4.parameters(), lr=0.0),
                            batch, train_cfg, device, grad_accum=4)

    assert one["grad_norm"] > 0, "fixture produced no gradient — nothing tested"
    ratio = four["grad_norm"] / one["grad_norm"]
    assert 0.8 < ratio < 1.25, (
        f"grad_norm={four['grad_norm']:.4f} vs {one['grad_norm']:.4f} "
        f"(ratio {ratio:.3f}) — it is being averaged across microbatches, "
        "not read once from the accumulated gradient"
    )


def test_search_outcome_correlation_is_none_when_undefined():
    """0.0 and "no data" are opposite diagnoses.

    A search that produced no roots must not report the same number as a
    search whose values are uncorrelated with the outcome — one says the
    determinizer is failing, the other condemns mcts_pi.
    """
    from ptcg_rl.mcts_train import _paired_corr

    assert _paired_corr([], []) is None
    assert _paired_corr([0.3], [1.0]) is None, "one pair defines no corr"
    assert _paired_corr([0.5, 0.5, 0.5], [1.0, -1.0, 1.0]) is None, (
        "a constant search value has no correlation — this is exactly the "
        "collapsed case, and 0.0 would hide that it was constant")
    r = _paired_corr([0.9, -0.8, 0.4, -0.5], [1.0, -1.0, 1.0, -1.0])
    assert r is not None and r > 0.9, r


def test_search_outcome_correlation_reaches_the_stats(train_cfg):
    import torch
    from ptcg_rl.mcts_train import train_step

    batch = _tiny_batch(n=4, with_mcts=True)
    # Make the search value track the outcome, so corr is defined and high.
    for s in batch:
        s["mcts_value"] = 0.8 * s["game_outcome"]

    policy = _tiny_policy(0)
    opt = torch.optim.SGD(policy.parameters(), lr=0.0)
    stats = train_step(policy, opt, batch, train_cfg, torch.device("cpu"))

    assert stats["search_outcome_corr"] is not None
    assert stats["search_outcome_corr"] > 0.99, stats["search_outcome_corr"]

    # Constant search values → undefined, not 0.0.
    for s in batch:
        s["mcts_value"] = 0.25
    stats2 = train_step(policy, opt, batch, train_cfg, torch.device("cpu"))
    assert stats2["search_outcome_corr"] is None


def test_k3_is_non_negative_where_the_raw_mean_is_not():
    """Minimising the raw mean of d drives log pi_theta(a) to -inf."""
    import torch
    d = torch.tensor([-2.0, -1.0, -0.5])   # theta below the reference
    assert d.mean() < 0, "fixture must exercise the negative case"
    k3 = (-d).exp() - 1.0 + d
    assert (k3 >= 0).all(), k3
    assert k3.mean() > 0


def test_kl_is_measured_on_the_actions_actually_taken():
    """`features` is featurized before the action exists.

    Its action_idx is the all -1 sentinel with action_len 0, and
    recompute_logp returns 0.0 for a row with no picks — so a KL computed off
    the raw feature batch is 0 - 0 for every sample.  It reads a plausible
    `kl=0.0000` forever and no amount of drift changes it.
    """
    import torch
    from ptcg_rl.mcts_train import _attach_actions
    from ptcg_rl.search import _collate_feat_list

    batch = _tiny_batch(n=3)
    device = torch.device("cpu")
    tb = _collate_feat_list([s["features"] for s in batch], device)

    # What the featurizer handed us, before the fix-up.
    assert int(tb["action_len"].sum()) == 0, (
        "fixture no longer carries the empty sentinel — this test would "
        "pass vacuously")
    assert bool((tb["action_idx"] == -1).all())

    n = _attach_actions(tb, batch, device)
    assert n == 3, f"only {n}/3 rows got their picks back"
    assert int(tb["action_len"].sum()) == 3
    assert int(tb["action_idx"][0, 0]) == 0


def test_attach_actions_leaves_actionless_rows_empty():
    import torch
    from ptcg_rl.mcts_train import _attach_actions
    from ptcg_rl.search import _collate_feat_list

    batch = _tiny_batch(n=3)
    batch[1]["action_len"] = 0          # e.g. an auto-advance decision
    device = torch.device("cpu")
    tb = _collate_feat_list([s["features"] for s in batch], device)
    n = _attach_actions(tb, batch, device)

    assert n == 2
    assert int(tb["action_len"][1]) == 0
    assert bool((tb["action_idx"][1] == -1).all()), (
        "an actionless row must keep the sentinel, not inherit a neighbour's "
        "picks")


def test_no_name_is_used_before_it_is_bound():
    """A name read above its own assignment is an UnboundLocalError.

    The training loop is the one path the unit tests cannot execute — it needs
    libcg — so a metric added to the console line above where its accumulator
    is bound crashes at the *end of the first training phase*, after minutes
    of self-play.  pyflakes sees it statically; skipped, not failed, when
    pyflakes is not installed, since it is not a project dependency.
    """
    import ast
    from pathlib import Path

    pytest.importorskip("pyflakes")
    from pyflakes.checker import Checker
    from pyflakes.messages import UndefinedLocal, UndefinedName

    root = Path(__file__).resolve().parents[1] / "ptcg_rl"
    checked = 0
    problems: list[str] = []
    for name in ("mcts_train.py", "rust_vec_env.py", "vec_env.py"):
        path = root / name
        tree = ast.parse(path.read_text(), filename=str(path))
        for msg in Checker(tree, filename=str(path)).messages:
            if isinstance(msg, (UndefinedName, UndefinedLocal)):
                problems.append(str(msg))
        checked += 1

    assert checked == 3, "fixture examined no files — the test proved nothing"
    assert not problems, "\n".join(problems)


def test_distillation_path_runs_end_to_end_with_the_anchor():
    """The configuration a fresh ``--mcts-distill`` run actually uses:
    pi targets, search values, and a live KL anchor in one step."""
    import torch
    from types import SimpleNamespace

    from ptcg_rl.mcts_train import train_step

    batch = _tiny_batch(n=4, with_mcts=True)
    for s in batch:
        # A visit distribution over the options the mask calls legal.
        s["mcts_pi"] = np.array([0.5, 0.3, 0.2], dtype=np.float32)

    policy = _tiny_policy(0)
    frozen = _tiny_policy(1)
    for p in frozen.parameters():
        p.requires_grad_(False)
    cfg = SimpleNamespace(c_pi=1.0, c_value=0.5, beta=0.1, grad_clip=1.0)
    opt = torch.optim.SGD(policy.parameters(), lr=1e-3)

    stats = train_step(policy, opt, batch, cfg, torch.device("cpu"),
                       frozen_il=frozen)

    for k in ("loss", "pi_ce", "v_mse", "kl", "kl_k3", "grad_norm"):
        assert np.isfinite(stats[k]), f"{k} = {stats[k]}"
    assert stats["n_pi"] == 4, "pi targets were silently skipped"
    assert stats["search_target_frac"] == 1.0
    assert stats["grad_norm"] > 0.0


# ── Seat-1 pilots: routing, ordering, recording ──────────────────────────
#
# θ used to answer both seats.  Seat 1 is dealt a *sampled* archetype and θ is
# a specialist on one deck, so the opponent was playing 60 cards it had never
# seen — the self-play win rate measured that, not θ's edge.  These cover the
# three ways the split can go wrong silently: wrong seat gets the wrong actor,
# picks come back in the wrong order, or the opponent's decisions leak into
# θ's replay buffer.


def _obs_seat(turn: int, seat: int) -> str:
    return json.dumps({"turn": turn, "current": {"yourIndex": seat}})


def _tagged_act_fn(tag, seen):
    """An act_fn that records what it was asked and stamps its replies."""
    def act(requests):
        seen.extend(json.loads(r["obs"]) if isinstance(r["obs"], str)
                    else r["obs"] for r in requests)
        return [
            {"picks": [tag], "features": {"who": tag}, "action_idx": [tag],
             "action_len": 1, "logp": float(tag), "value": float(tag)}
            for _ in requests
        ]
    return act


def _mixed_poll(seats, turn=1):
    """One poll cycle: *seats* battles each waiting on the named seat."""
    return [
        {"battle_idx": i, "obs_json": _obs_seat(turn, s), "sbi": "",
         "select_player": s}
        for i, s in enumerate(seats)
    ]


def _alternating_polls(seats):
    """Successive poll cycles on a *single* battle, one seat each.

    Recording is per battle, so the seats have to arrive on the same
    ``battle_idx`` over consecutive polls — which is also how a real battle
    behaves, alternating turns — for the trajectory to hold both of them.
    """
    return [
        [{"battle_idx": 0, "obs_json": _obs_seat(turn, s), "sbi": "",
          "select_player": s}]
        for turn, s in enumerate(seats, start=1)
    ] + [[]]


def test_seat_routing_sends_each_seat_to_its_own_actor():
    """Seat 0 → θ, every other seat → the opponent pilot."""
    seats = [0, 1, 1, 0, 1]
    env = _FakePolledEnv([_mixed_poll(seats), []])
    ours, theirs = [], []
    env.collect_batch(
        _tagged_act_fn(0, ours), n_decisions=1,
        act_fn_opp=_tagged_act_fn(1, theirs),
    )

    assert len(ours) == seats.count(0) > 0, "no seat-0 obs examined"
    assert len(theirs) == seats.count(1) > 0, "no seat-1 obs examined"
    assert {o["current"]["yourIndex"] for o in ours} == {0}
    assert {o["current"]["yourIndex"] for o in theirs} == {1}


def test_seat_routing_replies_stay_in_poll_order():
    """``vec_env_reply`` is positional against the batch ``vec_env_poll``
    handed out.  Returning the two seat groups concatenated would apply each
    seat's picks to the other seat's battles — every move legal, every move in
    the wrong game.  Mutation: return ``ours + theirs`` from ``_route_by_seat``
    and this goes red.
    """
    seats = [0, 1, 1, 0, 1]
    env = _FakePolledEnv([_mixed_poll(seats), []])
    env.collect_batch(
        _tagged_act_fn(0, []), n_decisions=1,
        act_fn_opp=_tagged_act_fn(1, []),
    )

    assert env.replies, "reply() was never called"
    picks = env.replies[0]
    assert len(picks) == len(seats)
    # Each actor stamps its own tag into picks, so the pick sequence must be
    # the seat sequence poll() emitted.
    assert [p[0] for p in picks] == seats


def test_seat_routing_records_only_theta_decisions():
    """The opponent's logp/value describe a different policy; letting them
    into ``traj.decisions`` puts another model's action statistics into θ's
    replay buffer."""
    env = _FakePolledEnv(_alternating_polls([0, 1, 0]))
    done = env.collect_batch(
        _tagged_act_fn(0, []), n_decisions=99,
        act_fn_opp=_tagged_act_fn(1, []),
    )

    assert len(done) == 1
    decisions = done[0].decisions
    assert len(decisions) == 2, "expected exactly θ's two seat-0 decisions"
    assert [d.your_index for d in decisions] == [0, 0]
    # _tagged_act_fn stamps its tag into logp; 1.0 would be the opponent's.
    assert {d.logp for d in decisions} == {0.0}, \
        "opponent's logp leaked into θ's replay buffer"


def test_collect_batch_without_opp_actor_records_both_seats():
    """The compatibility contract: ``act_fn_opp=None`` is the historical
    path — one actor answers both seats and both are recorded."""
    seats = [0, 1, 0]
    env = _FakePolledEnv(_alternating_polls(seats))
    done = env.collect_batch(_tagged_act_fn(0, []), n_decisions=99)

    assert len(done) == 1
    assert len(done[0].decisions) == len(seats)
    assert [d.your_index for d in done[0].decisions] == seats


def test_seat_routing_rejects_a_short_actor_reply():
    """An actor returning fewer replies than requests would shift every later
    pick by one and desynchronise the whole batch."""
    def short_act(requests):
        return []

    env = _FakePolledEnv([_mixed_poll([0, 1]), []])
    with pytest.raises(RuntimeError, match="unanswered"):
        env.collect_batch(
            _tagged_act_fn(0, []), n_decisions=1, act_fn_opp=short_act,
        )


# ── The opponent-expert registry ─────────────────────────────────────────


def test_parse_opp_experts_maps_ids_to_paths(tmp_path):
    from ptcg_rl.mcts_train import parse_opp_experts

    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b""); b.write_bytes(b"")
    assert parse_opp_experts([f"17={a}", f"25={b}"]) == {17: str(a), 25: str(b)}
    assert parse_opp_experts(None) == {}


@pytest.mark.parametrize("spec", ["17", "=x.pt", "17=", "a17=x.pt"])
def test_parse_opp_experts_rejects_malformed(spec, tmp_path):
    """A silently dropped expert is invisible — that deck just falls back to
    the generalist pilot and the branch runs to completion reporting a win
    rate against an opponent field the caller never actually configured."""
    from ptcg_rl.mcts_train import parse_opp_experts

    spec = spec.replace("x.pt", str(tmp_path / "x.pt"))
    (tmp_path / "x.pt").write_bytes(b"")
    with pytest.raises(SystemExit):
        parse_opp_experts([spec])


def test_parse_opp_experts_rejects_missing_file(tmp_path):
    from ptcg_rl.mcts_train import parse_opp_experts

    with pytest.raises(SystemExit, match="does not exist"):
        parse_opp_experts([f"17={tmp_path / 'nope.pt'}"])


def _registry(default, experts):
    from ptcg_rl.mcts_train import OpponentExperts
    import torch as _torch

    return OpponentExperts(
        default_ckpt=default, experts=experts, data_dir=Path("data"),
        device=_torch.device("cpu"), vocab={},
    )


def test_registry_prefers_the_deck_owner_then_the_default():
    reg = _registry("generalist.pt", {17: "a17.pt", 25: "a25.pt"})
    assert reg.ckpt_for(17) == "a17.pt"
    assert reg.ckpt_for(25) == "a25.pt"
    # Every other archetype in the 179-deck pool falls back to the generalist.
    assert reg.ckpt_for(3) == "generalist.pt"
    assert reg.enabled is True


def test_registry_disabled_without_any_pilot():
    """No flags → ``act_fn_opp=None`` → θ answers both seats, exactly as
    before this feature existed."""
    reg = _registry(None, {})
    assert reg.enabled is False
    assert reg.ckpt_for(17) is None
    assert reg.actor_for(17, seed=0) is None


# ── Rival share: how often a branch actually meets another branch ────────


def _share_decks(n=201):
    return [{"id": i, "deck": [i], "frequency": 1.0 / n, "count": 1,
             "name": f"a{i}"} for i in range(n)]


def _run_sampling(share, expert_ids, n_games=2000, n_pool=201):
    """Drive only run_self_play_games' opponent-sampling arithmetic.

    Reimplements the loop rather than calling it, because the real function
    needs libcg.  The assertion that this stays faithful is
    ``test_expert_share_flag_reaches_the_sampler`` below plus the shared
    ``_sample_weighted``.
    """
    from ptcg_rl.mcts_train import _sample_weighted

    decks = _share_decks(n_pool)
    rivals = [d for d in decks if d["id"] in expert_ids]
    rng = np.random.default_rng(0)
    faced = 0
    for _ in range(n_games):
        pool = decks
        if rivals and rng.random() < share:
            pool = rivals
        opp = _sample_weighted(pool, rng, {})
        if opp["id"] in expert_ids:
            faced += 1
    return faced / n_games


def test_zero_share_leaves_experts_a_rounding_error():
    """The number that motivated the flag: 4 rivals in a 201-deck pool is 2%
    of games, so 'branch a1 fights the other experts' is ~4 games in 200."""
    faced = _run_sampling(share=0.0, expert_ids={17, 25, 36, 16})
    assert faced < 0.05, f"expected ~2%, got {faced:.3f}"


def test_expert_share_lifts_the_rival_encounter_rate():
    """At share=0.5 roughly half the games must be against a rival expert —
    plus the few the full-pool draw lands on anyway."""
    faced = _run_sampling(share=0.5, expert_ids={17, 25, 36, 16})
    assert 0.48 <= faced <= 0.55, f"share=0.5 produced {faced:.3f}"


def test_share_of_one_still_keeps_every_game_on_a_rival():
    faced = _run_sampling(share=1.0, expert_ids={17, 25, 36, 16})
    assert faced == 1.0


def test_share_is_inert_when_no_expert_is_registered():
    """Round 1, branch 1: nobody else has a champion yet.  The share must not
    divide by zero or silently sample an empty pool."""
    faced = _run_sampling(share=1.0, expert_ids=set())
    assert faced == 0.0


def test_expert_share_flag_reaches_the_sampler():
    args = parse_args(_BASE_ARGV + ["--opp-expert-share", "0.5"])
    assert args.opp_expert_share == 0.5
    assert parse_args(_BASE_ARGV).opp_expert_share == 0.0


def test_is_expert_is_not_has_a_pilot():
    """``ckpt_for`` answers for every archetype once a default is set, so
    conflating the two would report a 100% rival share against a field that is
    almost entirely generalist."""
    reg = _registry("generalist.pt", {17: "a17.pt"})
    assert reg.is_expert(17) is True
    assert reg.is_expert(3) is False
    assert reg.ckpt_for(3) == "generalist.pt", "default still covers deck 3"


# ── ELO keys are stems, not filenames ────────────────────────────────────


def test_find_best_champion_matches_the_elo_key_format(tmp_path):
    """``EloTracker`` is keyed by bare model names (``elo_name`` registers
    ``ckpt-mcts-champion-001400``).  Looking up ``stem + suffix`` puts ``.pt``
    back on, matches nothing, and every champion falls to the default of 0 —
    a flat field where ``glob`` order decides the winner.  Observed: a run with
    a 1553-rated champion resumed from a 1522-rated one, logging "ELO: 0".
    """
    from ptcg_rl.mcts_train import EloTracker, _find_best_champion, elo_key

    ratings = {"ckpt-mcts-champion-000400": 1522.0,
               "ckpt-mcts-champion-001400": 1553.0,
               "ckpt-mcts-champion-001600": 1540.0}
    ckpts = []
    for name in ratings:
        p = tmp_path / f"{name}.pt"
        p.write_bytes(b"")
        ckpts.append(p)
    assert len(ckpts) == 3, "fixture must supply champions to rank"

    elo = EloTracker()
    elo.ratings.update(ratings)

    for p in ckpts:
        assert elo_key(p) in elo.ratings, f"{p.name} has no ELO key"

    best = _find_best_champion(ckpts, elo)
    assert best is not None
    assert best.name == "ckpt-mcts-champion-001400.pt", (
        f"picked {best.name}, not the 1553-rated champion"
    )


def test_find_best_champion_is_glob_order_independent(tmp_path):
    """The bug was invisible because it degraded to 'first file wins'.  Ranking
    must not depend on the order the paths arrive in."""
    from ptcg_rl.mcts_train import EloTracker, _find_best_champion

    elo = EloTracker()
    elo.ratings.update({"a": 10.0, "b": 99.0, "c": 50.0})
    paths = []
    for n in ("a", "b", "c"):
        p = tmp_path / f"{n}.pt"
        p.write_bytes(b"")
        paths.append(p)

    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2], [2, 0, 1]):
        best = _find_best_champion([paths[i] for i in order], elo)
        assert best.stem == "b", f"order {order} picked {best.stem}"


def test_find_best_champion_skips_missing_files(tmp_path):
    from ptcg_rl.mcts_train import EloTracker, _find_best_champion

    elo = EloTracker()
    elo.ratings.update({"gone": 9999.0, "here": 1.0})
    here = tmp_path / "here.pt"
    here.write_bytes(b"")
    best = _find_best_champion([tmp_path / "gone.pt", here], elo)
    assert best == here, "a top-rated rating for a deleted file must not win"


def test_champion_discovery_excludes_il_dir_and_ungated_last(tmp_path):
    """The candidate pool is gated champions from out_dir only.

    Folding in the --il-ckpt directory entered every IL checkpoint into the
    league twice, and made ``ckpt-best``'s *fixed* 1500 a resume candidate — a
    branch whose champions sat below it would rewind θ to the IL model while
    logging "Auto-resuming from best champion".  ``ckpt-mcts-last.pt`` is
    written gate or no gate, so it is not a result either.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for n in ("ckpt-mcts-champion-000200.pt", "ckpt-mcts-champion-000400.pt",
              "ckpt-mcts-last.pt"):
        (out_dir / n).write_bytes(b"")
    il_dir = tmp_path / "il"
    il_dir.mkdir()
    for n in ("ckpt-best.pt", "ckpt-last.pt", "ckpt-step-0001468.pt"):
        (il_dir / n).write_bytes(b"")

    found = sorted(p.name for p in out_dir.glob("ckpt-mcts-champion-*.pt"))
    assert found == ["ckpt-mcts-champion-000200.pt",
                     "ckpt-mcts-champion-000400.pt"], found
    assert "ckpt-mcts-last.pt" not in found
    assert not any(p.parent == il_dir for p in
                   out_dir.glob("ckpt-mcts-champion-*.pt"))


def test_fixed_il_anchor_cannot_win_auto_resume(tmp_path):
    """Regression for the coupled failure: with ELO keys fixed, ``ckpt-best``
    at a frozen 1500 outranks a struggling branch's champions."""
    from ptcg_rl.mcts_train import EloTracker, _find_best_champion

    elo = EloTracker()
    elo.add_model("ckpt-best", fixed=True)          # 1500, frozen
    elo.ratings["ckpt-mcts-champion-000200"] = 1440.0
    elo.ratings["ckpt-mcts-champion-000400"] = 1465.0

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for n in ("ckpt-mcts-champion-000200.pt", "ckpt-mcts-champion-000400.pt"):
        (out_dir / n).write_bytes(b"")
    (tmp_path / "ckpt-best.pt").write_bytes(b"")

    pool = sorted(out_dir.glob("ckpt-mcts-champion-*.pt"))
    assert len(pool) == 2, "fixture must supply champions below 1500"
    best = _find_best_champion(pool, elo)
    assert best.stem == "ckpt-mcts-champion-000400", (
        "resume must stay on the branch's own best champion"
    )


# ── Search must evaluate leaves with card identity ───────────────────────


# The premise — that a missing engine table zeroes card identity — is pinned in
# test_featurizer.py, not here.  Asserting it needs the real engine tables, and
# importing `cg` calls GameInitialize at import time; this module has already
# spent that call via the forest/RustVecEnv tests above, and a second one is a
# C++ abort no Python handler can catch.


def test_batch_evaluate_leaves_forwards_the_engine_tables():
    """MCTS leaf evaluation must featurize with card identity.

    Without it every prior and value in the search — and therefore the whole
    ``mcts_pi`` distillation target — is computed on a board where no card has
    an identity, while the rollout actor stays sighted.  Mutation: drop the two
    kwargs from the ``featurize`` call in ``batch_evaluate_leaves`` and this
    goes red.
    """
    import torch
    from ptcg_rl.search import batch_evaluate_leaves

    seen: list[dict] = []

    class _SpyPolicy(_StubPolicy):
        pass

    import ptcg_il.featurizer as fz
    real = fz.featurize

    def spy(obs, vocab, action=None, **kw):
        seen.append(kw)
        return real(obs, vocab, action, **kw)

    import ptcg_rl.search as search_mod
    orig = search_mod.featurize if hasattr(search_mod, "featurize") else None
    fz.featurize = spy
    try:
        leaves = [{
            "tree_id": 0, "obs_json": _leaf_obs(3, 1),
            "player_role": 0, "is_terminal": False, "n_options": 3,
        }]
        batch_evaluate_leaves(
            leaves, _SpyPolicy(), {}, torch.device("cpu"), bf16=False,
            engine_card_features={7: [1.0]}, engine_attack_features={9: [1.0]},
        )
    finally:
        fz.featurize = real
        if orig is not None:
            search_mod.featurize = orig

    assert seen, "featurize was never called — fixture examined nothing"
    assert seen[0].get("engine_card_features") == {7: [1.0]}, (
        f"engine_card_features did not reach featurize: {seen[0]}"
    )
    assert seen[0].get("engine_attack_features") == {9: [1.0]}, (
        f"engine_attack_features did not reach featurize: {seen[0]}"
    )


# ── The opponent archetype must survive poll → Trajectory ────────────────


def _poll_with_opp(seats, opp_id, turn=1):
    return [
        {"battle_idx": 0, "obs_json": _obs_seat(turn, s), "sbi": "",
         "select_player": s, "opp_id": opp_id}
        for s in seats
    ]


class _OppDrainEnv(_FakePolledEnv):
    """Fake whose drain() reports an opponent archetype, as Rust now does."""

    def __init__(self, polls, opp_id):
        super().__init__(polls)
        self._opp_id = opp_id

    def drain(self):
        trajs = super().drain()
        for t in trajs:
            t.opp_id = self._opp_id
        return trajs


def test_trajectory_carries_the_opponent_archetype():
    """``collect_batch`` builds the pending Trajectory itself, so anything
    ``drain()`` knows must be copied onto it explicitly.

    Dropping ``opp_id`` is silent and expensive: it stays at the -1 default, the
    caller's ``deck_by_id.get(-1, [])`` then hands MCTS an *empty* opponent
    deck, and every per-archetype statistic collapses onto one bogus id — which
    is exactly how it reached a benchmark reporting ``expert 0%`` while the
    sampler was correctly drawing rival decks.  Mutation: delete the
    ``pt.opp_id = t.opp_id`` copy and this goes red.
    """
    polls = [_poll_with_opp([0], 17), _poll_with_opp([0], 17, turn=2), []]
    env = _OppDrainEnv(polls, opp_id=17)
    done = env.collect_batch(_tagged_act_fn(0, []), n_decisions=99)

    assert len(done) == 1, "fixture examined nothing"
    assert done[0].opp_id == 17, (
        f"opponent archetype lost: got {done[0].opp_id}"
    )
    assert [d.opp_id for d in done[0].decisions] == [17, 17], (
        "per-decision opp_id must match the battle's opponent"
    )


def test_decision_opp_id_defaults_when_env_reports_none():
    """An env that reports no archetype must leave the sentinel, not 0 —
    archetype 0 is a real cluster id."""
    env = _FakePolledEnv([_mixed_poll([0]), []])
    done = env.collect_batch(_tagged_act_fn(0, []), n_decisions=99)
    assert done and done[0].decisions, "fixture examined nothing"
    assert done[0].decisions[0].opp_id == -1


# ── grad_clip must agree across every entry point ────────────────────────

_REPO = Path(__file__).resolve().parents[2]


def _shell_grad_clips() -> dict[str, float]:
    """Every grad-clip value the shipped scripts pass, by file."""
    import re

    pats = [
        ("scripts/run.sh", r"--grad-clip\s+([\d.]+)"),
        ("scripts/run_pipeline.sh", r"^MCTS_GRAD_CLIP=([\d.]+)"),
        ("scripts/run_mcts_train.sh", r"^GRAD_CLIP=([\d.]+)"),
        ("scripts/run_all_specialists.sh",
         r'^MCTS_GRAD_CLIP="\$\{MCTS_GRAD_CLIP:-([\d.]+)\}"'),
    ]
    found: dict[str, float] = {}
    for rel, pat in pats:
        path = _REPO / rel
        if not path.exists():
            continue
        m = re.search(pat, path.read_text(), re.M)
        assert m, f"{rel}: no grad-clip setting matched {pat!r}"
        found[rel] = float(m.group(1))
    return found


def test_every_entry_point_ships_the_same_grad_clip():
    """The CLI default and all four scripts must agree.

    They drifted once already — the scripts hardcoded 0.5 while the CLI default
    was 1.0 — so which value you got depended on how you started the run, and
    neither number was visible from the other.  0.5 in particular bound on
    100% of steps at 10-70x, which inverts loss scale rather than capping it.
    """
    cli = parse_args(_BASE_ARGV).grad_clip
    shell = _shell_grad_clips()
    assert shell, "no scripts examined"
    mismatched = {k: v for k, v in shell.items() if v != cli}
    assert not mismatched, (
        f"grad_clip is {cli} in the CLI but {mismatched} in these scripts"
    )


def test_grad_clip_is_a_spike_guard_not_a_normaliser():
    """Pinned against the measured distribution: raw norms on a real run were
    4.87-34.93 (median 12.68).  A clip at or below the median binds on most
    steps and rescales each by a different factor, which reverses the ordering
    it is supposed to preserve."""
    observed_median = 12.68
    assert parse_args(_BASE_ARGV).grad_clip > observed_median, (
        "grad_clip must sit above the typical gradient norm, or it normalises "
        "every step instead of catching spikes"
    )


# ── Adaptive KL anchor: the β controller and the top-ELO reference ───────
#
# A constant β cannot hold a trust region.  k3(d) = e^-d - 1 + d has
# derivative 1 - e^-d, which saturates at 1, so the anchor's restoring pull is
# capped at β while the CE/BC term is unbounded: once the policy term wins,
# KL grows with nothing to stop it.  Measured on five real runs, IL outranked
# every MCTS champion in four of them.


def test_beta_controller_flags_carry_the_specced_defaults():
    """κ matches RL_SPEC §9.3 (and ppo.py's R2 anchor) so the three stages
    report KL on one scale."""
    args = parse_args(_BASE_ARGV)
    assert args.kappa == 0.02
    assert args.beta_lr == 0.02
    assert args.beta_min <= args.beta <= args.beta_max


def test_the_cli_namespace_drives_ppo_update_beta_unchanged():
    """The controller is ppo.update_beta, not a second copy of it.  It reads
    κ/η_β/bounds off its cfg, so the MCTS Namespace must supply all four."""
    from ptcg_rl.ppo import update_beta

    args = parse_args(_BASE_ARGV)
    over = update_beta(args.beta, args.kappa * 5.0, args)
    under = update_beta(args.beta, args.kappa * 0.1, args)
    assert over > args.beta, "β must rise while KL sits above κ"
    assert under < args.beta, "β must fall back once KL is under κ"
    assert update_beta(args.beta_max, args.kappa * 1e3, args) <= args.beta_max
    assert update_beta(args.beta_min, 0.0, args) >= args.beta_min


def test_beta_converges_on_kappa_rather_than_saturating():
    """Dual ascent, not a ratchet: a KL that comes back under κ must give the
    β it earned back, or the anchor freezes the policy at beta_max."""
    from ptcg_rl.ppo import update_beta

    args = parse_args(_BASE_ARGV)
    beta = args.beta
    for _ in range(200):                      # one iteration's worth of steps
        beta = update_beta(beta, args.kappa * 4.0, args)
    climbed = beta
    assert climbed > args.beta * 5, climbed
    for _ in range(200):
        beta = update_beta(beta, args.kappa * 0.25, args)
    assert beta < climbed, "β never comes back down"


def test_train_step_takes_beta_as_an_argument_not_a_config_constant(train_cfg):
    """β changes every step now, so it cannot be read off the frozen config."""
    import torch
    from ptcg_rl.mcts_train import train_step

    policy = _tiny_policy(0)
    frozen = _tiny_policy(1)
    for p in frozen.parameters():
        p.requires_grad_(False)
    batch = _tiny_batch()
    device = torch.device("cpu")
    cfg = type(train_cfg)(c_pi=0.0, c_value=0.0, beta=0.0, grad_clip=1e9)
    opt = torch.optim.SGD(policy.parameters(), lr=0.0)

    off = train_step(policy, opt, batch, cfg, device, frozen_il=frozen,
                     beta=0.0)
    on = train_step(policy, opt, batch, cfg, device, frozen_il=frozen,
                    beta=4.0)
    assert off["grad_norm"] == pytest.approx(0.0, abs=1e-9), (
        "β=0 must switch the anchor off entirely")
    assert on["grad_norm"] > 0.0, (
        "the passed β was ignored — config.beta won instead")
    assert on["beta"] == 4.0, "train_step must report the β it actually used"


def test_train_step_scales_the_anchor_gradient_with_beta(train_cfg):
    import torch
    from ptcg_rl.mcts_train import train_step

    frozen = _tiny_policy(1)
    for p in frozen.parameters():
        p.requires_grad_(False)
    cfg = type(train_cfg)(c_pi=0.0, c_value=0.0, beta=0.0, grad_clip=1e9)
    grads = []
    for beta in (1.0, 2.0):
        policy = _tiny_policy(0)          # same weights both times
        opt = torch.optim.SGD(policy.parameters(), lr=0.0)
        grads.append(train_step(policy, opt, _tiny_batch(), cfg,
                                torch.device("cpu"), frozen_il=frozen,
                                beta=beta)["grad_norm"])
    # Both grads are 0 if β is ignored, and 0 == 2*0 — the proportionality
    # check alone passes vacuously on exactly the bug it is guarding.
    assert grads[0] > 0.0, "the anchor produced no gradient at all"
    assert grads[1] == pytest.approx(2.0 * grads[0], rel=1e-4), grads


def test_live_beta_not_the_flag_reaches_wandb():
    """`"beta": args.beta` logged a flat line by construction — the whole
    point of the controller is invisible if the series is the CLI default."""
    import inspect
    from ptcg_rl import mcts_train

    src = inspect.getsource(mcts_train.main)
    assert '"beta": args.beta' not in src, (
        "the β series is still pinned to the CLI flag")
    assert "update_beta" in src, "nothing in main ever moves β"


# ── The anchor reference follows the ELO leaderboard ─────────────────────


def _elo_with(ratings: dict) -> object:
    from ptcg_rl.mcts_train import EloTracker
    elo = EloTracker()
    elo.ratings.update(ratings)
    return elo


def test_anchor_candidates_map_every_leaderboard_name_to_a_file(tmp_path):
    from ptcg_rl.mcts_train import _anchor_candidates, elo_key

    il = tmp_path / "ckpt-best.pt"
    step = tmp_path / "ckpt-step-0002000.pt"
    champ = tmp_path / "ckpt-mcts-champion-000200.pt"
    for p in (il, step, champ):
        p.write_bytes(b"x")

    cands = _anchor_candidates(str(il), [("ckpt-step-0002000", str(step))],
                               [champ])
    assert cands["IL_baseline"] == str(il)
    assert cands["ckpt-step-0002000"] == str(step)
    assert cands[elo_key(champ)] == str(champ)


def test_anchor_follows_the_top_elo_model():
    from ptcg_rl.mcts_train import _resolve_anchor

    elo = _elo_with({"IL_baseline": 1500.0, "ckpt-step-0002000": 1589.3,
                     "ckpt-mcts-champion-001200": 1525.9})
    name, path = _resolve_anchor(elo, {
        "IL_baseline": "il.pt",
        "ckpt-step-0002000": "step.pt",
        "ckpt-mcts-champion-001200": "champ.pt",
    })
    assert (name, path) == ("ckpt-step-0002000", "step.pt")


def test_anchor_switches_to_a_champion_once_it_overtakes_il():
    """The reason for making the reference movable at all."""
    from ptcg_rl.mcts_train import _resolve_anchor

    cands = {"IL_baseline": "il.pt", "ckpt-mcts-champion-000800": "champ.pt"}
    before = _resolve_anchor(_elo_with(
        {"IL_baseline": 1580.0, "ckpt-mcts-champion-000800": 1520.0}), cands)
    after = _resolve_anchor(_elo_with(
        {"IL_baseline": 1580.0, "ckpt-mcts-champion-000800": 1611.0}), cands)
    assert before[0] == "IL_baseline"
    assert after[0] == "ckpt-mcts-champion-000800"


def test_an_unrated_model_never_outranks_a_rated_one():
    """A missing rating is a dict default, not an error — the bug that made
    `_find_best_champion` return glob order.  It must not resurface here."""
    from ptcg_rl.mcts_train import _resolve_anchor

    name, _ = _resolve_anchor(_elo_with({"IL_baseline": 1400.0}),
                              {"IL_baseline": "il.pt", "unrated": "u.pt"})
    assert name == "IL_baseline"


def test_anchor_ignores_a_leaderboard_name_with_no_checkpoint():
    from ptcg_rl.mcts_train import _resolve_anchor

    elo = _elo_with({"ghost": 9999.0, "IL_baseline": 1500.0})
    assert _resolve_anchor(elo, {"IL_baseline": "il.pt"})[0] == "IL_baseline"


def test_anchor_resolution_without_candidates_is_none_not_a_crash():
    from ptcg_rl.mcts_train import _resolve_anchor

    assert _resolve_anchor(_elo_with({"IL_baseline": 1500.0}), {}) is None


def test_the_il_baseline_opponent_stays_the_il_checkpoint():
    """The anchor moves; the *gate* must not.  Repointing `frozen_il` itself
    would enlist a champion under the name IL_baseline and quietly turn the
    regression check into a self-comparison."""
    import inspect
    from ptcg_rl import mcts_train

    src = inspect.getsource(mcts_train.main)
    assert '_enlist("IL_baseline", frozen_il, _deck_record_of(args.il_ckpt))' \
        in src, "the IL_baseline opponent is no longer the --il-ckpt model"
    assert "anchor_policy" in src, "the anchor has no variable of its own"


def test_the_moving_anchor_is_what_the_training_step_anchors_to():
    """Resolving an anchor and then training against the old one is the
    codebase's recurring failure shape: a number gets logged while nothing is
    restrained.  Mutating this line back to `frozen_il` passed every other
    test in this section."""
    import inspect
    from ptcg_rl import mcts_train

    src = inspect.getsource(mcts_train.main)
    assert "frozen_il=anchor_policy" in src, (
        "the training step is anchored to the fixed IL model again — the "
        "top-ELO reference is resolved but unused")


def test_the_anchor_is_re_resolved_after_every_eval():
    """Resolved once at startup, the reference is frozen for the whole run and
    can never follow a champion that overtakes IL."""
    import inspect
    from ptcg_rl import mcts_train

    src = inspect.getsource(mcts_train.main)
    assert src.count("_resolve_anchor_policy(") >= 3, (
        "the anchor is resolved fewer times than definition + startup + "
        "post-eval refresh — it cannot be following the leaderboard")
