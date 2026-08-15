"""Tests for ptcg_il.shard_writer — Phase 3 shard creation and meta.parquet builder.

Uses small synthetic episode dicts.  No network, no engine.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptcg_il.shard_writer import (
    META_COLUMNS,
    SAMPLES_PER_SHARD,
    _active_decisions,
    _is_kept_game,
    _split_of,
    build_shards,
)
from ptcg_mine.archetype import Archetype, assign_archetype, cluster_decks, canon


# ============================================================
# Test helpers
# ============================================================


def _make_deck(card_id: int, n: int = 60) -> list[int]:
    """All-same-card deck of length *n*."""
    return [card_id] * n


def _make_minimal_obs(
    your_index: int = 0,
    sel_type: int = 0,
    sel_ctx: int = 0,
    n_options: int = 2,
) -> dict:
    """A minimal-but-valid observation dict that featurize() can process."""
    base_player = {
        "active": [
            {
                "id": 1,
                "hp": 100,
                "maxHp": 100,
                "energies": [],
                "energyCards": [],
                "tools": [],
                "preEvolution": [],
                "appearThisTurn": False,
            }
        ],
        "bench": [],
        "hand": [{"id": 2}],
        "deckCount": 49,
        "handCount": 1,
        "benchMax": 5,
        "prize": [None, None, None, None, None, None],
        "discard": [],
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }

    options = []
    for i in range(n_options):
        options.append(
            {"type": 7, "index": i, "number": 0, "count": 0, "cardId": 2}
        )

    return {
        "select": {
            "type": sel_type,
            "context": sel_ctx,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
            "remainEnergyCost": 0,
            "remainDamageCounter": 0,
        },
        "current": {
            "yourIndex": your_index,
            "players": [dict(base_player), dict(base_player)],
            "turn": 1,
            "turnActionCount": 0,
            "firstPlayer": 0,
            "stadium": [],
        },
        "logs": [],
    }


def _make_synthetic_episode(
    episode_id: str = "test_001",
    teams: tuple[str, str] = ("expert_a", "expert_b"),
    rewards: tuple[int, int] = (-1, 1),
    deck_p0: list[int] | None = None,
    deck_p1: list[int] | None = None,
    n_decisions: int = 3,
) -> dict:
    """Create a small synthetic episode with deck steps and interleaved
    in-game decisions.

    After the deck steps (0-1), alternate P0-ACTIVE and P1-ACTIVE steps.
    With *n_decisions*=3, P0 gets 3 ACTIVE steps and P1 gets 2 (the final
    one is dropped by the off-by-one rule, matching the spec).
    """
    if deck_p0 is None:
        deck_p0 = [1] * 60
    if deck_p1 is None:
        deck_p1 = [2] * 60

    empty_current = _make_minimal_obs()["current"]

    steps = [
        # Step 0: deck-selection prompts (both ACTIVE, select=None)
        [
            {
                "observation": {"select": None, "current": empty_current},
                "action": [],
                "status": "ACTIVE",
                "reward": 0,
                "info": {},
            },
            {
                "observation": {"select": None, "current": empty_current},
                "action": [],
                "status": "ACTIVE",
                "reward": 0,
                "info": {},
            },
        ],
        # Step 1: deck submissions
        [
            {
                "observation": {"select": None, "current": empty_current},
                "action": deck_p0,
                "status": "ACTIVE",
                "reward": 0,
                "info": {},
            },
            {
                "observation": {"select": None, "current": empty_current},
                "action": deck_p1,
                "status": "ACTIVE",
                "reward": 0,
                "info": {},
            },
        ],
    ]

    # Game decisions: alternate P0, P1, P0, P1, ...
    for i in range(n_decisions * 2):
        p = i % 2
        obs = _make_minimal_obs(your_index=p)
        dummy_obs = {"select": None, "current": empty_current}
        step: list[dict] = [None, None]  # type: ignore[list-item]
        step[p] = {
            "observation": obs,
            "action": [0],
            "status": "ACTIVE",
            "reward": 0,
            "info": {},
        }
        step[1 - p] = {
            "observation": dummy_obs,
            "action": [],
            "status": "INACTIVE",
            "reward": 0,
            "info": {},
        }
        steps.append(step)

    return {
        "info": {"TeamNames": list(teams)},
        "rewards": list(rewards),
        "statuses": ["DONE", "DONE"],
        "steps": steps,
    }


def _build_test_vocab(episodes: list[dict]) -> dict:
    """Build a minimal vocab covering all deck card ids in *episodes*."""
    from ptcg_mine import vocab as vocab_mod

    return vocab_mod.build_vocab(episodes)


def _build_test_archetypes() -> tuple[list[Archetype], list[int], list[int]]:
    """Build archetypes from known 60-card decks.

    Returns (archetypes, self_ids, opp_ids).
    """
    # Deck A = all 1s, Deck B = all 2s, Deck C = all 3s
    freq: dict[tuple[int, ...], int] = {
        canon(_make_deck(1)): 10,
        canon(_make_deck(2)): 7,
        canon(_make_deck(3)): 4,
    }
    archetypes = cluster_decks(freq)
    self_ids = sorted(a.id for a in archetypes)  # all are both self and opp for convenience
    opp_ids = sorted(a.id for a in archetypes)
    return archetypes, self_ids, opp_ids


def _build_deck_with_overlap(base_id: int, swap_ids: list[int]) -> list[int]:
    """Build a 60-card deck that is mostly *base_id*, with a few cards
    swapped to *swap_ids* so it maps to a different archetype cluster."""
    deck = [base_id] * 60
    for i, sid in enumerate(swap_ids):
        deck[i] = sid
    return deck


# ============================================================
# Tests: _split_of
# ============================================================


class TestSplitOf:
    def test_deterministic(self):
        """Same episode_id always produces same split, verified against known hash value."""
        s1 = _split_of("abc123")
        s2 = _split_of("abc123")
        assert s1 == s2
        # Hard-coded expected value ensures cross-run reproducibility
        # (SHA-256, not Python's randomized hash)
        assert _split_of("test_ep_0000") == "train"

    def test_split_ranges(self):
        """Verify that the split distribution roughly follows the 80/10/10 plan."""
        counts: Counter = Counter()
        n = 10000
        for i in range(n):
            counts[_split_of(f"ep_{i}")] += 1
        # Ranges are approximate due to hash distribution
        assert counts["train"] > counts["val"]
        assert counts["train"] > counts["test"]
        # ~80 / ~10 / ~10, allowing generous slack for hash jitter
        assert 0.76 * n < counts["train"] < 0.84 * n
        assert 0.07 * n < counts["val"] < 0.13 * n
        assert 0.07 * n < counts["test"] < 0.13 * n

    def test_valid_splits_only(self):
        """Only returns train/val/test."""
        for i in range(500):
            assert _split_of(f"test_{i}") in ("train", "val", "test")


# ============================================================
# Tests: _is_kept_game
# ============================================================


class TestIsKeptGame:
    def test_expert_self_opp_all_match(self):
        """Kept when expert team, self deck in self_ids, opp deck in opp_ids."""
        ep = _make_synthetic_episode(
            episode_id="k1",
            teams=("expert_a", "random_b"),
            rewards=(1, -1),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
        )
        archetypes, self_ids, opp_ids = _build_test_archetypes()

        assert _is_kept_game(
            ep, 0, {"expert_a"}, set(self_ids), set(opp_ids), archetypes
        )

    def test_non_expert_team(self):
        """Not kept when player's team is not an expert."""
        ep = _make_synthetic_episode(
            teams=("random_a", "random_b"), deck_p0=_make_deck(1), deck_p1=_make_deck(2)
        )
        archetypes, self_ids, opp_ids = _build_test_archetypes()

        assert not _is_kept_game(
            ep, 0, {"expert_x"}, set(self_ids), set(opp_ids), archetypes
        )

    def test_self_deck_not_in_self_ids(self):
        """Not kept when expert's own deck archetype is not in self_ids."""
        ep = _make_synthetic_episode(
            teams=("expert_a", "other"),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
        )
        archetypes, self_ids_all, opp_ids_all = _build_test_archetypes()
        # self_ids without the archetype for deck of 1s
        # Archetype for deck of 1s is whichever got assigned to canon([1]*60)
        arch_1 = None
        for a in archetypes:
            if a.representative == tuple([1] * 60):
                arch_1 = a.id
                break
        assert arch_1 is not None, "Expected archetype for deck of all 1s"
        # Remove self_ids except arch_1
        other_self_ids = [sid for sid in self_ids_all if sid != arch_1]

        assert not _is_kept_game(
            ep, 0, {"expert_a"}, set(other_self_ids), set(opp_ids_all), archetypes
        )

    def test_opp_deck_not_in_opp_ids(self):
        """Not kept when opponent's deck archetype is not in opp_ids."""
        ep = _make_synthetic_episode(
            teams=("expert_a", "other"),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
        )
        archetypes, self_ids_all, opp_ids_all = _build_test_archetypes()
        arch_2 = None
        for a in archetypes:
            if a.representative == tuple([2] * 60):
                arch_2 = a.id
                break
        assert arch_2 is not None, "Expected archetype for deck of all 2s"
        other_opp_ids = [oid for oid in opp_ids_all if oid != arch_2]

        assert not _is_kept_game(
            ep, 0, {"expert_a"}, set(self_ids_all), set(other_opp_ids), archetypes
        )

    def test_both_won_and_lost_kept(self):
        """Both won (reward +1) and lost (reward -1) expert games are kept."""
        ep_won = _make_synthetic_episode(
            teams=("expert_a", "other"),
            rewards=(1, -1),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
        )
        ep_lost = _make_synthetic_episode(
            teams=("expert_a", "other"),
            rewards=(-1, 1),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
        )
        archetypes, self_ids, opp_ids = _build_test_archetypes()

        assert _is_kept_game(
            ep_won, 0, {"expert_a"}, set(self_ids), set(opp_ids), archetypes
        )
        assert _is_kept_game(
            ep_lost, 0, {"expert_a"}, set(self_ids), set(opp_ids), archetypes
        )

    def test_draw_not_distinguished(self):
        """A draw (reward 0) is still kept if archetype conditions pass."""
        ep_draw = _make_synthetic_episode(
            teams=("expert_a", "other"),
            rewards=(0, 0),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
        )
        archetypes, self_ids, opp_ids = _build_test_archetypes()

        assert _is_kept_game(
            ep_draw, 0, {"expert_a"}, set(self_ids), set(opp_ids), archetypes
        )

    def test_experts_none_keeps_a_team_no_filter_would_have_selected(self):
        """``experts=None`` -- the production path -- applies no team filter.

        The top-K filter discarded 92.3% of the real corpus (8027 of 104418
        (episode, player) pairs).  Skill is a per-row weight now, so a
        non-expert team's rows must be *written*, with ``skill_w`` deciding how
        much they count.  If this reverts to filtering, the corpus silently
        shrinks 13x and only the row count in the build log would show it.
        """
        ep = _make_synthetic_episode(
            teams=("nobody_special", "other"),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
        )
        archetypes, self_ids, opp_ids = _build_test_archetypes()

        assert _is_kept_game(ep, 0, None, set(self_ids), set(opp_ids), archetypes)
        # ...and an explicit set still filters, so the old corpus is reproducible.
        assert not _is_kept_game(
            ep, 0, {"expert_a"}, set(self_ids), set(opp_ids), archetypes
        )

    def test_experts_none_still_applies_the_archetype_filters(self):
        """Dropping the team filter must not drop the deck filters with it."""
        ep = _make_synthetic_episode(
            teams=("nobody_special", "other"),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
        )
        archetypes, self_ids, opp_ids = _build_test_archetypes()

        assert not _is_kept_game(ep, 0, None, set(), set(opp_ids), archetypes)
        assert not _is_kept_game(ep, 0, None, set(self_ids), set(), archetypes)

    def test_opp_ids_none_accepts_an_off_list_opponent(self):
        """``opp_ids=None`` keeps the row; ``self_ids`` is still enforced.

        These are deliberately asymmetric.  An off-list *opponent* still trains
        the policy and three of the four belief heads, abstaining only on
        ``bel_arch``; an off-list *own* deck is a different game entirely.
        """
        ep = _make_synthetic_episode(
            teams=("anyone", "other"),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
        )
        archetypes, self_ids, opp_ids = _build_test_archetypes()
        arch_2 = assign_archetype(_make_deck(2), archetypes)
        assert arch_2 is not None, "fixture must produce a real opponent archetype"
        without_opp = set(opp_ids) - {arch_2}

        assert not _is_kept_game(
            ep, 0, None, set(self_ids), without_opp, archetypes
        ), "an explicit set must still filter"
        assert _is_kept_game(ep, 0, None, set(self_ids), None, archetypes)
        # self_ids has no such escape
        assert not _is_kept_game(ep, 0, None, set(), None, archetypes)

    def test_opp_ids_none_accepts_an_opponent_matching_no_cluster_at_all(self):
        """The unclustered case, not just the off-list one.

        ``assign_archetype`` returns None when no representative matches at the
        jaccard threshold.  ``None in opp_ids`` was already False, so this row
        was dropped by the same condition -- but it is the case pass B used to
        drop a *second* time, and reinstating either one silently restores the
        filter.
        """
        ep = _make_synthetic_episode(
            teams=("anyone", "other"),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(999),  # matches no cluster
        )
        archetypes, self_ids, _ = _build_test_archetypes()
        assert assign_archetype(_make_deck(999), archetypes) is None

        assert _is_kept_game(ep, 0, None, set(self_ids), None, archetypes)

    def test_invalid_deck_caught(self):
        """An episode with a malformed deck should not crash _is_kept_game."""
        ep = _make_synthetic_episode(
            teams=("expert_a", "other"),
            deck_p0=[1] * 59,  # short deck
            deck_p1=_make_deck(2),
        )
        archetypes, self_ids, opp_ids = _build_test_archetypes()
        # _is_kept_game catches ValueError from deck_of and returns False
        result = _is_kept_game(
            ep, 0, {"expert_a"}, set(self_ids), set(opp_ids), archetypes
        )
        assert result is False


# ============================================================
# Tests: _active_decisions
# ============================================================


class TestActiveDecisions:
    def test_off_by_one_pairing(self):
        """Observation at steps[i], action from steps[i+1].

        The synthetic episode has: steps 0-1 = deck (both ACTIVE),
        steps 2,4,6 = P0 game decisions, steps 3,5,7 = P1 game decisions.
        P0 is ACTIVE at steps 0,1,2,4,6 (5 total: 2 deck + 3 game).
        """
        ep = _make_synthetic_episode(n_decisions=3)
        decisions = list(_active_decisions(ep, 0))
        # 5 ACTIVE records: steps 0,1 (deck) + 2,4,6 (game)
        assert len(decisions) == 5
        # Game decisions are at steps 2, 4, 6
        game_decisions = [(i, o, a) for i, o, a in decisions if o.get("select") is not None]
        assert len(game_decisions) == 3
        for step_i, obs, action in game_decisions:
            assert step_i in (2, 4, 6), f"Unexpected game step {step_i}"
            assert obs["select"]["type"] == 0  # MAIN

    def test_drops_final_active_decision(self):
        """The last ACTIVE step (step 7 for P1) has no step 8, so dropped."""
        ep = _make_synthetic_episode(n_decisions=3)
        # P1 ACTIVE at steps: 0,1 (deck) + 3,5,7 (game) = 5 total
        # step 7 has no action at step 8 → dropped → 4 yielded
        decisions = list(_active_decisions(ep, 1))
        assert len(decisions) == 4
        # Game decisions only: steps 3 and 5 (step 7 dropped)
        game_decisions = [(i, o, a) for i, o, a in decisions if o.get("select") is not None]
        assert len(game_decisions) == 2
        for step_i, _, _ in game_decisions:
            assert step_i in (3, 5)

    def test_deck_selection_steps_present(self):
        """Deck-selection steps (select=None) are yielded (caller must skip via
        featurizer rejection)."""
        ep = _make_synthetic_episode(n_decisions=3)
        deck_found = any(
            obs.get("select") is None for _, obs, _ in _active_decisions(ep, 0)
        )
        assert deck_found, "Deck-selection steps should appear in iterator"

    def test_inactive_status_skipped(self):
        """Only ACTIVE status records are yielded."""
        ep = _make_synthetic_episode(n_decisions=3)
        for p in (0, 1):
            for step_i, obs, _ in _active_decisions(ep, p):
                rec = ep["steps"][step_i][p]
                assert rec["status"] == "ACTIVE", f"Expected ACTIVE, got {rec['status']}"

    def test_deck_action_length(self):
        """The deck action for step 0 (deck prompt) comes from step 1 → 60 cards."""
        ep = _make_synthetic_episode(n_decisions=3)
        for step_i, obs, action in _active_decisions(ep, 0):
            if obs.get("select") is None and step_i == 0:
                assert len(action) == 60


# ============================================================
# Tests: build_shards (end-to-end)
# ============================================================


class TestBuildShards:
    @staticmethod
    def _setup_test_data(
        tmp_path, n_episodes=4, n_decks=3
    ) -> tuple[list[tuple[str, dict]], dict, list[Archetype], list[int], list[int], set[str]]:
        """Create synthetic episodes, vocab, and archetypes for testing."""
        episodes: list[tuple[str, dict]] = []
        for i in range(n_episodes):
            eid = f"test_ep_{i:04d}"
            # Alternate winners and some expert/non-expert variation
            deck_p0 = _make_deck(1)
            deck_p1 = _make_deck(2)
            ep = _make_synthetic_episode(
                episode_id=eid,
                teams=(f"expert_{i % 3}", "other"),
                rewards=((-1, 1) if i % 2 == 0 else (1, -1)),
                deck_p0=deck_p0,
                deck_p1=deck_p1,
                n_decisions=3,
            )
            episodes.append((eid, ep))

        # Build test vocab from all episodes
        vocab = _build_test_vocab([ep for _, ep in episodes])

        # Build test archetypes
        archetypes, self_ids, opp_ids = _build_test_archetypes()

        # Experts: the first few team names
        experts = {f"expert_{i}" for i in range(3)}

        return episodes, vocab, archetypes, self_ids, opp_ids, experts

    def test_build_shards_creates_files(self, tmp_path):
        """build_shards creates shards/ and meta.parquet in the configured out_dir."""
        from ptcg_mine.config import MineConfig

        episodes, vocab, archetypes, self_ids, opp_ids, experts = self._setup_test_data(tmp_path)

        out_dir = tmp_path / "data"
        out_dir.mkdir(parents=True)

        # Write vocab and archetypes to out_dir so config-based path can load them
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(out_dir / "archetypes.json", "w") as f:
            json.dump(
                {
                    "self_ids": self_ids,
                    "opp_ids": opp_ids,
                    "fixed_deck": [1] * 60,
                    "archetypes": [
                        {
                            "id": a.id,
                            "representative": list(a.representative),
                            "frequency": a.frequency,
                            "n_members": len(getattr(a, "members", [])),
                        }
                        for a in archetypes
                    ],
                },
                f,
            )

        config = MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir)

        summary = build_shards(
            config=config,
            episodes=episodes,
            vocab=vocab,
            archetypes_data=archetypes,
            self_ids=self_ids,
            opp_ids=opp_ids,
            experts=experts,
        )

        assert summary["total_samples"] > 0
        assert summary["n_shards"] > 0
        assert summary["n_kept_games"] > 0

        # Verify files actually exist on disk
        shards_dir = out_dir / "shards"
        assert shards_dir.is_dir(), f"Shards directory not found at {shards_dir}"
        shard_files = sorted(shards_dir.glob("*.npz"))
        assert len(shard_files) > 0, f"No shard files found in {shards_dir}"
        for sf in shard_files:
            assert sf.stat().st_size > 0, f"Shard file {sf.name} is empty"

        meta_path = out_dir / "meta.parquet"
        assert meta_path.exists(), f"meta.parquet not found at {meta_path}"
        assert meta_path.stat().st_size > 0, "meta.parquet is empty"
        meta = pd.read_parquet(meta_path)
        assert len(meta) == summary["total_samples"]

    def test_build_shards_with_temp_outdir(self, tmp_path):
        """build_shards writes to a config-specified out_dir."""
        from ptcg_mine.config import MineConfig

        episodes, vocab, archetypes, self_ids, opp_ids, experts = self._setup_test_data(tmp_path)

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Write vocab + archetypes to out_dir
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(out_dir / "archetypes.json", "w") as f:
            arch_json = {
                "self_ids": self_ids,
                "opp_ids": opp_ids,
                "fixed_deck": [1] * 60,
                "archetypes": [
                    {
                        "id": a.id,
                        "representative": list(a.representative),
                        "frequency": a.frequency,
                        "n_members": len(getattr(a, "members", [])),
                    }
                    for a in archetypes
                ],
            }
            json.dump(arch_json, f)

        config = MineConfig(raw_dir=raw_dir, out_dir=out_dir)

        summary = build_shards(
            config=config,
            episodes=episodes,
            vocab=vocab,
            archetypes_data=archetypes,
            self_ids=self_ids,
            opp_ids=opp_ids,
            experts=experts,
        )

        assert summary["total_samples"] > 0
        shards_dir = out_dir / "shards"
        shard_files = list(shards_dir.glob("*.npz"))
        assert len(shard_files) > 0

        meta_path = out_dir / "meta.parquet"
        assert meta_path.exists()
        meta = pd.read_parquet(meta_path)
        assert set(meta.columns) == set(META_COLUMNS)
        assert len(meta) == summary["total_samples"]

    def test_no_expert_filter_keeps_more_rows_and_weights_them_by_skill(self, tmp_path):
        """The production path (``experts=None``) writes every team, weighted.

        Two assertions that must hold together: the corpus **grows** (teams the
        top-K filter excluded now contribute rows), and ``skill_w`` **varies**
        across those teams (so the extra rows do not count as much as an
        expert's).  Either one alone is satisfiable by a bug -- keeping
        everything at weight 1.0 imitates the median player, and keeping the
        filter while writing a weight column changes nothing.
        """
        from ptcg_mine.config import MineConfig

        episodes, vocab, archetypes, self_ids, opp_ids, experts = self._setup_test_data(tmp_path)

        def _run(out_name, experts_arg):
            out_dir = tmp_path / out_name
            out_dir.mkdir()
            with open(out_dir / "vocab.json", "w") as f:
                json.dump(vocab, f)
            with open(out_dir / "archetypes.json", "w") as f:
                json.dump({
                    "self_ids": self_ids,
                    "opp_ids": opp_ids,
                    "fixed_deck": [1] * 60,
                    "archetypes": [
                        {"id": a.id, "representative": list(a.representative),
                         "frequency": a.frequency, "n_members": len(getattr(a, "members", []))}
                        for a in archetypes
                    ],
                }, f)
            summary = build_shards(
                config=MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir),
                episodes=episodes, vocab=vocab, archetypes_data=archetypes,
                self_ids=self_ids, opp_ids=opp_ids, experts=experts_arg,
            )
            return summary, pd.read_parquet(out_dir / "meta.parquet")

        filtered_summary, filtered_meta = _run("filtered", experts)
        all_summary, all_meta = _run("unfiltered", None)

        assert filtered_summary["total_samples"] > 0, "fixture produced no filtered rows"
        assert all_summary["n_kept_games"] > filtered_summary["n_kept_games"]
        assert set(all_meta["team"]) > set(filtered_meta["team"])

        assert "skill_w" in all_meta.columns
        assert (all_meta["skill_w"] > 0).all(), "a zeroed row is an invisible drop"
        assert all_meta["skill_w"].nunique() > 1, "skill_w must vary between teams"

    def test_off_list_opponents_are_written_with_the_ignore_sentinel(self, tmp_path):
        """Rows the D_opp filter used to drop are kept, and marked -1.

        Both halves matter.  If the rows are missing, dropping the filter did
        nothing.  If they are present but carry a real class index, the belief
        head is being trained on a label that means "archetype 0" when the
        truth is "not in the list" — which no loss curve would reveal.
        """
        from ptcg_mine.config import MineConfig

        episodes, vocab, archetypes, self_ids, opp_ids, _ = self._setup_test_data(tmp_path)

        def _run(out_name, filter_opp, opp_id_list):
            out_dir = tmp_path / out_name
            out_dir.mkdir()
            with open(out_dir / "vocab.json", "w") as f:
                json.dump(vocab, f)
            with open(out_dir / "archetypes.json", "w") as f:
                json.dump({
                    "self_ids": self_ids, "opp_ids": opp_id_list, "fixed_deck": [1] * 60,
                    "archetypes": [
                        {"id": a.id, "representative": list(a.representative),
                         "frequency": a.frequency, "n_members": len(getattr(a, "members", []))}
                        for a in archetypes
                    ],
                }, f)
            summary = build_shards(
                config=MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir),
                episodes=episodes, vocab=vocab, archetypes_data=archetypes,
                self_ids=self_ids, opp_ids=opp_id_list, experts=None,
                filter_opp=filter_opp,
            )
            return summary, pd.read_parquet(out_dir / "meta.parquet")

        # Shrink D_opp to a single archetype so some opponents fall off the list.
        narrow = opp_ids[:1]
        filtered, filtered_meta = _run("filtered", True, narrow)
        kept, kept_meta = _run("kept", False, narrow)

        assert filtered["total_samples"] > 0, "fixture produced no filtered rows"
        assert kept["n_kept_games"] > filtered["n_kept_games"]

        assert set(filtered_meta["archetype_opp"]) <= set(narrow), (
            "with the filter on, every opponent must be in D_opp"
        )
        off_list = kept_meta[~kept_meta["archetype_opp"].isin(narrow)]
        assert len(off_list) > 0, "no off-list rows were recovered"

        # Meta correctly records the real global cluster id (or -1) for every
        # opponent — the old bel_arch label is gone, but the meta column
        # remains for slicing eval and downstream analysis.
        assert len(off_list) > 0

    def test_an_unclustered_opponent_survives_both_passes(self, tmp_path):
        """The opponent deck that matches *no* cluster, end to end.

        Distinct from the off-list case and easy to lose: pass A decides to keep
        the row, then pass B recomputes ``opp_arch``, gets None, and used to
        ``continue`` on it -- silently reinstating the filter one pass after it
        was lifted, for exactly the most unusual decks.  The fixture's opponents
        must genuinely fail to cluster, or this passes vacuously.
        """
        from ptcg_mine.config import MineConfig

        _, vocab_src, archetypes, self_ids, opp_ids, _ = self._setup_test_data(tmp_path)
        assert assign_archetype(_make_deck(999), archetypes) is None

        episodes = []
        for i in range(4):
            eid = f"odd_ep_{i:04d}"
            episodes.append((eid, _make_synthetic_episode(
                episode_id=eid,
                teams=("anyone", "other"),
                rewards=((-1, 1) if i % 2 == 0 else (1, -1)),
                deck_p0=_make_deck(1),
                deck_p1=_make_deck(999),
                n_decisions=3,
            )))
        vocab = _build_test_vocab([ep for _, ep in episodes])

        out_dir = tmp_path / "unclustered"
        out_dir.mkdir()
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(out_dir / "archetypes.json", "w") as f:
            json.dump({
                "self_ids": self_ids, "opp_ids": opp_ids, "fixed_deck": [1] * 60,
                "archetypes": [
                    {"id": a.id, "representative": list(a.representative),
                     "frequency": a.frequency, "n_members": len(getattr(a, "members", []))}
                    for a in archetypes
                ],
            }, f)

        summary = build_shards(
            config=MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir),
            episodes=episodes, vocab=vocab, archetypes_data=archetypes,
            self_ids=self_ids, opp_ids=opp_ids, experts=None,
        )

        assert summary["total_samples"] > 0, "pass B dropped every unclustered row"
        meta = pd.read_parquet(out_dir / "meta.parquet")
        assert (meta["archetype_opp"] == -1).all()

        # The important invariant is that unclustered rows exist at all —
        # pass B used to silently reinstate the filter.  Meta correctness
        # (archetype_opp == -1) is already checked above.

    def test_skill_w_matches_the_team_leaderboard(self, tmp_path):
        """``skill_w`` is this corpus's own leaderboard, not a constant.

        Recomputed independently here: if ``build_shards`` ever writes a
        placeholder or reuses a stale table, every downstream weight is wrong
        and nothing else in the pipeline would notice.
        """
        from ptcg_mine.config import MineConfig
        from ptcg_mine.stats import team_leaderboard, team_skill_weights

        episodes, vocab, archetypes, self_ids, opp_ids, _ = self._setup_test_data(tmp_path)

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(out_dir / "archetypes.json", "w") as f:
            json.dump({
                "self_ids": self_ids, "opp_ids": opp_ids, "fixed_deck": [1] * 60,
                "archetypes": [
                    {"id": a.id, "representative": list(a.representative),
                     "frequency": a.frequency, "n_members": len(getattr(a, "members", []))}
                    for a in archetypes
                ],
            }, f)

        build_shards(
            config=MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir),
            episodes=episodes, vocab=vocab, archetypes_data=archetypes,
            self_ids=self_ids, opp_ids=opp_ids, experts=None,
        )
        meta = pd.read_parquet(out_dir / "meta.parquet")

        expected = team_skill_weights(team_leaderboard([ep for _, ep in episodes]))
        checked = 0
        for team, group in meta.groupby("team"):
            assert group["skill_w"].iloc[0] == pytest.approx(expected[team])
            checked += 1
        assert checked > 1, "fixture must contain more than one team to be meaningful"

    def test_meta_columns_and_types(self, tmp_path):
        """meta.parquet has correct columns and reasonable values."""
        from ptcg_mine.config import MineConfig

        episodes, vocab, archetypes, self_ids, opp_ids, experts = self._setup_test_data(tmp_path)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(out_dir / "archetypes.json", "w") as f:
            arch_json = {
                "self_ids": self_ids,
                "opp_ids": opp_ids,
                "fixed_deck": [1] * 60,
                "archetypes": [
                    {
                        "id": a.id,
                        "representative": list(a.representative),
                        "frequency": a.frequency,
                        "n_members": len(getattr(a, "members", [])),
                    }
                    for a in archetypes
                ],
            }
            json.dump(arch_json, f)

        config = MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir)

        summary = build_shards(
            config=config,
            episodes=episodes,
            vocab=vocab,
            archetypes_data=archetypes,
            self_ids=self_ids,
            opp_ids=opp_ids,
            experts=experts,
        )

        meta = pd.read_parquet(out_dir / "meta.parquet")

        # Column types (pandas 3.x uses StringDtype which is 'str' or 'string')
        assert str(meta["sample_uid"].dtype) in ("object", "string", "str")
        assert str(meta["shard"].dtype) in ("object", "string", "str")
        assert meta["row"].dtype in (np.int64, np.int32)
        assert meta["won"].dtype == bool

        # sample_uid should follow the format episode_player_step
        sample = meta.iloc[0]
        assert "_" in sample["sample_uid"]

        # shard name should match the split
        assert sample["shard"].endswith(".npz")
        split = sample["shard"].split("-")[0]
        assert split in ("train", "val", "test")

        # All rows in same shard should have same shard name and sequential rows
        for shard_name in meta["shard"].unique():
            shard_meta = meta[meta["shard"] == shard_name]
            assert list(shard_meta["row"]) == list(range(len(shard_meta)))

    def test_sample_uid_uniqueness(self, tmp_path):
        """Every sample should have a unique sample_uid."""
        from ptcg_mine.config import MineConfig

        episodes, vocab, archetypes, self_ids, opp_ids, experts = self._setup_test_data(tmp_path, n_episodes=6)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(out_dir / "archetypes.json", "w") as f:
            json.dump(
                {
                    "self_ids": self_ids,
                    "opp_ids": opp_ids,
                    "fixed_deck": [1] * 60,
                    "archetypes": [
                        {"id": a.id, "representative": list(a.representative), "frequency": a.frequency, "n_members": len(getattr(a, "members", []))}
                        for a in archetypes
                    ],
                },
                f,
            )

        config = MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir)

        summary = build_shards(
            config=config,
            episodes=episodes,
            vocab=vocab,
            archetypes_data=archetypes,
            self_ids=self_ids,
            opp_ids=opp_ids,
            experts=experts,
        )

        meta = pd.read_parquet(out_dir / "meta.parquet")
        assert meta["sample_uid"].is_unique
        assert len(meta) == summary["total_samples"]

    def test_shard_contents(self, tmp_path):
        """Shard .npz files contain all featurizer keys with correct stacking."""
        from ptcg_mine.config import MineConfig

        episodes, vocab, archetypes, self_ids, opp_ids, experts = self._setup_test_data(tmp_path, n_episodes=2)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(out_dir / "archetypes.json", "w") as f:
            json.dump(
                {
                    "self_ids": self_ids,
                    "opp_ids": opp_ids,
                    "fixed_deck": [1] * 60,
                    "archetypes": [
                        {"id": a.id, "representative": list(a.representative), "frequency": a.frequency, "n_members": len(getattr(a, "members", []))}
                        for a in archetypes
                    ],
                },
                f,
            )

        config = MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir)

        summary = build_shards(
            config=config,
            episodes=episodes,
            vocab=vocab,
            archetypes_data=archetypes,
            self_ids=self_ids,
            opp_ids=opp_ids,
            experts=experts,
        )

        assert summary["total_samples"] > 0

        shards_dir = out_dir / "shards"
        for shard_file in sorted(shards_dir.glob("*.npz")):
            data = np.load(shard_file)
            # Check essential keys are present.  Card features are *not* among
            # them: they are a gather from a frozen static table, so the shard
            # stores the ids and ShardDataset rebuilds the features on read.
            essential_keys = {
                "poke_card_id", "hand_card_id", "opt_card_id",
                "cls_feat", "opt_type",
                "opt_src_idx", "action_idx", "action_len", "sel_type", "sel_ctx",
                "value_target", "tok_mask", "opt_mask",
            }
            file_keys = set(data.keys())
            missing = essential_keys - file_keys
            assert not missing, f"Shard {shard_file.name} missing keys: {missing}"

            # The regression this format change fixes: materialised card
            # features made a 50k-sample shard 6.2 GB decompressed, and the
            # train split's mmap cache 47 GB against 30 GB of RAM.
            stored_feats = {k for k in file_keys if k.endswith("_card_feat")}
            assert not stored_feats, (
                f"Shard {shard_file.name} stores materialised card features: "
                f"{stored_feats}"
            )

            # All arrays should have batch dim = number of samples in shard
            n_samples = data["action_len"].shape[0]
            # action_len should be 1 (single-select), action_idx[0] should be valid
            for i in range(n_samples):
                action_len = int(data["action_len"][i])
                assert 0 <= action_len <= 64

    def test_no_kept_games_empty_meta(self, tmp_path):
        """When no games pass the filter, produce empty meta.parquet and no shards."""
        from ptcg_mine.config import MineConfig

        # Create episodes where NO game passes the filter (wrong experts)
        episodes: list[tuple[str, dict]] = []
        for i in range(3):
            ep = _make_synthetic_episode(
                episode_id=f"bad_{i}",
                teams=("noob_a", "noob_b"),
                rewards=(-1, 1),
                deck_p0=_make_deck(1),
                deck_p1=_make_deck(2),
                n_decisions=1,
            )
            episodes.append((f"bad_{i}", ep))

        vocab = _build_test_vocab([ep for _, ep in episodes])
        archetypes, self_ids, opp_ids = _build_test_archetypes()
        experts = {"some_other_expert"}

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(out_dir / "archetypes.json", "w") as f:
            json.dump(
                {
                    "self_ids": self_ids,
                    "opp_ids": opp_ids,
                    "fixed_deck": [1] * 60,
                    "archetypes": [
                        {"id": a.id, "representative": list(a.representative), "frequency": a.frequency, "n_members": len(getattr(a, "members", []))}
                        for a in archetypes
                    ],
                },
                f,
            )

        config = MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir)

        summary = build_shards(
            config=config,
            episodes=episodes,
            vocab=vocab,
            archetypes_data=archetypes,
            self_ids=self_ids,
            opp_ids=opp_ids,
            experts=experts,
        )

        assert summary["total_samples"] == 0
        assert summary["n_shards"] == 0
        assert summary["n_kept_games"] == 0

        meta_path = out_dir / "meta.parquet"
        assert meta_path.exists()
        meta = pd.read_parquet(meta_path)
        assert len(meta) == 0
        assert set(meta.columns) == set(META_COLUMNS)

    def test_no_episodes(self, tmp_path):
        """Empty episode list produces empty output without crashing."""
        from ptcg_mine.config import MineConfig

        vocab = _build_test_vocab([])
        archetypes_full = _build_test_archetypes()
        archetypes = archetypes_full[0]
        self_ids = archetypes_full[1]
        opp_ids = archetypes_full[2]
        # Actually _build_test_archetypes returns (archetypes, self_ids, opp_ids)
        archetypes_full = _build_test_archetypes()
        archetypes = archetypes_full[0]
        self_ids = archetypes_full[1]
        opp_ids = archetypes_full[2]

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        with open(out_dir / "vocab.json", "w") as f:
            json.dump({"id_to_index": {}, "index_to_id": ["PAD", "UNKNOWN"], "freq": {}, "size": 2, "freq_coverage": []}, f)
        with open(out_dir / "archetypes.json", "w") as f:
            json.dump(
                {
                    "self_ids": self_ids,
                    "opp_ids": opp_ids,
                    "fixed_deck": [1] * 60,
                    "archetypes": [
                        {"id": a.id, "representative": list(a.representative), "frequency": a.frequency, "n_members": len(getattr(a, "members", []))}
                        for a in archetypes
                    ],
                },
                f,
            )

        config = MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir)

        summary = build_shards(
            config=config,
            episodes=[],
            experts={"x"},
        )

        assert summary["total_samples"] == 0
        assert summary["n_kept_games"] == 0

    def test_won_flag_consistency(self, tmp_path):
        """meta 'won' column matches episode rewards."""
        from ptcg_mine.config import MineConfig

        # Create one won episode and one lost episode
        ep_won = _make_synthetic_episode(
            episode_id="won_001",
            teams=("expert_a", "other"),
            rewards=(1, -1),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
            n_decisions=2,
        )
        ep_lost = _make_synthetic_episode(
            episode_id="lost_001",
            teams=("expert_a", "other"),
            rewards=(-1, 1),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
            n_decisions=2,
        )
        episodes = [("won_001", ep_won), ("lost_001", ep_lost)]

        vocab = _build_test_vocab([ep for _, ep in episodes])
        archetypes, self_ids, opp_ids = _build_test_archetypes()
        experts = {"expert_a"}

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(out_dir / "archetypes.json", "w") as f:
            json.dump(
                {
                    "self_ids": self_ids,
                    "opp_ids": opp_ids,
                    "fixed_deck": [1] * 60,
                    "archetypes": [
                        {"id": a.id, "representative": list(a.representative), "frequency": a.frequency, "n_members": len(getattr(a, "members", []))}
                        for a in archetypes
                    ],
                },
                f,
            )

        config = MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir)

        build_shards(
            config=config,
            episodes=episodes,
            vocab=vocab,
            archetypes_data=archetypes,
            self_ids=self_ids,
            opp_ids=opp_ids,
            experts=experts,
        )

        meta = pd.read_parquet(out_dir / "meta.parquet")

        # All samples from won_001 should have won=True
        meta_won = meta[meta["episode_id"] == "won_001"]
        assert len(meta_won) > 0
        assert meta_won["won"].all()

        # All samples from lost_001 should have won=False
        meta_lost = meta[meta["episode_id"] == "lost_001"]
        assert len(meta_lost) > 0
        assert not meta_lost["won"].any()

    def test_value_target_in_shards(self, tmp_path):
        """value_target in shard: +1 for won, -1 for lost."""
        from ptcg_mine.config import MineConfig

        ep_won = _make_synthetic_episode(
            episode_id="vt_001",
            teams=("expert_a", "other"),
            rewards=(1, -1),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
            n_decisions=2,
        )
        ep_lost = _make_synthetic_episode(
            episode_id="vt_002",
            teams=("expert_a", "other"),
            rewards=(-1, 1),
            deck_p0=_make_deck(1),
            deck_p1=_make_deck(2),
            n_decisions=2,
        )
        episodes = [("vt_001", ep_won), ("vt_002", ep_lost)]

        vocab = _build_test_vocab([ep for _, ep in episodes])
        archetypes, self_ids, opp_ids = _build_test_archetypes()
        experts = {"expert_a"}

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f)
        with open(out_dir / "archetypes.json", "w") as f:
            json.dump(
                {
                    "self_ids": self_ids,
                    "opp_ids": opp_ids,
                    "fixed_deck": [1] * 60,
                    "archetypes": [
                        {"id": a.id, "representative": list(a.representative), "frequency": a.frequency, "n_members": len(getattr(a, "members", []))}
                        for a in archetypes
                    ],
                },
                f,
            )

        config = MineConfig(raw_dir=tmp_path / "raw", out_dir=out_dir)

        build_shards(
            config=config,
            episodes=episodes,
            vocab=vocab,
            archetypes_data=archetypes,
            self_ids=self_ids,
            opp_ids=opp_ids,
            experts=experts,
        )

        # Verify value_target in shards matches meta won flag
        meta = pd.read_parquet(out_dir / "meta.parquet")
        shards_dir = out_dir / "shards"

        seen_won = False
        seen_lost = False
        for shard_file in sorted(shards_dir.glob("*.npz")):
            data = np.load(shard_file)
            for i in range(len(data["value_target"])):
                vt = float(data["value_target"][i])
                # Find this sample in meta
                row_in_shard = i
                shard_name = shard_file.name
                meta_row = meta[(meta["shard"] == shard_name) & (meta["row"] == row_in_shard)]
                if len(meta_row) == 0:
                    continue
                won = meta_row.iloc[0]["won"]
                if won:
                    assert vt == 1.0
                    seen_won = True
                else:
                    assert vt == -1.0
                    seen_lost = True
        assert seen_won, "Should have at least one won sample"
        assert seen_lost, "Should have at least one lost sample"

    def test_build_shards_no_config_no_episodes(self):
        """build_shards raises ValueError if neither config nor episodes given."""
        with pytest.raises(ValueError, match="either config or episodes"):
            build_shards()

    def test_missing_vocab_file(self, tmp_path):
        """build_shards raises FileNotFoundError if vocab.json is missing."""
        from ptcg_mine.config import MineConfig

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        config = MineConfig(raw_dir=raw_dir, out_dir=out_dir)

        episodes, _, archetypes, self_ids, opp_ids, experts = self._setup_test_data(tmp_path, n_episodes=1)

        with pytest.raises(FileNotFoundError, match="vocab"):
            build_shards(
                config=config,
                episodes=episodes,
                archetypes_data=archetypes,
                self_ids=self_ids,
                opp_ids=opp_ids,
                experts=experts,
            )

    def test_missing_archetypes_file(self, tmp_path):
        """build_shards raises FileNotFoundError if archetypes.json is missing."""
        from ptcg_mine.config import MineConfig

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        with open(out_dir / "vocab.json", "w") as f:
            json.dump(_build_test_vocab([]), f)

        config = MineConfig(raw_dir=raw_dir, out_dir=out_dir)

        episodes, vocab, archetypes, self_ids, opp_ids, experts = self._setup_test_data(tmp_path, n_episodes=1)

        with pytest.raises(FileNotFoundError, match="archetypes"):
            build_shards(
                config=config,
                episodes=episodes,
                vocab=vocab,
            )

    def test_streaming_path_matches_in_memory_and_is_memory_bounded(self, tmp_path):
        """build_shards must stream the corpus off disk, not hold it resident.

        Whole parsed episodes cost ~14.5 MB each (3.5x their ~4 MB on disk), so
        retaining the corpus needs ~145 GB at the 10k-episode target. The
        disk-backed path must (a) produce exactly what the in-memory path does,
        and (b) peak well below the corpus size.
        """
        import tracemalloc

        from ptcg_mine.config import MineConfig

        episodes, vocab, archetypes, self_ids, opp_ids, experts = self._setup_test_data(
            tmp_path, n_episodes=40
        )

        def _write_artifacts(out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "vocab.json", "w") as f:
                json.dump(vocab, f)
            with open(out_dir / "archetypes.json", "w") as f:
                json.dump(
                    {
                        "self_ids": self_ids,
                        "opp_ids": opp_ids,
                        "archetypes": [
                            {
                                "id": a.id,
                                "representative": list(a.representative),
                                "frequency": a.frequency,
                                "n_members": len(getattr(a, "members", [])),
                            }
                            for a in archetypes
                        ],
                    },
                    f,
                )

        # Reference run: episodes handed over in memory.
        mem_out = tmp_path / "data_mem"
        _write_artifacts(mem_out)
        mem_summary = build_shards(
            config=MineConfig(raw_dir=tmp_path / "raw_unused", out_dir=mem_out),
            episodes=episodes,
            vocab=vocab,
            archetypes_data=archetypes,
            self_ids=self_ids,
            opp_ids=opp_ids,
            experts=experts,
        )

        # Same corpus on disk, with each step log inflated so the corpus is far
        # larger than its Phase-2 projection -- exactly the real-episode shape.
        #
        # The fixture has to be big enough to separate the two hypotheses, and
        # the separating variable is *episode count*, not padding: retaining the
        # corpus scales linearly with n, while a streaming build's peak is
        # roughly flat (one parse's scratch, plus shard buffers that scale with
        # sample count rather than corpus size).
        #
        # Parsing one episode costs ~12x its bytes at peak -- `load_episode`
        # uses orjson, whose transient scratch is ~3x the stdlib's for the same
        # retained object graph (measured on a real 6.32 MB episode: 24.3 MB
        # retained either way, 106.4 MB peak vs 29.7 MB).  That constant swamps
        # a small fixture: at the 6 episodes this test used to build, streaming
        # peaked at 16.2x the largest episode against 19.5x for retaining
        # everything -- only 1.20x apart, so no threshold could tell them apart
        # and the assertion below would have passed either way.
        #
        # Measured peak/corpus (streaming vs retain-all) by episode count:
        #   n=6   2.69 vs 3.25   n=12  1.45 vs 2.16   n=24  0.82 vs 1.62
        #   n=40  0.57 vs 1.40   n=60  0.45 vs 1.30
        # At n=40 one episode is ~3.9 MB against a ~155 MB corpus and the two
        # hypotheses straddle 1.0x with room on both sides, which is a wider
        # margin than this test ever had.
        raw_dir = tmp_path / "raw_stream"
        raw_dir.mkdir(parents=True)
        for eid, ep in episodes:
            fat = dict(ep)
            padding = [{"junk": "y" * 4000} for _ in range(2)]
            fat["steps"] = list(ep["steps"]) + [padding for _ in range(480)]
            (raw_dir / f"{eid}.json").write_text(json.dumps(fat))
        corpus_bytes = sum(p.stat().st_size for p in raw_dir.glob("*.json"))
        largest = max(p.stat().st_size for p in raw_dir.glob("*.json"))
        assert corpus_bytes > 4 * largest, (
            "fixture too small to tell 'one episode resident' from 'corpus "
            "resident'; raise the padding or the episode count"
        )

        stream_out = tmp_path / "data_stream"
        _write_artifacts(stream_out)

        tracemalloc.start()
        stream_summary = build_shards(
            config=MineConfig(raw_dir=raw_dir, out_dir=stream_out),
            vocab=vocab,
            archetypes_data=archetypes,
            self_ids=self_ids,
            opp_ids=opp_ids,
            experts=experts,
        )
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        # (a) identical work, whichever way the episodes arrived
        assert stream_summary["n_kept_games"] == mem_summary["n_kept_games"]
        assert stream_summary["total_samples"] == mem_summary["total_samples"]
        assert stream_summary["split_counts"] == mem_summary["split_counts"]
        assert stream_summary["n_kept_games"] > 0, "fixture kept nothing; test is vacuous"

        # (b) peak stays sub-corpus: one episode resident, not all of them
        assert peak < corpus_bytes, (
            f"peak {peak / 1e6:.1f} MB >= corpus {corpus_bytes / 1e6:.1f} MB — "
            "build_shards is still holding the corpus in memory"
        )


def test_opt_group_is_not_stored_as_fp16():
    from ptcg_il.shard_writer import _FP16_KEYS
    assert "opt_group" not in _FP16_KEYS, "opt_group is an index array, not a feature"


# ============================================================
# Tests: parallel pass B
# ============================================================


class TestParallelPassB:
    """Pass B featurizes episodes across a process pool.

    The pass is ~98% of build-shards' wall clock, and the only defensible way
    to parallelise something whose output feeds every downstream stage is for
    the parallel result to be *identical*, not merely equivalent -- `shard` and
    `row` in meta.parquet are positions into files, so any reordering silently
    repoints every training sample.
    """

    @staticmethod
    def _corpus_on_disk(tmp_path, n_episodes):
        """Write a synthetic corpus to raw/ and the artifacts each run needs."""
        episodes, vocab, archetypes, self_ids, opp_ids, _ = (
            TestBuildShards._setup_test_data(tmp_path, n_episodes=n_episodes)
        )
        raw_dir = tmp_path / "raw_corpus"
        raw_dir.mkdir(parents=True)
        for eid, ep in episodes:
            (raw_dir / f"{eid}.json").write_text(json.dumps(ep))

        def write_artifacts(out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "vocab.json", "w") as f:
                json.dump(vocab, f)
            with open(out_dir / "archetypes.json", "w") as f:
                json.dump(
                    {
                        "self_ids": self_ids,
                        "opp_ids": opp_ids,
                        "archetypes": [
                            {
                                "id": a.id,
                                "representative": list(a.representative),
                                "frequency": a.frequency,
                                "n_members": len(getattr(a, "members", [])),
                            }
                            for a in archetypes
                        ],
                    },
                    f,
                )

        return raw_dir, write_artifacts

    def test_parallel_output_is_identical_to_serial(self, tmp_path, monkeypatch):
        """Same shards, same rows, same meta -- byte for byte, not just in aggregate."""
        from ptcg_mine.config import MineConfig
        from ptcg_il import shard_writer

        raw_dir, write_artifacts = self._corpus_on_disk(tmp_path, n_episodes=40)
        # The production threshold is 256 episodes; drop it so a fixture small
        # enough to run in a test still takes the pool.
        monkeypatch.setattr(shard_writer, "_PARALLEL_FEATURIZE_MIN_EPISODES", 1)

        runs = {}
        for tag, jobs in (("serial", 1), ("parallel", 4)):
            out_dir = tmp_path / tag
            write_artifacts(out_dir)
            runs[tag] = build_shards(
                config=MineConfig(raw_dir=raw_dir, out_dir=out_dir),
                samples_per_shard=200,  # forces several flushes, so shard/row vary
                jobs=jobs,
            )

        assert runs["serial"]["total_samples"] > 0, "fixture produced nothing; test is vacuous"
        assert runs["serial"]["n_shards"] > 1, (
            "fixture fits in one shard, so shard/row placement is never exercised"
        )
        for key in ("total_samples", "n_shards", "n_kept_games", "split_counts",
                    "n_loaded", "n_invalid"):
            assert runs["serial"][key] == runs["parallel"][key], f"summary differs on {key}"

        meta_s = pd.read_parquet(tmp_path / "serial" / "meta.parquet")
        meta_p = pd.read_parquet(tmp_path / "parallel" / "meta.parquet")
        pd.testing.assert_frame_equal(meta_s, meta_p)

        names_s = sorted(p.name for p in (tmp_path / "serial" / "shards").glob("*.npz"))
        names_p = sorted(p.name for p in (tmp_path / "parallel" / "shards").glob("*.npz"))
        assert names_s == names_p

        n_compared = 0
        for name in names_s:
            a = np.load(tmp_path / "serial" / "shards" / name)
            b = np.load(tmp_path / "parallel" / "shards" / name)
            assert sorted(a.files) == sorted(b.files), f"{name}: key sets differ"
            for k in a.files:
                assert a[k].dtype == b[k].dtype, f"{name}:{k} dtype"
                assert np.array_equal(a[k], b[k]), f"{name}:{k} values"
                n_compared += 1
        assert n_compared > 0, "no arrays compared; test is vacuous"

    def test_imap_applies_backpressure(self, tmp_path):
        """The feeder must not race ahead of the consumer.

        `Pool.imap` buffers every result the consumer has not taken, and pass B's
        workers outrun the parent that stacks and compresses their output -- so
        without the semaphore the parent accumulates the whole corpus's samples.
        On the real corpus that is ~125 GB, which presents as a systemd-oomd
        kill of the terminal scope rather than a MemoryError.
        """
        from ptcg_il import shard_writer

        n_tasks = 300
        n_jobs = 2
        window = shard_writer._FEATURIZE_INFLIGHT_PER_WORKER * n_jobs
        pulled = []

        class CountingTasks:
            def __iter__(self):
                for i in range(n_tasks):
                    pulled.append(i)
                    # Path that does not exist: the worker returns an empty
                    # result, which is all this test needs.
                    yield (str(tmp_path / f"missing_{i}.json"), [])

        gen = shard_writer._imap_featurized(CountingTasks(), n_jobs, {})
        try:
            next(gen)  # take exactly one result
            # The feeder may legitimately be one window ahead plus whatever is
            # in flight; it must not have drained the whole task list.
            assert len(pulled) <= window + 2 * n_jobs + 1, (
                f"feeder pulled {len(pulled)} of {n_tasks} tasks after one result "
                f"was consumed (window is {window}) — backpressure is not applied"
            )
        finally:
            gen.close()

        assert pulled, "no tasks were pulled; test is vacuous"

    def test_abandoning_the_generator_does_not_hang(self, tmp_path):
        """Closing the generator early must tear the pool down, not deadlock.

        The pool's task handler blocks inside the backpressure semaphore, and
        `Pool.terminate()` joins that thread — so a non-interruptible acquire
        turns any early exit (an exception in the consumer, a `break`) into a
        permanent hang instead of the error being raised.
        """
        import threading

        from ptcg_il import shard_writer

        tasks = [(str(tmp_path / f"missing_{i}.json"), []) for i in range(500)]
        done = threading.Event()

        def run():
            gen = shard_writer._imap_featurized(tasks, 2, {})
            next(gen)
            gen.close()  # -> GeneratorExit -> finally -> pool.terminate()/join()
            done.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=60)
        assert done.is_set(), "abandoning the pass-B generator deadlocked on pool teardown"


# ============================================================
# Pass A: selection folded into the scan workers
# ============================================================


class TestPassASelection:
    """Selection runs inside the pass-A workers, not in the parent afterwards.

    It is 300-cluster multiset Jaccard per player -- measured 39.8 s for 3000
    episodes, ~12.3 min over the full corpus -- and it used to run serially in
    the parent over projections the workers had already shipped back.  Folding
    it in has to leave the keep decision *identical*, because `keep` is what
    decides which rows exist at all.
    """

    def test_parallel_pass_a_selects_exactly_what_serial_does(self, tmp_path, monkeypatch):
        """Same kept pairs, same shards, same meta -- pool or no pool.

        `test_parallel_output_is_identical_to_serial` covers pass B but leaves
        pass A serial in both arms (its fixture is under the 256-file pool
        threshold), so without dropping that threshold here the parallel
        selection path is never executed by any test.
        """
        from ptcg_mine.config import MineConfig
        from ptcg_il import shard_writer

        raw_dir, write_artifacts = TestParallelPassB._corpus_on_disk(
            tmp_path, n_episodes=40)
        monkeypatch.setattr(shard_writer, "_PARALLEL_SCAN_MIN_FILES", 1)
        monkeypatch.setattr(shard_writer, "_PARALLEL_FEATURIZE_MIN_EPISODES", 1)

        runs = {}
        for tag, jobs in (("serial", 1), ("parallel", 4)):
            out_dir = tmp_path / f"passa_{tag}"
            write_artifacts(out_dir)
            runs[tag] = build_shards(
                config=MineConfig(raw_dir=raw_dir, out_dir=out_dir),
                samples_per_shard=200,
                jobs=jobs,
            )

        assert runs["serial"]["n_kept_games"] > 0, "fixture kept nothing; test is vacuous"
        for key in ("n_kept_games", "total_samples", "n_shards", "split_counts",
                    "n_loaded", "n_invalid"):
            assert runs["serial"][key] == runs["parallel"][key], f"summary differs on {key}"

        meta_s = pd.read_parquet(tmp_path / "passa_serial" / "meta.parquet")
        meta_p = pd.read_parquet(tmp_path / "passa_parallel" / "meta.parquet")
        pd.testing.assert_frame_equal(meta_s, meta_p)

    def test_streamed_leaderboard_matches_stats(self, tmp_path):
        """The per-episode tally must reproduce `stats.team_leaderboard`.

        The parent no longer holds the projections that function takes a list
        of -- ~36 KB each, ~2.0 GiB over the 55k-episode corpus -- so it counts
        games and wins as pass A streams instead.  Same numbers or the skill
        weight silently changes for every row.
        """
        from ptcg_mine.episode import rewards, teams
        from ptcg_mine.stats import team_leaderboard
        from ptcg_il.shard_writer import _leaderboard_from_counts

        episodes, _, _, _, _, _ = TestBuildShards._setup_test_data(
            tmp_path, n_episodes=12)
        eps = [ep for _, ep in episodes]

        counts: dict[str, list[int]] = {}
        for ep in eps:
            t0, t1 = teams(ep)
            r0, r1 = rewards(ep)
            for team, r in ((t0, r0), (t1, r1)):
                c = counts.setdefault(team, [0, 0])
                c[0] += 1
                if r == 1:
                    c[1] += 1

        streamed = _leaderboard_from_counts(counts)
        assert streamed == team_leaderboard(eps)
        assert len(streamed) > 1, "fixture has one team; test cannot separate them"
        assert sum(v["games"] for v in streamed.values()) == 2 * len(eps)
        assert any(v["wins"] for v in streamed.values()), "no wins recorded; test is vacuous"


# ============================================================
# meta.parquet is streamed, and the shard buffers are budgeted
# ============================================================


class TestWriterMemoryBounds:
    """The two parent-side terms that decided whether a full build survives.

    Measured over the 55k-episode corpus: `meta_rows` as a list of dicts was
    781 B/row -> 2.67 GiB at 3.67M rows with no ceiling at all, and the three
    shard buffers were 3 x 50000 x 17.2 KiB = 2.58 GiB.  Together with the
    transient `np.stack` copy that is ~6.9 GiB in the parent while 240 GB of
    raw JSON streams through page cache -- the pattern that gets the terminal
    scope killed by systemd-oomd rather than raising MemoryError.
    """

    def test_meta_is_written_in_row_groups_not_one_frame(self, tmp_path, monkeypatch):
        """Streaming is observable in the file: several row groups, not one.

        This is the mutation-sensitive half -- reverting to "accumulate every
        row, then build one DataFrame" yields exactly one row group.
        """
        import pyarrow.parquet as pq
        from ptcg_mine.config import MineConfig
        from ptcg_il import shard_writer

        raw_dir, write_artifacts = TestParallelPassB._corpus_on_disk(
            tmp_path, n_episodes=12)
        monkeypatch.setattr(shard_writer, "_META_FLUSH_ROWS", 5)

        out_dir = tmp_path / "rowgroups"
        write_artifacts(out_dir)
        summary = build_shards(config=MineConfig(raw_dir=raw_dir, out_dir=out_dir),
                               samples_per_shard=200, jobs=1)

        assert summary["total_samples"] > 5, "fixture fits in one flush; test is vacuous"
        pf = pq.ParquetFile(out_dir / "meta.parquet")
        assert pf.num_row_groups > 1, (
            f"{summary['total_samples']} rows at a 5-row flush produced "
            f"{pf.num_row_groups} row group(s) -- meta is still accumulated in full"
        )
        assert pf.metadata.num_rows == summary["total_samples"]

    def test_row_group_boundaries_do_not_change_meta_content(self, tmp_path, monkeypatch):
        """Where the flushes land must not alter a single value, type or row order."""
        from ptcg_mine.config import MineConfig
        from ptcg_il import shard_writer

        raw_dir, write_artifacts = TestParallelPassB._corpus_on_disk(
            tmp_path, n_episodes=12)

        frames = {}
        for tag, flush in (("many", 7), ("one", 10_000_000)):
            monkeypatch.setattr(shard_writer, "_META_FLUSH_ROWS", flush)
            out_dir = tmp_path / f"flush_{tag}"
            write_artifacts(out_dir)
            build_shards(config=MineConfig(raw_dir=raw_dir, out_dir=out_dir),
                         samples_per_shard=200, jobs=1)
            frames[tag] = pd.read_parquet(out_dir / "meta.parquet")

        assert len(frames["many"]) > 7, "fixture fits in one flush; test is vacuous"
        pd.testing.assert_frame_equal(frames["many"], frames["one"])
        assert set(frames["one"].columns) == set(META_COLUMNS)

    def test_meta_residency_is_flat_in_corpus_size(self, tmp_path, monkeypatch):
        """Rows held in memory must stay bounded by the flush size, whatever n is.

        Asserted on the writer's own high-water mark rather than on process
        peak.  A whole-process `tracemalloc` bound cannot see this at test
        scale: 60 vs 120 meta rows is 47 KB vs 94 KB against shard buffers and
        one parsed episode, so an "accumulate everything" mutant survives it
        while the real corpus it models is 2.67 GiB.  Measuring the term
        directly is what makes the assertion sharp enough to mean anything.
        """
        from ptcg_mine.config import MineConfig
        from ptcg_il import shard_writer

        flush = 10
        monkeypatch.setattr(shard_writer, "_META_FLUSH_ROWS", flush)

        real_add = shard_writer._MetaWriter.add
        high_water: dict[int, int] = {}
        current_n = {"n": 0}

        def watched_add(self, row, split):
            real_add(self, row, split)
            n = current_n["n"]
            high_water[n] = max(high_water.get(n, 0), len(self._pending))

        monkeypatch.setattr(shard_writer._MetaWriter, "add", watched_add)

        rows = {}
        for n in (12, 24):
            current_n["n"] = n
            raw_dir, write_artifacts = TestParallelPassB._corpus_on_disk(
                tmp_path / f"n{n}", n_episodes=n)
            out_dir = tmp_path / f"n{n}" / "out"
            write_artifacts(out_dir)
            rows[n] = build_shards(
                config=MineConfig(raw_dir=raw_dir, out_dir=out_dir),
                samples_per_shard=20, jobs=1)["total_samples"]

        assert rows[24] > rows[12] * 1.5, (
            f"corpus did not grow enough to separate the hypotheses "
            f"({rows[12]} -> {rows[24]} rows); test is vacuous"
        )
        assert rows[12] > flush, "fixture never fills one flush; test is vacuous"
        for n in (12, 24):
            assert high_water[n] <= flush, (
                f"n={n}: held {high_water[n]} meta rows against a {flush}-row "
                f"flush -- meta is accumulating again"
            )
        assert high_water[24] == high_water[12], (
            f"rows held grew {high_water[12]} -> {high_water[24]} as the corpus "
            f"doubled; meta residency must not scale with n"
        )

    def test_budget_bounds_the_three_buffers_plus_the_flush_copy(self):
        """`samples_per_shard_for_budget` must respect the arithmetic it claims."""
        import numpy as np
        from ptcg_il.shard_writer import sample_nbytes, samples_per_shard_for_budget

        sample = {f"k{i}": np.zeros((64, 8), dtype=np.float32) for i in range(20)}
        resident, payload = sample_nbytes(sample)
        assert resident > payload, "resident must include per-ndarray overhead"

        for budget in (0.25, 1.0, 2.0, 8.0):
            n = samples_per_shard_for_budget(budget, sample)
            worst_case = n * (3 * resident + payload)
            assert worst_case <= budget * 1024 ** 3, (
                f"{budget} GiB budget -> {n} samples/shard = "
                f"{worst_case / 1024 ** 3:.2f} GiB worst case"
            )
            # and it must not be uselessly conservative
            assert (n + 1) * (3 * resident + payload) > budget * 1024 ** 3

    def test_budget_never_returns_zero(self):
        """A budget smaller than one sample still has to make progress."""
        import numpy as np
        from ptcg_il.shard_writer import samples_per_shard_for_budget

        huge = {"k": np.zeros((1024, 1024), dtype=np.float64)}  # 8 MiB
        assert samples_per_shard_for_budget(0.0001, huge) == 1

    def test_smaller_budget_yields_more_shards(self, tmp_path):
        """The knob has to actually move the shard layout, end to end."""
        from ptcg_mine.config import MineConfig

        raw_dir, write_artifacts = TestParallelPassB._corpus_on_disk(
            tmp_path, n_episodes=16)

        runs = {}
        for tag, budget in (("small", 0.002), ("large", 0.05)):
            out_dir = tmp_path / f"budget_{tag}"
            write_artifacts(out_dir)
            runs[tag] = build_shards(
                config=MineConfig(raw_dir=raw_dir, out_dir=out_dir),
                samples_per_shard=None, mem_budget_gb=budget, jobs=1)

        assert runs["small"]["total_samples"] == runs["large"]["total_samples"] > 0
        assert runs["small"]["n_shards"] > runs["large"]["n_shards"], (
            f"budget had no effect on layout: {runs['small']['n_shards']} vs "
            f"{runs['large']['n_shards']} shards"
        )

    def test_explicit_samples_per_shard_overrides_the_budget(self, tmp_path):
        """Pinning the shard size must ignore the budget entirely, so an old
        corpus stays reproducible."""
        from ptcg_mine.config import MineConfig

        raw_dir, write_artifacts = TestParallelPassB._corpus_on_disk(
            tmp_path, n_episodes=16)
        out_dir = tmp_path / "pinned"
        write_artifacts(out_dir)

        summary = build_shards(
            config=MineConfig(raw_dir=raw_dir, out_dir=out_dir),
            samples_per_shard=20, mem_budget_gb=0.001, jobs=1)

        meta = pd.read_parquet(out_dir / "meta.parquet")
        assert summary["total_samples"] > 20, "fixture never fills a shard; test is vacuous"
        assert meta.groupby("shard").size().max() == 20, (
            "shard size did not follow the pin -- the 0.001 GiB budget would "
            "have produced a much smaller shard"
        )

    def test_failed_build_leaves_no_readable_meta(self, tmp_path, monkeypatch):
        """A half-streamed meta.parquet must not survive as a valid file.

        Unlike the old build-it-all-then-write path, a streaming writer has
        already committed row groups by the time something downstream raises --
        and a meta that parses but names shards the run never flushed is worse
        than no meta at all.
        """
        from ptcg_mine.config import MineConfig
        from ptcg_il import shard_writer

        raw_dir, write_artifacts = TestParallelPassB._corpus_on_disk(
            tmp_path, n_episodes=12)
        monkeypatch.setattr(shard_writer, "_META_FLUSH_ROWS", 5)

        out_dir = tmp_path / "boom"
        write_artifacts(out_dir)

        real_write = shard_writer._write_shard
        state = {"n": 0}

        def exploding(split, idx, buffer, od):
            state["n"] += 1
            if state["n"] > 1:
                raise RuntimeError("disk full")
            return real_write(split, idx, buffer, od)

        monkeypatch.setattr(shard_writer, "_write_shard", exploding)

        with pytest.raises(RuntimeError, match="disk full"):
            build_shards(config=MineConfig(raw_dir=raw_dir, out_dir=out_dir),
                         samples_per_shard=50, jobs=1)

        assert state["n"] > 1, "fixture never reached a second flush; test is vacuous"
        meta_path = out_dir / "meta.parquet"
        if meta_path.exists():
            with pytest.raises(Exception):
                pd.read_parquet(meta_path)
