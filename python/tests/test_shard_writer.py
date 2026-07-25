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
from ptcg_mine.archetype import Archetype, cluster_decks, canon


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
            # Check essential keys are present
            essential_keys = {
                "poke_card_id", "hand_card_id", "cls_feat", "opt_type",
                "opt_src_idx", "action_idx", "action_len", "sel_type", "sel_ctx",
                "value_target", "tok_mask", "opt_mask",
            }
            file_keys = set(data.keys())
            missing = essential_keys - file_keys
            assert not missing, f"Shard {shard_file.name} missing keys: {missing}"

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
            tmp_path, n_episodes=6
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
        raw_dir = tmp_path / "raw_stream"
        raw_dir.mkdir(parents=True)
        for eid, ep in episodes:
            fat = dict(ep)
            padding = [{"junk": "y" * 4000} for _ in range(2)]
            fat["steps"] = list(ep["steps"]) + [padding for _ in range(120)]
            (raw_dir / f"{eid}.json").write_text(json.dumps(fat))
        corpus_bytes = sum(p.stat().st_size for p in raw_dir.glob("*.json"))

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
