"""Tests for ptcg_mine.artifacts: write_vocab_json, write_archetypes_json,
write_mining_report."""

import json

from ptcg_mine.archetype import Archetype
from ptcg_mine.artifacts import write_archetypes_json, write_mining_report, write_vocab_json
from ptcg_mine.config import MineConfig


def _vocab():
    return {
        "id_to_index": {10: 2, 20: 3, 30: 4},
        "index_to_id": ["PAD", "UNKNOWN", 10, 20, 30],
        "freq": {10: 5, 20: 3, 30: 1},
        "size": 5,
        "freq_coverage": [(10, 5, 5 / 9), (20, 3, 8 / 9), (30, 1, 1.0)],
    }


def test_write_vocab_json_roundtrip(tmp_path):
    vocab = _vocab()
    attack_id_to_index = {100: 1, 200: 2}
    config = MineConfig()
    path = tmp_path / "vocab.json"

    write_vocab_json(path, vocab, attack_id_to_index, config)

    assert path.exists()
    with open(path) as f:
        data = json.load(f)

    assert data["id_to_index"] == {"10": 2, "20": 3, "30": 4}
    assert data["index_to_id"] == ["PAD", "UNKNOWN", 10, 20, 30]
    assert data["size"] == 5
    assert data["attack_id_to_index"] == {"100": 1, "200": 2}

    assert data["norm"]["HP_N"] == 400
    assert data["norm"]["RETREAT_N"] == 4
    assert data["norm"]["ATKDMG_N"] == 350
    assert data["norm"]["ATKCOST_N"] == 5

    assert data["caps"]["h_max"] == config.h_max
    assert data["caps"]["o_max"] == config.o_max
    assert data["caps"]["d_max"] == config.d_max

    assert data["F_CARD"] == 94  # 52 base + 3 attacks × 14
    assert data["F_ATK"] == 14
    assert data["F_POKE"] == 26
    assert data["F_HAND"] == 2
    assert data["F_SUM"] == 11
    assert data["F_GLOBAL"] == 93
    assert data["F_OPT"] == 6

    assert data["w_lost"] == config.w_lost


def test_write_archetypes_json(tmp_path):
    archetypes = [
        Archetype(id=0, representative=tuple(range(60)), members=[tuple(range(60))], frequency=42),
        Archetype(
            id=1,
            representative=tuple(range(60, 120)),
            members=[tuple(range(60, 120))],
            frequency=7,
        ),
    ]
    self_ids = [0]
    opp_ids = [1]
    fixed_deck = list(range(60))
    path = tmp_path / "archetypes.json"

    write_archetypes_json(path, self_ids, opp_ids, archetypes, fixed_deck)

    assert path.exists()
    with open(path) as f:
        data = json.load(f)

    assert data["self_ids"] == [0]
    assert data["opp_ids"] == [1]
    assert data["fixed_deck"] == fixed_deck
    assert len(data["fixed_deck"]) == 60
    assert "archetypes" in data
    assert len(data["archetypes"]) == 2
    arch0 = next(a for a in data["archetypes"] if a["id"] == 0)
    assert arch0["representative"] == list(range(60))
    assert arch0["frequency"] == 42


def test_write_mining_report(tmp_path):
    leaderboard = {
        "teamA": {"games": 100, "wins": 70, "win_rate": 0.7},
        "teamB": {"games": 60, "wins": 20, "win_rate": 1 / 3},
    }
    experts = ["teamA"]
    archetypes = [
        Archetype(id=0, representative=tuple(range(60)), members=[tuple(range(60))], frequency=42),
    ]
    self_ids = [0]
    opp_ids = [0]
    vocab = _vocab()
    counts = {"episodes_loaded": 500, "episodes_valid": 480, "expert_games": 120}
    path = tmp_path / "mining_report.md"

    write_mining_report(path, leaderboard, experts, archetypes, self_ids, opp_ids, vocab, counts)

    assert path.exists()
    text = path.read_text()
    assert "teamA" in text
    assert "0.7" in text or "70.0%" in text or "70%" in text
    assert "vocab" in text.lower()
    assert str(vocab["size"]) in text
    assert "500" in text
    assert "480" in text
