"""The predictor against the real `data/archetypes.json`.

Fixtures cannot catch a shape assumption that only the shipped artifact
violates, and this artifact is now the entire opponent model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ptcg_il.deck_prior import OpponentDeckPredictor, all_archetype_ids

ARCHETYPES_PATH = Path(__file__).resolve().parents[1] / "data" / "archetypes.json"


@pytest.fixture(scope="module")
def archetypes() -> dict:
    if not ARCHETYPES_PATH.exists():
        pytest.skip(f"no mined artifact at {ARCHETYPES_PATH}")
    with open(ARCHETYPES_PATH) as f:
        return json.load(f)


def test_every_archetype_has_a_full_representative(archetypes):
    ids = all_archetype_ids(archetypes)
    assert ids, "artifact has no archetypes"
    entries = {int(a["id"]): a for a in archetypes["archetypes"]}
    short = [i for i in ids if len(entries[i].get("representative") or []) != 60]
    assert not short, f"{len(short)} archetypes lack a 60-card representative: {short[:5]}"


def test_predictor_commits_to_sixty_cards_before_any_evidence(archetypes):
    deck = OpponentDeckPredictor(archetypes).template()
    assert len(deck) == 60


def test_hypothesis_set_is_much_wider_than_opp_ids(archetypes):
    pred = OpponentDeckPredictor(archetypes)
    n_opp = len(archetypes.get("opp_ids") or [])
    assert n_opp > 0, "artifact has no opp_ids to compare against"
    assert len(pred.ids) > n_opp * 5, (
        f"expected the full archetype set, got {len(pred.ids)} against "
        f"{n_opp} opp_ids")


def test_evidence_from_a_real_decklist_recovers_that_decklist(archetypes):
    """Reveal 20 cards of a real archetype; the argmax should be that deck.

    Fails on zero examined: the loop asserts it saw every id it planned to.
    """
    pred = OpponentDeckPredictor(archetypes)
    entries = {int(a["id"]): a for a in archetypes["archetypes"]}
    rng = np.random.default_rng(0)
    # The ten most frequent archetypes — the ones actually worth recovering.
    top = sorted(entries, key=lambda i: -entries[i].get("frequency", 0))[:10]

    examined = 0
    hits = 0
    for aid in top:
        deck = [int(c) for c in entries[aid]["representative"]]
        revealed = [int(c) for c in rng.choice(deck, size=20, replace=False)]
        pred.reset()
        pred.observe_cards(revealed)
        if sorted(pred.template()) == sorted(deck):
            hits += 1
        examined += 1

    assert examined == 10, f"examined {examined}, expected 10"
    assert hits >= 8, f"recovered only {hits}/10 real decklists from 20 cards"
