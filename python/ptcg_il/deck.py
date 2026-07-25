"""Deck identity for a trained policy.

A checkpoint is only usable if you know *which deck it was trained to play*.
The six self archetypes in this corpus are near-disjoint decks — pairwise
multiset-Jaccard <= 0.17, and not one card is common to all six — so a policy
fitted on archetype 2 has essentially never seen archetype 9's cards.  Loading
the wrong pairing degrades silently rather than crashing: every unseen card just
maps to ``UNKNOWN_CARD``.

This module builds that identity record once, and writes it to two places:

* inside the ``.pt`` under the ``"deck"`` key (so a checkpoint is self-describing
  even when moved on its own), and
* a ``decks.json`` sidecar next to the checkpoints (so the pairing is greppable
  without importing torch).

Both carry ``vocab_sha1`` / ``archetypes_sha1``, which is what actually catches a
mismatched artifact pairing — the deck id alone would not, because archetype ids
are just cluster indices and get reassigned whenever mining is re-run.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DECK_KEY = "deck"
SIDECAR_NAME = "decks.json"


def _sha1(path: Path) -> str:
    """Short content hash of an artifact file."""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _sample_counts(data_dir: Path, archetype_self: int | None) -> dict[str, int]:
    """Per-split decision-point counts for this deck, read from meta.parquet."""
    meta_path = data_dir / "meta.parquet"
    if not meta_path.exists():
        return {}
    import pandas as pd

    meta = pd.read_parquet(meta_path, columns=["shard", "archetype_self"])
    if archetype_self is not None:
        meta = meta[meta["archetype_self"] == archetype_self]
    splits = meta["shard"].str.split("-").str[0]
    return {str(k): int(v) for k, v in splits.value_counts().items()}


def build_deck_metadata(
    data_dir: str | Path,
    archetype_self: int | None = None,
) -> dict[str, Any]:
    """Assemble the deck identity record for a training run.

    Parameters
    ----------
    data_dir : str or Path
        Directory holding ``archetypes.json``, ``vocab.json``, ``meta.parquet``.
    archetype_self : int, optional
        The archetype this policy specialises in.  ``None`` means the model was
        trained across every deck (generalist), in which case the recorded deck
        list is ``archetypes.json``'s ``fixed_deck``.

    Returns
    -------
    dict
        JSON-serialisable — deliberately no numpy/torch types, so the same
        object goes into both the ``.pt`` and the JSON sidecar.
    """
    data_dir = Path(data_dir)
    arch_path = data_dir / "archetypes.json"
    with open(arch_path) as f:
        arch_doc = json.load(f)

    by_id = {int(a["id"]): a for a in arch_doc["archetypes"]}

    if archetype_self is None:
        deck = list(arch_doc["fixed_deck"])
        entry: dict[str, Any] = {}
    else:
        if archetype_self not in by_id:
            raise KeyError(
                f"archetype id {archetype_self} not in {arch_path} "
                f"(self_ids={arch_doc.get('self_ids')})"
            )
        entry = by_id[archetype_self]
        deck = list(entry["representative"])

    counts = Counter(int(c) for c in deck)
    vocab_path = data_dir / "vocab.json"
    with open(vocab_path) as f:
        vocab = json.load(f)

    return {
        "archetype_self": archetype_self,
        "specialist": archetype_self is not None,
        "deck": [int(c) for c in deck],
        "deck_size": len(deck),
        "deck_counts": {str(cid): n for cid, n in sorted(counts.items())},
        "n_distinct_cards": len(counts),
        # Corpus support — how much data actually backed this deck.
        "archetype_frequency": int(entry.get("frequency", 0)) if entry else None,
        "archetype_n_members": int(entry.get("n_members", 0)) if entry else None,
        "samples": _sample_counts(data_dir, archetype_self),
        # Artifact pinning: a checkpoint is only valid against these exact files.
        "vocab_size": int(vocab.get("size", len(vocab.get("id_to_index", {})))),
        "vocab_sha1": _sha1(vocab_path),
        "archetypes_sha1": _sha1(arch_path),
        "is_fixed_deck": list(deck) == list(arch_doc["fixed_deck"]),
        "data_dir": str(data_dir),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_deck_csv(deck_meta: dict[str, Any], path: str | Path) -> Path:
    """Write the 60-card deck in the engine's ``deck.csv`` format.

    One raw card id per line, no header — matches
    ``sample_submission/deck.csv``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(c) for c in deck_meta["deck"]) + "\n")
    return path


def update_sidecar(
    out_dir: str | Path,
    deck_meta: dict[str, Any],
    checkpoints: list[str] | None = None,
) -> Path:
    """Merge this run's deck record into ``<out_dir>/decks.json``.

    Keyed by ``archetype_self`` (``"all"`` for the generalist) so repeated runs
    of the same deck overwrite rather than accumulate, and training several
    specialists into sibling directories still yields one readable index per
    directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SIDECAR_NAME

    doc: dict[str, Any] = {"decks": {}}
    if path.exists():
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass  # corrupt sidecar is not worth failing a training run over
        doc.setdefault("decks", {})

    key = "all" if deck_meta.get("archetype_self") is None else str(deck_meta["archetype_self"])
    record = dict(deck_meta)
    if checkpoints:
        record["checkpoints"] = checkpoints
    doc["decks"][key] = record
    doc["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path


def read_deck_metadata(checkpoint_path: str | Path) -> dict[str, Any] | None:
    """Read the deck record out of a ``.pt`` without building a model.

    Returns ``None`` for checkpoints written before deck labelling existed.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get(DECK_KEY)


def describe(deck_meta: dict[str, Any] | None) -> str:
    """One-line human summary for logs."""
    if not deck_meta:
        return "deck=<unlabelled>"
    who = (
        "all-decks"
        if deck_meta.get("archetype_self") is None
        else f"archetype {deck_meta['archetype_self']}"
    )
    n = sum(deck_meta.get("samples", {}).values())
    return (
        f"deck={who} ({deck_meta.get('n_distinct_cards')} distinct cards, "
        f"{n} samples, vocab={deck_meta.get('vocab_sha1')})"
    )
