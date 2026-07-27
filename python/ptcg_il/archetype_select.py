"""Which archetypes to train specialists for, derived from the artifacts.

The pipeline used to hardcode ``ARCHETYPES="0 2"``.  That is unsafe, and the
repo already knows why: archetype ids are **cluster indices**, so re-running
mining reassigns them.  On the corpus this module was written against,
``self_ids`` is ``[0, 1, 11, 3, 4, 5]`` — archetype 2 is not a 𝒟_self archetype
at all, and a hardcoded ``--archetype-self 2`` fails with "No samples for
archetype_self=2", or worse, silently trains on whatever cluster inherited the
id.

So the ids are read from ``archetypes.json`` and ranked by how much *training
data* each one actually has in ``meta.parquet``.  Rows are the right ranking key
because a specialist with a few hundred rows cannot be model-selected, let alone
gated: RL_SPEC §14.1 records archetype 20 having zero val rows for exactly this
reason.
"""

from __future__ import annotations

import json
from pathlib import Path

# An archetype needs enough held-out data for the val (model selection) and test
# (baseline / gate) splits to mean something.  Below this it is excluded from
# the automatic pick, though an explicit id always wins.
MIN_VAL_ROWS = 100
MIN_TEST_ROWS = 100


def split_of_shard(shard: str) -> str:
    """``"train-00007"`` → ``"train"``.

    ``meta.parquet`` has no split column; the split lives in the shard name.
    """
    return str(shard).split("-")[0]


def archetype_row_counts(data_dir: str | Path) -> dict[int, dict[str, int]]:
    """``{archetype_id: {"train": n, "val": n, "test": n}}`` from meta.parquet."""
    import pandas as pd

    meta_path = Path(data_dir) / "meta.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.parquet not found at {meta_path}")

    meta = pd.read_parquet(meta_path, columns=["shard", "archetype_self"])
    meta["split"] = meta["shard"].map(split_of_shard)

    counts: dict[int, dict[str, int]] = {}
    for (arch, split), n in meta.groupby(["archetype_self", "split"]).size().items():
        counts.setdefault(int(arch), {"train": 0, "val": 0, "test": 0})[str(split)] = int(n)
    return counts


def pick_archetypes(data_dir: str | Path, top: int = 2) -> list[int]:
    """The *top* 𝒟_self archetypes with usable data, best-supported first.

    Intersects ``archetypes.json``'s ``self_ids`` with what ``meta.parquet``
    actually contains, drops anything too thin to hold out, and ranks by train
    rows.  Returns fewer than *top* ids when fewer qualify — silently padding
    with an under-supported archetype would produce a specialist that cannot be
    evaluated.
    """
    arch_path = Path(data_dir) / "archetypes.json"
    if not arch_path.exists():
        raise FileNotFoundError(f"archetypes.json not found at {arch_path}")
    with open(arch_path) as f:
        self_ids = [int(i) for i in json.load(f).get("self_ids", [])]

    counts = archetype_row_counts(data_dir)

    usable = [
        aid for aid in self_ids
        if aid in counts
        and counts[aid]["val"] >= MIN_VAL_ROWS
        and counts[aid]["test"] >= MIN_TEST_ROWS
    ]
    usable.sort(key=lambda aid: counts[aid]["train"], reverse=True)
    return usable[:top]


def describe(data_dir: str | Path) -> str:
    """A human-readable table of every 𝒟_self archetype and its row counts."""
    arch_path = Path(data_dir) / "archetypes.json"
    with open(arch_path) as f:
        self_ids = [int(i) for i in json.load(f).get("self_ids", [])]
    counts = archetype_row_counts(data_dir)

    lines = ["archetype   train     val    test  usable"]
    for aid in sorted(self_ids, key=lambda a: counts.get(a, {}).get("train", 0), reverse=True):
        c = counts.get(aid, {"train": 0, "val": 0, "test": 0})
        ok = c["val"] >= MIN_VAL_ROWS and c["test"] >= MIN_TEST_ROWS
        lines.append(
            f"{aid:>9}  {c['train']:>6}  {c['val']:>6}  {c['test']:>6}  {'yes' if ok else 'no'}"
        )
    return "\n".join(lines)
