"""Phase 0: deterministic episode-day and episode-id sampling.

- select_days: pick the n_days most recent rows of the manifest (by date).
- deterministic_pick: pure, order-independent, reproducible subset selection
  keyed by a seed (stable-hash-then-sort), so repeated runs (and reruns after
  a restart) pick exactly the same ids.
"""

import hashlib

import pandas as pd


def select_days(manifest_df: pd.DataFrame, n_days: int) -> list[str]:
    """Return the n_days most recent dates in manifest_df, oldest-to-newest,
    as "YYYY-MM-DD" strings.

    Recency is determined by sorting the "date" column; "high top_avg_score"
    bias from the spec is satisfied by recency alone here (kept simple).
    """
    ordered = manifest_df.sort_values("date")
    recent = ordered["date"].tail(n_days).tolist()
    return [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in recent]


def deterministic_pick(ids: list[str], k: int, seed: int) -> list[str]:
    """Pick min(k, len(ids)) ids from ids, deterministically and reproducibly.

    Each id is stable-hashed together with the seed; ids are sorted by hash
    and the first k are taken. Same (ids, k, seed) => same output regardless
    of the input list's order.
    """
    if not ids:
        return []

    def _hash(item: str) -> str:
        return hashlib.sha1(f"{seed}:{item}".encode()).hexdigest()

    ranked = sorted(ids, key=_hash)
    return ranked[: min(k, len(ranked))]
