"""``il_baselines.json`` — the IL scores an RL run is allowed to regress against.

RL_SPEC §10.2 condition 3 gates promotion on non-trivial top-1 not falling more
than 5 points below the IL policy's own score.  That number has to come from
somewhere, and the obvious place — a constant in the spec — is exactly what went
wrong before: the ``opt_card_id`` fix (§6.4.2) changed the model's input
distribution, so the quoted 0.5394 / 0.7538 became measurements of features the
model no longer sees, while still looking like valid thresholds.

So the baseline is a pipeline *artifact*, written by stage 4c right after the
specialist that produced it, and pinned to that checkpoint by SHA-1.  A gate
handed a checkpoint whose SHA does not match the record refuses to run rather
than comparing against a number from a different model.  The failure mode this
removes is silent by nature: stale thresholds do not raise, they just quietly
pass or fail the wrong candidate.

Layout, keyed by archetype id as a string (JSON object keys are strings, and
round-tripping ints through them is a reliable source of lookup misses):

```json
{
  "2": {
    "ckpt_sha1": "3f9a…",
    "ckpt_path": "checkpoints_a2/ckpt-best.pt",
    "nontrivial_top1": 0.7538,
    "top1_macro": 0.83,
    "top1_micro": 0.891,
    "value_corr": 0.045,
    "value_std": 0.0006,
    "recorded_at": "2026-07-26T12:00:00+00:00"
  }
}
```
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASELINES_FILENAME = "il_baselines.json"

# Metrics copied out of an offline-eval result into the record.  Keeping the
# list explicit (rather than dumping every ``val/*`` key) keeps the artifact
# small and stable: it is read by a gate, not by a dashboard.
_RECORDED_METRICS = {
    "nontrivial_top1": "val/top1_nontrivial",
    "nontrivial_top1_firstlegal": "val/top1_nontrivial_firstlegal",
    "top1_macro": "val/top1_macro",
    "top1_micro": "val/top1_micro",
    "value_corr": "val/value_corr",
    "value_std": "val/value_std",
    "value_mse": "val/value_mse",
}


class BaselineMismatch(RuntimeError):
    """A baseline record does not belong to the checkpoint it was matched with."""


def checkpoint_sha1(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-1 of a checkpoint file, read in chunks.

    Checkpoints run to hundreds of MB, so this never loads the file whole.
    SHA-1 rather than a stronger hash because the threat here is *accident* —
    pairing a record with the wrong file — not forgery.
    """
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def baselines_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / BASELINES_FILENAME


def load_baselines(data_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Every recorded baseline, or ``{}`` when the file does not exist yet."""
    path = baselines_path(data_dir)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def record_baseline(
    data_dir: str | Path,
    archetype: int | str | None,
    ckpt_path: str | Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Write one archetype's baseline into ``il_baselines.json``.

    Merges into the existing file rather than replacing it, so training one
    specialist never erases another's record.  *archetype* may be ``None`` for a
    generalist run, which is stored under the key ``"generalist"``.

    Returns the record that was written.
    """
    data_dir = Path(data_dir)
    ckpt_path = Path(ckpt_path)

    record: dict[str, Any] = {
        "ckpt_sha1": checkpoint_sha1(ckpt_path),
        "ckpt_path": str(ckpt_path),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    for out_key, metric_key in _RECORDED_METRICS.items():
        if metric_key in metrics:
            record[out_key] = float(metrics[metric_key])

    all_records = load_baselines(data_dir)
    all_records[_key(archetype)] = record

    data_dir.mkdir(parents=True, exist_ok=True)
    path = baselines_path(data_dir)
    with open(path, "w") as f:
        json.dump(all_records, f, indent=2, sort_keys=True)
        f.write("\n")
    return record


def get_baseline(
    data_dir: str | Path,
    archetype: int | str | None,
    ckpt_path: str | Path | None = None,
) -> dict[str, Any]:
    """One archetype's baseline record, verified against *ckpt_path*.

    Raises ``FileNotFoundError`` when no baseline has been recorded for this
    archetype, and :class:`BaselineMismatch` when *ckpt_path* is given and its
    SHA-1 differs from the recorded one.

    Both are hard errors on purpose.  The alternative — proceeding with a
    warning — is how a run ends up gated against a number produced by a
    different model, which is worse than having no gate at all because it looks
    like one.
    """
    records = load_baselines(data_dir)
    key = _key(archetype)
    if key not in records:
        raise FileNotFoundError(
            f"no IL baseline recorded for archetype {key} in "
            f"{baselines_path(data_dir)} — run the pipeline's stage 4c "
            f"(`ptcg_il.cli train --eval-only --record-baseline`) first"
        )
    record = records[key]

    if ckpt_path is not None:
        actual = checkpoint_sha1(ckpt_path)
        if actual != record.get("ckpt_sha1"):
            raise BaselineMismatch(
                f"IL baseline for archetype {key} was recorded from "
                f"{record.get('ckpt_path')} (sha1 {record.get('ckpt_sha1')}), but "
                f"{ckpt_path} has sha1 {actual}. The baseline does not describe "
                f"this checkpoint; re-record it before gating against it."
            )
    return record


def _key(archetype: int | str | None) -> str:
    """Archetype id as the string key JSON will round-trip unchanged."""
    if archetype is None or archetype == "":
        return "generalist"
    return str(archetype)


def record_ensemble_baseline(
    data_dir: str | Path,
    archetype_self: int | None,
    ckpt_paths: list[str],
    ensemble_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Record ensemble eval scores, SHA-1-pinned to all member checkpoints.

    The record key is ``"ens-N-a<id>"`` (or ``"ens-N-generalist"``) — both
    member count and archetype, so two archetypes with the same ensemble size
    do not overwrite each other.
    Combinined SHA-1 ensures the gate detects when any member checkpoint changes.
    """
    data_dir = Path(data_dir)

    # Combined SHA of all member checkpoints
    combined_sha = _combined_sha1(ckpt_paths)

    # Key includes both member count AND archetype — two different archetypes
    # with the same ensemble size must not overwrite each other.
    key = f"ens-{len(ckpt_paths)}-{_key(archetype_self)}"
    record: dict[str, Any] = {
        "type": "ensemble",
        "member_count": len(ckpt_paths),
        "member_paths": [str(Path(p).resolve()) for p in ckpt_paths],
        "combined_sha1": combined_sha,
    }
    # Copy the recorded metrics
    for rec_key, metric_key in _RECORDED_METRICS.items():
        record[rec_key] = ensemble_metrics.get(metric_key, 0.0)
    record["recorded_at"] = datetime.now(timezone.utc).isoformat()
    if archetype_self is not None:
        record["archetype_self"] = archetype_self

    all_records = load_baselines(data_dir)
    all_records[key] = record

    data_dir.mkdir(parents=True, exist_ok=True)
    path = baselines_path(data_dir)
    with open(path, "w") as f:
        json.dump(all_records, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("Ensemble baseline recorded to %s", path)
    return record


def _combined_sha1(paths: list[str]) -> str:
    """SHA-1 of the concatenated SHA-1s of all checkpoint files."""
    h = hashlib.sha1()
    for p in sorted(paths):
        with open(p, "rb") as f:
            h.update(hashlib.sha1(f.read()).digest())
    return h.hexdigest()[:12]
