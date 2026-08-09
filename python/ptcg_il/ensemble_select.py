"""Pick the best *K*-member subset of a trained ensemble.

Kaggle caps how many checkpoints a submission may carry, so a 10-seed ensemble
has to be cut down.  Ranking members by their individual score and keeping the
top *K* is the obvious move and the wrong one: an ensemble gains from members
that are *decorrelated*, so the best 7 individually is not the best 7 together.
This module does greedy forward selection instead — start empty, repeatedly add
whichever remaining member most improves the **ensemble's** score.

The reason that is affordable: ``val/top1_nontrivial`` is computed entirely from
the single-select branch of :func:`ptcg_il.train.eval.offline_eval`, one
``argmax`` over one forward pass per row, and :class:`~ptcg_il.ensemble.
EnsemblePolicy` combines members as ``log(mean(softmax(masked logits)))``.  Log
is monotone and the ``+1e-10`` is uniform, so the ensemble's arg-max is the
arg-max of the *mean of member probabilities*, and every subset's score is
recoverable from per-member probabilities cached once.  Greedy over all M
members is then ``M(M+1)/2`` mean-and-argmax passes over a small array instead
of ``M(M+1)/2`` evaluations of the model — 55 numpy operations rather than 55
GPU passes for M=10, and the numbers are identical, not approximated.

That equivalence is specific to this metric.  The multi-select metrics come from
``select_multi``, where each member's msgru advances on the *ensemble's* chosen
option, so a subset's multi-select behaviour genuinely is not a function of the
members' independent outputs.  Nothing here reconstructs those; the chosen
subset is re-evaluated for real, on the held-out split, by the caller.

The order is fit on **val**.  Choosing the subset that maximises a *test* score
makes that test score a selection target rather than a held-out claim, and the
recorded number would then be optimistic by an unknown amount while looking like
every other baseline in the file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# The metric greedy maximises.  Named once here because it appears in the
# recorded artifact, and a reader of `il_baselines.json` has to be able to tell
# which number the ordering was fit on.
SELECTION_METRIC = "nontrivial_top1"


def collect_member_probs(
    members: Sequence[Any],
    loader: Iterable[dict],
    device: Any,
    *,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-member option probabilities on the rows ``top1_nontrivial`` scores.

    Returns ``(probs, targets)`` with ``probs`` of shape ``[M, N, O_MAX]``
    (float32) and ``targets`` of shape ``[N]`` (int64).

    The row filter is exactly the one in ``offline_eval``: single-select
    (``maxCount == 1``) *and* non-trivial (more than one legal option).  Rows
    whose target is ``-1`` (a declined single-select) are **kept**, because
    ``offline_eval`` keeps them too — no predicted index can equal ``-1``, so
    they sit in the denominator as permanent misses, and dropping them here
    would make every score computed from this cache disagree with the recorded
    baselines by a constant.
    """
    import torch

    if len(members) == 0:
        raise ValueError("collect_member_probs needs at least one member")

    device_type = getattr(device, "type", "cpu")
    if device_type not in ("cuda", "cpu"):
        device_type = "cpu"
    use_amp = device_type == "cuda"

    was_training = [m.training for m in members]
    for m in members:
        m.eval()

    per_member: list[list[np.ndarray]] = [[] for _ in members]
    targets: list[np.ndarray] = []
    n_rows = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch_gpu = {
                    k: v.to(device, non_blocking=(device_type == "cuda"))
                    for k, v in batch.items()
                    if k not in ("encoder_padding_mask",)
                }

                opt_mask = batch_gpu["opt_mask"]
                rows = (batch_gpu["maxCount"] == 1) & (opt_mask.sum(dim=-1) > 1)
                if not bool(rows.any()):
                    continue

                row_mask = opt_mask[rows]
                targets.append(
                    batch_gpu["action_idx"][rows, 0].detach().cpu().numpy().astype(np.int64)
                )
                n_rows += int(rows.sum().item())

                for i, member in enumerate(members):
                    if use_amp:
                        ctx = torch.amp.autocast(device_type, dtype=torch.bfloat16)
                    else:
                        from contextlib import nullcontext

                        ctx = nullcontext()
                    with ctx:
                        logits, _value, _history = member(batch_gpu)
                    # float() before softmax: the ensemble's own combination runs
                    # softmax outside autocast's bf16 list, and a bf16 softmax
                    # here would put ~3 decimal digits behind a mean whose ties
                    # decide the arg-max.
                    logits = logits[rows].float().masked_fill(~row_mask, -1e9)
                    probs = torch.softmax(logits, dim=-1)
                    per_member[i].append(probs.detach().cpu().numpy().astype(np.float32))
    finally:
        for m, t in zip(members, was_training):
            m.train(t)

    if n_rows == 0:
        raise ValueError(
            "no non-trivial single-select rows in this split — greedy selection "
            "would rank every subset identically at 0.0. Check --archetype-self "
            "and --eval-split point at a split that has data."
        )

    probs = np.stack([np.concatenate(chunks, axis=0) for chunks in per_member], axis=0)
    return probs, np.concatenate(targets, axis=0)


def subset_accuracy(probs: np.ndarray, targets: np.ndarray, subset: Sequence[int]) -> float:
    """``top1_nontrivial`` of the ensemble made of *subset*, from cached probs."""
    if len(subset) == 0:
        raise ValueError("subset_accuracy needs at least one member")
    mean_probs = probs[list(subset)].mean(axis=0)
    return float((mean_probs.argmax(axis=-1) == targets).mean())


def greedy_order(
    probs: np.ndarray, targets: np.ndarray
) -> tuple[list[int], list[float]]:
    """Greedy forward selection over all members.

    Returns ``(order, scores)``: *order* is every member index, best-first, and
    ``scores[i]`` is the ensemble's ``top1_nontrivial`` using ``order[:i+1]``.

    The full order is computed rather than stopping at *K* so one recorded
    ordering answers ``--ensemble-top`` for any *K*, and so ``scores`` shows
    where the ensemble actually peaks — which is the number that says whether
    dropping members costs anything.

    Ties go to the lowest member index, which makes the result reproducible
    across runs rather than dependent on dict ordering.
    """
    if probs.ndim != 3:
        raise ValueError(f"probs must be [M, N, O], got shape {probs.shape}")
    n_members, n_rows, _ = probs.shape
    if n_rows != targets.shape[0]:
        raise ValueError(
            f"probs has {n_rows} rows but targets has {targets.shape[0]}"
        )
    if n_rows == 0:
        raise ValueError("greedy_order needs at least one scored row")

    remaining = list(range(n_members))
    running = np.zeros(probs.shape[1:], dtype=np.float32)
    order: list[int] = []
    scores: list[float] = []

    while remaining:
        best_j, best_acc = -1, -1.0
        for j in remaining:  # ascending, so `>` leaves ties with the lowest index
            acc = float(((running + probs[j]).argmax(axis=-1) == targets).mean())
            if acc > best_acc:
                best_acc, best_j = acc, j
        running += probs[best_j]
        remaining.remove(best_j)
        order.append(best_j)
        scores.append(best_acc)

    return order, scores


def build_selection(
    order: Sequence[int],
    scores: Sequence[float],
    member_paths: Sequence[str],
    *,
    split: str,
    n_rows: int,
) -> dict[str, Any]:
    """The ``selection`` block written into the ensemble's baseline record."""
    from datetime import datetime, timezone

    return {
        "method": "greedy_forward",
        "metric": SELECTION_METRIC,
        "split": split,
        "n_rows": int(n_rows),
        "order": [int(i) for i in order],
        "order_paths": [str(Path(member_paths[i]).resolve()) for i in order],
        "scores": [float(s) for s in scores],
        "best_k": int(np.argmax(scores)) + 1 if len(scores) else 0,
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }


class SelectionUnavailable(RuntimeError):
    """No usable recorded selection covers the requested checkpoints."""


def select_from_baselines(
    ckpt_paths: Sequence[str],
    k: int,
    data_dir: str | Path,
    *,
    verify_sha1: bool = True,
) -> list[str]:
    """The best *k* of *ckpt_paths*, read from a recorded greedy ordering.

    Raises :class:`SelectionUnavailable` when no ensemble record in
    ``il_baselines.json`` holds a ``selection`` for exactly this member set, and
    :class:`~ptcg_il.baselines.BaselineMismatch` when a member's file no longer
    hashes to what the ordering was fit on.

    Both are hard errors.  A silent fallback to "the first *k* paths the glob
    happened to expand to" produces a submission that looks selected and is not,
    and nothing downstream would ever contradict it.
    """
    from ptcg_il.baselines import BaselineMismatch, checkpoint_sha1, load_baselines

    if k < 1:
        raise ValueError(f"--top must be at least 1, got {k}")
    if k > len(ckpt_paths):
        raise ValueError(
            f"asked for the top {k} of only {len(ckpt_paths)} checkpoints"
        )

    resolved = [str(Path(p).resolve()) for p in ckpt_paths]
    # Sorted *lists*, not sets: a record can legitimately hold the same path
    # twice (a glob expanded twice produces one), and a set comparison would
    # match a 20-entry record against 10 requested paths, then slice an order
    # indexing into the wrong list and hand back duplicate members.
    wanted = sorted(resolved)

    records = load_baselines(data_dir)
    match = None
    for key, rec in sorted(records.items()):
        if rec.get("type") != "ensemble" or not rec.get("selection"):
            continue
        if sorted(rec.get("member_paths") or []) == wanted:
            match = (key, rec)
            break

    if match is None:
        from ptcg_il.baselines import baselines_path

        raise SelectionUnavailable(
            f"no ensemble record in {baselines_path(data_dir)} carries a greedy "
            f"selection over exactly these {len(resolved)} checkpoints. Record "
            "one with:\n"
            "  cd python && uv run python -m ptcg_il.cli train --eval-only \\\n"
            "      --data-dir data --archetype-self <N> --eval-split val \\\n"
            f"      --ensemble-select {k} " + " ".join(f"--ckpt {p}" for p in resolved[:2])
            + " ..."
        )

    key, rec = match
    order = rec["selection"]["order"]
    member_paths = rec["member_paths"]
    members = rec.get("members") or []

    if verify_sha1 and members:
        by_path = {m["path"]: m for m in members}
        for path in resolved:
            recorded = by_path.get(path)
            if recorded is None or "ckpt_sha1" not in recorded:
                continue
            actual = checkpoint_sha1(path)
            if actual != recorded["ckpt_sha1"]:
                raise BaselineMismatch(
                    f"{path} has sha1 {actual[:12]} but the selection in {key} "
                    f"was fit on {recorded['ckpt_sha1'][:12]}. The recorded "
                    "ordering describes different weights — re-run "
                    "--ensemble-select before packaging a submission."
                )

    chosen = [member_paths[i] for i in order[:k]]
    logger.info(
        "Selected %d of %d members from %s (greedy on %s, %s=%.4f)",
        k, len(member_paths), key,
        rec["selection"].get("split", "?"), SELECTION_METRIC,
        rec["selection"]["scores"][k - 1] if len(rec["selection"]["scores"]) >= k else float("nan"),
    )
    return chosen


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ptcg_il.ensemble_select",
        description="Print the best K of a set of ensemble-member checkpoints, "
                    "using the greedy ordering recorded in il_baselines.json.",
    )
    parser.add_argument("--data-dir", default="data",
                        help="Directory holding il_baselines.json")
    parser.add_argument("--top", type=int, required=True,
                        help="How many members to keep")
    parser.add_argument("--ckpt", action="append", default=[], required=True,
                        help="Candidate checkpoint path (repeat per member)")
    parser.add_argument("--no-verify-sha1", action="store_true",
                        help="Skip re-hashing each checkpoint against the "
                             "ordering it was fit on. Only for a dry run.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    try:
        chosen = select_from_baselines(
            args.ckpt, args.top, args.data_dir,
            verify_sha1=not args.no_verify_sha1,
        )
    except Exception as e:  # noqa: BLE001 — this is a CLI boundary
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    for path in chosen:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
