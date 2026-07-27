"""Offline eval: per-context accuracy, value MSE/AUC (C.8).

Returns a flat dict of metrics designed to be logged directly to W&B.
Macro-averaged top-1 across ``sel_ctx`` is the early-stop criterion.

No live-engine eval here — that lives in a separate process-pool harness.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ptcg_il.model.policy import Policy


@torch.no_grad()
def offline_eval(
    policy: Policy,
    loader: DataLoader,
    device: torch.device,
    *,
    ema: Any = None,
    lambda_v: float = 0.5,
    label_smoothing: float = 0.05,
    max_batches: int | None = None,
    belief: bool = False,
) -> dict[str, float | list]:
    """Run offline metrics on the val split (C.8).

    Metrics (returned as a flat dict):
    - ``val/top1_macro``, ``val/top1_micro``, ``val/top3_micro``
    - ``val/top1_by_sel_type``, ``val/top1_by_sel_ctx`` — lists for W&B tables
    - ``val/multiselect_exact_set``, ``val/multiselect_perpick_top1``
    - ``val/value_mse``, ``val/value_auc``
    - ``val/best_top1_macro`` (placeholder, filled by loop)

    Parameters
    ----------
    policy : Policy
        Model to evaluate (switches to eval mode, restored after).
    loader : DataLoader
        Val split loader.
    device : torch.device
    ema : _EMA or None
        If provided, apply EMA weights before eval.
    lambda_v : float
        Value loss weight (for reporting consistency).
    label_smoothing : float
        Unused in eval; accepted for API compatibility.
    max_batches : int or None
        Cap eval to the first N batches (useful for testing).

    Returns
    -------
    dict
        Flat metrics dict.
    """
    was_training = policy.training
    policy.eval()

    # Save original weights if using EMA
    orig_weights = None
    if ema is not None:
        orig_weights = {n: p.data.clone() for n, p in policy.named_parameters()}
        ema.apply(policy)

    device_type = device.type if device.type in ("cuda", "cpu") else "cpu"
    use_amp = device_type == "cuda"

    # Accumulators
    total_correct_top1 = 0
    total_correct_top3 = 0
    total_samples = 0
    total_value_mse_sum = 0.0
    total_value_mse_n = 0
    all_values: list[float] = []
    all_targets: list[float] = []

    per_sel_type: dict[int, tuple[int, int]] = {}  # type → (correct, total)
    per_sel_ctx: dict[int, tuple[int, int]] = {}   # ctx → (correct, total)

    # Non-trivial = more than one legal option, i.e. the decision points where
    # the policy actually has a choice to get right.  Tracked separately because
    # forced (single-option) decisions score a free 1.0 and, together with the
    # macro average over near-degenerate sel_ctx buckets, can make a barely
    # better-than-baseline model look strong.
    nontrivial_correct_top1 = 0
    nontrivial_correct_top3 = 0
    nontrivial_samples = 0
    nontrivial_firstlegal_correct = 0

    total_multi_exact = 0
    total_multi_perpick_correct = 0
    total_multi_perpick_total = 0
    total_multi_samples = 0

    # (weighted sum, row count) per belief term; only touched when belief=True.
    belief_acc: dict[str, list[float]] = {
        k: [0.0, 0.0] for k in ("arch", "deck", "hidden", "hand")
    }

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch_gpu: dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if k in ("encoder_padding_mask",):
                continue
            batch_gpu[k] = v.to(device, non_blocking=(device.type == "cuda"))

        with (
            torch.amp.autocast(device_type, dtype=torch.bfloat16)
            if use_amp else nullcontext()
        ):
            if belief:
                logits, value, _history_h, belief_preds = policy.forward_with_belief(batch_gpu)
                _accum_belief(belief_acc, belief_preds, batch_gpu)
            else:
                logits, value, _history_h = policy(batch_gpu)  # [B, O], [B], [B, D]

        maxcount = batch_gpu["maxCount"]
        single_mask = (maxcount == 1)
        multi_mask = ~single_mask

        # --- Single-select accuracy ---
        if single_mask.any():
            single_logits = logits[single_mask]
            single_targets = batch_gpu["action_idx"][single_mask, 0]
            single_sel_type = batch_gpu["sel_type"][single_mask].cpu().numpy()
            single_sel_ctx = batch_gpu["sel_ctx"][single_mask].cpu().numpy()

            top1_pred = single_logits.argmax(dim=-1)  # [N_single]
            top3_pred = single_logits.topk(min(3, single_logits.shape[-1]), dim=-1).indices  # [N_single, 3]

            correct_top1 = (top1_pred == single_targets).sum().item()
            correct_top3 = (top3_pred == single_targets.unsqueeze(-1)).any(dim=-1).sum().item()

            total_correct_top1 += correct_top1
            total_correct_top3 += correct_top3
            total_samples += single_mask.sum().item()

            # Non-trivial subset + "always pick the first legal option" baseline
            single_optmask = batch_gpu["opt_mask"][single_mask]      # [N_single, O]
            n_opt = single_optmask.sum(dim=-1)                       # [N_single]
            nt = n_opt > 1
            if nt.any():
                nt_top1 = (top1_pred[nt] == single_targets[nt])
                nt_top3 = (top3_pred[nt] == single_targets[nt].unsqueeze(-1)).any(dim=-1)
                first_legal = single_optmask[nt].float().argmax(dim=-1)
                nontrivial_correct_top1 += nt_top1.sum().item()
                nontrivial_correct_top3 += nt_top3.sum().item()
                nontrivial_firstlegal_correct += (
                    first_legal == single_targets[nt]
                ).sum().item()
                nontrivial_samples += int(nt.sum().item())

            # Per-type/count accum
            correct_arr = (top1_pred == single_targets).cpu().numpy()
            for i, st in enumerate(single_sel_type):
                st = int(st)
                c, t = per_sel_type.get(st, (0, 0))
                per_sel_type[st] = (c + int(correct_arr[i]), t + 1)
            for i, sc in enumerate(single_sel_ctx):
                sc = int(sc)
                c, t = per_sel_ctx.get(sc, (0, 0))
                per_sel_ctx[sc] = (c + int(correct_arr[i]), t + 1)

        # --- Multi-select accuracy ---
        if multi_mask.any():
            # Greedy decode
            from ptcg_il.model.policy import select_multi
            multi_preds = select_multi(policy, {
                k: v[multi_mask] for k, v in batch_gpu.items()
            })  # [N_multi, O_MAX]
            multi_targets = batch_gpu["action_idx"][multi_mask]  # [N_multi, O_MAX]
            multi_len = batch_gpu["action_len"][multi_mask]      # [N_multi]

            for i in range(multi_mask.sum().item()):
                n = int(multi_len[i].item())
                pred = multi_preds[i, :n].tolist()
                tgt = multi_targets[i, :n].tolist()
                stop_col = int(batch_gpu["stop_column"][multi_mask][i].item())
                for step_idx, (p, t) in enumerate(zip(pred, tgt)):
                    if stop_col >= 0 and t == stop_col:
                        # STOP step: model should STOP (-2)
                        total_multi_perpick_correct += 1 if p == -2 else 0
                    else:
                        total_multi_perpick_correct += 1 if p == t else 0
                total_multi_perpick_total += n
                if sorted(pred) == sorted(tgt):
                    total_multi_exact += 1
                total_multi_samples += 1

        # --- Value metrics ---
        all_values.extend(value.detach().cpu().tolist())
        all_targets.extend(batch_gpu["value_target"].detach().cpu().tolist())
        # MSE
        v_mse = F.mse_loss(value, batch_gpu["value_target"], reduction="sum")
        total_value_mse_sum += v_mse.item()
        total_value_mse_n += value.shape[0]

    # Restore
    if orig_weights is not None:
        for n, p in policy.named_parameters():
            p.data.copy_(orig_weights[n])
    policy.train(was_training)

    # --- Compile results ---
    metrics: dict[str, Any] = {}

    # Top-1 / top-3
    if total_samples > 0:
        metrics["val/top1_micro"] = total_correct_top1 / total_samples
        metrics["val/top3_micro"] = total_correct_top3 / total_samples

        # Per sel_type
        type_accs = []
        for t in sorted(per_sel_type):
            c, n = per_sel_type[t]
            acc = c / n if n > 0 else 0.0
            type_accs.append((t, acc, n))
            metrics[f"val/top1@sel_type_{t}"] = acc
        metrics["val/top1_by_sel_type"] = type_accs  # for W&B table

        # Per sel_ctx
        ctx_accs = []
        for ctx in sorted(per_sel_ctx):
            c, n = per_sel_ctx[ctx]
            acc = c / n if n > 0 else 0.0
            ctx_accs.append((ctx, acc, n))
            metrics[f"val/top1@sel_ctx_{ctx}"] = acc
        metrics["val/top1_by_sel_ctx"] = ctx_accs  # for W&B table

        # Macro avg over sel_ctx.  Kept for continuity, but note it weights a
        # 3-sample forced-choice bucket the same as a 400-sample one, so it runs
        # well above the micro numbers — do not read it as "accuracy".
        n_ctx = len(per_sel_ctx)
        metrics["val/top1_macro"] = (
            sum(a for _, a, _ in ctx_accs) / n_ctx if n_ctx > 0 else 0.0
        )

        # Non-trivial micro metrics + baseline lift (early-stop criterion)
        if nontrivial_samples > 0:
            nt_top1 = nontrivial_correct_top1 / nontrivial_samples
            nt_baseline = nontrivial_firstlegal_correct / nontrivial_samples
            metrics["val/top1_nontrivial"] = nt_top1
            metrics["val/top3_nontrivial"] = nontrivial_correct_top3 / nontrivial_samples
            metrics["val/top1_nontrivial_firstlegal"] = nt_baseline
            metrics["val/top1_nontrivial_lift"] = nt_top1 - nt_baseline
            metrics["val/n_nontrivial"] = nontrivial_samples
        else:
            metrics["val/top1_nontrivial"] = 0.0
            metrics["val/top3_nontrivial"] = 0.0
            metrics["val/top1_nontrivial_firstlegal"] = 0.0
            metrics["val/top1_nontrivial_lift"] = 0.0
            metrics["val/n_nontrivial"] = 0
    else:
        metrics["val/top1_micro"] = 0.0
        metrics["val/top3_micro"] = 0.0
        metrics["val/top1_by_sel_type"] = []
        metrics["val/top1_by_sel_ctx"] = []
        metrics["val/top1_macro"] = 0.0
        metrics["val/top1_nontrivial"] = 0.0
        metrics["val/top3_nontrivial"] = 0.0
        metrics["val/top1_nontrivial_firstlegal"] = 0.0
        metrics["val/top1_nontrivial_lift"] = 0.0
        metrics["val/n_nontrivial"] = 0

    # Multi-select
    if total_multi_samples > 0:
        metrics["val/multiselect_exact_set"] = total_multi_exact / total_multi_samples
        metrics["val/multiselect_perpick_top1"] = (
            total_multi_perpick_correct / total_multi_perpick_total
            if total_multi_perpick_total > 0
            else 0.0
        )
    else:
        metrics["val/multiselect_exact_set"] = 0.0
        metrics["val/multiselect_perpick_top1"] = 0.0

    # Value
    if total_value_mse_n > 0:
        metrics["val/value_mse"] = total_value_mse_sum / total_value_mse_n
    else:
        metrics["val/value_mse"] = 0.0

    # AUC (simple — treat value sign as binary classifier)
    if len(all_values) > 1 and len(set(all_targets)) > 1:
        metrics["val/value_auc"] = _compute_auc(all_values, all_targets)
    else:
        metrics["val/value_auc"] = 0.5  # chance

    metrics.update(_value_health(all_values, all_targets))

    metrics["val/best_top1_macro"] = 0.0  # placeholder, filled by loop

    if belief:
        metrics.update(_belief_metrics(belief_acc))

    return metrics


def _accum_belief(
    acc: dict[str, list[float]],
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> None:
    """Accumulate belief quality over one batch, in place.

    The card heads are scored by *hit mass* -- how much of the predicted
    distribution lands on cards the opponent actually holds -- rather than by
    top-1 accuracy.  A decklist is a multiset of ~20 distinct cards, so "did
    the argmax match" answers almost nothing; hit mass is the quantity the
    determinizing sampler cares about, because it is the fraction of draws it
    will get right.
    """
    valid = batch["bel_valid"].bool()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return

    for key, tgt_key, mask in (
        ("deck", "bel_deck", valid),
        ("hidden", "bel_hidden", valid),
        ("hand", "bel_hand", valid & batch["bel_hand_valid"].bool()),
    ):
        n = int(mask.sum())
        if n == 0:
            continue
        p = torch.softmax(preds[key].float(), dim=-1)
        hit = (p * (batch[tgt_key] > 0).to(p.dtype)).sum(-1)
        acc[key][0] += float(hit[mask].sum())
        acc[key][1] += n

    arch_t = batch["bel_arch"].long()
    arch_mask = valid & (arch_t >= 0)
    n_arch = int(arch_mask.sum())
    if n_arch and preds["arch"].shape[-1] > 1:
        correct = preds["arch"].argmax(-1) == arch_t
        acc["arch"][0] += float(correct[arch_mask].sum())
        acc["arch"][1] += n_arch


def _belief_metrics(acc: dict[str, list[float]]) -> dict[str, float]:
    """Finalize :func:`_accum_belief` sums into logged scalars."""
    out: dict[str, float] = {}
    names = {"arch": "val/belief_arch_top1", "deck": "val/belief_deck_mass",
             "hidden": "val/belief_hidden_mass", "hand": "val/belief_hand_mass"}
    for key, name in names.items():
        total, n = acc[key]
        out[name] = total / n if n else 0.0
        out[f"{name}_n"] = float(n)
    return out


def _value_health(predictions: list[float], targets: list[float]) -> dict[str, float]:
    """Diagnostics for a collapsed value head (RL_SPEC §3).

    A dead critic is not visible in ``value_mse``: predicting a constant ~0
    against ±1 labels scores ~1.0, which looks like a merely-mediocre head
    rather than one carrying no signal at all.  ``value_std`` is what
    distinguishes them, and ``value_corr`` is the quantity R1 gates on
    (≥ 0.35 to proceed).

    Returns zeros when there is nothing to measure, and a correlation of 0.0
    when either side is constant — ``np.corrcoef`` would return NaN there, and a
    NaN silently poisons every downstream comparison.
    """
    if len(predictions) < 2:
        return {"val/value_std": 0.0, "val/value_corr": 0.0, "val/value_sign_agree": 0.0}

    pred = np.asarray(predictions, dtype=np.float64)
    targ = np.asarray(targets, dtype=np.float64)

    pred_std = float(pred.std())
    targ_std = float(targ.std())
    if pred_std <= 0.0 or targ_std <= 0.0:
        corr = 0.0
    else:
        corr = float(((pred - pred.mean()) * (targ - targ.mean())).mean() / (pred_std * targ_std))

    # Sign agreement excludes zero-valued targets, which have no sign to match.
    signed = targ != 0.0
    sign_agree = (
        float((np.sign(pred[signed]) == np.sign(targ[signed])).mean()) if signed.any() else 0.0
    )

    return {
        "val/value_std": pred_std,
        "val/value_corr": corr,
        "val/value_sign_agree": sign_agree,
    }


def _compute_auc(predictions: list[float], targets: list[float]) -> float:
    """Simple AUC for binary classification (value sign prediction).

    Uses the predicted value directly as score (not thresholded) against
    binary targets (+1/-1 mapped to 1/0).
    """
    pred_arr = np.array(predictions)
    tgt_arr = np.array(targets)
    # Map targets to 0/1: +1 → 1, -1 → 0
    y_true = (tgt_arr > 0).astype(int)
    y_score = pred_arr  # values already in (-1, 1)

    # Sort by score descending
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]

    n_pos = y_true_sorted.sum()
    n_neg = len(y_true_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate

    # TPR = TP / P, FPR = FP / N
    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0

    for i, y in enumerate(y_true_sorted):
        if y == 1:
            tp += 1
        else:
            fp += 1
            auc += (tp / n_pos) * ((fp - prev_fpr) / n_neg)
            prev_fpr = fp

    # Final segment
    auc += (tp / n_pos) * ((n_neg - prev_fpr) / n_neg)
    return float(auc)
