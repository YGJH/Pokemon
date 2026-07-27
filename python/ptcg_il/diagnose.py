"""Why is the IL policy's accuracy low?  A battery of diagnostics.

The model runs, produces legal actions, and beats the first-legal baseline — and
still tops out around 0.50 non-trivial top-1.  "It underperforms" has at least
six distinct causes, and they call for opposite responses, so the point of this
module is to tell them apart:

1. **The label ceiling.** Experts disagree with each other. Where two identical
   states carry two different expert actions, no model can get both right. If
   the ceiling is 0.60, a score of 0.50 is 83% of achievable and the model is
   nearly done; if the ceiling is 0.95, it is barely half-trained. Everything
   else is meaningless without this number.
2. **Underfitting vs overfitting.** Train and val accuracy together say which.
   Low-and-equal means undertrained or under-capacity; a wide gap means
   memorisation.
3. **Dead features.** Permute one feature group across the batch and re-measure:
   a group whose ablation costs nothing is a group the model ignores. This
   repo's `opt_card_id` was silently PAD for its entire history (RL_SPEC §6.4.1)
   and nothing caught it, because a model that ignores an input still trains,
   still converges, and still beats the trivial baseline.
4. **Where the errors are.** Accuracy by decision type and by option count. A
   model that is fine on 2-option decisions and chance-level on 20-option ones
   has a different problem from one that is uniformly mediocre.
5. **Degenerate strategies.** Predicting index 0, or the most common index,
   scores surprisingly well and looks like learning.
6. **Feature health.** OOV cards and PAD option identity, re-measured on the
   data the model was actually fitted to.

Usage::

    uv run python -m ptcg_il.diagnose --data-dir data \\
        --ckpt checkpoints_a0/ckpt-best.pt --archetype 0

Every section prints what it measured and how many rows it measured it on. A
diagnostic that quietly examined nothing is worse than no diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Feature groups for the ablation sweep.  Grouped by what they *mean*, not by
# tensor shape, so a zero-cost group names a concept the model is ignoring.
ABLATION_GROUPS: dict[str, tuple[str, ...]] = {
    "option card identity": ("opt_card_id",),
    "option type": ("opt_type",),
    "option target refs": ("opt_src_idx", "opt_tgt_idx"),
    "option scalars": ("opt_scalar",),
    "option attack": ("opt_attack_idx",),
    "board pokemon identity": ("poke_card_id",),
    "board pokemon features": ("poke_feat",),
    "hand identity": ("hand_card_id",),
    "hand features": ("hand_feat",),
    "global state (cls)": ("cls_feat",),
    "per-player summary": ("sum_feat",),
    "discard / prizes": ("discard_ids", "prize_ids"),
    "game log (belief)": ("log_feat",),
    "stadium / context": ("stadium_card_id", "context_card_id", "effect_card_id"),
}


@dataclass
class Report:
    """Collected findings, in the order they should be read."""

    sections: list[tuple[str, list[str]]] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def add(self, title: str, lines: list[str]) -> None:
        self.sections.append((title, lines))

    def verdict(self, text: str) -> None:
        self.verdicts.append(text)

    def render(self) -> str:
        out: list[str] = []
        for title, lines in self.sections:
            out.append(f"\n{'─' * 72}\n{title}\n{'─' * 72}")
            out.extend(lines)
        if self.verdicts:
            out.append(f"\n{'═' * 72}\nVERDICT\n{'═' * 72}")
            for i, v in enumerate(self.verdicts, 1):
                out.append(f"{i}. {v}")
        return "\n".join(out)


# ── 1. Label ceiling ────────────────────────────────────────────────────────


def state_fingerprint(batch: dict[str, np.ndarray], i: int) -> bytes:
    """A hash of everything that defines a decision, excluding the label.

    Two rows with the same fingerprint present the model with the same choice.
    Includes the option tensors, because the action is an *index into options* —
    two states with identical boards but differently-ordered options are not the
    same decision.
    """
    h = hashlib.blake2b(digest_size=16)
    for key in (
        "tok_type", "poke_card_id", "hand_card_id", "opt_type", "opt_card_id",
        "opt_src_idx", "opt_tgt_idx", "opt_mask", "sel_type", "sel_ctx",
        "minCount", "maxCount", "stadium_card_id",
    ):
        v = batch.get(key)
        if v is None:
            continue
        arr = np.asarray(v[i] if np.ndim(v) > 0 else v)
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.digest()


def label_ceiling(rows: list[tuple[bytes, int]]) -> dict[str, Any]:
    """Best achievable top-1 given that experts disagree on identical states.

    For each group of identical decisions, the most any deterministic policy can
    score is the frequency of the *modal* action. Summed over groups and divided
    by the row count, that is a hard upper bound on top-1 for this corpus.
    """
    groups: dict[bytes, Counter] = defaultdict(Counter)
    for fp, action in rows:
        groups[fp][action] += 1

    n_rows = len(rows)
    best = sum(c.most_common(1)[0][1] for c in groups.values())
    duplicated = {fp: c for fp, c in groups.items() if sum(c.values()) > 1}
    conflicted = {fp: c for fp, c in duplicated.items() if len(c) > 1}

    return {
        "n_rows": n_rows,
        "n_distinct_states": len(groups),
        "n_duplicated_states": len(duplicated),
        "n_conflicted_states": len(conflicted),
        "rows_in_conflicted": sum(sum(c.values()) for c in conflicted.values()),
        "ceiling_top1": best / n_rows if n_rows else 0.0,
    }


# ── 0. Feature health, without the model ────────────────────────────────────


def feature_health(data_dir: str | Path, split: str, max_shards: int = 3) -> dict[str, Any]:
    """Scan the raw shards for inputs that cannot carry information.

    Model-independent on purpose. A permutation ablation says "the model ignores
    this", which on an undertrained model is ambiguous — weak features and dead
    features look identical. A **zero-variance column is dead regardless of the
    model**, and that is a featurizer bug, not a training problem.

    Also re-measures the two rates RL_SPEC §6.4 cares about: cards that fell
    through to ``UNKNOWN_CARD`` (a vocab problem) and options left at
    ``PAD_CARD`` (~40% is correct post-R0b — deck slots and face-down prizes
    must stay PAD).
    """
    from ptcg_il.featurizer import PAD_CARD, UNKNOWN_CARD

    shards = sorted(Path(data_dir).glob(f"shards/{split}-*.npz"))[:max_shards]
    if not shards:
        return {}

    # Columns dead in *every* shard.  Intersecting rather than unioning matters:
    # a feature that only varies in one shard (a rare status condition, a card
    # that appears late in the corpus) is not dead, and unioning would report it
    # as such — turning a rare-but-real signal into a phantom featurizer bug.
    const_cols: dict[str, set[int]] = {}
    seen_keys: set[str] = set()
    stats: dict[str, Any] = {"shards": [s.name for s in shards], "n_rows_scanned": 0}
    unknown = pad = opt_total = 0
    state_unknown = state_total = 0

    for path in shards:
        d = np.load(path, allow_pickle=True)
        for key in ("cls_feat", "sum_feat", "poke_feat", "hand_feat", "opt_scalar", "log_feat"):
            if key not in d:
                continue
            arr = np.asarray(d[key], dtype=np.float32)
            flat = arr.reshape(-1, arr.shape[-1])
            dead = set(np.where(flat.std(axis=0) == 0)[0].tolist())
            const_cols[key] = dead if key not in seen_keys else (const_cols[key] & dead)
            seen_keys.add(key)
        stats["n_rows_scanned"] += int(np.asarray(d["maxCount"]).shape[0])

        if "opt_card_id" in d and "opt_mask" in d:
            m = np.asarray(d["opt_mask"], dtype=bool)
            ids = np.asarray(d["opt_card_id"])
            opt_total += int(m.sum())
            pad += int(((ids == PAD_CARD) & m).sum())
            unknown += int(((ids == UNKNOWN_CARD) & m).sum())
        for key in ("poke_card_id", "hand_card_id"):
            if key in d:
                ids = np.asarray(d[key])
                state_total += int((ids != PAD_CARD).sum())
                state_unknown += int((ids == UNKNOWN_CARD).sum())

        if "log_mask" in d:
            lm = np.asarray(d["log_mask"], dtype=bool)
            stats["log_mask_occupancy"] = float(lm.mean())

    stats["constant_columns"] = {k: sorted(v) for k, v in const_cols.items() if v}
    stats["opt_pad_rate"] = pad / opt_total if opt_total else 0.0
    stats["opt_unknown_rate"] = unknown / opt_total if opt_total else 0.0
    stats["state_unknown_rate"] = state_unknown / state_total if state_total else 0.0
    return stats


# ── 2-6. Model-side diagnostics ─────────────────────────────────────────────


def _first_legal(opt_mask: np.ndarray) -> np.ndarray:
    return opt_mask.astype(np.float32).argmax(axis=-1)


class Diagnoser:
    """Runs the model-side checks over a loader."""

    def __init__(self, policy, device, max_batches: int | None = None):
        self.policy = policy
        self.device = device
        self.max_batches = max_batches

    def _forward(self, batch):
        import torch

        gpu = {k: (v.to(self.device) if torch.is_tensor(v) else v)
               for k, v in batch.items()}
        with torch.no_grad():
            logits, value, _h = self.policy(gpu)
        return gpu, logits

    def collect(self, loader) -> dict[str, np.ndarray]:
        """One pass: predictions, targets, and every slicing key we need."""
        import torch

        acc: dict[str, list] = defaultdict(list)
        for bi, batch in enumerate(loader):
            if self.max_batches is not None and bi >= self.max_batches:
                break
            gpu, logits = self._forward(batch)

            single = (gpu["maxCount"] == 1)
            if not single.any():
                continue

            lg = logits[single]
            opt_mask = gpu["opt_mask"][single]
            masked = lg.masked_fill(~opt_mask, float("-inf"))
            probs = torch.softmax(masked.float(), dim=-1)

            top3 = masked.topk(min(3, masked.shape[-1]), dim=-1).indices
            acc["pred"].append(masked.argmax(-1).cpu().numpy())
            acc["top3"].append(top3.cpu().numpy())
            acc["target"].append(gpu["action_idx"][single, 0].cpu().numpy())
            acc["n_opts"].append(opt_mask.sum(-1).cpu().numpy())
            acc["sel_type"].append(gpu["sel_type"][single].cpu().numpy())
            acc["sel_ctx"].append(gpu["sel_ctx"][single].cpu().numpy())
            acc["first_legal"].append(_first_legal(opt_mask.cpu().numpy()))
            acc["maxprob"].append(probs.max(-1).values.cpu().numpy())
            ent = -(probs.clamp_min(1e-12).log() * probs).sum(-1)
            acc["entropy"].append(ent.cpu().numpy())

        if not acc:
            raise ValueError("no single-select rows found in the loader")
        return {k: np.concatenate(v) for k, v in acc.items()}

    def ablate(self, loader, groups: dict[str, tuple[str, ...]]) -> dict[str, float]:
        """Top-1 after permuting each feature group across the batch.

        Permutation (rather than zeroing) keeps each feature's marginal
        distribution intact and destroys only its association with the label, so
        a drop measures *use*, not distribution shift.
        """
        import torch

        batches = []
        for bi, batch in enumerate(loader):
            if bi >= (self.max_batches or 8):
                break
            batches.append(batch)
        if not batches:
            raise ValueError("no batches to ablate over")

        def score(mutate=None) -> float:
            correct = total = 0
            for batch in batches:
                b = dict(batch)
                if mutate is not None:
                    b = mutate(b)
                gpu, logits = self._forward(b)
                single = (gpu["maxCount"] == 1)
                nt = single & (gpu["opt_mask"].sum(-1) > 1)
                if not nt.any():
                    continue
                masked = logits[nt].masked_fill(~gpu["opt_mask"][nt], float("-inf"))
                pred = masked.argmax(-1)
                correct += int((pred == gpu["action_idx"][nt, 0]).sum())
                total += int(nt.sum())
            return correct / total if total else 0.0

        base = score()
        out = {"__baseline__": base}
        for name, keys in groups.items():
            present = [k for k in keys if k in batches[0]]
            if not present:
                continue

            def mutate(b, present=present):
                b = dict(b)
                n = b["maxCount"].shape[0]
                perm = torch.randperm(n)
                for k in present:
                    b[k] = b[k][perm].clone()
                return b

            out[name] = base - score(mutate)
        return out


# ── Report assembly ─────────────────────────────────────────────────────────


def _accuracy_table(res: dict[str, np.ndarray], key: str, label: str,
                    bucket=None) -> list[str]:
    correct = res["pred"] == res["target"]
    lines = [f"  {label:<24} {'n':>7} {'top-1':>8} {'first-legal':>12} {'lift':>8}"]
    vals = res[key] if bucket is None else bucket(res[key])
    for v in sorted(set(vals.tolist())):
        sel = vals == v
        n = int(sel.sum())
        if n < 30:
            continue
        acc = float(correct[sel].mean())
        fl = float((res["first_legal"][sel] == res["target"][sel]).mean())
        lines.append(f"  {str(v):<24} {n:>7} {acc:>8.4f} {fl:>12.4f} {acc - fl:>+8.4f}")
    return lines


def _opt_bucket(n_opts: np.ndarray) -> np.ndarray:
    edges = [1, 2, 3, 5, 9, 17, 33]
    labels = np.full(n_opts.shape, "33+", dtype=object)
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels[(n_opts >= lo) & (n_opts < hi)] = f"{lo}-{hi - 1}" if hi - 1 > lo else str(lo)
    return labels


def run_diagnosis(
    data_dir: str | Path,
    ckpt_path: str | Path,
    archetype: int | None,
    *,
    device: str | None = None,
    max_batches: int = 40,
    batch_size: int = 256,
    ablation_batches: int = 8,
    split: str = "val",
) -> Report:
    import torch
    from torch.utils.data import DataLoader

    from ptcg_il.model.policy import load_policy_state, policy_from_config
    from ptcg_il.train.checkpoint import load_checkpoint
    from ptcg_il.train.dataset import ShardDataset, collate_fn

    rep = Report()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = load_checkpoint(ckpt_path, device="cpu")
    config = ckpt.get("config") or {}
    if not config:
        raise SystemExit(
            f"{ckpt_path} has no 'config' record — retrain with the current "
            f"pipeline so the architecture does not have to be guessed."
        )
    policy = policy_from_config(config)
    load_policy_state(policy, ckpt["model_state_dict"])
    policy.to(dev).eval()

    def loader(sp: str, shuffle: bool = False):
        ds = ShardDataset(data_dir, split=sp, shuffle=shuffle, archetype_self=archetype)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    rep.add("MODEL", [
        f"  checkpoint : {ckpt_path}",
        f"  step       : {ckpt.get('step')}   <-- optimizer steps taken",
        f"  archetype  : {archetype if archetype is not None else 'generalist'}",
        f"  config     : {config}",
        f"  device     : {dev}",
    ])

    diag = Diagnoser(policy, dev, max_batches=max_batches)

    # ---- 0. Feature health (no model involved) ---------------------------
    # Scan the train split regardless of --split: it has the most shards, so the
    # "constant in every shard" intersection is strongest there, and it is the
    # data the model was actually fitted to.
    health = feature_health(data_dir, "train")
    rep.data["feature_health"] = health
    if health:
        lines = [
            f"  opt_card_id PAD rate     : {health['opt_pad_rate']:.4f}"
            "   (~0.40 is correct: deck slots and face-down prizes must stay PAD)",
            f"  opt_card_id UNKNOWN rate : {health['opt_unknown_rate']:.4f}"
            "   (should be ~0; anything else is a vocab/normalisation bug)",
            f"  state card UNKNOWN rate  : {health['state_unknown_rate']:.4f}",
        ]
        if "log_mask_occupancy" in health:
            lines.append(
                f"  game-log token occupancy : {health['log_mask_occupancy']:.4f}"
                "   (how full the belief module's input actually is)"
            )
        if health["constant_columns"]:
            lines += ["", "  ZERO-VARIANCE FEATURE COLUMNS (dead by construction):"]
            for key, cols in sorted(health["constant_columns"].items()):
                lines.append(f"    {key:<12} {len(cols):>3} dead columns: {cols}")
            lines.append(
                f"    (scanned {health['n_rows_scanned']} rows across "
                f"{len(health['shards'])} shard(s); a column counts as dead only "
                f"if it is constant in every one)"
            )
        else:
            lines += ["", "  No zero-variance feature columns."]
        rep.add("0. FEATURE HEALTH — can these inputs carry information at all?", lines)

        if health["constant_columns"]:
            total_dead = sum(len(c) for c in health["constant_columns"].values())
            rep.verdict(
                f"CONSTANT FEATURES: {total_dead} feature column(s) have zero "
                f"variance across the corpus — "
                + "; ".join(f"{k}{v}" for k, v in health["constant_columns"].items())
                + ". These are dead in the featurizer, not merely unused by the "
                "model, so no amount of training recovers them. Check whether the "
                "field is ever populated in the observation."
            )
        if health["opt_unknown_rate"] > 0.01:
            rep.verdict(
                f"VOCAB MISS: {health['opt_unknown_rate']:.1%} of legal options map "
                f"to UNKNOWN_CARD. `normalize_vocab` turns JSON string keys into "
                f"ints; skipping it makes every lookup miss silently."
            )

    # ---- 1. Label ceiling -------------------------------------------------
    rows: list[tuple[bytes, int]] = []
    ds = ShardDataset(data_dir, split=split, shuffle=False, archetype_self=archetype)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for bi, batch in enumerate(dl):
        if bi >= max_batches:
            break
        npb = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in batch.items()}
        for i in range(npb["maxCount"].shape[0]):
            if npb["maxCount"][i] != 1:
                continue
            rows.append((state_fingerprint(npb, i), int(npb["action_idx"][i, 0])))

    ceil = label_ceiling(rows)
    rep.data["ceiling"] = ceil
    dup_pct = 100.0 * ceil["n_duplicated_states"] / max(1, ceil["n_distinct_states"])
    rep.add("1. LABEL CEILING — how much of the gap is experts disagreeing?", [
        f"  single-select rows examined : {ceil['n_rows']}",
        f"  distinct decision states    : {ceil['n_distinct_states']}",
        f"  states seen more than once  : {ceil['n_duplicated_states']}  ({dup_pct:.1f}%)",
        f"  ...of those, with conflicting expert actions : {ceil['n_conflicted_states']}",
        f"  rows inside a conflicted state              : {ceil['rows_in_conflicted']}",
        "",
        f"  >>> CEILING on top-1 for this corpus: {ceil['ceiling_top1']:.4f}",
        "      No deterministic policy can beat this. Compare every accuracy",
        "      below against it, not against 1.0.",
    ])

    # ---- 2. Fit: train vs val --------------------------------------------
    res_val = diag.collect(loader(split))
    res_train = diag.collect(loader("train"))

    def _nt(res):
        sel = res["n_opts"] > 1
        return float((res["pred"][sel] == res["target"][sel]).mean()), int(sel.sum())

    val_nt, val_n = _nt(res_val)
    train_nt, train_n = _nt(res_train)
    gap = train_nt - val_nt
    rep.data["fit"] = {"train_nt_top1": train_nt, "val_nt_top1": val_nt, "gap": gap}
    rep.add("2. FIT — underfitting or overfitting?", [
        f"  train non-trivial top-1 : {train_nt:.4f}  (n={train_n})",
        f"  {split:<5} non-trivial top-1 : {val_nt:.4f}  (n={val_n})",
        f"  generalisation gap      : {gap:+.4f}",
        f"  headroom to ceiling     : {ceil['ceiling_top1'] - val_nt:+.4f}",
    ])
    if gap > 0.10:
        rep.verdict(
            f"OVERFITTING: train beats {split} by {gap:.3f}. More data, more "
            f"regularisation, or earlier stopping — not more steps."
        )
    elif train_nt < ceil["ceiling_top1"] - 0.10:
        rep.verdict(
            f"UNDERFITTING: the model scores {train_nt:.3f} on data it was "
            f"trained on, against a ceiling of {ceil['ceiling_top1']:.3f}. It has "
            f"not fitted the training set yet — train longer or increase capacity "
            f"before touching anything else."
        )

    # ---- 3. Feature ablation ---------------------------------------------
    abl_diag = Diagnoser(policy, dev, max_batches=ablation_batches)
    abl = abl_diag.ablate(loader(split), ABLATION_GROUPS)
    base = abl.pop("__baseline__")
    rep.data["ablation"] = abl
    lines = [f"  baseline non-trivial top-1 on this subset: {base:.4f}", "",
             f"  {'feature group':<26} {'top-1 drop when permuted':>26}"]
    for name, drop in sorted(abl.items(), key=lambda kv: -kv[1]):
        flag = "  <-- IGNORED" if drop <= 0.002 else ""
        lines.append(f"  {name:<26} {drop:>+26.4f}{flag}")
    dead = [n for n, d in abl.items() if d <= 0.002]
    underfitted = train_nt < ceil["ceiling_top1"] - 0.10
    if underfitted:
        lines += [
            "",
            "  NOTE: this model is underfitted (§2), so read the zero rows with",
            "  care — a model that has only learned the coarsest signals ignores",
            "  weak-but-real features and genuinely dead ones identically. Section",
            "  0 is the model-independent check; trust that one first, and re-run",
            "  this sweep once the model has actually fitted the training set.",
        ]
    rep.add("3. FEATURE ABLATION — which inputs does the model actually use?", lines)
    if dead and not underfitted:
        rep.verdict(
            "UNUSED FEATURES: permuting " + ", ".join(f"'{d}'" for d in dead) +
            " changes top-1 by <0.002, so a fitted model is ignoring them. Either "
            "the tensor is constant/PAD (see §0) or the architecture cannot reach "
            "it. This is how `opt_card_id` hid for the project's entire history."
        )
    elif dead:
        rep.verdict(
            "UNUSED FEATURES (INCONCLUSIVE): " + ", ".join(f"'{d}'" for d in dead) +
            " show no ablation effect, but the model is underfitted, so this does "
            "not yet distinguish 'dead' from 'not learned yet'. Re-run after "
            "training to convergence."
        )

    # ---- 4. Error decomposition ------------------------------------------
    rep.add("4. WHERE THE ERRORS ARE — by option count",
            _accuracy_table(res_val, "n_opts", "n legal options", bucket=_opt_bucket))
    rep.add("   ...and by SelectType", _accuracy_table(res_val, "sel_type", "sel_type"))

    # ---- 5. Degenerate baselines -----------------------------------------
    sel = res_val["n_opts"] > 1
    p, t = res_val["pred"][sel], res_val["target"][sel]
    fl = res_val["first_legal"][sel]
    top3 = res_val["top3"][sel]
    modal_idx = Counter(t.tolist()).most_common(1)[0][0]
    uniform = float((1.0 / res_val["n_opts"][sel]).mean())
    lines = [
        f"  model top-1                : {float((p == t).mean()):.4f}",
        f"  model top-3                : {float((top3 == t[:, None]).any(-1).mean()):.4f}",
        f"  always first legal option  : {float((fl == t).mean()):.4f}",
        f"  always index {modal_idx:<2} (most common) : "
        f"{float((t == modal_idx).mean()):.4f}",
        f"  uniform random             : {uniform:.4f}",
        f"  corpus ceiling             : {ceil['ceiling_top1']:.4f}",
        "",
        f"  fraction of predictions that ARE the first legal option: "
        f"{float((p == fl).mean()):.4f}",
        f"  fraction of targets  that ARE the first legal option: "
        f"{float((t == fl).mean()):.4f}",
    ]
    rep.add("5. BASELINES — is it beating the degenerate strategies?", lines)
    if float((p == fl).mean()) > 0.80:
        rep.verdict(
            f"POSITIONAL COLLAPSE: {float((p == fl).mean()):.1%} of predictions are "
            f"just the first legal option. The model is largely reproducing option "
            f"ORDER, not evaluating options."
        )

    # ---- 6. Confidence ----------------------------------------------------
    mp = res_val["maxprob"][sel]
    correct = (p == t)
    hi = mp >= np.quantile(mp, 0.75)
    lo = mp <= np.quantile(mp, 0.25)
    lines = [
        f"  mean max-probability : {float(mp.mean()):.4f}",
        f"  mean entropy         : {float(res_val['entropy'][sel].mean()):.4f}",
        f"  accuracy @ top-quartile confidence    : {float(correct[hi].mean()):.4f}",
        f"  accuracy @ bottom-quartile confidence : {float(correct[lo].mean()):.4f}",
    ]
    rep.add("6. CONFIDENCE — uncertain, or confidently wrong?", lines)
    mean_conf, acc_all = float(mp.mean()), float(correct.mean())
    if mean_conf - acc_all > 0.15:
        rep.verdict(
            f"OVERCONFIDENT: mean max-probability {mean_conf:.3f} against actual "
            f"accuracy {acc_all:.3f}. The policy commits hard to wrong options, "
            f"which matters beyond accuracy: RL samples from this distribution, so "
            f"an overconfident prior explores too little."
        )
    if float(correct[hi].mean()) - float(correct[lo].mean()) < 0.10:
        rep.verdict(
            "UNCALIBRATED: the model is no more accurate when confident than when "
            "unsure, so its probabilities carry no information. That also makes "
            "the RL entropy bonus and the KL anchor operate on noise."
        )

    if not rep.verdicts:
        rep.verdict(
            f"No single dominant failure. Non-trivial top-1 {val_nt:.4f} against a "
            f"ceiling of {ceil['ceiling_top1']:.4f} — the remaining gap is spread "
            f"across decision types rather than concentrated."
        )
    return rep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ptcg_il.diagnose",
        description="Diagnose why an IL policy's accuracy is low",
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--archetype", type=int, default=None)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--device", default=None)
    p.add_argument("--max-batches", type=int, default=40,
                   help="Batches per metric pass (default 40 x batch-size rows)")
    p.add_argument("--ablation-batches", type=int, default=8,
                   help="Batches for the ablation sweep; it runs one forward pass "
                        "per feature group, so keep it small")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--json", default=None, help="Also write raw numbers here")
    p.add_argument("--out", default=None, help="Write the report here as well as stdout")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    rep = run_diagnosis(
        args.data_dir, args.ckpt, args.archetype,
        device=args.device, max_batches=args.max_batches,
        batch_size=args.batch_size, ablation_batches=args.ablation_batches,
        split=args.split,
    )
    text = rep.render()
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(rep.data, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
