"""Save/load checkpoints + submission bundle assembly (C.7).

A checkpoint stores ``{step, model_state_dict, ema_state_dict,
optimizer_state_dict, scheduler_state_dict, rng_state}``.

The submission bundle packs best-val EMA weights together with the frozen
preprocessing artifacts (vocab.json, archetypes.json) so the agent runs
identically at inference.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ptcg_il.deck import DECK_KEY, require_deck_record, write_deck_csv


def _rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch RNG states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore all RNG states from a checkpoint."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])


def save_checkpoint(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: Any,
    *,
    step: int,
    save_dir: str | Path,
    tag: str | None = None,
    deck: dict[str, Any],
) -> Path:
    """Save a training checkpoint (C.7).

    Parameters
    ----------
    policy : nn.Module
        Model (should have EMA weights applied before calling).
    optimizer : Optimizer
    scheduler : LRScheduler
    ema : _EMA
        EMA tracker with .state_dict().
    step : int
        Global training step.
    save_dir : Path
        Output directory.
    tag : str or None
        Suffix for the filename (e.g. "best", "last", "step-0004000").
    deck : dict
        Deck identity record from :func:`ptcg_il.deck.build_deck_metadata`,
        stored under the ``"deck"`` key.  Without it a checkpoint does not say
        which of the near-disjoint archetype decks it was trained to play, and a
        wrong pairing fails silently (unseen cards just map to UNKNOWN).

        **Required.**  It used to default to ``None`` and be dropped, so an
        unlabelled checkpoint was indistinguishable from a labelled one until
        something downstream needed the deck.  Callers that genuinely have no
        deck have nothing shippable to save.

    Returns
    -------
    Path to the saved checkpoint.

    Raises
    ------
    ValueError
        If *deck* is not a usable record.  Every caller reaching here should
        already have failed at startup — :func:`ptcg_il.train.loop.train`
        builds the record before the first batch — so this is the backstop,
        not the expected error site.
    """
    require_deck_record(deck, f"save_checkpoint(tag={tag!r})")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fname = f"ckpt-{tag}.pt" if tag else "ckpt.pt"
    path = save_dir / fname

    ckpt: dict[str, Any] = {
        "step": step,
        "model_state_dict": {k: v.cpu() for k, v in policy.state_dict().items()},
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": _cpu_state_dict(optimizer.state_dict()),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": _rng_state(),
        # Architecture sizes, so a loader never has to infer V/A/D/heads/layers/ff
        # from tensor shapes.  ``Policy`` records these itself at construction;
        # anything else (a test double, say) simply gets an empty dict.
        "config": dict(getattr(policy, "config", {})),
        # Which optimizer wrote ``optimizer_state_dict``.  AdamW keeps two
        # moments per parameter and Muon keeps one momentum buffer for the trunk,
        # so the two state dicts are not interchangeable — and
        # ``Optimizer.load_state_dict`` matches groups by position, so a
        # mismatch can load quietly rather than raise.  ``train`` refuses on this
        # field; see ``require_matching_optimizer``.
        "optimizer": _optimizer_kind(optimizer),
    }
    ckpt[DECK_KEY] = deck

    torch.save(ckpt, path)
    return path


def _optimizer_kind(optimizer: torch.optim.Optimizer) -> str:
    """``"muon"`` or ``"adamw"`` — the name the ``--optimizer`` flag uses."""
    return "muon" if type(optimizer).__name__ == "Muon" else "adamw"


def _cpu_state_dict(sd: dict) -> dict:
    """Move all tensors in a state dict to CPU."""
    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu()
        else:
            out[k] = v
    return out


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint (C.7).

    Parameters
    ----------
    path : Path
        Path to the ``.pt`` checkpoint file.
    device : torch.device
        Device to load tensors to.

    Returns
    -------
    Dict with ``step, model_state_dict, ema_state_dict, optimizer_state_dict,
    scheduler_state_dict, rng_state``.
    """
    path = Path(path)
    if isinstance(device, str):
        device = torch.device(device)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Map model/optimizer state to requested device
    if device.type != "cpu":
        for v in [ckpt["model_state_dict"], ckpt["optimizer_state_dict"]]:
            for k in v:
                if isinstance(v[k], torch.Tensor):
                    v[k] = v[k].to(device)

    return ckpt


def build_submission_bundle(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    main_py: str | Path | None = None,
    deck_csv: str | Path | None = None,
) -> Path:
    """Assemble a Kaggle-ready submission bundle (C.7).

    Copies the EMA weights, ``vocab.json``, ``archetypes.json``, ``deck.csv``,
    and ``main.py`` / ``cg/`` into a flat ``submission/`` directory.

    Parameters
    ----------
    checkpoint_path : Path
        Path to a ``ckpt-best.pt`` (EMA weights already applied).
    data_dir : Path
        Directory containing ``vocab.json``, ``archetypes.json``.
    output_dir : Path
        Where to create the ``submission/`` directory.
    main_py : Path or None
        Path to the top-level ``main.py``.  Defaults to the repo root
        ``main.py`` relative to the caller's cwd.
    deck_csv : Path or None
        Path to ``deck.csv``.  Defaults to ``data_dir / "deck.csv"``.

    Returns
    -------
    Path to the submission directory.
    """
    checkpoint_path = Path(checkpoint_path)
    data_path = Path(data_dir)
    output_path = Path(output_dir) / "submission"
    output_path.mkdir(parents=True, exist_ok=True)

    # Model weights
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Strip optimizer/scheduler — we only need EMA weights
    # Ship the deck label alongside the weights — weights.pt is the file that
    # actually gets submitted, so it should say which deck it plays.
    weights: dict[str, Any] = {"model_state_dict": ckpt["model_state_dict"]}
    if ckpt.get(DECK_KEY):
        weights[DECK_KEY] = ckpt[DECK_KEY]
    # Carry the architecture through to the shipped file, so the packed main.py
    # can build the exact model instead of inferring it from tensor shapes.
    if ckpt.get("config"):
        weights["config"] = ckpt["config"]
    torch.save(weights, output_path / "weights.pt")

    # Vocab + archetypes
    for f in ("vocab.json", "archetypes.json"):
        src = data_path / f
        if src.exists():
            shutil.copy2(src, output_path / f)

    # main.py
    if main_py is None:
        main_py = Path("main.py")
    if Path(main_py).exists():
        shutil.copy2(main_py, output_path / "main.py")

    # deck.csv — prefer the deck the checkpoint says it was trained on.
    #
    # An explicit ``deck_csv`` still wins, but the default must not be a bare
    # path guess: shipping a deck that the policy was not trained for is a
    # silent failure (every unseen card maps to UNKNOWN), and the old default
    # (``data/deck.csv``, which is never produced by mining) shipped no deck at
    # all.  The label written by ``save_checkpoint`` removes the guesswork.
    if deck_csv is not None and Path(deck_csv).exists():
        shutil.copy2(deck_csv, output_path / "deck.csv")
    else:
        deck_meta = ckpt.get(DECK_KEY)
        if deck_meta and deck_meta.get("deck"):
            write_deck_csv(deck_meta, output_path / "deck.csv")
            # Keep the full identity record beside the weights so the bundle is
            # self-describing (which archetype, which vocab hash).
            (output_path / "deck.json").write_text(
                json.dumps(deck_meta, indent=2, ensure_ascii=False) + "\n"
            )
        else:
            for cand in (data_path / "deck.csv", Path("deck.csv")):
                if cand.exists():
                    shutil.copy2(cand, output_path / "deck.csv")
                    break
            else:
                raise FileNotFoundError(
                    "No deck for the submission bundle: checkpoint carries no "
                    f"'{DECK_KEY}' label and no deck.csv was found. Retrain with "
                    "--archetype-self (which stamps the deck into the .pt) or pass "
                    "deck_csv= explicitly."
                )

    # cg/ engine lib
    cg_src = Path("cg")
    if cg_src.exists():
        cg_dst = output_path / "cg"
        if cg_dst.exists():
            shutil.rmtree(cg_dst)
        shutil.copytree(cg_src, cg_dst, dirs_exist_ok=True)

    return output_path
