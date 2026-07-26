"""ShardDataset + collate_fn (TRANSFORMER_IL_SPEC.md C.1–C.4).

mmap-based dataset reading fixed-shape .npz shard slices, joined with
meta.parquet for per-sample metadata.  ``sample_weight`` is derived at load
time from meta columns via the C.3 formula.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ============================================================
# Constants from C.3 / C.10
# ============================================================
ALPHA_CTX: float = 0.5
ALPHA_ARCH: float = 0.5
W_LOST: float = 0.6

# Keys that should stay on CPU as int64 (indices, masks, types)
_INT_KEYS = frozenset({
    "poke_card_id", "hand_card_id", "stadium_card_id",
    "context_card_id", "effect_card_id", "discard_ids",
    "prize_ids", "tok_type", "tok_owner", "tok_zone",
    "opt_type", "opt_src_idx", "opt_tgt_idx", "opt_card_id",
    "opt_attack_idx", "sel_type", "sel_ctx", "action_idx",
    "minCount", "maxCount", "action_len", "stop_column", "log_len",
})

# Keys that should be float32 (can be cast to bf16)
_FLOAT_KEYS = frozenset({
    "cls_feat", "poke_feat", "hand_feat", "sum_feat",
    "stadium_present", "opt_scalar",
    "value_target", "sample_weight", "log_feat",
})

# Boolean masks
_BOOL_KEYS = frozenset({
    "tok_mask", "opt_mask", "discard_mask",
    "bel_valid", "bel_hand_valid",
})

# Belief labels are stored sparsely in the shards (index/count pairs) because a
# 60-card decklist touches at most 26 of ~300 vocab slots, so dense float32[V]
# rows would be ~97% zeros and would dominate shard size.  They are densified
# per sample here, on the way into the batch.
_BELIEF_SPARSE = (("bel_deck", "bel_deck_idx", "bel_deck_cnt"),
                  ("bel_hidden", "bel_hidden_idx", "bel_hidden_cnt"),
                  ("bel_hand", "bel_hand_idx", "bel_hand_cnt"))


def compute_sample_weights(
    meta: pd.DataFrame,
    alpha_ctx: float = ALPHA_CTX,
    alpha_arch: float = ALPHA_ARCH,
    w_lost: float = W_LOST,
) -> np.ndarray:
    """Compute per-sample weights from meta.parquet columns (C.3 formula).

    ``w = w_ctx[sel_ctx] * w_arch[archetype_self] * w_outcome``,
    normalized so that ``mean(weights) ≈ 1.0`` over the split.

    Parameters
    ----------
    meta : DataFrame
        Must contain columns ``sel_ctx``, ``archetype_self``, ``won``.
    alpha_ctx : float
        Exponent for rare-context balancing (0.5).
    alpha_arch : float
        Exponent for archetype balancing (0.5).
    w_lost : float
        Weight multiplier for lost-game decisions (0.6).

    Returns
    -------
    weights : float32 ndarray[Nsamples]
    """
    n = len(meta)
    if n == 0:
        return np.array([], dtype=np.float32)

    sel_ctx = meta["sel_ctx"].to_numpy(dtype=np.int64)
    archetype_self = meta["archetype_self"].to_numpy(dtype=np.int64)
    won = meta["won"].to_numpy(dtype=bool)

    # Context weights
    ctx_vals, ctx_counts = np.unique(sel_ctx, return_counts=True)
    ctx_count_map = dict(zip(ctx_vals, ctx_counts))
    w_ctx = np.array([n / ctx_count_map[c] for c in sel_ctx], dtype=np.float64)
    w_ctx = w_ctx ** alpha_ctx

    # Archetype weights
    arch_vals, arch_counts = np.unique(archetype_self, return_counts=True)
    arch_count_map = dict(zip(arch_vals, arch_counts))
    w_arch = np.array([n / arch_count_map[a] for a in archetype_self], dtype=np.float64)
    w_arch = w_arch ** alpha_arch

    # Outcome weight
    w_out = np.where(won, 1.0, w_lost).astype(np.float64)

    weights = w_ctx * w_arch * w_out
    weights /= weights.mean()  # normalize to mean ≈ 1.0

    return weights.astype(np.float32)


class ShardDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset over pre-featurized .npz shards, mmap'd for OS page-cache reuse.

    Each ``__getitem__`` reads one sample (dict-of-tensors) by loading the
    appropriate row slice from the underlying mmap'd shard array.  Batch
    assembly happens in ``collate_fn`` via ``torch.stack``.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``shards/`` and ``meta.parquet``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    shuffle : bool
        If True, shuffle indices on init (used for training).
    seed : int
        RNG seed for shuffle.
    archetype_self : int, optional
        If given, keep only decision points whose ``archetype_self`` matches —
        i.e. train a per-deck specialist.  The self archetypes are near-disjoint
        decks (pairwise multiset-Jaccard <= 0.17, no card common to all), so a
        single model trained across all of them is fitting several unrelated
        policies at once.  ``None`` (default) keeps every deck, the original
        behaviour.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        shuffle: bool = False,
        seed: int = 42,
        archetype_self: int | None = None,
    ):
        data_dir = Path(data_dir)
        self.split = split
        self.archetype_self = archetype_self

        # Vocab size, needed to densify the sparse belief labels.  Read from the
        # artifact rather than inferred from the shards: the sparse label rows
        # only reference the cards a deck actually contains, so the largest index
        # seen is a lower bound on V, not V.
        self.vocab_size: int | None = None
        vocab_path = data_dir / "vocab.json"
        if vocab_path.exists():
            from ptcg_il.featurizer import normalize_vocab

            # normalize_vocab, not a raw ["size"]: hand-written vocab files
            # (tests, older artifacts) carry only id_to_index, and a missing
            # key here would break loading for every dataset, belief or not.
            with open(vocab_path) as fh:
                size = normalize_vocab(json.load(fh)).get("size")
            self.vocab_size = int(size) if size else None

        # Load meta
        meta_path = data_dir / "meta.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.parquet not found at {meta_path}")
        self.meta = pd.read_parquet(meta_path)

        # Filter to split
        self.meta = self.meta[self.meta["shard"].str.startswith(split)].reset_index(drop=True)
        if len(self.meta) == 0:
            raise ValueError(f"No samples found for split '{split}' in meta.parquet")

        # Optional per-deck filter.  Applied after the split filter so the
        # 80/10/10 episode-level split still holds within each deck.
        if archetype_self is not None:
            if "archetype_self" not in self.meta.columns:
                raise ValueError(
                    "meta.parquet has no 'archetype_self' column — rebuild shards "
                    "with `ptcg_il.cli build-shards` to enable per-deck training"
                )
            self.meta = self.meta[
                self.meta["archetype_self"] == archetype_self
            ].reset_index(drop=True)
            if len(self.meta) == 0:
                raise ValueError(
                    f"No samples for archetype_self={archetype_self} in split "
                    f"'{split}'. Check `archetypes.json` for valid ids."
                )

        # Compute sample weights from meta columns (C.3)
        self.sample_weights = compute_sample_weights(self.meta)
        self.meta["sample_weight"] = self.sample_weights

        # Shuffle row order if requested
        self._indices = np.arange(len(self.meta))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self._indices)

        # Lazily open shards — mmap each unique shard once
        self._shard_cache: dict[str, dict[str, np.ndarray]] = {}
        self._shards_dir = data_dir / "shards"

    def _open_shard(self, shard_name: str) -> dict[str, np.ndarray]:
        """Open a .npz file with mmap_mode='r' and cache it."""
        if shard_name not in self._shard_cache:
            path = self._shards_dir / shard_name
            if not path.exists():
                raise FileNotFoundError(f"Shard file not found: {path}")
            self._shard_cache[shard_name] = dict(
                np.load(path, mmap_mode="r", allow_pickle=False)
            )
        return self._shard_cache[shard_name]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one sample as a dict-of-tensors.

        Keys match the featurizer contract (Appendix A.4) plus
        ``sample_weight``, ``value_target``.
        """
        row_idx = int(self._indices[idx])
        row = self.meta.iloc[row_idx]
        shard_name = str(row["shard"])
        sample_row = int(row["row"])

        shard = self._open_shard(shard_name)

        sample: dict[str, torch.Tensor] = {}
        for key in shard:
            arr = shard[key][sample_row]  # slice one row
            # Convert to torch tensor
            if key in _BOOL_KEYS:
                sample[key] = torch.from_numpy(np.asarray(arr).copy()).bool()
            elif key in _INT_KEYS:
                sample[key] = torch.from_numpy(np.asarray(arr).copy()).long()
            else:
                sample[key] = torch.from_numpy(np.asarray(arr).copy()).float()

        # Backward compat: old shards lack stop_column
        if "stop_column" not in sample:
            sample["stop_column"] = torch.tensor(-1, dtype=torch.long)

        # Attach sample_weight and value_target from meta
        sample["sample_weight"] = torch.tensor(
            float(row["sample_weight"]), dtype=torch.float32
        )

        self._densify_belief(sample)
        return sample

    def _densify_belief(self, sample: dict[str, torch.Tensor]) -> None:
        """Sparse belief labels -> normalised distributions over the vocab.

        Replaces each ``bel_*_idx``/``bel_*_cnt`` pair with a single
        ``float32[V]`` row summing to 1 (or to 0 when the label is absent, which
        the ``bel_*_valid`` masks flag).  Rows are *not* renormalised when empty
        -- an all-zero row must stay all-zero, or a missing label would silently
        become a uniform target.

        Shards written before belief labels existed simply have no ``bel_*``
        keys; those samples get zero-filled rows and ``bel_valid=False``, so an
        old corpus trains exactly as it did before.
        """
        V = self.vocab_size
        for dense_key, idx_key, cnt_key in _BELIEF_SPARSE:
            # No vocab.json => no way to size the row.  Fall through to the
            # zero-fill branch and mark the label invalid below, rather than
            # emitting a length-1 row the belief head would reject.
            if idx_key not in sample or V is None:
                sample[dense_key] = torch.zeros(V or 1, dtype=torch.float32)
                sample.pop(idx_key, None)
                sample.pop(cnt_key, None)
                continue
            idx = sample.pop(idx_key).long().clamp_(0, V - 1)
            cnt = sample.pop(cnt_key).float()
            row = torch.zeros(V, dtype=torch.float32)
            row.scatter_add_(0, idx, cnt)
            row[0] = 0.0  # PAD slot: padded entries land here carrying count 0
            total = row.sum()
            if total > 0:
                row /= total
            sample[dense_key] = row

        for key, dtype in (("bel_valid", torch.bool), ("bel_hand_valid", torch.bool),
                           ("bel_arch", torch.long)):
            if key not in sample or V is None:
                fill = -1 if dtype is torch.long else False
                sample[key] = torch.tensor(fill, dtype=dtype)
            else:
                sample[key] = sample[key].to(dtype)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for i in range(len(self)):
            yield self[i]

    def get_raw(self, idx: int) -> dict[str, np.ndarray]:
        """Return the raw numpy row without torch conversion (for testing)."""
        row_idx = int(self._indices[idx])
        row = self.meta.iloc[row_idx]
        shard_name = str(row["shard"])
        sample_row = int(row["row"])
        shard = self._open_shard(shard_name)
        return {k: np.asarray(shard[k][sample_row]).copy() for k in shard}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """torch.stack per-key along batch dim B (C.4).

    Derives ``encoder_padding_mask = ~tok_mask`` at collate time.

    Parameters
    ----------
    batch : list[dict]
        List of per-sample dicts from ShardDataset.__getitem__.

    Returns
    -------
    dict
        Batched dict with leading dim B for all tensors.
    """
    if not batch:
        return {}

    # torch.stack every key
    merged: dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for key in keys:
        tensors = [s[key] for s in batch]
        # Scalars need special handling — ensure they become [B] tensors
        shape0 = tensors[0].shape
        if shape0 == ():  # scalar
            merged[key] = torch.stack(tensors)  # → [B]
        else:
            merged[key] = torch.stack(tensors)

    # Derive encoder_padding_mask
    merged["encoder_padding_mask"] = ~merged["tok_mask"]

    return merged
