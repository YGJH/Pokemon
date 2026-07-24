"""ShardDataset + collate_fn (TRANSFORMER_IL_SPEC.md C.1–C.4).

mmap-based dataset reading fixed-shape .npz shard slices, joined with
meta.parquet for per-sample metadata.  ``sample_weight`` is derived at load
time from meta columns via the C.3 formula.
"""

from __future__ import annotations

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
    "minCount", "maxCount", "action_len",
})

# Keys that should be float32 (can be cast to bf16)
_FLOAT_KEYS = frozenset({
    "cls_feat", "poke_feat", "hand_feat", "sum_feat",
    "stadium_present", "opt_scalar",
    "value_target", "sample_weight",
})

# Boolean masks
_BOOL_KEYS = frozenset({
    "tok_mask", "opt_mask", "discard_mask",
})


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
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        shuffle: bool = False,
        seed: int = 42,
    ):
        data_dir = Path(data_dir)
        self.split = split

        # Load meta
        meta_path = data_dir / "meta.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.parquet not found at {meta_path}")
        self.meta = pd.read_parquet(meta_path)

        # Filter to split
        self.meta = self.meta[self.meta["shard"].str.startswith(split)].reset_index(drop=True)
        if len(self.meta) == 0:
            raise ValueError(f"No samples found for split '{split}' in meta.parquet")

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

        # Attach sample_weight and value_target from meta
        sample["sample_weight"] = torch.tensor(
            float(row["sample_weight"]), dtype=torch.float32
        )

        return sample

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
