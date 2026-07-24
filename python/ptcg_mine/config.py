"""MineConfig: shared configuration dataclass for the corpus-mining pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MineConfig:
    """Configuration for mining Kaggle Pokemon TCG episode replays.

    Fields are consumed across all pipeline phases (sampling, download,
    stats/archetype, vocab/static-tables/artifacts); this task only defines
    the dataclass.
    """

    days: list[str] | None = None  # None => auto-select recent
    n_days: int = 20
    target_episodes: int = 10000
    seed: int = 0
    k_experts: int = 10
    g_min: int = 50
    jaccard_thresh: float = 0.90
    n_self: int = 6
    n_opp: int = 6
    w_lost: float = 0.5
    vocab_mode: str = "all_corpus"
    n_vocab: int | None = None
    h_max: int = 30
    o_max: int = 64
    d_max: int = 60
    raw_dir: Path = Path("raw")
    out_dir: Path = Path("data")
    manifest_csv: Path = Path("archive/manifest.csv")
    dataset_prefix: str = "kaggle/pokemon-tcg-ai-battle-episodes-"
