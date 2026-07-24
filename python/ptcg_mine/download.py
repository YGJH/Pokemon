"""Phase 1: resumable download of sampled episodes via an injected Kaggle API.

The Kaggle API is always passed in as `api` (never imported/constructed here)
so tests can supply a fake and no real network call is ever made in this
module or its tests.

Expected `api` shape (thin adapter over kaggle.api.KaggleApi):
  - `api.dataset_list_files(slug, page_token=None)` -> object with
    `.files` (each having `.name`) and `.nextPageToken` (falsy on last page).
  - `api.dataset_download_file(slug, file_name, path=<dir>)` -> writes the
    file named `file_name` into directory `path`.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ptcg_mine.sampling import deterministic_pick, select_days

MAX_WORKERS = 8
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.05


def list_episode_files(api, slug: str) -> list[str]:
    """Page through api.dataset_list_files(slug, page_token=...), collecting
    all "<id>.json" file names until there is no next page token.

    Duck-typed against both response shapes seen in the wild: the brief's
    camelCase (`.files` / `.nextPageToken`) and the currently-vendored real
    `kaggle` SDK's snake_case `ApiListDatasetFilesResponse`
    (`.dataset_files` / `.next_page_token`).
    """
    names: list[str] = []
    page_token = None
    while True:
        response = api.dataset_list_files(slug, page_token=page_token)
        files = getattr(response, "dataset_files", None) or getattr(response, "files", None) or []
        names.extend(f.name for f in files if f.name.endswith(".json"))
        page_token = getattr(response, "next_page_token", None) or getattr(response, "nextPageToken", None)
        if not page_token:
            break
    return names


def download_episode(api, slug: str, filename: str, dest_dir: str | Path) -> Path:
    """Download `filename` from dataset `slug` into `dest_dir`, resuming
    (skipping) if the file is already present. Returns the destination path.
    """
    dest_dir = Path(dest_dir)
    dest_path = dest_dir / filename
    if dest_path.exists():
        return dest_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_file(slug, filename, path=str(dest_dir))
    return dest_path


def _episode_id(filename: str) -> str:
    return filename[: -len(".json")] if filename.endswith(".json") else filename


def _download_with_retry(api, slug: str, filename: str, dest_dir: Path) -> dict:
    """Attempt download_episode with simple retry/backoff. Returns a manifest
    row dict; never raises (failures are captured as ok=False).
    """
    episode_id = _episode_id(filename)
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            path = download_episode(api, slug, filename, dest_dir)
            return {
                "episode_id": episode_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "ok": True,
            }
        except Exception as exc:  # noqa: BLE001 - deliberately broad: recorded, not raised
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    return {
        "episode_id": episode_id,
        "path": str(dest_dir / filename),
        "bytes": 0,
        "ok": False,
        "error": str(last_error) if last_error is not None else None,
    }


def download_corpus(config, api) -> pd.DataFrame:
    """Sample and download episodes for config.n_days days, config.target_episodes
    total (target_episodes // n_days per day), using a bounded thread pool with
    simple retry/backoff. Writes and returns a manifest DataFrame with columns
    [day, episode_id, path, bytes, ok].
    """
    if config.days is not None:
        days = config.days
    else:
        manifest_df = pd.read_csv(config.manifest_csv)
        days = select_days(manifest_df, config.n_days)

    quota = max(1, config.target_episodes // max(config.n_days, 1))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for day in days:
            slug = config.dataset_prefix + day
            files = list_episode_files(api, slug)
            picked = deterministic_pick(files, quota, config.seed)
            dest_dir = Path(config.raw_dir) / day
            for filename in picked:
                fut = pool.submit(_download_with_retry, api, slug, filename, dest_dir)
                futures[fut] = day

        for fut in as_completed(futures):
            row = fut.result()
            row["day"] = futures[fut]
            rows.append(row)

    manifest = pd.DataFrame(rows, columns=["day", "episode_id", "path", "bytes", "ok"])
    manifest = manifest.sort_values(["day", "episode_id"]).reset_index(drop=True)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_dir / "downloaded.parquet", index=False)

    return manifest
