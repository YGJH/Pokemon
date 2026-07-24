"""Tests for ptcg_mine.download: list_episode_files, download_episode, download_corpus.

Uses fake Kaggle API objects only -- no real network access. Two listing
response shapes are exercised because they both occur in the wild:
  - the brief's camelCase: `.files` (each having `.name`) / `.nextPageToken`
  - the real, currently-vendored `kaggle==2.2.4` SDK's snake_case
    `ApiListDatasetFilesResponse`: `.dataset_files` / `.next_page_token`
`api.dataset_download_file(slug, file_name, path=...)` writes the file into
`path` in both cases.
"""

import dataclasses

import pandas as pd
import pytest

from ptcg_mine.config import MineConfig
from ptcg_mine.download import download_corpus, download_episode, list_episode_files


class FakeFile:
    def __init__(self, name):
        self.name = name


class FakeListResponse:
    def __init__(self, files, next_page_token=None):
        self.files = files
        self.nextPageToken = next_page_token


class FakeApi:
    """Fake Kaggle API: serves paged file listings and writes dummy downloads."""

    def __init__(self, files_by_slug, page_size=2, fail_names=frozenset()):
        self.files_by_slug = files_by_slug
        self.page_size = page_size
        self.fail_names = fail_names
        self.list_calls = []
        self.download_calls = []

    def dataset_list_files(self, slug, page_token=None):
        self.list_calls.append((slug, page_token))
        names = self.files_by_slug.get(slug, [])
        start = int(page_token) if page_token else 0
        end = start + self.page_size
        page = names[start:end]
        next_token = str(end) if end < len(names) else None
        return FakeListResponse([FakeFile(n) for n in page], next_page_token=next_token)

    def dataset_download_file(self, slug, file_name, path=None):
        self.download_calls.append((slug, file_name, path))
        if file_name in self.fail_names:
            raise RuntimeError(f"simulated failure downloading {file_name}")
        from pathlib import Path

        dest = Path(path) / file_name
        dest.write_text(f"dummy contents of {file_name}")


class RealShapedListResponse:
    """Mirrors the real kaggle SDK's ApiListDatasetFilesResponse: snake_case
    `dataset_files` / `next_page_token` (no `.files` / `.nextPageToken`).
    """

    def __init__(self, dataset_files, next_page_token=None):
        self.dataset_files = dataset_files
        self.next_page_token = next_page_token


class RealShapedApi:
    """Fake API using the real SDK's snake_case response attribute names,
    to guard against regressing to camelCase-only duck typing.
    """

    def __init__(self, files_by_slug, page_size=2):
        self.files_by_slug = files_by_slug
        self.page_size = page_size
        self.list_calls = []

    def dataset_list_files(self, slug, page_token=None):
        self.list_calls.append((slug, page_token))
        names = self.files_by_slug.get(slug, [])
        start = int(page_token) if page_token else 0
        end = start + self.page_size
        page = names[start:end]
        next_token = str(end) if end < len(names) else None
        return RealShapedListResponse([FakeFile(n) for n in page], next_page_token=next_token)


# ---------------------------------------------------------------------------
# list_episode_files
# ---------------------------------------------------------------------------


def test_list_episode_files_concatenates_pages():
    names = [f"ep{i}.json" for i in range(5)]
    api = FakeApi({"slug-a": names}, page_size=2)
    result = list_episode_files(api, "slug-a")
    assert result == names


def test_list_episode_files_single_page():
    names = ["ep0.json", "ep1.json"]
    api = FakeApi({"slug-a": names}, page_size=10)
    result = list_episode_files(api, "slug-a")
    assert result == names
    assert len(api.list_calls) == 1


def test_list_episode_files_concatenates_pages_real_sdk_shape():
    """Guards against regressing to camelCase-only duck typing: the real,
    currently-installed kaggle==2.2.4 SDK's ApiListDatasetFilesResponse uses
    snake_case `dataset_files` / `next_page_token`, not `.files` /
    `.nextPageToken`.
    """
    names = [f"ep{i}.json" for i in range(5)]
    api = RealShapedApi({"slug-a": names}, page_size=2)
    result = list_episode_files(api, "slug-a")
    assert result == names
    assert len(api.list_calls) == 3  # 2 + 2 + 1 items across 3 pages


def test_list_episode_files_empty():
    api = FakeApi({"slug-a": []}, page_size=2)
    result = list_episode_files(api, "slug-a")
    assert result == []


# ---------------------------------------------------------------------------
# download_episode
# ---------------------------------------------------------------------------


def test_download_episode_downloads_new_file(tmp_path):
    api = FakeApi({"slug-a": []})
    dest_dir = tmp_path / "raw" / "2026-06-20"
    result = download_episode(api, "slug-a", "ep0.json", dest_dir)
    assert result == dest_dir / "ep0.json"
    assert result.exists()
    assert len(api.download_calls) == 1


def test_download_episode_skips_existing_file(tmp_path):
    dest_dir = tmp_path / "raw" / "2026-06-20"
    dest_dir.mkdir(parents=True)
    existing = dest_dir / "ep0.json"
    existing.write_text("already here")

    api = FakeApi({"slug-a": []})
    result = download_episode(api, "slug-a", "ep0.json", dest_dir)

    assert result == existing
    assert existing.read_text() == "already here"  # untouched
    assert len(api.download_calls) == 0  # no network call made


# ---------------------------------------------------------------------------
# download_corpus
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest_csv(tmp_path):
    df = pd.DataFrame(
        {
            "date": ["2026-06-18", "2026-06-19", "2026-06-20"],
            "top_avg_score": [500.0, 600.0, 700.0],
        }
    )
    path = tmp_path / "manifest.csv"
    df.to_csv(path, index=False)
    return path


def _config(tmp_path, manifest_csv, **overrides):
    base = MineConfig(
        raw_dir=tmp_path / "raw",
        out_dir=tmp_path / "data",
        manifest_csv=manifest_csv,
        dataset_prefix="kaggle/pokemon-tcg-ai-battle-episodes-",
        n_days=2,
        target_episodes=6,
        seed=0,
    )
    return dataclasses.replace(base, **overrides)


def test_download_corpus_creates_manifest_and_files(tmp_path, manifest_csv):
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"ep{i}.json" for i in range(10)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"ep{i}.json" for i in range(10)],
    }
    api = FakeApi(files_by_slug, page_size=4)
    config = _config(tmp_path, manifest_csv)

    result = download_corpus(config, api)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["day", "episode_id", "path", "bytes", "ok"]
    # n_days=2 selects the 2 most recent days: 2026-06-19, 2026-06-20
    assert set(result["day"]) == {"2026-06-19", "2026-06-20"}
    # target_episodes // n_days = 3 per day
    assert len(result) == 6
    assert result["ok"].all()
    for _, row in result.iterrows():
        p = config.raw_dir / row["day"] / f"{row['episode_id']}.json"
        assert p.exists()
        assert p.read_text().startswith("dummy contents")
        assert row["bytes"] == p.stat().st_size

    parquet_path = config.out_dir / "downloaded.parquet"
    assert parquet_path.exists()
    reloaded = pd.read_parquet(parquet_path)
    assert len(reloaded) == len(result)


def test_download_corpus_resumes_skips_existing(tmp_path, manifest_csv):
    # Distinct filenames per day so a pick on one day can't collide with the other.
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(3)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(3)],
    }
    config = _config(tmp_path, manifest_csv, n_days=2, target_episodes=4)

    # Pre-seed: figure out deterministically which ids will be picked for
    # 2026-06-19 and pre-create one of its files so we can assert it's skipped.
    from ptcg_mine.sampling import deterministic_pick

    quota = config.target_episodes // config.n_days
    picked = deterministic_pick(files_by_slug["kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19"], quota, config.seed)
    pre_existing_name = picked[0]
    day_dir = config.raw_dir / "2026-06-19"
    day_dir.mkdir(parents=True)
    (day_dir / pre_existing_name).write_text("PRE-EXISTING SENTINEL")

    api = FakeApi(files_by_slug, page_size=2)
    result = download_corpus(config, api)

    assert (day_dir / pre_existing_name).read_text() == "PRE-EXISTING SENTINEL"
    # the pre-existing file was never passed to the fake api's download
    downloaded_names = {name for (_, name, _) in api.download_calls}
    assert pre_existing_name not in downloaded_names


def test_download_corpus_marks_failures_not_ok(tmp_path, manifest_csv):
    # Distinct filenames per day so failing one id can't affect the other day.
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(3)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(3)],
    }
    config = _config(tmp_path, manifest_csv, n_days=2, target_episodes=4)

    from ptcg_mine.sampling import deterministic_pick

    quota = config.target_episodes // config.n_days
    picked = deterministic_pick(files_by_slug["kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19"], quota, config.seed)
    always_fail = picked[0]

    api = FakeApi(files_by_slug, page_size=2, fail_names={always_fail})
    result = download_corpus(config, api)

    failing_rows = result[result["episode_id"] == always_fail[: -len(".json")]]
    assert len(failing_rows) == 1
    assert failing_rows.iloc[0]["ok"] == False  # noqa: E712
    assert failing_rows.iloc[0]["bytes"] == 0
    # everything else still succeeded
    assert result["ok"].sum() == len(result) - 1


def test_download_corpus_uses_explicit_days_when_given(tmp_path, manifest_csv):
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-18": [f"ep{i}.json" for i in range(5)],
    }
    api = FakeApi(files_by_slug, page_size=5)
    config = _config(tmp_path, manifest_csv, days=["2026-06-18"], n_days=1, target_episodes=3)

    result = download_corpus(config, api)

    assert set(result["day"]) == {"2026-06-18"}
    assert len(result) == 3
