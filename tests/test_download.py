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
from ptcg_mine.download import (
    DownloadError,
    download_corpus,
    download_episode,
    list_episode_files,
)


class FakeFile:
    def __init__(self, name):
        self.name = name


class FakeListResponse:
    def __init__(self, files, next_page_token=None):
        self.files = files
        self.nextPageToken = next_page_token


class FakeApi:
    """Fake Kaggle API: serves paged file listings and writes dummy downloads."""

    def __init__(
        self,
        files_by_slug,
        page_size=2,
        fail_names=frozenset(),
        fail_list_slugs=frozenset(),
        list_fail_countdown=0,
        list_fail_after_pages=None,
    ):
        self.files_by_slug = files_by_slug
        self.page_size = page_size
        self.fail_names = fail_names
        self.fail_list_slugs = fail_list_slugs
        self._list_fail_countdown = list_fail_countdown
        # After this many pages served, every further listing call fails --
        # simulates a listing that dies partway through pagination.
        self.list_fail_after_pages = list_fail_after_pages
        self._pages_served = 0
        self.list_calls = []
        self.download_calls = []

    def dataset_list_files(self, slug, page_token=None):
        self.list_calls.append((slug, page_token))
        if slug in self.fail_list_slugs:
            raise RuntimeError(f"simulated listing failure for {slug}")
        if (
            self.list_fail_after_pages is not None
            and self._pages_served >= self.list_fail_after_pages
        ):
            raise RuntimeError("simulated listing failure partway through pagination")
        if self._list_fail_countdown > 0:
            self._list_fail_countdown -= 1
            raise RuntimeError("simulated transient listing failure")
        self._pages_served += 1
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
    assert list(result.columns) == [
        "day", "episode_id", "path", "bytes", "ok", "error", "rate_limited", "skipped",
    ]
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
    # Pinned to sample mode: the expectation below is built from
    # deterministic_pick, which only stream-mode's page-order queueing bypasses.
    config = _config(tmp_path, manifest_csv, n_days=2, target_episodes=4, list_mode="sample")

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


# ---------------------------------------------------------------------------
# resilient listing (rate-limit / transient errors must not crash the run)
# ---------------------------------------------------------------------------


def test_download_corpus_skips_day_that_cannot_be_listed(tmp_path, manifest_csv, monkeypatch):
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    good = "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20"
    bad = "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19"
    files_by_slug = {
        good: [f"g{i}.json" for i in range(10)],
        bad: [f"b{i}.json" for i in range(10)],
    }
    # n_days=2 selects 2026-06-19 (fails to list) and 2026-06-20 (lists fine).
    api = FakeApi(files_by_slug, page_size=4, fail_list_slugs={bad})
    config = _config(tmp_path, manifest_csv)

    result = download_corpus(config, api)  # must not raise

    assert set(result["day"]) == {"2026-06-20"}
    assert len(result) == 3  # only the listable day's quota
    assert result["ok"].all()


def test_list_episode_files_retries_transient_listing_error(monkeypatch):
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    names = [f"ep{i}.json" for i in range(3)]
    api = FakeApi({"slug-a": names}, page_size=10, list_fail_countdown=2)

    result = list_episode_files(api, "slug-a")

    assert result == names
    assert len(api.list_calls) == 3  # 2 transient failures + 1 success


# ---------------------------------------------------------------------------
# rate limiting (HTTP 429): the failure mode that silently emptied a real run
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class RateLimitError(Exception):
    """Mimics an SDK HTTP error carrying a 429 response + Retry-After."""

    def __init__(self, retry_after=None):
        super().__init__("429 Client Error: Too Many Requests")
        headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
        self.response = FakeResponse(429, headers)


class RateLimitedApi(FakeApi):
    """Serves listings normally but 429s downloads after `allow_n` successes."""

    def __init__(self, *args, allow_n=0, retry_after=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_n = allow_n
        self.retry_after = retry_after
        self._n_ok = 0
        self._lock = __import__("threading").Lock()

    def dataset_download_file(self, slug, file_name, path=None):
        with self._lock:
            allowed = self._n_ok < self.allow_n
            if allowed:
                self._n_ok += 1
        if not allowed:
            self.download_calls.append((slug, file_name, path))
            raise RateLimitError(self.retry_after)
        return super().dataset_download_file(slug, file_name, path=path)


def test_is_rate_limit_detects_429_response_and_message():
    from ptcg_mine.download import _is_rate_limit

    assert _is_rate_limit(RateLimitError())
    assert _is_rate_limit(RuntimeError("HTTP 429 Too Many Requests"))
    assert _is_rate_limit(RuntimeError("too many requests, slow down"))
    assert not _is_rate_limit(RuntimeError("404 not found"))


def test_download_retry_honors_retry_after_and_is_patient_on_429(tmp_path, monkeypatch):
    """A 429 must get the patient, Retry-After-aware backoff -- not the 0.05s
    transient backoff that made the real run burn all its attempts instantly."""
    from ptcg_mine import download

    slept = []
    monkeypatch.setattr(download, "_sleep", lambda s: slept.append(s), raising=False)

    class AlwaysLimited:
        def __init__(self):
            self.n = 0

        def dataset_download_file(self, slug, file_name, path=None):
            self.n += 1
            raise RateLimitError(retry_after=7)

    api = AlwaysLimited()
    row = download._download_with_retry(api, "slug-a", "ep0.json", tmp_path / "raw")

    assert row["ok"] is False
    assert row["rate_limited"] is True
    # more patient than the 3-attempt transient budget
    assert api.n == download.RATE_LIMIT_MAX_RETRIES
    # and it actually waited the server-requested interval
    assert 7 in slept


def test_download_corpus_raises_when_no_day_can_be_listed(tmp_path, manifest_csv, monkeypatch):
    """All days rate-limited at listing time -> fail immediately and loudly,
    instead of returning an empty manifest and failing 3 phases later."""
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    api = FakeApi(
        {},
        page_size=4,
        fail_list_slugs={
            "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19",
            "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20",
        },
    )
    config = _config(tmp_path, manifest_csv)

    with pytest.raises(DownloadError) as exc:
        download_corpus(config, api)

    assert "could not be listed" in str(exc.value)


def test_download_corpus_raises_when_every_download_fails(tmp_path, manifest_csv, monkeypatch):
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(4)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(4)],
    }
    all_names = {n for names in files_by_slug.values() for n in names}
    api = FakeApi(files_by_slug, page_size=4, fail_names=all_names)
    config = _config(tmp_path, manifest_csv)

    with pytest.raises(DownloadError) as exc:
        download_corpus(config, api)

    assert "0 succeeded" in str(exc.value)


def test_download_corpus_raises_and_names_rate_limiting_as_the_cause(
    tmp_path, manifest_csv, monkeypatch
):
    """The real-world case: listing works, downloads 429 -> the error message
    must say so, so the user knows to wait/retry rather than debug the miner."""
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(4)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(4)],
    }
    api = RateLimitedApi(files_by_slug, page_size=4, allow_n=0, retry_after=1)
    config = _config(tmp_path, manifest_csv)

    with pytest.raises(DownloadError) as exc:
        download_corpus(config, api)

    msg = str(exc.value)
    assert "rate-limit" in msg.lower() or "429" in msg


def test_download_corpus_aborts_early_instead_of_grinding_through_all_files(
    tmp_path, manifest_csv, monkeypatch
):
    """Once sustained rate limiting is established, remaining downloads must be
    skipped rather than each burning its full retry budget against the limiter."""
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    monkeypatch.setattr(download, "RATE_LIMIT_ABORT_AFTER", 2, raising=False)
    names = [f"d20-ep{i}.json" for i in range(40)]
    files_by_slug = {"kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": names}
    api = RateLimitedApi(files_by_slug, page_size=40, allow_n=0)
    config = _config(tmp_path, manifest_csv, days=["2026-06-20"], n_days=1, target_episodes=40)

    with pytest.raises(DownloadError):
        download_corpus(config, api)

    # Far fewer API calls than 40 files x RATE_LIMIT_MAX_RETRIES attempts each.
    assert len(api.download_calls) < 40 * download.RATE_LIMIT_MAX_RETRIES


def test_stream_mode_stops_listing_once_quota_is_met(tmp_path, manifest_csv):
    """The whole point: don't page through 2000 files to download 8 of them."""
    names = [f"d20-ep{i:04d}.json" for i in range(400)]
    files_by_slug = {"kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": names}
    api = FakeApi(files_by_slug, page_size=20)
    # quota = target_episodes // n_days = 40 // 1 = 40 -> needs exactly 2 pages
    config = _config(tmp_path, manifest_csv, days=["2026-06-20"], n_days=1, target_episodes=40)

    result = download_corpus(config, api)

    assert len(api.list_calls) == 2, "should stop paging as soon as quota is filled"
    assert len(result) == 40
    assert result["ok"].all()


def test_sample_mode_still_lists_everything(tmp_path, manifest_csv):
    names = [f"d20-ep{i:04d}.json" for i in range(400)]
    files_by_slug = {"kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": names}
    api = FakeApi(files_by_slug, page_size=20)
    config = _config(tmp_path, manifest_csv, days=["2026-06-20"], n_days=1, target_episodes=40)
    config = dataclasses.replace(config, list_mode="sample")

    result = download_corpus(config, api)

    assert len(api.list_calls) == 20, "sample mode must see the full population"
    assert len(result) == 40


def test_stream_mode_keeps_files_queued_before_a_listing_failure(
    tmp_path, manifest_csv, monkeypatch
):
    """A listing that dies partway must not throw away the pages that worked --
    that is what discarded whole days in the real run."""
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    names = [f"d20-ep{i:04d}.json" for i in range(400)]
    files_by_slug = {"kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": names}
    api = FakeApi(files_by_slug, page_size=20, list_fail_after_pages=3)
    config = _config(tmp_path, manifest_csv, days=["2026-06-20"], n_days=1, target_episodes=400)

    result = download_corpus(config, api)  # must not raise

    assert len(result) == 60, "the 3 pages that succeeded should still be downloaded"
    assert result["ok"].all()


def test_stream_mode_reports_day_as_failed_only_when_nothing_was_queued(
    tmp_path, manifest_csv, monkeypatch
):
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    files_by_slug = {"kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": ["d20-ep0.json"]}
    api = FakeApi(files_by_slug, page_size=20, list_fail_after_pages=0)
    config = _config(tmp_path, manifest_csv, days=["2026-06-20"], n_days=1, target_episodes=40)

    with pytest.raises(DownloadError) as exc:
        download_corpus(config, api)

    assert "could not be listed" in str(exc.value)


def test_stream_mode_marks_failed_downloads_not_ok(tmp_path, manifest_csv):
    """Stream mode queues in listing order, so the first file of the day is the
    one guaranteed to be attempted."""
    names = [f"d20-ep{i}.json" for i in range(6)]
    files_by_slug = {"kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": names}
    api = FakeApi(files_by_slug, page_size=2, fail_names={names[0]})
    config = _config(tmp_path, manifest_csv, days=["2026-06-20"], n_days=1, target_episodes=4)

    result = download_corpus(config, api)

    failed = result[result["episode_id"] == "d20-ep0"]
    assert len(failed) == 1
    assert not bool(failed.iloc[0]["ok"])
    assert result["ok"].sum() == 3


def test_download_corpus_rejects_unknown_list_mode(tmp_path, manifest_csv):
    api = FakeApi({}, page_size=4)
    config = _config(tmp_path, manifest_csv, days=["2026-06-20"])
    config = dataclasses.replace(config, list_mode="telepathy")

    with pytest.raises(ValueError):
        download_corpus(config, api)


def test_order_days_defaults_to_newest_first():
    from ptcg_mine.download import _order_days

    days = ["2026-07-04", "2026-07-23", "2026-07-10"]
    assert _order_days(days, "recent-first") == ["2026-07-23", "2026-07-10", "2026-07-04"]
    assert _order_days(days, "oldest-first") == ["2026-07-04", "2026-07-10", "2026-07-23"]
    assert _order_days(days, "as-selected") == days


def test_order_days_rejects_unknown_order():
    from ptcg_mine.download import _order_days

    with pytest.raises(ValueError):
        _order_days(["2026-07-04"], "sideways")


def test_download_corpus_fetches_recent_days_first(tmp_path, manifest_csv):
    """select_days returns oldest-to-newest; Phase 1 must invert that so a run
    killed by rate limiting still leaves the freshest days on disk."""
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": ["d19-ep0.json"],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": ["d20-ep0.json"],
    }
    api = FakeApi(files_by_slug, page_size=4)
    config = _config(tmp_path, manifest_csv, days=["2026-06-19", "2026-06-20"])

    download_corpus(config, api)

    listed = [slug for slug, *_ in api.list_calls]
    assert listed[0].endswith("2026-06-20")  # newer day listed first
    assert listed[-1].endswith("2026-06-19")


def test_download_corpus_day_order_is_configurable(tmp_path, manifest_csv):
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": ["d19-ep0.json"],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": ["d20-ep0.json"],
    }
    api = FakeApi(files_by_slug, page_size=4)
    config = _config(tmp_path, manifest_csv, days=["2026-06-19", "2026-06-20"])
    config = dataclasses.replace(config, day_order="oldest-first")

    download_corpus(config, api)

    listed = [slug for slug, *_ in api.list_calls]
    assert listed[0].endswith("2026-06-19")


def test_day_order_does_not_change_which_episodes_are_picked(tmp_path, manifest_csv):
    """Ordering is a fetch-priority knob only -- the sampled set must be identical."""
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(8)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(8)],
    }
    base = _config(tmp_path / "a", manifest_csv, days=["2026-06-19", "2026-06-20"])
    recent = download_corpus(base, FakeApi(files_by_slug, page_size=8))

    other = _config(tmp_path / "b", manifest_csv, days=["2026-06-19", "2026-06-20"])
    other = dataclasses.replace(other, day_order="oldest-first")
    oldest = download_corpus(other, FakeApi(files_by_slug, page_size=8))

    assert sorted(recent["episode_id"]) == sorted(oldest["episode_id"])


def test_download_corpus_raises_when_no_days_selected(tmp_path, manifest_csv):
    api = FakeApi({}, page_size=4)
    config = _config(tmp_path, manifest_csv, days=[])

    with pytest.raises(DownloadError) as exc:
        download_corpus(config, api)

    assert "no days were selected" in str(exc.value)


def test_download_corpus_writes_manifest_even_when_it_raises(tmp_path, manifest_csv, monkeypatch):
    """Partial progress must survive the failure so a later run can resume."""
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(4)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(4)],
    }
    all_names = {n for names in files_by_slug.values() for n in names}
    api = FakeApi(files_by_slug, page_size=4, fail_names=all_names)
    config = _config(tmp_path, manifest_csv)

    with pytest.raises(DownloadError):
        download_corpus(config, api)

    assert (config.out_dir / "downloaded.parquet").exists()


def test_download_corpus_succeeds_when_rate_limit_clears(tmp_path, manifest_csv, monkeypatch):
    """A transient 429 that resolves must not fail the run."""
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)

    class BrieflyLimited(FakeApi):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._fail_left = 3

        def dataset_download_file(self, slug, file_name, path=None):
            if self._fail_left > 0:
                self._fail_left -= 1
                raise RateLimitError(retry_after=1)
            return super().dataset_download_file(slug, file_name, path=path)

    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(4)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(4)],
    }
    api = BrieflyLimited(files_by_slug, page_size=4)
    config = _config(tmp_path, manifest_csv)

    result = download_corpus(config, api)  # must not raise

    assert result["ok"].all()
