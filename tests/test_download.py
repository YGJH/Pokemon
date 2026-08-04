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


class RecoveringApi(FakeApi):
    """429s the first `fail_attempts` download attempts, then serves normally.

    Models the thing the sweep exists for: a throttle that eventually lifts.
    """

    def __init__(self, *args, fail_attempts=0, retry_after=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_attempts = fail_attempts
        self.retry_after = retry_after
        self.n_attempts = 0
        self._lock = __import__("threading").Lock()

    def dataset_download_file(self, slug, file_name, path=None):
        with self._lock:
            self.n_attempts += 1
            throttled = self.n_attempts <= self.fail_attempts
        if throttled:
            raise RateLimitError(self.retry_after)
        return super().dataset_download_file(slug, file_name, path=path)


def test_rate_limiting_delays_the_corpus_but_does_not_shrink_it(
    tmp_path, manifest_csv, monkeypatch
):
    """The contract: 429 decides *when* a file arrives, never whether it does.
    Every rate-limited file is retried on a later pass until it lands."""
    from ptcg_mine import download

    waits = []
    monkeypatch.setattr(download, "_sleep", lambda s: waits.append(s), raising=False)
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(4)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(4)],
    }
    # Enough attempts to burn through pass 1 entirely, then the throttle lifts.
    api = RecoveringApi(files_by_slug, page_size=4, fail_attempts=60, retry_after=1)
    config = _config(tmp_path, manifest_csv)

    result = download_corpus(config, api)  # must NOT raise

    assert len(result) == 6, "the corpus is short despite the throttle having lifted"
    assert result["ok"].all(), f"files left unfetched: {result[~result['ok']].to_dict('records')}"
    assert waits, "no backoff between passes -- the sweep never waited"


def test_sweep_backs_off_between_passes(tmp_path, manifest_csv, monkeypatch):
    from ptcg_mine import download

    waits = []
    monkeypatch.setattr(download, "_sleep", lambda s: waits.append(s), raising=False)
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(4)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(4)],
    }
    api = RecoveringApi(files_by_slug, page_size=4, fail_attempts=200)
    config = _config(tmp_path, manifest_csv)

    download_corpus(config, api)

    sweep_waits = [w for w in waits if w >= download.SWEEP_BACKOFF_BASE_SECONDS]
    assert sweep_waits, "no inter-pass backoff was applied"
    assert sweep_waits[0] == 60.0
    assert sweep_waits == sorted(sweep_waits), f"backoff did not grow: {sweep_waits}"
    assert max(sweep_waits) <= download.SWEEP_BACKOFF_CAP_SECONDS


def test_sustained_rate_limiting_ends_the_pass_early_not_the_run(
    tmp_path, manifest_csv, monkeypatch
):
    """Once sustained rate limiting is established, the rest of the pass is
    skipped rather than each file burning its full retry budget -- and those
    files come back on the next pass instead of being lost."""
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    monkeypatch.setattr(download, "RATE_LIMIT_ABORT_AFTER", 2, raising=False)
    names = [f"d20-ep{i}.json" for i in range(40)]
    files_by_slug = {"kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": names}
    api = RecoveringApi(files_by_slug, page_size=40, fail_attempts=10)
    config = _config(tmp_path, manifest_csv, days=["2026-06-20"], n_days=1, target_episodes=40)

    result = download_corpus(config, api)

    # The pass still ends early rather than letting all 40 files burn their full
    # retry budgets -- but it now ends the *pass*, not the run.
    assert len(api.download_calls) < 40 * download.RATE_LIMIT_MAX_RETRIES
    assert result["ok"].all(), "files were dropped rather than retried on a later pass"
    assert len(result) == 40


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


# ---------------------------------------------------------------------------
# Phase 1 volume / throughput reporting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_bytes,expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1024 * 1024 - 1, "1024.0 KB"),
        (1024 * 1024, "1.0 MB"),
        (225_050_624, "214.6 MB"),
        (1024**3, "1.0 GB"),
        (5 * 1024**4, "5120.0 GB"),  # saturates at GB rather than looping forever
    ],
)
def test_format_size(n_bytes, expected):
    from ptcg_mine.download import _format_size

    assert _format_size(n_bytes) == expected


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.0, "0.0s"), (47.32, "47.3s"), (59.9, "59.9s"), (60.0, "1m00s"),
     (412.7, "6m52s"), (3599.0, "59m59s"), (3600.0, "1h00m"), (7860.0, "2h11m")],
)
def test_format_duration(seconds, expected):
    from ptcg_mine.download import _format_duration

    assert _format_duration(seconds) == expected


def _summary_manifest(rows):
    return pd.DataFrame(
        rows, columns=["day", "episode_id", "path", "bytes", "ok", "error", "rate_limited", "skipped"]
    )


def _row(day, episode_id, n_bytes, ok=True):
    return {"day": day, "episode_id": episode_id, "path": f"raw/{day}/{episode_id}.json",
            "bytes": n_bytes, "ok": ok, "error": None, "rate_limited": False, "skipped": False}


def test_log_download_summary_reports_totals_and_per_day(caplog):
    from ptcg_mine.download import _log_download_summary

    manifest = _summary_manifest([
        _row("2026-06-19", "a", 1024 * 1024),
        _row("2026-06-19", "b", 1024 * 1024),
        _row("2026-06-20", "c", 2 * 1024 * 1024),
        _row("2026-06-20", "d", 0, ok=False),  # failures contribute neither count nor bytes
    ])

    with caplog.at_level("INFO", logger="ptcg_mine.download"):
        _log_download_summary(manifest, elapsed=4.0, n_ok=3, n_fail=1, n_rate_limited=0)

    lines = [r.getMessage() for r in caplog.records]
    assert lines, "no log records captured -- the summary would be silent in a real run"
    joined = "\n".join(lines)

    assert "complete in 4.0s — 3 ok, 1 failed (0 rate-limited)" in joined
    # 3 ok rows, 4 MiB total -- the failed row is excluded from both
    assert "3 file(s), 4.0 MB" in joined
    assert "1.00 MB/s, 0.8 file(s)/s" in joined
    # per-day breakdown, one line per day, sorted
    assert "2026-06-19   2 file(s), 2.0 MB" in joined
    assert "2026-06-20   1 file(s), 2.0 MB" in joined
    assert joined.index("2026-06-19") < joined.index("2026-06-20")


def test_log_download_summary_survives_empty_manifest(caplog):
    from ptcg_mine.download import _log_download_summary

    manifest = _summary_manifest([])

    with caplog.at_level("INFO", logger="ptcg_mine.download"):
        _log_download_summary(manifest, elapsed=1.5, n_ok=0, n_fail=0, n_rate_limited=0)

    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "0 file(s), 0 B" in joined
    assert "per day" not in joined


def test_log_download_summary_omits_throughput_when_elapsed_is_zero(caplog):
    from ptcg_mine.download import _log_download_summary

    manifest = _summary_manifest([_row("2026-06-19", "a", 2048)])

    with caplog.at_level("INFO", logger="ptcg_mine.download"):
        _log_download_summary(manifest, elapsed=0.0, n_ok=1, n_fail=0, n_rate_limited=0)

    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "1 file(s), 2.0 KB" in joined
    assert "throughput" not in joined  # no ZeroDivisionError, no infinite rate


def test_download_corpus_logs_summary(tmp_path, manifest_csv, caplog):
    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(4)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(4)],
    }
    api = FakeApi(files_by_slug, page_size=4)
    config = _config(tmp_path, manifest_csv)

    with caplog.at_level("INFO", logger="ptcg_mine.download"):
        result = download_corpus(config, api)

    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "fetched now:" in joined
    assert f"{len(result)} file(s)" in joined
    assert "per day:" in joined
    for day in sorted(set(result["day"])):
        assert day in joined


def test_download_corpus_silences_sdk_stdout(tmp_path, manifest_csv, capsys):
    """The Kaggle SDK prints "Dataset URL: ..." once per file from inside
    dataset_download_file, ignoring its own quiet= flag. None of it may reach
    the user's terminal."""

    class ChattyApi(FakeApi):
        def dataset_download_file(self, slug, file_name, path=None):
            print(f"Dataset URL: https://www.kaggle.com/datasets/{slug}")
            return super().dataset_download_file(slug, file_name, path=path)

    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(4)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(4)],
    }
    api = ChattyApi(files_by_slug, page_size=4)
    config = _config(tmp_path, manifest_csv)

    result = download_corpus(config, api)

    assert len(api.download_calls) > 0, "no download ran -- the assertion below is vacuous"
    assert "Dataset URL" not in capsys.readouterr().out
    assert result["ok"].all()  # silencing stdout must not break the downloads


# ---------------------------------------------------------------------------
# listing cache + resume (a rerun must fetch episodes it does not already have)
# ---------------------------------------------------------------------------


def test_max_list_pages_follows_the_quota():
    """A fixed 100-page cap silently truncated every day at ~2000 files, so a
    run asking for 4000/day got half of what it asked for."""
    from ptcg_mine.download import _max_list_pages

    assert _max_list_pages(100) == 100          # floor holds for small runs
    assert _max_list_pages(500) == 100
    assert _max_list_pages(4000) == 800         # 4 x ceil(4000/20)
    assert _max_list_pages(4000) * 20 > 4000, "budget cannot reach the quota"
    assert _max_list_pages(1) == 100


def test_rerun_fetches_files_it_does_not_have(tmp_path, manifest_csv):
    """The reported bug: run twice, and the second run must fetch NEW episodes
    rather than re-queueing the ones already on disk."""
    day = "2026-06-20"
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    files_by_slug = {slug: [f"ep{i:03d}.json" for i in range(20)]}
    config = _config(tmp_path, manifest_csv, days=[day], n_days=1, target_episodes=5)

    api1 = FakeApi(files_by_slug, page_size=4)
    first = download_corpus(config, api1)
    assert len(first) == 5
    first_names = {name for (_, name, _) in api1.download_calls}
    assert len(first_names) == 5

    api2 = FakeApi(files_by_slug, page_size=4)
    second = download_corpus(config, api2)
    second_names = {name for (_, name, _) in api2.download_calls}

    assert len(second_names) == 5, "second run fetched nothing new -- the resume bug"
    assert not (first_names & second_names), "second run re-fetched files already held"
    assert len(second) == 5
    # 10 distinct episodes on disk across the two runs
    assert len(list((config.raw_dir / day).glob("*.json"))) == 10


def test_rerun_reads_the_cache_instead_of_relisting(tmp_path, manifest_csv):
    day = "2026-06-20"
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    files_by_slug = {slug: [f"ep{i:03d}.json" for i in range(20)]}
    config = _config(tmp_path, manifest_csv, days=[day], n_days=1, target_episodes=5)

    # One page covers the whole day, so run 1 caches all 20 names and marks the
    # listing complete -- run 2's quota is then satisfiable from the cache alone
    # and it must not call the listing API at all.
    api1 = FakeApi(files_by_slug, page_size=20)
    download_corpus(config, api1)
    assert len(api1.list_calls) > 0

    api2 = FakeApi(files_by_slug, page_size=20)
    download_corpus(config, api2)
    assert api2.list_calls == [], f"re-listed despite a usable cache: {api2.list_calls}"
    assert len(api2.download_calls) == 5


def test_listing_cache_records_cursor_and_completeness(tmp_path, manifest_csv):
    day = "2026-06-20"
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    files_by_slug = {slug: [f"ep{i:03d}.json" for i in range(20)]}
    # quota 4 < 20 available, so pagination stops early and keeps a cursor
    config = _config(tmp_path, manifest_csv, days=[day], n_days=1, target_episodes=4)

    download_corpus(config, FakeApi(files_by_slug, page_size=4))

    from ptcg_mine.download import load_day_listing

    listing = load_day_listing(config.out_dir, day, slug)
    assert listing.files, "cache holds no names"
    assert not listing.complete, "listing stopped early but was marked complete"
    assert listing.next_page_token is not None, "no cursor to resume from"

    cache_file = config.out_dir / "listings" / f"{day}.json"
    assert cache_file.exists()
    # The cache must not sit in raw_dir: load_raw_episodes globs *.json there
    # and would try to parse it as an episode.
    assert not list(config.raw_dir.rglob("listings"))


def test_listing_cache_is_complete_when_the_day_is_exhausted(tmp_path, manifest_csv):
    day = "2026-06-20"
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    files_by_slug = {slug: [f"ep{i:03d}.json" for i in range(6)]}
    config = _config(tmp_path, manifest_csv, days=[day], n_days=1, target_episodes=100)

    download_corpus(config, FakeApi(files_by_slug, page_size=4))

    from ptcg_mine.download import load_day_listing

    listing = load_day_listing(config.out_dir, day, slug)
    assert listing.complete
    assert listing.next_page_token is None
    assert len(listing.files) == 6


def test_corrupt_listing_cache_degrades_to_listing_from_scratch(tmp_path, manifest_csv):
    day = "2026-06-20"
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    files_by_slug = {slug: [f"ep{i:03d}.json" for i in range(8)]}
    config = _config(tmp_path, manifest_csv, days=[day], n_days=1, target_episodes=4)

    cache_file = config.out_dir / "listings" / f"{day}.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("{ this is not json")

    api = FakeApi(files_by_slug, page_size=4)
    result = download_corpus(config, api)  # must not raise

    assert len(result) == 4
    assert len(api.list_calls) > 0, "a corrupt cache must not be trusted"


def test_listing_cache_from_another_slug_is_ignored(tmp_path, manifest_csv):
    """Cluster ids and dataset slugs both change under the pipeline; a cache
    keyed to a different dataset must not seed this one's file names."""
    import json as _json

    day = "2026-06-20"
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    files_by_slug = {slug: [f"ep{i:03d}.json" for i in range(8)]}
    config = _config(tmp_path, manifest_csv, days=[day], n_days=1, target_episodes=4)

    cache_file = config.out_dir / "listings" / f"{day}.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(_json.dumps({
        "version": 1, "slug": "someone-else/other-dataset",
        "files": ["bogus.json"], "next_page_token": None, "complete": True,
    }))

    api = FakeApi(files_by_slug, page_size=4)
    result = download_corpus(config, api)

    assert "bogus.json" not in {name for (_, name, _) in api.download_calls}
    assert len(result) == 4


def test_fully_satisfied_rerun_is_not_a_failure(tmp_path, manifest_csv):
    """Everything already on disk means zero downloads attempted -- which to
    every counter looks exactly like "every download failed"."""
    day = "2026-06-20"
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    files_by_slug = {slug: [f"ep{i:03d}.json" for i in range(4)]}
    config = _config(tmp_path, manifest_csv, days=[day], n_days=1, target_episodes=4)

    download_corpus(config, FakeApi(files_by_slug, page_size=4))

    api2 = FakeApi(files_by_slug, page_size=4)
    result = download_corpus(config, api2)  # must not raise DownloadError

    assert api2.download_calls == []
    assert result.empty


def test_already_held_files_are_reported_separately(tmp_path, manifest_csv, caplog):
    day = "2026-06-20"
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    files_by_slug = {slug: [f"ep{i:03d}.json" for i in range(20)]}
    config = _config(tmp_path, manifest_csv, days=[day], n_days=1, target_episodes=5)

    download_corpus(config, FakeApi(files_by_slug, page_size=4))
    # Re-run with a quota big enough to re-see the 5 held files plus new ones.
    config2 = _config(tmp_path, manifest_csv, days=[day], n_days=1, target_episodes=20)
    with caplog.at_level("INFO", logger="ptcg_mine.download"):
        download_corpus(config2, FakeApi(files_by_slug, page_size=4))

    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "already held:" in joined
    assert "5 file(s) skipped without an API call" in joined


def test_sample_mode_skips_held_files_but_keeps_the_same_draw(tmp_path, manifest_csv):
    """Sample mode's promise is reproducibility, so a rerun re-takes the same
    seeded draw; resuming makes it cheaper, not bigger."""
    day = "2026-06-20"
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{day}"
    files_by_slug = {slug: [f"ep{i:03d}.json" for i in range(20)]}
    config = _config(tmp_path, manifest_csv, days=[day], n_days=1,
                     target_episodes=5, list_mode="sample")

    api1 = FakeApi(files_by_slug, page_size=4)
    download_corpus(config, api1)
    first = {name for (_, name, _) in api1.download_calls}
    assert len(first) == 5

    api2 = FakeApi(files_by_slug, page_size=4)
    result = download_corpus(config, api2)

    assert api2.download_calls == [], "sample mode re-fetched its own draw"
    assert result.empty
    assert len(list((config.raw_dir / day).glob("*.json"))) == 5


def test_progress_advances_during_listing_not_only_after(tmp_path, manifest_csv):
    """Downloads run while later days are still being listed. If completions
    are only collected after the day loop, a multi-day run reports nothing for
    minutes while it is in fact working."""
    seen_during_listing = []

    class WatchingApi(FakeApi):
        def dataset_list_files(self, slug, page_token=None):
            # Record how much download progress has been observed by the time
            # the *last* day starts listing.
            if slug.endswith("2026-06-19") and page_token is None:
                seen_during_listing.append(len(self.download_calls))
            return super().dataset_list_files(slug, page_token=page_token)

    files_by_slug = {
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": [f"d20-ep{i}.json" for i in range(8)],
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19": [f"d19-ep{i}.json" for i in range(8)],
    }
    api = WatchingApi(files_by_slug, page_size=2)
    config = _config(tmp_path, manifest_csv, n_days=2, target_episodes=16)

    from ptcg_mine import download as download_mod

    advances = []
    real_advance = download_mod.ProgressReporter.advance

    def _spy(self, n=1, n_bytes=0):
        advances.append(len(advances) + 1)
        return real_advance(self, n, n_bytes)

    download_mod.ProgressReporter.advance = _spy
    try:
        result = download_corpus(config, api)
    finally:
        download_mod.ProgressReporter.advance = real_advance

    assert len(advances) == len(result), "not every completion advanced the bar"
    assert result["ok"].all()


def test_sweep_backoff_schedule_and_no_overflow():
    """The sweep has no attempt limit, so the backoff must stay finite forever.
    An unclamped 2**pass_n reaches an int too large to convert to float and
    turns 'retry patiently' into OverflowError after a few thousand passes."""
    from ptcg_mine.download import (SWEEP_BACKOFF_CAP_SECONDS,
                                    _sweep_backoff_seconds)

    assert _sweep_backoff_seconds(1) == 60.0
    assert _sweep_backoff_seconds(2) == 120.0
    assert _sweep_backoff_seconds(3) == 240.0
    assert _sweep_backoff_seconds(4) == 480.0
    assert _sweep_backoff_seconds(5) == 960.0
    assert _sweep_backoff_seconds(6) == SWEEP_BACKOFF_CAP_SECONDS
    for pass_n in (100, 10_000, 1_000_000):
        delay = _sweep_backoff_seconds(pass_n)   # must not raise
        assert delay == SWEEP_BACKOFF_CAP_SECONDS


def test_non_rate_limit_failures_are_not_retried_forever(tmp_path, manifest_csv, monkeypatch):
    """Only 429s get the endless sweep. A file that 404s would otherwise loop
    until the heat death of the universe."""
    from ptcg_mine import download

    waits = []
    monkeypatch.setattr(download, "_sleep", lambda s: waits.append(s), raising=False)
    names = [f"d20-ep{i}.json" for i in range(4)]
    files_by_slug = {"kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": names}
    # One file always fails with an ordinary error, never a 429.
    api = FakeApi(files_by_slug, page_size=4, fail_names={"d20-ep1.json"})
    config = _config(tmp_path, manifest_csv, days=["2026-06-20"], n_days=1, target_episodes=4)

    result = download_corpus(config, api)  # must terminate

    assert not [w for w in waits if w >= download.SWEEP_BACKOFF_BASE_SECONDS], (
        "a non-429 failure triggered an inter-pass backoff"
    )
    failed = result[~result["ok"]]
    assert len(failed) == 1
    assert failed.iloc[0]["episode_id"] == "d20-ep1"
    assert not failed.iloc[0]["rate_limited"]


def test_a_file_retried_across_passes_appears_once_as_a_success(
    tmp_path, manifest_csv, monkeypatch
):
    """Rows are keyed by (day, episode_id) and last write wins -- otherwise a
    file that 429'd twice then landed would show up three times, twice failed."""
    from ptcg_mine import download

    monkeypatch.setattr(download, "_sleep", lambda s: None, raising=False)
    names = [f"d20-ep{i}.json" for i in range(4)]
    files_by_slug = {"kaggle/pokemon-tcg-ai-battle-episodes-2026-06-20": names}
    api = RecoveringApi(files_by_slug, page_size=4, fail_attempts=30)
    config = _config(tmp_path, manifest_csv, days=["2026-06-20"], n_days=1, target_episodes=4)

    result = download_corpus(config, api)

    assert len(result) == 4, f"duplicate rows across passes: {result.to_dict('records')}"
    assert result["episode_id"].is_unique
    assert result["ok"].all()
    assert api.n_attempts > 30, "the throttle never lifted -- test is not testing recovery"
