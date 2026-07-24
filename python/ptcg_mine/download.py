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

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ptcg_mine.sampling import deterministic_pick, select_days

log = logging.getLogger(__name__)

MAX_WORKERS = 4
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.05

# Listing is the only unguarded, main-thread Kaggle call: a single rate-limited
# (HTTP 429) listing must not crash the whole run. It gets its own, more patient
# retry policy (Kaggle rate-limits want seconds, not milliseconds).
LIST_MAX_RETRIES = 5
LIST_BACKOFF_BASE_SECONDS = 1.0
LIST_BACKOFF_CAP_SECONDS = 60.0

# Rate limiting is a different animal from a transient error: the server is
# telling us to come back later, so 429s get their own, larger attempt budget
# and the patient listing-style backoff (seconds, Retry-After aware).
RATE_LIMIT_MAX_RETRIES = 6

# After this many downloads have failed *because of rate limiting*, the account
# is limited, not unlucky. Stop submitting work rather than letting every
# remaining file burn RATE_LIMIT_MAX_RETRIES doomed calls against the limiter.
RATE_LIMIT_ABORT_AFTER = 25

# Indirection point so tests can stub out real waiting.
_sleep = time.sleep


class DownloadError(RuntimeError):
    """Raised when Phase 1 could not produce a usable corpus (nothing listed,
    nothing downloaded, or sustained rate limiting). Raised at the point of
    failure so the run stops here, rather than surfacing much later as an
    empty-corpus error from expert selection."""


def _retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort read of a `Retry-After` header (in seconds) from an HTTP
    error, if the exception exposes a `.response` with headers. Returns None
    when unavailable or unparseable."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_rate_limit(exc: Exception) -> bool:
    """True if `exc` looks like an HTTP 429. Checks the structured response
    first, then falls back to the message text because the Kaggle SDK raises
    several exception types that carry the status only in their string form."""
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


class _RateLimitGate:
    """A process-wide cooperative pause shared by every download worker.

    A 429 is a statement about the *account*, not about one request: when one
    worker is limited, all the others are too. Without a shared gate each of
    MAX_WORKERS threads independently burns its retry budget hammering a server
    that has already said "stop", which is exactly how a real run turned one
    rate-limit into 425 failed downloads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resume_at = 0.0
        self.n_trips = 0

    def trip(self, seconds: float) -> None:
        """Record that we were rate-limited; hold all workers for `seconds`."""
        with self._lock:
            self._resume_at = max(self._resume_at, time.monotonic() + seconds)
            self.n_trips += 1

    def wait(self) -> None:
        """Block until the shared pause expires (a single sleep, never a spin)."""
        with self._lock:
            delay = self._resume_at - time.monotonic()
        if delay > 0:
            _sleep(delay)


def _list_backoff_seconds(attempt: int, exc: Exception) -> float:
    """Seconds to wait before the next listing retry: honor Retry-After when
    present, else exponential backoff with jitter, capped."""
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return retry_after
    delay = min(LIST_BACKOFF_CAP_SECONDS, LIST_BACKOFF_BASE_SECONDS * (2**attempt))
    return delay * (0.5 + random.random() * 0.5)


def _list_page_with_retry(api, slug: str, page_token):
    """One `dataset_list_files` page call with retry/backoff. Re-raises the
    last error only after LIST_MAX_RETRIES exhausted attempts."""
    last_error: Exception | None = None
    for attempt in range(LIST_MAX_RETRIES):
        try:
            return api.dataset_list_files(slug, page_token=page_token)
        except Exception as exc:  # noqa: BLE001 - transient API/HTTP errors are retried
            last_error = exc
            if attempt < LIST_MAX_RETRIES - 1:
                _sleep(_list_backoff_seconds(attempt, exc))
    raise last_error  # type: ignore[misc]


# Safety cap: a day with >100 pages (~10k files at 100/page) is pathological;
# bail rather than loop forever if the API keeps returning a page token.
_MAX_LIST_PAGES = 100


def list_episode_files(api, slug: str) -> list[str]:
    """Page through api.dataset_list_files(slug, page_token=...), collecting
    all "<id>.json" file names until there is no next page token.

    Each page call is retried with backoff (see `_list_page_with_retry`); if a
    page ultimately fails after all retries, the error propagates so the caller
    can skip the whole day rather than crash the run.

    Duck-typed against both response shapes seen in the wild: the brief's
    camelCase (`.files` / `.nextPageToken`) and the currently-vendored real
    `kaggle` SDK's snake_case `ApiListDatasetFilesResponse`
    (`.dataset_files` / `.next_page_token`).
    """
    names: list[str] = []
    page_n = 0
    for page_n, json_files in enumerate(iter_episode_file_pages(api, slug), start=1):
        names.extend(json_files)
        if page_n % 10 == 0 or page_n == 1:
            log.info("  listing %s: page %d, %d json files so far...", slug, page_n, len(names))
    log.info("listed %s: %d total episode files across %d page(s)", slug, len(names), page_n)
    return names


def iter_episode_file_pages(api, slug: str):
    """Yield each listing page's "<id>.json" names as they arrive.

    Streaming the pages lets a caller start downloading after the first page
    instead of waiting out the whole listing. That matters twice over: a day
    needs only ceil(quota / page_size) pages to fill its quota (pages hold ~20
    files, so 500 episodes needs ~25 pages, not the _MAX_LIST_PAGES=100 cap),
    and a listing that dies partway to a 429 has still produced real downloads
    rather than being discarded wholesale.
    """
    page_token = None
    page_n = 0
    while True:
        page_n += 1
        if page_n > _MAX_LIST_PAGES:
            log.warning(
                "iter_episode_file_pages: hit max-pages limit (%d) for %s — "
                "stopping pagination",
                _MAX_LIST_PAGES, slug,
            )
            return
        response = _list_page_with_retry(api, slug, page_token)
        files = getattr(response, "dataset_files", None) or getattr(response, "files", None) or []
        yield [f.name for f in files if f.name.endswith(".json")]
        page_token = getattr(response, "next_page_token", None) or getattr(response, "nextPageToken", None)
        if not page_token:
            return


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


def _download_with_retry(api, slug: str, filename: str, dest_dir: Path, gate=None, abort=None) -> dict:
    """Attempt download_episode with retry/backoff. Returns a manifest row
    dict; never raises (failures are captured as ok=False).

    Transient errors and rate limits are retried on separate budgets: a 429
    gets RATE_LIMIT_MAX_RETRIES patient, Retry-After-aware waits, while an
    ordinary error keeps the original MAX_RETRIES fast retries. `gate` (a
    _RateLimitGate) shares the rate-limit pause across all workers; `abort`
    (a threading.Event) lets the run give up without draining the queue.
    """
    episode_id = _episode_id(filename)

    def _row(ok, path, size, error=None, rate_limited=False, skipped=False):
        return {
            "episode_id": episode_id,
            "path": str(path),
            "bytes": size,
            "ok": ok,
            "error": error,
            "rate_limited": rate_limited,
            "skipped": skipped,
        }

    if abort is not None and abort.is_set():
        return _row(False, dest_dir / filename, 0,
                    error="skipped: run aborted after sustained rate limiting",
                    rate_limited=True, skipped=True)

    last_error: Exception | None = None
    n_transient = 0
    n_limited = 0
    while n_transient < MAX_RETRIES and n_limited < RATE_LIMIT_MAX_RETRIES:
        if gate is not None:
            gate.wait()
        try:
            path = download_episode(api, slug, filename, dest_dir)
            return _row(True, path, path.stat().st_size)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: recorded, not raised
            last_error = exc
            if _is_rate_limit(exc):
                n_limited += 1
                if n_limited >= RATE_LIMIT_MAX_RETRIES:
                    break
                delay = _list_backoff_seconds(n_limited - 1, exc)
                if gate is not None:
                    gate.trip(delay)
                    gate.wait()
                else:
                    _sleep(delay)
            else:
                n_transient += 1
                if n_transient >= MAX_RETRIES:
                    break
                _sleep(RETRY_BACKOFF_SECONDS * n_transient)

    return _row(
        False,
        dest_dir / filename,
        0,
        error=str(last_error) if last_error is not None else None,
        rate_limited=last_error is not None and _is_rate_limit(last_error),
    )


DAY_ORDERS = ("recent-first", "oldest-first", "as-selected")

# "stream": page the listing and queue each page immediately, stopping at quota.
#   Fewest API calls and downloads start at once, but the sample is whatever
#   Kaggle lists first (episode-id order) rather than a seeded random draw.
# "sample": list the whole day, then deterministic_pick(quota, seed) from it.
#   A reproducible unbiased draw, at the cost of listing every page up front.
LIST_MODES = ("stream", "sample")


def _order_days(days: list[str], order: str) -> list[str]:
    """Order the selected days for fetching.

    `select_days` returns oldest-to-newest, which is the worst order under rate
    limiting: a run that dies partway keeps only the stalest days. Days are ISO
    `YYYY-MM-DD`, so lexicographic sort is chronological.

    Ordering affects only *when* a day is fetched, never which episodes are
    picked -- `deterministic_pick` is seeded per-day and order-independent.
    """
    if order == "as-selected":
        return days
    if order not in DAY_ORDERS:
        raise ValueError(f"day_order must be one of {DAY_ORDERS}, got {order!r}")
    return sorted(days, reverse=(order == "recent-first"))


def download_corpus(config, api) -> pd.DataFrame:
    """Sample and download episodes for config.n_days days, config.target_episodes
    total (target_episodes // n_days per day), using a bounded thread pool with
    retry/backoff. Writes and returns a manifest DataFrame with columns
    [day, episode_id, path, bytes, ok, error, rate_limited, skipped].

    Raises DownloadError if the phase produced no usable corpus (no day could be
    listed, nothing downloaded, or the run was aborted under sustained rate
    limiting). The manifest is always written first, so a failed run still
    leaves resumable progress on disk.
    """
    if config.days is not None:
        days = list(config.days)
    else:
        manifest_df = pd.read_csv(config.manifest_csv)
        days = select_days(manifest_df, config.n_days)
    days = _order_days(days, getattr(config, "day_order", "recent-first"))
    list_mode = getattr(config, "list_mode", "stream")
    if list_mode not in LIST_MODES:
        raise ValueError(f"list_mode must be one of {LIST_MODES}, got {list_mode!r}")

    quota = max(1, config.target_episodes // max(config.n_days, 1))
    log.info(
        "download_corpus: %d day(s) to fetch (%s, list-mode=%s, %d episode(s)/day): %s",
        len(days), getattr(config, "day_order", "recent-first"), list_mode, quota, ", ".join(days),
    )

    gate = _RateLimitGate()
    abort = threading.Event()
    listing_errors: dict[str, Exception] = {}
    partial_listings: dict[str, Exception] = {}
    rows: list[dict] = []
    n_ok = 0
    n_fail = 0
    n_rate_limited = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures: dict = {}
        for day in days:
            if abort.is_set():
                log.warning("download_corpus: aborted — not listing remaining day(s)")
                break
            slug = config.dataset_prefix + day
            dest_dir = Path(config.raw_dir) / day
            log.info("download_corpus: listing files for %s...", day)

            def _submit(filename, _slug=slug, _dest=dest_dir, _day=day):
                fut = pool.submit(_download_with_retry, api, _slug, filename, _dest, gate, abort)
                futures[fut] = _day

            n_queued = 0
            if list_mode == "stream":
                # Queue each page as it arrives and stop paging at quota, so
                # downloads start immediately and we never list more than needed.
                try:
                    for page in iter_episode_file_pages(api, slug):
                        for filename in page:
                            if n_queued >= quota:
                                break
                            _submit(filename)
                            n_queued += 1
                        if n_queued >= quota or abort.is_set():
                            break
                except Exception as exc:  # noqa: BLE001 - recorded, not fatal
                    if n_queued == 0:
                        listing_errors[day] = exc
                        log.warning(
                            "download_corpus: skipping day %s -- listing failed before any "
                            "file was queued: %s%s",
                            day, exc, " (rate limited)" if _is_rate_limit(exc) else "",
                        )
                        continue
                    partial_listings[day] = exc
                    log.warning(
                        "download_corpus: day %s -- listing stopped early after %d file(s) "
                        "queued: %s%s",
                        day, n_queued, exc, " (rate limited)" if _is_rate_limit(exc) else "",
                    )
            else:
                try:
                    files = list_episode_files(api, slug)
                except Exception as exc:  # noqa: BLE001 - a day we cannot list is skipped, not fatal
                    listing_errors[day] = exc
                    log.warning(
                        "download_corpus: skipping day %s -- could not list files after "
                        "%d retries: %s%s",
                        day,
                        LIST_MAX_RETRIES,
                        exc,
                        " (rate limited)" if _is_rate_limit(exc) else "",
                    )
                    continue
                picked = deterministic_pick(files, quota, config.seed)
                log.info("download_corpus: %s — %d files listed, %d picked for download",
                         day, len(files), len(picked))
                for filename in picked:
                    _submit(filename)
                    n_queued += 1
            log.info("download_corpus: %s — %d episode(s) queued", day, n_queued)

        log.info("download_corpus: %d episode(s) submitted — downloading...", len(futures))
        for fut in as_completed(futures):
            row = fut.result()
            row["day"] = futures[fut]
            rows.append(row)
            if row["ok"]:
                n_ok += 1
                continue
            n_fail += 1
            if row.get("rate_limited"):
                n_rate_limited += 1
                # Trip the abort once rate limiting is clearly sustained rather
                # than incidental; queued tasks then return immediately.
                if n_rate_limited >= RATE_LIMIT_ABORT_AFTER and not abort.is_set():
                    abort.set()
                    log.error(
                        "download_corpus: aborting — %d downloads rate-limited (HTTP 429). "
                        "Kaggle is throttling this account; remaining files skipped.",
                        n_rate_limited,
                    )
        log.info(
            "download_corpus: downloads complete — %d ok, %d failed (%d rate-limited)",
            n_ok, n_fail, n_rate_limited,
        )

    manifest = pd.DataFrame(
        rows, columns=["day", "episode_id", "path", "bytes", "ok", "error", "rate_limited", "skipped"]
    )
    if not manifest.empty:
        manifest = manifest.sort_values(["day", "episode_id"]).reset_index(drop=True)

    # Persist before any raise: partial progress must survive so the next run
    # (downloads are resumable) can pick up where this one stopped.
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out_dir / "downloaded.parquet", index=False)

    if partial_listings:
        log.warning(
            "download_corpus: %d day(s) were listed only partially (fewer than the "
            "%d-episode quota queued): %s",
            len(partial_listings), quota, ", ".join(sorted(partial_listings)),
        )

    _check_download_outcome(days, listing_errors, n_ok, n_fail, n_rate_limited, abort.is_set())
    return manifest


def _check_download_outcome(days, listing_errors, n_ok, n_fail, n_rate_limited, aborted) -> None:
    """Fail the run here, at the point of failure, when Phase 1 did not produce
    a usable corpus. Without this the pipeline sails on and only dies three
    phases later at expert selection with a misleading "the corpus is empty".
    """
    rate_limited_days = [d for d, e in listing_errors.items() if _is_rate_limit(e)]
    hint = (
        "Kaggle is rate-limiting this account (HTTP 429). Wait ~15-60 min and rerun "
        "(downloads resume, so nothing already fetched is re-fetched), and/or lower "
        "--target-episodes / --n-days to make fewer API calls."
    )

    if not days:
        raise DownloadError(
            "no days were selected to download: --days was empty and day sampling "
            "returned nothing. Check --manifest-csv and --n-days."
        )

    if len(listing_errors) == len(days):
        detail = f" {len(rate_limited_days)} of them with HTTP 429." if rate_limited_days else ""
        raise DownloadError(
            f"all {len(days)} day(s) could not be listed.{detail} "
            f"First error: {next(iter(listing_errors.values()))}. "
            + (hint if rate_limited_days else "Check network access and Kaggle credentials.")
        )

    n_attempted = n_ok + n_fail
    if n_attempted == 0:
        raise DownloadError(
            f"nothing to download: {len(days) - len(listing_errors)} day(s) listed but "
            f"produced no episode files. Check --dataset-prefix and the day list."
        )

    if n_ok == 0:
        raise DownloadError(
            f"every download failed: {n_attempted} attempted, 0 succeeded "
            f"({n_rate_limited} rate-limited). "
            + (hint if n_rate_limited else "See the logged per-file errors above.")
        )

    if aborted:
        raise DownloadError(
            f"aborted after sustained rate limiting: only {n_ok} of {n_attempted} "
            f"download(s) succeeded ({n_rate_limited} rate-limited). " + hint
        )

    if n_fail:
        log.warning(
            "download_corpus: %d of %d download(s) failed (%d rate-limited) — "
            "the corpus is smaller than requested; rerun to resume.",
            n_fail, n_attempted, n_rate_limited,
        )
    if listing_errors:
        log.warning(
            "download_corpus: %d of %d day(s) could not be listed and were skipped: %s",
            len(listing_errors), len(days), ", ".join(sorted(listing_errors)),
        )
