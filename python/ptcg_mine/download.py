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

import contextlib
import dataclasses
import functools
import json
import logging
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# _format_size / _format_duration live in progress.py because the reporter
# needs them too; re-exported here since the download summary is their other
# caller and `ptcg_mine.download._format_size` is the established name.
from ptcg_mine.progress import ProgressReporter, _format_duration, _format_size
from ptcg_mine.sampling import deterministic_pick, select_days

log = logging.getLogger(__name__)

# Adaptive download concurrency.
#
# Kaggle throttles each connection hard -- measured at ~0.125 MB/s per stream
# against a link many times faster -- so wall-clock is set by how many streams
# run at once, not by bandwidth. A fixed 4 was leaving most of the pipe unused
# (0.5 MB/s aggregate; a 20000-episode corpus at ~4 MB/episode would have taken
# ~49 hours). The account-level ceiling before 429s is unknown and not
# documented, so rather than guess a constant, start where the old code was,
# climb while the server stays quiet, and halve on the first complaint.
CONCURRENCY_START = 4
CONCURRENCY_CEILING = 32
CONCURRENCY_STEP = 4
CONCURRENCY_FLOOR = 2
# Consecutive clean downloads before probing one step higher. Sized so a step
# costs a few seconds of evidence rather than one lucky file.
CONCURRENCY_PROBE_AFTER = 32
# After a 429, hold the reduced limit this long before probing upward again --
# otherwise the ramp walks straight back into the wall it just hit.
CONCURRENCY_COOLDOWN_SECONDS = 30.0
# One overload is one cut. Exceeding the server's limit 429s *everything then in
# flight*, so the complaints arrive as a burst; halving per complaint turned a
# single overload into 12 -> 6 -> 3 -> 2 and left the run crawling far below the
# level that had just been working. Collapse a burst into one decision.
CONCURRENCY_CUT_DEBOUNCE_SECONDS = 5.0

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
# is limited, not unlucky. End the current pass rather than letting every
# remaining file burn RATE_LIMIT_MAX_RETRIES doomed calls against the limiter --
# the sweep loop below will come back for them after a backoff.
RATE_LIMIT_ABORT_AFTER = 25

# Sweep loop: rate limiting delays a corpus, it does not decide its contents.
# Each pass downloads what it can, and whatever 429'd is retried after a wait,
# for as long as it takes. Passes are cheap to repeat -- the listing cache means
# no extra API calls, and files already on disk are never re-fetched.
SWEEP_BACKOFF_BASE_SECONDS = 60.0
SWEEP_BACKOFF_CAP_SECONDS = 1800.0
# Consecutive passes that fetched nothing at all before we start saying, loudly,
# that this no longer looks like throttling. The loop still continues -- there is
# deliberately no give-up -- but a revoked key must not hide behind a backoff.
SWEEP_STALL_WARN_AFTER = 3


def _sweep_backoff_seconds(pass_n: int) -> float:
    """Wait before pass `pass_n + 1`: 1m, 2m, 4m, 8m, 16m, then 30m.

    The exponent is clamped before it is used, not after: the sweep has no
    attempt limit, so an unclamped `2 ** pass_n` reaches an integer too large
    to convert to float and raises `OverflowError` -- turning "retry patiently
    forever" into a crash after a few thousand passes.
    """
    steps = min(max(0, pass_n - 1), _SWEEP_MAX_BACKOFF_STEPS)
    return min(SWEEP_BACKOFF_CAP_SECONDS, SWEEP_BACKOFF_BASE_SECONDS * (2**steps))


# Enough doublings to pass the cap (60s * 2^5 = 32m > 30m); beyond this the
# result is the cap regardless, so there is nothing to gain by computing it.
_SWEEP_MAX_BACKOFF_STEPS = 16

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

    def held_for(self) -> float:
        """Seconds remaining on the shared pause, 0 when not holding. Read by
        the progress display: "being throttled" and "hung" look identical
        otherwise, and that ambiguity is what made a rate-limited run
        unreadable."""
        with self._lock:
            return max(0.0, self._resume_at - time.monotonic())


class AdaptiveConcurrency:
    """How many downloads may run at once, adjusted from what the server says.

    A `ThreadPoolExecutor` cannot be resized, so the pool is created at the
    ceiling and this gates how many of its threads are actually allowed to be
    downloading. `acquire`/`release` bracket each attempt; `on_success` and
    `on_rate_limit` are the feedback.

    The policy is deliberately asymmetric -- climb slowly on evidence, cut
    immediately on a complaint. Overshooting costs a 429, and the account is
    already the shared resource that `_RateLimitGate` exists to protect.
    """

    def __init__(
        self,
        start: int = CONCURRENCY_START,
        ceiling: int = CONCURRENCY_CEILING,
        step: int = CONCURRENCY_STEP,
        floor: int = CONCURRENCY_FLOOR,
        probe_after: int = CONCURRENCY_PROBE_AFTER,
        cooldown: float = CONCURRENCY_COOLDOWN_SECONDS,
        cut_debounce: float = CONCURRENCY_CUT_DEBOUNCE_SECONDS,
    ) -> None:
        self.ceiling = max(1, ceiling)
        self.floor = max(1, min(floor, self.ceiling))
        self.step = max(1, step)
        self.probe_after = max(1, probe_after)
        self.cooldown = cooldown
        self.cut_debounce = cut_debounce
        self._limit = max(self.floor, min(start, self.ceiling))
        self.peak = self._limit
        self.n_cuts = 0
        self._active = 0
        self._clean = 0
        self._blocked_until = 0.0
        self._cut_until = 0.0
        self._cv = threading.Condition()

    @property
    def limit(self) -> int:
        with self._cv:
            return self._limit

    @property
    def active(self) -> int:
        """Downloads transferring right now."""
        with self._cv:
            return self._active

    def acquire(self) -> None:
        with self._cv:
            while self._active >= self._limit:
                self._cv.wait()
            self._active += 1

    def release(self) -> None:
        with self._cv:
            self._active -= 1
            self._cv.notify()

    def on_success(self) -> None:
        """One clean download. Steps the limit up once enough have accumulated
        and the post-429 cooldown has expired."""
        with self._cv:
            self._clean += 1
            if self._clean < self.probe_after or self._limit >= self.ceiling:
                return
            if time.monotonic() < self._blocked_until:
                return
            self._clean = 0
            self._limit = min(self.ceiling, self._limit + self.step)
            self.peak = max(self.peak, self._limit)
            log.info("download_corpus: raising concurrency to %d", self._limit)
            self._cv.notify_all()

    def on_rate_limit(self) -> None:
        """The server complained. Halve the limit and stop probing for a while.

        Threads already downloading are not interrupted -- `_active` may exceed
        the new limit until they drain, which is fine: the point is to stop
        *starting* more, and `_RateLimitGate` is what actually pauses the ones
        in flight.
        """
        with self._cv:
            now = time.monotonic()
            self._clean = 0
            self._blocked_until = now + self.cooldown
            # Every request that was in flight when the ceiling was crossed
            # reports the same overload. Cutting once per report compounds one
            # event into several halvings.
            if now < self._cut_until:
                return
            self._cut_until = now + self.cut_debounce
            # Remember the wall. Without this the ramp climbs back to the level
            # that just 429'd, gets cut again, and keeps provoking the server --
            # and since RATE_LIMIT_ABORT_AFTER kills the whole run at 25
            # rate-limited files, a couple of those cycles is a failed run
            # rather than a slow one. The failed level is out of bounds for the
            # rest of the run; the ramp converges downward instead.
            failed_at = self._limit
            self.ceiling = max(self.floor, failed_at - self.step)
            new_limit = max(self.floor, min(self._limit // 2, self.ceiling))
            if new_limit == self._limit:
                return
            self._limit = new_limit
            self.n_cuts += 1
            log.warning(
                "download_corpus: rate limited at %d — cutting concurrency to %d "
                "(ceiling now %d for the rest of this run)",
                failed_at, self._limit, self.ceiling,
            )


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


# Runaway guard on pagination -- NOT a decision about how big a day's corpus
# may be. A fixed 100-page cap was exactly that decision in disguise: pages hold
# ~20 files, so it silently truncated every day at 2000 episodes and a run asked
# for 4000/day got half, with only a warning to say so. The budget now follows
# the quota, with generous headroom for short pages, and the floor keeps small
# runs paging as far as they ever did.
_MIN_LIST_PAGES = 100
_ASSUMED_FILES_PER_PAGE = 20
_LIST_PAGE_SAFETY_FACTOR = 4


def _max_list_pages(quota: int) -> int:
    """Page budget for one day, derived from how many files that day wants."""
    needed = math.ceil(max(1, quota) / _ASSUMED_FILES_PER_PAGE)
    return max(_MIN_LIST_PAGES, _LIST_PAGE_SAFETY_FACTOR * needed)


def list_episode_files(api, slug: str, max_pages: int | None = None) -> list[str]:
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
    pages = iter_episode_file_pages(api, slug, max_pages=max_pages)
    for page_n, (json_files, _token) in enumerate(pages, start=1):
        names.extend(json_files)
        if page_n % 10 == 0 or page_n == 1:
            log.info("  listing %s: page %d, %d json files so far...", slug, page_n, len(names))
    log.info("listed %s: %d total episode files across %d page(s)", slug, len(names), page_n)
    return names


def iter_episode_file_pages(
    api, slug: str, *, start_token: str | None = None, max_pages: int | None = None
):
    """Yield `(names, next_page_token)` for each listing page as it arrives.

    Streaming the pages lets a caller start downloading after the first page
    instead of waiting out the whole listing. That matters twice over: a day
    needs only ceil(quota / page_size) pages to fill its quota, and a listing
    that dies partway to a 429 has still produced real downloads rather than
    being discarded wholesale.

    The token is yielded alongside the names so a caller can *persist the
    cursor* and resume paging on the next run instead of re-walking pages whose
    files it already has. `None` means the listing ended naturally -- that day
    is fully known and never needs listing again.
    """
    page_token = start_token
    page_n = 0
    budget = _MIN_LIST_PAGES if max_pages is None else max_pages
    while True:
        page_n += 1
        if page_n > budget:
            log.warning(
                "iter_episode_file_pages: hit max-pages limit (%d) for %s — "
                "stopping pagination",
                budget, slug,
            )
            return
        response = _list_page_with_retry(api, slug, page_token)
        files = getattr(response, "dataset_files", None) or getattr(response, "files", None) or []
        page_token = getattr(response, "next_page_token", None) or getattr(response, "nextPageToken", None)
        yield [f.name for f in files if f.name.endswith(".json")], page_token
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


def _download_with_retry(
    api, slug: str, filename: str, dest_dir: Path, gate=None, abort=None, limiter=None
) -> dict:
    """Attempt download_episode with retry/backoff. Returns a manifest row
    dict; never raises (failures are captured as ok=False).

    Transient errors and rate limits are retried on separate budgets: a 429
    gets RATE_LIMIT_MAX_RETRIES patient, Retry-After-aware waits, while an
    ordinary error keeps the original MAX_RETRIES fast retries. `gate` (a
    _RateLimitGate) shares the rate-limit pause across all workers; `abort`
    (a threading.Event) lets the run give up without draining the queue;
    `limiter` (an AdaptiveConcurrency) decides how many of us may be
    downloading at once and learns from the outcome.
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
            # The permit covers only the transfer. Holding it across the
            # backoff below would mean a rate-limited worker keeps a slot it
            # is not using, throttling the run twice over for one 429.
            if limiter is not None:
                limiter.acquire()
            try:
                path = download_episode(api, slug, filename, dest_dir)
            finally:
                if limiter is not None:
                    limiter.release()
            if limiter is not None:
                limiter.on_success()
            return _row(True, path, path.stat().st_size)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: recorded, not raised
            last_error = exc
            if _is_rate_limit(exc):
                if limiter is not None:
                    limiter.on_rate_limit()
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


# ---------------------------------------------------------------------------
# Per-day listing cache
# ---------------------------------------------------------------------------
#
# Without this, a rerun re-walks the same first N pages, finds the same files it
# already has, fills the quota with them, downloads nothing, and the corpus can
# never grow past what the first run managed. The cache records what a day's
# listing has told us so far plus where pagination stopped, so the next run
# resumes at the cursor instead of at page 1.
#
# It lives under `out_dir`, not `raw_dir`, for two reasons: `load_raw_episodes`
# walks `raw_dir` for `*.json` and would try to parse a cache file as an
# episode, and `stamp.py` fingerprints the raw corpus, so a file that changes
# every run would invalidate Phase 2's cache each time.

LISTING_CACHE_DIRNAME = "listings"
LISTING_CACHE_VERSION = 1


@dataclasses.dataclass
class DayListing:
    """What we know about one day's file listing, and where we stopped."""

    files: list[str] = dataclasses.field(default_factory=list)
    next_page_token: str | None = None
    complete: bool = False

    @property
    def known(self) -> int:
        return len(self.files)


def _listing_cache_path(out_dir, day: str) -> Path:
    return Path(out_dir) / LISTING_CACHE_DIRNAME / f"{day}.json"


def load_day_listing(out_dir, day: str, slug: str) -> DayListing:
    """Read the cached listing for `day`, or an empty one.

    Never raises. A cache that is missing, truncated, from another schema
    version, or from a different dataset slug degrades to "we know nothing
    about this day" -- listing from scratch is slow, but a hard failure here
    would block a download that is otherwise perfectly able to proceed.
    """
    path = _listing_cache_path(out_dir, day)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return DayListing()
    if (
        not isinstance(record, dict)
        or record.get("version") != LISTING_CACHE_VERSION
        or record.get("slug") != slug
    ):
        log.info("download_corpus: %s — ignoring unusable listing cache at %s", day, path)
        return DayListing()
    files = record.get("files")
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        return DayListing()
    token = record.get("next_page_token")
    return DayListing(
        files=files,
        next_page_token=token if isinstance(token, str) else None,
        complete=bool(record.get("complete")),
    )


def save_day_listing(out_dir, day: str, slug: str, listing: DayListing) -> None:
    """Persist `listing` for `day`. Written to a temp file and renamed, so an
    interrupted run cannot leave a half-written cache that the next run would
    read as authoritative. Failures are logged, never raised: losing the cache
    costs listing calls, not correctness."""
    path = _listing_cache_path(out_dir, day)
    record = {
        "version": LISTING_CACHE_VERSION,
        "slug": slug,
        "files": listing.files,
        "next_page_token": listing.next_page_token,
        "complete": listing.complete,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(path)
    except OSError as exc:
        log.warning("download_corpus: %s — could not save listing cache: %s", day, exc)


def _owned_episode_files(dest_dir) -> set[str]:
    """Episode files already on disk for a day. These must not consume quota:
    counting them is what made a rerun queue 4000 files it already had, download
    none of them, and report itself finished."""
    dest_dir = Path(dest_dir)
    if not dest_dir.is_dir():
        return set()
    return {p.name for p in dest_dir.glob("*.json") if p.is_file()}


def _queue_day_stream(api, slug, listing, owned, quota, page_budget, submit, abort):
    """Queue up to `quota` not-yet-owned files for one day, cache first.

    Two passes. The first spends no API calls at all: it walks the names the
    cache already holds. Only if that leaves the quota unmet does the second
    resume pagination *from the saved cursor*, which is the whole point --
    re-walking from page 1 finds the same owned files every run and is how a
    corpus gets permanently stuck at whatever the first run managed.

    Mutates `listing` in place (both the names and the cursor) so the caller
    persists whatever was learned, including after a mid-listing failure.
    Returns `(n_queued, n_owned, error)`; the error is returned rather than
    raised because a day that fails partway has still produced real downloads.
    """
    n_queued = 0
    n_owned = 0

    def _take(name):
        nonlocal n_queued, n_owned
        if name in owned:
            n_owned += 1
            return
        submit(name)
        n_queued += 1

    for name in listing.files:
        if n_queued >= quota:
            break
        _take(name)

    if n_queued >= quota or listing.complete:
        return n_queued, n_owned, None

    known = set(listing.files)
    try:
        pages = iter_episode_file_pages(
            api, slug, start_token=listing.next_page_token, max_pages=page_budget
        )
        for names, token in pages:
            for name in names:
                if name in known:
                    continue
                known.add(name)
                # Recorded even past quota: names are cheap and this is the
                # cache the next run reads instead of paging.
                listing.files.append(name)
                if n_queued < quota:
                    _take(name)
            listing.next_page_token = token
            listing.complete = token is None
            if n_queued >= quota or listing.complete or abort.is_set():
                break
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not fatal
        return n_queued, n_owned, exc
    return n_queued, n_owned, None


def _queue_day_sample(api, slug, listing, owned, quota, seed, page_budget, submit):
    """Queue one day's seeded random draw, skipping files already held.

    Sample mode's promise is a *reproducible* draw, so unlike stream mode the
    quota is not refilled with new files on a rerun: the draw is taken over the
    day's whole listing and the same run picks the same episodes. What resuming
    buys here is not a bigger corpus but a cheaper one -- already-downloaded
    members of the draw are simply not re-requested. Raise --target-episodes to
    grow a sample-mode corpus.
    """
    error = None
    if not listing.complete:
        known = set(listing.files)
        try:
            pages = iter_episode_file_pages(
                api, slug, start_token=listing.next_page_token, max_pages=page_budget
            )
            for names, token in pages:
                for name in names:
                    if name not in known:
                        known.add(name)
                        listing.files.append(name)
                listing.next_page_token = token
                listing.complete = token is None
                if listing.complete:
                    break
        except Exception as exc:  # noqa: BLE001 - reported to the caller, not fatal
            error = exc

    if not listing.files:
        return 0, 0, error

    picked = deterministic_pick(listing.files, quota, seed)
    n_queued = 0
    n_owned = 0
    for name in picked:
        if name in owned:
            n_owned += 1
            continue
        submit(name)
        n_queued += 1
    return n_queued, n_owned, error


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


@contextlib.contextmanager
def _quiet_sdk_stdout():
    """Swallow whatever the Kaggle SDK prints for the duration of the block.

    `dataset_download_file(..., quiet=True)` does *not* cover this: the SDK
    calls `_print_dataset_url_and_license` unconditionally, with a bare
    `print()`, once per file — so a 2300-episode run buries its own summary
    under 2300 "Dataset URL: ..." lines. Redirecting the module-global
    `sys.stdout` is the only lever, and it reaches all MAX_WORKERS threads
    because they share it.

    Safe only because nothing *we* want to see goes to stdout from inside:
    this module reports exclusively through `log` (stderr under the CLI's
    basicConfig), and every `print` in `ptcg_mine.mine` runs after Phase 1
    has returned.
    """
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def _log_download_summary(
    manifest: pd.DataFrame, elapsed: float, n_ok: int, n_fail: int,
    n_rate_limited: int, n_owned: int = 0, limiter=None,
) -> None:
    """Report what Phase 1 actually moved: file count, bytes and throughput,
    overall and per day.

    A pure function of the finished manifest, so it can be tested without a
    thread pool. These numbers are genuinely *this run's*: files already on
    disk are filtered out before anything is queued, so they never reach the
    manifest at all and are reported separately as `n_owned`. (That split used
    to be impossible -- owned files were submitted, hit `download_episode`'s
    resume path, and landed in the manifest indistinguishable from a real
    fetch. Not queueing them is what made the distinction free.)
    """
    log.info(
        "download_corpus: complete in %s — %d ok, %d failed (%d rate-limited)",
        _format_duration(elapsed), n_ok, n_fail, n_rate_limited,
    )
    ok_rows = manifest[manifest["ok"]] if not manifest.empty else manifest
    total_bytes = int(ok_rows["bytes"].sum()) if not ok_rows.empty else 0
    log.info(
        "  fetched now:    %d file(s), %s", len(ok_rows), _format_size(total_bytes)
    )
    if n_owned:
        log.info(
            "  already held:   %d file(s) skipped without an API call", n_owned
        )
    if elapsed > 0:
        log.info(
            "  throughput:     %.2f MB/s, %.1f file(s)/s",
            total_bytes / elapsed / (1024.0 * 1024.0), len(ok_rows) / elapsed,
        )
    if limiter is not None:
        # Where the ramp ended up is the number to reach for if this ever needs
        # tuning by hand -- and n_cuts says whether the ceiling was the server's
        # or ours.
        log.info(
            "  concurrency:    settled at %d (peak %d, ceiling %d, %d cut%s on 429)",
            limiter.limit, limiter.peak, limiter.ceiling,
            limiter.n_cuts, "" if limiter.n_cuts == 1 else "s",
        )
    if ok_rows.empty:
        return
    log.info("  per day:")
    for day, group in ok_rows.groupby("day", sort=True):
        log.info(
            "    %s   %d file(s), %s",
            day, len(group), _format_size(int(group["bytes"].sum())),
        )


@dataclasses.dataclass
class _BatchResult:
    """What one download pass produced."""

    rows: list[dict]
    retry: list[tuple[str, str]]  # (day, filename) that 429'd and deserve another pass
    n_ok: int
    n_fail: int
    n_rate_limited: int
    n_submitted: int


def _download_batch(api, gate, limiter, planned: int, description: str, queue_fn) -> _BatchResult:
    """Run one pass: `queue_fn(submit, abort)` queues the work, this owns the
    pool, the progress display and the collection of results.

    Factored out so the sweep loop can repeat it verbatim. `queue_fn` differs
    between the first pass (walk the days, page the listings, submit as it
    goes) and the retries (re-submit a known list of files), but everything
    around it -- concurrency, rate-limit feedback, reporting -- is identical.
    """
    rows: list[dict] = []
    retry: list[tuple[str, str]] = []
    n_ok = n_fail = n_rate_limited = 0
    lock = threading.Lock()
    abort = threading.Event()

    def _status() -> str:
        """What the downloader is doing right now, for the progress line.

        Exists because a rate-limited run and a hung run were indistinguishable:
        the bar only moved on completion, so being throttled looked like being
        stuck at "1/20000" for minutes.
        """
        held = gate.held_for()
        if held > 0:
            return f"rate limited, resuming in {held:.0f}s"
        failed = f", {n_fail} failed" if n_fail else ""
        return f"{limiter.active} in flight, limit {limiter.limit}{failed}"

    reporter = ProgressReporter(
        description, planned, logger=log, unit="file", track_bytes=True,
        status_fn=_status,
    )

    def _on_done(fut, day: str) -> None:
        """Record one finished download, on the worker thread that finished it.

        A single `as_completed` pass after the queueing loop could not begin
        until every day had been listed, so a five-day run reported nothing for
        its first several minutes while downloads were already running.
        """
        nonlocal n_ok, n_fail, n_rate_limited
        row = fut.result()  # _download_with_retry captures failures; never raises
        row["day"] = day
        with lock:
            rows.append(row)
            reporter.advance(1, int(row["bytes"]) if row["ok"] else 0)
            if row["ok"]:
                n_ok += 1
                return
            n_fail += 1
            if not row.get("rate_limited"):
                return  # a real error: retrying it forever would loop on a 404
            n_rate_limited += 1
            retry.append((day, f"{row['episode_id']}.json"))
            if n_rate_limited >= RATE_LIMIT_ABORT_AFTER and not abort.is_set():
                abort.set()
                log.warning(
                    "download_corpus: %d file(s) rate-limited — ending this pass early; "
                    "they are retried after a backoff, nothing is dropped.",
                    n_rate_limited,
                )

    n_submitted = 0
    # The pool is created at the ceiling; `limiter` decides how many of its
    # threads may actually be downloading, because a ThreadPoolExecutor cannot
    # be resized after construction. It is nested *inside* the stdout redirect
    # so its __exit__ joins every worker before stdout is restored -- a thread
    # still mid-print as the redirect unwound would write to a closed devnull.
    # The reporter is outermost: it owns the log handlers and must outlive both.
    with reporter, _quiet_sdk_stdout(), ThreadPoolExecutor(
        max_workers=CONCURRENCY_CEILING
    ) as pool:
        def _submit(day: str, slug: str, dest_dir: Path, filename: str) -> None:
            fut = pool.submit(
                _download_with_retry, api, slug, filename, dest_dir, gate, abort, limiter
            )
            fut.add_done_callback(functools.partial(_on_done, day=day))

        n_submitted = queue_fn(_submit, abort)
        log.info("download_corpus: %d episode(s) submitted — downloading...", n_submitted)
        reporter.set_total(n_submitted)

    return _BatchResult(rows, retry, n_ok, n_fail, n_rate_limited, n_submitted)


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
    limiter = AdaptiveConcurrency()
    listing_errors: dict[str, Exception] = {}
    partial_listings: dict[str, Exception] = {}
    n_owned_total = 0

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sized from config, then corrected: in stream mode listing is interleaved
    # with downloading, so the true count does not exist until every day has
    # been listed. Starting from the planned figure means the bar is honest
    # about intent from the first frame rather than sitting at an unknown total.
    planned = quota * len(days)
    page_budget = _max_list_pages(quota)

    def _queue_days(submit, abort) -> int:
        """First pass: walk the days, page each listing, submit as names arrive."""
        nonlocal n_owned_total
        n_submitted = 0
        for day in days:
            if abort.is_set():
                log.warning("download_corpus: pass ended early — not listing remaining day(s)")
                break
            slug = config.dataset_prefix + day
            dest_dir = Path(config.raw_dir) / day

            def _one(filename, _slug=slug, _dest=dest_dir, _day=day):
                submit(_day, _slug, _dest, filename)

            listing = load_day_listing(out_dir, day, slug)
            owned = _owned_episode_files(dest_dir)
            log.info(
                "download_corpus: %s — %d on disk, %d name(s) cached%s",
                day, len(owned), listing.known,
                " (listing complete)" if listing.complete else "",
            )

            try:
                if list_mode == "stream":
                    n_queued, n_owned, error = _queue_day_stream(
                        api, slug, listing, owned, quota, page_budget, _one, abort
                    )
                else:
                    n_queued, n_owned, error = _queue_day_sample(
                        api, slug, listing, owned, quota, config.seed, page_budget, _one
                    )
            finally:
                # Whatever pagination learned is worth keeping even if it then
                # failed: the next run resumes from here rather than page 1.
                save_day_listing(out_dir, day, slug, listing)

            if error is not None:
                if n_queued == 0:
                    listing_errors[day] = error
                    log.warning(
                        "download_corpus: skipping day %s -- listing failed before any "
                        "file was queued: %s%s",
                        day, error, " (rate limited)" if _is_rate_limit(error) else "",
                    )
                    continue
                partial_listings[day] = error
                log.warning(
                    "download_corpus: day %s -- listing stopped early after %d file(s) "
                    "queued: %s%s",
                    day, n_queued, error, " (rate limited)" if _is_rate_limit(error) else "",
                )

            n_owned_total += n_owned
            n_submitted += n_queued
            log.info(
                "download_corpus: %s — %d new episode(s) queued, %d already held",
                day, n_queued, n_owned,
            )
        return n_submitted

    def _queue_retries(targets):
        """Later passes: re-submit exactly the files that were rate-limited."""
        def _queue(submit, abort) -> int:
            for day, filename in targets:
                if abort.is_set():
                    break
                submit(day, config.dataset_prefix + day, Path(config.raw_dir) / day, filename)
            return len(targets)
        return _queue

    # The sweep. Rate limiting decides *when* the corpus arrives, not what is in
    # it, so a 429'd file is retried after a wait rather than dropped. There is
    # deliberately no attempt limit: the loop ends when nothing is left that a
    # retry could fix.
    t_start = time.monotonic()
    rows_by_key: dict[tuple[str, str], dict] = {}
    targets: list[tuple[str, str]] = []
    pass_n = 0
    n_barren_passes = 0

    while True:
        pass_n += 1
        if pass_n == 1:
            result = _download_batch(api, gate, limiter, planned, "downloading", _queue_days)
        else:
            log.info("download_corpus: pass %d — retrying %d rate-limited file(s)",
                     pass_n, len(targets))
            result = _download_batch(
                api, gate, limiter, len(targets), f"downloading (pass {pass_n})",
                _queue_retries(targets),
            )

        # Last write wins: a file that 429'd in pass 1 and succeeded in pass 3
        # must appear once, as a success.
        for row in result.rows:
            rows_by_key[(row["day"], row["episode_id"])] = row

        targets = result.retry
        if not targets:
            break

        n_barren_passes = n_barren_passes + 1 if result.n_ok == 0 else 0
        if n_barren_passes >= SWEEP_STALL_WARN_AFTER:
            # No give-up, by request -- but a revoked key or a withdrawn dataset
            # must not look like patience. Said at ERROR so it survives a log tail.
            sample = next((r.get("error") for r in result.rows if r.get("error")), None)
            log.error(
                "download_corpus: %d consecutive pass(es) fetched nothing while %d "
                "file(s) keep being rate-limited. This may no longer be throttling — "
                "check credentials and that the dataset still exists. Last error: %s. "
                "Still retrying (interrupt to stop; everything fetched is kept).",
                n_barren_passes, len(targets), sample,
            )

        delay = _sweep_backoff_seconds(pass_n)
        log.warning(
            "download_corpus: pass %d — %d file(s) still rate-limited, waiting %s "
            "before pass %d",
            pass_n, len(targets), _format_duration(delay), pass_n + 1,
        )
        _sleep(delay)

    elapsed = time.monotonic() - t_start
    rows = list(rows_by_key.values())
    # Counted from the final state of each file, not summed across passes: a
    # file that failed twice and then succeeded is one success, not two failures
    # and a success.
    n_ok = sum(1 for r in rows if r["ok"])
    n_fail = len(rows) - n_ok
    n_rate_limited = sum(1 for r in rows if not r["ok"] and r.get("rate_limited"))
    if pass_n > 1:
        log.info("download_corpus: finished after %d pass(es)", pass_n)

    manifest = pd.DataFrame(
        rows, columns=["day", "episode_id", "path", "bytes", "ok", "error", "rate_limited", "skipped"]
    )
    if not manifest.empty:
        manifest = manifest.sort_values(["day", "episode_id"]).reset_index(drop=True)

    # Persist before any raise: partial progress must survive so the next run
    # (downloads are resumable) can pick up where this one stopped.
    manifest.to_parquet(out_dir / "downloaded.parquet", index=False)

    # Before _check_download_outcome, which raises: a run cut short by rate
    # limiting should still say what it managed to fetch.
    _log_download_summary(
        manifest, elapsed, n_ok, n_fail, n_rate_limited, n_owned_total, limiter
    )

    if partial_listings:
        log.warning(
            "download_corpus: %d day(s) were listed only partially (fewer than the "
            "%d-episode quota queued): %s",
            len(partial_listings), quota, ", ".join(sorted(partial_listings)),
        )

    _check_download_outcome(
        days, listing_errors, n_ok, n_fail, n_rate_limited, n_owned_total
    )
    return manifest


def _check_download_outcome(
    days, listing_errors, n_ok, n_fail, n_rate_limited, n_owned=0
) -> None:
    """Fail the run here, at the point of failure, when Phase 1 did not produce
    a usable corpus. Without this the pipeline sails on and only dies three
    phases later at expert selection with a misleading "the corpus is empty".

    `n_owned` is what keeps a *fully satisfied* run from being read as a failed
    one. Now that already-downloaded files are never queued, the happy case of
    "the corpus is complete, there was nothing left to fetch" produces zero
    attempts and zero successes -- identical, to every counter here, to "every
    download failed".
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
        if n_owned:
            log.info(
                "download_corpus: nothing to fetch — all %d requested episode(s) "
                "were already on disk.",
                n_owned,
            )
            return
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

    # There is no longer an "aborted by rate limiting" outcome: the sweep loop
    # only returns once nothing is left that another pass could fix, so a run
    # that reaches here was not cut short by 429s.

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
