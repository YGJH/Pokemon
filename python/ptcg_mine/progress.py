"""Live progress for the two long, silent stretches of a mine run.

Phase 1 (download) and Phase 2's `load_raw_episodes` both run for minutes with
no output at all on the real corpus. `ProgressReporter` covers both with one
object, and deliberately renders two different ways:

  - **On a terminal**: a `rich.progress` bar, refreshed as work completes.
  - **Everywhere else**: one throttled `log.info` line every `interval`
    seconds. `run_pipeline.sh` redirects mine's output to `logs/2-mine.log`,
    and a live bar written to a file is thousands of redraw frames -- which is
    exactly the unreadable log the summary work was meant to fix.

The TTY check is on **stderr**, not stdout: `download.py` redirects stdout to
devnull while the pool runs (to swallow the Kaggle SDK's per-file prints), so
stdout is not a usable surface there and says nothing about whether a human is
watching.
"""

import logging
import sys
import threading
import time

log = logging.getLogger(__name__)

# Non-TTY fallback: seconds between status lines. Long enough that a 40-minute
# download adds ~80 lines to the log rather than one per file.
DEFAULT_LOG_INTERVAL_SECONDS = 30.0


def _format_size(n_bytes: float) -> str:
    """Human-readable byte count (binary units, e.g. "214.6 MB")."""
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024.0 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


def _format_duration(seconds: float) -> str:
    """Human-readable wall time ("47.3s", "6m52s", "1h04m")."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _format_eta(seconds: float) -> str:
    """Compact remaining-time string ("48s", "3m47s", "1h04m"). Differs from
    `_format_duration` only under a minute, where a decimal on an estimate
    implies a precision it does not have."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    return _format_duration(seconds)


class ProgressReporter:
    """Report progress toward `total` units of work, as a bar or as log lines.

    Used as a context manager. `advance()` is called once per completed unit;
    everything else (rendering, throttling, teardown) is internal.

    `total` may be a *planned* count that the run ends up falling short of --
    Phase 1 sizes the bar from config (quota x days) before it knows whether
    every day will list successfully. Finishing under 100% is a real signal,
    not a bug, and the final summary reports what actually happened.
    """

    def __init__(
        self,
        description: str,
        total: int,
        *,
        logger: logging.Logger | None = None,
        unit: str = "file",
        track_bytes: bool = False,
        interval: float | None = None,
        status_fn=None,
        heartbeat: float = 0.5,
    ) -> None:
        self.description = description
        self.total = max(0, int(total))
        self.unit = unit
        self.track_bytes = track_bytes
        # Read at call time, not bound as a default argument: as a default it
        # would freeze at import and the module constant that documents itself
        # as the knob could not actually be turned.
        self.interval = DEFAULT_LOG_INTERVAL_SECONDS if interval is None else interval
        self._log = logger or log

        # What the work is doing *right now* -- in flight, current concurrency,
        # whether the server is making us wait. Rendered on a heartbeat rather
        # than on completion, because the case that most needs explaining is
        # precisely the one where nothing is completing.
        self.status_fn = status_fn
        self.heartbeat = heartbeat

        self.completed = 0
        self.n_bytes = 0

        self._t0 = 0.0
        self._last_log = 0.0
        self._progress = None
        self._task = None
        self._retargeted: list[tuple[logging.Handler, object]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None

    # -- rendering surface ---------------------------------------------------

    def _use_bar(self) -> bool:
        """True when a human is watching stderr. Wrapped in try/except because
        a replaced stderr (pytest's capture, a StringIO) may not implement
        isatty at all."""
        try:
            return bool(sys.stderr.isatty())
        except (AttributeError, ValueError):
            return False

    def _retarget_log_handlers(self, original_stderr) -> None:
        """Point stderr log handlers at rich's stderr proxy for the duration.

        `rich.progress.Progress` keeps its bar intact around other output by
        proxying `sys.stderr` -- but `logging.StreamHandler` binds the stream
        object at construction, so handlers installed by `mine.main`'s
        `basicConfig` still hold the *original* stderr and write straight past
        the proxy. That is what turns a live bar into a garbled one: rich
        rewinds the cursor by N lines that something else has already scrolled.
        Restored in `__exit__`.

        Handlers are matched by *identity* against the stderr object rich
        replaced, not by name: `stream.name` does not exist on every stream
        (a StringIO has none), and identity is the exact question being asked
        -- is this handler writing to the surface the bar now owns? A file
        handler is left alone, which matters, since repointing it would send
        the log to the terminal and lose the file.
        """
        proxy = sys.stderr
        if proxy is original_stderr:  # rich did not install a proxy
            return
        for handler in logging.getLogger().handlers:
            if getattr(handler, "stream", None) is original_stderr:
                self._retargeted.append((handler, original_stderr))
                handler.stream = proxy

    def __enter__(self) -> "ProgressReporter":
        self._t0 = time.monotonic()
        self._last_log = self._t0
        if not self._use_bar():
            self._log.info("%s: 0/%d %ss...", self.description, self.total, self.unit)
            # The heartbeat matters at least as much here: `run_pipeline.sh`
            # pipes this to a log, and without a clock-driven line a throttled
            # download writes nothing at all between completions.
            self._start_ticker()
            return self

        from rich.console import Console
        from rich.progress import (BarColumn, Progress, SpinnerColumn,
                                   TextColumn, TimeRemainingColumn)

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[progress.percentage]{task.fields[detail]}"),
            TimeRemainingColumn(),
            console=Console(stderr=True),
        )
        original_stderr = sys.stderr  # captured before rich proxies it
        self._progress.start()
        self._task = self._progress.add_task(self.description, total=self.total, detail="")
        self._retarget_log_handlers(original_stderr)
        self._start_ticker()
        return self

    def _start_ticker(self) -> None:
        """Refresh on a clock, not on completions.

        A download that is being rate-limited completes nothing for minutes,
        so a completion-driven display sits at "1/20000" and reads as hung --
        which is exactly what it did. The heartbeat means the line keeps
        reporting *what is happening* even when nothing is finishing.
        """
        if self.heartbeat <= 0:
            return

        def _tick():
            while not self._stop.wait(self.heartbeat):
                self._render()

        self._ticker = threading.Thread(
            target=_tick, name="progress-heartbeat", daemon=True
        )
        self._ticker.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join(timeout=2.0)
            self._ticker = None
        for handler, stream in self._retargeted:
            handler.stream = stream
        self._retargeted.clear()
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    # -- reporting -----------------------------------------------------------

    def _detail(self, elapsed: float) -> str:
        """The middle column / log-line tail: bytes, rate, and live status."""
        parts = []
        if self.track_bytes:
            detail = _format_size(self.n_bytes)
            if elapsed > 0:
                detail += f", {self.n_bytes / elapsed / (1024.0 * 1024.0):.2f} MB/s"
            parts.append(detail)
        if self.status_fn is not None:
            try:
                status = self.status_fn()
            except Exception as exc:  # noqa: BLE001 - a status probe must never kill the run
                status = f"status unavailable ({exc})"
            if status:
                parts.append(status)
        return " · ".join(parts)

    def _render(self) -> None:
        """Draw the current state. Safe to call from the heartbeat thread, from
        `advance`, or from both at once."""
        with self._lock:
            elapsed = time.monotonic() - self._t0
            if self._progress is not None:
                self._progress.update(
                    self._task, completed=self.completed, detail=self._detail(elapsed)
                )
                return
            now = time.monotonic()
            if now - self._last_log < self.interval:
                return
            self._last_log = now
            detail = self._detail(elapsed)
            eta = ""
            if self.completed and self.total > self.completed and elapsed > 0:
                remaining = (self.total - self.completed) * elapsed / self.completed
                eta = f", eta {_format_eta(remaining)}"
            self._log.info(
                "%s: %d/%d %ss%s%s",
                self.description, self.completed, self.total, self.unit,
                f", {detail}" if detail else "", eta,
            )

    def set_total(self, total: int) -> None:
        """Correct the total once the real figure is known.

        Phase 1 sizes the bar from config before listing, because the true
        count does not exist until every day has been paged. Left uncorrected,
        a run that finds only 40 new episodes against a planned 20000 would
        show a bar frozen near zero with a meaningless ETA.
        """
        with self._lock:
            self.total = max(0, int(total))
            if self._progress is not None:
                self._progress.update(self._task, total=self.total)

    def advance(self, n: int = 1, n_bytes: int = 0) -> None:
        """Record `n` completed units (carrying `n_bytes` of payload).

        Thread-safe: callers may advance from worker threads while the
        heartbeat renders. Rendering is `_render`'s job, so a burst of
        completions costs one draw per heartbeat rather than one per item --
        which matters when an aborted run drains thousands of queued tasks at
        once.
        """
        with self._lock:
            self.completed += n
            self.n_bytes += n_bytes
        self._render()
