"""Tests for ptcg_mine.progress.ProgressReporter.

The two rendering paths are tested separately because they are chosen by an
`isatty()` check on stderr that neither pytest nor the pipeline satisfies: the
non-TTY path is what a real `run_pipeline.sh` run gets, the bar path is what an
interactive `python -m ptcg_mine.mine` gets, and only one of them can be
exercised without faking the terminal.
"""

import logging
import threading
import time

import pytest

from ptcg_mine.progress import (DEFAULT_LOG_INTERVAL_SECONDS, ProgressReporter,
                                _format_eta)


@pytest.mark.parametrize(
    "seconds,expected",
    [(0.0, "0s"), (48.4, "48s"), (59.6, "60s"), (60.0, "1m00s"),
     (227.0, "3m47s"), (3600.0, "1h00m")],
)
def test_format_eta(seconds, expected):
    assert _format_eta(seconds) == expected


def test_format_eta_drops_the_decimal_that_format_duration_keeps():
    """An estimate under a minute has no business claiming a tenth of a second."""
    from ptcg_mine.progress import _format_duration

    assert _format_duration(48.4) == "48.4s"
    assert _format_eta(48.4) == "48s"


class _FakeClock:
    """Monotonic time under test control, so throttling is deterministic."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr("ptcg_mine.progress.time.monotonic", clock)
    return clock


@pytest.fixture
def not_a_tty(monkeypatch):
    """Force the log-line path (the pipeline's path)."""
    monkeypatch.setattr(ProgressReporter, "_use_bar", lambda self: False)


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def test_logs_a_start_line_then_throttles(not_a_tty, fake_clock, caplog):
    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter("downloading", 100, unit="file") as reporter:
            for _ in range(10):
                reporter.advance()

    lines = _messages(caplog)
    # The start line always fires; nothing else does, because no time passed.
    assert lines == ["downloading: 0/100 files..."]
    assert reporter.completed == 10


def test_logs_one_line_per_interval(not_a_tty, fake_clock, caplog):
    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter("downloading", 100, unit="file", interval=30.0) as reporter:
            for i in range(90):
                fake_clock.tick(1.0)  # 90 advances over 90 simulated seconds
                reporter.advance()

    progress_lines = [line for line in _messages(caplog) if "/100 files," in line]
    assert len(progress_lines) == 3, f"expected one line per 30s, got {progress_lines}"
    assert "30/100 files" in progress_lines[0]
    assert "60/100 files" in progress_lines[1]


def test_log_line_carries_eta(not_a_tty, fake_clock, caplog):
    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter("downloading", 100, unit="file", interval=10.0) as reporter:
            for _ in range(25):
                fake_clock.tick(1.0)
                reporter.advance()

    joined = "\n".join(_messages(caplog))
    # 10 files in 10s -> 90 remaining at 1/s -> 1m30s
    assert "eta 1m30s" in joined


def test_log_line_carries_bytes_and_rate_when_tracked(not_a_tty, fake_clock, caplog):
    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter(
            "downloading", 100, unit="file", track_bytes=True, interval=10.0
        ) as reporter:
            for _ in range(10):
                fake_clock.tick(1.0)
                reporter.advance(1, 1024 * 1024)

    progress_lines = [line for line in _messages(caplog) if "10/100" in line]
    assert progress_lines, "no progress line emitted -- the assertions below are vacuous"
    assert "10.0 MB" in progress_lines[0]
    assert "1.00 MB/s" in progress_lines[0]


def test_no_bytes_or_rate_when_not_tracked(not_a_tty, fake_clock, caplog):
    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter("parsing episodes", 100, unit="episode", interval=10.0) as reporter:
            for _ in range(10):
                fake_clock.tick(1.0)
                reporter.advance()

    progress_lines = [line for line in _messages(caplog) if "10/100" in line]
    assert progress_lines
    assert "MB" not in progress_lines[0]


def test_no_eta_once_total_is_reached(not_a_tty, fake_clock, caplog):
    """A run that overshoots or exactly meets its planned total must not print
    a negative or zero ETA."""
    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter("downloading", 5, unit="file", interval=1.0) as reporter:
            for _ in range(8):
                fake_clock.tick(2.0)
                reporter.advance()

    progress_lines = [line for line in _messages(caplog) if "/5 files" in line and "..." not in line]
    assert progress_lines, "no progress line emitted -- the assertion below is vacuous"
    overshoot = [line for line in progress_lines if "eta" in line and "-" in line.split("eta")[1]]
    assert not overshoot, f"negative ETA emitted: {overshoot}"


def test_zero_total_does_not_divide_by_zero(not_a_tty, fake_clock, caplog):
    """An empty raw_dir gives load_raw_episodes a total of 0."""
    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter("parsing episodes", 0, unit="episode"):
            pass
    assert "parsing episodes: 0/0 episodes..." in _messages(caplog)


def test_uses_the_logger_it_is_given(not_a_tty, fake_clock, caplog):
    """Phase 1's lines must read as download_corpus's, not the helper's."""
    custom = logging.getLogger("ptcg_mine.download")
    with caplog.at_level("INFO", logger="ptcg_mine.download"):
        with ProgressReporter("downloading", 10, logger=custom, unit="file"):
            pass
    assert [r.name for r in caplog.records] == ["ptcg_mine.download"]


def test_default_interval_is_used_when_unspecified():
    reporter = ProgressReporter("downloading", 10)
    assert reporter.interval == DEFAULT_LOG_INTERVAL_SECONDS


# ---------------------------------------------------------------------------
# bar path
# ---------------------------------------------------------------------------


@pytest.fixture
def is_a_tty(monkeypatch):
    monkeypatch.setattr(ProgressReporter, "_use_bar", lambda self: True)


def test_bar_path_emits_no_log_lines(is_a_tty, caplog):
    """The bar replaces the log lines rather than doubling them up."""
    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter("downloading", 10, unit="file", track_bytes=True) as reporter:
            for _ in range(10):
                reporter.advance(1, 1024)

    assert _messages(caplog) == []
    assert reporter.completed == 10
    assert reporter.n_bytes == 10 * 1024


def test_bar_is_torn_down_on_exit(is_a_tty):
    with ProgressReporter("downloading", 10) as reporter:
        assert reporter._progress is not None
    assert reporter._progress is None


def test_bar_is_torn_down_when_the_body_raises(is_a_tty):
    """Phase 1 raises DownloadError from inside the reporter's scope; a live
    rich bar left running holds the terminal in a broken state."""
    reporter = ProgressReporter("downloading", 10)
    with pytest.raises(RuntimeError):
        with reporter:
            assert reporter._progress is not None
            raise RuntimeError("boom")
    assert reporter._progress is None


@pytest.fixture
def rich_sees_a_terminal(monkeypatch):
    """Make rich believe stderr is a real terminal.

    rich only installs its stdout/stderr proxies when `console.is_terminal`,
    and pytest's capture is not a terminal -- so without this the retarget has
    nothing to retarget *to* and the tests below would pass vacuously against
    a no-op.
    """
    from rich.console import Console

    monkeypatch.setattr(Console, "is_terminal", property(lambda self: True))


def test_stderr_log_handlers_are_retargeted_and_restored(is_a_tty, rich_sees_a_terminal):
    """Without the retarget the bar is garbled: logging.StreamHandler binds the
    original stderr at construction and writes straight past rich's proxy."""
    import sys

    root = logging.getLogger()
    handler = logging.StreamHandler()  # binds sys.stderr as it is right now
    original = handler.stream
    root.addHandler(handler)
    try:
        with ProgressReporter("downloading", 10):
            assert sys.stderr is not original, "rich installed no proxy -- test is vacuous"
            assert handler.stream is sys.stderr, "handler still writes past rich's proxy"
        assert handler.stream is original, "handler stream not restored"
    finally:
        root.removeHandler(handler)


def test_non_stderr_handlers_are_left_alone(is_a_tty, rich_sees_a_terminal, tmp_path):
    """A file handler is not the bar's problem and must not be repointed at
    the terminal -- that would send the log to the console and lose the file."""
    root = logging.getLogger()
    log_path = tmp_path / "run.log"
    handler = logging.FileHandler(log_path)
    root.addHandler(handler)
    try:
        with ProgressReporter("downloading", 10):
            assert handler.stream.name == str(log_path)
    finally:
        root.removeHandler(handler)
        handler.close()


def test_no_retarget_when_rich_installs_no_proxy(is_a_tty):
    """The pipeline's case: output is piped, rich does not proxy, and touching
    the handlers would be meddling with a stream nothing else owns."""
    root = logging.getLogger()
    handler = logging.StreamHandler()
    original = handler.stream
    root.addHandler(handler)
    try:
        with ProgressReporter("downloading", 10) as reporter:
            assert handler.stream is original
            assert reporter._retargeted == []
    finally:
        root.removeHandler(handler)


# ---------------------------------------------------------------------------
# heartbeat: the display must report even when nothing completes
# ---------------------------------------------------------------------------


def test_heartbeat_reports_while_nothing_completes(not_a_tty, caplog):
    """The bug that made a throttled download unreadable: the line only moved
    on completion, so being rate-limited for minutes looked identical to being
    hung at 1/20000."""
    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter(
            "downloading", 100, unit="file", interval=0.05, heartbeat=0.02,
            status_fn=lambda: "4 in flight, limit 4",
        ) as reporter:
            time.sleep(0.35)
            assert reporter.completed == 0, "something completed -- test is not testing a stall"

    lines = [line for line in _messages(caplog) if "in flight" in line]
    assert len(lines) >= 2, f"heartbeat did not report during a stall: {_messages(caplog)}"
    assert "0/100 files" in lines[0]


def test_status_fn_appears_in_the_line(not_a_tty, fake_clock, caplog):
    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter(
            "downloading", 100, unit="file", interval=1.0, heartbeat=0,
            status_fn=lambda: "rate limited, resuming in 42s",
        ) as reporter:
            fake_clock.tick(2.0)
            reporter.advance()

    joined = "\n".join(_messages(caplog))
    assert "rate limited, resuming in 42s" in joined


def test_a_failing_status_probe_does_not_kill_the_run(not_a_tty, fake_clock, caplog):
    """The status probe reads live state from several objects; a bug in it must
    degrade the display, not abort a multi-hour download."""
    def _boom():
        raise RuntimeError("probe exploded")

    with caplog.at_level("INFO", logger="ptcg_mine.progress"):
        with ProgressReporter(
            "downloading", 100, unit="file", interval=1.0, heartbeat=0, status_fn=_boom
        ) as reporter:
            fake_clock.tick(2.0)
            reporter.advance()          # must not raise

    joined = "\n".join(_messages(caplog))
    assert "status unavailable" in joined
    assert "probe exploded" in joined


def test_heartbeat_thread_stops_on_exit(not_a_tty):
    reporter = ProgressReporter("downloading", 10, heartbeat=0.02)
    with reporter:
        assert reporter._ticker is not None
        assert reporter._ticker.is_alive()
    assert reporter._ticker is None


def test_advance_is_safe_from_many_threads(not_a_tty):
    """An aborted run drains thousands of queued tasks at once, each advancing
    from its own worker thread."""
    with ProgressReporter("downloading", 4000, unit="file", interval=10**9,
                          heartbeat=0.01, track_bytes=True) as reporter:
        def _burst():
            for _ in range(500):
                reporter.advance(1, 10)

        threads = [threading.Thread(target=_burst) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10.0)

    assert reporter.completed == 4000, f"lost updates under concurrency: {reporter.completed}"
    assert reporter.n_bytes == 40000
