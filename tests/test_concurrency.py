"""Tests for ptcg_mine.download.AdaptiveConcurrency and PooledKaggleApi.

The ramp is the piece that decides how hard this hits Kaggle, so the policy is
pinned here rather than left to be inferred from a live run: climb only on
accumulated evidence, cut immediately, and never probe upward during the
post-429 cooldown.
"""

import threading
import time

import pytest

from ptcg_mine.download import AdaptiveConcurrency


def _limiter(**kw):
    defaults = dict(start=4, ceiling=32, step=4, floor=2, probe_after=8, cooldown=30.0)
    defaults.update(kw)
    return AdaptiveConcurrency(**defaults)


# ---------------------------------------------------------------------------
# ramp policy
# ---------------------------------------------------------------------------


def test_starts_at_the_configured_limit():
    assert _limiter().limit == 4


def test_does_not_climb_before_enough_clean_downloads():
    limiter = _limiter(probe_after=8)
    for _ in range(7):
        limiter.on_success()
    assert limiter.limit == 4, "stepped up on partial evidence"


def test_climbs_one_step_per_probe_window():
    limiter = _limiter(probe_after=8, step=4)
    for _ in range(8):
        limiter.on_success()
    assert limiter.limit == 8
    for _ in range(8):
        limiter.on_success()
    assert limiter.limit == 12


def test_never_climbs_past_the_ceiling():
    limiter = _limiter(probe_after=1, step=4, ceiling=12)
    for _ in range(100):
        limiter.on_success()
    assert limiter.limit == 12
    assert limiter.peak == 12


def test_rate_limit_halves_immediately():
    # cut_debounce=0 so each call counts as a distinct overload; the coalescing
    # of a burst is covered by test_a_burst_of_429s_costs_exactly_one_cut.
    limiter = _limiter(probe_after=1, step=4, ceiling=32, cut_debounce=0.0)
    for _ in range(10):
        limiter.on_success()
    assert limiter.limit == 32
    limiter.on_rate_limit()
    assert limiter.limit == 16
    limiter.on_rate_limit()
    assert limiter.limit == 8


def test_rate_limit_never_goes_below_the_floor():
    limiter = _limiter(floor=2, cut_debounce=0.0)
    for _ in range(20):
        limiter.on_rate_limit()
    assert limiter.limit == 2


def test_cooldown_blocks_probing_after_a_cut(monkeypatch):
    """Without the cooldown the ramp walks straight back into the wall: one
    cut, then probe_after clean files, then the same limit that just 429'd."""
    clock = [1000.0]
    monkeypatch.setattr("ptcg_mine.download.time.monotonic", lambda: clock[0])

    # Start high enough that the post-cut ceiling still leaves room to climb;
    # otherwise this would pass for the wrong reason (nowhere to go).
    limiter = _limiter(start=16, probe_after=2, step=4, cooldown=30.0)
    limiter.on_rate_limit()          # 16 -> 8, ceiling drops to 12
    assert limiter.limit == 8
    for _ in range(50):
        limiter.on_success()
    assert limiter.limit == 8, "probed upward during the cooldown"

    clock[0] += 31.0
    for _ in range(2):
        limiter.on_success()
    assert limiter.limit == 12, "never resumed probing after the cooldown"


def test_cut_resets_the_clean_counter(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("ptcg_mine.download.time.monotonic", lambda: clock[0])

    limiter = _limiter(probe_after=4, step=4, ceiling=32, cooldown=0.0)
    for _ in range(3):
        limiter.on_success()          # 3 of 4 toward a step
    limiter.on_rate_limit()
    limiter.on_success()              # would have been the 4th
    assert limiter.limit == 2, "a cut did not discard the accumulated evidence"


def test_peak_and_cut_count_are_recorded():
    limiter = _limiter(probe_after=1, step=4, ceiling=16, cut_debounce=0.0)
    for _ in range(10):
        limiter.on_success()
    assert limiter.peak == 16
    limiter.on_rate_limit()
    limiter.on_rate_limit()
    assert limiter.peak == 16, "peak must record the high-water mark, not the current limit"
    assert limiter.n_cuts == 2


def test_a_cut_that_changes_nothing_is_not_counted():
    """At the floor there is nothing left to give back."""
    limiter = _limiter(start=2, floor=2)
    limiter.on_rate_limit()
    assert limiter.n_cuts == 0


# ---------------------------------------------------------------------------
# the permit actually gates concurrency
# ---------------------------------------------------------------------------


def test_acquire_blocks_past_the_limit():
    limiter = _limiter(start=2, probe_after=10**9)
    limiter.acquire()
    limiter.acquire()

    entered = threading.Event()

    def _third():
        limiter.acquire()
        entered.set()

    t = threading.Thread(target=_third, daemon=True)
    t.start()
    assert not entered.wait(0.2), "a third download started against a limit of 2"

    limiter.release()
    assert entered.wait(2.0), "releasing a permit did not wake a waiter"
    t.join(2.0)


def test_raising_the_limit_wakes_waiters():
    # floor=1 too: the floor clamps `start`, so a floor of 2 would silently
    # make this a limit of 2 and the first assertion below would be vacuous.
    limiter = _limiter(start=1, floor=1, step=4, probe_after=1, ceiling=8)
    limiter.acquire()

    entered = threading.Event()
    threading.Thread(target=lambda: (limiter.acquire(), entered.set()), daemon=True).start()
    assert not entered.wait(0.2)

    limiter.on_success()  # 1 -> 5, must notify_all
    assert entered.wait(2.0), "a raised limit did not wake blocked workers"


def test_observed_concurrency_never_exceeds_the_limit():
    """The property that matters: whatever the ramp does, the number of
    simultaneous holders stays within the current limit."""
    limiter = _limiter(start=3, step=1, probe_after=5, ceiling=6)
    active = 0
    peak_seen = 0
    breach = []
    lock = threading.Lock()
    stop = threading.Event()

    def _worker():
        nonlocal active, peak_seen
        while not stop.is_set():
            limiter.acquire()
            with lock:
                active += 1
                peak_seen = max(peak_seen, active)
                if active > limiter.ceiling:
                    breach.append(active)
            time.sleep(0.001)
            with lock:
                active -= 1
            limiter.release()
            limiter.on_success()

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(12)]
    for t in threads:
        t.start()
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join(2.0)

    assert peak_seen > 0, "no worker ever ran -- the assertion below is vacuous"
    assert not breach, f"concurrency exceeded the ceiling: {breach}"
    assert peak_seen <= limiter.ceiling


# ---------------------------------------------------------------------------
# PooledKaggleApi
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, owner):
        self.owner = owner
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True

    def http_client(self):
        return "http-client"

    @property
    def datasets(self):
        return self

    @property
    def dataset_api_client(self):
        return self

    def download_dataset(self, request):
        self.owner.requests.append(request.file_name)
        url = f"https://x/{request.file_name}?sig=1"
        inner = type("R", (), {"url": url})()
        return type("Resp", (), {"request": inner})()


class _FakeSdkApi:
    """Enough of KaggleApi for the pooled path."""

    def __init__(self):
        self.n_clients_built = 0
        self.requests = []
        self.written = []
        self.slow_path_calls = []

    def build_kaggle_client(self):
        self.n_clients_built += 1
        return _FakeClient(self)

    def split_dataset_string(self, slug):
        owner, name = slug.split("/", 1)
        return owner, name, None

    def download_file(self, response, outfile, http_client, quiet, resume):
        self.written.append(outfile)

    def dataset_download_file(self, dataset, file_name, path=None):
        self.slow_path_calls.append(file_name)


def test_pooled_api_builds_one_client_per_thread(tmp_path):
    from ptcg_mine.kaggle_adapter import PooledKaggleApi

    sdk = _FakeSdkApi()
    with PooledKaggleApi(sdk) as api:
        for i in range(10):
            api.dataset_download_file("owner/ds", f"ep{i}.json", path=str(tmp_path))

    assert sdk.n_clients_built == 1, (
        f"built {sdk.n_clients_built} clients for 10 files in one thread"
    )
    assert api.n_fast == 10
    assert api.n_fallback == 0
    assert len(sdk.written) == 10


def test_pooled_api_uses_one_client_per_thread_not_one_globally(tmp_path):
    from ptcg_mine.kaggle_adapter import PooledKaggleApi

    sdk = _FakeSdkApi()
    with PooledKaggleApi(sdk) as api:
        def _work(n):
            api.dataset_download_file("owner/ds", f"t{n}.json", path=str(tmp_path))

        threads = [threading.Thread(target=_work, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5.0)

    assert sdk.n_clients_built == 4, "clients are not per-thread"


def test_pooled_api_closes_its_clients(tmp_path):
    from ptcg_mine.kaggle_adapter import PooledKaggleApi

    sdk = _FakeSdkApi()
    built = []
    real_build = sdk.build_kaggle_client

    def _spy():
        client = real_build()
        built.append(client)
        return client

    sdk.build_kaggle_client = _spy
    with PooledKaggleApi(sdk) as api:
        api.dataset_download_file("owner/ds", "ep0.json", path=str(tmp_path))

    assert built, "no client built -- the assertion below is vacuous"
    assert all(c.closed for c in built), "clients leaked; connections stay open"


def test_pooled_api_falls_back_permanently_when_the_sdk_moved(tmp_path, caplog):
    """An SDK that no longer exposes the internals must degrade to the
    supported entry point, once, loudly -- not fail the download."""
    from ptcg_mine.kaggle_adapter import PooledKaggleApi

    sdk = _FakeSdkApi()
    sdk.build_kaggle_client = None  # simulate the SDK moving underneath us

    with caplog.at_level("WARNING", logger="ptcg_mine.kaggle_adapter"):
        with PooledKaggleApi(sdk) as api:
            for i in range(3):
                api.dataset_download_file("owner/ds", f"ep{i}.json", path=str(tmp_path))

    assert sdk.slow_path_calls == ["ep0.json", "ep1.json", "ep2.json"]
    assert api.n_fast == 0
    assert api.n_fallback == 3
    warnings = [r for r in caplog.records if "falling back" in r.getMessage()]
    assert len(warnings) == 1, "the downgrade must be announced exactly once"


def test_pooled_api_does_not_swallow_network_errors(tmp_path):
    """Network failures belong to _download_with_retry's budgets. Catching them
    here would silently retry them on the slow path and hide rate limiting."""
    from ptcg_mine.kaggle_adapter import PooledKaggleApi

    class _Boom(_FakeSdkApi):
        def download_file(self, *a, **kw):
            raise ConnectionError("429 too many requests")

    sdk = _Boom()
    with PooledKaggleApi(sdk) as api:
        with pytest.raises(ConnectionError):
            api.dataset_download_file("owner/ds", "ep0.json", path=str(tmp_path))
    assert sdk.slow_path_calls == [], "a network error wrongly triggered the fallback"


def test_pooled_api_passes_listing_through():
    from ptcg_mine.kaggle_adapter import PooledKaggleApi

    class _Listing(_FakeSdkApi):
        def dataset_list_files(self, slug, page_token=None):
            return ("listed", slug, page_token)

    api = PooledKaggleApi(_Listing())
    assert api.dataset_list_files("owner/ds", page_token="7") == ("listed", "owner/ds", "7")


def test_a_burst_of_429s_costs_exactly_one_cut(monkeypatch):
    """Crossing the server's limit 429s everything then in flight, so the
    complaints arrive together. Halving per complaint compounded one overload
    into 12 -> 6 -> 3 -> 2 and left the run far below what had been working."""
    clock = [1000.0]
    monkeypatch.setattr("ptcg_mine.download.time.monotonic", lambda: clock[0])

    limiter = _limiter(start=16, floor=2, cut_debounce=5.0)
    for _ in range(20):           # 20 in-flight downloads all report the 429
        limiter.on_rate_limit()
    assert limiter.limit == 8, "a single overload was compounded into many cuts"
    assert limiter.n_cuts == 1

    clock[0] += 6.0               # a genuinely separate overload, later
    limiter.on_rate_limit()
    assert limiter.limit == 4
    assert limiter.n_cuts == 2


def test_debounced_burst_still_extends_the_probe_cooldown(monkeypatch):
    """Coalescing the cut must not also coalesce away the cooldown -- the later
    complaints are still evidence that the server is unhappy right now."""
    clock = [1000.0]
    monkeypatch.setattr("ptcg_mine.download.time.monotonic", lambda: clock[0])

    limiter = _limiter(start=16, probe_after=1, step=4, cooldown=30.0, cut_debounce=5.0)
    limiter.on_rate_limit()
    clock[0] += 4.0
    limiter.on_rate_limit()       # debounced cut, but refreshes the cooldown

    clock[0] += 27.0              # past the first cooldown, inside the second
    for _ in range(10):
        limiter.on_success()
    assert limiter.limit == 8, "probed upward while the server was still complaining"


def test_a_cut_lowers_the_ceiling_for_the_rest_of_the_run(monkeypatch):
    """The level that 429'd must become off-limits. Otherwise the ramp climbs
    straight back to it, gets cut, climbs again -- and since the run aborts
    after RATE_LIMIT_ABORT_AFTER rate-limited files, that cycle turns a slow
    run into a failed one."""
    clock = [1000.0]
    monkeypatch.setattr("ptcg_mine.download.time.monotonic", lambda: clock[0])

    limiter = _limiter(start=16, step=4, probe_after=1, ceiling=32, cooldown=1.0)
    limiter.on_rate_limit()
    assert limiter.ceiling == 12, "the failed level is still reachable"
    assert limiter.limit == 8

    clock[0] += 2.0
    for _ in range(100):
        limiter.on_success()
    assert limiter.limit == 12, "climbed past the remembered wall"
    assert limiter.limit < 16


def test_repeated_walls_converge_downward(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("ptcg_mine.download.time.monotonic", lambda: clock[0])

    limiter = _limiter(start=32, step=4, floor=2, probe_after=1, ceiling=32, cooldown=1.0)
    ceilings = []
    for _ in range(5):
        limiter.on_rate_limit()
        ceilings.append(limiter.ceiling)
        clock[0] += 2.0
        for _ in range(100):
            limiter.on_success()

    assert ceilings == sorted(ceilings, reverse=True), f"ceiling did not fall: {ceilings}"
    assert limiter.ceiling >= limiter.floor
    assert limiter.limit >= limiter.floor, "converged below the floor"
