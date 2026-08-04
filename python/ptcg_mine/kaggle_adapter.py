"""A Kaggle API adapter that reuses one SDK client per thread.

`KaggleApi.dataset_download_file` calls `build_kaggle_client()` for every single
file, so a 20000-episode run builds 20000 clients and opens 20000 fresh TLS
connections. This wraps the real API and keeps one client per worker thread
instead, alive for the run.

Be clear about the size of the prize: at the measured ~0.125 MB/s per stream, a
4 MB episode spends ~32 s transferring, against maybe 0.1-0.3 s of handshake.
This is a low-single-digit percentage, not the fix -- `AdaptiveConcurrency` in
`download.py` is what actually addresses the throughput. It is here because it
is nearly free once the per-thread client exists.

The cost is coupling to SDK internals (`ApiDownloadDatasetRequest`, the
`datasets.dataset_api_client` path), which have already changed shape once in
this project's lifetime. So the fast path is guarded: anything that looks like
the SDK having moved underneath us disables it permanently for the process and
falls back to `dataset_download_file`, which is the supported entry point.
Network and HTTP errors are *not* caught here -- they belong to
`_download_with_retry`'s retry and rate-limit budgets.

`ptcg_mine.download` still never imports kaggle: it receives whichever adapter
it is handed, and every test hands it a fake.
"""

import logging
import os
import threading

log = logging.getLogger(__name__)


class SdkShapeError(RuntimeError):
    """The vendored SDK does not expose the internals the fast path needs."""


class PooledKaggleApi:
    """Delegates to a real `KaggleApi`, but downloads over a per-thread client.

    Only `dataset_download_file` is special-cased; `dataset_list_files` and
    anything else pass straight through, so this is a drop-in for the adapter
    shape `download_corpus` expects.
    """

    def __init__(self, api):
        self._api = api
        self._local = threading.local()
        self._clients: list = []          # every client built, for teardown
        self._clients_lock = threading.Lock()
        self._fast_path = True
        self.n_fast = 0
        self.n_fallback = 0

    # -- pass-through --------------------------------------------------------

    def dataset_list_files(self, slug, page_token=None):
        return self._api.dataset_list_files(slug, page_token=page_token)

    def __getattr__(self, name):
        return getattr(self._api, name)

    # -- per-thread client ---------------------------------------------------

    def _client(self):
        """The calling thread's SDK client, built once and kept open.

        The SDK hands these out as context managers; we deliberately do not
        close them per call -- that is the entire saving. They are closed
        together in `close()`.
        """
        client = getattr(self._local, "client", None)
        if client is not None:
            return client
        build = getattr(self._api, "build_kaggle_client", None)
        if build is None:
            raise SdkShapeError("KaggleApi has no build_kaggle_client")
        client = build().__enter__()
        self._local.client = client
        with self._clients_lock:
            self._clients.append(client)
        return client

    def _download_url_response(self, client, slug: str, file_name: str):
        """The signed-download response for one file, via the reused client."""
        try:
            from kagglesdk.datasets.types.dataset_api_service import \
                ApiDownloadDatasetRequest
        except ImportError as exc:  # pragma: no cover - depends on SDK version
            raise SdkShapeError(f"ApiDownloadDatasetRequest unavailable: {exc}") from exc

        owner_slug, dataset_slug, version = self._api.split_dataset_string(slug)
        request = ApiDownloadDatasetRequest()
        request.owner_slug = owner_slug
        request.dataset_slug = dataset_slug
        request.dataset_version_number = int(version) if version else None
        request.file_name = file_name
        try:
            return client.datasets.dataset_api_client.download_dataset(request)
        except AttributeError as exc:
            raise SdkShapeError(f"download_dataset path moved: {exc}") from exc

    # -- the one method that matters -----------------------------------------

    def dataset_download_file(self, slug, file_name, path=None):
        """Download one file, reusing this thread's client when possible."""
        if self._fast_path:
            try:
                return self._download_pooled(slug, file_name, path)
            except SdkShapeError as exc:
                # Said once, loudly: the run continues correctly but slower, and
                # a silent permanent downgrade is exactly the kind of thing that
                # gets discovered months later.
                self._fast_path = False
                log.warning(
                    "PooledKaggleApi: falling back to dataset_download_file for the "
                    "rest of this run — the vendored SDK does not expose the pooled "
                    "path (%s). Downloads still work; each one opens a new "
                    "connection.",
                    exc,
                )
        self.n_fallback += 1
        return self._api.dataset_download_file(slug, file_name, path=path)

    def _download_pooled(self, slug, file_name, path):
        client = self._client()
        response = self._download_url_response(client, slug, file_name)
        url = getattr(getattr(response, "request", None), "url", None)
        if not url:
            raise SdkShapeError("download response carries no request URL")
        outfile = os.path.join(path, url.split("?")[0].split("/")[-1])

        download_file = getattr(self._api, "download_file", None)
        http_client = getattr(client, "http_client", None)
        if download_file is None or http_client is None:
            raise SdkShapeError("download_file/http_client unavailable")

        # The SDK's own streaming writer: keeps its chunking, retry and resume
        # behaviour. All we have changed is who owns the connection.
        download_file(response, outfile, http_client(), True, False)
        self.n_fast += 1
        return True

    # -- teardown ------------------------------------------------------------

    def close(self) -> None:
        """Close every client this adapter opened. Safe to call twice."""
        with self._clients_lock:
            clients, self._clients = self._clients, []
        for client in clients:
            try:
                client.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 - teardown must not mask a real error
                log.debug("PooledKaggleApi: error closing client: %s", exc)
        self._local = threading.local()

    def __enter__(self) -> "PooledKaggleApi":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
