"""Telegram notifications on CLI exit (ptcg_il.notify + the cli.main wrapper).

The bits worth pinning: a missing env var or a dead network must never crash
the run, and each exit path sends exactly one message — including the
silent ``SystemExit(0)`` case (``--help``), which must send none.
"""

import json

import pytest

from ptcg_il import cli, notify


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(notify.ENV_TOKEN, raising=False)
    monkeypatch.delenv(notify.ENV_CHAT_ID, raising=False)


class TestSendTelegram:
    def test_missing_env_is_a_noop(self):
        assert notify.send_telegram("hi") is False

    def test_posts_chat_id_and_text(self, monkeypatch):
        monkeypatch.setenv(notify.ENV_TOKEN, "tok123")
        monkeypatch.setenv(notify.ENV_CHAT_ID, "456")
        sent = {}

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout):
            sent["url"] = req.full_url
            sent["payload"] = json.loads(req.data)
            return _Resp()

        monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
        assert notify.send_telegram("hello") is True
        assert "bottok123" in sent["url"]
        assert sent["payload"] == {"chat_id": "456", "text": "hello"}

    def test_network_error_does_not_raise(self, monkeypatch):
        monkeypatch.setenv(notify.ENV_TOKEN, "t")
        monkeypatch.setenv(notify.ENV_CHAT_ID, "c")

        def boom(req, timeout):
            raise OSError("no route to host")

        monkeypatch.setattr(notify.urllib.request, "urlopen", boom)
        assert notify.send_telegram("x") is False

    def test_long_messages_are_truncated(self, monkeypatch):
        monkeypatch.setenv(notify.ENV_TOKEN, "t")
        monkeypatch.setenv(notify.ENV_CHAT_ID, "c")
        sent = {}

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout):
            sent["payload"] = json.loads(req.data)
            return _Resp()

        monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
        notify.send_telegram("x" * 5000)
        assert len(sent["payload"]["text"]) == notify._MAX_LEN


class TestMainWrapper:
    @pytest.fixture
    def captured(self, monkeypatch):
        msgs = []
        monkeypatch.setattr(cli, "send_telegram", lambda text: msgs.append(text))
        return msgs

    def test_success_notifies(self, captured, monkeypatch):
        monkeypatch.setattr(cli, "_dispatch", lambda argv: 0)
        assert cli.main(["train"]) == 0
        assert len(captured) == 1
        assert captured[0].startswith("✅")

    def test_nonzero_return_notifies(self, captured, monkeypatch):
        monkeypatch.setattr(cli, "_dispatch", lambda argv: 1)
        assert cli.main(["train"]) == 1
        assert len(captured) == 1
        assert "❌" in captured[0] and "exit 1" in captured[0]

    def test_exception_notifies_and_reraises(self, captured, monkeypatch):
        def boom(argv):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(cli, "_dispatch", boom)
        with pytest.raises(RuntimeError):
            cli.main(["train"])
        assert len(captured) == 1
        assert "❌" in captured[0] and "kaboom" in captured[0]

    def test_system_exit_nonzero_notifies(self, captured, monkeypatch):
        def bad(argv):
            raise SystemExit(2)

        monkeypatch.setattr(cli, "_dispatch", bad)
        with pytest.raises(SystemExit):
            cli.main(["train"])
        assert len(captured) == 1
        assert "exit 2" in captured[0]

    def test_system_exit_message_is_attached(self, captured, monkeypatch):
        def bad(argv):
            raise SystemExit("missing engine_card_features.npy")

        monkeypatch.setattr(cli, "_dispatch", bad)
        with pytest.raises(SystemExit):
            cli.main(["train"])
        assert "missing engine_card_features.npy" in captured[0]

    def test_system_exit_zero_stays_silent(self, captured, monkeypatch):
        def ok(argv):
            raise SystemExit(0)

        monkeypatch.setattr(cli, "_dispatch", ok)
        with pytest.raises(SystemExit):
            cli.main(["train"])
        assert captured == []

    def test_keyboard_interrupt_notifies(self, captured, monkeypatch):
        def interrupt(argv):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_dispatch", interrupt)
        with pytest.raises(KeyboardInterrupt):
            cli.main(["train"])
        assert len(captured) == 1
        assert "⚠️" in captured[0]

    def test_invoked_command_is_in_the_message(self, captured, monkeypatch):
        monkeypatch.setattr(cli, "_dispatch", lambda argv: 0)
        cli.main(["train", "--epochs", "3"])
        assert "train --epochs 3" in captured[0]
