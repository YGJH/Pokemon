"""Telegram notifications for long-running CLI jobs.

Transport-only helper.  ``send_telegram`` never raises and never blocks long:
a missing env var, a down network, or a rejected message must not take a
training run with it.

Credentials come from the environment:

- ``TELEGRAM_BOT_TOKEN_WHISPER`` — bot token (BotFather)
- ``TELEGRAM_CHAT_ID`` — numeric chat id (or @channelusername)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

ENV_TOKEN = "TELEGRAM_BOT_TOKEN_WHISPER"
ENV_CHAT_ID = "TELEGRAM_CHAT_ID"

_TIMEOUT_S = 10
#: Telegram caps messages at 4096 chars; leave headroom for the wrapper text.
_MAX_LEN = 3900


def send_telegram(text: str) -> bool:
    """Send *text* to the configured chat.  Returns True on a 200 response.

    Any failure — unset env vars, network error, non-200 API response — is
    logged as a warning and reported as False, never raised.
    """
    token = os.environ.get(ENV_TOKEN)
    chat_id = os.environ.get(ENV_CHAT_ID)
    if not token or not chat_id:
        logger.warning(
            "telegram notify skipped: %s and/or %s not set", ENV_TOKEN, ENV_CHAT_ID
        )
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text[:_MAX_LEN]}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            if resp.status != 200:
                logger.warning("telegram notify failed: HTTP %d", resp.status)
                return False
    except Exception as e:  # notification must never crash the run
        logger.warning("telegram notify failed: %s", e)
        return False
    return True
