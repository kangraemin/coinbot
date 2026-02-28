import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import config as cfg
import report


async def test_send_trade_alert(monkeypatch):
    """send_trade_alert calls Telegram API with correct URL and payload."""
    monkeypatch.setattr(cfg, "TELEGRAM_TOKEN", "test")
    monkeypatch.setattr(cfg, "TELEGRAM_CHAT_ID", "123")

    resp_mock = MagicMock()
    resp_mock.status = 200
    resp_mock.text = AsyncMock(return_value="OK")

    resp_ctx = AsyncMock()
    resp_ctx.__aenter__ = AsyncMock(return_value=resp_mock)
    resp_ctx.__aexit__ = AsyncMock(return_value=False)

    session_mock = MagicMock()
    session_mock.post = MagicMock(return_value=resp_ctx)

    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session_mock)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=session_ctx):
        await report.send_trade_alert(
            "long", 50000.0, 0.1, 51500.0, 49750.0, symbol="BTC/USDT:USDT"
        )

    session_mock.post.assert_called_once()
    url_arg = session_mock.post.call_args[0][0]
    assert "bottest/sendMessage" in url_arg

    payload = session_mock.post.call_args[1]["json"]
    assert payload["chat_id"] == "123"
    assert "text" in payload
    assert payload["parse_mode"] == "Markdown"
