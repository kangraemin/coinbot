"""bot_listener.py unit tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque

import numpy as np


# ── fake candle data (250 candles needed for _compute_indicators) ──

def _make_candles(n=250, base_price=85000.0):
    """Generate fake OHLCV candles for testing."""
    candles = []
    ts = 1700000000000
    for i in range(n):
        o = base_price + np.sin(i * 0.1) * 500
        h = o + 200
        l = o - 200
        c = o + np.cos(i * 0.1) * 100
        v = 1000.0
        candles.append([ts + i * 14400000, o, h, l, c, v])
    return candles


@pytest.fixture
def mock_exchange():
    exc = AsyncMock()
    exc.fetch_ohlcv = AsyncMock(return_value=_make_candles())
    exc.fetch_positions = AsyncMock(return_value=[])
    exc.fetch_balance = AsyncMock(return_value={
        "total": {"USDT": 1500.0},
        "free": {"USDT": 1200.0},
    })
    return exc


@pytest.mark.asyncio
async def test_build_status_all_symbols(mock_exchange):
    """TC-1: build_status returns all symbols summary."""
    from bot_listener import build_status

    result = await build_status(mock_exchange)
    assert "coinbot 현황" in result
    assert "BTC" in result
    assert "ETH" in result
    assert "XRP" in result
    assert "잔액" in result


@pytest.mark.asyncio
async def test_build_status_single_coin(mock_exchange):
    """TC-2: build_status with symbol returns detailed view."""
    from bot_listener import build_status

    result = await build_status(mock_exchange, "BTC")
    assert "BTC" in result
    assert "진입 조건" in result
    assert "용어 설명" in result
    # Should not contain other coins in detailed mode
    assert "ETH" not in result


@pytest.mark.asyncio
async def test_build_status_unknown_symbol(mock_exchange):
    """TC-3: Unknown symbol returns error message."""
    from bot_listener import build_status

    result = await build_status(mock_exchange, "DOGE")
    assert "알 수 없는 코인" in result
    assert "BTC" in result  # available coins listed


def test_build_help():
    """TC-4: build_help returns commands and symbols."""
    from bot_listener import build_help

    result = build_help()
    assert "/status" in result
    assert "/help" in result
    assert "BTC" in result


@pytest.mark.asyncio
async def test_handle_message_unauthorized():
    """TC-5: Unauthorized chat_id is ignored."""
    from bot_listener import handle_message

    session = AsyncMock()
    exchange = AsyncMock()
    msg = {"text": "/status", "chat": {"id": "99999"}}

    with patch("bot_listener.ALLOWED_CHAT_ID", "12345"):
        with patch("bot_listener.send_reply") as mock_send:
            await handle_message(session, exchange, msg)
            mock_send.assert_not_called()
