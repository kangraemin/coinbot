import pytest
from unittest.mock import AsyncMock, patch

from bot import risk


async def test_check_daily_loss_halted(exchange, shared_state):
    """PnL -6% ≤ -5% threshold → trading_halted=True + Telegram alert."""
    exchange.fetch_balance.return_value = {"total": {"USDT": 1000.0}}

    with patch("bot.journal.get_daily_pnl", return_value=-60.0), \
         patch("bot.report.send_telegram", new_callable=AsyncMock) as mock_telegram:
        result = await risk.check_daily_loss(exchange, shared_state)

    assert result is True
    assert shared_state.get("trading_halted") is True
    mock_telegram.assert_called_once()


async def test_check_daily_loss_normal(exchange, shared_state):
    """PnL -1% > -5% threshold → trading_halted not set."""
    exchange.fetch_balance.return_value = {"total": {"USDT": 1000.0}}

    with patch("bot.journal.get_daily_pnl", return_value=-10.0), \
         patch("bot.report.send_telegram", new_callable=AsyncMock) as mock_telegram:
        result = await risk.check_daily_loss(exchange, shared_state)

    assert result is False
    assert "trading_halted" not in shared_state
    mock_telegram.assert_not_called()
