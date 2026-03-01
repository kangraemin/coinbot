import pytest
from collections import deque
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from bot import strategy
import config as cfg

SYMBOL = "BTC/USDT:USDT"

FAKE_IND = {
    "timestamp": 1000000,
    "close": 50000.0,
    "bb_upper": 52000.0,
    "bb_lower": 48000.0,
    "bb_mid": 50000.0,
    "rsi": 25.0,
    "atr": 1000.0,
    "ema200": 45000.0,
}


@pytest.fixture(autouse=True)
def reset_pos():
    strategy._pos[SYMBOL] = {
        "has_position":   False,
        "direction":      None,
        "trade_id":       None,
        "entry_price":    0.0,
        "amount":         0.0,
        "atr_at_entry":   0.0,
        "tp_order_id":    None,
        "sl_order_id":    None,
        "entry_time":     None,
        "last_signal_ts": None,
    }
    yield


# ── _place_entry ────────────────────────────────────

async def test_place_entry_long(exchange):
    exchange.fetch_balance.return_value = {"total": {"USDT": 1000.0}}
    exchange.create_order.return_value = {
        "id": "order123", "average": 50000.0, "filled": 0.042,
    }

    result = await strategy._place_entry(exchange, SYMBOL, "long", FAKE_IND)

    assert result is not None
    order_id, fill_price, fill_amt = result
    assert order_id == "order123"
    assert fill_price == 50000.0

    args = exchange.create_order.call_args[0]
    assert args[1] == "market"
    assert args[2] == "buy"


async def test_place_entry_short(exchange):
    exchange.fetch_balance.return_value = {"total": {"USDT": 1000.0}}
    exchange.create_order.return_value = {
        "id": "order456", "average": 52000.0, "filled": 0.04,
    }

    ind = dict(FAKE_IND, close=52000.0)
    result = await strategy._place_entry(exchange, SYMBOL, "short", ind)

    assert result is not None
    args = exchange.create_order.call_args[0]
    assert args[2] == "sell"


# ── _place_tp_sl ────────────────────────────────────

async def test_place_tp_sl_long(exchange):
    """Long: TP=ep+ATR×tp_mult(3), SL=ep-ATR×sl_mult(2), both sell."""
    strategy._pos[SYMBOL]["amount"] = 0.1
    exchange.create_order = AsyncMock(side_effect=[{"id": "tp1"}, {"id": "sl1"}])

    with patch("bot.report.send_trade_alert", new_callable=AsyncMock):
        tp_id, sl_id = await strategy._place_tp_sl(
            exchange, SYMBOL, "long", 50000.0, 1000.0
        )

    assert tp_id == "tp1"
    assert sl_id == "sl1"

    tp_args = exchange.create_order.call_args_list[0][0]
    sl_args = exchange.create_order.call_args_list[1][0]

    assert tp_args[1] == "limit"
    assert tp_args[2] == "sell"
    assert tp_args[4] == 53000.0   # 50000 + 1000 * 3

    assert sl_args[1] == "STOP_MARKET"
    assert sl_args[2] == "sell"
    assert sl_args[5]["stopPrice"] == 48000.0   # 50000 - 1000 * 2


async def test_place_tp_sl_short(exchange):
    """Short: TP=ep-ATR×tp_mult(3), SL=ep+ATR×sl_mult(2), both buy."""
    strategy._pos[SYMBOL]["amount"] = 0.1
    exchange.create_order = AsyncMock(side_effect=[{"id": "tp1"}, {"id": "sl1"}])

    with patch("bot.report.send_trade_alert", new_callable=AsyncMock):
        tp_id, sl_id = await strategy._place_tp_sl(
            exchange, SYMBOL, "short", 50000.0, 1000.0
        )

    tp_args = exchange.create_order.call_args_list[0][0]
    sl_args = exchange.create_order.call_args_list[1][0]

    assert tp_args[2] == "buy"
    assert tp_args[4] == 47000.0   # 50000 - 1000 * 3

    assert sl_args[2] == "buy"
    assert sl_args[5]["stopPrice"] == 52000.0   # 50000 + 1000 * 2


# ── _handle_symbol Phase A ──────────────────────────

async def test_handle_symbol_flow_A_long(exchange, shared_state):
    """Phase A: long signal → entry + TP/SL placed."""
    shared_state[SYMBOL] = {
        "candles": deque([object()] * 230, maxlen=250),
        "last_price": 50000.0,
    }

    with patch.object(strategy, "_compute_indicators", return_value=FAKE_IND), \
         patch.object(strategy, "_check_signal", return_value="long"), \
         patch.object(strategy, "_place_entry", new_callable=AsyncMock,
                      return_value=("order1", 50000.0, 0.042)) as mock_entry, \
         patch("bot.journal.record_trade", return_value=1), \
         patch.object(strategy, "_place_tp_sl", new_callable=AsyncMock,
                      return_value=("tp1", "sl1")) as mock_tp_sl:
        await strategy._handle_symbol(exchange, SYMBOL, shared_state)

    mock_entry.assert_called_once_with(exchange, SYMBOL, "long", FAKE_IND)
    mock_tp_sl.assert_called_once_with(exchange, SYMBOL, "long", 50000.0, 1000.0)
    assert strategy._pos[SYMBOL]["has_position"] is True
    assert strategy._pos[SYMBOL]["direction"] == "long"
    assert strategy._pos[SYMBOL]["tp_order_id"] == "tp1"
    assert strategy._pos[SYMBOL]["sl_order_id"] == "sl1"


async def test_handle_symbol_flow_A_no_signal(exchange, shared_state):
    """Phase A: no signal → no entry."""
    shared_state[SYMBOL] = {
        "candles": deque([object()] * 230, maxlen=250),
        "last_price": 50000.0,
    }

    with patch.object(strategy, "_compute_indicators", return_value=FAKE_IND), \
         patch.object(strategy, "_check_signal", return_value=None), \
         patch.object(strategy, "_place_entry", new_callable=AsyncMock) as mock_entry:
        await strategy._handle_symbol(exchange, SYMBOL, shared_state)

    mock_entry.assert_not_called()


async def test_handle_symbol_flow_A_same_candle(exchange, shared_state):
    """Phase A: same candle timestamp → skip (no double entry)."""
    strategy._pos[SYMBOL]["last_signal_ts"] = FAKE_IND["timestamp"]
    shared_state[SYMBOL] = {
        "candles": deque([object()] * 230, maxlen=250),
        "last_price": 50000.0,
    }

    with patch.object(strategy, "_compute_indicators", return_value=FAKE_IND), \
         patch.object(strategy, "_place_entry", new_callable=AsyncMock) as mock_entry:
        await strategy._handle_symbol(exchange, SYMBOL, shared_state)

    mock_entry.assert_not_called()


# ── _handle_symbol Phase C ──────────────────────────

async def test_handle_symbol_flow_C_closed(exchange, shared_state):
    """Phase C: position closed by exchange → cancel TP/SL + close trade."""
    strategy._pos[SYMBOL].update({
        "has_position": True,
        "direction":    "long",
        "trade_id":     1,
        "entry_price":  50000.0,
        "amount":       0.1,
        "tp_order_id":  "tp1",
        "sl_order_id":  "sl1",
        "entry_time":   datetime.now(timezone.utc),
    })
    shared_state[SYMBOL] = {"candles": deque(), "last_price": 53000.0}
    exchange.fetch_positions = AsyncMock(return_value=[])

    with patch.object(strategy, "_cancel_safe", new_callable=AsyncMock) as mock_cancel, \
         patch("bot.journal.close_trade") as mock_close, \
         patch("bot.report.send_close_alert", new_callable=AsyncMock) as mock_alert:
        await strategy._handle_symbol(exchange, SYMBOL, shared_state)

    assert mock_cancel.call_count == 2
    cancelled_ids = {c[0][1] for c in mock_cancel.call_args_list}
    assert "tp1" in cancelled_ids
    assert "sl1" in cancelled_ids

    mock_close.assert_called_once()
    mock_alert.assert_called_once()
    assert strategy._pos[SYMBOL]["has_position"] is False


async def test_handle_symbol_flow_C_timeout(exchange, shared_state):
    """Phase C: timeout expired → force market close."""
    strategy._pos[SYMBOL].update({
        "has_position": True,
        "direction":    "short",
        "trade_id":     2,
        "entry_price":  52000.0,
        "amount":       0.1,
        "tp_order_id":  "tp2",
        "sl_order_id":  "sl2",
        "entry_time":   datetime.now(timezone.utc) - timedelta(hours=cfg.SIGNAL_TIMEOUT_HOURS + 1),
    })
    shared_state[SYMBOL] = {"candles": deque(), "last_price": 50000.0}
    exchange.create_order = AsyncMock(return_value={"id": "close1", "average": 50000.0, "price": 50000.0})

    with patch.object(strategy, "_cancel_safe", new_callable=AsyncMock), \
         patch("bot.journal.close_trade"), \
         patch("bot.report.send_close_alert", new_callable=AsyncMock):
        await strategy._handle_symbol(exchange, SYMBOL, shared_state)

    # market buy to close short position
    exchange.create_order.assert_called_once()
    close_args = exchange.create_order.call_args[0]
    assert close_args[2] == "buy"
    assert strategy._pos[SYMBOL]["has_position"] is False
