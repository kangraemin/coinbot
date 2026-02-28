import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

import strategy
import config as cfg

SYMBOL = "BTC/USDT:USDT"


@pytest.fixture(autouse=True)
def reset_pos():
    """Reset _pos[SYMBOL] to clean state before each strategy test."""
    strategy._pos[SYMBOL] = {
        "has_position": False,
        "trade_id": None,
        "entry_price": 0.0,
        "amount": 0.0,
        "entry_order_id": None,
        "tp_order_id": None,
        "sl_order_id": None,
        "last_prev_close": 0.0,
    }
    yield


async def test_place_entry(exchange):
    exchange.fetch_balance.return_value = {"total": {"USDT": 1000.0}}
    exchange.create_order.return_value = {"id": "order123"}

    await strategy._place_entry(exchange, SYMBOL, 50000.0)

    entry_price = 50000.0 * (1 - cfg.ENTRY_DROP_PCT / 100)  # 49250.0
    expected_amount = 1000.0 * cfg.POSITION_RATIO * cfg.LEVERAGE / entry_price

    exchange.create_order.assert_called_once_with(
        SYMBOL, "limit", "buy", expected_amount, entry_price,
        {"timeInForce": "GTC"},
    )
    assert strategy._pos[SYMBOL]["entry_order_id"] == "order123"


async def test_place_tp_sl(exchange):
    exchange.create_order = AsyncMock(side_effect=[
        {"id": "tp_order_id"},
        {"id": "sl_order_id"},
    ])

    with patch("report.send_trade_alert", new_callable=AsyncMock) as mock_alert:
        await strategy._place_tp_sl(exchange, SYMBOL, 50000.0, 0.1)

    assert exchange.create_order.call_count == 2

    tp_args = exchange.create_order.call_args_list[0][0]
    sl_args = exchange.create_order.call_args_list[1][0]

    # TP: limit sell with reduceOnly
    assert tp_args[0] == SYMBOL
    assert tp_args[1] == "limit"
    assert tp_args[2] == "sell"
    assert tp_args[3] == 0.1
    assert tp_args[4] == 51500.0
    assert tp_args[5] == {"reduceOnly": True, "timeInForce": "GTC"}

    # SL: STOP_MARKET sell with stopPrice
    # BTC sl_pct=1.5% (SYMBOL_PARAMS 기준) → 50000 * 0.985 = 49250.0
    assert sl_args[0] == SYMBOL
    assert sl_args[1] == "STOP_MARKET"
    assert sl_args[2] == "sell"
    assert sl_args[3] == 0.1
    assert sl_args[4] is None
    assert sl_args[5]["stopPrice"] == 49250.0
    assert sl_args[5]["reduceOnly"] is True

    mock_alert.assert_called_once()
    assert strategy._pos[SYMBOL]["tp_order_id"] == "tp_order_id"
    assert strategy._pos[SYMBOL]["sl_order_id"] == "sl_order_id"


async def test_handle_symbol_flow_A(exchange, shared_state):
    """Flow A: no position, no pending order → calls _place_entry."""
    shared_state[SYMBOL] = {"prev_close": 50000.0}

    with patch.object(strategy, "_place_entry", new_callable=AsyncMock) as mock_entry:
        await strategy._handle_symbol(exchange, SYMBOL, shared_state)

    mock_entry.assert_called_once_with(exchange, SYMBOL, 50000.0)


async def test_handle_symbol_flow_B_filled(exchange, shared_state):
    """Flow B: pending order filled → record trade + place TP/SL."""
    strategy._pos[SYMBOL]["entry_order_id"] = "order123"
    shared_state[SYMBOL] = {"prev_close": 49250.0}

    exchange.fetch_order = AsyncMock(return_value={
        "id": "order123", "status": "closed",
        "average": 49250.0, "filled": 0.1,
    })

    with patch("journal.record_trade", return_value=1) as mock_record, \
         patch.object(strategy, "_place_tp_sl", new_callable=AsyncMock) as mock_tp_sl:
        await strategy._handle_symbol(exchange, SYMBOL, shared_state)

    mock_record.assert_called_once_with("long", SYMBOL, 49250.0, 0.1, "order123")
    mock_tp_sl.assert_called_once_with(exchange, SYMBOL, 49250.0, 0.1)
    assert strategy._pos[SYMBOL]["has_position"] is True
    assert strategy._pos[SYMBOL]["trade_id"] == 1
    assert strategy._pos[SYMBOL]["entry_price"] == 49250.0
    assert strategy._pos[SYMBOL]["amount"] == 0.1


async def test_handle_symbol_flow_B_price_change(exchange, shared_state):
    """Flow B: pending order open + prev_close drifted >0.5% → cancel + re-place."""
    strategy._pos[SYMBOL]["entry_order_id"] = "order123"
    strategy._pos[SYMBOL]["last_prev_close"] = 50000.0
    shared_state[SYMBOL] = {"prev_close": 50300.0}  # 0.6% change > 0.5%

    exchange.fetch_order = AsyncMock(return_value={"status": "open"})

    with patch.object(strategy, "_cancel_safe", new_callable=AsyncMock) as mock_cancel, \
         patch.object(strategy, "_place_entry", new_callable=AsyncMock) as mock_entry:
        await strategy._handle_symbol(exchange, SYMBOL, shared_state)

    mock_cancel.assert_called_once_with(exchange, "order123", SYMBOL)
    mock_entry.assert_called_once_with(exchange, SYMBOL, 50300.0)


async def test_handle_symbol_flow_C(exchange, shared_state):
    """Flow C: has position + empty fetch_positions → cancel TP/SL + close trade."""
    strategy._pos[SYMBOL].update({
        "has_position": True,
        "trade_id": 1,
        "entry_price": 50000.0,
        "amount": 0.1,
        "tp_order_id": "tp1",
        "sl_order_id": "sl1",
    })
    shared_state[SYMBOL] = {"prev_close": 50000.0, "last_price": 51000.0}

    exchange.fetch_positions = AsyncMock(return_value=[])

    with patch.object(strategy, "_cancel_safe", new_callable=AsyncMock) as mock_cancel, \
         patch("journal.close_trade") as mock_close, \
         patch("report.send_close_alert", new_callable=AsyncMock) as mock_alert:
        await strategy._handle_symbol(exchange, SYMBOL, shared_state)

    assert mock_cancel.call_count == 2
    cancelled_ids = {c[0][1] for c in mock_cancel.call_args_list}
    assert "tp1" in cancelled_ids
    assert "sl1" in cancelled_ids

    mock_close.assert_called_once()
    close_args = mock_close.call_args[0]
    assert close_args[0] == 1         # trade_id
    assert close_args[1] == 51000.0   # exit_price

    mock_alert.assert_called_once()
    assert strategy._pos[SYMBOL]["has_position"] is False
