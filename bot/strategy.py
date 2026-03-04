"""4H BB+RSI 양방향 평균회귀 전략.

진입:
  롱: close < BB_lower(20,2σ) AND RSI(14) < rsi_long  AND close > EMA(200)
  숏: close > BB_upper(20,2σ) AND RSI(14) > rsi_short AND close < EMA(200)

TP: 진입가 ± ATR × tp_mult  (리밋, reduceOnly)
SL: 진입가 ± ATR × sl_mult  (스탑마켓, reduceOnly)
타임아웃: SIGNAL_TIMEOUT_HOURS 경과 → 시장가 강제청산
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone, timedelta

import numpy as np

import config as cfg
from . import journal
from . import report

logger = logging.getLogger(__name__)

# ── 심볼별 포지션 상태 ─────────────────────────────────
_pos: dict[str, dict] = {
    symbol: {
        "has_position":  False,
        "direction":     None,   # 'long' | 'short'
        "trade_id":      None,
        "entry_price":   0.0,
        "amount":        0.0,
        "atr_at_entry":  0.0,
        "tp_order_id":   None,
        "sl_order_id":   None,
        "entry_time":    None,   # datetime(UTC) — 타임아웃 기준
        "last_signal_ts": None,  # 같은 봉 재진입 방지
    }
    for symbol in cfg.SYMBOLS
}


# ── 인디케이터 ────────────────────────────────────────

def _ewm(values: np.ndarray, alpha: float) -> float:
    """EWM (adjust=False) 마지막 값 반환."""
    result = float(values[0])
    for v in values[1:]:
        result = result * (1.0 - alpha) + float(v) * alpha
    return result


def _compute_indicators(candles: deque) -> dict | None:
    """마지막 확정봉 기준 BB/RSI/ATR/EMA200 계산.

    candles[-1]은 현재 진행 중인 봉이므로 제외하고,
    candles[-2]에 해당하는 마지막 확정봉 값을 반환한다.
    """
    if len(candles) < 220:
        return None

    arr    = list(candles)[:-1]        # 현재 봉 제외 → 마지막 = 확정봉
    closes = np.array([c[4] for c in arr], dtype=float)
    highs  = np.array([c[2] for c in arr], dtype=float)
    lows   = np.array([c[3] for c in arr], dtype=float)

    # BB(20, 2σ)
    last_20 = closes[-20:]
    bb_mid  = float(last_20.mean())
    bb_std  = float(last_20.std(ddof=1))

    # RSI(14) — Wilder EWM, alpha = 1/14
    delta     = np.diff(closes)
    gains     = np.where(delta > 0, delta, 0.0)
    losses    = np.where(delta < 0, -delta, 0.0)
    alpha_rsi = 1.0 / 14.0
    avg_gain  = _ewm(gains,  alpha_rsi)
    avg_loss  = _ewm(losses, alpha_rsi)
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) if avg_loss > 1e-10 else 100.0

    # ATR(14) — Wilder EWM
    prev_close        = np.empty_like(closes)
    prev_close[0]     = closes[0]
    prev_close[1:]    = closes[:-1]
    tr                = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    tr[0] = highs[0] - lows[0]
    atr   = _ewm(tr, alpha_rsi)

    # EMA(200) — span=200, alpha = 2/201
    ema200 = _ewm(closes, 2.0 / 201.0)

    return {
        "timestamp": arr[-1][0],   # 밀리초 Unix (봉 open time)
        "close":    float(closes[-1]),
        "bb_upper": bb_mid + 2 * bb_std,
        "bb_lower": bb_mid - 2 * bb_std,
        "bb_mid":   bb_mid,
        "rsi":      rsi,
        "atr":      atr,
        "ema200":   ema200,
    }


def _check_signal(ind: dict, symbol: str) -> str | None:
    """BB+RSI+EMA200 진입 조건 → 'long' | 'short' | None."""
    p = cfg.SYMBOL_STRATEGY.get(symbol, {})
    if not p:
        return None

    close  = ind["close"]
    rsi    = ind["rsi"]
    ema200 = ind["ema200"]

    if close < ind["bb_lower"] and rsi < p["rsi_long"] and close > ema200:
        return "long"
    if close > ind["bb_upper"] and rsi > p["rsi_short"] and close < ema200:
        return "short"
    return None


# ── 주문 유틸 ─────────────────────────────────────────

async def _get_total_balance(exchange) -> float:
    try:
        data = await exchange.fetch_balance()
        return float(data.get("total", {}).get("USDT", 0) or 0)
    except Exception as e:
        logger.error("잔액 조회 실패: %s", e)
        return 0.0


async def _cancel_safe(exchange, order_id: str, symbol: str) -> None:
    try:
        await exchange.cancel_order(order_id, symbol)
        logger.debug("[%s] 주문 취소: %s", symbol, order_id)
    except Exception:
        pass


async def _place_entry(
    exchange, symbol: str, direction: str, ind: dict
) -> tuple[str, float, float] | None:
    """시장가 진입 주문 → (order_id, fill_price, fill_amount) 또는 None."""
    balance = await _get_total_balance(exchange)
    if balance <= 0:
        logger.warning("[%s] 잔액 없음, 진입 건너뜀", symbol)
        return None

    notional = balance * cfg.POSITION_RATIO * cfg.LEVERAGE
    raw_amt  = notional / ind["close"]
    try:
        amount = float(exchange.amount_to_precision(symbol, raw_amt))
    except Exception:
        amount = round(raw_amt, 6)

    if amount * ind["close"] < 10:
        logger.warning("[%s] 최소 주문금액 미달 (%.2f USDT)", symbol, amount * ind["close"])
        return None

    side = "buy" if direction == "long" else "sell"
    try:
        order      = await exchange.create_order(symbol, "market", side, amount)
        fill_price = float(order.get("average") or order.get("price") or ind["close"])
        fill_amt   = float(order.get("filled")  or order.get("amount") or amount)
        logger.info("[%s] %s 진입 — %.6f @ %.4f", symbol, direction, fill_amt, fill_price)
        return order["id"], fill_price, fill_amt
    except Exception as e:
        logger.error("[%s] 진입 주문 실패: %s", symbol, e)
        return None


async def _place_tp_sl(
    exchange, symbol: str, direction: str, entry_price: float, atr: float
) -> tuple[str | None, str | None]:
    """TP 리밋 + SL 스탑마켓 발행 → (tp_order_id, sl_order_id)."""
    p       = cfg.SYMBOL_STRATEGY[symbol]
    amount  = _pos[symbol]["amount"]

    if direction == "long":
        tp_raw = entry_price + atr * p["tp_mult"]
        sl_raw = entry_price - atr * p["sl_mult"]
        tp_side = sl_side = "sell"
    else:
        tp_raw = entry_price - atr * p["tp_mult"]
        sl_raw = entry_price + atr * p["sl_mult"]
        tp_side = sl_side = "buy"

    try:
        tp_price = float(exchange.price_to_precision(symbol, tp_raw))
        sl_price = float(exchange.price_to_precision(symbol, sl_raw))
    except Exception:
        tp_price = round(tp_raw, 4)
        sl_price = round(sl_raw, 4)

    tp_id = sl_id = None

    try:
        tp_order = await exchange.create_order(
            symbol, "limit", tp_side, amount, tp_price,
            {"reduceOnly": True, "timeInForce": "GTC"},
        )
        tp_id = tp_order["id"]
        logger.info("[%s] TP 주문 — @ %.4f", symbol, tp_price)
    except Exception as e:
        logger.error("[%s] TP 주문 실패: %s", symbol, e)

    try:
        sl_order = await exchange.create_order(
            symbol, "STOP_MARKET", sl_side, 0, None,
            {"stopPrice": sl_price, "closePosition": True},
        )
        sl_id = sl_order["id"]
        logger.info("[%s] SL 주문 — @ %.4f", symbol, sl_price)
    except Exception as e:
        logger.error("[%s] SL 주문 실패: %s", symbol, e)

    await report.send_trade_alert(
        direction, entry_price, amount, tp_price, sl_price,
        symbol=symbol, leverage=cfg.LEVERAGE,
    )
    return tp_id, sl_id


# ── 포지션 종료 공통 처리 ───────────────────────────────

async def _close_position(exchange, symbol: str, exit_price: float, reason: str) -> None:
    """저널 업데이트 + 알림 + 상태 초기화."""
    state     = _pos[symbol]
    direction = state["direction"]

    if direction == "long":
        pnl = (exit_price - state["entry_price"]) * state["amount"] * cfg.LEVERAGE
    else:
        pnl = (state["entry_price"] - exit_price) * state["amount"] * cfg.LEVERAGE
    fee = exit_price * state["amount"] * 0.0005 * 2

    if state["trade_id"]:
        journal.close_trade(state["trade_id"], exit_price, fee, pnl)

    logger.info("[%s] 포지션 종료 (%s) — %s @ %.4f | PnL %.2f", symbol, reason, direction, exit_price, pnl)
    await report.send_close_alert(state["entry_price"], exit_price, pnl, fee, symbol=symbol)

    state.update({
        "has_position": False, "direction": None,
        "trade_id": None, "entry_price": 0.0, "amount": 0.0,
        "atr_at_entry": 0.0, "tp_order_id": None, "sl_order_id": None,
        "entry_time": None,
    })


# ── 상태 머신 ─────────────────────────────────────────

async def _handle_symbol(exchange, symbol: str, shared_state: dict) -> None:
    """단일 심볼 전략 처리."""
    state = _pos[symbol]
    sym   = shared_state.get(symbol, {})

    # ── C: 포지션 보유 중 ────────────────────────────────
    if state["has_position"]:
        # 타임아웃 체크
        if state["entry_time"] and datetime.now(timezone.utc) >= (
            state["entry_time"] + timedelta(hours=cfg.SIGNAL_TIMEOUT_HOURS)
        ):
            logger.info("[%s] 포지션 타임아웃 — 시장가 강제청산", symbol)
            for oid in (state["tp_order_id"], state["sl_order_id"]):
                if oid:
                    await _cancel_safe(exchange, oid, symbol)
            side = "sell" if state["direction"] == "long" else "buy"
            try:
                order = await exchange.create_order(
                    symbol, "market", side, state["amount"], {"reduceOnly": True}
                )
                exit_price = float(order.get("average") or order.get("price") or 0)
            except Exception as e:
                logger.error("[%s] 타임아웃 청산 실패: %s", symbol, e)
                return
            await _close_position(exchange, symbol, exit_price, "timeout")
            return

        # TP/SL 체결 여부 확인
        try:
            positions = await exchange.fetch_positions([symbol])
            open_pos  = next(
                (p for p in positions if abs(float(p.get("contracts") or 0)) > 1e-8),
                None,
            )
            if open_pos is None:
                for oid in (state["tp_order_id"], state["sl_order_id"]):
                    if oid:
                        await _cancel_safe(exchange, oid, symbol)
                last_price = sym.get("last_price", state["entry_price"])
                await _close_position(exchange, symbol, last_price, "tp_sl")
        except Exception as e:
            logger.error("[%s] 포지션 조회 오류: %s", symbol, e)
        return

    # ── A: 신호 확인 + 진입 ─────────────────────────────
    candles = sym.get("candles")
    if not candles:
        return

    ind = _compute_indicators(candles)
    if ind is None:
        logger.debug("[%s] 인디케이터 부족 (캔들 %d개)", symbol, len(candles))
        return

    # 같은 확정봉에서 이미 처리했으면 건너뜀
    if state["last_signal_ts"] == ind["timestamp"]:
        return

    logger.debug(
        "[%s] 봉 %s | close=%.4f RSI=%.1f EMA200=%.4f BB=[%.4f,%.4f]",
        symbol, ind["timestamp"], ind["close"], ind["rsi"], ind["ema200"],
        ind["bb_lower"], ind["bb_upper"],
    )

    signal = _check_signal(ind, symbol)
    state["last_signal_ts"] = ind["timestamp"]

    if signal is None:
        return

    logger.info("[%s] 신호: %s | close=%.4f RSI=%.1f", symbol, signal, ind["close"], ind["rsi"])

    result = await _place_entry(exchange, symbol, signal, ind)
    if result is None:
        return

    order_id, fill_price, fill_amt = result
    trade_id = journal.record_trade(signal, symbol, fill_price, fill_amt, order_id)

    state.update({
        "has_position": True,
        "direction":    signal,
        "trade_id":     trade_id,
        "entry_price":  fill_price,
        "amount":       fill_amt,
        "atr_at_entry": ind["atr"],
        "entry_time":   datetime.now(timezone.utc),
    })

    tp_id, sl_id = await _place_tp_sl(exchange, symbol, signal, fill_price, ind["atr"])
    state["tp_order_id"] = tp_id
    state["sl_order_id"] = sl_id


async def _restore_state(exchange) -> None:
    """재시작 시 기존 포지션 복원."""
    for symbol in cfg.SYMBOLS:
        state = _pos[symbol]
        try:
            positions = await exchange.fetch_positions([symbol])
            open_pos  = next(
                (p for p in positions if abs(float(p.get("contracts") or 0)) > 1e-8),
                None,
            )
            if open_pos:
                contracts = float(open_pos.get("contracts") or 0)
                side_str  = (open_pos.get("side") or "").lower()
                if side_str in ("long", "short"):
                    direction = side_str
                else:
                    direction = "long" if contracts > 0 else "short"
                state.update({
                    "has_position": True,
                    "direction":    direction,
                    "entry_price":  float(open_pos.get("entryPrice") or 0),
                    "amount":       abs(contracts),
                    "entry_time":   datetime.now(timezone.utc),  # 정확한 진입 시각 불명
                })
                logger.info(
                    "[%s] 포지션 복원 — %s %.6f @ %.4f",
                    symbol, state["direction"], state["amount"], state["entry_price"],
                )
                try:
                    open_orders = await exchange.fetch_open_orders(symbol)
                    for o in open_orders:
                        ot = (o.get("type") or "").upper()
                        if ot == "LIMIT":
                            state["tp_order_id"] = o["id"]
                            logger.info("[%s] TP 복원: %s", symbol, o["id"])
                        elif "STOP" in ot:
                            state["sl_order_id"] = o["id"]
                            logger.info("[%s] SL 복원: %s", symbol, o["id"])

                    # SL 누락 시 TP 가격으로 ATR 역산 → SL 재등록
                    if state["tp_order_id"] and not state["sl_order_id"]:
                        tp_order = next(
                            (o for o in open_orders if o["id"] == state["tp_order_id"]), None
                        )
                        if tp_order:
                            p         = cfg.SYMBOL_STRATEGY.get(symbol, {})
                            tp_price  = float(tp_order.get("price") or 0)
                            atr       = abs(tp_price - state["entry_price"]) / p["tp_mult"]
                            sl_raw    = (
                                state["entry_price"] - atr * p["sl_mult"]
                                if direction == "long"
                                else state["entry_price"] + atr * p["sl_mult"]
                            )
                            try:
                                sl_price = float(exchange.price_to_precision(symbol, sl_raw))
                            except Exception:
                                sl_price = round(sl_raw, 4)
                            sl_side = "sell" if direction == "long" else "buy"
                            try:
                                sl_order = await exchange.create_order(
                                    symbol, "STOP_MARKET", sl_side, 0, None,
                                    {"stopPrice": sl_price, "closePosition": True},
                                )
                                state["sl_order_id"] = sl_order["id"]
                                logger.info("[%s] SL 재등록 완료 — @ %.4f", symbol, sl_price)
                            except Exception as e:
                                logger.error("[%s] SL 재등록 실패: %s", symbol, e)
                        else:
                            logger.warning("[%s] SL 없음 + TP 주문 조회 실패 — SL 재등록 불가", symbol)
                    elif not state["sl_order_id"]:
                        logger.warning("[%s] SL 없음 + TP도 없음 — 포지션 수동 확인 필요", symbol)

                except Exception as e:
                    logger.warning("[%s] 열린 주문 복원 실패: %s", symbol, e)
        except Exception as e:
            logger.error("[%s] 상태 복원 오류: %s", symbol, e)


async def strategy_loop(exchange, shared_state: dict) -> None:
    """전략 메인 루프 — 30초마다 모든 심볼 순회."""
    logger.info("전략 루프 시작 (4H BB+RSI) — %s", cfg.SYMBOLS)
    await asyncio.sleep(5)
    await _restore_state(exchange)

    while True:
        try:
            if not shared_state.get("trading_halted", False):
                for symbol in cfg.SYMBOLS:
                    await _handle_symbol(exchange, symbol, shared_state)
            else:
                logger.debug("거래 중단 상태 — 전략 루프 대기")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("전략 루프 오류: %s", e)

        await asyncio.sleep(30)
