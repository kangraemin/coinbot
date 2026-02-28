"""전략 B: ADX 레짐 필터 평균회귀 — 지표 계산, 진입 조건, 익절/손절.

진입 조건 (4개):
  1. ADX(14) < 25  — 횡보 레짐 (추세장에선 평균회귀 무효)
  2. 가격 ≤ BB 하단(20, 2σ) — 과매도 구간 진입
  3. RSI(14) ≤ 35  — 모멘텀 과매도 확인
  4. RSI 반등 시작 — 바닥 확인 (rsi[0] > rsi[-1])

익절: BB 중심선 (진정한 평균회귀 타겟)
손절: 진입가 - ATR(14) × 1.5
"""

import asyncio
import logging

import pandas as pd
import ta as ta_lib

import config as cfg
import journal
import report

logger = logging.getLogger(__name__)


# 모든 지표 안정화에 필요한 최소 캔들 수 (ADX ~28, BB 20, RSI 14)
_CANDLE_MIN = 50


# ── 지표 계산 ────────────────────────────────────────


def compute_indicators(shared_state: dict) -> dict | None:
    """shared_state의 캔들 deque로 지표를 계산하여 반환한다."""
    candles = shared_state["candles"]
    if len(candles) < _CANDLE_MIN:
        return None

    df = pd.DataFrame(
        list(candles),
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Bollinger Bands (20, 2σ)
    bb = ta_lib.volatility.BollingerBands(close, window=cfg.BB_PERIOD, window_dev=cfg.BB_STD)
    bb_lower = bb.bollinger_lband()
    bb_middle = bb.bollinger_mavg()  # 평균회귀 TP 타겟

    # RSI (14)
    rsi = ta_lib.momentum.RSIIndicator(close, window=cfg.RSI_PERIOD).rsi()

    # ATR (14)
    atr = ta_lib.volatility.AverageTrueRange(high, low, close, window=cfg.ATR_PERIOD).average_true_range()

    # ADX (14) — 레짐 감지 (횡보 vs 추세)
    adx = ta_lib.trend.ADXIndicator(high, low, close, window=cfg.ADX_PERIOD).adx()

    indicators = {
        "bb_lower": bb_lower.iloc[-1],
        "bb_middle": bb_middle.iloc[-1],
        "rsi": rsi.iloc[-1],
        "rsi_prev": rsi.iloc[-2] if len(rsi) >= 2 else None,
        "atr": atr.iloc[-1],
        "adx": adx.iloc[-1],
    }

    # NaN 체크 — 지표 미완성 시 None 반환
    if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in indicators.values()):
        return None

    shared_state["indicators"] = indicators
    return indicators


# ── 진입 조건 체크 ───────────────────────────────────


def check_entry_conditions(price: float, indicators: dict) -> bool:
    """4개 진입 조건 — ADX 레짐 + BB 하단 + RSI 과매도 + RSI 반등."""
    # 1. ADX < 25 — 횡보 레짐 (평균회귀 유효 구간)
    #    추세장(ADX >= 25)에서 평균회귀는 손절 반복으로 이어짐
    if indicators["adx"] >= cfg.ADX_TREND_THRESHOLD:
        return False

    # 2. 가격 ≤ BB 하단 (20, 2σ) — 과매도 구간 진입
    if price > indicators["bb_lower"]:
        return False

    # 3. RSI(14) ≤ 35 — 모멘텀 과매도 확인
    if indicators["rsi"] > cfg.RSI_THRESHOLD:
        return False

    # 4. RSI 반등 시작 — 바닥 확인 (단순 하락 중 진입 방지)
    if indicators["rsi_prev"] is None or indicators["rsi"] <= indicators["rsi_prev"]:
        return False

    return True


# ── 주문 실행 ────────────────────────────────────────


async def place_entry_order(exchange, price: float, indicators: dict) -> dict | None:
    """지정가 매수 주문을 실행한다. 시장가 주문 금지."""
    amount = cfg.TRADE_AMOUNT_USDT / price

    try:
        order = await exchange.create_order(
            symbol=cfg.SYMBOL,
            type="limit",
            side="buy",
            amount=amount,
            price=price,
            params={"postOnly": True},
        )
        logger.info(
            "매수 주문 생성 — 가격: %.2f, 수량: %.6f, 주문ID: %s",
            price,
            amount,
            order["id"],
        )
        return order
    except Exception as e:
        logger.warning("매수 주문 실패: %s", e)
        return None


async def place_tp_sl_orders(
    exchange, entry_price: float, amount: float, atr: float, bb_middle: float
) -> tuple[dict | None, dict | None]:
    """익절(BB 중심선)/손절(ATR×1.5) 지정가 주문을 배치한다."""
    tp_price = bb_middle                      # 평균회귀 타겟 = BB 중심선
    sl_price = entry_price - atr * cfg.SL_ATR_MULT

    tp_order = None
    sl_order = None

    # 익절 — 지정가 매도
    try:
        tp_order = await exchange.create_order(
            symbol=cfg.SYMBOL,
            type="limit",
            side="sell",
            amount=amount,
            price=tp_price,
            params={"postOnly": True},
        )
        logger.info("익절 주문 — 가격: %.2f, 주문ID: %s", tp_price, tp_order["id"])
    except Exception as e:
        logger.warning("익절 주문 실패: %s", e)

    # 손절 — 스탑로스 지정가
    try:
        sl_order = await exchange.create_order(
            symbol=cfg.SYMBOL,
            type="stop",
            side="sell",
            amount=amount,
            price=sl_price,
            params={"stopPrice": sl_price},
        )
        logger.info("손절 주문 — 가격: %.2f, 주문ID: %s", sl_price, sl_order["id"])
    except Exception as e:
        logger.warning("손절 주문 실패: %s", e)

    return tp_order, sl_order


# ── 전략 루프 ────────────────────────────────────────


async def strategy_loop(exchange, shared_state: dict) -> None:
    """전략 메인 루프 — 캔들 업데이트마다 지표 계산 + 조건 체크."""
    has_position = False
    current_trade_id: int | None = None

    while True:
        try:
            # 일일 손실 한도 체크
            if shared_state.get("trading_halted"):
                await asyncio.sleep(30)
                continue

            # 캔들 데이터 충분한지 확인
            if len(shared_state["candles"]) < _CANDLE_MIN:
                await asyncio.sleep(1)
                continue

            indicators = compute_indicators(shared_state)
            if indicators is None:
                await asyncio.sleep(1)
                continue

            price = shared_state["last_price"]

            # 포지션 확인
            if not has_position:
                positions = await exchange.fetch_positions([cfg.SYMBOL])
                open_positions = [
                    p for p in positions
                    if float(p.get("contracts", 0)) > 0
                ]
                if len(open_positions) >= cfg.MAX_POSITIONS:
                    has_position = True
                    await asyncio.sleep(cfg.RECONNECT_DELAY)
                    continue

                # 진입 조건 체크
                if check_entry_conditions(price, indicators):
                    logger.info(
                        "진입 조건 충족 — 가격: %.2f, RSI: %.1f, ADX: %.1f, BB하단: %.2f",
                        price,
                        indicators["rsi"],
                        indicators["adx"],
                        indicators["bb_lower"],
                    )
                    order = await place_entry_order(exchange, price, indicators)
                    if order:
                        amount = float(order.get("amount", 0))
                        atr = indicators["atr"]
                        bb_middle = indicators["bb_middle"]
                        tp_order, sl_order = await place_tp_sl_orders(
                            exchange, price, amount, atr, bb_middle
                        )

                        # 매매 일지 기록
                        tp_id = tp_order["id"] if tp_order else None
                        sl_id = sl_order["id"] if sl_order else None
                        current_trade_id = journal.record_trade(
                            side="buy",
                            symbol=cfg.SYMBOL,
                            entry_price=price,
                            amount=amount,
                            order_id=order["id"],
                            tp_order_id=tp_id,
                            sl_order_id=sl_id,
                        )

                        # Telegram 알림
                        tp_price = bb_middle
                        sl_price = price - atr * cfg.SL_ATR_MULT
                        await report.send_trade_alert(
                            "buy", price, amount, tp_price, sl_price
                        )
                        has_position = True
            else:
                # 포지션 종료 확인
                positions = await exchange.fetch_positions([cfg.SYMBOL])
                open_positions = [
                    p for p in positions
                    if float(p.get("contracts", 0)) > 0
                ]
                if not open_positions:
                    has_position = False

                    # 매매 종료 기록
                    if current_trade_id is not None:
                        trade = journal.get_open_trade()
                        if trade:
                            entry_price = trade["entry_price"]
                            amount = trade["amount"]
                            fee = price * amount * 0.0002  # maker 0.02%
                            pnl = (price - entry_price) * amount - fee
                            journal.close_trade(current_trade_id, price, fee, pnl)
                            await report.send_close_alert(entry_price, price, pnl, fee)
                        current_trade_id = None

                    logger.info("포지션 종료 감지 — 새 진입 대기")

            await asyncio.sleep(1)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("전략 루프 오류: %s", e)
            await asyncio.sleep(cfg.RECONNECT_DELAY)
