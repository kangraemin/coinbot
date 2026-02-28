"""리스크 관리 — 미체결 주문 타임아웃, 일일 손실 한도."""

import asyncio
import logging
import time

import config as cfg
import report

logger = logging.getLogger(__name__)


async def cancel_stale_orders(exchange) -> None:
    """ORDER_TIMEOUT_MIN을 초과한 미체결 진입 주문을 취소한다.
    reduceOnly 주문(TP/SL)은 취소하지 않는다."""
    now_ms = int(time.time() * 1000)
    timeout_ms = cfg.ORDER_TIMEOUT_MIN * 60 * 1000

    for symbol in cfg.SYMBOLS:
        try:
            open_orders = await exchange.fetch_open_orders(symbol)
            for order in open_orders:
                # TP/SL(reduceOnly) 주문은 건너뜀
                if order.get("reduceOnly", False):
                    continue
                order_time = order.get("timestamp", 0)
                if now_ms - order_time > timeout_ms:
                    await exchange.cancel_order(order["id"], symbol)
                    elapsed = (now_ms - order_time) / 60000
                    logger.info(
                        "[%s] 미체결 주문 취소 (타임아웃) — 주문ID: %s, 경과: %.1f분",
                        symbol, order["id"], elapsed,
                    )
                    await report.send_telegram(
                        f"⏰ *미체결 주문 자동 취소*\n심볼: {symbol}\n주문ID: {order['id']}\n경과: {elapsed:.1f}분"
                    )
        except Exception as e:
            logger.error("[%s] 미체결 주문 취소 중 오류: %s", symbol, e)


async def check_daily_loss(exchange, shared_state: dict) -> bool:
    """일일 PnL을 확인하여 한도 초과 시 거래를 중단한다.

    Returns:
        True이면 거래 중단 상태.
    """
    try:
        # journal.py에서 일일 PnL 가져오기
        from journal import get_daily_pnl

        daily_pnl = get_daily_pnl()
        balance = await exchange.fetch_balance()
        total_balance = float(balance.get("total", {}).get("USDT", 0))

        if total_balance <= 0:
            return False

        daily_loss_pct = (daily_pnl / total_balance) * 100

        if daily_loss_pct <= cfg.MAX_DAILY_LOSS_PCT:
            shared_state["trading_halted"] = True
            logger.warning(
                "일일 손실 한도 도달 — PnL: %.2f USDT (%.1f%%), 거래 중단",
                daily_pnl,
                daily_loss_pct,
            )
            await report.send_telegram(
                f"🚨 *일일 손실 한도 도달 — 거래 중단*\n손실: {daily_pnl:.2f} USDT ({daily_loss_pct:.1f}%)\n한도: {cfg.MAX_DAILY_LOSS_PCT}%"
            )
            return True

        return False
    except Exception as e:
        logger.error("일일 손실 체크 오류: %s", e)
        return False


async def risk_loop(exchange, shared_state: dict) -> None:
    """리스크 관리 루프 — 30초마다 미체결 주문 타임아웃 + 일일 손실 체크."""
    while True:
        try:
            await cancel_stale_orders(exchange)
            await check_daily_loss(exchange, shared_state)
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("리스크 루프 오류: %s", e)
            await asyncio.sleep(cfg.RECONNECT_DELAY)
