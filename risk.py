"""리스크 관리 — 미체결 주문 타임아웃, 일일 손실 한도."""

import asyncio
import logging
import time

import config as cfg

logger = logging.getLogger(__name__)


async def cancel_stale_orders(exchange) -> None:
    """ORDER_TIMEOUT_MIN을 초과한 미체결 주문을 취소한다."""
    try:
        open_orders = await exchange.fetch_open_orders(cfg.SYMBOL)
        now_ms = int(time.time() * 1000)
        timeout_ms = cfg.ORDER_TIMEOUT_MIN * 60 * 1000

        for order in open_orders:
            order_time = order.get("timestamp", 0)
            if now_ms - order_time > timeout_ms:
                await exchange.cancel_order(order["id"], cfg.SYMBOL)
                logger.info(
                    "미체결 주문 취소 (타임아웃) — 주문ID: %s, 경과: %.1f분",
                    order["id"],
                    (now_ms - order_time) / 60000,
                )
    except Exception as e:
        logger.error("미체결 주문 취소 중 오류: %s", e)


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
