"""리스크 관리 — 일일 손실 한도."""

import asyncio
import logging

import config as cfg
import report

logger = logging.getLogger(__name__)


async def check_daily_loss(exchange, shared_state: dict) -> bool:
    """일일 PnL을 확인하여 한도 초과 시 거래를 중단한다.

    Returns:
        True이면 거래 중단 상태.
    """
    try:
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
    """리스크 관리 루프 — 30초마다 일일 손실 체크."""
    while True:
        try:
            await check_daily_loss(exchange, shared_state)
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("리스크 루프 오류: %s", e)
            await asyncio.sleep(cfg.RECONNECT_DELAY)
