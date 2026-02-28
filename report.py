"""리포트 — 일일 요약 + Telegram 알림."""

import logging

import aiohttp

import config as cfg

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


async def send_telegram(message: str) -> bool:
    """Telegram Bot API로 메시지를 발송한다."""
    if not cfg.TELEGRAM_TOKEN or not cfg.TELEGRAM_CHAT_ID:
        logger.warning("Telegram 설정 없음 — 알림 건너뜀")
        return False

    url = TELEGRAM_API_URL.format(token=cfg.TELEGRAM_TOKEN)
    payload = {
        "chat_id": cfg.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return True
                body = await resp.text()
                logger.warning("Telegram 발송 실패 — %d: %s", resp.status, body)
                return False
    except Exception as e:
        logger.error("Telegram 발송 오류: %s", e)
        return False


async def send_trade_alert(
    side: str, price: float, amount: float, tp_price: float | None = None, sl_price: float | None = None
) -> None:
    """매매 체결 알림을 발송한다."""
    msg = (
        f"📊 *coinbot 주문 체결*\n"
        f"방향: {side.upper()}\n"
        f"가격: {price:,.2f} USDT\n"
        f"수량: {amount:.6f}\n"
    )
    if tp_price:
        msg += f"익절: {tp_price:,.2f}\n"
    if sl_price:
        msg += f"손절: {sl_price:,.2f}\n"

    await send_telegram(msg)


async def send_close_alert(entry_price: float, exit_price: float, pnl: float, fee: float) -> None:
    """포지션 종료 알림을 발송한다."""
    emoji = "✅" if pnl >= 0 else "❌"
    msg = (
        f"{emoji} *coinbot 포지션 종료*\n"
        f"진입: {entry_price:,.2f}\n"
        f"종료: {exit_price:,.2f}\n"
        f"수수료: {fee:,.4f} USDT\n"
        f"순익: {pnl:,.2f} USDT\n"
    )
    await send_telegram(msg)


async def send_daily_report(trades: list[dict]) -> None:
    """일일 요약 리포트를 발송한다."""
    if not trades:
        await send_telegram("📋 *coinbot 일일 리포트*\n오늘 거래 없음")
        return

    total_trades = len(trades)
    closed = [t for t in trades if t.get("status") == "closed"]
    total_pnl = sum(t.get("pnl", 0) for t in closed)
    total_fee = sum(t.get("fee", 0) for t in closed)
    wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
    win_rate = (wins / len(closed) * 100) if closed else 0

    msg = (
        f"📋 *coinbot 일일 리포트*\n"
        f"총 거래: {total_trades}건\n"
        f"체결: {len(closed)}건\n"
        f"승률: {win_rate:.0f}%\n"
        f"총 수수료: {total_fee:,.4f} USDT\n"
        f"순익합계: {total_pnl:,.2f} USDT\n"
    )
    await send_telegram(msg)
