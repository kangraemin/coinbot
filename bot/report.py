"""리포트 — 일일 요약 + Telegram 알림."""

import logging
from collections import defaultdict

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


async def send_entry_order_alert(
    symbol: str,
    entry_price: float,
    amount: float,
    tp_price: float,
    sl_price: float,
) -> None:
    """진입 리밋 주문 배치 알림."""
    coin = symbol.split("/")[0]
    notional = entry_price * amount
    msg = (
        f"🟡 *coinbot 진입 주문 대기*\n"
        f"코인: {coin}\n"
        f"주문가: {entry_price:,.4f} USDT\n"
        f"수량: {amount:.6f} ({notional:,.2f} USDT)\n"
        f"익절: {tp_price:,.4f}\n"
        f"손절: {sl_price:,.4f}\n"
    )
    await send_telegram(msg)


async def send_order_update_alert(
    symbol: str,
    old_prev_close: float,
    new_prev_close: float,
    old_entry_price: float,
    new_entry_price: float,
) -> None:
    """주문 갱신 알림 (prev_close 변동으로 재주문 시)."""
    coin = symbol.split("/")[0]
    change_pct = (new_prev_close - old_prev_close) / old_prev_close * 100
    msg = (
        f"🔄 *coinbot 주문 갱신*\n"
        f"코인: {coin}\n"
        f"기준가: {old_prev_close:,.4f} → {new_prev_close:,.4f} ({change_pct:+.2f}%)\n"
        f"주문가: {old_entry_price:,.4f} → {new_entry_price:,.4f}\n"
    )
    await send_telegram(msg)


async def send_trade_alert(
    side: str,
    price: float,
    amount: float,
    tp_price: float | None = None,
    sl_price: float | None = None,
    symbol: str | None = None,
    leverage: int = 1,
) -> None:
    """매매 체결 알림을 발송한다."""
    coin = symbol.split("/")[0] if symbol else "?"
    msg = (
        f"📊 *coinbot 주문 체결*\n"
        f"코인: {coin}\n"
        f"방향: {side.upper()}\n"
        f"가격: {price:,.4f} USDT\n"
        f"수량: {amount:.6f}\n"
        f"레버리지: {leverage}x\n"
    )
    if tp_price:
        pnl_pct = abs(tp_price - price) / price * leverage * 100
        pnl_usdt = abs(tp_price - price) * amount
        msg += f"익절: {tp_price:,.4f} (+{pnl_pct:.1f}% / +${pnl_usdt:.2f})\n"
    if sl_price:
        loss_pct = abs(sl_price - price) / price * leverage * 100
        loss_usdt = abs(sl_price - price) * amount
        msg += f"손절: {sl_price:,.4f} (-{loss_pct:.1f}% / -${loss_usdt:.2f})\n"

    await send_telegram(msg)


async def send_close_alert(
    entry_price: float,
    exit_price: float,
    pnl: float,
    fee: float,
    symbol: str | None = None,
) -> None:
    """포지션 종료 알림을 발송한다."""
    coin = symbol.split("/")[0] if symbol else "?"
    emoji = "✅" if pnl >= 0 else "❌"
    msg = (
        f"{emoji} *coinbot 포지션 종료*\n"
        f"코인: {coin}\n"
        f"진입: {entry_price:,.4f}\n"
        f"종료: {exit_price:,.4f}\n"
        f"수수료: {fee:,.4f} USDT\n"
        f"순익: {pnl:+,.2f} USDT\n"
    )
    await send_telegram(msg)


async def send_capital_alert(balance: float, target: float) -> None:
    """원금 2배 달성 출금 권고 알림."""
    msg = (
        f"🎉 *coinbot 원금 2배 달성!*\n"
        f"현재 잔액: {balance:,.2f} USDT\n"
        f"목표 달성: {target:,.2f} USDT\n\n"
        f"💡 *원금 출금 권고*\n"
        f"바이낸스에서 원금(1,300 USDT)을 출금하고\n"
        f"수익분만 봇에 남겨두세요.\n"
        f"→ 이후 수익분만 복리 운용"
    )
    await send_telegram(msg)


async def send_daily_report(trades: list[dict], balance: float = 0.0) -> None:
    """일일 요약 리포트를 발송한다."""
    if not trades:
        msg = f"📋 *coinbot 일일 리포트*\n오늘 거래 없음\n\n💰 현재 잔액: {balance:,.2f} USDT"
        await send_telegram(msg)
        return

    closed = [t for t in trades if t.get("status") == "closed"]
    total_pnl = sum(t.get("pnl", 0) for t in closed)
    total_fee = sum(t.get("fee", 0) for t in closed)
    wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
    win_rate = (wins / len(closed) * 100) if closed else 0

    coin_pnl: dict = defaultdict(float)
    coin_cnt: dict = defaultdict(int)
    for t in closed:
        coin = t.get("symbol", "?").split("/")[0]
        coin_pnl[coin] += t.get("pnl", 0)
        coin_cnt[coin] += 1

    coin_lines = "\n".join(
        f"  {coin}: {coin_cnt[coin]}건 / {coin_pnl[coin]:+.2f} USDT"
        for coin in sorted(coin_cnt)
    )

    msg = (
        f"📋 *coinbot 일일 리포트*\n"
        f"총 체결: {len(closed)}건 | 승률: {win_rate:.0f}%\n"
        f"총 수수료: {total_fee:,.4f} USDT\n"
        f"순익합계: {total_pnl:+,.2f} USDT\n"
        f"\n코인별:\n{coin_lines}\n"
        f"\n💰 현재 잔액: {balance:,.2f} USDT"
    )
    await send_telegram(msg)
