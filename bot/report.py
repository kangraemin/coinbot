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


def _rsi_bar(rsi_val: float) -> str:
    """RSI를 5칸 블록 바로 표현."""
    if rsi_val < 20:
        return "🟩⬜⬜⬜⬜ 극과매도"
    elif rsi_val < 40:
        return "🟦🟩⬜⬜⬜ 과매도"
    elif rsi_val < 60:
        return "⬜⬜🟩⬜⬜ 중립"
    elif rsi_val < 80:
        return "⬜⬜⬜🟩🟥 과매수"
    else:
        return "⬜⬜⬜⬜🟩 극과매수"


def _candle_action_hint(symbol: str, rsi: float, bb_pct: float,
                        ema_diff_pct: float, has_position: bool) -> str:
    """4H 봉 마감 기준 행동 지침 한 줄."""
    p = cfg.SYMBOL_STRATEGY.get(symbol, {})
    rsi_long = p.get("rsi_long", 30)
    rsi_short = p.get("rsi_short", 65)

    if has_position:
        return ""  # 포지션 보유 시 별도 표시 불필요

    hints = []

    # 롱 임박 체크
    long_gap = rsi - rsi_long
    if ema_diff_pct > 0 and long_gap <= 10 and bb_pct < 30:
        if long_gap <= 0:
            hints.append(f"🟢 롱 진입 조건 충족! RSI {rsi:.0f} < {rsi_long}")
        else:
            hints.append(f"🟡 롱 진입 임박 — RSI {rsi:.0f}, 목표 {rsi_long}까지 {long_gap:.0f} 남음")

    # 숏 임박 체크
    short_gap = rsi_short - rsi
    if ema_diff_pct < 0 and short_gap <= 10 and bb_pct > 70:
        if short_gap <= 0:
            hints.append(f"🔴 숏 진입 조건 충족! RSI {rsi:.0f} > {rsi_short}")
        else:
            hints.append(f"🟡 숏 진입 임박 — RSI {rsi:.0f}, 목표 {rsi_short}까지 {short_gap:.0f} 남음")

    if hints:
        return "\n".join(hints)
    return "⚪ 관망 — 진입 조건 미충족"


async def send_candle_status(statuses: list[dict]) -> None:
    """4H 봉 마감 인디케이터 상태 알림 (통합 메시지)."""
    if not statuses:
        return

    lines = ["📊 *4H 봉 마감 상태*"]
    for s in statuses:
        coin = s["symbol"].split("/")[0]
        close = s["close"]
        rsi = s["rsi"]
        bb_lower = s["bb_lower"]
        bb_upper = s["bb_upper"]
        ema200 = s["ema200"]

        # 포지션 상태
        has_position = s.get("has_position") and s.get("direction")
        if has_position:
            arrow = "📈" if s["direction"] == "long" else "📉"
            side_kr = "롱" if s["direction"] == "long" else "숏"
            pos_lines = [f"{arrow} 포지션: {side_kr} @ {s['entry_price']:,.4f}"]
            if s.get("tp_price"):
                pos_lines.append(f"  익절(TP): {s['tp_price']:,.4f}")
            if s.get("sl_price"):
                pos_lines.append(f"  손절(SL): {s['sl_price']:,.4f}")
            if s.get("entry_time"):
                from datetime import datetime, timezone, timedelta
                elapsed = datetime.now(timezone.utc) - s["entry_time"]
                elapsed_h = elapsed.total_seconds() / 3600
                timeout_h = 192  # SIGNAL_TIMEOUT_HOURS
                remain_h = max(0, timeout_h - elapsed_h)
                pos_lines.append(f"  경과: {elapsed_h:.0f}h / 타임아웃: {remain_h:.0f}h 남음")
            pos_str = "\n".join(pos_lines)
        else:
            pos_str = "⏳ 포지션: 대기 중"

        # BB 대비 위치 (%)
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_pct = (close - bb_lower) / bb_range * 100
        else:
            bb_pct = 50.0

        # EMA200 대비 위치
        ema_diff_pct = (close - ema200) / ema200 * 100

        # 해석
        comments = []
        if rsi < 30:
            comments.append("RSI 과매도 구간")
        elif rsi > 70:
            comments.append("RSI 과매수 구간")
        elif rsi < 40:
            comments.append("RSI 약세")
        elif rsi > 60:
            comments.append("RSI 강세")
        else:
            comments.append("RSI 중립")

        if bb_pct < 20:
            comments.append("BB 하단 근접")
        elif bb_pct > 80:
            comments.append("BB 상단 근접")

        if ema_diff_pct > 0:
            comments.append("EMA200 위 (상승 추세)")
        else:
            comments.append("EMA200 아래 (하락 추세)")

        comment_str = " / ".join(comments)

        # 행동 지침
        hint = _candle_action_hint(s["symbol"], rsi, bb_pct, ema_diff_pct, bool(has_position))

        # 코인별 진입 조건
        p = s.get("strategy", {})
        rsi_long = p.get("rsi_long", "?")
        rsi_short = p.get("rsi_short", "?")
        sl_mult = p.get("sl_mult", "?")
        tp_mult = p.get("tp_mult", "?")
        cond_str = (
            f"🟢 롱: 종가<BB하단 & RSI<{rsi_long} & 종가>EMA200\n"
            f"🔴 숏: 종가>BB상단 & RSI>{rsi_short} & 종가<EMA200\n"
            f"  TP: ATR×{tp_mult} / SL: ATR×{sl_mult}"
        )

        block = (
            f"\n🔸 *{coin}*\n"
            f"종가: {close:,.4f}\n"
            f"RSI(14): {rsi:.1f} {_rsi_bar(rsi)}\n"
            f"BB 상단: {bb_upper:,.4f}\n"
            f"BB 하단: {bb_lower:,.4f}\n"
            f"BB 위치: {bb_pct:.0f}%\n"
            f"EMA200: {ema200:,.4f} ({ema_diff_pct:+.1f}%)\n"
            f"{pos_str}\n"
            f"💬 {comment_str}"
        )
        if hint:
            block += f"\n→ {hint}"
        block += f"\n📋 진입 조건:\n{cond_str}"

        lines.append(block)

    # 용어 설명
    lines.append(
        "\n─────────────\n"
        "📖 *용어 설명*\n"
        "RSI: 상대강도지수 (30↓ 과매도, 70↑ 과매수)\n"
        "BB: 볼린저밴드 (가격 변동 범위)\n"
        "BB 위치: 0%=하단, 100%=상단\n"
        "EMA200: 200봉 지수이동평균 (장기 추세)"
    )

    msg = "\n".join(lines)
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
