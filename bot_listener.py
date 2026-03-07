"""Telegram bot listener -- /status command for real-time coin status.

Long polling, separate process from main bot.
Usage: python bot_listener.py
Systemd: coinbot-listener.service
"""

import asyncio
import logging
import traceback
from collections import deque

import aiohttp

import config as cfg
from bot.exchange import create_exchange
from bot.strategy import _compute_indicators

logging.basicConfig(format=cfg.LOG_FORMAT, level=cfg.LOG_LEVEL)
logger = logging.getLogger(__name__)

BOT_TOKEN = cfg.TELEGRAM_TOKEN
ALLOWED_CHAT_ID = cfg.TELEGRAM_CHAT_ID
POLL_TIMEOUT = 30

# symbol shorthand mapping: "BTC" -> "BTC/USDT:USDT"
_COIN_MAP: dict[str, str] = {}
for _s in cfg.SYMBOLS:
    _coin = _s.split("/")[0]
    _COIN_MAP[_coin] = _s


async def get_updates(session: aiohttp.ClientSession, offset: int | None = None) -> list:
    """Telegram getUpdates (long polling)."""
    params = {"timeout": POLL_TIMEOUT}
    if offset is not None:
        params["offset"] = offset
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=POLL_TIMEOUT + 10)) as resp:
            data = await resp.json()
            if data.get("ok"):
                return data.get("result", [])
    except Exception:
        await asyncio.sleep(5)
    return []


async def send_reply(session: aiohttp.ClientSession, chat_id: str, text: str) -> None:
    """Telegram message send."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.warning("Send error: %d %s", resp.status, body)
    except Exception as e:
        logger.warning("Send error: %s", e)


def _rsi_bar(rsi_val: float) -> str:
    """RSI를 5칸 블록 바로 표현."""
    if rsi_val < 20:
        return "\U0001f7e9\u2b1c\u2b1c\u2b1c\u2b1c 극과매도"
    elif rsi_val < 40:
        return "\U0001f7e6\U0001f7e9\u2b1c\u2b1c\u2b1c 과매도"
    elif rsi_val < 60:
        return "\u2b1c\u2b1c\U0001f7e9\u2b1c\u2b1c 중립"
    elif rsi_val < 80:
        return "\u2b1c\u2b1c\u2b1c\U0001f7e9\U0001f7e5 과매수"
    else:
        return "\u2b1c\u2b1c\u2b1c\u2b1c\U0001f7e9 극과매수"


def _action_hint(symbol: str, rsi: float, bb_pct: float, ema_diff_pct: float,
                 position: dict | None) -> str:
    """현재 상태 기반 행동 지침 한 줄."""
    p = cfg.SYMBOL_STRATEGY.get(symbol, {})
    rsi_long = p.get("rsi_long", 30)
    rsi_short = p.get("rsi_short", 65)

    has_pos = position and abs(position.get("contracts", 0)) > 0

    if has_pos:
        side = position.get("side", "")
        unrealized_pnl = float(position.get("unrealizedPnl", 0))
        arrow = "\U0001f4c8" if side == "long" else "\U0001f4c9"
        side_kr = "롱" if side == "long" else "숏"
        return f"{arrow} {side_kr} 보유 중 \u2014 미실현 {unrealized_pnl:+,.2f} USDT"

    hints = []

    # 롱 임박 체크: RSI가 rsi_long 근처이고 EMA200 위
    long_gap = rsi - rsi_long
    if ema_diff_pct > 0 and long_gap <= 10 and bb_pct < 30:
        if long_gap <= 0:
            hints.append(f"\U0001f7e2 롱 진입 조건 충족! RSI {rsi:.0f} < {rsi_long}")
        else:
            hints.append(f"\U0001f7e1 롱 진입 임박 \u2014 RSI {rsi:.0f}, 목표 {rsi_long}까지 {long_gap:.0f} 남음")

    # 숏 임박 체크: RSI가 rsi_short 근처이고 EMA200 아래
    short_gap = rsi_short - rsi
    if ema_diff_pct < 0 and short_gap <= 10 and bb_pct > 70:
        if short_gap <= 0:
            hints.append(f"\U0001f534 숏 진입 조건 충족! RSI {rsi:.0f} > {rsi_short}")
        else:
            hints.append(f"\U0001f7e1 숏 진입 임박 \u2014 RSI {rsi:.0f}, 목표 {rsi_short}까지 {short_gap:.0f} 남음")

    if hints:
        return "\n".join(hints)
    return "\u26aa 관망 \u2014 진입 조건 미충족"


def _format_coin_status(ind: dict, symbol: str, position: dict | None, detailed: bool = False) -> str:
    """코인별 상태 포맷."""
    coin = symbol.split("/")[0]
    p = cfg.SYMBOL_STRATEGY.get(symbol, {})

    close = ind["close"]
    rsi = ind["rsi"]
    bb_lower = ind["bb_lower"]
    bb_upper = ind["bb_upper"]
    ema200 = ind["ema200"]

    # BB 위치 %
    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / bb_range * 100 if bb_range > 0 else 50.0

    # EMA200 대비 %
    ema_diff_pct = (close - ema200) / ema200 * 100

    # 포지션 상태
    if position and abs(position.get("contracts", 0)) > 0:
        side = position.get("side", "")
        entry = float(position.get("entryPrice", 0))
        contracts = abs(float(position.get("contracts", 0)))
        unrealized_pnl = float(position.get("unrealizedPnl", 0))
        arrow = "\U0001f4c8" if side == "long" else "\U0001f4c9"
        side_kr = "롱" if side == "long" else "숏"

        pos_lines = [f"{arrow} 포지션: {side_kr} @ {entry:,.4f}"]
        pos_lines.append(f"  수량: {contracts:.6f} ({contracts * entry:,.2f} USDT)")
        pnl_emoji = "+" if unrealized_pnl >= 0 else ""
        pos_lines.append(f"  미실현 손익: {pnl_emoji}{unrealized_pnl:,.2f} USDT")

        # TP/SL 추정 (ATR 기반)
        atr = ind.get("atr", 0)
        if atr > 0:
            tp_mult = p.get("tp_mult", 3.0)
            sl_mult = p.get("sl_mult", 2.0)
            if side == "long":
                tp_price = entry + atr * tp_mult
                sl_price = entry - atr * sl_mult
            else:
                tp_price = entry - atr * tp_mult
                sl_price = entry + atr * sl_mult
            pos_lines.append(f"  익절(TP): {tp_price:,.4f} / 손절(SL): {sl_price:,.4f}")

        pos_str = "\n".join(pos_lines)
    else:
        pos_str = "\u23f3 포지션: 대기 중"

    # 해석
    comments = []
    if rsi < 30:
        comments.append("RSI 과매도")
    elif rsi > 70:
        comments.append("RSI 과매수")
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
    hint = _action_hint(symbol, rsi, bb_pct, ema_diff_pct, position)

    lines = [
        f"\n\U0001f538 *{coin}*",
        f"종가: {close:,.4f}",
        f"RSI(14): {rsi:.1f} {_rsi_bar(rsi)}",
        f"BB 상단: {bb_upper:,.4f}",
        f"BB 하단: {bb_lower:,.4f}",
        f"BB 위치: {bb_pct:.0f}%",
        f"EMA200: {ema200:,.4f} ({ema_diff_pct:+.1f}%)",
        pos_str,
        f"\U0001f4ac {comment_str}",
        f"\u2192 {hint}",
    ]

    if detailed:
        rsi_long = p.get("rsi_long", "?")
        rsi_short = p.get("rsi_short", "?")
        sl_mult = p.get("sl_mult", "?")
        tp_mult = p.get("tp_mult", "?")
        lines.append(
            f"\U0001f4cb 진입 조건:\n"
            f"\U0001f7e2 롱: 종가<BB하단 & RSI<{rsi_long} & 종가>EMA200\n"
            f"\U0001f534 숏: 종가>BB상단 & RSI>{rsi_short} & 종가<EMA200\n"
            f"  TP: ATR\u00d7{tp_mult} / SL: ATR\u00d7{sl_mult}"
        )

    return "\n".join(lines)


async def build_status(exchange, symbol_key: str | None = None) -> str:
    """Build /status response with live data from Binance."""
    if symbol_key:
        full_symbol = _COIN_MAP.get(symbol_key.upper())
        if not full_symbol:
            available = ", ".join(_COIN_MAP.keys())
            return f"알 수 없는 코인: {symbol_key}\n사용 가능: {available}"
        symbols = [full_symbol]
        detailed = True
    else:
        symbols = cfg.SYMBOLS
        detailed = False

    # Fetch positions
    try:
        positions = await exchange.fetch_positions()
        pos_map = {p["symbol"]: p for p in positions if abs(p.get("contracts", 0)) > 0}
    except Exception as e:
        logger.warning("Position fetch error: %s", e)
        pos_map = {}

    lines = ["\U0001f4ca *coinbot 현황*"]

    for sym in symbols:
        try:
            ohlcv = await exchange.fetch_ohlcv(sym, cfg.TIMEFRAME, limit=cfg.INITIAL_CANDLE_LOAD)
            candles = deque(ohlcv, maxlen=cfg.CANDLE_BUFFER_SIZE)
            ind = _compute_indicators(candles)
            if ind is None:
                lines.append(f"\n\U0001f538 *{sym.split('/')[0]}* - 데이터 부족")
                continue

            position = pos_map.get(sym)
            lines.append(_format_coin_status(ind, sym, position, detailed=detailed))

        except Exception as e:
            logger.warning("[%s] Status error: %s", sym, e)
            lines.append(f"\n\U0001f538 *{sym.split('/')[0]}* - 오류")

    # Balance
    try:
        bal = await exchange.fetch_balance()
        total = float(bal.get("total", {}).get("USDT", 0) or 0)
        free = float(bal.get("free", {}).get("USDT", 0) or 0)
        lines.append(f"\n\U0001f4b0 잔액: {total:,.2f} USDT (가용: {free:,.2f})")
    except Exception as e:
        logger.warning("Balance error: %s", e)

    if detailed:
        lines.append(
            "\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            "\U0001f4d6 *용어 설명*\n"
            "RSI: 상대강도지수 (30\u2193 과매도, 70\u2191 과매수)\n"
            "BB: 볼린저밴드 (가격 변동 범위)\n"
            "BB 위치: 0%=하단, 100%=상단\n"
            "EMA200: 200봉 지수이동평균 (장기 추세)"
        )

    return "\n".join(lines)


def build_help() -> str:
    """/help 응답."""
    coins = ", ".join(_COIN_MAP.keys())
    return (
        "\U0001f4cb *사용 가능한 명령어*\n\n"
        f"/status \u2014 전체 코인 요약\n"
        f"/status BTC \u2014 개별 코인 상세\n"
        f"/help \u2014 도움말\n\n"
        f"코인: {coins}"
    )


async def handle_message(
    session: aiohttp.ClientSession,
    exchange,
    msg: dict,
) -> None:
    """Parse message and dispatch command."""
    text = msg.get("text", "")
    chat_id = str(msg["chat"]["id"])

    if ALLOWED_CHAT_ID and chat_id != ALLOWED_CHAT_ID:
        return

    if text.startswith("/status"):
        parts = text.split()
        symbol_key = parts[1].upper() if len(parts) > 1 else None
        reply = await build_status(exchange, symbol_key)
        await send_reply(session, chat_id, reply)
    elif text.startswith("/help"):
        reply = build_help()
        await send_reply(session, chat_id, reply)


async def main() -> None:
    """Long polling main loop."""
    exchange = create_exchange()
    await exchange.load_markets()
    logger.info("Bot listener started. Waiting for commands...")

    offset = None
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                updates = await get_updates(session, offset)
                for u in updates:
                    if "message" in u:
                        await handle_message(session, exchange, u["message"])
                    offset = u["update_id"] + 1
            except KeyboardInterrupt:
                logger.info("Bot listener stopped.")
                break
            except Exception:
                traceback.print_exc()
                await asyncio.sleep(5)

    await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
