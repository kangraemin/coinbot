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
from bot.fng_alert import fetch_current_fng, get_fear_streak, build_fng_alert
from bot.format import format_coin_status
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

            # exchange position → 공통 pos dict 변환
            raw_pos = pos_map.get(sym)
            pos = None
            if raw_pos and abs(raw_pos.get("contracts", 0)) > 0:
                side = raw_pos.get("side", "")
                pos = {
                    "side": side,
                    "entry_price": float(raw_pos.get("entryPrice", 0)),
                    "contracts": abs(float(raw_pos.get("contracts", 0))),
                    "unrealized_pnl": float(raw_pos.get("unrealizedPnl", 0)),
                }
            lines.append(format_coin_status(ind, sym, pos))

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
        lines.append("\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        lines.append("")
        lines.append("\U0001f4d6 *용어 설명*")
        lines.append("RSI: 상대강도지수 (30↓ 과매도, 70↑ 과매수)")
        lines.append("BB: 볼린저밴드 (가격 변동 범위)")
        lines.append("BB 위치: 0%=하단, 100%=상단")
        lines.append("EMA200: 200봉 지수이동평균 (장기 추세)")

    # F&G 섹션 추가
    fng_section = await _build_fng_section()
    if fng_section:
        lines.append("\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        lines.append(fng_section)

    return "\n".join(lines)


async def _build_fng_section() -> str | None:
    """F&G 알림 섹션 생성."""
    try:
        fng_value = await fetch_current_fng()
        if fng_value is None:
            return None
        streak = get_fear_streak()
        return build_fng_alert(fng_value, streak)
    except Exception as e:
        logger.warning("F&G section error: %s", e)
        return None


def build_help() -> str:
    """/help 응답."""
    coins = ", ".join(_COIN_MAP.keys())
    return (
        "\U0001f4cb *사용 가능한 명령어*\n\n"
        f"/status \u2014 전체 코인 요약\n"
        f"/status BTC \u2014 개별 코인 상세\n"
        f"/fng \u2014 F&G 공포/탐욕 지수 + DCA 추천\n"
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
    elif text.startswith("/fng"):
        fng_section = await _build_fng_section()
        reply = fng_section or "F&G 데이터 조회 실패"
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
