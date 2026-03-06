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


def _format_coin_status(ind: dict, symbol: str, position: dict | None, detailed: bool = False) -> str:
    """Single coin status formatting."""
    coin = symbol.split("/")[0]
    p = cfg.SYMBOL_STRATEGY.get(symbol, {})

    close = ind["close"]
    rsi = ind["rsi"]
    bb_lower = ind["bb_lower"]
    bb_upper = ind["bb_upper"]
    ema200 = ind["ema200"]

    # BB position %
    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / bb_range * 100 if bb_range > 0 else 50.0

    # EMA200 relative %
    ema_diff_pct = (close - ema200) / ema200 * 100

    # Position status
    if position and abs(position.get("contracts", 0)) > 0:
        side = position.get("side", "")
        entry = float(position.get("entryPrice", 0))
        contracts = abs(float(position.get("contracts", 0)))
        unrealized_pnl = float(position.get("unrealizedPnl", 0))
        arrow = "\U0001f4c8" if side == "long" else "\U0001f4c9"

        pos_lines = [f"{arrow} position: {side.upper()} @ {entry:,.4f}"]
        pos_lines.append(f"  size: {contracts:.6f} ({contracts * entry:,.2f} USDT)")
        pnl_emoji = "+" if unrealized_pnl >= 0 else ""
        pos_lines.append(f"  unrealized PnL: {pnl_emoji}{unrealized_pnl:,.2f} USDT")

        # TP/SL estimate from ATR
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
            pos_lines.append(f"  TP: {tp_price:,.4f} / SL: {sl_price:,.4f}")

        pos_str = "\n".join(pos_lines)
    else:
        pos_str = "\u23f3 position: standby"

    # Interpretation
    comments = []
    if rsi < 30:
        comments.append("RSI oversold")
    elif rsi > 70:
        comments.append("RSI overbought")
    elif rsi < 40:
        comments.append("RSI weak")
    elif rsi > 60:
        comments.append("RSI strong")
    else:
        comments.append("RSI neutral")

    if bb_pct < 20:
        comments.append("BB lower near")
    elif bb_pct > 80:
        comments.append("BB upper near")

    if ema_diff_pct > 0:
        comments.append("above EMA200 (uptrend)")
    else:
        comments.append("below EMA200 (downtrend)")

    comment_str = " / ".join(comments)

    lines = [
        f"\n\U0001f538 *{coin}*",
        f"close: {close:,.4f}",
        f"RSI(14): {rsi:.1f}",
        f"BB upper: {bb_upper:,.4f}",
        f"BB lower: {bb_lower:,.4f}",
        f"BB position: {bb_pct:.0f}%",
        f"EMA200: {ema200:,.4f} ({ema_diff_pct:+.1f}%)",
        pos_str,
        f"\U0001f4ac {comment_str}",
    ]

    if detailed:
        rsi_long = p.get("rsi_long", "?")
        rsi_short = p.get("rsi_short", "?")
        sl_mult = p.get("sl_mult", "?")
        tp_mult = p.get("tp_mult", "?")
        lines.append(
            f"\U0001f4cb entry conditions:\n"
            f"\U0001f7e2 Long: close<BB_lower & RSI<{rsi_long} & close>EMA200\n"
            f"\U0001f534 Short: close>BB_upper & RSI>{rsi_short} & close<EMA200\n"
            f"  TP: ATR\u00d7{tp_mult} / SL: ATR\u00d7{sl_mult}"
        )

    return "\n".join(lines)


async def build_status(exchange, symbol_key: str | None = None) -> str:
    """Build /status response with live data from Binance."""
    if symbol_key:
        full_symbol = _COIN_MAP.get(symbol_key.upper())
        if not full_symbol:
            available = ", ".join(_COIN_MAP.keys())
            return f"Unknown coin: {symbol_key}\nAvailable: {available}"
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

    lines = ["\U0001f4ca *coinbot status*"]

    for sym in symbols:
        try:
            ohlcv = await exchange.fetch_ohlcv(sym, cfg.TIMEFRAME, limit=cfg.INITIAL_CANDLE_LOAD)
            candles = deque(ohlcv, maxlen=cfg.CANDLE_BUFFER_SIZE)
            ind = _compute_indicators(candles)
            if ind is None:
                lines.append(f"\n\U0001f538 *{sym.split('/')[0]}* - insufficient data")
                continue

            position = pos_map.get(sym)
            lines.append(_format_coin_status(ind, sym, position, detailed=detailed))

        except Exception as e:
            logger.warning("[%s] Status error: %s", sym, e)
            lines.append(f"\n\U0001f538 *{sym.split('/')[0]}* - error")

    # Balance
    try:
        bal = await exchange.fetch_balance()
        total = float(bal.get("total", {}).get("USDT", 0) or 0)
        free = float(bal.get("free", {}).get("USDT", 0) or 0)
        lines.append(f"\n\U0001f4b0 balance: {total:,.2f} USDT (free: {free:,.2f})")
    except Exception as e:
        logger.warning("Balance error: %s", e)

    if detailed:
        lines.append(
            "\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            "\U0001f4d6 *Glossary*\n"
            "RSI: Relative Strength Index (30\u2193 oversold, 70\u2191 overbought)\n"
            "BB: Bollinger Bands (price volatility range)\n"
            "BB position: 0%=lower, 100%=upper\n"
            "EMA200: 200-period Exponential MA (long-term trend)"
        )

    return "\n".join(lines)


def build_help() -> str:
    """/help response."""
    coins = ", ".join(_COIN_MAP.keys())
    return (
        "\U0001f4cb *Available commands*\n\n"
        f"/status \u2014 All coins summary\n"
        f"/status BTC \u2014 Single coin detail\n"
        f"/help \u2014 This help\n\n"
        f"Coins: {coins}"
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
