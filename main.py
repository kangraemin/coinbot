"""coinbot 진입점 — asyncio 이벤트 루프, WebSocket 캔들 수신, 재연결."""

import asyncio
import logging
import signal
from collections import deque
from datetime import datetime, timezone, timedelta

import config as cfg
from bot import report
from bot.exchange import create_exchange, setup_leverage
from bot.risk import risk_loop
from bot.strategy import strategy_loop

logging.basicConfig(format=cfg.LOG_FORMAT, level=cfg.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ── shared_state: 심볼별 캔들/가격 + 전역 trading_halted ──
shared_state: dict = {
    "trading_halted": False,
    **{
        symbol: {
            "candles": deque(maxlen=cfg.CANDLE_BUFFER_SIZE),
            "last_price": 0.0,
            "prev_close": 0.0,
        }
        for symbol in cfg.SYMBOLS
    },
}


async def load_initial_candles(exchange, symbol: str) -> None:
    """과거 캔들 N개를 REST로 로드하여 shared_state에 채운다."""
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, cfg.TIMEFRAME, limit=cfg.INITIAL_CANDLE_LOAD)
        sym_data = shared_state[symbol]
        for candle in ohlcv:
            sym_data["candles"].append(candle)
        if len(ohlcv) >= 2:
            sym_data["prev_close"] = ohlcv[-2][4]  # 마지막 완료된 봉의 종가
            sym_data["last_price"] = ohlcv[-1][4]
        logger.info("[%s] 초기 캔들 로드 완료 (prev_close=%.4f)", symbol, sym_data["prev_close"])
    except Exception as e:
        logger.error("[%s] 초기 캔들 로드 실패: %s", symbol, e)


async def data_loop(exchange, symbol: str) -> None:
    """WebSocket으로 1분봉을 수신하여 shared_state를 업데이트한다."""
    sym_data = shared_state[symbol]
    candles = sym_data["candles"]

    while True:
        try:
            ohlcv = await exchange.watch_ohlcv(symbol, cfg.TIMEFRAME)
            for candle in ohlcv:
                if candles and candles[-1][0] == candle[0]:
                    # 동일 봉 업데이트 (현재 형성 중)
                    candles[-1] = candle
                else:
                    # 새 봉 시작 → 직전 봉 마감
                    if candles:
                        sym_data["prev_close"] = candles[-1][4]
                        logger.debug("[%s] 봉 마감 prev_close=%.4f", symbol, sym_data["prev_close"])
                    candles.append(candle)
                sym_data["last_price"] = candle[4]
        except Exception as e:
            logger.error("[%s] WebSocket 오류: %s", symbol, e)
            logger.info("[%s] %d초 후 재연결...", symbol, cfg.RECONNECT_DELAY)
            await asyncio.sleep(cfg.RECONNECT_DELAY)


async def heartbeat_loop(exchange) -> None:
    """1시간마다 봇 상태를 Telegram으로 알린다."""
    while True:
        try:
            await asyncio.sleep(3600)
            bal = await exchange.fetch_balance()
            free = float(bal.get("USDT", {}).get("free", 0))
            total = float(bal.get("USDT", {}).get("total", 0))
            symbols_str = " / ".join(s.split("/")[0] for s in cfg.SYMBOLS)
            await report.send_telegram(
                f"🟢 *coinbot 가동 중*\n"
                f"심볼: {symbols_str}\n"
                f"잔액: {free:,.2f} / {total:,.2f} USDT"
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("하트비트 루프 오류: %s", e)


async def daily_report_loop(exchange) -> None:
    """매일 오전 7시(KST)에 일일 리포트를 발송한다."""
    KST = timezone(timedelta(hours=9))
    while True:
        try:
            now = datetime.now(KST)
            next_7am = now.replace(hour=7, minute=0, second=0, microsecond=0)
            if now >= next_7am:
                next_7am += timedelta(days=1)
            await asyncio.sleep((next_7am - now).total_seconds())

            from bot.journal import get_daily_trades
            trades = get_daily_trades()
            balance_data = await exchange.fetch_balance()
            balance = float(balance_data.get("USDT", {}).get("free", 0))
            await report.send_daily_report(trades, balance)
            logger.info("일일 리포트 발송 완료")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("일일 리포트 루프 오류: %s", e)
            await asyncio.sleep(60)


async def main() -> None:
    """메인 루프: 거래소 초기화 → 캔들 로드 → 루프 실행."""
    exchange = create_exchange()

    try:
        await setup_leverage(exchange)

        for symbol in cfg.SYMBOLS:
            await load_initial_candles(exchange, symbol)

        symbols_str = " / ".join(s.split("/")[0] for s in cfg.SYMBOLS)
        logger.info("coinbot 시작 — %s %s", symbols_str, cfg.TIMEFRAME)
        param_lines = "\n".join(
            f"  {s.split('/')[0]}: -{p.get('entry_pct', cfg.ENTRY_DROP_PCT)}% | TP +{p.get('tp_pct', cfg.TP_PCT)}% | SL -{p.get('sl_pct', cfg.SL_PCT)}%"
            for s in cfg.SYMBOLS
            for p in [cfg.SYMBOL_PARAMS.get(s, {})]
        )
        await report.send_telegram(
            f"🤖 *coinbot 시작*\n"
            f"심볼: {symbols_str}\n"
            f"파라미터 (코인별):\n{param_lines}\n"
            f"레버리지: {cfg.LEVERAGE}x | 포지션 비율: {int(cfg.POSITION_RATIO * 100)}%/코인"
        )

        await asyncio.gather(
            *[data_loop(exchange, symbol) for symbol in cfg.SYMBOLS],
            strategy_loop(exchange, shared_state),
            risk_loop(exchange, shared_state),
            daily_report_loop(exchange),
            heartbeat_loop(exchange),
        )
    except asyncio.CancelledError:
        logger.info("봇 종료 요청 수신")
        await report.send_telegram("🛑 *coinbot 종료*")
    finally:
        await exchange.close()
        logger.info("거래소 연결 종료")


def _handle_shutdown(loop: asyncio.AbstractEventLoop) -> None:
    """SIGINT/SIGTERM 시 모든 태스크를 취소한다."""
    logger.info("종료 시그널 수신, 정리 중...")
    for task in asyncio.all_tasks(loop):
        task.cancel()


if __name__ == "__main__":
    import os, sys

    PID_FILE = "/tmp/coinbot.pid"

    # 중복 실행 방지
    if os.path.exists(PID_FILE):
        try:
            old_pid = int(open(PID_FILE).read().strip())
            os.kill(old_pid, 0)  # 프로세스 존재 확인
            logger.error("이미 실행 중 (PID %d). 종료합니다.", old_pid)
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass  # 기존 PID가 죽었으면 계속 진행

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    loop = asyncio.new_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_shutdown, loop)

    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
        try:
            os.remove(PID_FILE)
        except FileNotFoundError:
            pass
        logger.info("coinbot 종료")
