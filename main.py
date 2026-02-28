"""coinbot 진입점 — asyncio 이벤트 루프, WebSocket 캔들 수신, 재연결."""

import asyncio
import logging
import signal
from collections import deque

import config as cfg
from exchange import create_exchange, setup_leverage
from risk import risk_loop
from strategy import strategy_loop

logging.basicConfig(format=cfg.LOG_FORMAT, level=cfg.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ── shared_state: 모든 루프가 공유하는 전역 상태 ────
shared_state: dict = {
    "candles": deque(maxlen=cfg.CANDLE_BUFFER_SIZE),
    "last_price": 0.0,
    "indicators": {},
    "trading_halted": False,
}


async def load_initial_candles(exchange) -> None:
    """과거 캔들 100개를 REST로 로드하여 shared_state에 채운다."""
    try:
        ohlcv = await exchange.fetch_ohlcv(
            cfg.SYMBOL, cfg.TIMEFRAME, limit=cfg.INITIAL_CANDLE_LOAD
        )
        for candle in ohlcv:
            shared_state["candles"].append(candle)
        if ohlcv:
            shared_state["last_price"] = ohlcv[-1][4]  # close price
        logger.info("초기 캔들 %d개 로드 완료", len(ohlcv))
    except Exception as e:
        logger.error("초기 캔들 로드 실패: %s", e)


async def data_loop(exchange) -> None:
    """WebSocket으로 캔들을 수신하여 shared_state를 업데이트한다."""
    while True:
        try:
            ohlcv = await exchange.watch_ohlcv(cfg.SYMBOL, cfg.TIMEFRAME)
            for candle in ohlcv:
                # 마지막 캔들과 같은 타임스탬프면 업데이트, 아니면 추가
                if (
                    shared_state["candles"]
                    and shared_state["candles"][-1][0] == candle[0]
                ):
                    shared_state["candles"][-1] = candle
                else:
                    shared_state["candles"].append(candle)
                shared_state["last_price"] = candle[4]
        except Exception as e:
            logger.error("WebSocket 캔들 수신 오류: %s", e)
            logger.info("%d초 후 재연결 시도...", cfg.RECONNECT_DELAY)
            await asyncio.sleep(cfg.RECONNECT_DELAY)


async def main() -> None:
    """메인 루프: 거래소 초기화 → 캔들 로드 → 루프 실행."""
    exchange = create_exchange()

    try:
        await setup_leverage(exchange)
        await load_initial_candles(exchange)

        logger.info("coinbot 시작 — %s %s", cfg.SYMBOL, cfg.TIMEFRAME)

        await asyncio.gather(
            data_loop(exchange),
            strategy_loop(exchange, shared_state),
            risk_loop(exchange, shared_state),
        )
    except asyncio.CancelledError:
        logger.info("봇 종료 요청 수신")
    finally:
        await exchange.close()
        logger.info("거래소 연결 종료")


def _handle_shutdown(loop: asyncio.AbstractEventLoop) -> None:
    """SIGINT/SIGTERM 시 모든 태스크를 취소한다."""
    logger.info("종료 시그널 수신, 정리 중...")
    for task in asyncio.all_tasks(loop):
        task.cancel()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_shutdown, loop)

    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
        logger.info("coinbot 종료")
