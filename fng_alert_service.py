"""F&G 공포/탐욕 지수 알림 서비스.

독립 실행: python fng_alert_service.py
systemd: coinbot-fng-alert.service
"""

import asyncio
import logging
import signal

import config as cfg
from bot.fng_alert import fng_alert_loop

logging.basicConfig(format=cfg.LOG_FORMAT, level=cfg.LOG_LEVEL)
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("F&G 알림 서비스 시작")
    try:
        await fng_alert_loop()
    except asyncio.CancelledError:
        logger.info("F&G 알림 서비스 종료 요청")


def _handle_shutdown(loop: asyncio.AbstractEventLoop) -> None:
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
        logger.info("F&G 알림 서비스 종료")
