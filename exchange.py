"""거래소 연결 — ccxt.pro 초기화, 격리마진/레버리지 세팅."""

import logging

import ccxt.pro as ccxtpro

import config as cfg

logger = logging.getLogger(__name__)


def create_exchange() -> ccxtpro.binance:
    """ccxt.pro 바이낸스 선물 인스턴스를 생성한다."""
    exchange = ccxtpro.binance(
        {
            "apiKey": cfg.API_KEY,
            "secret": cfg.API_SECRET,
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
        }
    )
    if cfg.TESTNET:
        exchange.set_sandbox_mode(True)
        logger.info("Testnet 모드 활성화")
    return exchange


async def setup_leverage(exchange: ccxtpro.binance) -> None:
    """격리마진 + 레버리지를 설정한다."""
    try:
        await exchange.set_margin_mode(cfg.MARGIN_TYPE, cfg.SYMBOL)
        logger.info("마진 모드 설정: %s", cfg.MARGIN_TYPE)
    except Exception as e:
        # 이미 설정된 경우 에러 무시
        logger.warning("마진 모드 설정 중 예외 (이미 설정됨?): %s", e)

    try:
        await exchange.set_leverage(cfg.LEVERAGE, cfg.SYMBOL)
        logger.info("레버리지 설정: %dx", cfg.LEVERAGE)
    except Exception as e:
        logger.warning("레버리지 설정 중 예외: %s", e)
