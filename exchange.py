"""거래소 연결 — ccxt.pro 초기화, 격리마진/레버리지 세팅."""

import logging

import ccxt.pro as ccxtpro

import config as cfg

logger = logging.getLogger(__name__)


def create_exchange() -> ccxtpro.binance:
    """ccxt.pro 바이낸스 선물 인스턴스를 생성한다.

    TESTNET=true이면 Binance Futures Testnet을 사용한다.
    testnet.binancefuture.com 에서 발급한 API 키 필요.
    공개 시장 데이터(캔들 등)는 메인넷에서, 주문/포지션은 테스트넷에서 실행.
    """
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
    """모든 심볼에 격리마진 + 레버리지를 설정한다."""
    for symbol in cfg.SYMBOLS:
        try:
            await exchange.set_margin_mode(cfg.MARGIN_TYPE, symbol)
            logger.info("[%s] 마진 모드 설정: %s", symbol, cfg.MARGIN_TYPE)
        except Exception as e:
            logger.warning("[%s] 마진 모드 설정 중 예외 (이미 설정됨?): %s", symbol, e)

        try:
            await exchange.set_leverage(cfg.LEVERAGE, symbol)
            logger.info("[%s] 레버리지 설정: %dx", symbol, cfg.LEVERAGE)
        except Exception as e:
            logger.warning("[%s] 레버리지 설정 중 예외: %s", symbol, e)
