"""coinbot 설정 — 모든 상수/파라미터를 여기서 정의."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# ── API ──────────────────────────────────────────────
API_KEY: str = os.getenv("API_KEY", "")
API_SECRET: str = os.getenv("API_SECRET", "")
TESTNET: bool = os.getenv("TESTNET", "true").lower() == "true"

# ── Telegram ─────────────────────────────────────────
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── 심볼 & 타임프레임 ───────────────────────────────
SYMBOL: str = "BTC/USDT:USDT"
TIMEFRAME: str = "15m"

# ── 레버리지 & 마진 ─────────────────────────────────
LEVERAGE: int = 3
MARGIN_TYPE: str = "isolated"

# ── 캔들 버퍼 ────────────────────────────────────────
CANDLE_BUFFER_SIZE: int = 200
INITIAL_CANDLE_LOAD: int = 100

# ── 지표 파라미터 ────────────────────────────────────
BB_PERIOD: int = 20
BB_STD: float = 2.0
RSI_PERIOD: int = 14
EMA_LONG: int = 200
EMA_SHORT: int = 50
ATR_PERIOD: int = 14
VOLUME_MA_PERIOD: int = 20

# ── 진입 조건 임계값 ────────────────────────────────
RSI_THRESHOLD: float = 35.0
VOLUME_MULTIPLIER: float = 1.2

# ── 익절/손절 (ATR 배수) ────────────────────────────
TP_ATR_MULT: float = 2.0
SL_ATR_MULT: float = 1.0

# ── 리스크 ───────────────────────────────────────────
TRADE_AMOUNT_USDT: float = float(os.getenv("TRADE_AMOUNT_USDT", "100"))
MAX_DAILY_LOSS_PCT: float = -5.0
ORDER_TIMEOUT_MIN: int = 10
MAX_POSITIONS: int = 1
ENABLE_SHORT: bool = False

# ── 재연결 ───────────────────────────────────────────
RECONNECT_DELAY: int = 5

# ── 로깅 ─────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(message)s"
LOG_LEVEL: int = logging.INFO
