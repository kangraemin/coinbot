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
CANDLE_BUFFER_SIZE: int = 100
INITIAL_CANDLE_LOAD: int = 100

# ── 지표 파라미터 ────────────────────────────────────
BB_PERIOD: int = 20
BB_STD: float = 2.0
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14

# ── 진입 조건 임계값 ────────────────────────────────
RSI_THRESHOLD: float = 35.0         # 롱 진입: RSI ≤ 이 값 (과매도)
RSI_OVERBOUGHT: float = 65.0        # 숏 진입: RSI ≥ 이 값 (과매수)
ADX_TREND_THRESHOLD: float = 25.0   # ADX >= 25 → 추세장, 평균회귀 스킵

# ── 익절/손절 ────────────────────────────────────────
# TP: BB 중심선 (평균회귀 타겟) — TP_ATR_MULT 미사용
TP_ATR_MULT: float = 2.0            # 레거시, 미사용
SL_ATR_MULT: float = 1.5            # 진입가 - ATR × 1.5

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
