"""coinbot 설정 — 모든 상수/파라미터를 여기서 정의."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# ── API ──────────────────────────────────────────────
TESTNET: bool = os.getenv("TESTNET", "true").lower() == "true"

_key_var = "TESTNET_API_KEY" if TESTNET else "REAL_API_KEY"
_secret_var = "TESTNET_API_SECRET" if TESTNET else "REAL_API_SECRET"
API_KEY: str = os.getenv(_key_var, "")
API_SECRET: str = os.getenv(_secret_var, "")

# ── Telegram ─────────────────────────────────────────
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── 심볼 & 타임프레임 ───────────────────────────────
SYMBOLS: list[str] = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "XRP/USDT:USDT",
]
TIMEFRAME: str = "4h"

# ── 레버리지 & 마진 ─────────────────────────────────
LEVERAGE: int = 3
MARGIN_TYPE: str = "isolated"
POSITION_RATIO: float = 0.30   # 코인당 자본 비율 (30%) — 백테스트 기준값

# ── 4H BB+RSI 양방향 전략 파라미터 ───────────────────
# 백테스트 근거 (2022~2025): BTC +157.2%, ETH +157.8%, XRP +133.6%
# SL=ATR×sl_mult, TP=ATR×tp_mult, EMA200 필터 ON
SYMBOL_STRATEGY: dict[str, dict] = {
    "BTC/USDT:USDT": {"rsi_long": 30, "rsi_short": 65, "sl_mult": 2.0, "tp_mult": 3.0},
    "ETH/USDT:USDT": {"rsi_long": 25, "rsi_short": 65, "sl_mult": 2.0, "tp_mult": 2.0},
    "XRP/USDT:USDT": {"rsi_long": 25, "rsi_short": 65, "sl_mult": 2.0, "tp_mult": 3.0},
}

# 포지션 타임아웃: 백테스트 48봉 × 4h = 192시간
SIGNAL_TIMEOUT_HOURS: int = 192

# ── 리스크 ───────────────────────────────────────────
MAX_DAILY_LOSS_PCT: float = -5.0

# ── 캔들 버퍼 (EMA200 + 여유) ────────────────────────
CANDLE_BUFFER_SIZE: int = 250
INITIAL_CANDLE_LOAD: int = 250

# ── 재연결 ───────────────────────────────────────────
RECONNECT_DELAY: int = 5

# ── 로깅 ─────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(message)s"
LOG_LEVEL: int = logging.INFO
