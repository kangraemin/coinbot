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
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
]
TIMEFRAME: str = "1m"

# ── 레버리지 & 마진 ─────────────────────────────────
LEVERAGE: int = 5
MARGIN_TYPE: str = "isolated"

# ── 전략 파라미터 (코인별 최적화, 자본 손실 기준 SL 5년 백테스트 기준) ──────────────
# 공통 기본값
ENTRY_DROP_PCT: float = 1.5   # prev_close 대비 이 % 이상 하락 시 진입
TP_PCT: float = 3.0           # 익절 기본값
SL_PCT: float = 0.2           # 손절 기본값 (자본 1% / 5x = 가격 0.2%)
POSITION_RATIO: float = 0.20  # 코인당 자본 비율 (20%)
MAX_POSITIONS: int = 4        # 코인당 1개

# 심볼별 파라미터 오버라이드 (없으면 위 기본값 사용)
# 자본 손실 1% 기준 SL (sl_pct = sl_capital / leverage = 1% / 5x = 0.2% 가격)
# BTC: entry 1.5% / TP 2.0% / SL 0.2% → 수익 +88%, MDD 4.0%
# ETH: entry 1.0% / TP 1.0% / SL 0.2% → 수익 +451%, MDD 12.0%
# SOL: entry 2.0% / TP 5.0% / SL 0.2% → 수익 +3701%, MDD 10.7%
# XRP: entry 1.5% / TP 3.0% / SL 0.2% → 수익 +2596%, MDD 9.9%
SYMBOL_PARAMS: dict[str, dict] = {
    "BTC/USDT:USDT": {"entry_pct": 1.5, "tp_pct": 2.0, "sl_pct": 0.2},
    "ETH/USDT:USDT": {"entry_pct": 1.0, "tp_pct": 1.0, "sl_pct": 0.2},
    "SOL/USDT:USDT": {"entry_pct": 2.0, "tp_pct": 5.0, "sl_pct": 0.2},
    "XRP/USDT:USDT": {"entry_pct": 1.5, "tp_pct": 3.0, "sl_pct": 0.2},
}

# ── 리스크 ───────────────────────────────────────────
MAX_DAILY_LOSS_PCT: float = -5.0

# ── 캔들 버퍼 ────────────────────────────────────────
CANDLE_BUFFER_SIZE: int = 10
INITIAL_CANDLE_LOAD: int = 5

# ── 재연결 ───────────────────────────────────────────
RECONNECT_DELAY: int = 5

# ── 로깅 ─────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(message)s"
LOG_LEVEL: int = logging.INFO
