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
LEVERAGE: int = 7
MARGIN_TYPE: str = "isolated"

# ── 전략 파라미터 (코인별 최적화, 7x/20% 5년 백테스트 기준) ──────────────
# 공통 기본값
ENTRY_DROP_PCT: float = 1.5   # prev_close 대비 이 % 이상 하락 시 진입
TP_PCT: float = 3.0           # 익절 기본값
SL_PCT: float = 0.5           # 손절 기본값
POSITION_RATIO: float = 0.20  # 코인당 자본 비율 (20%)
MAX_POSITIONS: int = 4        # 코인당 1개

# 심볼별 파라미터 오버라이드 (없으면 위 기본값 사용)
# BTC: sl 1.5% — 노이즈 손절 방지, 승률 26.9%→45.2%, 수익 +93%→+112%
# ETH: tp 2.0% — 빠른 익절, MDD 17.9%→12.5%, 수익 +140%→+153%
SYMBOL_PARAMS: dict[str, dict] = {
    "BTC/USDT:USDT": {"entry_pct": 1.5, "tp_pct": 3.0, "sl_pct": 1.5},
    "ETH/USDT:USDT": {"entry_pct": 1.5, "tp_pct": 2.0, "sl_pct": 0.5},
    "SOL/USDT:USDT": {"entry_pct": 1.5, "tp_pct": 3.0, "sl_pct": 0.5},
    "XRP/USDT:USDT": {"entry_pct": 1.5, "tp_pct": 3.0, "sl_pct": 0.5},
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
