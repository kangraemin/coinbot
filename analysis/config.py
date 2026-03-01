"""분석 모듈 설정."""

from pathlib import Path

# 대상 코인 및 타임프레임
SYMBOLS = ['BTC/USDT', 'ETH/USDT']
COIN_NAMES = ['btc', 'eth']
TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']
LOOKBACK_DAYS = 90  # 최근 3개월

# 데이터 경로
DATA_DIR = Path(__file__).parent.parent / 'data' / 'market'
OUTPUT_DIR = Path(__file__).parent / 'output'

# 백테스팅 파라미터
TP_PCT = 0.02       # 2% 익절
SL_PCT = 0.01       # 1% 손절
LEVERAGE = [3, 5, 7]
FEE_RATE = 0.0005   # 0.05%
SLIPPAGE = 0.0003   # 3bp
TIMEOUT_BARS = 48   # 타임아웃 봉 수

# 리샘플링 규칙
RESAMPLE_RULES = {
    '3m':  '3min',
    '5m':  '5min',
    '15m': '15min',
    '30m': '30min',
    '1h':  '1h',
    '2h':  '2h',
    '4h':  '4h',
    '1d':  '1D',
}
