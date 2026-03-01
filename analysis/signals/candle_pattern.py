"""캔들 패턴 신호 — 망치형(hammer) 및 강세장악형(bullish engulfing)."""

import pandas as pd

NAME = 'candle_pattern'

# 망치형 파라미터
HAMMER_BODY_RATIO = 0.3   # 몸통 비율 (전체 범위 대비) 최대
HAMMER_LOWER_RATIO = 0.6  # 아래 꼬리 비율 (전체 범위 대비) 최소
HAMMER_UPPER_RATIO = 0.1  # 위 꼬리 비율 (전체 범위 대비) 최대


def _hammer(df: pd.DataFrame) -> pd.Series:
    """망치형 캔들 탐지.

    조건:
      - 아래 꼬리가 전체 범위의 60% 이상
      - 몸통이 전체 범위의 30% 이하
      - 위 꼬리가 전체 범위의 10% 이하
    """
    total_range = df['high'] - df['low']
    body = (df['close'] - df['open']).abs()
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)

    # 범위가 0인 봉 제외
    valid = total_range > 0

    body_ratio = body / total_range
    lower_ratio = lower_shadow / total_range
    upper_ratio = upper_shadow / total_range

    return (
        valid &
        (body_ratio <= HAMMER_BODY_RATIO) &
        (lower_ratio >= HAMMER_LOWER_RATIO) &
        (upper_ratio <= HAMMER_UPPER_RATIO)
    ).fillna(False)


def _bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """강세장악형(bullish engulfing) 캔들 탐지.

    조건:
      - 이전 봉: 음봉 (close < open)
      - 현재 봉: 양봉 (close > open)
      - 현재 봉 시가 <= 이전 봉 종가 (이전 음봉 종가 아래서 시작)
      - 현재 봉 종가 >= 이전 봉 시가 (이전 음봉 시가 위에서 마감)
    """
    prev_bearish = df['close'].shift(1) < df['open'].shift(1)
    curr_bullish = df['close'] > df['open']
    engulfs_low = df['open'] <= df['close'].shift(1)
    engulfs_high = df['close'] >= df['open'].shift(1)

    return (prev_bearish & curr_bullish & engulfs_low & engulfs_high).fillna(False)


def detect(df: pd.DataFrame) -> pd.Series:
    """망치형 또는 강세장악형 캔들 패턴을 진입 신호로 반환.

    Args:
        df: timestamp, open, high, low, close, volume 컬럼을 가진 DataFrame

    Returns:
        bool Series (True = 진입 신호)
    """
    return (_hammer(df) | _bullish_engulfing(df)).fillna(False)
