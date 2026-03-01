"""지지선 반등 신호 — rolling min 기반 지지선에서 반등 (ta 라이브러리 미사용)."""

import pandas as pd

NAME = 'support_bounce'
SUPPORT_PERIOD = 20   # 지지선 계산 기간 (봉)
TOUCH_THRESHOLD = 0.003  # 지지선 ±0.3% 이내를 "터치"로 판단


def detect(df: pd.DataFrame) -> pd.Series:
    """rolling min 기반 지지선 근처에서 반등하는 봉을 진입 신호로 반환.

    조건:
      - 지지선 = 최근 SUPPORT_PERIOD봉의 저가 rolling min (현재 봉 제외)
      - 이전 봉 저가가 지지선 ±TOUCH_THRESHOLD 이내 (지지선 터치)
      - 현재 봉 종가 > 이전 봉 종가 (반등 확인)

    Args:
        df: timestamp, open, high, low, close, volume 컬럼을 가진 DataFrame

    Returns:
        bool Series (True = 진입 신호)
    """
    # 현재 봉 제외한 이전 SUPPORT_PERIOD봉의 저가 최솟값
    support = df['low'].shift(1).rolling(window=SUPPORT_PERIOD).min()

    # 이전 봉 저가가 지지선 근처 (±threshold)
    prev_low = df['low'].shift(1)
    near_support = (
        (prev_low <= support * (1 + TOUCH_THRESHOLD)) &
        (prev_low >= support * (1 - TOUCH_THRESHOLD))
    )

    # 현재 봉에서 반등 확인 (종가 > 이전 봉 종가)
    bouncing = df['close'] > df['close'].shift(1)

    return (near_support & bouncing).fillna(False)
