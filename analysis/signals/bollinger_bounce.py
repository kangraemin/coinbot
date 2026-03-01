"""볼린저밴드 하단 반등 신호 — BB 하단 이탈 후 다음봉에서 하단 위로 회복."""

import pandas as pd
import ta as ta_lib

NAME = 'bollinger_bounce'
BB_PERIOD = 20
BB_STD = 2.0


def detect(df: pd.DataFrame) -> pd.Series:
    """BB 하단 이탈 후 다음봉에서 종가가 하단 위로 회복하는 봉을 진입 신호로 반환.

    조건:
      - 이전 봉 종가 < BB 하단 (하단 이탈)
      - 현재 봉 종가 >= BB 하단 (회복)

    Args:
        df: timestamp, open, high, low, close, volume 컬럼을 가진 DataFrame

    Returns:
        bool Series (True = 진입 신호)
    """
    bb = ta_lib.volatility.BollingerBands(
        df['close'], window=BB_PERIOD, window_dev=BB_STD
    )
    bb_lower = bb.bollinger_lband()

    prev_below = df['close'].shift(1) < bb_lower.shift(1)
    curr_above = df['close'] >= bb_lower

    return (prev_below & curr_above).fillna(False)
