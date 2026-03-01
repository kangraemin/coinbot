"""EMA 크로스오버 신호 — fast EMA(9)가 slow EMA(21)를 상향 돌파."""

import pandas as pd

NAME = 'ema_cross'
EMA_FAST = 9
EMA_SLOW = 21


def detect(df: pd.DataFrame) -> pd.Series:
    """fast EMA(9)가 slow EMA(21)를 상향 돌파하는 봉을 진입 신호로 반환.

    Args:
        df: timestamp, open, high, low, close, volume 컬럼을 가진 DataFrame

    Returns:
        bool Series (True = 진입 신호)
    """
    close = df['close']
    ema_fast = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=EMA_SLOW, adjust=False).mean()

    # 이전 봉: fast <= slow, 현재 봉: fast > slow
    prev_below = ema_fast.shift(1) <= ema_slow.shift(1)
    curr_above = ema_fast > ema_slow

    return (prev_below & curr_above).fillna(False)
