"""RSI 과매도 신호 — RSI(14) < 30."""

import pandas as pd
import ta as ta_lib

NAME = 'rsi_oversold'
RSI_PERIOD = 14
RSI_THRESHOLD = 30.0


def detect(df: pd.DataFrame) -> pd.Series:
    """RSI(14) < 30 인 봉을 진입 신호로 반환.

    Args:
        df: timestamp, open, high, low, close, volume 컬럼을 가진 DataFrame

    Returns:
        bool Series (True = 진입 신호)
    """
    rsi = ta_lib.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
    return (rsi < RSI_THRESHOLD).fillna(False)
