"""MACD 골든크로스 신호 — MACD 라인이 시그널 라인을 상향 돌파."""

import pandas as pd
import ta as ta_lib

NAME = 'macd_cross'
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def detect(df: pd.DataFrame) -> pd.Series:
    """MACD 라인이 시그널 라인을 상향 돌파하는 봉을 진입 신호로 반환.

    Args:
        df: timestamp, open, high, low, close, volume 컬럼을 가진 DataFrame

    Returns:
        bool Series (True = 진입 신호)
    """
    macd_ind = ta_lib.trend.MACD(
        df['close'],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL,
    )
    macd_line = macd_ind.macd()
    signal_line = macd_ind.macd_signal()

    # 이전 봉: macd <= signal, 현재 봉: macd > signal
    prev_below = macd_line.shift(1) <= signal_line.shift(1)
    curr_above = macd_line > signal_line

    return (prev_below & curr_above).fillna(False)
