"""신호 탐지 모듈 — 각 전략은 detect(df) -> pd.Series[bool] 반환."""

from analysis.signals import (
    rsi_oversold,
    macd_cross,
    ema_cross,
    bollinger_bounce,
    support_bounce,
    candle_pattern,
)

STRATEGIES = {
    'rsi_oversold':     rsi_oversold,
    'macd_cross':       macd_cross,
    'ema_cross':        ema_cross,
    'bollinger_bounce': bollinger_bounce,
    'support_bounce':   support_bounce,
    'candle_pattern':   candle_pattern,
}

__all__ = ['STRATEGIES']
