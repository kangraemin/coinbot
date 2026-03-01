"""특징 분석 모듈 — 진입 포인트 시장 상태 분석 및 복합 신호 비교."""

from typing import Any

import numpy as np
import pandas as pd
import ta as ta_lib

from analysis.signals import STRATEGIES
from analysis.backtest_engine import run_backtest
from analysis.config import TP_PCT, SL_PCT, FEE_RATE, SLIPPAGE, TIMEOUT_BARS


def analyze_entry_context(df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    """진입 포인트 발생 시점의 시장 상태 분석.

    각 신호 발생 봉의 RSI, EMA 위치, 볼린저밴드 위치, 볼륨 등을 추가.

    Args:
        df: OHLCV DataFrame
        signals: bool Series

    Returns:
        신호 발생 봉의 시장 상태 DataFrame
    """
    df_ctx = df.copy()

    # RSI
    df_ctx['rsi'] = ta_lib.momentum.RSIIndicator(df['close'], window=14).rsi()

    # EMA
    df_ctx['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df_ctx['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df_ctx['above_ema20'] = df['close'] > df_ctx['ema20']
    df_ctx['above_ema50'] = df['close'] > df_ctx['ema50']

    # 볼린저밴드
    bb = ta_lib.volatility.BollingerBands(df['close'], window=20, window_dev=2.0)
    df_ctx['bb_upper'] = bb.bollinger_hband()
    df_ctx['bb_lower'] = bb.bollinger_lband()
    df_ctx['bb_pct'] = bb.bollinger_pband()  # 0~1 위치

    # 볼륨 비율 (현재/MA20)
    df_ctx['vol_ma20'] = df['volume'].rolling(20).mean()
    df_ctx['vol_ratio'] = df['volume'] / df_ctx['vol_ma20']

    # 신호 발생 봉만 추출
    entry_rows = df_ctx[signals.values].copy()
    return entry_rows[[
        'rsi', 'above_ema20', 'above_ema50', 'bb_pct', 'vol_ratio'
    ]]


def compare_combined_signals(
    df: pd.DataFrame,
    leverage: int = 5,
) -> pd.DataFrame:
    """복합 신호 (2개 이상 동시 발생) 승률 비교.

    각 전략 신호와 복합 신호별 백테스팅 결과를 비교 DataFrame으로 반환.

    Args:
        df: OHLCV DataFrame (timestamp 컬럼)
        leverage: 레버리지

    Returns:
        전략/복합신호별 결과 DataFrame
    """
    # 각 전략 신호 계산
    signal_series: dict[str, pd.Series] = {}
    for name, module in STRATEGIES.items():
        try:
            signal_series[name] = module.detect(df)
        except Exception as e:
            print(f"  [경고] {name} 신호 계산 실패: {e}")
            signal_series[name] = pd.Series(False, index=df.index)

    rows = []

    # 단일 전략
    for name, sig in signal_series.items():
        _, summary = run_backtest(
            df, sig, tp_pct=TP_PCT, sl_pct=SL_PCT,
            leverage=leverage, fee_rate=FEE_RATE,
            slippage=SLIPPAGE, timeout_bars=TIMEOUT_BARS,
        )
        rows.append({
            'strategy': name,
            'signal_count': int(sig.sum()),
            **summary,
        })

    # 복합 신호 (2개 이상 동시 발생)
    names = list(signal_series.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            combined = signal_series[n1] & signal_series[n2]
            if combined.sum() == 0:
                continue
            _, summary = run_backtest(
                df, combined, tp_pct=TP_PCT, sl_pct=SL_PCT,
                leverage=leverage, fee_rate=FEE_RATE,
                slippage=SLIPPAGE, timeout_bars=TIMEOUT_BARS,
            )
            rows.append({
                'strategy': f'{n1}+{n2}',
                'signal_count': int(combined.sum()),
                **summary,
            })

    return pd.DataFrame(rows).sort_values('win_rate', ascending=False)


def compare_timeframes(
    tf_data: dict[str, pd.DataFrame],
    strategy_name: str,
    leverage: int = 5,
) -> pd.DataFrame:
    """타임프레임별 성과 비교.

    Args:
        tf_data: {'5m': df, '15m': df, ...} 형태의 dict
        strategy_name: STRATEGIES 키 중 하나
        leverage: 레버리지

    Returns:
        타임프레임별 결과 DataFrame
    """
    module = STRATEGIES[strategy_name]
    rows = []
    for tf, df in tf_data.items():
        try:
            sig = module.detect(df)
        except Exception as e:
            print(f"  [경고] {strategy_name}/{tf} 신호 실패: {e}")
            continue

        _, summary = run_backtest(
            df, sig, tp_pct=TP_PCT, sl_pct=SL_PCT,
            leverage=leverage, fee_rate=FEE_RATE,
            slippage=SLIPPAGE, timeout_bars=TIMEOUT_BARS,
        )
        rows.append({
            'timeframe': tf,
            'signal_count': int(sig.sum()),
            **summary,
        })

    return pd.DataFrame(rows)
