"""백테스팅 엔진 — 신호 기반 진입/청산 시뮬레이션."""

import numpy as np
import pandas as pd

from analysis.config import TP_PCT, SL_PCT, FEE_RATE, SLIPPAGE, TIMEOUT_BARS


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    leverage: int = 5,
    fee_rate: float = FEE_RATE,
    slippage: float = SLIPPAGE,
    timeout_bars: int = TIMEOUT_BARS,
) -> tuple[pd.DataFrame, dict]:
    """신호 기반 롱 포지션 백테스팅.

    진입: 신호 발생 다음봉 시가 (slippage 적용)
    청산: TP / SL 히트 또는 타임아웃
    수수료: 진입·청산 각 fee_rate (편도)

    Args:
        df: OHLCV DataFrame (timestamp 또는 index 기반)
        signals: bool Series — True인 봉 다음봉에 진입
        tp_pct: 익절 비율 (예: 0.02 = 2%)
        sl_pct: 손절 비율 (예: 0.01 = 1%)
        leverage: 레버리지
        fee_rate: 수수료율 (편도)
        slippage: 슬리피지 (진입가 가산)
        timeout_bars: 타임아웃 봉 수

    Returns:
        (trades_df, summary_dict)
    """
    opens  = df['open'].to_numpy(dtype=float)
    highs  = df['high'].to_numpy(dtype=float)
    lows   = df['low'].to_numpy(dtype=float)
    closes = df['close'].to_numpy(dtype=float)

    # timestamp 컬럼 또는 인덱스 처리
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].to_numpy()
    else:
        timestamps = df.index.to_numpy()

    sig = signals.to_numpy(dtype=bool)
    n = len(df)

    trades = []
    i = 0
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue

        # 다음봉 시가에 진입 (slippage 가산)
        entry_idx = i + 1
        entry_price = opens[entry_idx] * (1 + slippage)
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

        exit_price = None
        exit_reason = None
        exit_idx = entry_idx

        for j in range(entry_idx, min(entry_idx + timeout_bars + 1, n)):
            # SL 먼저 (같은 봉에서 SL/TP 동시 도달 시 SL 우선)
            if lows[j] <= sl_price:
                exit_price = sl_price
                exit_reason = 'sl'
                exit_idx = j
                break
            if highs[j] >= tp_price:
                exit_price = tp_price
                exit_reason = 'tp'
                exit_idx = j
                break
            if j == min(entry_idx + timeout_bars, n - 1):
                exit_price = closes[j]
                exit_reason = 'timeout'
                exit_idx = j
                break

        if exit_price is None:
            # 데이터 끝
            exit_price = closes[min(entry_idx + timeout_bars, n - 1)]
            exit_reason = 'timeout'
            exit_idx = min(entry_idx + timeout_bars, n - 1)

        pnl_pct = (exit_price - entry_price) / entry_price
        fee_total = fee_rate * 2  # 진입 + 청산
        pnl_pct_net = pnl_pct - fee_total
        pnl_leveraged = pnl_pct_net * leverage

        trades.append({
            'entry_time':    timestamps[entry_idx],
            'exit_time':     timestamps[exit_idx],
            'entry_price':   entry_price,
            'exit_price':    exit_price,
            'pnl_pct':       round(pnl_pct_net, 6),
            'pnl_leveraged': round(pnl_leveraged, 6),
            'exit_reason':   exit_reason,
        })

        i = exit_idx + 1

    if not trades:
        empty_df = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'pnl_pct', 'pnl_leveraged', 'exit_reason',
        ])
        return empty_df, _empty_summary()

    trades_df = pd.DataFrame(trades)
    summary = _compute_summary(trades_df)
    return trades_df, summary


def _empty_summary() -> dict:
    return {
        'total_trades': 0,
        'win_rate':     0.0,
        'total_return': 0.0,
        'avg_return':   0.0,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
    }


def _compute_summary(trades_df: pd.DataFrame) -> dict:
    """trades DataFrame에서 요약 통계 계산."""
    total = len(trades_df)
    wins = (trades_df['pnl_leveraged'] > 0).sum()
    win_rate = wins / total if total > 0 else 0.0

    pnl = trades_df['pnl_leveraged'].to_numpy()
    total_return = float(np.sum(pnl))
    avg_return = float(np.mean(pnl))

    # 최대 낙폭 (누적 PnL 기준)
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe ratio
    if pnl.std() > 0:
        sharpe = float(pnl.mean() / pnl.std() * np.sqrt(len(pnl)))
    else:
        sharpe = 0.0

    return {
        'total_trades': total,
        'win_rate':     round(win_rate, 4),
        'total_return': round(total_return, 4),
        'avg_return':   round(avg_return, 4),
        'max_drawdown': round(max_drawdown, 4),
        'sharpe_ratio': round(sharpe, 4),
    }
