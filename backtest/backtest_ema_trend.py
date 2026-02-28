"""EMA 크로스오버 추세 추종 전략 — 5m / 15m 그리드 서치.

전략:
  - fast EMA가 slow EMA를 골든크로스 → 롱 진입 (종가 기준)
  - TP: 진입가 × (1 + tp_pct%)
  - SL: 진입가 × (1 - sl_capital / leverage %)
  - 추가 청산: 데드크로스 발생 시 종가 청산

1분봉 데이터를 리샘플링해서 5m/15m 생성 (별도 다운로드 불필요).
"""

import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE_RATE = 0.0005

# ── 파라미터 그리드 ──────────────────────────────────
FAST_EMAS   = [5, 9, 12]
SLOW_EMAS   = [21, 26, 50]
TP_PCTS     = [2.0, 3.0, 5.0]
SL_CAPITAL  = [1.0, 2.0, 3.0]   # 자본 손실 %
LEVERAGES   = [3, 5, 7]
POS_RATIOS  = [0.1, 0.2, 0.3]
TIMEFRAMES  = ["5m", "15m"]

COINS = ["btc", "eth", "sol", "xrp"]


def load_1m(coin: str) -> pd.DataFrame:
    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.startswith(f"{coin}_1m_") and f.endswith(".parquet")
    ])
    if not files:
        raise FileNotFoundError(f"{coin} 1분봉 데이터 없음: {DATA_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """1분봉 → 지정 타임프레임으로 리샘플링."""
    rule = tf.replace("m", "min")  # "5m" → "5min"
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return resampled.reset_index()


def compute_ema(series: np.ndarray, period: int) -> np.ndarray:
    ema = np.empty_like(series)
    ema[:period - 1] = np.nan
    ema[period - 1] = series[:period].mean()
    k = 2.0 / (period + 1)
    for i in range(period, len(series)):
        ema[i] = series[i] * k + ema[i - 1] * (1 - k)
    return ema


def run_backtest(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    tp_pct: float,
    sl_capital_pct: float,
    leverage: int,
    pos_ratio: float,
    initial_balance: float = 1000.0,
) -> dict:
    closes = df["close"].to_numpy(dtype=float)
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    n = len(closes)

    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)

    sl_price_pct = sl_capital_pct / leverage

    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    trades = wins = 0
    total_pnl = 0.0

    in_position = False
    entry_price = tp_price = sl_price = position_amount = 0.0

    start = max(fast, slow)

    for i in range(start, n):
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            continue

        if not in_position:
            if balance <= 0:
                break
            # 골든크로스: 이전 봉에서 fast <= slow, 현재 봉에서 fast > slow
            if (not np.isnan(ema_fast[i - 1]) and
                    ema_fast[i - 1] <= ema_slow[i - 1] and
                    ema_fast[i] > ema_slow[i]):
                entry_price = closes[i]
                margin = balance * pos_ratio
                notional = margin * leverage
                position_amount = notional / entry_price
                fee = notional * TAKER_FEE_RATE
                balance -= fee

                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_price_pct / 100)
                in_position = True
        else:
            exit_price = None
            won = False

            # SL 먼저 (같은 봉 SL+TP 동시 도달 시 SL 우선)
            if lows[i] <= sl_price:
                exit_price = sl_price
            elif highs[i] >= tp_price:
                exit_price = tp_price
                won = True
            # 데드크로스: EMA 역전 → 종가 청산
            elif ema_fast[i] < ema_slow[i]:
                exit_price = closes[i]
                won = exit_price > entry_price

            if exit_price is not None:
                pnl = (exit_price - entry_price) * position_amount
                fee = exit_price * position_amount * TAKER_FEE_RATE
                balance += pnl - fee
                total_pnl += pnl - fee
                trades += 1
                if won:
                    wins += 1
                in_position = False

            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100
            if dd > max_drawdown:
                max_drawdown = dd

    if balance <= 0:
        ret = -100.0
    else:
        ret = (balance - initial_balance) / initial_balance * 100

    win_rate = wins / trades * 100 if trades > 0 else 0.0
    calmar = ret / max_drawdown if max_drawdown > 0 else 0.0

    return {
        "fast_ema":     fast,
        "slow_ema":     slow,
        "tp_pct":       tp_pct,
        "sl_capital":   sl_capital_pct,
        "sl_price_pct": round(sl_price_pct, 4),
        "leverage":     leverage,
        "pos_ratio":    pos_ratio,
        "trades":       trades,
        "win_rate":     round(win_rate, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(max_drawdown, 2),
        "calmar":       round(calmar, 2),
    }


def worker(args: tuple) -> str:
    coin, tf = args
    print(f"[{coin.upper()} {tf}] 시작...", flush=True)

    try:
        df_1m = load_1m(coin)
        df = resample(df_1m, tf)
    except Exception as e:
        print(f"[{coin.upper()} {tf}] 데이터 로드 실패: {e}", flush=True)
        return f"{coin}/{tf}: FAILED"

    combos = list(itertools.product(
        FAST_EMAS, SLOW_EMAS, TP_PCTS, SL_CAPITAL, LEVERAGES, POS_RATIOS
    ))
    # fast >= slow 조합 제외
    combos = [(f, s, tp, sl, lev, pos) for f, s, tp, sl, lev, pos in combos if f < s]

    results = []
    for fast, slow, tp_pct, sl_cap, leverage, pos_ratio in combos:
        r = run_backtest(df, fast, slow, tp_pct, sl_cap, leverage, pos_ratio)
        r["timeframe"] = tf
        results.append(r)

    df_res = pd.DataFrame(results).sort_values("return_pct", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"ema_trend_{coin}_{tf}.csv")
    df_res.to_csv(out_path, index=False)

    best = df_res.iloc[0]
    print(
        f"[{coin.upper()} {tf}] 완료 — {len(results)}조합 | "
        f"Best: EMA {best['fast_ema']}/{best['slow_ema']} tp={best['tp_pct']}% "
        f"sl_cap={best['sl_capital']}% lev={best['leverage']}x "
        f"pos={best['pos_ratio']*100:.0f}% → "
        f"수익률={best['return_pct']:+.1f}% MDD={best['max_drawdown']:.1f}% "
        f"승률={best['win_rate']:.1f}% 거래={best['trades']}건 Calmar={best['calmar']:.2f}",
        flush=True
    )
    return out_path


def main():
    tasks = list(itertools.product(COINS, TIMEFRAMES))
    total_combos = len([
        1 for f, s, *_ in itertools.product(FAST_EMAS, SLOW_EMAS, TP_PCTS, SL_CAPITAL, LEVERAGES, POS_RATIOS)
        if f < s
    ])
    print(f"EMA 추세 추종 그리드 서치")
    print(f"타임프레임: {TIMEFRAMES} | 코인: {[c.upper() for c in COINS]}")
    print(f"조합 수: {total_combos}개 × {len(tasks)}태스크 = {total_combos * len(tasks)}개 백테스트\n")

    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                result = fut.result()
                print(f"  저장: {result}")
            except Exception as e:
                print(f"  [{task}] 오류: {e}")


if __name__ == "__main__":
    main()
