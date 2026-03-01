"""BB+RSI 과매도 반등 1h v2 — XRP/SOL/ETH, 2023~2025, 볼륨 필터 없음.

4h v2 대비:
  - 타임프레임: 4h → 1h
  - 코인: XRP, SOL, ETH (BTC 제외)
  - 기간: 2023-01-01 ~ 2025-12-31
  - 볼륨 필터 없음 (v2와 동일)
"""

import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE_RATE = 0.0005

START_YEAR = 2023
END_YEAR   = 2025

RSI_THRESHOLDS = [30, 35, 40, 45]
SL_ATR_MULTS   = [1.0, 1.5, 2.0]
TP_MODES       = ["bb_mid", "atr_2x", "atr_3x"]
LEVERAGES      = [3, 5, 7]
POS_RATIOS     = [0.1, 0.2, 0.3]
USE_EMA200     = [True, False]

COINS = ["xrp", "sol", "eth"]


def load_range(coin: str) -> pd.DataFrame:
    """2022 마지막 60일 + 2023~2025 전체 로드 (EMA200 웜업용)."""
    frames = []
    for y in range(2022, END_YEAR + 1):
        path = os.path.join(DATA_DIR, f"{coin}_1m_{y}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"{coin} 데이터 없음")
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    resampled = df.resample("1h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return resampled.reset_index()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std(ddof=1)
    bb_lower = bb_mid - 2 * bb_std

    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr    = tr.ewm(com=13, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    df = df.copy()
    df["bb_mid"]   = bb_mid
    df["bb_lower"] = bb_lower
    df["rsi"]      = rsi
    df["atr"]      = atr
    df["ema200"]   = ema200
    return df


def run_backtest(
    df: pd.DataFrame,
    rsi_thresh: float,
    sl_atr_mult: float,
    tp_mode: str,
    leverage: int,
    pos_ratio: float,
    use_ema200: bool,
    timeout_bars: int = 48,
    initial_balance: float = 1000.0,
) -> dict:
    closes   = df["close"].to_numpy(dtype=float)
    highs    = df["high"].to_numpy(dtype=float)
    lows     = df["low"].to_numpy(dtype=float)
    bb_mid   = df["bb_mid"].to_numpy(dtype=float)
    bb_lower = df["bb_lower"].to_numpy(dtype=float)
    rsi_arr  = df["rsi"].to_numpy(dtype=float)
    atr_arr  = df["atr"].to_numpy(dtype=float)
    ema200   = df["ema200"].to_numpy(dtype=float)
    n = len(closes)

    balance = initial_balance
    peak    = initial_balance
    mdd = 0.0
    trades = wins = 0

    in_pos = False
    entry_price = sl_price = tp_price = 0.0
    pos_amt = 0.0
    entry_bar = 0

    for i in range(1, n):
        if np.isnan(bb_lower[i]) or np.isnan(rsi_arr[i]) or \
                np.isnan(atr_arr[i]) or np.isnan(ema200[i]):
            continue

        if not in_pos:
            if balance <= 0:
                break
            if use_ema200 and closes[i] <= ema200[i]:
                continue
            if closes[i] < bb_lower[i] and rsi_arr[i] < rsi_thresh:
                entry_price = closes[i]
                notional  = balance * pos_ratio * leverage
                pos_amt   = notional / entry_price
                balance  -= notional * TAKER_FEE_RATE
                atr_val   = atr_arr[i]
                sl_price  = entry_price - atr_val * sl_atr_mult
                if tp_mode == "atr_2x":
                    tp_price = entry_price + atr_val * 2.0
                elif tp_mode == "atr_3x":
                    tp_price = entry_price + atr_val * 3.0
                else:
                    tp_price = 0.0
                entry_bar = i
                in_pos = True
        else:
            exit_p = None
            tp_check = bb_mid[i] if tp_mode == "bb_mid" else tp_price
            if lows[i] <= sl_price:
                exit_p = sl_price
            elif tp_check > entry_price and highs[i] >= tp_check:
                exit_p = tp_check
                wins += 1
            elif i - entry_bar >= timeout_bars:
                exit_p = closes[i]
                if exit_p > entry_price:
                    wins += 1
            if exit_p is not None:
                pnl     = (exit_p - entry_price) * pos_amt
                balance += pnl - exit_p * pos_amt * TAKER_FEE_RATE
                trades  += 1
                in_pos   = False
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > mdd:
                mdd = dd

    if balance <= 0:
        ret = -100.0
    else:
        ret = (balance - initial_balance) / initial_balance * 100

    win_rate = wins / trades * 100 if trades > 0 else 0.0
    calmar   = ret / mdd if mdd > 0 else 0.0

    return {
        "rsi_thresh":   rsi_thresh,
        "sl_atr_mult":  sl_atr_mult,
        "tp_mode":      tp_mode,
        "leverage":     leverage,
        "pos_ratio":    pos_ratio,
        "use_ema200":   use_ema200,
        "trades":       trades,
        "trades_per_yr": round(trades / 3, 1),
        "win_rate":     round(win_rate, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(mdd, 2),
        "calmar":       round(calmar, 2),
        "timeframe":    "1h",
    }


def worker(coin: str) -> str:
    print(f"[{coin.upper()} 1h] 시작...", flush=True)

    df_1m = load_range(coin)
    df_1h = resample_1h(df_1m)
    df_1h = compute_indicators(df_1h)

    # 평가 구간: 2023~2025
    s = pd.Timestamp(f"{START_YEAR}-01-01", tz="UTC")
    e = pd.Timestamp(f"{END_YEAR}-12-31 23:59:59", tz="UTC")
    df_eval = df_1h[(df_1h["timestamp"] >= s) & (df_1h["timestamp"] <= e)].reset_index(drop=True)

    combos = list(itertools.product(
        RSI_THRESHOLDS, SL_ATR_MULTS, TP_MODES, LEVERAGES, POS_RATIOS, USE_EMA200,
    ))

    results = []
    for rsi_thresh, sl_atr, tp_mode, leverage, pos_ratio, ema200_flag in combos:
        r = run_backtest(df_eval, rsi_thresh, sl_atr, tp_mode, leverage, pos_ratio, ema200_flag)
        results.append(r)

    df_res = pd.DataFrame(results).sort_values("return_pct", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"bb_rsi_1h_v2_{coin}.csv")
    df_res.to_csv(out_path, index=False)

    df30 = df_res[df_res["trades"] >= 30]
    best = df30.iloc[0] if len(df30) > 0 else df_res.iloc[0]
    print(
        f"[{coin.upper()} 1h] 완료 — {len(results)}조합 | "
        f"Best(≥30거래): 수익={best['return_pct']:+.1f}%  MDD={best['max_drawdown']:.1f}%  "
        f"승률={best['win_rate']:.1f}%  거래={best['trades']}건({best['trades_per_yr']}/년)  "
        f"Calmar={best['calmar']:.2f} | "
        f"RSI<{best['rsi_thresh']}  sl×{best['sl_atr_mult']}  tp={best['tp_mode']}  "
        f"lev={best['leverage']}x  pos={best['pos_ratio']*100:.0f}%  "
        f"EMA200={'ON' if best['use_ema200'] else 'OFF'}",
        flush=True
    )
    return out_path


def main():
    total = len(list(itertools.product(
        RSI_THRESHOLDS, SL_ATR_MULTS, TP_MODES, LEVERAGES, POS_RATIOS, USE_EMA200,
    )))
    print("BB+RSI 1h v2 — XRP/SOL/ETH, 2023~2025, 볼륨필터 없음")
    print(f"타임프레임: 1h | 코인: {[c.upper() for c in COINS]}")
    print(f"조합: {total}개 × {len(COINS)}코인 = {total*len(COINS)}개\n")

    with ProcessPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {executor.submit(worker, c): c for c in COINS}
        for fut in as_completed(futures):
            coin = futures[fut]
            try:
                print(f"  저장: {fut.result()}")
            except Exception as e:
                print(f"  [{coin}] 오류: {e}")


if __name__ == "__main__":
    main()
