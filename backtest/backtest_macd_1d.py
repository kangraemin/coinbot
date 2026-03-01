"""일봉 MACD 추세추종 백테스트 — 2022~2025, BTC/ETH/XRP.

롱: MACD 골든크로스 + ADX(14) > adx_thresh
숏: MACD 데드크로스 + ADX(14) > adx_thresh
청산: 반대 신호 역전 OR ATR×sl_mult SL (sl_mult=0이면 신호까지 보유)
"""

import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0
START_DATE  = "2022-01-01"
END_DATE    = "2025-12-31"

MACD_FAST  = [8, 12]
MACD_SLOW  = [21, 26]
MACD_SIG   = [9]
ADX_THRESH = [20, 25]
SL_MULTS   = [2.0, 3.0, 0]   # 0 = SL 없음
LEVERAGES  = [2, 3, 5]
POS_RATIOS = [0.3, 0.5, 1.0]

COINS  = ["btc", "eth", "xrp"]
BH_REF = {"btc": 89.4, "eth": -19.4, "xrp": 121.6}


# ── 데이터 준비 ───────────────────────────────────────

def load_range(coin: str) -> pd.DataFrame:
    frames = []
    for y in range(2021, 2026):
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
    return df.sort_values("timestamp").set_index("timestamp")


def resample_1d(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last"),
        volume=("volume","sum"),
    ).dropna().reset_index()


def add_indicators(df: pd.DataFrame, fast: int, slow: int, sig: int) -> pd.DataFrame:
    c  = df["close"].astype(float)
    h  = df["high"].astype(float)
    lo = df["low"].astype(float)

    # MACD
    ema_fast  = c.ewm(span=fast, adjust=False).mean()
    ema_slow  = c.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=sig, adjust=False).mean()

    # ATR14 (Wilder)
    tr  = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    # ADX14 (Wilder)
    h_diff  = h.diff()
    l_diff  = -lo.diff()
    dm_plus  = h_diff.where((h_diff > l_diff) & (h_diff > 0), 0.0)
    dm_minus = l_diff.where((l_diff > h_diff) & (l_diff > 0), 0.0)
    atr_adx  = tr.ewm(alpha=1/14, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=1/14, adjust=False).mean() / atr_adx.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(alpha=1/14, adjust=False).mean() / atr_adx.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx      = dx.ewm(alpha=1/14, adjust=False).mean()

    df = df.copy()
    df["macd"]    = macd_line
    df["signal"]  = sig_line
    df["atr"]     = atr
    df["adx"]     = adx
    return df


# ── 백테스트 코어 ─────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    adx_thresh: float,
    sl_mult: float,
    leverage: int,
    pos_ratio: float,
    initial_balance: float = INITIAL_BAL,
) -> dict:
    closes  = df["close"].to_numpy(float)
    highs   = df["high"].to_numpy(float)
    lows    = df["low"].to_numpy(float)
    macd    = df["macd"].to_numpy(float)
    sig     = df["signal"].to_numpy(float)
    atr     = df["atr"].to_numpy(float)
    adx     = df["adx"].to_numpy(float)
    n = len(closes)

    balance = initial_balance
    peak    = initial_balance
    mdd = 0.0

    in_pos    = False
    direction = 0
    ep = sl = pos_amt = 0.0
    entry_bar = 0

    long_trades = long_wins = 0
    short_trades = short_wins = 0

    for i in range(1, n):
        if any(np.isnan(x[i]) for x in [macd, sig, atr, adx]):
            continue
        if np.isnan(macd[i-1]) or np.isnan(sig[i-1]):
            continue

        long_cross  = (macd[i-1] <= sig[i-1]) and (macd[i] > sig[i]) and (adx[i] >= adx_thresh)
        short_cross = (macd[i-1] >= sig[i-1]) and (macd[i] < sig[i]) and (adx[i] >= adx_thresh)

        if in_pos:
            exit_p = None

            if direction == 1:
                if sl_mult > 0 and lows[i] <= sl:
                    exit_p = sl
                elif short_cross:
                    exit_p = closes[i]
            else:
                if sl_mult > 0 and highs[i] >= sl:
                    exit_p = sl
                elif long_cross:
                    exit_p = closes[i]

            if exit_p is not None:
                if direction == 1:
                    pnl = (exit_p - ep) * pos_amt
                    long_trades += 1
                    if exit_p > ep:
                        long_wins += 1
                else:
                    pnl = (ep - exit_p) * pos_amt
                    short_trades += 1
                    if exit_p < ep:
                        short_wins += 1
                balance += pnl - exit_p * pos_amt * TAKER_FEE
                in_pos    = False
                direction = 0

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > mdd:
                mdd = dd

        if not in_pos and balance > 0:
            notional = balance * pos_ratio * leverage
            if long_cross:
                ep       = closes[i]
                pos_amt  = notional / ep
                balance -= notional * TAKER_FEE
                sl       = ep - atr[i] * sl_mult if sl_mult > 0 else 0.0
                in_pos   = True
                direction = 1
            elif short_cross:
                ep       = closes[i]
                pos_amt  = notional / ep
                balance -= notional * TAKER_FEE
                sl       = ep + atr[i] * sl_mult if sl_mult > 0 else 0.0
                in_pos   = True
                direction = -1

    # 미청산 포지션 강제 청산
    if in_pos and n > 0:
        exit_p = closes[-1]
        if direction == 1:
            pnl = (exit_p - ep) * pos_amt
            long_trades += 1
            if exit_p > ep:
                long_wins += 1
        else:
            pnl = (ep - exit_p) * pos_amt
            short_trades += 1
            if exit_p < ep:
                short_wins += 1
        balance += pnl - exit_p * pos_amt * TAKER_FEE
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100
        if dd > mdd:
            mdd = dd

    ret = (balance - initial_balance) / initial_balance * 100 if balance > 0 else -100.0
    total = long_trades + short_trades
    lwr   = long_wins  / long_trades  * 100 if long_trades  > 0 else 0.0
    swr   = short_wins / short_trades * 100 if short_trades > 0 else 0.0
    wr    = (long_wins + short_wins) / total * 100 if total > 0 else 0.0
    calmar = ret / mdd if mdd > 0 else 0.0

    return {
        "adx_thresh":   adx_thresh,
        "sl_mult":      sl_mult,
        "leverage":     leverage,
        "pos_ratio":    pos_ratio,
        "trades":       total,
        "trades_per_yr": round(total / 4, 1),
        "long_trades":  long_trades,
        "short_trades": short_trades,
        "win_rate":     round(wr, 1),
        "long_win_rate":  round(lwr, 1),
        "short_win_rate": round(swr, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(mdd, 2),
        "calmar":       round(calmar, 2),
    }


# ── 워커 ──────────────────────────────────────────────

def worker(coin: str) -> str:
    print(f"[{coin.upper()} 1D MACD] 시작...", flush=True)

    df_1m = load_range(coin)
    df_1d = resample_1d(df_1m)

    s = pd.Timestamp(START_DATE, tz="UTC")
    e = pd.Timestamp(END_DATE + " 23:59:59", tz="UTC")

    results = []
    for fast, slow, sig_p in itertools.product(MACD_FAST, MACD_SLOW, MACD_SIG):
        df_ind  = add_indicators(df_1d, fast, slow, sig_p)
        df_eval = df_ind[(df_ind["timestamp"] >= s) & (df_ind["timestamp"] <= e)].reset_index(drop=True)

        for adx_t, sl_m, lev, pos in itertools.product(ADX_THRESH, SL_MULTS, LEVERAGES, POS_RATIOS):
            r = run_backtest(df_eval, adx_t, sl_m, lev, pos)
            r["macd_fast"] = fast
            r["macd_slow"] = slow
            r["macd_sig"]  = sig_p
            results.append(r)

    df_res = pd.DataFrame(results).sort_values("return_pct", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"macd_1d_{coin}.csv")
    df_res.to_csv(out_path, index=False)

    bh = BH_REF[coin]
    df10 = df_res[df_res["trades"] >= 10]
    best = df10.iloc[0] if len(df10) > 0 else df_res.iloc[0]
    beat = "✓ B&H 초과" if best["return_pct"] > bh else "✗ B&H 미달"

    print(
        f"[{coin.upper()} 1D MACD] 완료 — {len(results)}조합 | "
        f"Best(≥10거래): 수익={best['return_pct']:+.1f}%  MDD={best['max_drawdown']:.1f}%  "
        f"Calmar={best['calmar']:.2f}  거래={best['trades']}건({best['trades_per_yr']}/년)  "
        f"L={best['long_trades']}건({best['long_win_rate']:.0f}%) "
        f"S={best['short_trades']}건({best['short_win_rate']:.0f}%) | "
        f"MACD({best['macd_fast']},{best['macd_slow']},{best['macd_sig']}) "
        f"ADX>{best['adx_thresh']} sl×{best['sl_mult']} "
        f"lev={best['leverage']}x pos={best['pos_ratio']*100:.0f}% | "
        f"B&H={bh:+.1f}% → {beat}",
        flush=True,
    )
    return out_path


def main():
    total = (len(MACD_FAST) * len(MACD_SLOW) * len(MACD_SIG)
             * len(ADX_THRESH) * len(SL_MULTS) * len(LEVERAGES) * len(POS_RATIOS))
    print("일봉 MACD 추세추종 — BTC/ETH/XRP, 2022~2025")
    print(f"조합: {total}개 × {len(COINS)}코인 = {total*len(COINS)}개\n")

    with ProcessPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {executor.submit(worker, c): c for c in COINS}
        for fut in as_completed(futures):
            coin = futures[fut]
            try:
                print(f"  저장: {fut.result()}")
            except Exception as exc:
                print(f"  [{coin}] 오류: {exc}")


if __name__ == "__main__":
    main()
