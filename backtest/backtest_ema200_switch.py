"""일봉 EMA 스위처 — EMA 위아래로 포지션 전환.

규칙:
  long_only  : EMA 위 → 롱 유지, EMA 아래 → 현금
  long_short : EMA 위 → 롱, EMA 아래 → 숏

코인: BTC, ETH, XRP | 기간: 2022~2025
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

EMA_PERIODS = [150, 200, 250]
MODES       = ["long_only", "long_short"]
LEVERAGES   = [1, 2, 3, 5]
POS_RATIOS  = [0.5, 1.0]

COINS  = ["btc", "eth", "xrp"]
BH_REF = {"btc": 89.4, "eth": -19.4, "xrp": 121.6}


# ── 데이터 준비 ───────────────────────────────────────

def load_range(coin: str) -> pd.DataFrame:
    frames = []
    for y in range(2020, 2026):   # 충분한 EMA 웜업
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


# ── 백테스트 코어 ─────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    ema_period: int,
    mode: str,
    leverage: int,
    pos_ratio: float,
    initial_balance: float = INITIAL_BAL,
) -> dict:
    closes = df["close"].to_numpy(float)
    highs  = df["high"].to_numpy(float)
    lows   = df["low"].to_numpy(float)

    # EMA 계산
    ema_ser = pd.Series(closes).ewm(span=ema_period, adjust=False).mean().to_numpy(float)
    n = len(closes)

    balance = initial_balance
    peak    = initial_balance
    mdd = 0.0

    in_pos    = False
    direction = 0
    ep = pos_amt = 0.0

    long_trades = long_wins = 0
    short_trades = short_wins = 0

    for i in range(1, n):
        if np.isnan(ema_ser[i]) or np.isnan(ema_ser[i-1]):
            continue

        above_now  = closes[i]   > ema_ser[i]
        above_prev = closes[i-1] > ema_ser[i-1]

        cross_up   = (not above_prev) and above_now    # EMA 위로 교차
        cross_down = above_prev and (not above_now)    # EMA 아래로 교차

        if in_pos:
            exit_p = None

            if direction == 1 and cross_down:
                exit_p = closes[i]
            elif direction == -1 and cross_up:
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
            if cross_up:
                ep      = closes[i]
                pos_amt = notional / ep
                balance -= notional * TAKER_FEE
                in_pos   = True
                direction = 1
            elif cross_down and mode == "long_short":
                ep      = closes[i]
                pos_amt = notional / ep
                balance -= notional * TAKER_FEE
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
    wr    = (long_wins + short_wins) / total * 100 if total > 0 else 0.0
    calmar = ret / mdd if mdd > 0 else 0.0

    return {
        "ema_period":   ema_period,
        "mode":         mode,
        "leverage":     leverage,
        "pos_ratio":    pos_ratio,
        "trades":       total,
        "long_trades":  long_trades,
        "short_trades": short_trades,
        "win_rate":     round(wr, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(mdd, 2),
        "calmar":       round(calmar, 2),
    }


# ── 워커 & 연도별 ─────────────────────────────────────

def worker(coin: str) -> str:
    print(f"[{coin.upper()} EMA Switch] 시작...", flush=True)

    df_1m = load_range(coin)
    df_1d = resample_1d(df_1m)

    s = pd.Timestamp(START_DATE, tz="UTC")
    e = pd.Timestamp(END_DATE + " 23:59:59", tz="UTC")

    combos = list(itertools.product(EMA_PERIODS, MODES, LEVERAGES, POS_RATIOS))
    results = []
    for ema_p, mode, lev, pos in combos:
        df_eval = df_1d[(df_1d["timestamp"] >= s) & (df_1d["timestamp"] <= e)].reset_index(drop=True)
        r = run_backtest(df_eval, ema_p, mode, lev, pos)
        results.append(r)

    df_res = pd.DataFrame(results).sort_values("return_pct", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"ema_switch_{coin}.csv")
    df_res.to_csv(out_path, index=False)

    bh   = BH_REF[coin]
    best = df_res.iloc[0]
    beat = "✓ B&H 초과" if best["return_pct"] > bh else "✗ B&H 미달"

    print(
        f"[{coin.upper()} EMA Switch] 완료 — {len(results)}조합 | "
        f"Best: 수익={best['return_pct']:+.1f}%  MDD={best['max_drawdown']:.1f}%  "
        f"Calmar={best['calmar']:.2f}  거래={best['trades']}건 | "
        f"EMA{best['ema_period']} {best['mode']} lev={best['leverage']}x pos={best['pos_ratio']*100:.0f}% | "
        f"B&H={bh:+.1f}% → {beat}",
        flush=True,
    )
    return out_path


def run_yearly(coin: str):
    """Best 파라미터로 연도별 성과 출력."""
    df_1m = load_range(coin)
    df_1d = resample_1d(df_1m)

    csv = os.path.join(RESULTS_DIR, f"ema_switch_{coin}.csv")
    df_res = pd.read_csv(csv)
    best = df_res.iloc[0]
    ema_p = int(best["ema_period"])
    mode  = best["mode"]
    lev   = int(best["leverage"])
    pos   = float(best["pos_ratio"])

    rows = []
    print(f"\n[{coin.upper()}] Best: EMA{ema_p} {mode} {lev}x {pos*100:.0f}%")
    print(f"  {'연도':6} {'수익률':>9} {'MDD':>7} {'Calmar':>8} {'롱건':>5} {'숏건':>5}")
    print(f"  {'-'*45}")

    for year in range(2022, 2027):
        s = pd.Timestamp(f"{year}-01-01", tz="UTC")
        e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
        df_y = df_1d[(df_1d["timestamp"] >= s) & (df_1d["timestamp"] <= e)].reset_index(drop=True)
        if len(df_y) < 5:
            continue
        r = run_backtest(df_y, ema_p, mode, lev, pos)
        suffix = " (YTD)" if year == 2026 else ""
        print(
            f"  {year}{suffix:6} {r['return_pct']:>+8.1f}% "
            f"{r['max_drawdown']:>6.1f}%  {r['calmar']:>7.2f} "
            f"{r['long_trades']:>5}건 {r['short_trades']:>5}건"
        )
        rows.append({"coin": coin.upper(), "year": year,
                     "ema_period": ema_p, "mode": mode, "leverage": lev, "pos_ratio": pos,
                     **{k: r[k] for k in ["return_pct","max_drawdown","calmar","trades","win_rate","long_trades","short_trades"]}})
    return rows


def main():
    total = len(list(itertools.product(EMA_PERIODS, MODES, LEVERAGES, POS_RATIOS)))
    print("일봉 EMA 스위처 — BTC/ETH/XRP, 2022~2025")
    print(f"조합: {total}개 × {len(COINS)}코인 = {total*len(COINS)}개\n")

    with ProcessPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {executor.submit(worker, c): c for c in COINS}
        for fut in as_completed(futures):
            coin = futures[fut]
            try:
                print(f"  저장: {fut.result()}")
            except Exception as exc:
                print(f"  [{coin}] 오류: {exc}")

    print("\n── 연도별 견고성 ──")
    all_rows = []
    for coin in COINS:
        all_rows.extend(run_yearly(coin))

    pd.DataFrame(all_rows).to_csv(
        os.path.join(RESULTS_DIR, "ema_switch_robustness.csv"), index=False
    )
    print(f"\n저장: {os.path.join(RESULTS_DIR, 'ema_switch_robustness.csv')}")


if __name__ == "__main__":
    main()
