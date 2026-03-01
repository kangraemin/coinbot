"""BB+RSI v2 견고성 검증 (Robustness Test) — 연도별 성과 분리.

v2 Best 파라미터를 연도별로 나눠 성과 일관성 확인.
과적합이라면 특정 연도에만 수익이 집중됨.

검증 코인: ETH, SOL, XRP
연도: 2023, 2024, 2025, 2026
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE_RATE = 0.0005

# ── Best 파라미터 고정 (v2 그리드 서치 결과) ──────────
BEST_PARAMS = {
    "eth": dict(rsi_thresh=45, sl_atr_mult=2.0, tp_mode="atr_2x", leverage=7, pos_ratio=0.3, use_ema200=True),
    "sol": dict(rsi_thresh=45, sl_atr_mult=1.0, tp_mode="atr_2x", leverage=7, pos_ratio=0.3, use_ema200=True),
    "xrp": dict(rsi_thresh=30, sl_atr_mult=1.5, tp_mode="bb_mid", leverage=7, pos_ratio=0.3, use_ema200=False),
}

COINS = ["eth", "sol", "xrp"]
YEARS = [2023, 2024, 2025, 2026]


def load_year(coin: str, year: int) -> pd.DataFrame:
    """특정 연도 1분봉 로드."""
    path = os.path.join(DATA_DIR, f"{coin}_1m_{year}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    resampled = df.resample("4h").agg({
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
    atr = tr.ewm(com=13, adjust=False).mean()

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
    peak_balance = initial_balance
    max_drawdown = 0.0
    trades = wins = 0

    in_position = False
    entry_price = sl_price = tp_price = 0.0
    position_amount = 0.0
    entry_bar = 0

    # 연도별 실행 시 웜업 없이 전체 구간 사용 (단, NaN 체크로 자연 웜업)
    for i in range(1, n):
        if (np.isnan(bb_lower[i]) or np.isnan(rsi_arr[i]) or
                np.isnan(atr_arr[i]) or np.isnan(ema200[i])):
            continue

        if not in_position:
            if balance <= 0:
                break

            if use_ema200 and closes[i] <= ema200[i]:
                continue

            if closes[i] < bb_lower[i] and rsi_arr[i] < rsi_thresh:
                entry_price = closes[i]
                margin = balance * pos_ratio
                notional = margin * leverage
                position_amount = notional / entry_price
                fee = notional * TAKER_FEE_RATE
                balance -= fee

                atr_val = atr_arr[i]
                sl_price = entry_price - atr_val * sl_atr_mult
                if tp_mode == "atr_2x":
                    tp_price = entry_price + atr_val * 2.0
                elif tp_mode == "atr_3x":
                    tp_price = entry_price + atr_val * 3.0
                else:
                    tp_price = 0.0
                entry_bar = i
                in_position = True
        else:
            exit_price = None
            won = False

            tp_check = bb_mid[i] if tp_mode == "bb_mid" else tp_price

            if lows[i] <= sl_price:
                exit_price = sl_price
            elif tp_check > entry_price and highs[i] >= tp_check:
                exit_price = tp_check
                won = True
            elif i - entry_bar >= timeout_bars:
                exit_price = closes[i]
                won = exit_price > entry_price

            if exit_price is not None:
                pnl = (exit_price - entry_price) * position_amount
                fee = exit_price * position_amount * TAKER_FEE_RATE
                balance += pnl - fee
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
        "trades":       trades,
        "win_rate":     round(win_rate, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(max_drawdown, 2),
        "calmar":       round(calmar, 2),
    }


def worker(args: tuple) -> dict:
    coin, year = args
    params = BEST_PARAMS[coin]

    df_1m = load_year(coin, year)
    if df_1m.empty:
        return {"coin": coin.upper(), "year": year, "trades": 0,
                "win_rate": 0, "return_pct": 0, "max_drawdown": 0, "calmar": 0,
                "note": "데이터 없음"}

    # 연도별 독립 실행이지만 EMA200 웜업을 위해 전년도 마지막 데이터 일부 포함
    # 전년도 60일치 앞에 붙여서 지표 안정화
    prev_year_path = os.path.join(DATA_DIR, f"{coin}_1m_{year-1}.parquet")
    if os.path.exists(prev_year_path):
        df_prev = pd.read_parquet(prev_year_path)
        df_prev.columns = [c.lower() for c in df_prev.columns]
        df_prev["timestamp"] = pd.to_datetime(df_prev["timestamp"], utc=True)
        df_prev = df_prev.sort_values("timestamp").set_index("timestamp")
        # 마지막 60일 (4h봉 기준 약 360봉 = EMA200 웜업 충분)
        cutoff = df_prev.index[-1] - pd.Timedelta(days=60)
        df_prev = df_prev[df_prev.index >= cutoff]
        df_combined = pd.concat([df_prev, df_1m])
    else:
        df_combined = df_1m

    df_4h = resample_4h(df_combined)
    df_4h = compute_indicators(df_4h)

    # 실제 평가 구간: 해당 연도만
    year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    year_end   = pd.Timestamp(f"{year+1}-01-01", tz="UTC")
    df_eval = df_4h[(df_4h["timestamp"] >= year_start) & (df_4h["timestamp"] < year_end)].reset_index(drop=True)

    if len(df_eval) < 20:
        return {"coin": coin.upper(), "year": year, "trades": 0,
                "win_rate": 0, "return_pct": 0, "max_drawdown": 0, "calmar": 0,
                "note": "데이터 부족"}

    result = run_backtest(df_eval, **params)
    result["coin"] = coin.upper()
    result["year"] = year
    result["note"] = ""
    return result


def main():
    print("BB+RSI v2 견고성 검증 — 연도별 성과 분리")
    print(f"코인: {[c.upper() for c in COINS]} | 연도: {YEARS}\n")

    tasks = [(coin, year) for coin in COINS for year in YEARS]

    results = []
    with ProcessPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            coin, year = futures[fut]
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                print(f"  [{coin.upper()} {year}] 오류: {e}")
                results.append({"coin": coin.upper(), "year": year,
                                 "trades": 0, "win_rate": 0, "return_pct": 0,
                                 "max_drawdown": 0, "calmar": 0, "note": str(e)})

    df_res = pd.DataFrame(results).sort_values(["coin", "year"]).reset_index(drop=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "bb_rsi_robustness.csv")
    df_res.to_csv(out_path, index=False)

    # 결과 출력
    print(f"\n{'코인':6} {'연도':6} {'수익률':>9} {'MDD':>8} {'승률':>7} {'거래수':>7} {'Calmar':>8}")
    print("-" * 58)
    for _, row in df_res.iterrows():
        sign = "+" if row["return_pct"] >= 0 else ""
        print(f"{row['coin']:6} {row['year']:6} "
              f"{sign}{row['return_pct']:>8.2f}% "
              f"{row['max_drawdown']:>7.1f}% "
              f"{row['win_rate']:>6.1f}% "
              f"{row['trades']:>7}건 "
              f"{row['calmar']:>8.2f}")
        if row.get("note"):
            print(f"       ※ {row['note']}")

    # 코인별 요약
    print("\n== 코인별 연도 일관성 요약 ==")
    for coin in [c.upper() for c in COINS]:
        sub = df_res[df_res["coin"] == coin]
        pos = (sub["return_pct"] > 0).sum()
        avg = sub["return_pct"].mean()
        print(f"{coin}: 플러스 연도 {pos}/{len(sub)} | 평균 수익률 {avg:+.1f}%")

    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
