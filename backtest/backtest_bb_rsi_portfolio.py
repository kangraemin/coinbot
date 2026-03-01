"""XRP + SOL 포트폴리오 상관관계 검증.

목적:
  - XRP(50%)와 SOL(50%)을 동시 운영 시 손실 겹침 여부 확인
  - 월별 수익률 상관계수 계산 → 독립성 검증
  - 합산 포트폴리오 성과 (총수익률, MDD, 샤프, Calmar)

설정:
  - 기간: 2023-01-01 ~ 2025-12-31
  - 타임프레임: 4h봉
  - XRP: RSI<30, sl_atr=1.5, tp=bb_mid, lev=7x, pos=0.15, EMA200=OFF
  - SOL: RSI<45, sl_atr=1.0, tp=atr_2x, lev=7x, pos=0.15, EMA200=ON
  - 총 배분: XRP 50% + SOL 50% (각 pos_ratio=0.15로 나머지 절반은 현금 유보)
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE_RATE = 0.0005

INITIAL_BALANCE = 1000.0
START_DATE = "2023-01-01"
END_DATE   = "2025-12-31"

PARAMS = {
    "xrp": dict(rsi_thresh=30, sl_atr_mult=1.5, tp_mode="bb_mid",
                leverage=7, pos_ratio=0.15, use_ema200=False),
    "sol": dict(rsi_thresh=45, sl_atr_mult=1.0, tp_mode="atr_2x",
                leverage=7, pos_ratio=0.15, use_ema200=True),
}


# ── 데이터 로드 ─────────────────────────────────────

def load_range(coin: str, start: str, end: str) -> pd.DataFrame:
    """2022~2025 전체 로드 후 구간 필터 (EMA200 웜업 포함)."""
    years = [2022, 2023, 2024, 2025]
    frames = []
    for y in years:
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


# ── 백테스트 (거래 기록 반환) ─────────────────────────

def run_backtest_trades(
    df: pd.DataFrame,
    rsi_thresh: float,
    sl_atr_mult: float,
    tp_mode: str,
    leverage: int,
    pos_ratio: float,
    use_ema200: bool,
    timeout_bars: int = 48,
    initial_balance: float = INITIAL_BALANCE,
) -> tuple[list, list]:
    """(balance_series, trade_records) 반환."""
    closes   = df["close"].to_numpy(dtype=float)
    highs    = df["high"].to_numpy(dtype=float)
    lows     = df["low"].to_numpy(dtype=float)
    bb_mid   = df["bb_mid"].to_numpy(dtype=float)
    bb_lower = df["bb_lower"].to_numpy(dtype=float)
    rsi_arr  = df["rsi"].to_numpy(dtype=float)
    atr_arr  = df["atr"].to_numpy(dtype=float)
    ema200   = df["ema200"].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy()
    n = len(closes)

    balance = initial_balance
    balance_series = []  # (timestamp, balance)
    trades = []          # (exit_ts, pnl_pct)

    in_position = False
    entry_price = sl_price = tp_price = 0.0
    position_amount = 0.0
    entry_bar = 0
    entry_balance = 0.0

    for i in range(1, n):
        balance_series.append((timestamps[i], balance))

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
                entry_balance = balance
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
            tp_check = bb_mid[i] if tp_mode == "bb_mid" else tp_price
            if lows[i] <= sl_price:
                exit_price = sl_price
            elif tp_check > entry_price and highs[i] >= tp_check:
                exit_price = tp_check
            elif i - entry_bar >= timeout_bars:
                exit_price = closes[i]

            if exit_price is not None:
                pnl = (exit_price - entry_price) * position_amount
                fee = exit_price * position_amount * TAKER_FEE_RATE
                balance += pnl - fee
                trade_pnl_pct = (balance - entry_balance) / entry_balance * 100
                trades.append((timestamps[i], trade_pnl_pct))
                in_position = False

    return balance_series, trades


# ── 월별 수익률 계산 ──────────────────────────────────

def monthly_returns(balance_series: list, initial: float) -> pd.Series:
    """balance_series → 월별 수익률(%)."""
    if not balance_series:
        return pd.Series(dtype=float)
    df = pd.DataFrame(balance_series, columns=["ts", "balance"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    monthly = df["balance"].resample("ME").last().ffill()
    monthly_ret = monthly.pct_change() * 100
    # 첫 달: 초기 잔고 대비
    first_month = monthly.index[0]
    monthly_ret.loc[first_month] = (monthly.iloc[0] - initial) / initial * 100
    return monthly_ret


def portfolio_stats(balance_series: list, initial: float) -> dict:
    if not balance_series:
        return {"return_pct": 0, "max_drawdown": 0, "sharpe": 0, "calmar": 0}
    df_b = pd.DataFrame(balance_series, columns=["ts", "balance"])
    df_b = df_b.set_index("ts").resample("4h").last().ffill()
    balances = df_b["balance"].to_numpy(dtype=float)

    final = balances[-1]
    ret = (final - initial) / initial * 100

    peak = np.maximum.accumulate(balances)
    dd = (peak - balances) / peak * 100
    mdd = dd.max()

    # 샤프: 4h 수익률 → 연율화 (4h × 6 = 일, × 365)
    rets = np.diff(balances) / balances[:-1]
    sharpe = (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(6 * 365) if rets.std() > 0 else 0.0

    calmar = ret / mdd if mdd > 0 else 0.0

    return {
        "return_pct":   round(ret, 2),
        "max_drawdown": round(mdd, 2),
        "sharpe":       round(sharpe, 2),
        "calmar":       round(calmar, 2),
    }


# ── 포트폴리오 합산 잔고 ──────────────────────────────

def merge_balance_series(xrp_series: list, sol_series: list) -> list:
    """두 잔고 시리즈를 합산 (동일 타임스탬프 기준)."""
    df_x = pd.DataFrame(xrp_series, columns=["ts", "xrp"]).set_index("ts")
    df_s = pd.DataFrame(sol_series, columns=["ts", "sol"]).set_index("ts")
    merged = pd.merge(df_x, df_s, left_index=True, right_index=True, how="outer")
    merged = merged.ffill().bfill()
    merged["total"] = merged["xrp"] + merged["sol"]
    return list(zip(merged.index, merged["total"]))


# ── 메인 ─────────────────────────────────────────────

def main():
    print("XRP + SOL 포트폴리오 상관관계 검증")
    print(f"기간: {START_DATE} ~ {END_DATE} | 타임프레임: 4h\n")

    results = {}
    balance_series_map = {}

    for coin, params in PARAMS.items():
        print(f"[{coin.upper()}] 데이터 로드 및 백테스트 중...", flush=True)
        df_1m = load_range(coin, START_DATE, END_DATE)
        df_4h = resample_4h(df_1m)
        df_4h = compute_indicators(df_4h)

        # 평가 구간 필터
        s = pd.Timestamp(START_DATE, tz="UTC")
        e = pd.Timestamp(END_DATE + " 23:59:59", tz="UTC")
        df_eval = df_4h[(df_4h["timestamp"] >= s) & (df_4h["timestamp"] <= e)].reset_index(drop=True)

        bal_series, trades = run_backtest_trades(df_eval, **params)
        balance_series_map[coin] = bal_series
        results[coin] = {
            "bal_series": bal_series,
            "monthly":    monthly_returns(bal_series, INITIAL_BALANCE / 2),
            "stats":      portfolio_stats(bal_series, INITIAL_BALANCE / 2),
            "trades":     trades,
        }
        st = results[coin]["stats"]
        print(f"  → 수익={st['return_pct']:+.1f}%  MDD={st['max_drawdown']:.1f}%  "
              f"샤프={st['sharpe']:.2f}  Calmar={st['calmar']:.2f}  "
              f"거래={len(trades)}건")

    # ── 포트폴리오 합산 ──
    print("\n[포트폴리오] 합산 계산 중...", flush=True)
    port_series = merge_balance_series(
        balance_series_map["xrp"], balance_series_map["sol"]
    )
    port_stats = portfolio_stats(port_series, INITIAL_BALANCE)
    port_monthly = monthly_returns(port_series, INITIAL_BALANCE)

    # ── 월별 수익률 테이블 ──
    xrp_m = results["xrp"]["monthly"]
    sol_m = results["sol"]["monthly"]

    all_months = xrp_m.index.union(sol_m.index).union(port_monthly.index)
    df_monthly = pd.DataFrame(index=all_months)
    df_monthly["xrp"] = xrp_m.reindex(all_months)
    df_monthly["sol"] = sol_m.reindex(all_months)
    df_monthly["portfolio"] = port_monthly.reindex(all_months)
    df_monthly = df_monthly.sort_index()
    df_monthly["year_month"] = df_monthly.index.strftime("%Y-%m")

    print(f"\n{'월':>8}  {'XRP':>9}  {'SOL':>9}  {'합산':>9}")
    print("-" * 42)
    for _, row in df_monthly.iterrows():
        xrp_v = f"{row['xrp']:+.1f}%" if pd.notna(row["xrp"]) else "   —   "
        sol_v = f"{row['sol']:+.1f}%" if pd.notna(row["sol"]) else "   —   "
        prt_v = f"{row['portfolio']:+.1f}%" if pd.notna(row["portfolio"]) else "   —   "
        print(f"{row['year_month']:>8}  {xrp_v:>9}  {sol_v:>9}  {prt_v:>9}")

    # ── 상관계수 ──
    common = df_monthly[["xrp", "sol"]].dropna()
    corr = common["xrp"].corr(common["sol"]) if len(common) > 2 else float("nan")

    print(f"\n== 상관계수 (월간 수익률 XRP vs SOL) ==")
    print(f"  Pearson r = {corr:.3f}  ", end="")
    if corr < 0.3:
        print("→ 낮음 (독립적, 포트폴리오 분산 효과 있음)")
    elif corr < 0.6:
        print("→ 중간 (부분적 독립)")
    else:
        print("→ 높음 (함께 움직임, 분산 효과 제한)")

    # ── 포트폴리오 vs 단독 ──
    xrp_st = results["xrp"]["stats"]
    sol_st = results["sol"]["stats"]

    print(f"\n== 성과 비교 (2023~2025) ==")
    print(f"{'':12} {'수익률':>9}  {'MDD':>8}  {'샤프':>7}  {'Calmar':>8}")
    print("-" * 50)
    print(f"{'XRP 단독':12} {xrp_st['return_pct']:>+8.1f}%  {xrp_st['max_drawdown']:>7.1f}%  "
          f"{xrp_st['sharpe']:>7.2f}  {xrp_st['calmar']:>8.2f}")
    print(f"{'SOL 단독':12} {sol_st['return_pct']:>+8.1f}%  {sol_st['max_drawdown']:>7.1f}%  "
          f"{sol_st['sharpe']:>7.2f}  {sol_st['calmar']:>8.2f}")
    print(f"{'포트폴리오':12} {port_stats['return_pct']:>+8.1f}%  {port_stats['max_drawdown']:>7.1f}%  "
          f"{port_stats['sharpe']:>7.2f}  {port_stats['calmar']:>8.2f}")

    # ── CSV 저장 ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "bb_rsi_portfolio.csv")
    save_df = df_monthly.copy()
    save_df = save_df.round(2)
    save_df.to_csv(out_path, index=False)

    # 요약 행 추가
    summary = pd.DataFrame([{
        "year_month": "SUMMARY",
        "xrp": xrp_st["return_pct"],
        "sol": sol_st["return_pct"],
        "portfolio": port_stats["return_pct"],
    }, {
        "year_month": "MDD",
        "xrp": xrp_st["max_drawdown"],
        "sol": sol_st["max_drawdown"],
        "portfolio": port_stats["max_drawdown"],
    }, {
        "year_month": "SHARPE",
        "xrp": xrp_st["sharpe"],
        "sol": sol_st["sharpe"],
        "portfolio": port_stats["sharpe"],
    }, {
        "year_month": "CALMAR",
        "xrp": xrp_st["calmar"],
        "sol": sol_st["calmar"],
        "portfolio": port_stats["calmar"],
    }, {
        "year_month": "CORRELATION",
        "xrp": round(corr, 3),
        "sol": round(corr, 3),
        "portfolio": round(corr, 3),
    }])
    full_df = pd.concat([save_df, summary], ignore_index=True)
    full_df.to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
