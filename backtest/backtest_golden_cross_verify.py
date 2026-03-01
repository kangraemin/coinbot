"""SMA 골든크로스 동일 파라미터 3코인 검증.

코인마다 최적 파라미터가 다르면 과적합 의심 → 동일 파라미터로 재검증.

세트 A: SMA20/150  long_only  1x 100%
세트 B: SMA50/200  long_only  1x 100%
세트 C: SMA100/200 long_only  1x 100%

코인: BTC, ETH, XRP | 기간: 2022~2025
"""

import os
import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0

COINS  = ["btc", "eth", "xrp"]
BH_REF = {"btc": 89.4, "eth": -19.4, "xrp": 121.6}

PARAM_SETS = {
    "A": dict(sma_fast=20,  sma_slow=150),
    "B": dict(sma_fast=50,  sma_slow=200),
    "C": dict(sma_fast=100, sma_slow=200),
}


# ── 데이터 준비 ───────────────────────────────────────

def load_range(coin: str) -> pd.DataFrame:
    frames = []
    for y in range(2020, 2026):
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
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),    close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna().reset_index()


def add_sma(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["sma_fast"] = c.rolling(fast, min_periods=fast).mean()
    df["sma_slow"] = c.rolling(slow, min_periods=slow).mean()
    return df


# ── 백테스트 코어 ─────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = INITIAL_BAL,
) -> dict:
    """long_only 1x 100% 고정. df에 sma_fast/sma_slow/open/close 필요."""
    closes = df["close"].to_numpy(float)
    opens  = df["open"].to_numpy(float)
    sma_f  = df["sma_fast"].to_numpy(float)
    sma_s  = df["sma_slow"].to_numpy(float)
    n = len(closes)

    balance = initial_balance
    peak    = initial_balance
    mdd     = 0.0

    in_pos    = False
    ep = pos_amt = 0.0
    trades = wins = 0
    signal_next = None

    for i in range(1, n):
        if (np.isnan(sma_f[i]) or np.isnan(sma_s[i])
                or np.isnan(sma_f[i-1]) or np.isnan(sma_s[i-1])):
            signal_next = None
            continue

        # ① 이전 봉 신호를 오늘 시가에 실행
        if signal_next == "golden" and not in_pos and balance > 0:
            pos_amt  = balance / opens[i]
            balance -= balance * TAKER_FEE
            ep       = opens[i]
            in_pos   = True
        elif signal_next == "death" and in_pos:
            exit_p  = opens[i]
            pnl     = (exit_p - ep) * pos_amt
            balance += pnl - exit_p * pos_amt * TAKER_FEE
            trades  += 1
            if pnl > 0:
                wins += 1
            in_pos = False

        signal_next = None

        # ② 크로스오버 감지
        golden = (sma_f[i-1] < sma_s[i-1]) and (sma_f[i] >= sma_s[i])
        death  = (sma_f[i-1] > sma_s[i-1]) and (sma_f[i] <= sma_s[i])

        if golden and not in_pos:
            signal_next = "golden"
        elif death and in_pos:
            signal_next = "death"

        # ③ 포트폴리오 가치
        pv = balance + (closes[i] - ep) * pos_amt if in_pos else balance
        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak * 100
        if dd > mdd:
            mdd = dd

    # 미청산 포지션 강제 청산
    if in_pos:
        exit_p  = closes[-1]
        pnl     = (exit_p - ep) * pos_amt
        balance += pnl - exit_p * pos_amt * TAKER_FEE
        trades  += 1
        if pnl > 0:
            wins += 1

    ret    = (balance - initial_balance) / initial_balance * 100 if balance > 0 else -100.0
    wr     = wins / trades * 100 if trades > 0 else 0.0
    calmar = ret / mdd if mdd > 0 else 0.0

    return {
        "trades":       trades,
        "win_rate":     round(wr, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(mdd, 2),
        "calmar":       round(calmar, 2),
    }


# ── 메인 ──────────────────────────────────────────────

def main():
    print("SMA 골든크로스 동일 파라미터 검증 — BTC/ETH/XRP, 2022~2025")
    print("long_only | 1x 레버리지 | 비중 100%\n")

    # 데이터 캐시
    coin_data = {}
    for coin in COINS:
        df_1m = load_range(coin)
        df_1d = resample_1d(df_1m)
        coin_data[coin] = df_1d

    all_rows = []

    for set_name, params in PARAM_SETS.items():
        sf = params["sma_fast"]
        ss = params["sma_slow"]
        print(f"{'='*60}")
        print(f"세트 {set_name}: SMA{sf}/{ss} long_only 1x 100%")
        print(f"{'='*60}")
        print(f"  {'코인':5} {'수익률':>9} {'MDD':>7} {'Calmar':>8} {'거래':>5} {'B&H':>9} {'판정':>10}")
        print(f"  {'-'*58}")

        set_results = {}
        for coin in COINS:
            df_sma  = add_sma(coin_data[coin], sf, ss)
            s = pd.Timestamp("2022-01-01", tz="UTC")
            e = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
            df_eval = df_sma[(df_sma["timestamp"] >= s) & (df_sma["timestamp"] <= e)].reset_index(drop=True)
            r = run_backtest(df_eval)
            bh   = BH_REF[coin]
            beat = "✓ B&H 초과" if r["return_pct"] > bh else "✗ B&H 미달"
            print(
                f"  {coin.upper():5} {r['return_pct']:>+8.1f}% "
                f"{r['max_drawdown']:>6.1f}%  {r['calmar']:>7.2f} "
                f"{r['trades']:>4}건 {bh:>+8.1f}% {beat:>10}"
            )
            set_results[coin] = r
            all_rows.append({"set": set_name, "sma_fast": sf, "sma_slow": ss,
                              "coin": coin.upper(), **r, "bh_ref": bh})

        # 세트 판정
        all_beat = all(set_results[c]["return_pct"] > BH_REF[c] for c in COINS)
        print(f"\n  → 3코인 모두 B&H 초과: {'✅ YES' if all_beat else '❌ NO'}")

        # 연도별
        print(f"\n  연도별 수익률 (SMA{sf}/{ss} long_only 1x 100%):")
        print(f"  {'연도':6}", end="")
        for coin in COINS:
            print(f"  {coin.upper():>10}", end="")
        print()
        print(f"  {'-'*42}")

        for year in range(2022, 2026):
            s = pd.Timestamp(f"{year}-01-01", tz="UTC")
            e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
            print(f"  {year:4}  ", end="")
            for coin in COINS:
                df_sma = add_sma(coin_data[coin], sf, ss)
                df_y   = df_sma[(df_sma["timestamp"] >= s) & (df_sma["timestamp"] <= e)].reset_index(drop=True)
                if len(df_y) < 5:
                    print(f"  {'N/A':>10}", end="")
                    continue
                r = run_backtest(df_y)
                flag = "✓" if r["return_pct"] >= 0 else "✗"
                print(f"  {r['return_pct']:>+8.1f}%{flag}", end="")
            print()
        print()

    # 결과 저장
    df_out = pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "golden_cross_verify.csv")
    df_out.to_csv(out_path, index=False)

    # 최종 요약
    print(f"{'='*60}")
    print("최종 요약")
    print(f"{'='*60}")
    print(f"  {'세트':6} {'SMA':>12} {'BTC':>10} {'ETH':>10} {'XRP':>10} {'3코인 모두':>10}")
    print(f"  {'-'*58}")
    for set_name, params in PARAM_SETS.items():
        sf = params["sma_fast"]
        ss = params["sma_slow"]
        row_data = {r["coin"]: r for r in all_rows if r["set"] == set_name}
        btc_r = row_data.get("BTC", {}).get("return_pct", 0)
        eth_r = row_data.get("ETH", {}).get("return_pct", 0)
        xrp_r = row_data.get("XRP", {}).get("return_pct", 0)
        all_beat = (btc_r > BH_REF["btc"] and eth_r > BH_REF["eth"] and xrp_r > BH_REF["xrp"])
        print(
            f"  {set_name:4}  SMA{sf:3}/{ss:3}  "
            f"{btc_r:>+8.1f}%  {eth_r:>+8.1f}%  {xrp_r:>+8.1f}%  "
            f"{'✅' if all_beat else '❌'}"
        )
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
