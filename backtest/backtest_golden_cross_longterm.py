"""SMA 골든크로스 장기 검증 — 1h 전체 데이터 사용.

데이터: data/market/{coin}_1h_full.parquet (1h봉 → 일봉 리샘플)
  BTC/ETH: 2017~2026 | XRP: 2018~2026

세트 A: SMA20/150  long_only 1x 100%
세트 B: SMA50/200  long_only 1x 100%

핵심 확인:
  1. 2017~2018 불장 → 2018 하락장 사이클
  2. 2020~2021 불장 → 2022 하락장 회피
  3. 장기 B&H 대비 성과 (BTC +1433%, ETH +552%, XRP +48%)
"""

import os
import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0

COINS  = ["btc", "eth", "xrp"]
BH_REF = {"btc": 1433.0, "eth": 552.0, "xrp": 48.0}   # 전체 기간 B&H

# 코인별 평가 시작일 (SMA 웜업 후 실거래 시작)
EVAL_START = {"btc": "2017-01-01", "eth": "2017-01-01", "xrp": "2018-01-01"}
EVAL_END   = "2025-12-31"

PARAM_SETS = {
    "A": dict(sma_fast=20,  sma_slow=150),
    "B": dict(sma_fast=50,  sma_slow=200),
}


# ── 데이터 준비 ───────────────────────────────────────

def load_1h(coin: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{coin}_1h_full.parquet")
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def resample_1d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp")
    daily = df.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),    close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna().reset_index()
    return daily


def add_sma(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """전체 df 기준으로 SMA 계산 (히스토리 보존)."""
    df = df.copy()
    c = df["close"]
    df["sma_fast"] = c.rolling(fast, min_periods=fast).mean()
    df["sma_slow"] = c.rolling(slow, min_periods=slow).mean()
    return df


# ── 백테스트 코어 ─────────────────────────────────────

def run_backtest(df: pd.DataFrame, initial_balance: float = INITIAL_BAL) -> dict:
    """long_only 1x 100% 고정. df에 sma_fast/sma_slow/open/close 필요."""
    closes = df["close"].to_numpy(float)
    opens  = df["open"].to_numpy(float)
    sma_f  = df["sma_fast"].to_numpy(float)
    sma_s  = df["sma_slow"].to_numpy(float)
    n = len(closes)

    balance = initial_balance
    peak    = initial_balance
    mdd     = 0.0

    in_pos  = False
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
            exit_p   = opens[i]
            pnl      = (exit_p - ep) * pos_amt
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

        pv = balance + (closes[i] - ep) * pos_amt if in_pos else balance
        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak * 100
        if dd > mdd:
            mdd = dd

    # 미청산 강제 청산
    if in_pos:
        exit_p   = closes[-1]
        pnl      = (exit_p - ep) * pos_amt
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
    print("SMA 골든크로스 장기 검증 — 1h 전체 데이터")
    print("long_only | 1x 레버리지 | 비중 100%\n")

    # 데이터 로딩 및 일봉 변환
    coin_daily = {}
    for coin in COINS:
        df_1h  = load_1h(coin)
        df_1d  = resample_1d(df_1h)
        coin_daily[coin] = df_1d
        start  = df_1d["timestamp"].iloc[0].strftime("%Y-%m-%d")
        end    = df_1d["timestamp"].iloc[-1].strftime("%Y-%m-%d")
        print(f"  {coin.upper()}: {len(df_1d)}일봉 ({start} ~ {end})")

    print()
    all_rows = []

    for set_name, params in PARAM_SETS.items():
        sf = params["sma_fast"]
        ss = params["sma_slow"]
        print(f"{'='*70}")
        print(f"세트 {set_name}: SMA{sf}/{ss} long_only 1x 100%")
        print(f"{'='*70}")

        # ── 전체 기간 성과 ──
        print(f"\n  [전체 기간 성과]")
        print(f"  {'코인':5} {'수익률':>10} {'MDD':>8} {'Calmar':>8} {'거래':>5} "
              f"{'B&H':>9} {'판정':>12}")
        print(f"  {'-'*62}")

        set_results = {}
        for coin in COINS:
            df_sma = add_sma(coin_daily[coin], sf, ss)
            s = pd.Timestamp(EVAL_START[coin], tz="UTC")
            e = pd.Timestamp(EVAL_END + " 23:59:59", tz="UTC")
            df_eval = df_sma[(df_sma["timestamp"] >= s) &
                             (df_sma["timestamp"] <= e)].reset_index(drop=True)
            r   = run_backtest(df_eval)
            bh  = BH_REF[coin]
            beat = "✓ B&H 초과" if r["return_pct"] > bh else "✗ B&H 미달"
            print(
                f"  {coin.upper():5} {r['return_pct']:>+9.1f}% "
                f"{r['max_drawdown']:>7.1f}%  {r['calmar']:>7.2f} "
                f"{r['trades']:>4}건 {bh:>+8.1f}% {beat:>12}"
            )
            set_results[coin] = r
            all_rows.append({
                "set": set_name, "sma_fast": sf, "sma_slow": ss,
                "coin": coin.upper(), "period": "full", **r, "bh_ref": bh,
            })

        all_beat = all(set_results[c]["return_pct"] > BH_REF[c] for c in COINS)
        print(f"\n  → 3코인 모두 B&H 초과: {'✅ YES' if all_beat else '❌ NO'}")

        # ── 연도별 수익률 ──
        print(f"\n  [연도별 수익률]")
        print(f"  {'연도':6}", end="")
        for coin in COINS:
            print(f"  {coin.upper():>12}", end="")
        print()
        print(f"  {'-'*48}")

        # 연도 범위: 전체 가용 데이터 기준
        all_years = sorted(set(
            year
            for coin in COINS
            for year in range(
                int(EVAL_START[coin][:4]),
                2026
            )
        ))

        for year in all_years:
            s = pd.Timestamp(f"{year}-01-01", tz="UTC")
            e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
            suffix = " (YTD)" if year == 2026 else ""
            print(f"  {year}{suffix:4}", end="")

            for coin in COINS:
                # 해당 코인의 평가 시작 연도 이전이면 스킵
                if year < int(EVAL_START[coin][:4]):
                    print(f"  {'—':>12}", end="")
                    continue

                df_sma = add_sma(coin_daily[coin], sf, ss)
                df_y   = df_sma[(df_sma["timestamp"] >= s) &
                                (df_sma["timestamp"] <= e)].reset_index(drop=True)
                if len(df_y) < 5:
                    print(f"  {'N/A':>12}", end="")
                    continue

                r    = run_backtest(df_y)
                ret  = r["return_pct"]
                flag = "✓" if ret >= 0 else "✗"
                note = "(회피)" if r["trades"] == 0 else ""
                val  = f"{ret:>+7.1f}%{flag}{note}"
                print(f"  {val:>12}", end="")

                all_rows.append({
                    "set": set_name, "sma_fast": sf, "sma_slow": ss,
                    "coin": coin.upper(), "period": str(year), **r, "bh_ref": BH_REF[coin],
                })

            print()
        print()

    # 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "golden_cross_longterm.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)

    # 최종 비교 요약
    print(f"{'='*70}")
    print("최종 요약 (전체 기간)")
    print(f"{'='*70}")
    print(f"  {'세트':5} {'SMA':>12} {'BTC':>12} {'ETH':>12} {'XRP':>12} {'전체 판정':>10}")
    print(f"  {'-'*64}")
    for set_name, params in PARAM_SETS.items():
        sf = params["sma_fast"]
        ss = params["sma_slow"]
        row_data = {r["coin"]: r for r in all_rows
                    if r["set"] == set_name and r["period"] == "full"}
        btc_r = row_data.get("BTC", {}).get("return_pct", 0)
        eth_r = row_data.get("ETH", {}).get("return_pct", 0)
        xrp_r = row_data.get("XRP", {}).get("return_pct", 0)
        all_beat = (btc_r > BH_REF["btc"]
                    and eth_r > BH_REF["eth"]
                    and xrp_r > BH_REF["xrp"])
        print(
            f"  {set_name:4}  SMA{sf:3}/{ss:3}  "
            f"{btc_r:>+10.1f}%  {eth_r:>+10.1f}%  {xrp_r:>+10.1f}%  "
            f"{'✅' if all_beat else '❌'}"
        )
    print(f"\n  B&H:        BTC {BH_REF['btc']:>+10.1f}%  "
          f"ETH {BH_REF['eth']:>+10.1f}%  XRP {BH_REF['xrp']:>+10.1f}%")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
