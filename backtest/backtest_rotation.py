"""코인 간 모멘텀 로테이션 백테스트.

규칙:
  매월 1일: BTC/ETH/XRP의 직전 N개월 수익률 계산
  1위 선택 → 다음 달 내내 보유
  3코인 모두 음수 → CASH 보유

코인 풀: BTC, ETH, XRP | 기간: 2022~2025
"""

import os
import itertools

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0
START_DATE  = "2022-01-01"
END_DATE    = "2025-12-31"

LOOKBACKS  = [1, 2, 3, 6]
LEVERAGES  = [1, 2, 3]
POS_RATIOS = [0.5, 1.0]

COINS  = ["btc", "eth", "xrp"]
BH_REF = {"btc": 89.4, "eth": -19.4, "xrp": 121.6}


# ── 데이터 준비 ───────────────────────────────────────

def load_range(coin: str) -> pd.DataFrame:
    frames = []
    for y in range(2020, 2026):   # 6개월 룩백 웜업
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


def build_merged() -> pd.DataFrame:
    """BTC/ETH/XRP 일봉 종가를 하나의 DataFrame으로 병합."""
    dfs = {}
    for coin in COINS:
        df_1m = load_range(coin)
        df_1d = resample_1d(df_1m).set_index("timestamp")
        dfs[coin] = df_1d["close"]
    merged = pd.concat(dfs.values(), axis=1, join="inner")
    merged.columns = COINS
    return merged.reset_index()   # columns: timestamp, btc, eth, xrp


# ── 백테스트 코어 ─────────────────────────────────────

def run_backtest(
    df_all: pd.DataFrame,
    lookback: int,
    leverage: int,
    pos_ratio: float,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    initial_balance: float = INITIAL_BAL,
) -> dict:
    """
    df_all: columns=[timestamp, btc, eth, xrp], 2020~2025 전체 일봉 종가
    """
    df = df_all.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    eval_s = pd.Timestamp(start_date, tz="UTC")
    eval_e = pd.Timestamp(end_date + " 23:59:59", tz="UTC")

    n = len(df)
    balance = initial_balance
    peak    = initial_balance
    mdd     = 0.0

    in_pos       = False
    current_coin = None
    ep = pos_amt = 0.0

    trades = wins = 0
    prev_ym = None
    last_eval_i = 0

    for i in range(n):
        ts      = df.at[i, "timestamp"]
        ym      = (ts.year, ts.month)
        in_eval = eval_s <= ts <= eval_e

        if in_eval:
            last_eval_i = i

        # 월 시작 + 평가 기간: 리밸런싱
        if ym != prev_ym and in_eval:
            lb_ts = ts - pd.DateOffset(months=lookback)

            # 룩백 시점 가격: lb_ts 이전 마지막 available row
            lb_mask = df["timestamp"] <= lb_ts
            if not lb_mask.any():
                chosen = None
            else:
                lb_i  = lb_mask.values.nonzero()[0][-1]
                returns = {}
                for coin in COINS:
                    lb_price  = df.at[lb_i, coin]
                    cur_price = df.at[i, coin]
                    if lb_price > 0:
                        returns[coin] = (cur_price - lb_price) / lb_price

                if not returns or all(v < 0 for v in returns.values()):
                    chosen = None   # CASH
                else:
                    chosen = max(returns, key=returns.get)

            # 기존 포지션 청산
            if in_pos:
                exit_p   = df.at[i, current_coin]
                pnl      = (exit_p - ep) * pos_amt
                balance += pnl - exit_p * pos_amt * TAKER_FEE
                trades  += 1
                if pnl > 0:
                    wins += 1
                in_pos = False

            # 새 포지션 진입
            if chosen and balance > 0:
                entry_p   = df.at[i, chosen]
                notional  = balance * pos_ratio * leverage
                pos_amt   = notional / entry_p
                balance  -= notional * TAKER_FEE
                ep        = entry_p
                current_coin = chosen
                in_pos    = True

        prev_ym = ym

        # 일별 포트폴리오 가치 (MDD 추적)
        if in_eval:
            pv = (balance + (df.at[i, current_coin] - ep) * pos_amt) if in_pos else balance
            if pv > peak:
                peak = pv
            dd = (peak - pv) / peak * 100
            if dd > mdd:
                mdd = dd

    # 미청산 포지션 강제 청산
    if in_pos:
        exit_p   = df.at[last_eval_i, current_coin]
        pnl      = (exit_p - ep) * pos_amt
        balance += pnl - exit_p * pos_amt * TAKER_FEE
        trades  += 1
        if pnl > 0:
            wins += 1

    ret    = (balance - initial_balance) / initial_balance * 100 if balance > 0 else -100.0
    wr     = wins / trades * 100 if trades > 0 else 0.0
    calmar = ret / mdd if mdd > 0 else 0.0

    return {
        "lookback":     lookback,
        "leverage":     leverage,
        "pos_ratio":    pos_ratio,
        "trades":       trades,
        "win_rate":     round(wr, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(mdd, 2),
        "calmar":       round(calmar, 2),
    }


# ── 메인 ──────────────────────────────────────────────

def main():
    print("코인 간 모멘텀 로테이션 — BTC/ETH/XRP, 2022~2025")
    combos = list(itertools.product(LOOKBACKS, LEVERAGES, POS_RATIOS))
    print(f"조합: {len(combos)}개\n")

    print("데이터 로딩 중...", flush=True)
    df_merged = build_merged()

    results = []
    for lb, lev, pos in combos:
        r = run_backtest(df_merged, lb, lev, pos)
        results.append(r)

    df_res = pd.DataFrame(results).sort_values("return_pct", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "rotation.csv")
    df_res.to_csv(out_path, index=False)

    best    = df_res.iloc[0]
    bh_max  = max(BH_REF.values())   # XRP +121.6%
    beat    = "✓ B&H 최대(XRP) 초과" if best["return_pct"] > bh_max else "✗ B&H 최대(XRP) 미달"

    df_1x   = df_res[(df_res["leverage"] == 1) & (df_res["pos_ratio"] == 1.0)]
    best_1x = df_1x.iloc[0] if len(df_1x) > 0 else None

    print(f"{'룩백':>5} {'레버리지':>5} {'비중':>5} {'수익률':>9} {'MDD':>7} {'Calmar':>8} {'거래':>5} {'승률':>7}")
    print(f"{'-'*64}")
    for _, row in df_res.iterrows():
        print(
            f"  {row['lookback']:>3}M  {row['leverage']:>4}x  {row['pos_ratio']*100:>4.0f}%  "
            f"{row['return_pct']:>+8.1f}%  {row['max_drawdown']:>6.1f}%  "
            f"{row['calmar']:>7.2f}  {row['trades']:>4}건  {row['win_rate']:>6.1f}%"
        )

    print(
        f"\nBest: 수익={best['return_pct']:+.1f}%  MDD={best['max_drawdown']:.1f}%  "
        f"Calmar={best['calmar']:.2f} | "
        f"lookback={best['lookback']}M lev={best['leverage']}x pos={best['pos_ratio']*100:.0f}%"
    )
    print(f"B&H 최대 (XRP): {bh_max:+.1f}% → {beat}")

    if best_1x is not None:
        beat_1x = "✓ B&H 최대 초과" if best_1x["return_pct"] > bh_max else "✗ B&H 최대 미달"
        print(
            f"\n[1x 100%] Best: 수익={best_1x['return_pct']:+.1f}%  "
            f"MDD={best_1x['max_drawdown']:.1f}%  Calmar={best_1x['calmar']:.2f}  "
            f"lookback={best_1x['lookback']}M → {beat_1x}"
        )

    # ── 연도별 견고성 ──
    print(f"\n── 연도별 견고성 (Best 파라미터) ──")
    lb_best  = int(best["lookback"])
    lev_best = int(best["leverage"])
    pos_best = float(best["pos_ratio"])
    print(f"파라미터: lookback={lb_best}M, lev={lev_best}x, pos={pos_best*100:.0f}%")
    print(f"  {'연도':6} {'수익률':>9} {'MDD':>7} {'Calmar':>8} {'거래':>5}")
    print(f"  {'-'*40}")

    yearly_rows = []
    for year in range(2022, 2027):
        s = f"{year}-01-01"
        e = f"{year}-12-31"
        r = run_backtest(df_merged, lb_best, lev_best, pos_best, start_date=s, end_date=e)
        suffix = " (YTD)" if year == 2026 else ""
        print(
            f"  {year}{suffix:6} {r['return_pct']:>+8.1f}%  "
            f"{r['max_drawdown']:>6.1f}%  {r['calmar']:>7.2f}  {r['trades']:>4}건"
        )
        yearly_rows.append({"year": year, **r})

    # 1x 연도별 (best_1x 파라미터)
    if best_1x is not None and (int(best_1x["lookback"]) != lb_best
                                 or int(best_1x["leverage"]) != lev_best
                                 or float(best_1x["pos_ratio"]) != pos_best):
        lb_1x  = int(best_1x["lookback"])
        lev_1x = int(best_1x["leverage"])
        pos_1x = float(best_1x["pos_ratio"])
        print(f"\n── 연도별 견고성 (1x Best: lookback={lb_1x}M) ──")
        print(f"  {'연도':6} {'수익률':>9} {'MDD':>7} {'Calmar':>8} {'거래':>5}")
        print(f"  {'-'*40}")
        for year in range(2022, 2027):
            s = f"{year}-01-01"
            e = f"{year}-12-31"
            r = run_backtest(df_merged, lb_1x, lev_1x, pos_1x, start_date=s, end_date=e)
            suffix = " (YTD)" if year == 2026 else ""
            print(
                f"  {year}{suffix:6} {r['return_pct']:>+8.1f}%  "
                f"{r['max_drawdown']:>6.1f}%  {r['calmar']:>7.2f}  {r['trades']:>4}건"
            )

    pd.DataFrame(yearly_rows).to_csv(
        os.path.join(RESULTS_DIR, "rotation_robustness.csv"), index=False
    )
    print(f"\n저장: {out_path}")
    print(f"저장: {os.path.join(RESULTS_DIR, 'rotation_robustness.csv')}")


if __name__ == "__main__":
    main()
