"""4H BB+RSI 양방향 전략 — 전체 기간(2017~2026) 그리드 서치.

데이터: {coin}_1h_full.parquet → 4H 리샘플
  BTC/ETH: 2017-08-17 ~ 2026-03
  XRP:     2018-05-04 ~ 2026-03

B&H 기준 (전체 기간):
  BTC +1419% / ETH +539% / XRP +49%

정렬 기준:
  1) B&H 초과 조합 → Calmar 내림차순
  2) 전체 조합     → return_pct 내림차순

평가 포인트:
  - 전체 기간 수익률
  - MDD (최대낙폭)
  - 연도별 플러스 비율
  - B&H 초과 여부
"""

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0

# ── 파라미터 그리드 ───────────────────────────────────────
RSI_LONG   = [20, 25, 30, 35]
RSI_SHORT  = [65, 70, 75]
SL_MULTS   = [1.5, 2.0, 2.5, 3.0]
TP_MODES   = ["atr_2x", "atr_3x"]
LEVERAGES  = [1, 2, 3]
POS_RATIOS = [0.30, 0.50, 0.70]
USE_EMA200 = [True, False]

COINS = ["btc", "eth", "xrp"]

BH_FULL = {"btc": 1419.1, "eth": 539.0, "xrp": 49.4}


# ── 데이터 ────────────────────────────────────────────────

def load_4h(coin: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{coin}_1h_full.parquet")
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df.resample("4h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),     close=("close","last"),
        volume=("volume","sum"),
    ).dropna().reset_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c  = df["close"].astype(float)
    h  = df["high"].astype(float)
    lo = df["low"].astype(float)

    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std(ddof=1)
    delta    = c.diff()
    gain     = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss     = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi      = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    tr       = pd.concat([h-lo,(h-c.shift(1)).abs(),(lo-c.shift(1)).abs()],axis=1).max(axis=1)
    atr      = tr.ewm(com=13, adjust=False).mean()
    ema200   = c.ewm(span=200, adjust=False).mean()

    df = df.copy()
    df["bb_mid"]   = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["rsi"]      = rsi
    df["atr"]      = atr
    df["ema200"]   = ema200
    return df


# ── 백테스트 코어 ────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    rsi_long: float,
    rsi_short: float,
    sl_mult: float,
    tp_mode: str,
    leverage: int,
    pos_ratio: float,
    use_ema200: bool,
    timeout_bars: int = 48,
) -> dict:
    closes   = df["close"].to_numpy(float)
    highs    = df["high"].to_numpy(float)
    lows     = df["low"].to_numpy(float)
    bb_mid   = df["bb_mid"].to_numpy(float)
    bb_upper = df["bb_upper"].to_numpy(float)
    bb_lower = df["bb_lower"].to_numpy(float)
    rsi_arr  = df["rsi"].to_numpy(float)
    atr_arr  = df["atr"].to_numpy(float)
    ema200   = df["ema200"].to_numpy(float)
    timestamps = df["timestamp"].tolist()
    n = len(closes)

    balance = INITIAL_BAL
    peak    = INITIAL_BAL
    mdd     = 0.0

    in_pos    = False
    direction = 0
    ep = sl = tp = pos_amt = 0.0
    entry_bar = 0

    long_trades = long_wins = 0
    short_trades = short_wins = 0

    year_returns: dict[int, float] = {}
    year_start_bal: dict[int, float] = {}

    for i in range(1, n):
        yr = timestamps[i].year
        if yr not in year_start_bal:
            year_start_bal[yr] = balance

        if any(np.isnan(x[i]) for x in [bb_mid, bb_upper, bb_lower, rsi_arr, atr_arr, ema200]):
            continue

        if in_pos:
            exit_p = None

            if direction == 1:
                tp_check = bb_mid[i] if tp_mode == "bb_mid" else tp
                if lows[i] <= sl:
                    exit_p = sl
                elif tp_check > ep and highs[i] >= tp_check:
                    exit_p = tp_check; long_wins += 1
                elif i - entry_bar >= timeout_bars:
                    exit_p = closes[i]
                    if exit_p > ep: long_wins += 1
            else:
                tp_check = bb_mid[i] if tp_mode == "bb_mid" else tp
                if highs[i] >= sl:
                    exit_p = sl
                elif tp_check < ep and lows[i] <= tp_check:
                    exit_p = tp_check; short_wins += 1
                elif i - entry_bar >= timeout_bars:
                    exit_p = closes[i]
                    if exit_p < ep: short_wins += 1

            if exit_p is not None:
                pnl     = (exit_p - ep) * pos_amt if direction == 1 else (ep - exit_p) * pos_amt
                fee     = exit_p * pos_amt * TAKER_FEE
                balance += pnl - fee
                if direction == 1: long_trades += 1
                else: short_trades += 1
                in_pos    = False
                direction = 0

            if balance > peak: peak = balance
            dd = (peak - balance) / peak * 100
            if dd > mdd: mdd = dd

        if not in_pos:
            if balance <= 0: break
            notional = balance * pos_ratio * leverage

            ema_long_ok  = (not use_ema200) or (closes[i] > ema200[i])
            ema_short_ok = (not use_ema200) or (closes[i] < ema200[i])

            if ema_long_ok and closes[i] < bb_lower[i] and rsi_arr[i] < rsi_long:
                ep=closes[i]; pos_amt=notional/ep; balance-=notional*TAKER_FEE
                sl=ep-atr_arr[i]*sl_mult
                tp=ep+atr_arr[i]*(3.0 if tp_mode=="atr_3x" else 2.0)
                entry_bar=i; in_pos=True; direction=1
                continue

            if ema_short_ok and closes[i] > bb_upper[i] and rsi_arr[i] > rsi_short:
                ep=closes[i]; pos_amt=notional/ep; balance-=notional*TAKER_FEE
                sl=ep+atr_arr[i]*sl_mult
                tp=ep-atr_arr[i]*(3.0 if tp_mode=="atr_3x" else 2.0)
                entry_bar=i; in_pos=True; direction=-1

    # 연도별 수익률 계산
    all_years = sorted(year_start_bal.keys())
    for idx, yr in enumerate(all_years):
        if idx + 1 < len(all_years):
            next_yr = all_years[idx + 1]
            end_bal = year_start_bal[next_yr]
        else:
            end_bal = balance
        s_bal = year_start_bal[yr]
        if s_bal > 0:
            year_returns[yr] = round((end_bal - s_bal) / s_bal * 100, 1)

    total_trades = long_trades + short_trades
    total_wins   = long_wins + short_wins
    win_rate     = total_wins / total_trades * 100 if total_trades > 0 else 0.0
    long_wr      = long_wins  / long_trades  * 100 if long_trades  > 0 else 0.0
    short_wr     = short_wins / short_trades * 100 if short_trades > 0 else 0.0

    if balance <= 0:
        ret = -100.0
    else:
        ret = (balance - INITIAL_BAL) / INITIAL_BAL * 100

    calmar = ret / mdd if mdd > 0 else 0.0
    n_years = len(year_returns)
    positive_years = sum(1 for v in year_returns.values() if v > 0)

    return {
        "rsi_long":     rsi_long,
        "rsi_short":    rsi_short,
        "sl_mult":      sl_mult,
        "tp_mode":      tp_mode,
        "leverage":     leverage,
        "pos_ratio":    pos_ratio,
        "use_ema200":   use_ema200,
        "return_pct":   round(ret, 2),
        "max_drawdown": round(mdd, 2),
        "calmar":       round(calmar, 2),
        "trades":       total_trades,
        "win_rate":     round(win_rate, 1),
        "long_trades":  long_trades,
        "long_win_rate":  round(long_wr, 1),
        "short_trades": short_trades,
        "short_win_rate": round(short_wr, 1),
        "positive_years": positive_years,
        "total_years":    n_years,
        "pos_year_rate":  round(positive_years / n_years * 100, 1) if n_years > 0 else 0.0,
        **{f"y{yr}": v for yr, v in year_returns.items()},
    }


# ── 워커 ────────────────────────────────────────────────

def worker(coin: str) -> str:
    print(f"[{coin.upper()}] 데이터 로드...", flush=True)
    df_4h = add_indicators(load_4h(coin))
    bh    = BH_FULL[coin]

    first_yr = df_4h["timestamp"].min().year
    last_yr  = df_4h["timestamp"].max().year
    print(f"[{coin.upper()}] {first_yr}~{last_yr} ({len(df_4h):,}봉) 서치 시작...", flush=True)

    combos = list(itertools.product(
        RSI_LONG, RSI_SHORT, SL_MULTS, TP_MODES, LEVERAGES, POS_RATIOS, USE_EMA200,
    ))

    results = []
    for rsi_l, rsi_s, sl_m, tp_m, lev, pos, ema in combos:
        r = run_backtest(df_4h, rsi_l, rsi_s, sl_m, tp_m, lev, pos, ema)
        results.append(r)

    df_res = pd.DataFrame(results)
    df_res["bh_excess"] = df_res["return_pct"] - bh

    # B&H 초과 & MDD<50% 필터 (XRP는 현실적, BTC/ETH는 드물 것)
    df_beat = df_res[
        (df_res["return_pct"] > bh) &
        (df_res["max_drawdown"] < 50) &
        (df_res["trades"] >= 15)
    ].sort_values(["calmar", "return_pct"], ascending=False)

    # 전체 기간 양수 + MDD<60% + 거래수 필터
    df_decent = df_res[
        (df_res["return_pct"] > 0) &
        (df_res["max_drawdown"] < 60) &
        (df_res["trades"] >= 15) &
        (df_res["pos_year_rate"] >= 60)
    ].sort_values("calmar", ascending=False)

    # 전체 저장
    df_all = df_res.sort_values("return_pct", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_all.to_csv(os.path.join(RESULTS_DIR, f"fullgrid_{coin}_all.csv"), index=False)
    df_beat.to_csv(os.path.join(RESULTS_DIR, f"fullgrid_{coin}_beat_bh.csv"), index=False)
    df_decent.to_csv(os.path.join(RESULTS_DIR, f"fullgrid_{coin}_decent.csv"), index=False)

    # 연도 컬럼 추출
    yr_cols = sorted([c for c in df_all.columns if c.startswith("y2")])

    print(f"\n{'='*65}")
    print(f"[{coin.upper()}] {len(results)}조합 완료 | B&H={bh:+.1f}%")
    print(f"  B&H 초과 + MDD<50%: {len(df_beat)}개")
    print(f"  수익플러스 + MDD<60% + 연플러스60%↑: {len(df_decent)}개")

    if len(df_beat) > 0:
        print(f"\n  ── B&H 초과 Top5 ──")
        for _, row in df_beat.head(5).iterrows():
            yr_str = " ".join(f"{c[1:]}:{row[c]:+.0f}%" for c in yr_cols if c in row)
            print(
                f"  return={row['return_pct']:+.1f}%  MDD={row['max_drawdown']:.1f}%  "
                f"Calmar={row['calmar']:.2f}  거래={row['trades']}건  "
                f"RSI_L<{row['rsi_long']} RSI_S>{row['rsi_short']} "
                f"sl×{row['sl_mult']} {row['tp_mode']} "
                f"{row['leverage']}x×{row['pos_ratio']*100:.0f}% "
                f"EMA={'Y' if row['use_ema200'] else 'N'}"
            )
            print(f"    연도별: {yr_str}")

    if len(df_decent) > 0:
        print(f"\n  ── 양수수익 + 연도별안정 Top5 (Calmar순) ──")
        for _, row in df_decent.head(5).iterrows():
            yr_str = " ".join(f"{c[1:]}:{row[c]:+.0f}%" for c in yr_cols if c in row)
            print(
                f"  return={row['return_pct']:+.1f}%  MDD={row['max_drawdown']:.1f}%  "
                f"Calmar={row['calmar']:.2f}  거래={row['trades']}건  "
                f"연플러스={row['positive_years']}/{row['total_years']}년  "
                f"RSI_L<{row['rsi_long']} RSI_S>{row['rsi_short']} "
                f"sl×{row['sl_mult']} {row['tp_mode']} "
                f"{row['leverage']}x×{row['pos_ratio']*100:.0f}% "
                f"EMA={'Y' if row['use_ema200'] else 'N'}"
            )
            print(f"    연도별: {yr_str}")

    if len(df_beat) == 0 and len(df_decent) == 0:
        # 그냥 Top3 보여주기
        print(f"\n  ── 전체 Top3 (return_pct) ──")
        for _, row in df_all.head(3).iterrows():
            yr_str = " ".join(f"{c[1:]}:{row[c]:+.0f}%" for c in yr_cols if c in row)
            print(
                f"  return={row['return_pct']:+.1f}%  MDD={row['max_drawdown']:.1f}%  "
                f"Calmar={row['calmar']:.2f}  거래={row['trades']}건"
            )
            print(f"    연도별: {yr_str}")

    return coin


def main():
    total = len(list(itertools.product(
        RSI_LONG, RSI_SHORT, SL_MULTS, TP_MODES, LEVERAGES, POS_RATIOS, USE_EMA200,
    )))
    print("4H BB+RSI 전체 기간(2017~2026) 그리드 서치")
    print(f"코인: {[c.upper() for c in COINS]}")
    print(f"조합: {total}개/코인 × {len(COINS)}코인 = {total*len(COINS)}개\n")

    with ProcessPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {executor.submit(worker, c): c for c in COINS}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  오류: {e}")

    print("\n완료. 결과 파일:")
    for coin in COINS:
        print(f"  data/results/fullgrid_{coin}_all.csv")
        print(f"  data/results/fullgrid_{coin}_beat_bh.csv")
        print(f"  data/results/fullgrid_{coin}_decent.csv")


if __name__ == "__main__":
    main()
