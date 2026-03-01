"""SMA 골든/데스크로스 백테스트 — 교차 다음 봉 시가 진입.

신호:
  골든크로스: SMA(fast) > SMA(slow)로 전환 → 다음 봉 시가에 롱 진입
  데스크로스: SMA(fast) < SMA(slow)로 전환 → 다음 봉 시가에 청산 (long_only) / 숏 (long_short)

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

SMA_FAST   = [20, 50, 100]
SMA_SLOW   = [150, 200]
MODES      = ["long_only", "long_short"]
LEVERAGES  = [1, 2, 3]
POS_RATIOS = [0.5, 1.0]

COINS  = ["btc", "eth", "xrp"]
BH_REF = {"btc": 89.4, "eth": -19.4, "xrp": 121.6}


# ── 데이터 준비 ───────────────────────────────────────

def load_range(coin: str) -> pd.DataFrame:
    frames = []
    for y in range(2020, 2026):   # SMA200 웜업을 위해 2020부터 로드
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
    """전체 df에 SMA 컬럼 추가 (히스토리 보존)."""
    df = df.copy()
    c = df["close"]
    df["sma_fast"] = c.rolling(fast, min_periods=fast).mean()
    df["sma_slow"] = c.rolling(slow, min_periods=slow).mean()
    return df


# ── 백테스트 코어 ─────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    mode: str,
    leverage: int,
    pos_ratio: float,
    initial_balance: float = INITIAL_BAL,
) -> dict:
    """df에 sma_fast, sma_slow, open, close 컬럼이 있어야 함."""
    closes = df["close"].to_numpy(float)
    opens  = df["open"].to_numpy(float)
    sma_f  = df["sma_fast"].to_numpy(float)
    sma_s  = df["sma_slow"].to_numpy(float)
    n = len(closes)

    balance = initial_balance
    peak    = initial_balance
    mdd     = 0.0

    in_pos    = False
    direction = 0
    ep = pos_amt = 0.0

    long_trades = long_wins = 0
    short_trades = short_wins = 0

    signal_next = None  # "golden" or "death" — 다음 봉 시가 실행

    for i in range(1, n):
        if (np.isnan(sma_f[i]) or np.isnan(sma_s[i])
                or np.isnan(sma_f[i-1]) or np.isnan(sma_s[i-1])):
            signal_next = None
            continue

        # ① 이전 봉 신호를 오늘 시가에 실행
        if signal_next == "golden":
            # 숏 포지션 먼저 청산
            if in_pos and direction == -1:
                exit_p = opens[i]
                pnl = (ep - exit_p) * pos_amt
                short_trades += 1
                if pnl > 0:
                    short_wins += 1
                balance += pnl - exit_p * pos_amt * TAKER_FEE
                in_pos = False
            # 롱 진입
            if not in_pos and balance > 0:
                notional  = balance * pos_ratio * leverage
                pos_amt   = notional / opens[i]
                balance  -= notional * TAKER_FEE
                ep        = opens[i]
                in_pos    = True
                direction = 1

        elif signal_next == "death":
            # 롱 포지션 청산
            if in_pos and direction == 1:
                exit_p = opens[i]
                pnl = (exit_p - ep) * pos_amt
                long_trades += 1
                if pnl > 0:
                    long_wins += 1
                balance += pnl - exit_p * pos_amt * TAKER_FEE
                in_pos = False
            # 숏 진입 (long_short 모드)
            if mode == "long_short" and not in_pos and balance > 0:
                notional  = balance * pos_ratio * leverage
                pos_amt   = notional / opens[i]
                balance  -= notional * TAKER_FEE
                ep        = opens[i]
                in_pos    = True
                direction = -1

        signal_next = None

        # ② 오늘 종가 기준 크로스오버 감지
        golden = (sma_f[i-1] < sma_s[i-1]) and (sma_f[i] >= sma_s[i])
        death  = (sma_f[i-1] > sma_s[i-1]) and (sma_f[i] <= sma_s[i])

        if golden and (not in_pos or direction == -1):
            signal_next = "golden"
        elif death and (in_pos or (mode == "long_short" and not in_pos)):
            signal_next = "death"

        # ③ 포트폴리오 가치
        if in_pos:
            unrealized = (closes[i] - ep) * pos_amt if direction == 1 else (ep - closes[i]) * pos_amt
            pv = balance + unrealized
        else:
            pv = balance

        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak * 100
        if dd > mdd:
            mdd = dd

    # 미청산 포지션 강제 청산
    if in_pos:
        exit_p = closes[-1]
        if direction == 1:
            pnl = (exit_p - ep) * pos_amt
            long_trades += 1
            if pnl > 0:
                long_wins += 1
        else:
            pnl = (ep - exit_p) * pos_amt
            short_trades += 1
            if pnl > 0:
                short_wins += 1
        balance += pnl - exit_p * pos_amt * TAKER_FEE

    ret    = (balance - initial_balance) / initial_balance * 100 if balance > 0 else -100.0
    total  = long_trades + short_trades
    lwr    = long_wins  / long_trades  * 100 if long_trades  > 0 else 0.0
    swr    = short_wins / short_trades * 100 if short_trades > 0 else 0.0
    wr     = (long_wins + short_wins)  / total * 100 if total > 0 else 0.0
    calmar = ret / mdd if mdd > 0 else 0.0

    return {
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
    print(f"[{coin.upper()} 골든크로스] 시작...", flush=True)

    df_1m = load_range(coin)
    df_1d = resample_1d(df_1m)

    s = pd.Timestamp(START_DATE, tz="UTC")
    e = pd.Timestamp(END_DATE + " 23:59:59", tz="UTC")

    valid_pairs = [(sf, ss) for sf, ss in itertools.product(SMA_FAST, SMA_SLOW) if sf < ss]

    results = []
    for sf, ss in valid_pairs:
        df_sma  = add_sma(df_1d, sf, ss)            # 전체 기간으로 SMA 계산
        df_eval = df_sma[(df_sma["timestamp"] >= s) & (df_sma["timestamp"] <= e)].reset_index(drop=True)

        for mode, lev, pos in itertools.product(MODES, LEVERAGES, POS_RATIOS):
            r = run_backtest(df_eval, mode, lev, pos)
            r["sma_fast"] = sf
            r["sma_slow"] = ss
            results.append(r)

    df_res = pd.DataFrame(results).sort_values("return_pct", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"golden_cross_{coin}.csv")
    df_res.to_csv(out_path, index=False)

    bh   = BH_REF[coin]
    best = df_res.iloc[0]
    beat = "✓ B&H 초과" if best["return_pct"] > bh else "✗ B&H 미달"

    # 1x 100% 결과
    df_1x   = df_res[(df_res["leverage"] == 1) & (df_res["pos_ratio"] == 1.0)]
    best_1x = df_1x.iloc[0] if len(df_1x) > 0 else None

    print(
        f"[{coin.upper()} 골든크로스] 완료 — {len(results)}조합 | "
        f"Best: 수익={best['return_pct']:+.1f}%  MDD={best['max_drawdown']:.1f}%  "
        f"Calmar={best['calmar']:.2f}  거래={best['trades']}건 | "
        f"SMA{best['sma_fast']}/{best['sma_slow']} {best['mode']} "
        f"lev={best['leverage']}x pos={best['pos_ratio']*100:.0f}% | "
        f"B&H={bh:+.1f}% → {beat}",
        flush=True,
    )
    if best_1x is not None:
        beat_1x = "✓" if best_1x["return_pct"] > bh else "✗"
        print(
            f"  [1x 100%] Best: 수익={best_1x['return_pct']:+.1f}%  "
            f"MDD={best_1x['max_drawdown']:.1f}%  거래={best_1x['trades']}건  "
            f"SMA{best_1x['sma_fast']}/{best_1x['sma_slow']} {best_1x['mode']} → {beat_1x}",
            flush=True,
        )

    return out_path


def run_yearly(coin: str, df_1d: pd.DataFrame) -> list:
    """Best 파라미터로 연도별 성과 계산."""
    csv    = os.path.join(RESULTS_DIR, f"golden_cross_{coin}.csv")
    df_res = pd.read_csv(csv)
    best   = df_res.iloc[0]
    sf     = int(best["sma_fast"])
    ss     = int(best["sma_slow"])
    mode   = best["mode"]
    lev    = int(best["leverage"])
    pos    = float(best["pos_ratio"])

    df_sma = add_sma(df_1d, sf, ss)  # 전체 히스토리 기반 SMA

    rows = []
    print(f"\n[{coin.upper()}] Best: SMA{sf}/{ss} {mode} {lev}x {pos*100:.0f}%")
    print(f"  {'연도':6} {'수익률':>9} {'MDD':>7} {'Calmar':>8} {'롱건':>5} {'숏건':>5}")
    print(f"  {'-'*45}")

    for year in range(2022, 2027):
        s = pd.Timestamp(f"{year}-01-01", tz="UTC")
        e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
        df_y = df_sma[(df_sma["timestamp"] >= s) & (df_sma["timestamp"] <= e)].reset_index(drop=True)
        if len(df_y) < 10:
            continue
        r = run_backtest(df_y, mode, lev, pos)
        suffix = " (YTD)" if year == 2026 else ""
        print(
            f"  {year}{suffix:6} {r['return_pct']:>+8.1f}% "
            f"{r['max_drawdown']:>6.1f}%  {r['calmar']:>7.2f} "
            f"{r['long_trades']:>5}건 {r['short_trades']:>5}건"
        )
        rows.append({
            "coin": coin.upper(), "year": year,
            "sma_fast": sf, "sma_slow": ss, "mode": mode,
            "leverage": lev, "pos_ratio": pos,
            **{k: r[k] for k in ["return_pct", "max_drawdown", "calmar",
                                   "trades", "win_rate", "long_trades", "short_trades"]},
        })
    return rows


# ── 메인 ──────────────────────────────────────────────

def main():
    n_pairs = sum(1 for sf, ss in itertools.product(SMA_FAST, SMA_SLOW) if sf < ss)
    total   = n_pairs * len(MODES) * len(LEVERAGES) * len(POS_RATIOS)
    print("SMA 골든/데스크로스 — BTC/ETH/XRP, 2022~2025")
    print(f"조합: {total}개 × {len(COINS)}코인 = {total * len(COINS)}개\n")

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
        df_1m = load_range(coin)
        df_1d = resample_1d(df_1m)
        all_rows.extend(run_yearly(coin, df_1d))

    out_rob = os.path.join(RESULTS_DIR, "golden_cross_robustness.csv")
    pd.DataFrame(all_rows).to_csv(out_rob, index=False)
    print(f"\n저장: {out_rob}")


if __name__ == "__main__":
    main()
