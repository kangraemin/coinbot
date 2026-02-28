"""타임프레임별 파라미터 그리드 서치 (1m/3m/5m/15m 병렬).

1m 데이터를 리샘플링하여 각 타임프레임에서 그리드 서치 후 결과 저장.

사용법:
    python backtest_grid_tf.py                    # BTC, 1m/3m/5m/15m
    python backtest_grid_tf.py --coin eth
    python backtest_grid_tf.py --tfs 1m 5m 15m   # 특정 타임프레임만
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TAKER_FEE_RATE = 0.0005

ALL_TIMEFRAMES = ["1m", "3m", "5m", "15m"]

ENTRY_PCTS = [0.3, 0.5, 0.8, 1.0, 1.5]
TP_PCTS    = [1.0, 1.5, 2.0, 3.0]
SL_PCTS    = [0.5, 0.8, 1.0, 1.5, 2.0]
LEVERAGES  = [3, 5, 7, 10]
POS_RATIOS = [10, 20, 30]


# ── 데이터 로드 & 리샘플링 ──────────────────────────────────────────────────

def load_1m(coin: str) -> pd.DataFrame:
    frames = []
    for year in [2022, 2023, 2024, 2025, 2026]:
        path = os.path.join(DATA_DIR, f"{coin}_1m_{year}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"{coin} 1m 데이터 없음 ({DATA_DIR})")
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "1m":
        return df
    rule = tf.replace("m", "min")
    df2 = (
        df.set_index("timestamp")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
        .reset_index()
    )
    return df2


# ── 백테스트 로직 (backtest_grid.py 동일) ──────────────────────────────────

def run_backtest(df, entry_pct, tp_pct, sl_pct, leverage, pos_ratio):
    fee_rate = TAKER_FEE_RATE * leverage * 2
    equity = 1.0
    equity_curve = [1.0]
    trades = []
    in_position = False
    entry_price = tp_price = sl_price = 0.0

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    for i in range(1, len(df)):
        prev_close = closes[i - 1]
        high = highs[i]
        low  = lows[i]

        if not in_position:
            trigger = prev_close * (1 - entry_pct / 100)
            if low <= trigger:
                entry_price = trigger
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
                in_position = True
            continue

        hit_tp = high >= tp_price
        hit_sl = low  <= sl_price

        if hit_sl or (hit_tp and hit_sl):
            exit_price, outcome = sl_price, "sl"
        elif hit_tp:
            exit_price, outcome = tp_price, "tp"
        else:
            continue

        price_return = (exit_price - entry_price) / entry_price
        trade_return = price_return * leverage - fee_rate
        equity *= 1 + trade_return * pos_ratio / 100
        equity_curve.append(equity)
        trades.append({"outcome": outcome, "trade_return_pct": trade_return * 100})
        in_position = False

    return trades, equity_curve


def calc_stats(trades, equity_curve, entry_pct, tp_pct, sl_pct, leverage, pos_ratio):
    if not trades:
        return None
    wins   = [t for t in trades if t["outcome"] == "tp"]
    losses = [t for t in trades if t["outcome"] == "sl"]
    win_rate  = len(wins) / len(trades) * 100
    avg_tp    = sum(t["trade_return_pct"] for t in wins)   / len(wins)   if wins   else 0.0
    avg_sl    = sum(t["trade_return_pct"] for t in losses) / len(losses) if losses else 0.0
    expectancy = win_rate / 100 * avg_tp + (1 - win_rate / 100) * avg_sl
    total_return = (equity_curve[-1] - 1) * 100

    peak = mdd = 0.0
    peak = equity_curve[0]
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > mdd:
            mdd = dd

    max_consec = cur_consec = 0
    for t in trades:
        if t["outcome"] == "sl":
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    sharpe = total_return / mdd if mdd > 0 else 0.0

    return {
        "entry_pct": entry_pct, "tp_pct": tp_pct, "sl_pct": sl_pct,
        "leverage": leverage, "pos_ratio": pos_ratio,
        "rr_ratio": round(tp_pct / sl_pct, 2),
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "avg_tp_pct": round(avg_tp, 2),
        "avg_sl_pct": round(avg_sl, 2),
        "expectancy": round(expectancy, 3),
        "total_return_pct": round(total_return, 2),
        "mdd_pct": round(mdd, 2),
        "max_consec_losses": max_consec,
        "sharpe": round(sharpe, 3),
    }


# ── 워커 (프로세스별 실행) ──────────────────────────────────────────────────

def worker(coin: str, tf: str) -> dict:
    """단일 (coin, tf) 조합의 전체 그리드 서치 실행."""
    df_1m = load_1m(coin)
    df    = resample(df_1m, tf)

    combos = list(product(ENTRY_PCTS, TP_PCTS, SL_PCTS, LEVERAGES, POS_RATIOS))
    results = []
    for entry_pct, tp_pct, sl_pct, leverage, pos_ratio in combos:
        trades, eq = run_backtest(df, entry_pct, tp_pct, sl_pct, leverage, pos_ratio)
        row = calc_stats(trades, eq, entry_pct, tp_pct, sl_pct, leverage, pos_ratio)
        if row:
            results.append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = os.path.join(RESULTS_DIR, f"grid_{coin}_{tf}.csv")

    df_res = (
        pd.DataFrame(results)
        .sort_values("total_return_pct", ascending=False)
        .reset_index(drop=True)
    )
    df_res.to_csv(output, index=False)

    best = df_res.iloc[0] if len(df_res) else None
    date_range = f"{df['timestamp'].iloc[0].date()} ~ {df['timestamp'].iloc[-1].date()}"

    return {
        "coin": coin,
        "tf": tf,
        "rows": len(df),
        "date_range": date_range,
        "n_combos": len(df_res),
        "best_return_pct": round(float(best["total_return_pct"]), 2) if best is not None else None,
        "best_win_rate":   round(float(best["win_rate"]), 1)         if best is not None else None,
        "best_mdd_pct":    round(float(best["mdd_pct"]), 2)          if best is not None else None,
        "best_params": {
            "entry_pct": best["entry_pct"], "tp_pct": best["tp_pct"],
            "sl_pct": best["sl_pct"], "leverage": int(best["leverage"]),
            "pos_ratio": int(best["pos_ratio"]),
        } if best is not None else {},
        "output": output,
    }


# ── 메인 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", default="btc", help="코인 소문자 (기본: btc)")
    parser.add_argument("--tfs", nargs="+", default=ALL_TIMEFRAMES,
                        choices=ALL_TIMEFRAMES, help="타임프레임 목록 (기본: 1m 3m 5m 15m)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tasks = [(args.coin, tf) for tf in args.tfs]
    print(f"[{args.coin.upper()}] 타임프레임 {args.tfs} 그리드 서치 시작 (workers={args.workers})")
    print(f"조합 수: {len(list(product(ENTRY_PCTS, TP_PCTS, SL_PCTS, LEVERAGES, POS_RATIOS)))} × {len(args.tfs)} TF\n")

    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(worker, coin, tf): (coin, tf) for coin, tf in tasks}
        for future in as_completed(future_map):
            coin, tf = future_map[future]
            try:
                result = future.result()
                summaries.append(result)
                print(
                    f"  ✓ {tf:4s}  rows={result['rows']:,}  "
                    f"best_return={result['best_return_pct']:+.1f}%  "
                    f"win={result['best_win_rate']}%  "
                    f"mdd={result['best_mdd_pct']}%  "
                    f"→ {result['output']}"
                )
            except Exception as e:
                print(f"  ✗ {tf}  오류: {e}")

    # ── 타임프레임 비교 요약 ────────────────────────────────────────────────
    summaries.sort(key=lambda x: ALL_TIMEFRAMES.index(x["tf"]))
    print(f"\n{'='*70}")
    print(f"{'TF':6s} {'rows':>8s} {'best_ret%':>10s} {'win%':>6s} {'MDD%':>7s}  best_params")
    print(f"{'-'*70}")
    for s in summaries:
        p = s["best_params"]
        params_str = (
            f"entry={p.get('entry_pct')}% tp={p.get('tp_pct')}% sl={p.get('sl_pct')}% "
            f"{p.get('leverage')}x pos={p.get('pos_ratio')}%"
        ) if p else "N/A"
        print(
            f"{s['tf']:6s} {s['rows']:>8,} {s['best_return_pct']:>+10.1f} "
            f"{s['best_win_rate']:>6.1f} {s['best_mdd_pct']:>7.1f}  {params_str}"
        )
    print(f"{'='*70}")
    print(f"\n결과 파일: {RESULTS_DIR}/grid_{args.coin}_{{tf}}.csv")


if __name__ == "__main__":
    main()
