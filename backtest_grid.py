"""1분봉 파라미터 그리드 서치.

사용법:
    python backtest_grid.py
    python backtest_grid.py --output my_results.csv
"""

import argparse
import os
from itertools import product
from datetime import datetime, timezone

import pandas as pd

DATA_DIR = "data"
TAKER_FEE_RATE = 0.0005

ENTRY_PCTS = [0.3, 0.5, 0.8, 1.0, 1.5]
TP_PCTS = [1.0, 1.5, 2.0, 3.0]
SL_PCTS = [0.5, 0.8, 1.0, 1.5, 2.0]
LEVERAGES = [3, 5, 7, 10]
POS_RATIOS = [10, 20, 30]


def load_data() -> pd.DataFrame:
    frames = []
    for year in [2025, 2026]:
        path = os.path.join(DATA_DIR, f"btc_1m_{year}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def run_backtest(df, entry_pct, tp_pct, sl_pct, leverage, pos_ratio):
    fee_rate = TAKER_FEE_RATE * leverage * 2
    equity = 1.0
    equity_curve = [1.0]
    trades = []
    in_position = False
    entry_price = tp_price = sl_price = 0.0
    entry_idx = 0

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    for i in range(1, len(df)):
        prev_close = closes[i - 1]
        high = highs[i]
        low = lows[i]

        if not in_position:
            trigger = prev_close * (1 - entry_pct / 100)
            if low <= trigger:
                entry_price = trigger
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
                in_position = True
                entry_idx = i
            continue

        hit_tp = high >= tp_price
        hit_sl = low <= sl_price

        if hit_tp and hit_sl:
            exit_price, outcome = sl_price, "sl"
        elif hit_sl:
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

    wins = [t for t in trades if t["outcome"] == "tp"]
    losses = [t for t in trades if t["outcome"] == "sl"]
    win_rate = len(wins) / len(trades) * 100
    avg_tp = sum(t["trade_return_pct"] for t in wins) / len(wins) if wins else 0.0
    avg_sl = sum(t["trade_return_pct"] for t in losses) / len(losses) if losses else 0.0
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
        "entry_pct": entry_pct,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "leverage": leverage,
        "pos_ratio": pos_ratio,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="grid_results.csv")
    args = parser.parse_args()

    print("데이터 로드 중...")
    df = load_data()
    print(f"로드 완료: {len(df):,}행 ({df['timestamp'].iloc[0].date()} ~ {df['timestamp'].iloc[-1].date()})")

    combos = list(product(ENTRY_PCTS, TP_PCTS, SL_PCTS, LEVERAGES, POS_RATIOS))
    total = len(combos)
    print(f"총 {total}개 조합 실행\n")

    results = []
    for idx, (entry_pct, tp_pct, sl_pct, leverage, pos_ratio) in enumerate(combos, 1):
        trades, equity_curve = run_backtest(df, entry_pct, tp_pct, sl_pct, leverage, pos_ratio)
        row = calc_stats(trades, equity_curve, entry_pct, tp_pct, sl_pct, leverage, pos_ratio)
        if row:
            results.append(row)

        if idx % 100 == 0 or idx == total:
            print(f"  [{idx}/{total}] {idx/total*100:.0f}% 완료...")

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("total_return_pct", ascending=False).reset_index(drop=True)
    df_res.to_csv(args.output, index=False)

    print(f"\n결과 저장: {args.output} ({len(df_res)}개 조합)")
    print("\n=== 상위 20개 (수익률 기준) ===")
    top20 = df_res.head(20)[["entry_pct", "tp_pct", "sl_pct", "leverage", "pos_ratio",
                               "total_trades", "win_rate", "total_return_pct", "mdd_pct", "sharpe"]]
    print(top20.to_string(index=False))

    print("\n=== 상위 20개 (Sharpe 기준) ===")
    top_sharpe = df_res.nlargest(20, "sharpe")[["entry_pct", "tp_pct", "sl_pct", "leverage", "pos_ratio",
                                                  "total_trades", "win_rate", "total_return_pct", "mdd_pct", "sharpe"]]
    print(top_sharpe.to_string(index=False))


if __name__ == "__main__":
    main()
