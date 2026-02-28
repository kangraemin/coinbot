"""멀티코인 포트폴리오 백테스팅 (BTC/ETH/SOL/XRP 1:1:1:1 동일 비중).

전략: prev_close 대비 entry_pct% 하락 시 롱 진입
포트폴리오: 각 코인 25% 동일 배분, 독립 운용 후 포트폴리오 합산

사용법:
    python backtest_multicoin.py
    python backtest_multicoin.py --leverage 7 --pos 20
    python backtest_multicoin.py --entry 1.5 --tp 3.0 --sl 0.5
"""

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TAKER_FEE_RATE = 0.0005
COINS = ["btc", "eth", "sol", "xrp"]
ALLOC = 1.0 / len(COINS)  # 25% each


def load_data(coin: str) -> pd.DataFrame:
    frames = []
    for year in [2022, 2023, 2024, 2025, 2026]:
        path = os.path.join(DATA_DIR, f"{coin}_1m_{year}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"{coin} 데이터 없음: {DATA_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def run_single(df: pd.DataFrame, entry_pct, tp_pct, sl_pct, leverage, pos_ratio, init_equity=1.0):
    """단일 코인 백테스트 — 거래 목록(timestamp + equity 포함) 반환."""
    fee_rate = TAKER_FEE_RATE * leverage * 2
    equity = init_equity
    trades = []
    in_position = False
    entry_price = tp_price = sl_price = 0.0
    entry_idx = 0

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df["timestamp"].values

    for i in range(1, len(df)):
        prev_close = closes[i - 1]
        high = highs[i]
        low = lows[i]
        ts = timestamps[i]

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

        trades.append({
            "coin": None,  # 호출 후 채움
            "ts": ts,
            "entry_ts": timestamps[entry_idx],
            "outcome": outcome,
            "trade_return_pct": trade_return * 100,
            "equity": equity,
        })
        in_position = False

    return trades, equity


def calc_portfolio_stats(all_trades, portfolio_equity_curve, coins_final):
    """포트폴리오 통계 계산."""
    total_return = (portfolio_equity_curve[-1] - 1.0) * 100

    peak = portfolio_equity_curve[0]
    mdd = 0.0
    for eq in portfolio_equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > mdd:
            mdd = dd

    wins = [t for t in all_trades if t["outcome"] == "tp"]
    losses = [t for t in all_trades if t["outcome"] == "sl"]
    total = len(all_trades)
    win_rate = len(wins) / total * 100 if total else 0

    avg_tp = sum(t["trade_return_pct"] for t in wins) / len(wins) if wins else 0
    avg_sl = sum(t["trade_return_pct"] for t in losses) / len(losses) if losses else 0
    expectancy = win_rate / 100 * avg_tp + (1 - win_rate / 100) * avg_sl

    sharpe = total_return / mdd if mdd > 0 else 0

    return {
        "total_trades": total,
        "win_rate": win_rate,
        "avg_tp_pct": avg_tp,
        "avg_sl_pct": avg_sl,
        "expectancy": expectancy,
        "total_return_pct": total_return,
        "mdd_pct": mdd,
        "sharpe": sharpe,
    }


def calc_monthly(all_trades):
    rows = []
    for t in all_trades:
        ts = pd.Timestamp(t["ts"])
        rows.append({"year": ts.year, "month": ts.month, "pnl": t["trade_return_pct"]})
    df = pd.DataFrame(rows)
    monthly = df.groupby(["year", "month"])["pnl"].agg(["sum", "count"]).reset_index()
    monthly.columns = ["연도", "월", "수익(%)", "거래수"]
    monthly["수익(%)"] = monthly["수익(%)"].round(2)
    return monthly


def main():
    parser = argparse.ArgumentParser(description="멀티코인 1:1:1:1 포트폴리오 백테스팅")
    parser.add_argument("--entry", type=float, default=1.5, dest="entry_pct")
    parser.add_argument("--tp", type=float, default=3.0)
    parser.add_argument("--sl", type=float, default=0.5)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--pos", type=float, default=30.0)
    args = parser.parse_args()

    print(f"파라미터: entry={args.entry_pct}%, TP={args.tp}%, SL={args.sl}%, {args.leverage}x, pos={args.pos}%")
    print(f"포트폴리오: {' / '.join(c.upper() for c in COINS)} = 1:1:1:1 (각 {ALLOC*100:.0f}%)\n")

    # 코인별 백테스트
    coin_results = {}
    for coin in COINS:
        print(f"  [{coin.upper()}] 로드 중...", end=" ", flush=True)
        df = load_data(coin)
        print(f"{len(df):,}행", end=" → ", flush=True)
        trades, final_eq = run_single(
            df, args.entry_pct, args.tp, args.sl, args.leverage, args.pos,
            init_equity=ALLOC
        )
        for t in trades:
            t["coin"] = coin
        coin_results[coin] = {"trades": trades, "final_equity": final_eq}
        ret = (final_eq / ALLOC - 1) * 100
        print(f"거래 {len(trades)}건 | 수익률 {ret:+.1f}%")

    # 포트폴리오 합산 equity curve (시간순 정렬)
    # 각 코인 현재 equity 추적
    coin_equity = {c: ALLOC for c in COINS}
    all_trades = []
    for coin, res in coin_results.items():
        all_trades.extend(res["trades"])

    all_trades_sorted = sorted(all_trades, key=lambda x: x["ts"])

    portfolio_equity_curve = [1.0]  # 초기 포트폴리오 = 합 1.0
    for t in all_trades_sorted:
        coin_equity[t["coin"]] = t["equity"]
        portfolio_equity_curve.append(sum(coin_equity.values()))

    # 통계
    stats = calc_portfolio_stats(all_trades_sorted, portfolio_equity_curve, coin_equity)
    monthly = calc_monthly(all_trades_sorted)

    # 코인별 요약
    print("\n" + "=" * 60)
    print("코인별 개별 성과")
    print("=" * 60)
    print(f"{'코인':>6} {'거래수':>6} {'승률':>7} {'수익률':>12} {'최종자산':>10}")
    print("-" * 60)
    for coin in COINS:
        t_list = coin_results[coin]["trades"]
        eq = coin_results[coin]["final_equity"]
        ret = (eq / ALLOC - 1) * 100
        wins = sum(1 for t in t_list if t["outcome"] == "tp")
        wr = wins / len(t_list) * 100 if t_list else 0
        print(f"{coin.upper():>6} {len(t_list):>6}건 {wr:>6.1f}% {ret:>+11.1f}% {eq:>10.4f}")
    print(f"{'합계':>6} {stats['total_trades']:>6}건")

    # 포트폴리오 통계
    print("\n" + "=" * 60)
    print("포트폴리오 통합 성과")
    print("=" * 60)
    print(f"총 거래수     : {stats['total_trades']}건")
    print(f"승률          : {stats['win_rate']:.1f}%")
    print(f"평균 익절     : {stats['avg_tp_pct']:.2f}%")
    print(f"평균 손절     : {stats['avg_sl_pct']:.2f}%")
    print(f"기대값        : {stats['expectancy']:.2f}%")
    print("-" * 60)
    print(f"복리 총수익   : {stats['total_return_pct']:+.2f}%")
    print(f"MDD           : {stats['mdd_pct']:.2f}%")
    print(f"Sharpe        : {stats['sharpe']:.3f}")
    print("=" * 60)

    if not monthly.empty:
        print("\n월별 수익표 (전체 거래 합산):")
        print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
