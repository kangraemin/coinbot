"""자본 손실 기준 그리드 서치 백테스트.

SL 정의:
  기존: entry × (1 - sl_pct / 100)           → 가격 하락 %
  신규: entry × (1 - sl_pct / leverage / 100) → 자본 손실 %
"""

import os
import sys
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE_RATE = 0.0005

# ── 파라미터 그리드 ──────────────────────────────────
ENTRY_PCTS  = [0.5, 1.0, 1.5, 2.0, 3.0]   # 진입: 가격 하락 %
TP_PCTS     = [1.0, 2.0, 3.0, 5.0]         # 익절: 가격 상승 %
SL_CAPITAL  = [1.0, 2.0, 3.0, 5.0, 10.0]  # 손절: 자본 손실 %
LEVERAGES   = [1, 2, 3, 4, 5]
POS_RATIOS  = [0.1, 0.2, 0.3]

COINS = ["btc", "eth", "sol", "xrp"]


def load_data(coin: str) -> pd.DataFrame:
    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.startswith(f"{coin}_1m_") and f.endswith(".parquet")
    ])
    if not files:
        raise FileNotFoundError(f"{coin} 데이터 파일 없음: {DATA_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def run_backtest(df: pd.DataFrame, entry_pct: float, tp_pct: float,
                 sl_capital_pct: float, leverage: int, pos_ratio: float,
                 initial_balance: float = 1000.0) -> dict:
    """단순 1분봉 하락 진입 백테스트 — SL은 자본 손실 기준."""
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    trades = 0
    wins = 0
    total_pnl = 0.0

    # SL 가격 하락률 = 자본 손실% / 레버리지
    sl_price_pct = sl_capital_pct / leverage

    in_position = False
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    position_amount = 0.0

    closes = df["close"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    n = len(closes)

    for i in range(1, n):
        prev_close = closes[i - 1]
        high = highs[i]
        low  = lows[i]

        if not in_position:
            if balance <= 0:
                break
            entry_price = prev_close * (1 - entry_pct / 100)
            if low <= entry_price:
                margin = balance * pos_ratio
                notional = margin * leverage
                position_amount = notional / entry_price
                fee = notional * TAKER_FEE_RATE
                balance -= fee

                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_price_pct / 100)
                in_position = True
        else:
            # SL 먼저 확인 (같은 봉에서 SL+TP 동시 도달 시 SL 우선)
            if low <= sl_price:
                pnl = (sl_price - entry_price) * position_amount
                fee = sl_price * position_amount * TAKER_FEE_RATE
                balance += pnl - fee
                total_pnl += pnl - fee
                trades += 1
                in_position = False
            elif high >= tp_price:
                pnl = (tp_price - entry_price) * position_amount
                fee = tp_price * position_amount * TAKER_FEE_RATE
                balance += pnl - fee
                total_pnl += pnl - fee
                trades += 1
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
    sharpe = (ret / max_drawdown) if max_drawdown > 0 else 0.0

    return {
        "entry_pct":    entry_pct,
        "tp_pct":       tp_pct,
        "sl_capital":   sl_capital_pct,
        "sl_price_pct": round(sl_price_pct, 4),
        "leverage":     leverage,
        "pos_ratio":    pos_ratio,
        "trades":       trades,
        "win_rate":     round(win_rate, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe":       round(sharpe, 2),
    }


def worker(coin: str) -> str:
    print(f"[{coin.upper()}] 시작...", flush=True)
    try:
        df = load_data(coin)
    except Exception as e:
        print(f"[{coin.upper()}] 데이터 로드 실패: {e}", flush=True)
        return f"{coin}: FAILED"

    combos = list(itertools.product(
        ENTRY_PCTS, TP_PCTS, SL_CAPITAL, LEVERAGES, POS_RATIOS
    ))
    results = []
    for entry_pct, tp_pct, sl_capital, leverage, pos_ratio in combos:
        r = run_backtest(df, entry_pct, tp_pct, sl_capital, leverage, pos_ratio)
        results.append(r)

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("return_pct", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"grid_capital_{coin}.csv")
    df_res.to_csv(out_path, index=False)

    best = df_res.iloc[0]
    print(
        f"[{coin.upper()}] 완료 — {len(results)}조합 | "
        f"Best: entry={best['entry_pct']}% tp={best['tp_pct']}% "
        f"sl_cap={best['sl_capital']}% lev={best['leverage']}x "
        f"pos={best['pos_ratio']*100:.0f}% → "
        f"수익률={best['return_pct']:+.1f}% MDD={best['max_drawdown']:.1f}% "
        f"승률={best['win_rate']:.1f}% 거래={best['trades']}건",
        flush=True
    )
    return out_path


def main():
    print(f"총 {len(list(itertools.product(ENTRY_PCTS, TP_PCTS, SL_CAPITAL, LEVERAGES, POS_RATIOS)))}조합 × {len(COINS)}코인 = "
          f"{len(list(itertools.product(ENTRY_PCTS, TP_PCTS, SL_CAPITAL, LEVERAGES, POS_RATIOS))) * len(COINS)}개 백테스트")
    print("SL = 자본 손실 기준 (sl_price_pct = sl_capital / leverage)\n")

    with ProcessPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {executor.submit(worker, coin): coin for coin in COINS}
        for fut in as_completed(futures):
            coin = futures[fut]
            try:
                result = fut.result()
                print(f"  저장: {result}")
            except Exception as e:
                print(f"  [{coin.upper()}] 오류: {e}")


if __name__ == "__main__":
    main()
