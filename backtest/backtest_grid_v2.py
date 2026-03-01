"""동적 그리드 트레이딩 백테스트 — ATR(20)×0.5 간격, 10 그리드, ±15% 범위.

구조:
  - 현재가 기준 아래 10개 매수 레벨 (ATR×0.5 간격)
  - 각 레벨: 자본/10 × 레버리지 크기의 롱 포지션
  - TP: 진입가 + ATR×0.5 (다음 상위 그리드)
  - 리셋: 가격이 최저 그리드 아래로 이탈 시 전체 청산 후 재설정

코인: BTC, ETH, XRP | 기간: 2022~2025
"""

import os

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0
START_DATE  = "2022-01-01"
END_DATE    = "2025-12-31"

N_GRIDS    = 10
LEVERAGES  = [1, 2, 3]
COINS      = ["btc", "eth", "xrp"]
BH_REF     = {"btc": 89.4, "eth": -19.4, "xrp": 121.6}


# ── 데이터 준비 ───────────────────────────────────────

def load_range(coin: str) -> pd.DataFrame:
    frames = []
    for y in range(2021, 2026):
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
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last"),
        volume=("volume","sum"),
    ).dropna().reset_index()


def compute_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    c  = df["close"].astype(float)
    h  = df["high"].astype(float)
    lo = df["low"].astype(float)
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ── 백테스트 코어 ─────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    leverage: int,
    n_grids: int = N_GRIDS,
    initial_balance: float = INITIAL_BAL,
) -> dict:
    closes  = df["close"].to_numpy(float)
    highs   = df["high"].to_numpy(float)
    lows    = df["low"].to_numpy(float)
    atr_arr = compute_atr(df).to_numpy(float)
    n = len(closes)

    balance = initial_balance
    peak    = initial_balance
    mdd = 0.0

    capital_per_grid = initial_balance / n_grids

    # 초기 그리드 설정
    def make_grid(center, atr_val):
        spacing = atr_val * 0.5
        # n_grids 개 매수 레벨: center 아래로 내려가며 배치
        buy_levels = [center - (k + 1) * spacing for k in range(n_grids)]
        tp_levels  = [lvl + spacing for lvl in buy_levels]
        return buy_levels, tp_levels, spacing

    # 그리드 상태
    grid_buy   = []   # 각 레벨의 매수가
    grid_tp    = []   # 각 레벨의 TP가
    grid_pos   = []   # pos_amt (0이면 미진입)
    grid_spacing = 0.0

    def reset_grid(center_price, atr_val):
        nonlocal grid_buy, grid_tp, grid_pos, grid_spacing
        buy_levels, tp_levels, spacing = make_grid(center_price, atr_val)
        grid_buy   = buy_levels
        grid_tp    = tp_levels
        grid_pos   = [0.0] * n_grids
        grid_spacing = spacing

    trades = wins = 0
    port_vals = np.empty(n)

    for i in range(n):
        if np.isnan(atr_arr[i]):
            unrealized = sum((closes[i] - grid_buy[k]) * grid_pos[k]
                             for k in range(len(grid_buy)) if grid_pos[k] > 0)
            port_vals[i] = balance + unrealized
            continue

        # 첫 번째 유효 바에서 그리드 초기화
        if not grid_buy:
            reset_grid(closes[i], atr_arr[i])
            port_vals[i] = balance
            continue

        c_prev = closes[i - 1] if i > 0 else closes[i]

        # ① TP 체크 (가격 상승 → TP 도달)
        for k in range(n_grids):
            if grid_pos[k] > 0 and highs[i] >= grid_tp[k]:
                exit_p = grid_tp[k]
                pnl    = (exit_p - grid_buy[k]) * grid_pos[k]
                fee    = exit_p * grid_pos[k] * TAKER_FEE
                balance += pnl - fee
                grid_pos[k] = 0.0
                trades += 1
                wins   += 1

        # ② 매수 체크 (가격 하락 → 그리드 레벨 도달)
        for k in range(n_grids):
            if grid_pos[k] == 0 and lows[i] <= grid_buy[k] < c_prev:
                notional    = capital_per_grid * leverage
                pos_amt     = notional / grid_buy[k]
                balance    -= notional * TAKER_FEE
                grid_pos[k] = pos_amt

        # ③ 리셋 체크: 가격이 최저 그리드 아래로 이탈
        min_level = min(grid_buy)
        if closes[i] < min_level:
            # 전체 포지션 강제 청산 (손절)
            for k in range(n_grids):
                if grid_pos[k] > 0:
                    pnl     = (closes[i] - grid_buy[k]) * grid_pos[k]
                    fee     = closes[i] * grid_pos[k] * TAKER_FEE
                    balance += pnl - fee
                    grid_pos[k] = 0.0
                    trades  += 1
                    # 손절이므로 win 아님
            reset_grid(closes[i], atr_arr[i])

        # 포트폴리오 가치
        unrealized  = sum((closes[i] - grid_buy[k]) * grid_pos[k]
                          for k in range(n_grids) if grid_pos[k] > 0)
        port_val    = balance + unrealized
        port_vals[i] = port_val

        if port_val > peak:
            peak = port_val
        dd = (peak - port_val) / peak * 100
        if dd > mdd:
            mdd = dd

    final_val = port_vals[-1] if n > 0 else initial_balance
    ret    = (final_val - initial_balance) / initial_balance * 100
    wr     = wins / trades * 100 if trades > 0 else 0.0
    calmar = ret / mdd if mdd > 0 else 0.0

    return {
        "leverage":     leverage,
        "trades":       trades,
        "trades_per_yr": round(trades / 4, 1),
        "win_rate":     round(wr, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(mdd, 2),
        "calmar":       round(calmar, 2),
    }


# ── 메인 ──────────────────────────────────────────────

def main():
    print("동적 그리드 트레이딩 — BTC/ETH/XRP, 2022~2025")
    print(f"그리드: {N_GRIDS}개 | 간격: ATR(20)×0.5 | 범위: ±최저그리드 이탈 시 리셋")
    print(f"레버리지: {LEVERAGES}\n")

    all_rows = []
    print(f"  {'코인':5} {'레버리지':>5} {'수익률':>9} {'MDD':>7} {'Calmar':>8} "
          f"{'거래/년':>7} {'승률':>7} {'B&H':>9} {'판정':>10}")
    print(f"  {'-'*72}")

    for coin in COINS:
        df_1m = load_range(coin)
        df_1d = resample_1d(df_1m)

        s = pd.Timestamp(START_DATE, tz="UTC")
        e = pd.Timestamp(END_DATE + " 23:59:59", tz="UTC")
        df_eval = df_1d[(df_1d["timestamp"] >= s) & (df_1d["timestamp"] <= e)].reset_index(drop=True)

        bh = BH_REF[coin]
        for lev in LEVERAGES:
            r    = run_backtest(df_eval, lev)
            beat = "✓ B&H 초과" if r["return_pct"] > bh else "✗ B&H 미달"
            print(
                f"  {coin.upper():5} {lev:>5}x "
                f"{r['return_pct']:>+8.1f}% "
                f"{r['max_drawdown']:>6.1f}% "
                f"{r['calmar']:>8.2f} "
                f"{r['trades_per_yr']:>6.1f}/년 "
                f"{r['win_rate']:>6.1f}% "
                f"{bh:>+8.1f}% "
                f"{beat:>10}"
            )
            row = {"coin": coin.upper(), **r, "bh_ref": bh}
            all_rows.append(row)

        print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "grid_v2_results.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
