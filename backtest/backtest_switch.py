"""레짐 스위칭 전략 백테스트.

현행 BB+RSI vs 스위칭 전략 비교.

스위칭 전략:
  불장 (close > EMA200): EMA50 추세추종 롱 (close > EMA50 진입, EMA50 이탈 청산)
  약세/중립 (close < EMA200): BB+RSI 숏 전용 (현행과 동일)

비교 기준: 코인별 B&H, 현행 BB+RSI
데이터: {coin}_1h_full.parquet → 4H 리샘플 (2017~2026)
"""

import os
import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0

PARAMS = {
    "btc": dict(rsi_long=30, rsi_short=65, sl_mult=2.0, tp_mult=3.0, leverage=3, pos_ratio=0.70),
    "eth": dict(rsi_long=25, rsi_short=65, sl_mult=2.0, tp_mult=2.0, leverage=3, pos_ratio=0.70),
    "xrp": dict(rsi_long=25, rsi_short=65, sl_mult=2.0, tp_mult=3.0, leverage=3, pos_ratio=0.70),
}
COINS = ["btc", "eth", "xrp"]


# ── 데이터 ────────────────────────────────────────────

def load_4h(coin: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{coin}_1h_full.parquet")
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df.resample("4h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),    close=("close","last"),
        volume=("volume","sum"),
    ).dropna().reset_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c  = df["close"].astype(float)
    h  = df["high"].astype(float)
    lo = df["low"].astype(float)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std(ddof=1)

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    tr    = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr   = tr.ewm(com=13, adjust=False).mean()

    ema200 = c.ewm(span=200, adjust=False).mean()
    ema50  = c.ewm(span=50,  adjust=False).mean()

    df = df.copy()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["rsi"]      = rsi
    df["atr"]      = atr
    df["ema200"]   = ema200
    df["ema50"]    = ema50
    return df


# ── 백테스트 코어 ─────────────────────────────────────

def _run(
    closes, highs, lows, bb_upper, bb_lower,
    rsi_arr, atr_arr, ema200, ema50,
    p: dict, mode: str, timeout_bars: int = 48
) -> dict:
    """mode: 'bbrsi' | 'switch'"""
    n = len(closes)
    balance = INITIAL_BAL
    peak    = INITIAL_BAL
    mdd     = 0.0
    in_pos  = False
    direction = 0
    ep = sl = tp = pos_amt = 0.0
    entry_bar = 0
    pos_mode  = ""   # 포지션이 어떤 모드로 진입했는지

    bbrsi_t = bbrsi_w = 0
    trend_t = trend_w = 0

    for i in range(1, n):
        if any(np.isnan(x[i]) for x in [bb_upper, bb_lower, rsi_arr, atr_arr, ema200, ema50]):
            continue

        is_bull = closes[i] > ema200[i]

        # ── 청산 ──────────────────────────────────────
        if in_pos:
            exit_p = None

            if pos_mode == "trend":
                # 추세추종: EMA50 이탈 시 청산 (또는 SL)
                if direction == 1:
                    if lows[i] <= sl:
                        exit_p = sl
                    elif closes[i] < ema50[i]:
                        exit_p = closes[i]
                        trend_w += int(exit_p > ep)
                    elif i - entry_bar >= timeout_bars * 6:   # 최대 24일 홀드
                        exit_p = closes[i]
                        trend_w += int(exit_p > ep)
            else:
                # BB+RSI: ATR TP/SL
                if direction == 1:
                    if lows[i] <= sl:
                        exit_p = sl
                    elif highs[i] >= tp:
                        exit_p = tp
                        bbrsi_w += 1
                    elif i - entry_bar >= timeout_bars:
                        exit_p = closes[i]
                        bbrsi_w += int(exit_p > ep)
                else:
                    if highs[i] >= sl:
                        exit_p = sl
                    elif lows[i] <= tp:
                        exit_p = tp
                        bbrsi_w += 1
                    elif i - entry_bar >= timeout_bars:
                        exit_p = closes[i]
                        bbrsi_w += int(exit_p < ep)

            if exit_p is not None:
                pnl = (exit_p - ep) * pos_amt if direction == 1 else (ep - exit_p) * pos_amt
                balance += pnl - exit_p * pos_amt * TAKER_FEE
                if pos_mode == "trend":
                    trend_t += 1
                else:
                    bbrsi_t += 1
                in_pos = False

            peak = max(peak, balance)
            mdd  = max(mdd, (peak - balance) / peak * 100)

        # ── 진입 ──────────────────────────────────────
        if not in_pos and balance > 0:
            notional = balance * p["pos_ratio"] * p["leverage"]

            if mode == "bbrsi":
                # ── 현행 BB+RSI ──
                if closes[i] < bb_lower[i] and rsi_arr[i] < p["rsi_long"] and closes[i] > ema200[i]:
                    ep      = closes[i]
                    pos_amt = notional / ep
                    balance -= notional * TAKER_FEE
                    sl = ep - atr_arr[i] * p["sl_mult"]
                    tp = ep + atr_arr[i] * p["tp_mult"]
                    entry_bar = i; in_pos = True; direction = 1; pos_mode = "bbrsi"
                    continue
                if closes[i] > bb_upper[i] and rsi_arr[i] > p["rsi_short"] and closes[i] < ema200[i]:
                    ep      = closes[i]
                    pos_amt = notional / ep
                    balance -= notional * TAKER_FEE
                    sl = ep + atr_arr[i] * p["sl_mult"]
                    tp = ep - atr_arr[i] * p["tp_mult"]
                    entry_bar = i; in_pos = True; direction = -1; pos_mode = "bbrsi"

            else:
                # ── 스위칭 전략 ──
                if is_bull:
                    # 불장: EMA50 위 + 직전봉 EMA50 아래 → 크로스오버 진입
                    if closes[i] > ema50[i] and closes[i - 1] <= ema50[i - 1]:
                        ep      = closes[i]
                        pos_amt = notional / ep
                        balance -= notional * TAKER_FEE
                        sl = ep - atr_arr[i] * p["sl_mult"]
                        tp = 0  # EMA50 이탈 청산 (TP 없음)
                        entry_bar = i; in_pos = True; direction = 1; pos_mode = "trend"
                else:
                    # 약세: BB+RSI 숏만
                    if closes[i] > bb_upper[i] and rsi_arr[i] > p["rsi_short"]:
                        ep      = closes[i]
                        pos_amt = notional / ep
                        balance -= notional * TAKER_FEE
                        sl = ep + atr_arr[i] * p["sl_mult"]
                        tp = ep - atr_arr[i] * p["tp_mult"]
                        entry_bar = i; in_pos = True; direction = -1; pos_mode = "bbrsi"

    ret    = -100.0 if balance <= 0 else (balance - INITIAL_BAL) / INITIAL_BAL * 100
    trades = bbrsi_t + trend_t
    wins   = bbrsi_w + trend_w
    wr     = wins / trades * 100 if trades > 0 else 0.0
    calmar = ret / mdd if mdd > 0 else 0.0
    return {
        "return_pct":   round(ret, 2),
        "mdd":          round(mdd, 2),
        "calmar":       round(calmar, 2),
        "trades":       trades,
        "win_rate":     round(wr, 1),
        "bbrsi_trades": bbrsi_t,
        "trend_trades": trend_t,
    }


def run_coin(coin: str):
    print(f"\n{'='*60}")
    print(f"▶ {coin.upper()} 로드 중...", end=" ", flush=True)
    df = add_indicators(load_4h(coin))
    p  = PARAMS[coin]
    print(f"{df['timestamp'].min().year}~{df['timestamp'].max().year} ({len(df):,}봉)")

    closes   = df["close"].to_numpy(float)
    highs    = df["high"].to_numpy(float)
    lows     = df["low"].to_numpy(float)
    bb_upper = df["bb_upper"].to_numpy(float)
    bb_lower = df["bb_lower"].to_numpy(float)
    rsi_arr  = df["rsi"].to_numpy(float)
    atr_arr  = df["atr"].to_numpy(float)
    ema200   = df["ema200"].to_numpy(float)
    ema50    = df["ema50"].to_numpy(float)

    args = (closes, highs, lows, bb_upper, bb_lower, rsi_arr, atr_arr, ema200, ema50, p)

    r_bb  = _run(*args, mode="bbrsi")
    r_sw  = _run(*args, mode="switch")
    bh    = round((closes[-1] - closes[0]) / closes[0] * 100, 1)

    print(f"\n  전체 기간 ({df['timestamp'].min().year}~{df['timestamp'].max().year})")
    print(f"  {'전략':14} {'수익':>9} {'MDD':>7} {'Calmar':>7} {'거래':>5} {'승률':>6} {'BB거래':>5} {'추세거래':>6}")
    print(f"  {'-'*65}")
    for name, r in [("현행 BB+RSI", r_bb), ("스위칭", r_sw)]:
        print(f"  {name:14} {r['return_pct']:>+8.1f}% {r['mdd']:>6.1f}% {r['calmar']:>7.2f}"
              f" {r['trades']:>5}건 {r['win_rate']:>5.1f}% {r['bbrsi_trades']:>5}건 {r['trend_trades']:>6}건")
    print(f"  {'B&H':14} {bh:>+8.1f}%")

    # 연도별
    years = sorted(df["timestamp"].dt.year.unique())
    print(f"\n  연도별 비교:")
    print(f"  {'연도':6} {'BB+RSI':>9} {'스위칭':>9} {'B&H':>9} {'우위':>6}")
    print(f"  {'-'*45}")

    rows = []
    for yr in years:
        s   = pd.Timestamp(f"{yr}-01-01", tz="UTC")
        e   = pd.Timestamp(f"{yr}-12-31 23:59:59", tz="UTC")
        df_y = df[(df["timestamp"] >= s) & (df["timestamp"] <= e)].reset_index(drop=True)
        if len(df_y) < 20:
            continue

        c2 = df_y["close"].to_numpy(float)
        h2 = df_y["high"].to_numpy(float)
        l2 = df_y["low"].to_numpy(float)
        bu = df_y["bb_upper"].to_numpy(float)
        bl = df_y["bb_lower"].to_numpy(float)
        ri = df_y["rsi"].to_numpy(float)
        at = df_y["atr"].to_numpy(float)
        e2 = df_y["ema200"].to_numpy(float)
        e5 = df_y["ema50"].to_numpy(float)

        a2  = (c2, h2, l2, bu, bl, ri, at, e2, e5, p)
        rb  = _run(*a2, mode="bbrsi")["return_pct"]
        rs  = _run(*a2, mode="switch")["return_pct"]
        bhy = round((c2[-1] - c2[0]) / c2[0] * 100, 1)
        ytd = "(YTD)" if yr == 2026 else ""
        best = "SW" if rs > rb else "BB"
        emoji = "✅" if rs > rb else "  "
        print(f"  {yr}{ytd:6} {rb:>+8.1f}% {rs:>+8.1f}% {bhy:>+8.1f}% {emoji}{best:>4}")
        rows.append({"coin": coin.upper(), "year": str(yr) + ytd,
                     "bbrsi": rb, "switch": rs, "bh": bhy})

    return rows, r_bb, r_sw, bh


def main():
    print("레짐 스위칭 전략 백테스트")
    print("불장(close>EMA200): EMA50 크로스오버 롱 | 약세: BB+RSI 숏")
    print("비교: 현행 BB+RSI vs 스위칭\n")

    all_rows = []
    summary  = []

    for coin in COINS:
        rows, r_bb, r_sw, bh = run_coin(coin)
        all_rows.extend(rows)
        summary.append({
            "coin":         coin.upper(),
            "bbrsi_ret":    r_bb["return_pct"],
            "bbrsi_mdd":    r_bb["mdd"],
            "bbrsi_calmar": r_bb["calmar"],
            "switch_ret":   r_sw["return_pct"],
            "switch_mdd":   r_sw["mdd"],
            "switch_calmar":r_sw["calmar"],
            "bh":           bh,
        })

    print(f"\n\n{'='*60}")
    print("전체 요약")
    print(f"{'코인':6} {'BB수익':>9} {'BB MDD':>7} {'SW수익':>9} {'SW MDD':>7} {'B&H':>9} {'SW 우위':>8}")
    print(f"{'-'*60}")
    for s in summary:
        better = "✅" if s["switch_ret"] > s["bbrsi_ret"] else "❌"
        diff = s["switch_ret"] - s["bbrsi_ret"]
        print(f"{s['coin']:6} {s['bbrsi_ret']:>+8.1f}% {s['bbrsi_mdd']:>6.1f}% "
              f"{s['switch_ret']:>+8.1f}% {s['switch_mdd']:>6.1f}% {s['bh']:>+8.1f}% "
              f"{better} {diff:>+6.1f}%p")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(os.path.join(RESULTS_DIR, "switch_yearly.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, "switch_summary.csv"), index=False)
    print(f"\n저장: {RESULTS_DIR}/switch_*.csv")


if __name__ == "__main__":
    main()
