"""레짐 스위칭 전략 백테스트 — BTC EMA200 글로벌 레짐.

스위칭 전략:
  불장 (BTC close > BTC EMA200): EMA50 추세추종 롱
  약세/중립 (BTC close < BTC EMA200): BB+RSI 숏 전용

XRP도 BTC 레짐 기준으로 판단. 기존 코인별 EMA200과 비교.
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


def add_indicators(df: pd.DataFrame, btc_ema200: pd.Series = None) -> pd.DataFrame:
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

    ema200_self = c.ewm(span=200, adjust=False).mean()
    ema50       = c.ewm(span=50,  adjust=False).mean()

    df = df.copy()
    df["bb_upper"]    = bb_mid + 2 * bb_std
    df["bb_lower"]    = bb_mid - 2 * bb_std
    df["rsi"]         = rsi
    df["atr"]         = atr
    df["ema200_self"] = ema200_self   # 코인 자체 EMA200
    df["ema50"]       = ema50

    # BTC EMA200을 외부에서 받으면 글로벌 레짐으로 사용
    if btc_ema200 is not None:
        df["ema200"] = btc_ema200.reindex(df["timestamp"]).values
    else:
        df["ema200"] = ema200_self

    return df


def _run(
    closes, highs, lows, bb_upper, bb_lower,
    rsi_arr, atr_arr, ema200, ema50,
    p: dict, mode: str, timeout_bars: int = 48
) -> dict:
    n = len(closes)
    balance = INITIAL_BAL
    peak    = INITIAL_BAL
    mdd     = 0.0
    in_pos  = False
    direction = 0
    ep = sl = tp = pos_amt = 0.0
    entry_bar = 0
    pos_mode  = ""

    bbrsi_t = bbrsi_w = 0
    trend_t = trend_w = 0

    for i in range(1, n):
        if any(np.isnan(x[i]) for x in [bb_upper, bb_lower, rsi_arr, atr_arr, ema200, ema50]):
            continue

        is_bull = closes[i] > ema200[i]

        if in_pos:
            exit_p = None

            if pos_mode == "trend":
                if direction == 1:
                    if lows[i] <= sl:
                        exit_p = sl
                    elif closes[i] < ema50[i]:
                        exit_p = closes[i]
                        trend_w += int(exit_p > ep)
                    elif i - entry_bar >= timeout_bars * 6:
                        exit_p = closes[i]
                        trend_w += int(exit_p > ep)
            else:
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

        if not in_pos and balance > 0:
            notional = balance * p["pos_ratio"] * p["leverage"]

            if mode == "bbrsi":
                if closes[i] < bb_lower[i] and rsi_arr[i] < p["rsi_long"] and closes[i] > ema200[i]:
                    ep = closes[i]; pos_amt = notional / ep
                    balance -= notional * TAKER_FEE
                    sl = ep - atr_arr[i] * p["sl_mult"]; tp = ep + atr_arr[i] * p["tp_mult"]
                    entry_bar = i; in_pos = True; direction = 1; pos_mode = "bbrsi"; continue
                if closes[i] > bb_upper[i] and rsi_arr[i] > p["rsi_short"] and closes[i] < ema200[i]:
                    ep = closes[i]; pos_amt = notional / ep
                    balance -= notional * TAKER_FEE
                    sl = ep + atr_arr[i] * p["sl_mult"]; tp = ep - atr_arr[i] * p["tp_mult"]
                    entry_bar = i; in_pos = True; direction = -1; pos_mode = "bbrsi"
            else:
                if is_bull:
                    if closes[i] > ema50[i] and closes[i - 1] <= ema50[i - 1]:
                        ep = closes[i]; pos_amt = notional / ep
                        balance -= notional * TAKER_FEE
                        sl = ep - atr_arr[i] * p["sl_mult"]; tp = 0
                        entry_bar = i; in_pos = True; direction = 1; pos_mode = "trend"
                else:
                    if closes[i] > bb_upper[i] and rsi_arr[i] > p["rsi_short"]:
                        ep = closes[i]; pos_amt = notional / ep
                        balance -= notional * TAKER_FEE
                        sl = ep + atr_arr[i] * p["sl_mult"]; tp = ep - atr_arr[i] * p["tp_mult"]
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


def main():
    print("레짐 스위칭 백테스트 — BTC EMA200 글로벌 레짐")
    print("불장 판단: 코인 자체 EMA200 vs BTC EMA200\n")

    # BTC EMA200을 먼저 계산해 공유
    print("BTC EMA200 로드 중...", end=" ", flush=True)
    btc_raw = load_4h("btc")
    btc_ema200_series = btc_raw["close"].astype(float).ewm(span=200, adjust=False).mean()
    btc_ema200_series.index = btc_raw["timestamp"]
    print(f"{len(btc_raw):,}봉 완료\n")

    all_rows = []

    for coin in COINS:
        print(f"{'='*60}")
        print(f"▶ {coin.upper()}")
        df_self = add_indicators(load_4h(coin))                       # 자체 EMA200
        df_btc  = add_indicators(load_4h(coin), btc_ema200=btc_ema200_series)  # BTC EMA200

        p = PARAMS[coin]

        def arrays(df):
            return (
                df["close"].to_numpy(float), df["high"].to_numpy(float),
                df["low"].to_numpy(float),   df["bb_upper"].to_numpy(float),
                df["bb_lower"].to_numpy(float), df["rsi"].to_numpy(float),
                df["atr"].to_numpy(float),   df["ema200"].to_numpy(float),
                df["ema50"].to_numpy(float),
            )

        r_self = _run(*arrays(df_self), p=p, mode="switch")
        r_btc  = _run(*arrays(df_btc),  p=p, mode="switch")
        bh     = round((df_self["close"].iloc[-1] - df_self["close"].iloc[0]) / df_self["close"].iloc[0] * 100, 1)

        print(f"  {'전략':18} {'수익':>9} {'MDD':>7} {'Calmar':>7} {'거래':>5} {'승률':>6}")
        print(f"  {'-'*55}")
        for name, r in [("자체 EMA200 스위칭", r_self), ("BTC EMA200 스위칭", r_btc)]:
            print(f"  {name:18} {r['return_pct']:>+8.1f}% {r['mdd']:>6.1f}% {r['calmar']:>7.2f}"
                  f" {r['trades']:>5}건 {r['win_rate']:>5.1f}%")
        print(f"  {'B&H':18} {bh:>+8.1f}%")

        # 연도별
        years = sorted(df_self["timestamp"].dt.year.unique())
        print(f"\n  {'연도':8} {'자체SW':>9} {'BTC-SW':>9} {'B&H':>9} {'우위':>8}")
        print(f"  {'-'*48}")

        for yr in years:
            s = pd.Timestamp(f"{yr}-01-01", tz="UTC")
            e = pd.Timestamp(f"{yr}-12-31 23:59:59", tz="UTC")

            dy = df_self[(df_self["timestamp"] >= s) & (df_self["timestamp"] <= e)].reset_index(drop=True)
            db = df_btc[ (df_btc["timestamp"]  >= s) & (df_btc["timestamp"]  <= e)].reset_index(drop=True)
            if len(dy) < 20:
                continue

            rs = _run(*arrays(dy), p=p, mode="switch")["return_pct"]
            rb = _run(*arrays(db), p=p, mode="switch")["return_pct"]
            bhy = round((dy["close"].iloc[-1] - dy["close"].iloc[0]) / dy["close"].iloc[0] * 100, 1)
            ytd = "(YTD)" if yr == 2026 else ""

            best_val = max(rs, rb)
            if best_val == rs and best_val == rb:
                marker = "="
            elif rs >= rb:
                marker = "자체"
            else:
                marker = "BTC "
            win = "✅" if max(rs, rb) > bhy else "  "

            print(f"  {yr}{ytd:6} {rs:>+8.1f}% {rb:>+8.1f}% {bhy:>+8.1f}% {win}{marker}")

            all_rows.append({
                "coin": coin.upper(), "year": str(yr) + ytd,
                "self_sw": rs, "btc_sw": rb, "bh": bhy,
            })

        print()

    # 장세별 요약
    df_res = pd.DataFrame(all_rows)

    def regime(bh):
        if bh > 30:   return "상승장"
        if bh < -20:  return "하락장"
        return "횡보장"

    df_res["regime"] = df_res["bh"].apply(regime)

    print(f"\n{'='*60}")
    print("장세별 평균 (전 코인)")
    print(f"  {'장세':5} {'건수':>4} {'자체SW':>9} {'BTC-SW':>9} {'B&H':>9}")
    print(f"  {'-'*45}")
    for reg in ["상승장", "하락장", "횡보장"]:
        sub = df_res[df_res["regime"] == reg]
        print(f"  {reg:5} {len(sub):>4}건"
              f" {sub['self_sw'].mean():>+8.1f}%"
              f" {sub['btc_sw'].mean():>+8.1f}%"
              f" {sub['bh'].mean():>+8.1f}%")

    # XRP만 따로
    print(f"\n  XRP만 (BTC 레짐 효과)")
    print(f"  {'장세':5} {'건수':>4} {'자체SW':>9} {'BTC-SW':>9} {'B&H':>9}")
    print(f"  {'-'*45}")
    for reg in ["상승장", "하락장", "횡보장"]:
        sub = df_res[(df_res["coin"] == "XRP") & (df_res["regime"] == reg)]
        if len(sub) == 0:
            continue
        print(f"  {reg:5} {len(sub):>4}건"
              f" {sub['self_sw'].mean():>+8.1f}%"
              f" {sub['btc_sw'].mean():>+8.1f}%"
              f" {sub['bh'].mean():>+8.1f}%")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_res.to_csv(os.path.join(RESULTS_DIR, "switch_btcregime.csv"), index=False)
    print(f"\n저장: {RESULTS_DIR}/switch_btcregime.csv")


if __name__ == "__main__":
    main()
