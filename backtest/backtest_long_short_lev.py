"""4H BB+RSI 롱숏 — 레버리지×비중 조합 비교 테스트.

기존 BEST_PARAMS 고정, 레버리지×비중 6조합 × 3코인 검증.
목적: 강제청산 위험 줄이면서 실질 노출은 유지.

청산 거리 = 1/leverage × 100%

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

COINS  = ["btc", "eth", "xrp"]
BH_REF = {"btc": 89.4, "eth": -19.4, "xrp": 121.6}

# 고정 파라미터 (기존 BEST_PARAMS에서 leverage/pos_ratio 제외)
BASE_PARAMS = {
    "btc": dict(rsi_long=30, rsi_short=65, sl_mult=2.0, tp_mode="atr_3x", use_ema200=True),
    "eth": dict(rsi_long=25, rsi_short=65, sl_mult=2.0, tp_mode="atr_2x", use_ema200=True),
    "xrp": dict(rsi_long=25, rsi_short=65, sl_mult=2.0, tp_mode="atr_3x", use_ema200=True),
}

# 테스트할 (leverage, pos_ratio) 조합
COMBOS = [
    (2, 1.0),   # 2x×100% = 200% 노출, 청산 -50%
    (3, 0.7),   # 3x×70%  = 210% 노출, 청산 -33%
    (3, 1.0),   # 3x×100% = 300% 노출, 청산 -33%
    (5, 0.4),   # 5x×40%  = 200% 노출, 청산 -20%
    (5, 0.5),   # 5x×50%  = 250% 노출, 청산 -20%
    (7, 0.3),   # 7x×30%  = 210% 노출, 청산 -14% ← 기존
]


# ── 데이터 준비 ───────────────────────────────────────

def load_range(coin: str) -> pd.DataFrame:
    frames = []
    for y in range(2021, 2027):
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


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("4h").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),    close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna().reset_index()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c  = df["close"].astype(float)
    h  = df["high"].astype(float)
    lo = df["low"].astype(float)

    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std(ddof=1)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    tr     = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr    = tr.ewm(com=13, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()

    df = df.copy()
    df["bb_mid"]   = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["rsi"]      = rsi
    df["atr"]      = atr
    df["ema200"]   = ema200
    return df


# ── 백테스트 코어 ─────────────────────────────────────

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
    initial_balance: float = INITIAL_BAL,
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
    n = len(closes)

    balance = initial_balance
    peak    = initial_balance
    mdd     = 0.0

    in_pos    = False
    direction = 0
    ep = sl = tp = pos_amt = 0.0
    entry_bar = 0

    long_trades = long_wins = 0
    short_trades = short_wins = 0

    for i in range(1, n):
        if any(np.isnan(x[i]) for x in [bb_mid, bb_upper, bb_lower, rsi_arr, atr_arr, ema200]):
            continue

        if in_pos:
            exit_p = None

            if direction == 1:
                tp_check = bb_mid[i] if tp_mode == "bb_mid" else tp
                if lows[i] <= sl:
                    exit_p = sl
                elif tp_check > ep and highs[i] >= tp_check:
                    exit_p = tp_check
                    long_wins += 1
                elif i - entry_bar >= timeout_bars:
                    exit_p = closes[i]
                    if exit_p > ep:
                        long_wins += 1
            else:
                tp_check = bb_mid[i] if tp_mode == "bb_mid" else tp
                if highs[i] >= sl:
                    exit_p = sl
                elif tp_check < ep and lows[i] <= tp_check:
                    exit_p = tp_check
                    short_wins += 1
                elif i - entry_bar >= timeout_bars:
                    exit_p = closes[i]
                    if exit_p < ep:
                        short_wins += 1

            if exit_p is not None:
                if direction == 1:
                    pnl = (exit_p - ep) * pos_amt
                    long_trades += 1
                else:
                    pnl = (ep - exit_p) * pos_amt
                    short_trades += 1
                balance += pnl - exit_p * pos_amt * TAKER_FEE
                in_pos    = False
                direction = 0

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > mdd:
                mdd = dd

        if not in_pos:
            if balance <= 0:
                break
            notional = balance * pos_ratio * leverage

            ema_long_ok  = (not use_ema200) or (closes[i] > ema200[i])
            ema_short_ok = (not use_ema200) or (closes[i] < ema200[i])

            if ema_long_ok and closes[i] < bb_lower[i] and rsi_arr[i] < rsi_long:
                ep        = closes[i]
                pos_amt   = notional / ep
                balance  -= notional * TAKER_FEE
                sl        = ep - atr_arr[i] * sl_mult
                tp        = ep + atr_arr[i] * (2.0 if tp_mode == "atr_2x" else 3.0) if tp_mode != "bb_mid" else 0.0
                entry_bar = i
                in_pos    = True
                direction = 1

            elif ema_short_ok and closes[i] > bb_upper[i] and rsi_arr[i] > rsi_short:
                ep        = closes[i]
                pos_amt   = notional / ep
                balance  -= notional * TAKER_FEE
                sl        = ep + atr_arr[i] * sl_mult
                tp        = ep - atr_arr[i] * (2.0 if tp_mode == "atr_2x" else 3.0) if tp_mode != "bb_mid" else 0.0
                entry_bar = i
                in_pos    = True
                direction = -1

    ret    = (balance - initial_balance) / initial_balance * 100 if balance > 0 else -100.0
    total  = long_trades + short_trades
    lwr    = long_wins  / long_trades  * 100 if long_trades  > 0 else 0.0
    swr    = short_wins / short_trades * 100 if short_trades > 0 else 0.0
    wr     = (long_wins + short_wins)  / total * 100 if total > 0 else 0.0
    calmar = ret / mdd if mdd > 0 else 0.0

    return {
        "trades":       total,
        "long_trades":  long_trades,
        "short_trades": short_trades,
        "win_rate":     round(wr, 1),
        "long_wr":      round(lwr, 1),
        "short_wr":     round(swr, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(mdd, 2),
        "calmar":       round(calmar, 2),
    }


# ── 메인 ──────────────────────────────────────────────

def main():
    print("4H BB+RSI 롱숏 — 레버리지×비중 조합 비교 (2022~2025)")
    print("청산거리 = 1/레버리지 × 100%\n")

    # 데이터 준비
    coin_df = {}
    for coin in COINS:
        df_1m = load_range(coin)
        df_4h = resample_4h(df_1m)
        df_ind = compute_indicators(df_4h)
        s = pd.Timestamp(START_DATE, tz="UTC")
        e = pd.Timestamp(END_DATE + " 23:59:59", tz="UTC")
        coin_df[coin] = df_ind[(df_ind["timestamp"] >= s) &
                                (df_ind["timestamp"] <= e)].reset_index(drop=True)
        # 연도별용 전체 df (2022-2026 포함)
    coin_df_full = {}
    for coin in COINS:
        df_1m = load_range(coin)
        df_4h = resample_4h(df_1m)
        coin_df_full[coin] = compute_indicators(df_4h)

    all_rows = []

    for coin in COINS:
        bp  = BASE_PARAMS[coin]
        bh  = BH_REF[coin]
        df  = coin_df[coin]
        dff = coin_df_full[coin]

        print(f"{'='*72}")
        rsi_l = bp['rsi_long']; rsi_s = bp['rsi_short']
        sl_m  = bp['sl_mult'];  tp_m  = bp['tp_mode']
        print(f"{coin.upper()} — RSI_L<{rsi_l} RSI_S>{rsi_s} sl×{sl_m} {tp_m} EMA200=ON")
        print(f"{'='*72}")
        print(f"  {'레버리지':>5} {'비중':>5} {'노출':>5} {'청산거리':>7} "
              f"{'수익률':>9} {'MDD':>7} {'Calmar':>8} {'거래':>5} {'B&H':>8} {'판정':>10}")
        print(f"  {'-'*72}")

        combo_results = []
        for lev, pos in COMBOS:
            exposure    = lev * pos * 100
            liq_dist    = 100 / lev
            is_base     = (lev == 7 and pos == 0.3)
            r = run_backtest(df, **bp, leverage=lev, pos_ratio=pos)
            beat = "✓ B&H 초과" if r["return_pct"] > bh else "✗ B&H 미달"
            mark = " ← 기존" if is_base else ""
            print(
                f"  {lev:>4}x  {pos*100:>4.0f}%  "
                f"{exposure:>4.0f}%   -{liq_dist:>5.0f}%  "
                f"{r['return_pct']:>+8.1f}%  {r['max_drawdown']:>6.1f}%  "
                f"{r['calmar']:>7.2f}  {r['trades']:>4}건  "
                f"{bh:>+7.1f}%  {beat}{mark}"
            )
            combo_results.append((lev, pos, r))
            all_rows.append({"coin": coin.upper(), "leverage": lev, "pos_ratio": pos,
                              "exposure_pct": exposure, "liq_dist_pct": liq_dist,
                              **r, "bh_ref": bh})

        # 연도별 수익률
        print(f"\n  연도별 수익률:")
        header = f"  {'조합':>10}"
        for year in range(2022, 2026):
            header += f"  {year:>9}"
        print(header)
        print(f"  {'-'*50}")

        for lev, pos, _ in combo_results:
            is_base = (lev == 7 and pos == 0.3)
            mark = "←기존" if is_base else "     "
            line = f"  {lev}x/{pos*100:.0f}%{mark}"
            for year in range(2022, 2026):
                s = pd.Timestamp(f"{year}-01-01", tz="UTC")
                e = pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
                df_y = dff[(dff["timestamp"] >= s) & (dff["timestamp"] <= e)].reset_index(drop=True)
                if len(df_y) < 10:
                    line += f"  {'N/A':>9}"
                    continue
                ry = run_backtest(df_y, **BASE_PARAMS[coin], leverage=lev, pos_ratio=pos)
                flag = "✓" if ry["return_pct"] >= 0 else "✗"
                line += f"  {ry['return_pct']:>+7.1f}%{flag}"
            print(line)
        print()

    # 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "long_short_lev_compare.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)

    # 최종 요약
    print(f"{'='*72}")
    print("최종 요약 — 조합별 3코인 수익률")
    print(f"{'='*72}")
    print(f"  {'조합':>10} {'노출':>6} {'청산':>7}  "
          f"{'BTC':>10} {'ETH':>10} {'XRP':>10}  {'모두 B&H초과':>10}")
    print(f"  {'-'*68}")
    for lev, pos in COMBOS:
        exposure = lev * pos * 100
        liq_dist = 100 / lev
        is_base  = (lev == 7 and pos == 0.3)
        mark     = " ←기존" if is_base else ""
        row_map  = {r["coin"]: r for r in all_rows if r["leverage"] == lev and r["pos_ratio"] == pos}
        btc_r    = row_map.get("BTC", {}).get("return_pct", 0)
        eth_r    = row_map.get("ETH", {}).get("return_pct", 0)
        xrp_r    = row_map.get("XRP", {}).get("return_pct", 0)
        all_beat = (btc_r > BH_REF["btc"] and eth_r > BH_REF["eth"] and xrp_r > BH_REF["xrp"])
        print(
            f"  {lev}x/{pos*100:.0f}%    {exposure:>4.0f}%  -{liq_dist:>4.0f}%   "
            f"{btc_r:>+8.1f}%  {eth_r:>+8.1f}%  {xrp_r:>+8.1f}%  "
            f"{'✅' if all_beat else '❌'}{mark}"
        )
    print(f"\n  B&H:              "
          f"  {BH_REF['btc']:>+8.1f}%  {BH_REF['eth']:>+8.1f}%  {BH_REF['xrp']:>+8.1f}%")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
