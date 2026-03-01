"""4H BB+RSI 양방향 전략 — 전체 기간 견고성 검증.

확정 파라미터 고정 (파라미터 탐색 없음):
  BTC: RSI_L<30, RSI_S>65, SL=ATR×2, TP=ATR×3, EMA200=ON, 3x×70%
  ETH: RSI_L<25, RSI_S>65, SL=ATR×2, TP=ATR×2, EMA200=ON, 3x×70%
  XRP: RSI_L<25, RSI_S>65, SL=ATR×2, TP=ATR×3, EMA200=ON, 3x×70%

데이터: {coin}_1h_full.parquet → 4H 리샘플링
  BTC/ETH: 2017-08-17 ~ 2026-03
  XRP:     2018-05-04 ~ 2026-03
"""

import os
import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0

FIXED_PARAMS = {
    "btc": dict(rsi_long=30, rsi_short=65, sl_mult=2.0, tp_mode="atr_3x", leverage=3, pos_ratio=0.70, use_ema200=True),
    "eth": dict(rsi_long=25, rsi_short=65, sl_mult=2.0, tp_mode="atr_2x", leverage=3, pos_ratio=0.70, use_ema200=True),
    "xrp": dict(rsi_long=25, rsi_short=65, sl_mult=2.0, tp_mode="atr_3x", leverage=3, pos_ratio=0.70, use_ema200=True),
}

COINS = ["btc", "eth", "xrp"]


# ── 데이터 로드 / 인디케이터 ──────────────────────────────

def load_1h_full(coin: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{coin}_1h_full.parquet")
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
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

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    tr     = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr    = tr.ewm(com=13, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()

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

            if direction == 1:  # 롱
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
            else:  # 숏
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
                pnl     = (exit_p - ep) * pos_amt if direction == 1 else (ep - exit_p) * pos_amt
                fee     = exit_p * pos_amt * TAKER_FEE
                balance += pnl - fee
                if direction == 1:
                    long_trades += 1
                else:
                    short_trades += 1
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

            # 롱 신호
            ema_ok = (not use_ema200) or (closes[i] > ema200[i])
            if ema_ok and closes[i] < bb_lower[i] and rsi_arr[i] < rsi_long:
                ep        = closes[i]
                pos_amt   = notional / ep
                balance  -= notional * TAKER_FEE
                sl        = ep - atr_arr[i] * sl_mult
                tp        = (ep + atr_arr[i] * 3.0) if tp_mode == "atr_3x" else \
                            (ep + atr_arr[i] * 2.0) if tp_mode == "atr_2x" else 0.0
                entry_bar = i
                in_pos    = True
                direction = 1
                continue

            # 숏 신호
            ema_ok = (not use_ema200) or (closes[i] < ema200[i])
            if ema_ok and closes[i] > bb_upper[i] and rsi_arr[i] > rsi_short:
                ep        = closes[i]
                pos_amt   = notional / ep
                balance  -= notional * TAKER_FEE
                sl        = ep + atr_arr[i] * sl_mult
                tp        = (ep - atr_arr[i] * 3.0) if tp_mode == "atr_3x" else \
                            (ep - atr_arr[i] * 2.0) if tp_mode == "atr_2x" else 0.0
                entry_bar = i
                in_pos    = True
                direction = -1

    if balance <= 0:
        ret = -100.0
    else:
        ret = (balance - initial_balance) / initial_balance * 100

    total_trades = long_trades + short_trades
    total_wins   = long_wins + short_wins
    win_rate     = total_wins / total_trades * 100 if total_trades > 0 else 0.0
    long_wr      = long_wins  / long_trades  * 100 if long_trades  > 0 else 0.0
    short_wr     = short_wins / short_trades * 100 if short_trades > 0 else 0.0
    calmar       = ret / mdd if mdd > 0 else 0.0

    return {
        "return_pct":     round(ret, 2),
        "max_drawdown":   round(mdd, 2),
        "calmar":         round(calmar, 2),
        "trades":         total_trades,
        "win_rate":       round(win_rate, 1),
        "long_trades":    long_trades,
        "long_win_rate":  round(long_wr, 1),
        "short_trades":   short_trades,
        "short_win_rate": round(short_wr, 1),
    }


# ── B&H 계산 ────────────────────────────────────────────

def bh_return(df: pd.DataFrame) -> float:
    first = df["close"].iloc[0]
    last  = df["close"].iloc[-1]
    return round((last - first) / first * 100, 1)


# ── 메인 ────────────────────────────────────────────────

def main():
    print("4H BB+RSI 양방향 전략 — 전체 기간 견고성 검증")
    print("파라미터: BTC(RSI_L<30/S>65,TP=atr3x) ETH(RSI_L<25/S>65,TP=atr2x) XRP(RSI_L<25/S>65,TP=atr3x)")
    print("공통: SL=ATR×2, EMA200=ON, 3x×70%\n")

    rows = []

    for coin in COINS:
        print(f"▶ {coin.upper()} 로드 중...", end=" ", flush=True)
        df_1h = load_1h_full(coin)
        df_4h = resample_4h(df_1h)
        df_4h = compute_indicators(df_4h)
        p     = FIXED_PARAMS[coin]
        print(f"{df_4h['timestamp'].min().year}~{df_4h['timestamp'].max().year} ({len(df_4h):,}봉)")

        # 전체 기간
        r_all = run_backtest(df_4h, **p)
        bh_all = bh_return(df_4h)
        first_year = df_4h["timestamp"].min().year
        last_year  = df_4h["timestamp"].max().year
        years_n    = last_year - first_year + 1

        print(f"\n{'='*60}")
        print(f"{coin.upper()} 전체 ({first_year}~{last_year}, {years_n}년)")
        print(f"  수익률: {r_all['return_pct']:+.1f}%  MDD: {r_all['max_drawdown']:.1f}%  "
              f"Calmar: {r_all['calmar']:.2f}")
        print(f"  거래: {r_all['trades']}건  승률: {r_all['win_rate']:.1f}%  "
              f"(롱 {r_all['long_trades']}건/{r_all['long_win_rate']:.0f}%  "
              f"숏 {r_all['short_trades']}건/{r_all['short_win_rate']:.0f}%)")
        print(f"  B&H: {bh_all:+.1f}%  →  전략 초과: {r_all['return_pct']-bh_all:+.1f}%p")

        rows.append({
            "coin": coin.upper(), "period": f"{first_year}~{last_year}",
            **r_all, "bh_return": bh_all,
        })

        # 연도별 breakdown
        all_years = sorted(df_4h["timestamp"].dt.year.unique())
        print(f"\n  연도별 ({coin.upper()}):")
        print(f"  {'연도':6} {'수익률':>9} {'MDD':>7} {'승률':>7} {'롱':>4} {'숏':>4} {'B&H':>8}")
        print(f"  {'-'*55}")

        for yr in all_years:
            s = pd.Timestamp(f"{yr}-01-01", tz="UTC")
            e = pd.Timestamp(f"{yr}-12-31 23:59:59", tz="UTC")
            df_y = df_4h[(df_4h["timestamp"] >= s) & (df_4h["timestamp"] <= e)].reset_index(drop=True)
            if len(df_y) < 10:
                continue
            r_y   = run_backtest(df_y, **p)
            bh_y  = bh_return(df_y)
            ytd   = " (YTD)" if yr == 2026 else ""
            sign  = "✅" if r_y["return_pct"] > 0 else "❌"
            print(
                f"  {sign} {yr}{ytd:6} "
                f"{r_y['return_pct']:>+8.1f}% "
                f"{r_y['max_drawdown']:>6.1f}% "
                f"{r_y['win_rate']:>6.1f}% "
                f"{r_y['long_trades']:>4}건 "
                f"{r_y['short_trades']:>4}건 "
                f"B&H{bh_y:>+7.1f}%"
            )
            rows.append({
                "coin": coin.upper(), "period": str(yr) + ytd,
                **r_y, "bh_return": bh_y,
            })

        print()

    # CSV 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "fullperiod_bb_rsi_longshort.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
