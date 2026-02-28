"""backtest_wf_all.py — V1/V2/V3 롤링 윈도우 OOS 검증.

파라미터 고정 상태로 6개월 윈도우를 시계열로 슬라이딩하여
각 전략의 시간대별 성과 분포와 통계적 유의성을 검증한다.

윈도우: 2022-01 ~ 2026-02 (6개월 × 8개 + 부분 윈도우 1개)
전략 파라미터: 원본 스크립트와 동일 (변경 없음)
비용 모델: 원본과 동일 (COMMISSION_RATE=0.0002, LEVERAGE=3)

V1: ADX<25 평균회귀 (BB하단+RSI<35+ADX<25 → TP=BB중심, SL=ATR×1.5)
V2: EMA20/50 크로스 추세추종 (ADX>20 → TP=ATR×3, SL=ATR×1.5)
V3: RSI 다이버전스 (MACD+OBV+Volume → TP=ATR×2.5, SL=ATR×1.2)

사용법:
  python backtest_wf_all.py
  python backtest_wf_all.py --coin eth
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import ta as ta_lib
from scipy import stats as scipy_stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── 비용 / 포지션 ────────────────────────────────────────────
LEVERAGE = 3
COMMISSION_RATE = 0.0002   # 진입+청산 수수료 (원본 동일)
INITIAL_BALANCE = 10_000

# ── 공통 지표 파라미터 ────────────────────────────────────────
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

# ── V1 파라미터 ───────────────────────────────────────────────
V1_RSI_THRESH = 35.0
V1_RSI_OVERBOUGHT = 65.0
V1_ADX_THRESH = 25.0
V1_SL_ATR = 1.5
V1_MAX_HOLD = 96       # 24시간

# ── V2 파라미터 ───────────────────────────────────────────────
V2_EMA_FAST = 20
V2_EMA_SLOW = 50
V2_ADX_MIN = 20.0
V2_TP_ATR = 3.0
V2_SL_ATR = 1.5
V2_MAX_HOLD = 480      # 5일
V2_FUNDING_LONG_MAX = 0.0005
V2_FUNDING_SHORT_MIN = -0.0005

# ── V3 파라미터 ───────────────────────────────────────────────
V3_TP_ATR = 2.5
V3_SL_ATR = 1.2
V3_MAX_HOLD = 192      # 48시간
V3_DIV_LOOKBACK = 20
V3_VOLUME_MULT = 1.2
V3_OBV_SMA = 10
V3_RSI_EXIT_LONG = 75.0
V3_RSI_EXIT_SHORT = 25.0
V3_MACD_FAST = 12
V3_MACD_SLOW_P = 26
V3_MACD_SIGNAL = 9

WARMUP_ROWS = 200  # 슬라이스 앞에 붙이는 지표 안정화 버퍼

# 6개월 OOS 윈도우 (전체 5년 + 2026년 1~2월)
WINDOWS = [
    ("2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30"),
    ("2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
    ("2026-01-01", "2026-02-28"),
]


# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────

def load_and_resample(coin: str) -> pd.DataFrame:
    frames = []
    for year in range(2022, 2027):
        path = os.path.join(DATA_DIR, f"{coin}_1m_{year}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"데이터 없음: {coin}")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    df_15m = df.resample("15min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return df_15m.reset_index()


# ─────────────────────────────────────────────────────────────
# V1 — ADX 평균회귀
# ─────────────────────────────────────────────────────────────

def add_v1_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low = df["close"], df["high"], df["low"]
    bb = ta_lib.volatility.BollingerBands(close, window=BB_PERIOD, window_dev=BB_STD)
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()
    df["rsi"] = ta_lib.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()
    df["rsi_prev"] = df["rsi"].shift(1)
    df["atr"] = ta_lib.volatility.AverageTrueRange(high, low, close, window=ATR_PERIOD).average_true_range()
    df["adx"] = ta_lib.trend.ADXIndicator(high, low, close, window=ADX_PERIOD).adx()
    return df


def run_v1(df: pd.DataFrame, w_start: pd.Timestamp, w_end: pd.Timestamp) -> list[dict]:
    trades = []
    in_trade = False
    side = "long"
    entry_price = tp = sl = 0.0
    entry_idx = 0

    for i in range(WARMUP_ROWS, len(df)):
        row = df.iloc[i]
        if pd.isna(row["adx"]) or pd.isna(row["rsi_prev"]):
            continue

        if in_trade:
            hold = i - entry_idx
            exit_price = result = None
            if side == "long":
                if bool(row["low"] <= sl) and (not bool(row["high"] >= tp) or sl <= tp):
                    exit_price, result = sl, "loss"
                elif bool(row["high"] >= tp):
                    exit_price, result = tp, "win"
                elif hold >= V1_MAX_HOLD:
                    exit_price, result = row["close"], "timeout"
            else:
                if bool(row["high"] >= sl) and (not bool(row["low"] <= tp) or sl >= tp):
                    exit_price, result = sl, "loss"
                elif bool(row["low"] <= tp):
                    exit_price, result = tp, "win"
                elif hold >= V1_MAX_HOLD:
                    exit_price, result = row["close"], "timeout"

            if result is not None:
                raw_pnl = (
                    (exit_price - entry_price) / entry_price * LEVERAGE * 100
                    if side == "long"
                    else (entry_price - exit_price) / entry_price * LEVERAGE * 100
                )
                fee_pct = COMMISSION_RATE * 2 * LEVERAGE * 100
                entry_dt = df.iloc[entry_idx]["timestamp"]
                if w_start <= entry_dt <= w_end:
                    trades.append({
                        "entry_dt": entry_dt, "exit_dt": row["timestamp"],
                        "side": side, "result": result,
                        "pnl_pct": round(raw_pnl - fee_pct, 4),
                    })
                in_trade = False

        if not in_trade:
            regime_ok = row["adx"] < V1_ADX_THRESH
            long_ok = (
                regime_ok
                and row["close"] <= row["bb_lower"]
                and row["rsi"] <= V1_RSI_THRESH
                and row["rsi"] > row["rsi_prev"]
                and row["bb_middle"] > row["close"]
            )
            short_ok = (
                regime_ok
                and row["close"] >= row["bb_upper"]
                and row["rsi"] >= V1_RSI_OVERBOUGHT
                and row["rsi"] < row["rsi_prev"]
                and row["bb_middle"] < row["close"]
            )
            if long_ok:
                entry_price = row["close"]
                tp = row["bb_middle"]
                sl = entry_price - row["atr"] * V1_SL_ATR
                side, in_trade, entry_idx = "long", True, i
            elif short_ok:
                entry_price = row["close"]
                tp = row["bb_middle"]
                sl = entry_price + row["atr"] * V1_SL_ATR
                side, in_trade, entry_idx = "short", True, i

    return trades


# ─────────────────────────────────────────────────────────────
# V2 — EMA 크로스 추세추종
# ─────────────────────────────────────────────────────────────

def add_v2_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low = df["close"], df["high"], df["low"]
    df["ema_fast"] = ta_lib.trend.EMAIndicator(close, window=V2_EMA_FAST).ema_indicator()
    df["ema_slow"] = ta_lib.trend.EMAIndicator(close, window=V2_EMA_SLOW).ema_indicator()
    df["ema_fast_prev"] = df["ema_fast"].shift(1)
    df["ema_slow_prev"] = df["ema_slow"].shift(1)
    df["adx"] = ta_lib.trend.ADXIndicator(high, low, close, window=ADX_PERIOD).adx()
    df["atr"] = ta_lib.volatility.AverageTrueRange(high, low, close, window=ATR_PERIOD).average_true_range()
    df["long_cross"] = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast_prev"] <= df["ema_slow_prev"])
    df["short_cross"] = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast_prev"] >= df["ema_slow_prev"])
    df["funding_rate"] = 0.0   # 로컬 데이터 없음 — 필터 실질 비활성화
    return df


def run_v2(df: pd.DataFrame, w_start: pd.Timestamp, w_end: pd.Timestamp) -> list[dict]:
    trades = []
    in_trade = False
    side = "long"
    entry_price = tp = sl = 0.0
    entry_idx = 0

    for i in range(WARMUP_ROWS, len(df)):
        row = df.iloc[i]
        if pd.isna(row["adx"]) or pd.isna(row["ema_fast"]):
            continue

        if in_trade:
            hold = i - entry_idx
            exit_price = result = None
            if side == "long":
                if bool(row["low"] <= sl) and (not bool(row["high"] >= tp) or sl <= tp):
                    exit_price, result = sl, "loss"
                elif bool(row["high"] >= tp):
                    exit_price, result = tp, "win"
                elif bool(row["short_cross"]):
                    exit_price, result = row["close"], "cross_exit"
                elif hold >= V2_MAX_HOLD:
                    exit_price, result = row["close"], "timeout"
            else:
                if bool(row["high"] >= sl) and (not bool(row["low"] <= tp) or sl >= tp):
                    exit_price, result = sl, "loss"
                elif bool(row["low"] <= tp):
                    exit_price, result = tp, "win"
                elif bool(row["long_cross"]):
                    exit_price, result = row["close"], "cross_exit"
                elif hold >= V2_MAX_HOLD:
                    exit_price, result = row["close"], "timeout"

            if result is not None:
                raw_pnl = (
                    (exit_price - entry_price) / entry_price * LEVERAGE * 100
                    if side == "long"
                    else (entry_price - exit_price) / entry_price * LEVERAGE * 100
                )
                fee_pct = COMMISSION_RATE * 2 * LEVERAGE * 100
                entry_dt = df.iloc[entry_idx]["timestamp"]
                if w_start <= entry_dt <= w_end:
                    trades.append({
                        "entry_dt": entry_dt, "exit_dt": row["timestamp"],
                        "side": side, "result": result,
                        "pnl_pct": round(raw_pnl - fee_pct, 4),
                    })
                in_trade = False

        if not in_trade:
            adx_ok = row["adx"] > V2_ADX_MIN
            fr = float(row["funding_rate"])
            if bool(row["long_cross"]) and adx_ok and fr < V2_FUNDING_LONG_MAX:
                entry_price = row["close"]
                tp = entry_price + row["atr"] * V2_TP_ATR
                sl = entry_price - row["atr"] * V2_SL_ATR
                side, in_trade, entry_idx = "long", True, i
            elif bool(row["short_cross"]) and adx_ok and fr > V2_FUNDING_SHORT_MIN:
                entry_price = row["close"]
                tp = entry_price - row["atr"] * V2_TP_ATR
                sl = entry_price + row["atr"] * V2_SL_ATR
                side, in_trade, entry_idx = "short", True, i

    return trades


# ─────────────────────────────────────────────────────────────
# V3 — RSI 다이버전스
# ─────────────────────────────────────────────────────────────

def add_v3_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    df["rsi"] = ta_lib.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()
    macd = ta_lib.trend.MACD(
        close, window_fast=V3_MACD_FAST,
        window_slow=V3_MACD_SLOW_P, window_sign=V3_MACD_SIGNAL,
    )
    df["macd_hist"] = macd.macd_diff()
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    df["atr"] = ta_lib.volatility.AverageTrueRange(high, low, close, window=ATR_PERIOD).average_true_range()
    df["vol_sma20"] = volume.rolling(window=V3_DIV_LOOKBACK).mean()
    df["obv"] = ta_lib.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["obv_sma10"] = df["obv"].rolling(window=V3_OBV_SMA).mean()

    # 다이버전스 탐지 (numpy 루프)
    rsi_vals = df["rsi"].values
    low_vals = df["low"].values
    high_vals = df["high"].values
    n = len(df)
    psl = np.full(n, np.nan)
    rsl = np.full(n, np.nan)
    psh = np.full(n, np.nan)
    rsh = np.full(n, np.nan)

    for i in range(V3_DIV_LOOKBACK, n):
        ws = i - V3_DIV_LOOKBACK
        wr = rsi_vals[ws:i]
        if np.any(np.isnan(wr)):
            continue
        wl = low_vals[ws:i]
        wh = high_vals[ws:i]
        mi = int(np.argmin(wl))
        psl[i] = wl[mi]; rsl[i] = wr[mi]
        xi = int(np.argmax(wh))
        psh[i] = wh[xi]; rsh[i] = wr[xi]

    df["past_swing_low"] = psl
    df["rsi_at_swing_low"] = rsl
    df["past_swing_high"] = psh
    df["rsi_at_swing_high"] = rsh
    return df


def run_v3(df: pd.DataFrame, w_start: pd.Timestamp, w_end: pd.Timestamp) -> list[dict]:
    trades = []
    in_trade = False
    side = "long"
    entry_price = tp = sl = 0.0
    entry_idx = 0

    for i in range(WARMUP_ROWS, len(df)):
        row = df.iloc[i]
        if pd.isna(row["rsi"]) or pd.isna(row["macd_hist"]) or pd.isna(row["atr"]):
            continue

        if in_trade:
            hold = i - entry_idx
            exit_price = result = None
            if side == "long":
                if bool(row["low"] <= sl) and (not bool(row["high"] >= tp) or sl <= tp):
                    exit_price, result = sl, "loss"
                elif bool(row["high"] >= tp):
                    exit_price, result = tp, "win"
                elif bool(row["rsi"] > V3_RSI_EXIT_LONG):
                    exit_price, result = row["close"], "rsi_exit"
                elif hold >= V3_MAX_HOLD:
                    exit_price, result = row["close"], "timeout"
            else:
                if bool(row["high"] >= sl) and (not bool(row["low"] <= tp) or sl >= tp):
                    exit_price, result = sl, "loss"
                elif bool(row["low"] <= tp):
                    exit_price, result = tp, "win"
                elif bool(row["rsi"] < V3_RSI_EXIT_SHORT):
                    exit_price, result = row["close"], "rsi_exit"
                elif hold >= V3_MAX_HOLD:
                    exit_price, result = row["close"], "timeout"

            if result is not None:
                raw_pnl = (
                    (exit_price - entry_price) / entry_price * LEVERAGE * 100
                    if side == "long"
                    else (entry_price - exit_price) / entry_price * LEVERAGE * 100
                )
                fee_pct = COMMISSION_RATE * 2 * LEVERAGE * 100
                entry_dt = df.iloc[entry_idx]["timestamp"]
                if w_start <= entry_dt <= w_end:
                    trades.append({
                        "entry_dt": entry_dt, "exit_dt": row["timestamp"],
                        "side": side, "result": result,
                        "pnl_pct": round(raw_pnl - fee_pct, 4),
                    })
                in_trade = False

        if not in_trade:
            if (
                pd.isna(row["past_swing_low"]) or pd.isna(row["rsi_at_swing_low"])
                or pd.isna(row["macd_hist_prev"]) or pd.isna(row["vol_sma20"])
                or pd.isna(row["obv_sma10"])
            ):
                continue

            vol_ok = bool(row["volume"] > row["vol_sma20"] * V3_VOLUME_MULT)
            bull_div = bool(row["low"] < row["past_swing_low"] and row["rsi"] > row["rsi_at_swing_low"])
            macd_up = bool(row["macd_hist"] > 0 and row["macd_hist_prev"] <= 0)
            obv_up = bool(row["obv"] > row["obv_sma10"])

            if bull_div and macd_up and vol_ok and obv_up:
                entry_price = float(row["close"])
                atr = float(row["atr"])
                tp = entry_price + atr * V3_TP_ATR
                sl = entry_price - atr * V3_SL_ATR
                side, in_trade, entry_idx = "long", True, i
                continue

            if pd.isna(row["rsi_at_swing_high"]):
                continue

            bear_div = bool(row["high"] > row["past_swing_high"] and row["rsi"] < row["rsi_at_swing_high"])
            macd_dn = bool(row["macd_hist"] < 0 and row["macd_hist_prev"] >= 0)
            obv_dn = bool(row["obv"] < row["obv_sma10"])

            if bear_div and macd_dn and vol_ok and obv_dn:
                entry_price = float(row["close"])
                atr = float(row["atr"])
                tp = entry_price - atr * V3_TP_ATR
                sl = entry_price + atr * V3_SL_ATR
                side, in_trade, entry_idx = "short", True, i

    return trades


# ─────────────────────────────────────────────────────────────
# 통계
# ─────────────────────────────────────────────────────────────

def compute_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "win_rate": 0, "ret": 0, "mdd": 0, "sharpe": 0, "exp": 0}
    pnl = np.array([t["pnl_pct"] for t in trades])
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    mdd = 0.0
    for p in pnl:
        balance *= 1 + p / 100
        peak = max(peak, balance)
        mdd = max(mdd, (peak - balance) / peak * 100)
    ret = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    wr = np.mean(pnl > 0) * 100
    sharpe = (np.mean(pnl) / np.std(pnl)) * np.sqrt(len(pnl)) if np.std(pnl) > 0 else 0.0
    return {"n": len(pnl), "win_rate": wr, "ret": ret, "mdd": mdd, "sharpe": sharpe, "exp": float(np.mean(pnl))}


# ─────────────────────────────────────────────────────────────
# Walk-Forward 실행
# ─────────────────────────────────────────────────────────────

def run_wf(df: pd.DataFrame, indicator_fn, backtest_fn, name: str, coin: str) -> None:
    print(f"\n{'='*64}")
    print(f"  {name}  —  {coin.upper()} 15m 롤링 OOS 검증")
    print(f"{'='*64}")

    print("  지표 계산 중 ...", end="", flush=True)
    df_ind = indicator_fn(df)
    print(" 완료")

    ts = df_ind["timestamp"]
    all_trades: list[dict] = []
    window_stats: list[dict] = []

    for w_start_str, w_end_str in WINDOWS:
        w_start = pd.Timestamp(w_start_str, tz="UTC")
        w_end = pd.Timestamp(w_end_str, tz="UTC") + pd.Timedelta(hours=23, minutes=59)

        # 윈도우 시작 200행 앞 → 지표 안정화 버퍼
        start_mask = ts >= w_start
        if not start_mask.any():
            continue
        start_pos = max(0, start_mask.idxmax() - WARMUP_ROWS)

        end_mask = ts <= w_end
        if not end_mask.any():
            continue
        end_pos = end_mask[::-1].idxmax()

        slice_df = df_ind.iloc[start_pos:end_pos + 1].reset_index(drop=True)
        trades = backtest_fn(slice_df, w_start, w_end)
        stats = compute_stats(trades)
        stats["period"] = f"{w_start_str[:7]} ~ {w_end_str[:7]}"
        window_stats.append(stats)
        all_trades.extend(trades)

    # 윈도우별 표
    print(f"\n  {'기간':>17}  {'거래':>5}  {'승률':>6}  {'수익':>9}  {'MDD':>7}  {'Sharpe':>7}")
    print(f"  {'-'*60}")
    pos_windows = 0
    for s in window_stats:
        flag = "+" if s["ret"] > 0 else " "
        if s["ret"] > 0:
            pos_windows += 1
        print(
            f"  {s['period']:>17}  {s['n']:>5}  {s['win_rate']:>5.1f}%  "
            f"{flag}{s['ret']:>8.1f}%  -{s['mdd']:>5.1f}%  {s['sharpe']:>7.2f}"
        )

    total_w = len(window_stats)
    print(f"\n  수익 윈도우: {pos_windows}/{total_w}  "
          f"({'%.0f' % (pos_windows/total_w*100) if total_w else 0}%)")

    # 전체 OOS 합산
    if all_trades:
        all_pnl = [t["pnl_pct"] for t in all_trades]
        t_stat, p_val = scipy_stats.ttest_1samp(all_pnl, 0)
        ov = compute_stats(all_trades)
        sig = "✓ 유의 (p<0.05)" if p_val < 0.05 else "✗ 비유의"
        print(f"\n  ── 전체 OOS 합산 ({ov['n']}건) ──")
        print(f"  승률: {ov['win_rate']:.1f}%  기대값: {ov['exp']:+.3f}%  Sharpe: {ov['sharpe']:.2f}")
        print(f"  누적 수익: {ov['ret']:+.1f}%  MDD: -{ov['mdd']:.1f}%")
        print(f"  t-stat: {t_stat:.2f}  p-value: {p_val:.4f}  {sig}")

        # CSV 저장
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"wf_all_{name.split()[0].lower()}_{coin}.csv")
        pd.DataFrame(all_trades).to_csv(out_path, index=False)
        print(f"  → {out_path}")
    else:
        print(f"\n  전체 거래 없음")


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="V1/V2/V3 Walk-Forward 검증")
    parser.add_argument("--coin", default="btc", help="코인 (btc/eth/sol/xrp)")
    args = parser.parse_args()
    coin = args.coin.lower()

    print(f"데이터 로드: {coin.upper()} ...", end="", flush=True)
    df = load_and_resample(coin)
    print(f" {len(df):,}개 캔들  ({df['timestamp'].iloc[0].date()} ~ {df['timestamp'].iloc[-1].date()})")

    run_wf(df, add_v1_indicators, run_v1, "V1 (ADX 평균회귀)", coin)
    run_wf(df, add_v2_indicators, run_v2, "V2 (EMA 크로스 추세추종)", coin)
    run_wf(df, add_v3_indicators, run_v3, "V3 (RSI 다이버전스)", coin)

    print(f"\n{'='*64}")
    print("  완료")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
