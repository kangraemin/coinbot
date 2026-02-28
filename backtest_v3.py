"""RSI 다이버전스 + 볼륨 확인 모멘텀 반전 백테스팅 (V3).

전략:
  롱: 불리시 다이버전스 + MACD 히스토그램 음→양 전환 + 볼륨 급증 + OBV 상승
  숏: 베어리시 다이버전스 + MACD 히스토그램 양→음 전환 + 볼륨 급증 + OBV 하락

  TP: 진입가 ± ATR × 2.5
  SL: 진입가 ∓ ATR × 1.2
  청산: TP/SL/RSI 과열(롱>75, 숏<25)/48시간 타임아웃

사용법:
  python backtest_v3.py          # 기본 90일
  python backtest_v3.py 365      # 365일
"""

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import ta as ta_lib

import config as cfg

# ── 파라미터 ──────────────────────────────────────────
BACKTEST_DAYS: int = 90
COMMISSION_RATE: float = 0.0002
INITIAL_BALANCE: float = 10_000
MAX_HOLD_CANDLES: int = 192    # 48시간 (15분 × 192)

RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
DIVERGENCE_LOOKBACK: int = 20
VOLUME_MULT: float = 1.2
OBV_SMA: int = 10
TP_ATR_MULT: float = 2.5
SL_ATR_MULT: float = 1.2
RSI_EXIT_LONG: float = 75.0
RSI_EXIT_SHORT: float = 25.0
CANDLE_WARMUP: int = 60


# ── 데이터 수집 ───────────────────────────────────────

def fetch_ohlcv(days: int = BACKTEST_DAYS) -> pd.DataFrame:
    exchange = ccxt.binanceusdm()
    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    cutoff_ms = int(time.time() * 1000) - 2 * 15 * 60 * 1000
    all_ohlcv: list = []

    print(f"OHLCV {days}일 수집 중", end="", flush=True)
    while since_ms < cutoff_ms:
        batch = exchange.fetch_ohlcv(
            cfg.SYMBOL, cfg.TIMEFRAME, since=since_ms, limit=1000
        )
        if not batch:
            break
        all_ohlcv.extend(batch)
        since_ms = batch[-1][0] + 1
        print(".", end="", flush=True)
        time.sleep(0.1)

    print(f" {len(all_ohlcv)}개")
    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return (
        df.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ── 지표 계산 ─────────────────────────────────────────

def add_indicators(df_candles: pd.DataFrame) -> pd.DataFrame:
    df = df_candles.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # RSI(14)
    df["rsi"] = ta_lib.momentum.RSIIndicator(close, window=RSI_PERIOD).rsi()

    # MACD 히스토그램 (fast=12, slow=26, signal=9)
    macd = ta_lib.trend.MACD(
        close,
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL,
    )
    df["macd_hist"] = macd.macd_diff()
    df["macd_hist_prev"] = df["macd_hist"].shift(1)

    # ATR(14)
    df["atr"] = ta_lib.volatility.AverageTrueRange(
        high, low, close, window=cfg.ATR_PERIOD
    ).average_true_range()

    # 볼륨 20봉 이동평균
    df["vol_sma20"] = volume.rolling(window=DIVERGENCE_LOOKBACK).mean()

    # OBV & OBV SMA(10)
    df["obv"] = ta_lib.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["obv_sma10"] = df["obv"].rolling(window=OBV_SMA).mean()

    # ── 다이버전스 탐지 ─────────────────────────────
    # 각 봉에서 과거 DIVERGENCE_LOOKBACK봉 내 스윙 로우/하이와 그 시점의 RSI 기록
    rsi_vals = df["rsi"].values
    low_vals = df["low"].values
    high_vals = df["high"].values
    n = len(df)

    past_swing_low = np.full(n, np.nan)
    rsi_at_swing_low = np.full(n, np.nan)
    past_swing_high = np.full(n, np.nan)
    rsi_at_swing_high = np.full(n, np.nan)

    for i in range(DIVERGENCE_LOOKBACK, n):
        ws = i - DIVERGENCE_LOOKBACK
        we = i  # 현재 봉 제외

        window_rsi = rsi_vals[ws:we]
        if np.any(np.isnan(window_rsi)):
            continue

        window_lows = low_vals[ws:we]
        window_highs = high_vals[ws:we]

        min_idx = int(np.argmin(window_lows))
        past_swing_low[i] = window_lows[min_idx]
        rsi_at_swing_low[i] = window_rsi[min_idx]

        max_idx = int(np.argmax(window_highs))
        past_swing_high[i] = window_highs[max_idx]
        rsi_at_swing_high[i] = window_rsi[max_idx]

    df["past_swing_low"] = past_swing_low
    df["rsi_at_swing_low"] = rsi_at_swing_low
    df["past_swing_high"] = past_swing_high
    df["rsi_at_swing_high"] = rsi_at_swing_high

    return df.reset_index(drop=True)


# ── 백테스팅 시뮬레이션 ───────────────────────────────

def run_backtest(df: pd.DataFrame) -> list[dict]:
    """RSI 다이버전스 신호로 롱/숏 진입/청산 시뮬레이션.

    청산 우선순위: SL > TP > RSI 과열 청산 > 타임아웃
    """
    trades: list[dict] = []
    in_trade = False
    side = "long"
    entry_price = tp = sl = 0.0
    entry_idx = 0

    for i in range(CANDLE_WARMUP, len(df)):
        row = df.iloc[i]

        if pd.isna(row["rsi"]) or pd.isna(row["macd_hist"]) or pd.isna(row["atr"]):
            continue

        # ── 보유 중: 청산 체크 ────────────────────
        if in_trade:
            hold = i - entry_idx
            exit_price = result = None

            if side == "long":
                hit_sl = bool(row["low"] <= sl)
                hit_tp = bool(row["high"] >= tp)
                rsi_exit = bool(row["rsi"] > RSI_EXIT_LONG)

                if hit_sl and (not hit_tp or sl <= tp):
                    exit_price, result = sl, "loss"
                elif hit_tp:
                    exit_price, result = tp, "win"
                elif rsi_exit:
                    exit_price, result = row["close"], "rsi_exit"
                elif hold >= MAX_HOLD_CANDLES:
                    exit_price, result = row["close"], "timeout"
            else:
                hit_sl = bool(row["high"] >= sl)
                hit_tp = bool(row["low"] <= tp)
                rsi_exit = bool(row["rsi"] < RSI_EXIT_SHORT)

                if hit_sl and (not hit_tp or sl >= tp):
                    exit_price, result = sl, "loss"
                elif hit_tp:
                    exit_price, result = tp, "win"
                elif rsi_exit:
                    exit_price, result = row["close"], "rsi_exit"
                elif hold >= MAX_HOLD_CANDLES:
                    exit_price, result = row["close"], "timeout"

            if result is not None:
                raw_pnl = (
                    (exit_price - entry_price) / entry_price * cfg.LEVERAGE * 100
                    if side == "long"
                    else (entry_price - exit_price) / entry_price * cfg.LEVERAGE * 100
                )
                fee_pct = COMMISSION_RATE * 2 * cfg.LEVERAGE * 100
                trades.append({
                    "entry_dt": df.iloc[entry_idx]["dt"],
                    "exit_dt": row["dt"],
                    "side": side,
                    "entry": entry_price,
                    "exit": exit_price,
                    "tp": tp,
                    "sl": sl,
                    "result": result,
                    "pnl_pct": round(raw_pnl - fee_pct, 4),
                    "hold": hold,
                })
                in_trade = False

        # ── 미보유: 다이버전스 진입 신호 체크 ──────────
        if not in_trade:
            # 다이버전스 계산값 유효 여부 확인
            if (
                pd.isna(row["past_swing_low"])
                or pd.isna(row["rsi_at_swing_low"])
                or pd.isna(row["macd_hist_prev"])
                or pd.isna(row["vol_sma20"])
                or pd.isna(row["obv_sma10"])
            ):
                continue

            vol_ok = bool(row["volume"] > row["vol_sma20"] * VOLUME_MULT)

            # 불리시 다이버전스: 가격 신저점 + RSI 상승
            bullish_div = bool(
                row["low"] < row["past_swing_low"]
                and row["rsi"] > row["rsi_at_swing_low"]
            )
            macd_cross_up = bool(
                row["macd_hist"] > 0 and row["macd_hist_prev"] <= 0
            )
            obv_rising = bool(row["obv"] > row["obv_sma10"])

            if bullish_div and macd_cross_up and vol_ok and obv_rising:
                entry_price = float(row["close"])
                atr = float(row["atr"])
                tp = entry_price + atr * TP_ATR_MULT
                sl = entry_price - atr * SL_ATR_MULT
                side, in_trade, entry_idx = "long", True, i
                continue

            # 베어리시 다이버전스: 가격 신고점 + RSI 하락
            if pd.isna(row["rsi_at_swing_high"]):
                continue

            bearish_div = bool(
                row["high"] > row["past_swing_high"]
                and row["rsi"] < row["rsi_at_swing_high"]
            )
            macd_cross_down = bool(
                row["macd_hist"] < 0 and row["macd_hist_prev"] >= 0
            )
            obv_falling = bool(row["obv"] < row["obv_sma10"])

            if bearish_div and macd_cross_down and vol_ok and obv_falling:
                entry_price = float(row["close"])
                atr = float(row["atr"])
                tp = entry_price - atr * TP_ATR_MULT
                sl = entry_price + atr * SL_ATR_MULT
                side, in_trade, entry_idx = "short", True, i

    return trades


# ── 결과 출력 ─────────────────────────────────────────

def _side_stats(df_t: pd.DataFrame, label: str) -> None:
    if df_t.empty:
        print(f"  [{label:5}] 거래 없음")
        return
    profitable = df_t[df_t["pnl_pct"] > 0]
    wr = len(profitable) / len(df_t) * 100
    aw = profitable["pnl_pct"].mean() if len(profitable) else 0.0
    losing = df_t[df_t["pnl_pct"] <= 0]
    al = losing["pnl_pct"].mean() if len(losing) else 0.0
    rr = abs(aw / al) if al != 0 else float("inf")
    wins = len(df_t[df_t["result"] == "win"])
    losses = len(df_t[df_t["result"] == "loss"])
    rsi_exits = len(df_t[df_t["result"] == "rsi_exit"])
    print(
        f"  [{label:5}] {len(df_t):2}건  승률 {wr:.0f}%  "
        f"익절 {aw:+.2f}%  손절 {al:.2f}%  RR {rr:.2f}  "
        f"(TP {wins}/SL {losses}/RSI청산 {rsi_exits})"
    )


def print_report(trades: list[dict], days: int = BACKTEST_DAYS) -> None:
    if not trades:
        print("\n거래 신호 없음 (다이버전스 조건이 너무 엄격할 수 있음)")
        return

    df_t = pd.DataFrame(trades)
    profitable = df_t[df_t["pnl_pct"] > 0]
    losing = df_t[df_t["pnl_pct"] <= 0]
    longs = df_t[df_t["side"] == "long"]
    shorts = df_t[df_t["side"] == "short"]

    win_rate = len(profitable) / len(df_t) * 100
    avg_win = profitable["pnl_pct"].mean() if len(profitable) else 0.0
    avg_loss = losing["pnl_pct"].mean() if len(losing) else 0.0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    avg_hold_h = df_t["hold"].mean() * 15 / 60

    # 누적 잔액 & 최대 낙폭
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    max_dd = 0.0
    for pnl in df_t["pnl_pct"]:
        balance *= 1 + pnl / 100
        peak = max(peak, balance)
        max_dd = max(max_dd, (peak - balance) / peak * 100)

    final_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    ret_arr = df_t["pnl_pct"].values / 100
    sharpe = (
        (ret_arr.mean() / ret_arr.std()) * (len(ret_arr) ** 0.5)
        if ret_arr.std() > 0 else 0.0
    )

    wins = df_t[df_t["result"] == "win"]
    losses_sl = df_t[df_t["result"] == "loss"]
    rsi_exits = df_t[df_t["result"] == "rsi_exit"]
    timeouts = df_t[df_t["result"] == "timeout"]

    print("\n" + "=" * 60)
    print(f"  백테스팅 V3 (RSI 다이버전스)  {days}일 / {cfg.SYMBOL} {cfg.TIMEFRAME} / {cfg.LEVERAGE}x")
    print("=" * 60)
    start = df_t["entry_dt"].iloc[0].strftime("%Y-%m-%d")
    end = df_t["exit_dt"].iloc[-1].strftime("%Y-%m-%d")
    print(f"  기간     : {start} ~ {end}")
    print(
        f"  거래     : {len(df_t)}건  "
        f"(TP {len(wins)} / SL {len(losses_sl)} / RSI청산 {len(rsi_exits)} / 타임아웃 {len(timeouts)})"
    )
    print(f"  승률     : {win_rate:.1f}%  (수익 거래 기준)")
    print(f"  평균 수익 : +{avg_win:.2f}%")
    print(f"  평균 손실 : {avg_loss:.2f}%")
    print(f"  손익비   : {rr:.2f}")
    print(f"  평균 보유 : {avg_hold_h:.1f}시간")
    print(f"  누적 수익 : {final_return:+.1f}%  ({INITIAL_BALANCE:,.0f} → {balance:,.0f} USDT)")
    print(f"  최대 낙폭 : -{max_dd:.1f}%")
    print(f"  Sharpe   : {sharpe:.2f}")
    print("-" * 60)
    _side_stats(longs, "LONG")
    _side_stats(shorts, "SHORT")
    print("=" * 60)

    # V1/V2 대비 요약
    print(f"\n  ── V1/V2 대비 (365일 기준) ──")
    print(f"  V1 (ADX 평균회귀): 80건, 승률 32.5%, 누적 -8.4%, MDD -21.2%, Sharpe -0.35")
    print(f"  V2 (EMA 추세추종): 320건, 승률 36.6%, 누적 +20.3%, MDD -21.4%, Sharpe 0.65")
    print(f"  V3 (RSI 다이버전스): {len(df_t)}건, 승률 {win_rate:.1f}%, 누적 {final_return:+.1f}%, MDD -{max_dd:.1f}%, Sharpe {sharpe:.2f}")

    # 최근 거래
    show = df_t.tail(15)
    print(f"\n  최근 {len(show)}건:")
    print(f"  {'진입일시':>16}  {'방향':>5}  {'청산유형':>10}  {'진입가':>10}  {'청산가':>10}  {'PnL%':>7}")
    print("  " + "-" * 72)
    for _, t in show.iterrows():
        print(
            f"  {t['entry_dt'].strftime('%m-%d %H:%M'):>16}  "
            f"{t['side'].upper():>5}  "
            f"{t['result']:>10}  "
            f"{t['entry']:>10,.0f}  "
            f"{t['exit']:>10,.0f}  "
            f"{t['pnl_pct']:>+7.2f}%"
        )

    out = Path("backtest_v3_trades.csv")
    df_t.to_csv(out, index=False)
    print(f"\n  전체 내역 → {out}")


# ── 진입점 ────────────────────────────────────────────

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else BACKTEST_DAYS
    df_ohlcv = fetch_ohlcv(days)
    df = add_indicators(df_ohlcv)
    trades = run_backtest(df)
    print_report(trades, days)
