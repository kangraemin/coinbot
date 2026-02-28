"""STARS v1.0 — 멀티 레짐 전략 백테스팅 (V4).

전략:
  레짐 판단 (ADX 기반):
    - ADX > 25 → 추세 레짐: EMA 방향 + HH/HL(롱) / LH/LL(숏) + 풀백 + RSI + 볼륨
    - ADX < 20 + BB 스퀴즈 → 브레이크아웃 레짐: BB 돌파 + 볼륨
    - 그 외 → 관망

  청산:
    - SL = entry ∓ ATR × 1.5
    - TP = 스윙 구조 기반 (추세) 또는 ATR × 2.25 (브레이크아웃)
    - MAX_HOLD = 240봉 (60시간)

사용법:
  python backtest_v4.py          # 기본 90일
  python backtest_v4.py 365      # 365일
"""

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ccxt
import numpy as np
import pandas as pd
import ta as ta_lib

import config as cfg

# ── 파라미터 ──────────────────────────────────────────
BACKTEST_DAYS: int = 90
COMMISSION_RATE: float = 0.0002
INITIAL_BALANCE: float = 10_000

# 레짐
ADX_PERIOD: int = 14
ADX_TREND: float = 25.0
ADX_RANGE: float = 20.0

# 구조
SWING_PERIOD: int = 20
EMA_FAST: int = 20
EMA_SLOW: int = 50

# 진입
RSI_PERIOD: int = 14
RSI_MAX_TREND: float = 70.0
RSI_MIN_TREND: float = 30.0
VOL_PERIOD: int = 20
SUPPORT_ATR: float = 0.5

# BB (브레이크아웃용)
BB_PERIOD: int = 20
BB_STD: float = 2.0
BB_SQUEEZE_RATIO: float = 0.5

# 청산
SL_ATR_MULT: float = 1.5
TP_TREND_ATR_MULT: float = 2.5
TP_BREAKOUT_ATR_MULT: float = 2.25
MIN_RR: float = 1.5
MAX_HOLD_CANDLES: int = 240

CANDLE_WARMUP: int = 100


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

    # EMA
    df["ema_fast"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=EMA_SLOW, adjust=False).mean()

    # ADX(14)
    df["adx"] = ta_lib.trend.ADXIndicator(
        high, low, close, window=ADX_PERIOD
    ).adx()

    # RSI(14)
    rsi_ind = ta_lib.momentum.RSIIndicator(close, window=RSI_PERIOD)
    df["rsi"] = rsi_ind.rsi()

    # ATR(14)
    df["atr"] = ta_lib.volatility.AverageTrueRange(
        high, low, close, window=14
    ).average_true_range()

    # BB
    bb = ta_lib.volatility.BollingerBands(
        close, window=BB_PERIOD, window_dev=BB_STD
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_width_ma"] = df["bb_width"].rolling(window=BB_PERIOD).mean()

    # 볼륨 이동평균
    df["vol_ma"] = volume.rolling(window=VOL_PERIOD).mean()

    # 스윙 고점/저점 (최근 SWING_PERIOD봉)
    df["swing_high"] = high.rolling(window=SWING_PERIOD).max()
    df["swing_low"] = low.rolling(window=SWING_PERIOD).min()

    # 이전 스윙 (shift로 과거 스윙 참조)
    df["prev_swing_high"] = df["swing_high"].shift(SWING_PERIOD)
    df["prev_swing_low"] = df["swing_low"].shift(SWING_PERIOD)

    return df.reset_index(drop=True)


# ── 백테스팅 시뮬레이션 ───────────────────────────────

def run_backtest(df: pd.DataFrame) -> list[dict]:
    """STARS v1.0 멀티레짐 진입/청산 시뮬레이션."""
    trades: list[dict] = []
    in_trade = False
    side = "long"
    entry_price = tp = sl = 0.0
    entry_idx = 0
    trade_regime = "trend"

    required_cols = ["adx", "ema_fast", "ema_slow", "rsi", "atr",
                     "bb_upper", "bb_lower", "bb_width", "bb_width_ma",
                     "vol_ma", "swing_high", "swing_low",
                     "prev_swing_high", "prev_swing_low"]

    for i in range(CANDLE_WARMUP, len(df)):
        row = df.iloc[i]

        if any(pd.isna(row[c]) for c in required_cols):
            continue

        # ── 보유 중: 청산 체크 ────────────────────
        if in_trade:
            hold = i - entry_idx
            exit_price = result = None

            if side == "long":
                hit_sl = bool(row["low"] <= sl)
                hit_tp = bool(row["high"] >= tp)

                if hit_sl and (not hit_tp or sl <= tp):
                    exit_price, result = sl, "loss"
                elif hit_tp:
                    exit_price, result = tp, "win"
                elif hold >= MAX_HOLD_CANDLES:
                    exit_price, result = row["close"], "timeout"
            else:
                hit_sl = bool(row["high"] >= sl)
                hit_tp = bool(row["low"] <= tp)

                if hit_sl and (not hit_tp or sl >= tp):
                    exit_price, result = sl, "loss"
                elif hit_tp:
                    exit_price, result = tp, "win"
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
                    "regime": trade_regime,
                    "entry": entry_price,
                    "exit": exit_price,
                    "tp": tp,
                    "sl": sl,
                    "result": result,
                    "pnl_pct": round(raw_pnl - fee_pct, 4),
                    "hold": hold,
                })
                in_trade = False

        # ── 미보유: 신호 체크 ────────────────────
        if in_trade:
            continue

        adx = float(row["adx"])
        atr = float(row["atr"])

        # 레짐 판단
        is_trend = adx > ADX_TREND
        is_range_adx = adx < ADX_RANGE
        is_squeeze = bool(row["bb_width"] < row["bb_width_ma"] * BB_SQUEEZE_RATIO)
        is_breakout_regime = is_range_adx and is_squeeze

        # ── 추세 레짐 진입 ──────────────────────
        if is_trend:
            trend_long = (
                bool(row["ema_fast"] > row["ema_slow"])
                and bool(row["swing_high"] > row["prev_swing_high"])  # HH
                and bool(row["swing_low"] > row["prev_swing_low"])    # HL
                and bool(row["close"] <= row["swing_low"] + atr * SUPPORT_ATR)
                and bool(row["rsi"] < RSI_MAX_TREND)
                and bool(row["volume"] > row["vol_ma"])
            )
            trend_short = (
                bool(row["ema_fast"] < row["ema_slow"])
                and bool(row["swing_high"] < row["prev_swing_high"])  # LH
                and bool(row["swing_low"] < row["prev_swing_low"])    # LL
                and bool(row["close"] >= row["swing_high"] - atr * SUPPORT_ATR)
                and bool(row["rsi"] > RSI_MIN_TREND)
                and bool(row["volume"] > row["vol_ma"])
            )

            if trend_long:
                entry_price = float(row["close"])
                sl = entry_price - atr * SL_ATR_MULT
                tp_swing = float(row["swing_high"])
                tp_atr = entry_price + atr * TP_TREND_ATR_MULT
                tp = min(tp_swing, tp_atr)
                rr = abs(tp - entry_price) / abs(entry_price - sl)
                if rr < MIN_RR:
                    continue
                side, in_trade, entry_idx, trade_regime = "long", True, i, "trend"
                continue

            if trend_short:
                entry_price = float(row["close"])
                sl = entry_price + atr * SL_ATR_MULT
                tp_swing = float(row["swing_low"])
                tp_atr = entry_price - atr * TP_TREND_ATR_MULT
                tp = max(tp_swing, tp_atr)
                rr = abs(entry_price - tp) / abs(sl - entry_price)
                if rr < MIN_RR:
                    continue
                side, in_trade, entry_idx, trade_regime = "short", True, i, "trend"
                continue

        # ── 브레이크아웃 레짐 진입 ──────────────
        if is_breakout_regime:
            breakout_long = (
                bool(row["close"] > row["bb_upper"])
                and bool(row["volume"] > row["vol_ma"])
            )
            breakout_short = (
                bool(row["close"] < row["bb_lower"])
                and bool(row["volume"] > row["vol_ma"])
            )

            if breakout_long:
                entry_price = float(row["close"])
                sl = entry_price - atr * SL_ATR_MULT
                tp = entry_price + atr * TP_BREAKOUT_ATR_MULT
                rr = abs(tp - entry_price) / abs(entry_price - sl)
                if rr < MIN_RR:
                    continue
                side, in_trade, entry_idx, trade_regime = "long", True, i, "breakout"
                continue

            if breakout_short:
                entry_price = float(row["close"])
                sl = entry_price + atr * SL_ATR_MULT
                tp = entry_price - atr * TP_BREAKOUT_ATR_MULT
                rr = abs(entry_price - tp) / abs(sl - entry_price)
                if rr < MIN_RR:
                    continue
                side, in_trade, entry_idx, trade_regime = "short", True, i, "breakout"

    return trades


# ── 결과 출력 ─────────────────────────────────────────

def _regime_stats(df_t: pd.DataFrame, label: str) -> None:
    sub = df_t[df_t["regime"] == label.strip().lower()]
    if sub.empty:
        print(f"  [{label:10}] 거래 없음")
        return
    profitable = sub[sub["pnl_pct"] > 0]
    losing = sub[sub["pnl_pct"] <= 0]
    wr = len(profitable) / len(sub) * 100
    aw = profitable["pnl_pct"].mean() if len(profitable) else 0.0
    al = losing["pnl_pct"].mean() if len(losing) else 0.0
    rr = abs(aw / al) if al != 0 else float("inf")
    print(
        f"  [{label:10}] {len(sub):3}건  승률 {wr:.0f}%  "
        f"익절 {aw:+.2f}%  손절 {al:.2f}%  RR {rr:.2f}"
    )


def _side_stats(df_t: pd.DataFrame, label: str) -> None:
    sub = df_t[df_t["side"] == label.lower()]
    if sub.empty:
        print(f"  [{label:5}] 거래 없음")
        return
    profitable = sub[sub["pnl_pct"] > 0]
    losing = sub[sub["pnl_pct"] <= 0]
    wr = len(profitable) / len(sub) * 100
    aw = profitable["pnl_pct"].mean() if len(profitable) else 0.0
    al = losing["pnl_pct"].mean() if len(losing) else 0.0
    rr = abs(aw / al) if al != 0 else float("inf")
    wins = len(sub[sub["result"] == "win"])
    losses = len(sub[sub["result"] == "loss"])
    print(
        f"  [{label:5}] {len(sub):3}건  승률 {wr:.0f}%  "
        f"익절 {aw:+.2f}%  손절 {al:.2f}%  RR {rr:.2f}  "
        f"(TP {wins}/SL {losses})"
    )


def print_report(trades: list[dict], days: int = BACKTEST_DAYS) -> None:
    if not trades:
        print("\n거래 신호 없음 (레짐 조건이 충족되지 않음)")
        return

    df_t = pd.DataFrame(trades)
    profitable = df_t[df_t["pnl_pct"] > 0]
    losing = df_t[df_t["pnl_pct"] <= 0]

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
    timeouts = df_t[df_t["result"] == "timeout"]

    print("\n" + "=" * 60)
    print(f"  백테스팅 V4 (STARS v1.0)  {days}일 / {cfg.SYMBOL} {cfg.TIMEFRAME} / {cfg.LEVERAGE}x")
    print("=" * 60)
    start = df_t["entry_dt"].iloc[0].strftime("%Y-%m-%d")
    end = df_t["exit_dt"].iloc[-1].strftime("%Y-%m-%d")
    print(f"  기간     : {start} ~ {end}")
    print(
        f"  거래     : {len(df_t)}건  "
        f"(TP {len(wins)} / SL {len(losses_sl)} / 타임아웃 {len(timeouts)})"
    )
    print(f"  승률     : {win_rate:.1f}%")
    print(f"  평균 수익 : +{avg_win:.2f}%")
    print(f"  평균 손실 : {avg_loss:.2f}%")
    print(f"  손익비   : {rr:.2f}")
    print(f"  평균 보유 : {avg_hold_h:.1f}시간")
    print(f"  누적 수익 : {final_return:+.1f}%  ({INITIAL_BALANCE:,.0f} → {balance:,.0f} USDT)")
    print(f"  최대 낙폭 : -{max_dd:.1f}%")
    print(f"  Sharpe   : {sharpe:.2f}")
    print("-" * 60)
    print("  ── 레짐별 ──")
    _regime_stats(df_t, "TREND    ")
    _regime_stats(df_t, "BREAKOUT ")
    print("-" * 60)
    print("  ── 방향별 ──")
    _side_stats(df_t, "long")
    _side_stats(df_t, "short")

    # V1~V4 비교표
    print("\n" + "=" * 56)
    print(f"  V1 (ADX 평균회귀):    80건  -8.4%  MDD -21.2%  Sharpe -0.35")
    print(f"  V2 (EMA 추세추종):   320건 +20.3%  MDD -21.4%  Sharpe  0.65")
    print(f"  V3 (RSI 다이버전스):  45건  +7.6%  MDD  -8.3%  Sharpe  0.61")
    print(
        f"  V4 (STARS v1.0):   {len(df_t):3}건  {final_return:+.1f}%  "
        f"MDD -{max_dd:.1f}%  Sharpe {sharpe:.2f}"
    )
    print("=" * 56)

    # 최근 거래
    show = df_t.tail(15)
    print(f"\n  최근 {len(show)}건:")
    print(f"  {'진입일시':>16}  {'방향':>5}  {'레짐':>9}  {'청산유형':>8}  {'진입가':>10}  {'청산가':>10}  {'PnL%':>7}")
    print("  " + "-" * 80)
    for _, t in show.iterrows():
        print(
            f"  {t['entry_dt'].strftime('%m-%d %H:%M'):>16}  "
            f"{t['side'].upper():>5}  "
            f"{t['regime']:>9}  "
            f"{t['result']:>8}  "
            f"{t['entry']:>10,.0f}  "
            f"{t['exit']:>10,.0f}  "
            f"{t['pnl_pct']:>+7.2f}%"
        )

    out = Path("backtest_v4_trades.csv")
    df_t.to_csv(out, index=False)
    print(f"\n  전체 내역 → {out}")


# ── 진입점 ────────────────────────────────────────────

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else BACKTEST_DAYS
    df_ohlcv = fetch_ohlcv(days)
    df = add_indicators(df_ohlcv)
    trades = run_backtest(df)
    print_report(trades, days)
