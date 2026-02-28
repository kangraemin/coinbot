"""EMA 크로스 + ADX + 펀딩레이트 추세 추종 백테스팅 (V2).

20년차 트레이더 합의 전략:
  롱: EMA20 > EMA50 골든크로스 + ADX > 20 + 펀딩레이트 < 0.05% (과열 아님)
  숏: EMA20 < EMA50 데드크로스 + ADX > 20 + 펀딩레이트 > -0.05% (과열 아님)

  TP: 진입가 ± ATR × 3.0
  SL: 진입가 ∓ ATR × 1.5
  청산: TP/SL/반대크로스/5일 타임아웃

사용법:
  python backtest_v2.py          # 기본 90일
  python backtest_v2.py 180      # 180일
"""

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import pandas as pd
import ta as ta_lib

import config as cfg

# ── 파라미터 ──────────────────────────────────────────
BACKTEST_DAYS: int = 90
COMMISSION_RATE: float = 0.0002
INITIAL_BALANCE: float = 10_000
MAX_HOLD_CANDLES: int = 480    # 최대 보유 5일 (추세 추종은 더 오래 들고 감)

EMA_FAST: int = 20
EMA_SLOW: int = 50
ADX_MIN: float = 20.0
TP_ATR_MULT: float = 3.0
SL_ATR_MULT: float = 1.5
FUNDING_LONG_MAX: float = 0.0005    # 0.05%: 롱 과열 차단
FUNDING_SHORT_MIN: float = -0.0005  # -0.05%: 숏 과열 차단
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


def fetch_funding_rates(days: int = BACKTEST_DAYS) -> pd.DataFrame:
    """펀딩레이트 이력 수집 (8시간 간격, 인증 불필요)."""
    exchange = ccxt.binanceusdm()
    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    all_rates: list = []

    print("펀딩레이트 수집 중", end="", flush=True)
    while True:
        batch = exchange.fetch_funding_rate_history(
            cfg.SYMBOL, since=since_ms, limit=1000
        )
        if not batch:
            break
        all_rates.extend(batch)
        if len(batch) < 1000:
            break
        since_ms = batch[-1]["timestamp"] + 1
        print(".", end="", flush=True)
        time.sleep(0.1)

    print(f" {len(all_rates)}건")
    rows = [
        {"timestamp": r["timestamp"], "funding_rate": r["fundingRate"]}
        for r in all_rates
    ]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp", "funding_rate"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ── 지표 계산 ─────────────────────────────────────────

def add_indicators(
    df_candles: pd.DataFrame, df_funding: pd.DataFrame
) -> pd.DataFrame:
    df = df_candles.copy()
    close, high, low = df["close"], df["high"], df["low"]

    # EMA 크로스
    df["ema_fast"] = ta_lib.trend.EMAIndicator(close, window=EMA_FAST).ema_indicator()
    df["ema_slow"] = ta_lib.trend.EMAIndicator(close, window=EMA_SLOW).ema_indicator()
    df["ema_fast_prev"] = df["ema_fast"].shift(1)
    df["ema_slow_prev"] = df["ema_slow"].shift(1)

    # ADX (14)
    df["adx"] = ta_lib.trend.ADXIndicator(
        high, low, close, window=cfg.ADX_PERIOD
    ).adx()

    # ATR (14)
    df["atr"] = ta_lib.volatility.AverageTrueRange(
        high, low, close, window=cfg.ATR_PERIOD
    ).average_true_range()

    # 크로스 감지: 전봉 EMA 역전 → 현봉 정배열
    df["long_cross"] = (
        (df["ema_fast"] > df["ema_slow"])
        & (df["ema_fast_prev"] <= df["ema_slow_prev"])
    )
    df["short_cross"] = (
        (df["ema_fast"] < df["ema_slow"])
        & (df["ema_fast_prev"] >= df["ema_slow_prev"])
    )

    # 펀딩레이트 병합 — 캔들 시간 기준 직전 값(backward fill)
    if not df_funding.empty:
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            df_funding[["timestamp", "funding_rate"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
    else:
        df["funding_rate"] = 0.0

    df["funding_rate"] = df["funding_rate"].fillna(0.0)
    return df.reset_index(drop=True)


# ── 백테스팅 시뮬레이션 ───────────────────────────────

def run_backtest(df: pd.DataFrame) -> list[dict]:
    """EMA 크로스 신호로 롱/숏 진입/청산 시뮬레이션.

    청산 우선순위: SL > TP > 반대크로스 > 타임아웃
    """
    trades: list[dict] = []
    in_trade = False
    side = "long"
    entry_price = tp = sl = 0.0
    entry_idx = 0

    for i in range(CANDLE_WARMUP, len(df)):
        row = df.iloc[i]

        if pd.isna(row["adx"]) or pd.isna(row["ema_fast"]):
            continue

        # ── 보유 중: 방향별 청산 체크 ────────────────────
        if in_trade:
            hold = i - entry_idx
            exit_price = result = None

            if side == "long":
                hit_sl = bool(row["low"] <= sl)
                hit_tp = bool(row["high"] >= tp)
                opp = bool(row["short_cross"])

                if hit_sl and (not hit_tp or sl <= tp):
                    exit_price, result = sl, "loss"
                elif hit_tp:
                    exit_price, result = tp, "win"
                elif opp:
                    exit_price, result = row["close"], "cross_exit"
                elif hold >= MAX_HOLD_CANDLES:
                    exit_price, result = row["close"], "timeout"
            else:
                hit_sl = bool(row["high"] >= sl)
                hit_tp = bool(row["low"] <= tp)
                opp = bool(row["long_cross"])

                if hit_sl and (not hit_tp or sl >= tp):
                    exit_price, result = sl, "loss"
                elif hit_tp:
                    exit_price, result = tp, "win"
                elif opp:
                    exit_price, result = row["close"], "cross_exit"
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
                    "funding_at_entry": df.iloc[entry_idx]["funding_rate"],
                })
                in_trade = False

        # ── 미보유: EMA 크로스 + 필터 체크 ──────────────
        if not in_trade:
            adx_ok = bool(row["adx"] > ADX_MIN)
            fr = float(row["funding_rate"])

            if row["long_cross"] and adx_ok and fr < FUNDING_LONG_MAX:
                entry_price = row["close"]
                tp = entry_price + row["atr"] * TP_ATR_MULT
                sl = entry_price - row["atr"] * SL_ATR_MULT
                side, in_trade, entry_idx = "long", True, i

            elif row["short_cross"] and adx_ok and fr > FUNDING_SHORT_MIN:
                entry_price = row["close"]
                tp = entry_price - row["atr"] * TP_ATR_MULT
                sl = entry_price + row["atr"] * SL_ATR_MULT
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
    al = df_t[df_t["pnl_pct"] <= 0]["pnl_pct"].mean() if len(df_t[df_t["pnl_pct"] <= 0]) else 0.0
    rr = abs(aw / al) if al != 0 else float("inf")
    wins = len(df_t[df_t["result"] == "win"])
    losses = len(df_t[df_t["result"] == "loss"])
    cx = len(df_t[df_t["result"] == "cross_exit"])
    print(
        f"  [{label:5}] {len(df_t):2}건  승률 {wr:.0f}%  "
        f"익절 {aw:+.2f}%  손절 {al:.2f}%  RR {rr:.2f}  "
        f"(TP {wins}/SL {losses}/크로스청산 {cx})"
    )


def print_report(trades: list[dict], days: int = BACKTEST_DAYS) -> None:
    if not trades:
        print(f"\n거래 신호 없음 (펀딩레이트 필터가 너무 엄격할 수 있음)")
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
    cross_exits = df_t[df_t["result"] == "cross_exit"]
    timeouts = df_t[df_t["result"] == "timeout"]

    print("\n" + "=" * 60)
    print(f"  백테스팅 V2 (추세추종)  {days}일 / {cfg.SYMBOL} {cfg.TIMEFRAME} / {cfg.LEVERAGE}x")
    print("=" * 60)
    start = df_t["entry_dt"].iloc[0].strftime("%Y-%m-%d")
    end = df_t["exit_dt"].iloc[-1].strftime("%Y-%m-%d")
    print(f"  기간     : {start} ~ {end}")
    print(
        f"  거래     : {len(df_t)}건  "
        f"(TP {len(wins)} / SL {len(losses_sl)} / 크로스청산 {len(cross_exits)} / 타임아웃 {len(timeouts)})"
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

    # V1 대비 요약
    print(f"\n  ── V1 (평균회귀) 대비 ──")
    print(f"  V1: 12건, 승률 16.7%, 누적 +0.4%, MDD -5.6%")
    print(f"  V2: {len(df_t)}건, 승률 {win_rate:.1f}%, 누적 {final_return:+.1f}%, MDD -{max_dd:.1f}%")

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

    out = Path("backtest_v2_trades.csv")
    df_t.to_csv(out, index=False)
    print(f"\n  전체 내역 → {out}")


# ── 진입점 ────────────────────────────────────────────

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else BACKTEST_DAYS
    df_ohlcv = fetch_ohlcv(days)
    df_funding = fetch_funding_rates(days)
    df = add_indicators(df_ohlcv, df_funding)
    trades = run_backtest(df)
    print_report(trades, days)
