"""BTC/USDT:USDT 15m — ADX 레짐 평균회귀 전략 백테스팅 (롱 + 숏).

사용법:
  python backtest.py          # 기본 90일
  python backtest.py 180      # 180일
"""

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ccxt
import pandas as pd
import ta as ta_lib

import config as cfg

# ── 파라미터 ──────────────────────────────────────────
BACKTEST_DAYS: int = 90
COMMISSION_RATE: float = 0.0002   # maker 수수료 0.02% (진입+청산)
INITIAL_BALANCE: float = 10_000  # 가상 초기 잔액 USDT
MAX_HOLD_CANDLES: int = 96        # 최대 보유 96봉 = 24시간 강제 청산
CANDLE_WARMUP: int = 50           # 지표 안정화 워밍업


# ── 데이터 수집 ───────────────────────────────────────

def fetch_ohlcv(days: int = BACKTEST_DAYS) -> pd.DataFrame:
    """Binance USDM에서 과거 15m OHLCV 수집 (인증 불필요)."""
    exchange = ccxt.binanceusdm()
    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    all_ohlcv: list = []
    cutoff_ms = int(time.time() * 1000) - 2 * 15 * 60 * 1000  # 최신 2봉 전까지

    print(f"Binance USDM {cfg.SYMBOL} {cfg.TIMEFRAME} {days}일 수집 중", end="", flush=True)
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

    print(f" {len(all_ohlcv)}개 완료")

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

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """전체 데이터프레임에 전략 지표를 추가한다."""
    close, high, low = df["close"], df["high"], df["low"]

    bb = ta_lib.volatility.BollingerBands(
        close, window=cfg.BB_PERIOD, window_dev=cfg.BB_STD
    )
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()

    df["rsi"] = ta_lib.momentum.RSIIndicator(close, window=cfg.RSI_PERIOD).rsi()
    df["rsi_prev"] = df["rsi"].shift(1)
    df["atr"] = ta_lib.volatility.AverageTrueRange(
        high, low, close, window=cfg.ATR_PERIOD
    ).average_true_range()
    df["adx"] = ta_lib.trend.ADXIndicator(
        high, low, close, window=cfg.ADX_PERIOD
    ).adx()

    return df


# ── 백테스팅 시뮬레이션 ───────────────────────────────

def run_backtest(df: pd.DataFrame) -> list[dict]:
    """캔들을 순회하며 롱/숏 진입/청산을 시뮬레이션한다.

    가정:
    - 신호 발생 캔들의 close 가격으로 limit 주문 (진입가)
    - 이후 캔들의 high/low로 TP(BB 중심선)/SL(ATR×1.5) 체크
    - SL과 TP 동시 도달 시 SL 우선 (보수적)
    - MAX_HOLD_CANDLES 초과 시 close 가격으로 강제 청산

    롱 진입: ADX<25, price≤BB하단, RSI≤35, RSI반등
    숏 진입: ADX<25, price≥BB상단, RSI≥65, RSI하락
    """
    trades: list[dict] = []
    in_trade = False
    side = "long"
    entry_price = tp = sl = 0.0
    entry_idx = 0

    for i in range(CANDLE_WARMUP, len(df)):
        row = df.iloc[i]
        has_indicators = not pd.isna(row["adx"]) and not pd.isna(row["rsi_prev"])

        # ── 보유 중: 방향별 TP/SL 체크 ──────────────────
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
            else:  # short
                hit_sl = bool(row["high"] >= sl)
                hit_tp = bool(row["low"] <= tp)
                if hit_sl and (not hit_tp or sl >= tp):
                    exit_price, result = sl, "loss"
                elif hit_tp:
                    exit_price, result = tp, "win"
                elif hold >= MAX_HOLD_CANDLES:
                    exit_price, result = row["close"], "timeout"

            if result is not None:
                if side == "long":
                    raw_pnl = (exit_price - entry_price) / entry_price * cfg.LEVERAGE * 100
                else:
                    raw_pnl = (entry_price - exit_price) / entry_price * cfg.LEVERAGE * 100
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

        # ── 미보유: 롱/숏 진입 조건 체크 ────────────────
        if not in_trade and has_indicators:
            regime_ok = row["adx"] < cfg.ADX_TREND_THRESHOLD

            # 롱: BB 하단 이탈 + RSI 과매도 + RSI 반등
            long_signal = (
                regime_ok
                and row["close"] <= row["bb_lower"]
                and row["rsi"] <= cfg.RSI_THRESHOLD
                and row["rsi"] > row["rsi_prev"]
                and row["bb_middle"] > row["close"]
            )
            # 숏: BB 상단 이탈 + RSI 과매수 + RSI 하락
            short_signal = (
                regime_ok
                and row["close"] >= row["bb_upper"]
                and row["rsi"] >= cfg.RSI_OVERBOUGHT
                and row["rsi"] < row["rsi_prev"]
                and row["bb_middle"] < row["close"]
            )

            if long_signal:
                entry_price = row["close"]
                tp = row["bb_middle"]
                sl = entry_price - row["atr"] * cfg.SL_ATR_MULT
                side, in_trade, entry_idx = "long", True, i
            elif short_signal:
                entry_price = row["close"]
                tp = row["bb_middle"]
                sl = entry_price + row["atr"] * cfg.SL_ATR_MULT
                side, in_trade, entry_idx = "short", True, i

    return trades


# ── 결과 출력 ─────────────────────────────────────────

def _side_stats(df_t: pd.DataFrame, label: str) -> None:
    """롱 또는 숏 단독 통계를 출력한다."""
    if df_t.empty:
        print(f"  [{label}] 거래 없음")
        return
    wins = df_t[df_t["result"] == "win"]
    losses = df_t[df_t["result"] == "loss"]
    wr = len(wins) / len(df_t) * 100
    aw = wins["pnl_pct"].mean() if len(wins) else 0.0
    al = losses["pnl_pct"].mean() if len(losses) else 0.0
    rr = abs(aw / al) if al != 0 else float("inf")
    print(
        f"  [{label:5}] {len(df_t):2}건  승률 {wr:.0f}%  "
        f"익절 {aw:+.2f}%  손절 {al:.2f}%  RR {rr:.2f}"
    )


def print_report(trades: list[dict], days: int = BACKTEST_DAYS) -> None:
    if not trades:
        print("\n거래 신호 없음 — 조건 강화 또는 기간 확대 고려")
        return

    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["result"] == "win"]
    losses = df_t[df_t["result"] == "loss"]
    timeouts = df_t[df_t["result"] == "timeout"]
    longs = df_t[df_t["side"] == "long"]
    shorts = df_t[df_t["side"] == "short"]

    win_rate = len(wins) / len(df_t) * 100
    avg_win = wins["pnl_pct"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl_pct"].mean() if len(losses) else 0.0
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

    # 간이 Sharpe (per-trade)
    ret_arr = df_t["pnl_pct"].values / 100
    sharpe = (
        (ret_arr.mean() / ret_arr.std()) * (len(ret_arr) ** 0.5)
        if ret_arr.std() > 0 else 0.0
    )

    print("\n" + "=" * 56)
    print(f"  백테스팅 결과  {days}일 / {cfg.SYMBOL} {cfg.TIMEFRAME} / {cfg.LEVERAGE}x")
    print("=" * 56)
    start = df_t["entry_dt"].iloc[0].strftime("%Y-%m-%d")
    end = df_t["exit_dt"].iloc[-1].strftime("%Y-%m-%d")
    print(f"  기간     : {start} ~ {end}")
    print(f"  거래     : {len(df_t)}건  (익절 {len(wins)} / 손절 {len(losses)} / 타임아웃 {len(timeouts)})")
    print(f"  승률     : {win_rate:.1f}%")
    print(f"  평균 익절 : +{avg_win:.2f}%")
    print(f"  평균 손절 : {avg_loss:.2f}%")
    print(f"  손익비   : {rr:.2f}")
    print(f"  평균 보유 : {avg_hold_h:.1f}시간")
    print(f"  누적 수익 : {final_return:+.1f}%  ({INITIAL_BALANCE:,.0f} → {balance:,.0f} USDT)")
    print(f"  최대 낙폭 : -{max_dd:.1f}%")
    print(f"  Sharpe   : {sharpe:.2f}")
    print("-" * 56)
    _side_stats(longs, "LONG")
    _side_stats(shorts, "SHORT")
    print("=" * 56)

    # 최근 거래
    show = df_t.tail(15)
    print(f"\n  최근 {len(show)}건:")
    print(f"  {'진입일시':>16}  {'방향':>5}  {'결과':>8}  {'진입가':>10}  {'청산가':>10}  {'PnL%':>7}")
    print("  " + "-" * 68)
    for _, t in show.iterrows():
        print(
            f"  {t['entry_dt'].strftime('%m-%d %H:%M'):>16}  "
            f"{t['side'].upper():>5}  "
            f"{t['result']:>8}  "
            f"{t['entry']:>10,.0f}  "
            f"{t['exit']:>10,.0f}  "
            f"{t['pnl_pct']:>+7.2f}%"
        )

    # CSV 저장
    out = Path("backtest_trades.csv")
    df_t.to_csv(out, index=False)
    print(f"\n  전체 내역 → {out}")


# ── 진입점 ────────────────────────────────────────────

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else BACKTEST_DAYS
    df = fetch_ohlcv(days)
    df = add_indicators(df)
    trades = run_backtest(df)
    print_report(trades, days)
