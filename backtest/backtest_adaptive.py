"""매크로 적응형 4H BB+RSI 전략 그리드 서치.

핵심 아이디어:
  EMA(macro_span) 로 시장 국면 감지
    → 불장(close > ema_macro): 롱만 허용, RSI 조건 완화
    → 하락/중립장(close < ema_macro): 현재 전략 (숏 위주 평균회귀)

파라미터 그리드:
  macro_span      : EMA 스팬 (4H봉 수)  ≈ 500=83일, 800=133일, 1200=200일
  bull_rsi_long   : 불장 롱 진입 RSI (완화)
  bear_rsi_long   : 하락장 롱 진입 RSI (엄격)
  rsi_short       : 숏 진입 RSI
  sl_mult         : SL = ATR × sl_mult
  tp_mode         : atr_2x / atr_3x
  pos_ratio, leverage : 포지션

코인: BTC / ETH / XRP  |  데이터: {coin}_1h_full.parquet → 4H 리샘플
"""

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0

# ── 파라미터 그리드 ────────────────────────────────────────
MACRO_SPANS    = [500, 800, 1200]   # 4H봉 수 ≈ 83일 / 133일 / 200일
BULL_RSI_LONG  = [35, 40, 45]       # 불장 롱 RSI (완화)
BEAR_RSI_LONG  = [20, 25]           # 하락장 롱 RSI (엄격)
RSI_SHORT      = [65, 70]
SL_MULTS       = [1.5, 2.0]
TP_MODES       = ["atr_2x", "atr_3x"]
LEVERAGES      = [3]
POS_RATIOS     = [0.50, 0.70]

COINS = ["btc", "eth", "xrp"]

# 비교 기준 (전체 기간 B&H)
BH_FULL = {"btc": 1419.1, "eth": 539.0, "xrp": 49.4}


# ── 데이터 ────────────────────────────────────────────────

def load_4h(coin: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{coin}_1h_full.parquet")
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df.resample("4h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),     close=("close","last"),
        volume=("volume","sum"),
    ).dropna().reset_index()


def add_indicators(df: pd.DataFrame, macro_span: int) -> pd.DataFrame:
    c  = df["close"].astype(float)
    h  = df["high"].astype(float)
    lo = df["low"].astype(float)

    # BB(20, 2σ)
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std(ddof=1)

    # RSI(14)
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ATR(14)
    tr  = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean()

    # EMA 단기(200봉 ≈ 33일): 불장 필터용
    ema_short = c.ewm(span=200, adjust=False).mean()

    # EMA 매크로 (macro_span봉): 시장 국면 판단
    ema_macro = c.ewm(span=macro_span, adjust=False).mean()

    df = df.copy()
    df["bb_mid"]    = bb_mid
    df["bb_upper"]  = bb_mid + 2 * bb_std
    df["bb_lower"]  = bb_mid - 2 * bb_std
    df["rsi"]       = rsi
    df["atr"]       = atr
    df["ema_short"] = ema_short
    df["ema_macro"] = ema_macro
    return df


# ── 백테스트 코어 ─────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    macro_span: int,
    bull_rsi_long: float,
    bear_rsi_long: float,
    rsi_short: float,
    sl_mult: float,
    tp_mode: str,
    leverage: int,
    pos_ratio: float,
    timeout_bars: int = 48,
    initial_balance: float = INITIAL_BAL,
) -> dict:
    closes    = df["close"].to_numpy(float)
    highs     = df["high"].to_numpy(float)
    lows      = df["low"].to_numpy(float)
    bb_mid    = df["bb_mid"].to_numpy(float)
    bb_upper  = df["bb_upper"].to_numpy(float)
    bb_lower  = df["bb_lower"].to_numpy(float)
    rsi_arr   = df["rsi"].to_numpy(float)
    atr_arr   = df["atr"].to_numpy(float)
    ema_short = df["ema_short"].to_numpy(float)
    ema_macro = df["ema_macro"].to_numpy(float)
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
        if any(np.isnan(x[i]) for x in [bb_mid, rsi_arr, atr_arr, ema_short, ema_macro]):
            continue

        is_bull = closes[i] > ema_macro[i]   # 매크로 불장 여부

        if in_pos:
            exit_p = None

            if direction == 1:  # 롱 청산
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
            else:  # 숏 청산
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

            if is_bull:
                # ── 불장: 롱만, 조건 완화 (close < BB_lower, RSI < bull_rsi_long, close > ema_short)
                if (closes[i] < bb_lower[i] and
                        rsi_arr[i] < bull_rsi_long and
                        closes[i] > ema_short[i]):
                    ep        = closes[i]
                    pos_amt   = notional / ep
                    balance  -= notional * TAKER_FEE
                    sl        = ep - atr_arr[i] * sl_mult
                    tp        = ep + atr_arr[i] * (3.0 if tp_mode == "atr_3x" else 2.0)
                    entry_bar = i
                    in_pos    = True
                    direction = 1

            else:
                # ── 하락/중립장: 숏 우선 + 극단적 RSI일 때만 롱
                # 숏 신호
                if closes[i] > bb_upper[i] and rsi_arr[i] > rsi_short:
                    ep        = closes[i]
                    pos_amt   = notional / ep
                    balance  -= notional * TAKER_FEE
                    sl        = ep + atr_arr[i] * sl_mult
                    tp        = ep - atr_arr[i] * (3.0 if tp_mode == "atr_3x" else 2.0)
                    entry_bar = i
                    in_pos    = True
                    direction = -1

                # 롱 신호 (하락장 극단 반등)
                elif (closes[i] < bb_lower[i] and
                        rsi_arr[i] < bear_rsi_long):
                    ep        = closes[i]
                    pos_amt   = notional / ep
                    balance  -= notional * TAKER_FEE
                    sl        = ep - atr_arr[i] * sl_mult
                    tp        = ep + atr_arr[i] * (3.0 if tp_mode == "atr_3x" else 2.0)
                    entry_bar = i
                    in_pos    = True
                    direction = 1

    if balance <= 0:
        ret = -100.0
    else:
        ret = (balance - initial_balance) / initial_balance * 100

    total_trades = long_trades + short_trades
    total_wins   = long_wins + short_wins
    long_wr  = long_wins  / long_trades  * 100 if long_trades  > 0 else 0.0
    short_wr = short_wins / short_trades * 100 if short_trades > 0 else 0.0
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0.0
    calmar   = ret / mdd if mdd > 0 else 0.0

    return {
        "macro_span":     macro_span,
        "bull_rsi_long":  bull_rsi_long,
        "bear_rsi_long":  bear_rsi_long,
        "rsi_short":      rsi_short,
        "sl_mult":        sl_mult,
        "tp_mode":        tp_mode,
        "leverage":       leverage,
        "pos_ratio":      pos_ratio,
        "trades":         total_trades,
        "long_trades":    long_trades,
        "short_trades":   short_trades,
        "win_rate":       round(win_rate, 1),
        "long_win_rate":  round(long_wr, 1),
        "short_win_rate": round(short_wr, 1),
        "return_pct":     round(ret, 2),
        "max_drawdown":   round(mdd, 2),
        "calmar":         round(calmar, 2),
    }


# ── 워커 ─────────────────────────────────────────────────

def worker(coin: str) -> str:
    print(f"[{coin.upper()}] 시작...", flush=True)

    df_4h_base = load_4h(coin)
    bh_ref     = BH_FULL[coin]

    combos = list(itertools.product(
        MACRO_SPANS, BULL_RSI_LONG, BEAR_RSI_LONG, RSI_SHORT,
        SL_MULTS, TP_MODES, LEVERAGES, POS_RATIOS,
    ))

    results = []
    for macro_sp, bull_rsi_l, bear_rsi_l, rsi_s, sl_m, tp_m, lev, pos in combos:
        df = add_indicators(df_4h_base, macro_sp)
        r  = run_backtest(df, macro_sp, bull_rsi_l, bear_rsi_l, rsi_s, sl_m, tp_m, lev, pos)
        results.append(r)

    df_res = pd.DataFrame(results)

    # 전체 기간 B&H 초과하는 것 중 MDD < 60% 필터
    df_beat = df_res[(df_res["return_pct"] > bh_ref) & (df_res["max_drawdown"] < 60)]
    df_beat = df_beat.sort_values(["calmar", "return_pct"], ascending=False)

    # 전체 결과도 저장 (수익률 top)
    df_all  = df_res.sort_values("return_pct", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_all  = os.path.join(RESULTS_DIR, f"adaptive_{coin}_all.csv")
    out_beat = os.path.join(RESULTS_DIR, f"adaptive_{coin}_beat_bh.csv")
    df_all.to_csv(out_all, index=False)
    df_beat.to_csv(out_beat, index=False)

    print(f"\n[{coin.upper()}] 완료 — {len(results)}조합 | B&H={bh_ref:+.1f}%")
    print(f"  B&H 초과 + MDD<60%: {len(df_beat)}개")
    if len(df_beat) > 0:
        b = df_beat.iloc[0]
        print(
            f"  Best: return={b['return_pct']:+.1f}%  MDD={b['max_drawdown']:.1f}%  "
            f"Calmar={b['calmar']:.2f}  거래={b['trades']}건  "
            f"macro={b['macro_span']} bull_rsi<{b['bull_rsi_long']} "
            f"bear_rsi<{b['bear_rsi_long']} rsi_s>{b['rsi_short']} "
            f"sl×{b['sl_mult']} tp={b['tp_mode']} "
            f"lev={b['leverage']}x pos={b['pos_ratio']*100:.0f}%"
        )
    else:
        b = df_all.iloc[0]
        print(f"  (B&H 초과 없음) Top: return={b['return_pct']:+.1f}%  MDD={b['max_drawdown']:.1f}%")

    return out_beat


def main():
    total = len(list(itertools.product(
        MACRO_SPANS, BULL_RSI_LONG, BEAR_RSI_LONG, RSI_SHORT,
        SL_MULTS, TP_MODES, LEVERAGES, POS_RATIOS,
    )))
    print("매크로 적응형 BB+RSI 전략 그리드 서치")
    print(f"코인: {[c.upper() for c in COINS]} | 조합: {total}개/코인 × {len(COINS)}코인 = {total*len(COINS)}개\n")

    with ProcessPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {executor.submit(worker, c): c for c in COINS}
        for fut in as_completed(futures):
            try:
                print(f"  저장: {fut.result()}")
            except Exception as e:
                print(f"  오류: {e}")


if __name__ == "__main__":
    main()
