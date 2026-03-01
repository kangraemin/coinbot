"""롱숏 양방향 BB+RSI 평균회귀 백테스트 — 4h봉, 2022~2025.

롱 진입: close < BB_lower(20,2σ) AND RSI(14) < rsi_long_thresh
숏 진입: close > BB_upper(20,2σ) AND RSI(14) > rsi_short_thresh
청산: BB 중심선 / ATR×mult / 타임아웃(48봉)
EMA200=True: 롱은 price>EMA200, 숏은 price<EMA200

코인: BTC, ETH, XRP | 기간: 2022~2025
"""

import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0

START_DATE = "2022-01-01"
END_DATE   = "2025-12-31"

# 그리드
RSI_LONG   = [25, 30, 35]
RSI_SHORT  = [65, 70, 75]
SL_MULTS   = [1.0, 1.5, 2.0]
TP_MODES   = ["bb_mid", "atr_2x", "atr_3x"]
LEVERAGES  = [3, 5, 7]
POS_RATIOS = [0.1, 0.2, 0.3]
USE_EMA200 = [True, False]

COINS = ["btc", "eth", "xrp"]

# B&H 벤치마크 (2022~2025 참고치)
BH_REF = {"btc": 89.4, "eth": -19.4, "xrp": 121.6}


# ── 데이터 로드 / 인디케이터 ──────────────────────────

def load_range(coin: str) -> pd.DataFrame:
    """2021 마지막 60일 + 2022~2025 전체 로드 (EMA200 웜업)."""
    frames = []
    for y in range(2021, 2026):
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
    mdd = 0.0

    in_pos    = False
    direction = 0       # 1=롱, -1=숏
    ep = sl = tp = pos_amt = 0.0
    entry_bar = 0

    long_trades = long_wins = 0
    short_trades = short_wins = 0

    for i in range(1, n):
        if any(np.isnan(x[i]) for x in [bb_mid, bb_upper, bb_lower, rsi_arr, atr_arr, ema200]):
            continue

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

            else:  # 숏 청산 (direction == -1)
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
                fee     = exit_p * pos_amt * TAKER_FEE
                balance += pnl - fee
                in_pos   = False
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
            ema_long_ok = (not use_ema200) or (closes[i] > ema200[i])
            if ema_long_ok and closes[i] < bb_lower[i] and rsi_arr[i] < rsi_long:
                ep        = closes[i]
                pos_amt   = notional / ep
                balance  -= notional * TAKER_FEE
                sl        = ep - atr_arr[i] * sl_mult
                tp        = (ep + atr_arr[i] * 2.0) if tp_mode == "atr_2x" else \
                            (ep + atr_arr[i] * 3.0) if tp_mode == "atr_3x" else 0.0
                entry_bar = i
                in_pos    = True
                direction = 1
                continue

            # 숏 신호
            ema_short_ok = (not use_ema200) or (closes[i] < ema200[i])
            if ema_short_ok and closes[i] > bb_upper[i] and rsi_arr[i] > rsi_short:
                ep        = closes[i]
                pos_amt   = notional / ep
                balance  -= notional * TAKER_FEE
                sl        = ep + atr_arr[i] * sl_mult
                tp        = (ep - atr_arr[i] * 2.0) if tp_mode == "atr_2x" else \
                            (ep - atr_arr[i] * 3.0) if tp_mode == "atr_3x" else 0.0
                entry_bar = i
                in_pos    = True
                direction = -1

    if balance <= 0:
        ret = -100.0
    else:
        ret = (balance - initial_balance) / initial_balance * 100

    total_trades = long_trades + short_trades
    long_wr  = long_wins  / long_trades  * 100 if long_trades  > 0 else 0.0
    short_wr = short_wins / short_trades * 100 if short_trades > 0 else 0.0
    calmar   = ret / mdd if mdd > 0 else 0.0

    return {
        "rsi_long":     rsi_long,
        "rsi_short":    rsi_short,
        "sl_mult":      sl_mult,
        "tp_mode":      tp_mode,
        "leverage":     leverage,
        "pos_ratio":    pos_ratio,
        "use_ema200":   use_ema200,
        "trades":       total_trades,
        "trades_per_yr": round(total_trades / 4, 1),
        "long_trades":  long_trades,
        "short_trades": short_trades,
        "long_win_rate":  round(long_wr, 1),
        "short_win_rate": round(short_wr, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(mdd, 2),
        "calmar":       round(calmar, 2),
    }


# ── 워커 ──────────────────────────────────────────────

def worker(coin: str) -> str:
    print(f"[{coin.upper()} 4h] 시작...", flush=True)

    df_1m = load_range(coin)
    df_4h = resample_4h(df_1m)
    df_4h = compute_indicators(df_4h)

    s = pd.Timestamp(START_DATE, tz="UTC")
    e = pd.Timestamp(END_DATE + " 23:59:59", tz="UTC")
    df_eval = df_4h[(df_4h["timestamp"] >= s) & (df_4h["timestamp"] <= e)].reset_index(drop=True)

    combos = list(itertools.product(
        RSI_LONG, RSI_SHORT, SL_MULTS, TP_MODES, LEVERAGES, POS_RATIOS, USE_EMA200,
    ))

    results = []
    for rsi_l, rsi_s, sl_m, tp_m, lev, pos, ema in combos:
        r = run_backtest(df_eval, rsi_l, rsi_s, sl_m, tp_m, lev, pos, ema)
        results.append(r)

    df_res = pd.DataFrame(results).sort_values("return_pct", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"long_short_{coin}_4h.csv")
    df_res.to_csv(out_path, index=False)

    bh = BH_REF[coin]
    df20 = df_res[df_res["trades"] >= 20]
    best = df20.iloc[0] if len(df20) > 0 else df_res.iloc[0]
    beat_bh = "✓ B&H 초과" if best["return_pct"] > bh else "✗ B&H 미달"

    print(
        f"[{coin.upper()} 4h] 완료 — {len(results)}조합 | "
        f"Best(≥20거래): 수익={best['return_pct']:+.1f}%  MDD={best['max_drawdown']:.1f}%  "
        f"Calmar={best['calmar']:.2f}  거래={best['trades']}건({best['trades_per_yr']}/년)  "
        f"L={best['long_trades']}건({best['long_win_rate']:.0f}%) "
        f"S={best['short_trades']}건({best['short_win_rate']:.0f}%) | "
        f"RSI_L<{best['rsi_long']} RSI_S>{best['rsi_short']} "
        f"sl×{best['sl_mult']} tp={best['tp_mode']} "
        f"lev={best['leverage']}x pos={best['pos_ratio']*100:.0f}% "
        f"EMA200={'ON' if best['use_ema200'] else 'OFF'} | "
        f"B&H={bh:+.1f}% → {beat_bh}",
        flush=True,
    )
    return out_path


def main():
    total = len(list(itertools.product(
        RSI_LONG, RSI_SHORT, SL_MULTS, TP_MODES, LEVERAGES, POS_RATIOS, USE_EMA200,
    )))
    print("롱숏 양방향 BB+RSI 평균회귀 — 4h봉, 2022~2025")
    print(f"코인: {[c.upper() for c in COINS]}")
    print(f"조합: {total}개 × {len(COINS)}코인 = {total*len(COINS)}개\n")

    with ProcessPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {executor.submit(worker, c): c for c in COINS}
        for fut in as_completed(futures):
            coin = futures[fut]
            try:
                print(f"  저장: {fut.result()}")
            except Exception as e:
                print(f"  [{coin}] 오류: {e}")


if __name__ == "__main__":
    main()
