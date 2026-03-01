"""RSI + 볼린저밴드 과매도 반등 전략 — 1h 그리드 서치.

전략:
  - close < BB_lower AND RSI < rsi_thresh AND volume > vol_MA × vol_mult → 롱 진입
  - TP: BB 중간선 도달(bb_mid 모드) 또는 ATR × tp_atr_mult
  - SL: 진입가 - ATR(14) × sl_atr_mult
  - 타임아웃: 48봉 강제 청산

1분봉 데이터를 1시간봉으로 리샘플링해서 사용.
"""

import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE_RATE = 0.0005

# ── 파라미터 그리드 ──────────────────────────────────
RSI_THRESH   = [25, 30, 35]
VOL_MULT     = [1.2, 1.5, 2.0]
SL_ATR_MULT  = [1.0, 1.5, 2.0]
TP_ATR_MULT  = [2.0, 3.0]
TP_MODES     = ["bb_mid", "atr"]
LEVERAGES    = [3, 5, 7]
POS_RATIOS   = [0.1, 0.2, 0.3]

COINS = ["btc", "eth", "sol", "xrp"]


def load_1m(coin: str) -> pd.DataFrame:
    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.startswith(f"{coin}_1m_") and f.endswith(".parquet")
    ])
    if not files:
        raise FileNotFoundError(f"{coin} 1분봉 데이터 없음: {DATA_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    """1분봉 → 1시간봉으로 리샘플링."""
    resampled = df.resample("1h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return resampled.reset_index()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """BB(20,2σ), RSI(14), vol_MA(20), ATR(14) 계산."""
    closes = df["close"].to_numpy(dtype=float)
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    volumes = df["volume"].to_numpy(dtype=float)
    n = len(closes)

    # Bollinger Bands (20, 2σ)
    bb_period = 20
    bb_mid  = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    for i in range(bb_period - 1, n):
        window = closes[i - bb_period + 1 : i + 1]
        m = window.mean()
        s = window.std(ddof=1)
        bb_mid[i]   = m
        bb_upper[i] = m + 2 * s
        bb_lower[i] = m - 2 * s

    # RSI (14)
    rsi_period = 14
    rsi = np.full(n, np.nan)
    if n > rsi_period:
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains[:rsi_period].mean()
        avg_loss = losses[:rsi_period].mean()
        for i in range(rsi_period, n):
            avg_gain = (avg_gain * (rsi_period - 1) + gains[i - 1]) / rsi_period
            avg_loss = (avg_loss * (rsi_period - 1) + losses[i - 1]) / rsi_period
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    # Volume MA (20)
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period - 1, n):
        vol_ma[i] = volumes[i - vol_period + 1 : i + 1].mean()

    # ATR (14)
    atr_period = 14
    atr = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    if n > atr_period:
        atr[atr_period] = tr[1 : atr_period + 1].mean()
        for i in range(atr_period + 1, n):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

    df = df.copy()
    df["bb_mid"]   = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["rsi"]      = rsi
    df["vol_ma"]   = vol_ma
    df["atr"]      = atr
    return df


def run_backtest(
    df: pd.DataFrame,
    rsi_thresh: float,
    vol_mult: float,
    sl_atr_mult: float,
    tp_atr_mult: float,
    tp_mode: str,
    leverage: int,
    pos_ratio: float,
    timeout_bars: int = 48,
    initial_balance: float = 1000.0,
) -> dict:
    closes   = df["close"].to_numpy(dtype=float)
    highs    = df["high"].to_numpy(dtype=float)
    lows     = df["low"].to_numpy(dtype=float)
    volumes  = df["volume"].to_numpy(dtype=float)
    bb_mid   = df["bb_mid"].to_numpy(dtype=float)
    bb_lower = df["bb_lower"].to_numpy(dtype=float)
    rsi_arr  = df["rsi"].to_numpy(dtype=float)
    vol_ma   = df["vol_ma"].to_numpy(dtype=float)
    atr_arr  = df["atr"].to_numpy(dtype=float)
    n = len(closes)

    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    trades = wins = 0
    total_pnl = 0.0

    in_position = False
    entry_price = sl_price = tp_price = 0.0
    bb_mid_at_entry = 0.0
    position_amount = 0.0
    entry_bar = 0

    # 지표가 모두 준비되는 인덱스부터 시작
    start = 34  # max(bb_period=20, rsi=14, vol_ma=20, atr=14) + 여유

    for i in range(start, n):
        if np.isnan(bb_lower[i]) or np.isnan(rsi_arr[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_arr[i]):
            continue

        if not in_position:
            if balance <= 0:
                break
            # 진입 조건: BB 하단 이탈 + RSI 과매도 + 볼륨 급증
            if (closes[i] < bb_lower[i] and
                    rsi_arr[i] < rsi_thresh and
                    volumes[i] > vol_ma[i] * vol_mult):
                entry_price = closes[i]
                margin = balance * pos_ratio
                notional = margin * leverage
                position_amount = notional / entry_price
                fee = notional * TAKER_FEE_RATE
                balance -= fee

                atr_val = atr_arr[i]
                sl_price = entry_price - atr_val * sl_atr_mult
                if tp_mode == "atr":
                    tp_price = entry_price + atr_val * tp_atr_mult
                else:
                    tp_price = 0.0  # bb_mid 모드: 동적으로 체크
                bb_mid_at_entry = bb_mid[i]
                entry_bar = i
                in_position = True
        else:
            exit_price = None
            won = False

            # bb_mid 모드: 현재 bb_mid 기준으로 TP 체크
            if tp_mode == "bb_mid":
                tp_check = bb_mid[i]
            else:
                tp_check = tp_price

            # SL 우선 (같은 봉에서 SL+TP 동시 도달 시 SL)
            if lows[i] <= sl_price:
                exit_price = sl_price
                won = False
            elif highs[i] >= tp_check and tp_check > entry_price:
                exit_price = tp_check
                won = True
            # 타임아웃: 48봉 초과
            elif i - entry_bar >= timeout_bars:
                exit_price = closes[i]
                won = exit_price > entry_price

            if exit_price is not None:
                pnl = (exit_price - entry_price) * position_amount
                fee = exit_price * position_amount * TAKER_FEE_RATE
                balance += pnl - fee
                total_pnl += pnl - fee
                trades += 1
                if won:
                    wins += 1
                in_position = False

            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100
            if dd > max_drawdown:
                max_drawdown = dd

    if balance <= 0:
        ret = -100.0
    else:
        ret = (balance - initial_balance) / initial_balance * 100

    win_rate = wins / trades * 100 if trades > 0 else 0.0
    calmar = ret / max_drawdown if max_drawdown > 0 else 0.0

    return {
        "rsi_thresh":   rsi_thresh,
        "vol_mult":     vol_mult,
        "sl_atr_mult":  sl_atr_mult,
        "tp_atr_mult":  tp_atr_mult,
        "tp_mode":      tp_mode,
        "leverage":     leverage,
        "pos_ratio":    pos_ratio,
        "trades":       trades,
        "win_rate":     round(win_rate, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(max_drawdown, 2),
        "calmar":       round(calmar, 2),
    }


def worker(coin: str) -> str:
    print(f"[{coin.upper()} 1h] 시작...", flush=True)

    try:
        df_1m = load_1m(coin)
        df = resample_1h(df_1m)
        df = compute_indicators(df)
    except Exception as e:
        print(f"[{coin.upper()} 1h] 데이터 로드 실패: {e}", flush=True)
        return f"{coin}/1h: FAILED"

    combos = list(itertools.product(
        RSI_THRESH, VOL_MULT, SL_ATR_MULT, TP_ATR_MULT, TP_MODES, LEVERAGES, POS_RATIOS
    ))

    results = []
    for rsi_thresh, vol_mult, sl_atr, tp_atr, tp_mode, leverage, pos_ratio in combos:
        r = run_backtest(df, rsi_thresh, vol_mult, sl_atr, tp_atr, tp_mode, leverage, pos_ratio)
        results.append(r)

    df_res = pd.DataFrame(results).sort_values("return_pct", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"bb_rsi_reversion_{coin}_1h.csv")
    df_res.to_csv(out_path, index=False)

    best = df_res.iloc[0]
    print(
        f"[{coin.upper()} 1h] 완료 — {len(results)}조합 | "
        f"Best: RSI<{best['rsi_thresh']} vol×{best['vol_mult']} "
        f"sl_atr×{best['sl_atr_mult']} tp={best['tp_mode']}(×{best['tp_atr_mult']}) "
        f"lev={best['leverage']}x pos={best['pos_ratio']*100:.0f}% → "
        f"수익률={best['return_pct']:+.1f}% MDD={best['max_drawdown']:.1f}% "
        f"승률={best['win_rate']:.1f}% 거래={best['trades']}건 Calmar={best['calmar']:.2f}",
        flush=True
    )
    return out_path


def main():
    combos_total = len(list(itertools.product(
        RSI_THRESH, VOL_MULT, SL_ATR_MULT, TP_ATR_MULT, TP_MODES, LEVERAGES, POS_RATIOS
    )))
    print("RSI + 볼린저밴드 과매도 반등 전략 그리드 서치")
    print(f"타임프레임: 1h | 코인: {[c.upper() for c in COINS]}")
    print(f"조합 수: {combos_total}개 × {len(COINS)}코인 = {combos_total * len(COINS)}개 백테스트\n")

    with ProcessPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {executor.submit(worker, coin): coin for coin in COINS}
        for fut in as_completed(futures):
            coin = futures[fut]
            try:
                result = fut.result()
                print(f"  저장: {result}")
            except Exception as e:
                print(f"  [{coin}] 오류: {e}")


if __name__ == "__main__":
    main()
