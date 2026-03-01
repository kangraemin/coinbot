"""RSI + 볼린저밴드 과매도 반등 전략 — 1h 그리드 서치.

전략:
  진입 조건 (ALL 충족):
    1. close < BB_lower (20, 2σ)
    2. RSI(14) < rsi_thresh
    3. volume > vol_MA(20) × vol_mult
    4. [선택] close > EMA(200) — 불마켓 필터

  청산:
    - TP 모드:
        "bb_mid" : 현재 BB 중심선 도달
        "atr_2x" : 진입가 + ATR × 2
        "atr_3x" : 진입가 + ATR × 3
    - SL: 진입가 - ATR(14) × sl_atr_mult
    - 타임아웃: 48봉 (48시간)
    - 같은 봉 SL/TP 동시 도달 시 SL 우선

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
RSI_THRESHOLDS  = [25, 30, 35]
VOL_MULTIPLIERS = [1.2, 1.5, 2.0]
SL_ATR_MULTS    = [1.0, 1.5, 2.0]
TP_MODES        = ["bb_mid", "atr_2x", "atr_3x"]
LEVERAGES       = [3, 5, 7]
POS_RATIOS      = [0.1, 0.2, 0.3]
USE_EMA200      = [True, False]

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
    """BB(20,2σ), RSI(14), vol_MA(20), ATR(14), EMA(200) 계산."""
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # Bollinger Bands (20, 2σ)
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std(ddof=1)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # RSI(14) — Wilder EMA (ewm com = period-1)
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Volume MA(20)
    vol_ma = volume.rolling(20).mean()

    # ATR(14) — Wilder EMA
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean()

    # EMA(200)
    ema200 = close.ewm(span=200, adjust=False).mean()

    df = df.copy()
    df["bb_mid"]   = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["rsi"]      = rsi
    df["vol_ma"]   = vol_ma
    df["atr"]      = atr
    df["ema200"]   = ema200
    return df


def run_backtest(
    df: pd.DataFrame,
    rsi_thresh: float,
    vol_mult: float,
    sl_atr_mult: float,
    tp_mode: str,
    leverage: int,
    pos_ratio: float,
    use_ema200: bool,
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
    ema200   = df["ema200"].to_numpy(dtype=float)
    n = len(closes)

    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    trades = wins = 0
    total_pnl = 0.0

    in_position = False
    entry_price = sl_price = tp_price = 0.0
    position_amount = 0.0
    entry_bar = 0

    # EMA200 웜업 포함 — 220봉부터 신호 체크
    start = 220

    for i in range(start, n):
        if (np.isnan(bb_lower[i]) or np.isnan(rsi_arr[i]) or
                np.isnan(vol_ma[i]) or np.isnan(atr_arr[i]) or
                np.isnan(ema200[i])):
            continue

        if not in_position:
            if balance <= 0:
                break

            # EMA200 트렌드 필터 (선택)
            if use_ema200 and closes[i] <= ema200[i]:
                continue

            # 진입 조건
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
                if tp_mode == "atr_2x":
                    tp_price = entry_price + atr_val * 2.0
                elif tp_mode == "atr_3x":
                    tp_price = entry_price + atr_val * 3.0
                else:
                    tp_price = 0.0  # bb_mid: 매 봉 동적 체크
                entry_bar = i
                in_position = True
        else:
            exit_price = None
            won = False

            # bb_mid 모드: 현재 BB 중심선
            tp_check = bb_mid[i] if tp_mode == "bb_mid" else tp_price

            # SL 우선 (동시 도달 시 SL)
            if lows[i] <= sl_price:
                exit_price = sl_price
            elif tp_check > entry_price and highs[i] >= tp_check:
                exit_price = tp_check
                won = True
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
        "tp_mode":      tp_mode,
        "leverage":     leverage,
        "pos_ratio":    pos_ratio,
        "use_ema200":   use_ema200,
        "trades":       trades,
        "win_rate":     round(win_rate, 1),
        "return_pct":   round(ret, 2),
        "max_drawdown": round(max_drawdown, 2),
        "calmar":       round(calmar, 2),
        "timeframe":    "1h",
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
        RSI_THRESHOLDS, VOL_MULTIPLIERS, SL_ATR_MULTS,
        TP_MODES, LEVERAGES, POS_RATIOS, USE_EMA200,
    ))

    results = []
    for rsi_thresh, vol_mult, sl_atr, tp_mode, leverage, pos_ratio, ema200_flag in combos:
        r = run_backtest(
            df, rsi_thresh, vol_mult, sl_atr, tp_mode,
            leverage, pos_ratio, ema200_flag,
        )
        results.append(r)

    df_res = pd.DataFrame(results).sort_values("return_pct", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"bb_rsi_reversion_{coin}_1h.csv")
    df_res.to_csv(out_path, index=False)

    best = df_res.iloc[0]
    print(
        f"[{coin.upper()} 1h] 완료 — {len(results)}조합 | "
        f"Best: RSI<{best['rsi_thresh']} vol×{best['vol_mult']} "
        f"sl_atr×{best['sl_atr_mult']} tp={best['tp_mode']} "
        f"lev={best['leverage']}x pos={best['pos_ratio']*100:.0f}% "
        f"EMA200={'ON' if best['use_ema200'] else 'OFF'} → "
        f"수익률={best['return_pct']:+.1f}% MDD={best['max_drawdown']:.1f}% "
        f"승률={best['win_rate']:.1f}% 거래={best['trades']}건 Calmar={best['calmar']:.2f}",
        flush=True
    )
    return out_path


def main():
    combos_total = len(list(itertools.product(
        RSI_THRESHOLDS, VOL_MULTIPLIERS, SL_ATR_MULTS,
        TP_MODES, LEVERAGES, POS_RATIOS, USE_EMA200,
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
