"""backtest_pro.py — 프랍 스타일 Walk-Forward 백테스팅.

전략 철학:
  "왜 돈을 버는가" 가설: 크립토는 레짐 의존적이다.
  - 평균회귀 레짐(Hurst < 0.5)에서 과매도 딥은 반등 가능성이 통계적으로 높다.
  - 추세 레짐(Hurst > 0.5)에서는 딥바이가 패배 전략이다.
  - 베어마켓(EMA200 아래)에서는 딥바이가 패배 전략이다.
  → 레짐 필터 + 트렌드 필터로 진입 구간 선별.

진입 로직 (4가지 필터):
  1. [선택] close > EMA(200)        — 트렌드 필터 (불마켓만)
  2. Hurst Exponent < hurst_thresh  — 레짐 필터 (평균회귀 구간)
  3. RSI(14) < rsi_thresh           — 과매도 확인
  4. close < prev_close × (1 - entry_pct%) — 직전 봉 대비 X% 딥

청산 로직:
  - SL = entry - ATR(14) × sl_atr
  - TP = entry + (SL 거리) × rr_ratio
  - 타임아웃 = 48봉 (15분봉 기준 12시간)

비용 모델 (현실적):
  - 슬리피지: 진입 3bp 불리 (limit order partial fill 가정)
  - 수수료: Binance 테이커 0.05% × 2
  - 펀딩피: 8시간마다 0.01% (포지션 보유 시)

포지션 사이징:
  - 훈련 구간 거래 기반 1/4 Kelly 산출
  - 최대 30% 자본

Walk-Forward 검증:
  - 훈련: 18개월 → 그리드 서치 (Sharpe 최대화)
  - 검증: 6개월 OOS (훈련과 완전 분리)
  - 윈도우: 6개월 스텝 × 5개
  - 최종 판단: 전체 OOS 트레이드 통합 성과

그리드 (432개 조합):
  entry_pct   : [0.3, 0.5, 1.0, 1.5]   # 직전 종가 대비 딥 %
  rsi_thresh  : [35, 40, 45]            # RSI 상한
  sl_atr      : [1.0, 1.5, 2.0]         # SL 거리 (ATR 배수)
  rr_ratio    : [1.5, 2.0, 2.5]         # TP/SL 손익비
  hurst_thresh: [0.62, 0.65, 0.68]      # 레짐 필터 (15분봉 BTC Hurst 실측: 0.54~0.75)
  use_ema200  : [True, False]           # EMA200 트렌드 필터

중요 발견:
  15분봉 BTC의 Hurst는 항상 0.54 이상 (평균 0.64, max 0.75).
  즉, 15분봉 레벨에서는 BTC가 "항상 추세장"이다.
  → H < 0.50 같은 교과서 기준은 15분봉에서 신호를 전부 차단한다.
  → 대신 상대적 Hurst (해당 구간 내 하위 분위) 기준을 사용해 "덜 추세적인" 구간 선별.

사용법:
  python backtest_pro.py               # BTC 기본 (Hurst 활성화)
  python backtest_pro.py --coin eth
  python backtest_pro.py --no-hurst    # Hurst 비활성화 (빠른 실행, 비교용)
"""

import argparse
import os
import time
from itertools import product
from scipy import stats as scipy_stats

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ─────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market")

# 비용 모델
SLIPPAGE = 0.0003        # 진입 시 3bp 불리
TAKER_FEE = 0.0005       # 바이낸스 테이커 0.05%
FUNDING_PER_8H = 0.0001  # 8시간당 0.01%
FUNDING_CANDLES = 32     # 15분봉 × 32 = 8시간

# 전략 고정값
HURST_WINDOW = 100       # Hurst 윈도우 (~25시간)
EMA200_PERIOD = 200
ATR_PERIOD = 14
RSI_PERIOD = 14
MAX_HOLD = 48            # 최대 보유봉 (12시간)
WARMUP = 220             # 지표 웜업 (EMA200 포함)

# Walk-Forward
TRAIN_MONTHS = 18
TEST_MONTHS = 6
STEP_MONTHS = 6

# Kelly
KELLY_FRACTION = 0.25
MAX_POS = 0.30
MIN_TRADES_FOR_KELLY = 20

# 그리드
GRID = {
    "entry_pct":    [0.3, 0.5, 1.0, 1.5],
    "rsi_thresh":   [35, 40, 45],
    "sl_atr":       [1.0, 1.5, 2.0],
    "rr_ratio":     [1.5, 2.0, 2.5],
    "hurst_thresh": [0.62, 0.65, 0.68],
    "use_ema200":   [True, False],
}


# ─────────────────────────────────────────────────────────────
# 데이터 로드 및 전처리
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
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")
    df_15m = df.resample("15min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return df_15m.reset_index()


# ─────────────────────────────────────────────────────────────
# 지표 계산
# ─────────────────────────────────────────────────────────────

def _hurst_rs(prices: np.ndarray) -> float:
    """R/S 방법으로 Hurst 지수 추정."""
    log_ret = np.diff(np.log(np.maximum(prices, 1e-10)))
    n = len(log_ret)
    lags = [4, 8, 16, 32]
    rs_vals, lag_used = [], []
    for lag in lags:
        if lag >= n:
            continue
        pieces = n // lag
        rs_list = []
        for j in range(pieces):
            seg = log_ret[j * lag:(j + 1) * lag]
            dev = np.cumsum(seg - seg.mean())
            R = dev.max() - dev.min()
            S = seg.std(ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)
        if rs_list:
            rs_vals.append(np.mean(rs_list))
            lag_used.append(lag)
    if len(rs_vals) < 2:
        return 0.5
    try:
        return float(np.polyfit(np.log(lag_used), np.log(rs_vals), 1)[0])
    except Exception:
        return 0.5


def compute_hurst_column(closes: np.ndarray, window: int = HURST_WINDOW) -> np.ndarray:
    n = len(closes)
    hurst = np.full(n, np.nan)
    wins = sliding_window_view(closes, window)
    total = len(wins)
    report_step = max(1, total // 10)
    print(f"  Hurst 계산: {total:,}봉 ", end="", flush=True)
    t0 = time.time()
    for i, w in enumerate(wins):
        hurst[i + window - 1] = _hurst_rs(w)
        if (i + 1) % report_step == 0:
            print(".", end="", flush=True)
    print(f" {time.time()-t0:.0f}초")
    return hurst


def add_indicators(df: pd.DataFrame, use_hurst: bool = True) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI(14) — Wilder EMA
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ATR(14)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()

    # EMA200
    df["ema200"] = close.ewm(span=EMA200_PERIOD, adjust=False).mean()

    # 직전 종가 (진입 기준)
    df["prev_close"] = close.shift(1)

    # Hurst
    if use_hurst:
        df["hurst"] = compute_hurst_column(close.values, HURST_WINDOW)
    else:
        df["hurst"] = 0.49  # 항상 통과

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 백테스팅 엔진
# ─────────────────────────────────────────────────────────────

def run_backtest(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    rsi_arr: np.ndarray,
    atr_arr: np.ndarray,
    hurst_arr: np.ndarray,
    ema200_arr: np.ndarray,
    prev_close_arr: np.ndarray,
    params: dict,
) -> list:
    entry_pct = params["entry_pct"]
    rsi_thresh = params["rsi_thresh"]
    sl_atr = params["sl_atr"]
    rr_ratio = params["rr_ratio"]
    hurst_thresh = params["hurst_thresh"]
    use_ema200 = params["use_ema200"]

    n = len(closes)
    trades = []
    in_pos = False
    entry_price = sl = tp = 0.0
    entry_idx = 0
    accrued_funding = 0.0

    for i in range(WARMUP, n):
        if (np.isnan(rsi_arr[i]) or np.isnan(atr_arr[i])
                or np.isnan(hurst_arr[i]) or np.isnan(prev_close_arr[i])
                or np.isnan(ema200_arr[i])):
            continue

        close = closes[i]
        high = highs[i]
        low = lows[i]

        # ── 보유 중: 청산 체크 ─────────────────────────
        if in_pos:
            hold = i - entry_idx
            if hold > 0 and hold % FUNDING_CANDLES == 0:
                accrued_funding += FUNDING_PER_8H

            hit_sl = low <= sl
            hit_tp = high >= tp

            if hit_sl:
                exit_price, result = sl, "sl"
            elif hit_tp:
                exit_price, result = tp, "tp"
            elif hold >= MAX_HOLD:
                exit_price, result = close, "timeout"
            else:
                continue

            gross_ret = (exit_price - entry_price) / entry_price
            cost = TAKER_FEE * 2 + SLIPPAGE + accrued_funding
            net_ret = gross_ret - cost

            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": i,
                "entry": entry_price,
                "exit": exit_price,
                "hold": hold,
                "result": result,
                "net_ret": net_ret,
                "hurst": float(hurst_arr[entry_idx]),
                "rsi_entry": float(rsi_arr[entry_idx]),
            })
            in_pos = False
            accrued_funding = 0.0
            continue

        # ── 미보유: 진입 신호 ──────────────────────────
        # 1. EMA200 트렌드 필터 (선택)
        if use_ema200 and close <= ema200_arr[i]:
            continue

        # 2. Hurst 레짐 필터
        if hurst_arr[i] >= hurst_thresh:
            continue

        # 3. RSI 과매도
        if rsi_arr[i] >= rsi_thresh:
            continue

        # 4. 직전 종가 대비 X% 딥
        ref = prev_close_arr[i]
        if ref <= 0 or close > ref * (1 - entry_pct / 100):
            continue

        # 5. ATR 유효성
        atr = atr_arr[i]
        if atr <= 0:
            continue

        # 진입 (슬리피지)
        entry_price = close * (1 + SLIPPAGE)
        sl_dist = atr * sl_atr
        if sl_dist <= 0:
            continue
        sl = entry_price - sl_dist
        tp = entry_price + sl_dist * rr_ratio

        in_pos = True
        entry_idx = i
        accrued_funding = 0.0

    return trades


# ─────────────────────────────────────────────────────────────
# 포지션 사이징 (1/4 Kelly)
# ─────────────────────────────────────────────────────────────

def kelly_pos_size(trades: list) -> float:
    if len(trades) < MIN_TRADES_FOR_KELLY:
        return 0.05
    wins = [t for t in trades if t["result"] == "tp"]
    losses = [t for t in trades if t["result"] != "tp"]
    if not wins or not losses:
        return 0.05
    win_rate = len(wins) / len(trades)
    avg_win = np.mean([t["net_ret"] for t in wins])
    avg_loss = abs(np.mean([t["net_ret"] for t in losses]))
    if avg_loss < 1e-10:
        return MAX_POS
    rr = avg_win / avg_loss
    full_kelly = (win_rate * rr - (1 - win_rate)) / rr
    return float(np.clip(full_kelly * KELLY_FRACTION, 0.01, MAX_POS))


# ─────────────────────────────────────────────────────────────
# 통계 계산
# ─────────────────────────────────────────────────────────────

def calc_stats(trades: list, pos_size: float) -> dict | None:
    if len(trades) < 5:
        return None

    wins = [t for t in trades if t["result"] == "tp"]
    losses = [t for t in trades if t["result"] != "tp"]
    sl_only = [t for t in trades if t["result"] == "sl"]
    timeout = [t for t in trades if t["result"] == "timeout"]

    win_rate = len(wins) / len(trades) * 100
    avg_win = np.mean([t["net_ret"] for t in wins]) * 100 if wins else 0.0
    avg_loss = np.mean([t["net_ret"] for t in losses]) * 100 if losses else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for t in trades:
        equity *= 1 + t["net_ret"] * pos_size
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
    total_ret = (equity - 1.0) * 100

    rets = np.array([t["net_ret"] * pos_size for t in trades])
    sharpe = float(rets.mean() / rets.std() * np.sqrt(len(rets))) if rets.std() > 1e-10 else 0.0

    gross_profit = sum(t["net_ret"] for t in wins) if wins else 0.0
    gross_loss = abs(sum(t["net_ret"] for t in losses)) if losses else 1e-10
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    max_consec = cur = 0
    for t in trades:
        if t["result"] != "tp":
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    return {
        "n": len(trades),
        "n_tp": len(wins),
        "n_sl": len(sl_only),
        "n_timeout": len(timeout),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "expectancy": round(expectancy, 4),
        "total_ret": round(total_ret, 2),
        "mdd": round(mdd, 2),
        "sharpe": round(sharpe, 3),
        "profit_factor": round(pf, 3),
        "max_consec_loss": max_consec,
        "pos_size": round(pos_size, 4),
    }


# ─────────────────────────────────────────────────────────────
# Walk-Forward 검증
# ─────────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, coin: str, use_hurst: bool) -> dict:
    ts_arr = df["timestamp"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    rsi_arr = df["rsi"].values
    atr_arr = df["atr"].values
    hurst_arr = df["hurst"].values
    ema200_arr = df["ema200"].values
    prev_close_arr = df["prev_close"].values

    # 파라미터 조합
    keys = list(GRID.keys())
    combos = [dict(zip(keys, v)) for v in product(*GRID.values())]

    # Hurst 비활성화 시 hurst_thresh 고정 (중복 제거)
    if not use_hurst:
        seen = set()
        unique_combos = []
        for c in combos:
            key = tuple(c[k] for k in keys if k != "hurst_thresh")
            if key not in seen:
                seen.add(key)
                c["hurst_thresh"] = 0.50
                unique_combos.append(c)
        combos = unique_combos

    print(f"\n[{coin.upper()}] {pd.Timestamp(ts_arr[0]).date()} ~ {pd.Timestamp(ts_arr[-1]).date()}")
    print(f"  15분봉 총 {len(df):,}봉 | 그리드 조합: {len(combos)}개")

    # Walk-Forward 윈도우
    start_ts = pd.Timestamp(ts_arr[0]).tz_localize(None)
    end_ts = pd.Timestamp(ts_arr[-1]).tz_localize(None)
    windows = []
    win_start = start_ts
    while True:
        train_end = win_start + pd.DateOffset(months=TRAIN_MONTHS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)
        if test_end > end_ts:
            break
        windows.append((win_start, train_end, test_end))
        win_start += pd.DateOffset(months=STEP_MONTHS)

    print(f"  Walk-Forward 윈도우: {len(windows)}개 ({TRAIN_MONTHS}개월 훈련 / {TEST_MONTHS}개월 검증)")

    if not windows:
        print("  [경고] 데이터 부족")
        return {}

    ts_naive = pd.DatetimeIndex(ts_arr).tz_localize(None)
    all_oos_trades = []
    wf_results = []

    def _sl(arr, idx0, idx1):
        return arr[idx0:idx1 + 1]

    for w_idx, (train_s, train_e, test_e) in enumerate(windows):
        mask_train = (ts_naive >= train_s) & (ts_naive < train_e)
        mask_test = (ts_naive >= train_e) & (ts_naive < test_e)
        idx_train = np.where(mask_train)[0]
        idx_test = np.where(mask_test)[0]

        if len(idx_train) < WARMUP * 2 or len(idx_test) < WARMUP:
            continue

        i0, i1 = idx_train[0], idx_train[-1]
        j0, j1 = idx_test[0], idx_test[-1]

        tr_args = (
            _sl(closes, i0, i1), _sl(highs, i0, i1), _sl(lows, i0, i1),
            _sl(rsi_arr, i0, i1), _sl(atr_arr, i0, i1), _sl(hurst_arr, i0, i1),
            _sl(ema200_arr, i0, i1), _sl(prev_close_arr, i0, i1),
        )
        te_args = (
            _sl(closes, j0, j1), _sl(highs, j0, j1), _sl(lows, j0, j1),
            _sl(rsi_arr, j0, j1), _sl(atr_arr, j0, j1), _sl(hurst_arr, j0, j1),
            _sl(ema200_arr, j0, j1), _sl(prev_close_arr, j0, j1),
        )
        ts_te = _sl(ts_arr, j0, j1)

        print(f"\n  ── WF {w_idx+1}/{len(windows)} ─────────────────────────────────────")
        print(f"  훈련: {train_s.date()}~{train_e.date()} ({len(idx_train):,}봉)  "
              f"검증: {train_e.date()}~{test_e.date()} ({len(idx_test):,}봉)")
        print(f"  그리드 서치 중...", end="", flush=True)
        t0 = time.time()

        best_sharpe = -np.inf
        best_params = None

        for params in combos:
            trades = run_backtest(*tr_args, params)
            if len(trades) < MIN_TRADES_FOR_KELLY:
                continue
            pos = kelly_pos_size(trades)
            stats = calc_stats(trades, pos)
            if stats and stats["sharpe"] > best_sharpe and stats["sharpe"] > 0:
                best_sharpe = stats["sharpe"]
                best_params = params

        print(f" {time.time()-t0:.1f}초")

        if best_params is None:
            print("  → Sharpe > 0인 파라미터 없음, 스킵")
            continue

        p = best_params
        print(f"  → 최적: entry={p['entry_pct']}%  RSI<{p['rsi_thresh']}  "
              f"SL×{p['sl_atr']}  RR={p['rr_ratio']}  H<{p['hurst_thresh']}  "
              f"EMA200={'ON' if p['use_ema200'] else 'OFF'}")
        print(f"  → 훈련 Sharpe: {best_sharpe:.3f}")

        # OOS 검증
        oos_raw = run_backtest(*te_args, best_params)

        # Kelly는 훈련 기반
        train_trades = run_backtest(*tr_args, best_params)
        pos_size = kelly_pos_size(train_trades)
        oos_stats = calc_stats(oos_raw, pos_size)

        if oos_stats is None:
            print("  → OOS 거래 5건 미만, 스킵")
            continue

        print(f"  → OOS: {oos_stats['n']}건 | 승률 {oos_stats['win_rate']}% | "
              f"수익 {oos_stats['total_ret']:+.1f}% | MDD {oos_stats['mdd']:.1f}% | "
              f"Sharpe {oos_stats['sharpe']:.3f} | PF {oos_stats['profit_factor']:.2f}")

        for t in oos_raw:
            t["window"] = w_idx + 1
            t["ts_entry"] = ts_te[t["entry_idx"]] if t["entry_idx"] < len(ts_te) else None
            t["ts_exit"] = ts_te[t["exit_idx"]] if t["exit_idx"] < len(ts_te) else None

        all_oos_trades.extend(oos_raw)
        wf_results.append({
            "window": w_idx + 1,
            "train_start": train_s,
            "train_end": train_e,
            "test_end": test_e,
            "best_params": best_params,
            "train_sharpe": best_sharpe,
            "oos_stats": oos_stats,
            "pos_size": pos_size,
        })

    return {"coin": coin, "windows": wf_results, "all_oos_trades": all_oos_trades}


# ─────────────────────────────────────────────────────────────
# 최종 리포트
# ─────────────────────────────────────────────────────────────

def print_report(result: dict, use_hurst: bool) -> None:
    coin = result["coin"]
    windows = result["windows"]
    all_oos = result["all_oos_trades"]
    mode = "Hurst ON" if use_hurst else "Hurst OFF"

    print("\n" + "=" * 72)
    print(f"  {coin.upper()} Walk-Forward OOS 성과 [{mode}]")
    print("=" * 72)

    if not windows:
        print("  유효 윈도우 없음 — 모든 구간에서 양의 Sharpe 파라미터 없음")
        print("  (→ 이 전략은 이 코인/기간에서 통계적 엣지 없음)")
        return

    print(f"\n  {'WF':>3}  {'OOS 기간':>17}  {'거래':>5}  {'승률':>6}  "
          f"{'수익%':>8}  {'MDD%':>6}  {'Sharpe':>7}  {'PF':>5}  {'pos%':>5}  {'결과':>4}")
    print("  " + "-" * 70)
    oos_pos = 0
    for w in windows:
        s = w["oos_stats"]
        period = f"{w['train_end'].strftime('%y/%m')}~{w['test_end'].strftime('%y/%m')}"
        marker = "✓" if s["total_ret"] > 0 else "✗"
        if s["total_ret"] > 0:
            oos_pos += 1
        print(f"  {w['window']:>3}  {period:>17}  {s['n']:>5}  {s['win_rate']:>5.1f}%  "
              f"{s['total_ret']:>+7.1f}%  {s['mdd']:>5.1f}%  {s['sharpe']:>7.3f}  "
              f"{s['profit_factor']:>4.2f}  {w['pos_size']*100:>4.1f}%  {marker}")

    print(f"\n  수익 OOS 구간: {oos_pos}/{len(windows)}")

    # 전체 OOS 통합
    if all_oos:
        avg_pos = np.mean([w["pos_size"] for w in windows])
        total = calc_stats(all_oos, avg_pos)
        if total:
            print("\n  ── 전체 OOS 통합 ──")
            print(f"  총 거래수     : {total['n']}건  "
                  f"(TP {total['n_tp']} / SL {total['n_sl']} / 타임아웃 {total['n_timeout']})")
            print(f"  승률          : {total['win_rate']}%")
            print(f"  평균 수익     : {total['avg_win']:+.3f}% / 트레이드")
            print(f"  평균 손실     : {total['avg_loss']:+.3f}% / 트레이드")
            print(f"  기대값        : {total['expectancy']:+.4f}% / 트레이드")
            print(f"  복리 총수익   : {total['total_ret']:+.2f}%")
            print(f"  최대 낙폭     : -{total['mdd']:.2f}%")
            print(f"  Sharpe        : {total['sharpe']:.3f}")
            print(f"  Profit Factor : {total['profit_factor']:.3f}")
            print(f"  연속 최대 손실: {total['max_consec_loss']}연패")
            print(f"  포지션 비율   : {avg_pos*100:.1f}% (1/4 Kelly 평균)")

    # 파라미터 안정성
    print("\n  ── 파라미터 선택 (수렴도가 과적합 판단 기준) ──")
    for w in windows:
        p = w["best_params"]
        print(f"  WF{w['window']}: entry={p['entry_pct']}%  RSI<{p['rsi_thresh']}  "
              f"SL×{p['sl_atr']}  RR={p['rr_ratio']}  H<{p['hurst_thresh']}  "
              f"EMA200={'ON' if p['use_ema200'] else 'OFF'}")

    if len(windows) >= 3:
        entry_v = [w["best_params"]["entry_pct"] for w in windows]
        rsi_v = [w["best_params"]["rsi_thresh"] for w in windows]
        ema_v = [w["best_params"]["use_ema200"] for w in windows]
        print(f"\n  entry_pct σ={np.std(entry_v):.2f}  rsi_thresh σ={np.std(rsi_v):.2f}  "
              f"EMA200 ON={sum(ema_v)}/{len(ema_v)}  "
              f"(σ 작고 EMA200 수렴 → 과적합↓)")

    # ── 통계적 유의성 검정 ──
    if all_oos and len(all_oos) >= 20:
        rets = np.array([t["net_ret"] for t in all_oos])
        t_stat, p_val = scipy_stats.ttest_1samp(rets, 0)
        n = len(rets)
        se = rets.std() / np.sqrt(n)
        ci_lo = rets.mean() - 1.96 * se
        ci_hi = rets.mean() + 1.96 * se
        significant = p_val < 0.05 and rets.mean() > 0
        print(f"\n  ── 통계적 유의성 (중요!) ──")
        print(f"  표본 수       : {n}건")
        print(f"  기대값 평균   : {rets.mean()*100:+.4f}%")
        print(f"  95% 신뢰구간  : [{ci_lo*100:+.4f}%, {ci_hi*100:+.4f}%]")
        print(f"  t-stat / p-val: {t_stat:.3f} / {p_val:.4f}")
        if significant:
            print(f"  → ✓ 통계적으로 유의한 양의 기대값 (p < 0.05)")
        elif p_val < 0.05:
            print(f"  → ✗ 통계적으로 유의한 음의 기대값")
        else:
            print(f"  → △ 통계적으로 유의하지 않음 (p = {p_val:.3f})")
            print(f"     (엣지가 있더라도 거래 수 {n}건으로는 증명 불가)")

    print("=" * 72)

    # CSV 저장
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if use_hurst else "_no_hurst"

    if all_oos:
        trades_path = os.path.join(out_dir, f"pro_oos_trades_{coin}{suffix}.csv")
        pd.DataFrame([
            {k: v for k, v in t.items() if k not in ("entry_idx", "exit_idx")}
            for t in all_oos
        ]).to_csv(trades_path, index=False)
        print(f"\n  OOS 거래 내역 → {trades_path}")

    if windows:
        rows = []
        for w in windows:
            s = w["oos_stats"]
            row = {
                "window": w["window"],
                "train_start": w["train_start"].date(),
                "train_end": w["train_end"].date(),
                "test_end": w["test_end"].date(),
                "train_sharpe": round(w["train_sharpe"], 3),
                **{f"oos_{k}": v for k, v in s.items() if k not in ("equity_curve",)},
                **{f"param_{k}": v for k, v in w["best_params"].items()},
            }
            rows.append(row)
        summary_path = os.path.join(out_dir, f"pro_wf_summary_{coin}{suffix}.csv")
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"  WF 요약      → {summary_path}")


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="프랍 스타일 Walk-Forward 백테스팅")
    parser.add_argument("--coin", default="btc", help="코인 심볼 소문자 (기본: btc)")
    parser.add_argument("--no-hurst", action="store_true",
                        help="Hurst 비활성화 (빠른 실행, RSI+EMA200만 사용)")
    args = parser.parse_args()
    use_hurst = not args.no_hurst

    t_start = time.time()
    print(f"\n{'='*52}")
    print(f"  프랍 Walk-Forward 백테스팅 — {args.coin.upper()}")
    print(f"  모드: {'Hurst 레짐 필터 ON' if use_hurst else 'Hurst OFF (RSI+EMA200 only)'}")
    print(f"  비용: 슬리피지 {SLIPPAGE*1e4:.0f}bp + 테이커 {TAKER_FEE*100:.2f}% × 2 + 펀딩 {FUNDING_PER_8H*100:.2f}%/8h")
    print(f"  포지션: 1/4 Kelly (최대 {MAX_POS*100:.0f}%)")
    print(f"{'='*52}")

    print(f"\n[1/3] 데이터 로드 및 15분봉 리샘플 중...")
    df_15m = load_and_resample(args.coin)
    print(f"  15분봉 {len(df_15m):,}행  "
          f"({df_15m['timestamp'].iloc[0].date()} ~ {df_15m['timestamp'].iloc[-1].date()})")

    print(f"\n[2/3] 지표 계산 중...")
    df = add_indicators(df_15m, use_hurst=use_hurst)

    print(f"\n[3/3] Walk-Forward 실행 중...")
    result = walk_forward(df, args.coin, use_hurst)

    print_report(result, use_hurst)
    print(f"\n  총 소요시간: {time.time()-t_start:.1f}초")


if __name__ == "__main__":
    main()
