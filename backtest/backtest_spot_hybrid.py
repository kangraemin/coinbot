"""현물 보유 + Tier 1 선물 조합 — 시나리오 비교.

시나리오:
  BuyHold          : 현물 100% 단순 보유
  FuturesOnly      : 선물 전용 (Tier1 4h 파라미터)
  B (Partial Hybrid): Tier1 신호 시 현물 30% → 선물 전환, 청산 후 현물 복귀

코인: XRP, SOL | 기간: 2022~2025 (4년, 2022 하락장 포함)
"""

import os

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0
START       = "2022-01-01"
END         = "2025-12-31"

# Tier1 확정 파라미터 (4h)
TIER1 = {
    "xrp": dict(rsi_thresh=30, sl_atr_mult=1.5, tp_mode="bb_mid", leverage=7,
                pos_ratio=0.30, use_ema200=False, timeout_1h=192),
    "sol": dict(rsi_thresh=45, sl_atr_mult=1.0, tp_mode="atr_2x", leverage=7,
                pos_ratio=0.30, use_ema200=True, timeout_1h=192),
}

COINS = ["xrp", "sol"]


# ── 데이터 준비 ──────────────────────────────────────

def load_data(coin: str) -> pd.DataFrame:
    frames = []
    for y in range(2021, 2026):  # 2021 포함: EMA200 웜업용
        p = os.path.join(DATA_DIR, f"{coin}_1m_{y}.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df.columns = [c.lower() for c in df.columns]
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").set_index("timestamp")


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last"),
        volume=("volume","sum"),
    ).dropna().reset_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    lo = df["low"].astype(float)
    bb_mid   = c.rolling(20).mean()
    bb_lower = bb_mid - 2 * c.rolling(20).std(ddof=1)
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    tr    = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr    = tr.ewm(com=13, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    df = df.copy()
    df["bb_mid"]   = bb_mid
    df["bb_lower"] = bb_lower
    df["rsi"]      = rsi
    df["atr"]      = atr
    df["ema200"]   = ema200
    return df


def build_arrays(coin: str):
    df_1m = load_data(coin)
    df_1h = resample(df_1m, "1h")
    df_4h = resample(df_1m, "4h")
    df_1h = add_indicators(df_1h)
    df_4h = add_indicators(df_4h)

    df_4h_idx = df_4h.set_index("timestamp")[
        ["close", "bb_mid", "bb_lower", "rsi", "atr", "ema200"]
    ].rename(columns=lambda x: f"t1_{x}")
    df = df_1h.join(df_4h_idx, on="timestamp", how="left")
    df["t1_bb_mid"] = df["t1_bb_mid"].ffill()  # TP 체크용 ffill

    s = pd.Timestamp(START, tz="UTC")
    e = pd.Timestamp(END + " 23:59:59", tz="UTC")
    df = df[(df["timestamp"] >= s) & (df["timestamp"] <= e)].reset_index(drop=True)

    return df


# ── 공통 Tier1 신호/포지션 관리 헬퍼 ─────────────────

def tier1_signal(i, t1_close, t1_bb_low, t1_rsi, t1_atr, t1_ema, p: dict) -> bool:
    if np.isnan(t1_close[i]):
        return False
    if any(np.isnan(x[i]) for x in [t1_bb_low, t1_rsi, t1_atr, t1_ema]):
        return False
    ema_ok = (not p["use_ema200"]) or (t1_close[i] > t1_ema[i])
    return ema_ok and t1_close[i] < t1_bb_low[i] and t1_rsi[i] < p["rsi_thresh"]


def calc_tp(entry: float, atr_val: float, tp_mode: str, bb_mid_val: float) -> float:
    if tp_mode == "atr_2x":
        return entry + atr_val * 2.0
    elif tp_mode == "atr_3x":
        return entry + atr_val * 3.0
    return 0.0  # bb_mid: 동적


def calc_sl(entry: float, atr_val: float, sl_atr_mult: float) -> float:
    return entry - atr_val * sl_atr_mult


# ── 시나리오별 포트폴리오 추적 ────────────────────────

def portfolio_stats(values: np.ndarray, monthly_rets: list) -> dict:
    if len(values) < 2:
        return dict(return_pct=0, max_drawdown=0, sharpe=0, calmar=0)
    ret = (values[-1] - INITIAL_BAL) / INITIAL_BAL * 100
    peak = np.maximum.accumulate(values)
    mdd  = ((peak - values) / peak * 100).max()
    calmar = ret / mdd if mdd > 0 else 0.0
    sharpe = 0.0
    if len(monthly_rets) > 1:
        mr = np.array(monthly_rets)
        sharpe = round(mr.mean() / (mr.std(ddof=1) + 1e-10) * (12**0.5), 2)
    return dict(
        return_pct=round(ret, 2),
        max_drawdown=round(mdd, 2),
        sharpe=sharpe,
        calmar=round(calmar, 2),
    )


def monthly_series(ts_arr, values: np.ndarray) -> list:
    df = pd.DataFrame({"ts": ts_arr, "v": values})
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    m = df["v"].resample("ME").last().ffill()
    rets = m.pct_change().dropna() * 100
    return list(rets.values)


def run_scenario_a(df: pd.DataFrame, p: dict) -> dict:
    """Overlay: 현물 100% 유지 + 선물 추가."""
    closes   = df["close"].to_numpy(float)
    highs    = df["high"].to_numpy(float)
    lows     = df["low"].to_numpy(float)
    ts       = df["timestamp"].to_numpy()
    t1_close = df["t1_close"].to_numpy(float)
    t1_bb_low= df["t1_bb_lower"].to_numpy(float)
    t1_bb_mid= df["t1_bb_mid"].to_numpy(float)
    t1_rsi   = df["t1_rsi"].to_numpy(float)
    t1_atr   = df["t1_atr"].to_numpy(float)
    t1_ema   = df["t1_ema200"].to_numpy(float)
    n = len(closes)

    price_0 = closes[0]
    fut_balance = 0.0   # 선물 누적 PnL (수수료 포함)

    in_pos   = False
    ep = sl = tp = pos_amt = entry_bar = 0.0
    tp_mode  = p["tp_mode"]
    trades   = wins = 0

    port_vals = np.empty(n)

    for i in range(n):
        spot_val = INITIAL_BAL * closes[i] / price_0
        unreal   = (closes[i] - ep) * pos_amt if in_pos else 0.0
        port_vals[i] = spot_val + fut_balance + unreal

        if in_pos:
            tp_check = t1_bb_mid[i] if tp_mode == "bb_mid" else tp
            exit_p = None
            if lows[i] <= sl:
                exit_p = sl
            elif tp_check > ep and highs[i] >= tp_check:
                exit_p = tp_check; wins += 1
            elif i - entry_bar >= p["timeout_1h"]:
                exit_p = closes[i]
                if exit_p > ep: wins += 1
            if exit_p is not None:
                pnl = (exit_p - ep) * pos_amt
                fee = exit_p * pos_amt * TAKER_FEE
                fut_balance += pnl - fee
                trades += 1
                in_pos = False

        if not in_pos and tier1_signal(i, t1_close, t1_bb_low, t1_rsi, t1_atr, t1_ema, p):
            ep  = t1_close[i]
            notional = INITIAL_BAL * p["pos_ratio"] * p["leverage"]
            pos_amt  = notional / ep
            fut_balance -= notional * TAKER_FEE
            sl  = calc_sl(ep, t1_atr[i], p["sl_atr_mult"])
            tp  = calc_tp(ep, t1_atr[i], tp_mode, t1_bb_mid[i])
            entry_bar = i
            in_pos = True

    mr = monthly_series(ts, port_vals)
    st = portfolio_stats(port_vals, mr)
    st["trades"] = trades
    st["win_rate"] = round(wins/trades*100, 1) if trades else 0.0
    st["scenario"] = "A_overlay"
    return st, port_vals, ts


def run_scenario_b(df: pd.DataFrame, p: dict) -> dict:
    """Partial Hybrid: Tier1 신호 시 현물 30% → 선물."""
    closes   = df["close"].to_numpy(float)
    highs    = df["high"].to_numpy(float)
    lows     = df["low"].to_numpy(float)
    ts       = df["timestamp"].to_numpy()
    t1_close = df["t1_close"].to_numpy(float)
    t1_bb_low= df["t1_bb_lower"].to_numpy(float)
    t1_bb_mid= df["t1_bb_mid"].to_numpy(float)
    t1_rsi   = df["t1_rsi"].to_numpy(float)
    t1_atr   = df["t1_atr"].to_numpy(float)
    t1_ema   = df["t1_ema200"].to_numpy(float)
    n = len(closes)

    spot_units = INITIAL_BAL / closes[0]
    cash = 0.0

    in_pos   = False
    ep = sl = tp = pos_amt = entry_bar = 0.0
    tp_mode  = p["tp_mode"]
    trades   = wins = 0

    port_vals = np.empty(n)

    for i in range(n):
        spot_val = spot_units * closes[i]
        unreal   = (closes[i] - ep) * pos_amt if in_pos else 0.0
        port_vals[i] = spot_val + cash + unreal

        if in_pos:
            tp_check = t1_bb_mid[i] if tp_mode == "bb_mid" else tp
            exit_p = None
            if lows[i] <= sl:
                exit_p = sl
            elif tp_check > ep and highs[i] >= tp_check:
                exit_p = tp_check; wins += 1
            elif i - entry_bar >= p["timeout_1h"]:
                exit_p = closes[i]
                if exit_p > ep: wins += 1
            if exit_p is not None:
                pnl = (exit_p - ep) * pos_amt
                fee = exit_p * pos_amt * TAKER_FEE
                cash += pnl - fee
                trades += 1
                in_pos = False
                # 현금 전량을 현물로 재매수 (선물 청산 후 완전히 현물로 복귀)
                spot_units += cash / closes[i] * (1 - TAKER_FEE)
                cash = 0.0

        if not in_pos and tier1_signal(i, t1_close, t1_bb_low, t1_rsi, t1_atr, t1_ema, p):
            ep = t1_close[i]
            total_now = spot_units * ep + cash
            sell_val  = total_now * 0.30
            # 현물 30% 매도
            sell_units = sell_val / ep
            spot_units = max(0.0, spot_units - sell_units)
            cash += sell_val * (1 - TAKER_FEE)
            # 선물 진입
            notional = sell_val * p["leverage"]
            pos_amt  = notional / ep
            cash -= sell_val * TAKER_FEE  # 선물 진입 수수료
            sl  = calc_sl(ep, t1_atr[i], p["sl_atr_mult"])
            tp  = calc_tp(ep, t1_atr[i], tp_mode, t1_bb_mid[i])
            entry_bar = i
            in_pos = True

    mr = monthly_series(ts, port_vals)
    st = portfolio_stats(port_vals, mr)
    st["trades"] = trades
    st["win_rate"] = round(wins/trades*100, 1) if trades else 0.0
    st["scenario"] = "B_partial"
    return st, port_vals, ts


def run_scenario_c(df: pd.DataFrame, p: dict) -> dict:
    """Full Switch: Tier1 신호 시 현물 전량 → 선물 (30% 비중)."""
    closes   = df["close"].to_numpy(float)
    highs    = df["high"].to_numpy(float)
    lows     = df["low"].to_numpy(float)
    ts       = df["timestamp"].to_numpy()
    t1_close = df["t1_close"].to_numpy(float)
    t1_bb_low= df["t1_bb_lower"].to_numpy(float)
    t1_bb_mid= df["t1_bb_mid"].to_numpy(float)
    t1_rsi   = df["t1_rsi"].to_numpy(float)
    t1_atr   = df["t1_atr"].to_numpy(float)
    t1_ema   = df["t1_ema200"].to_numpy(float)
    n = len(closes)

    spot_units = INITIAL_BAL / closes[0]
    cash = 0.0  # stablecoin + realized futures pnl

    in_pos   = False
    ep = sl = tp = pos_amt = entry_bar = 0.0
    tp_mode  = p["tp_mode"]
    trades   = wins = 0

    port_vals = np.empty(n)

    for i in range(n):
        spot_val = spot_units * closes[i]
        unreal   = (closes[i] - ep) * pos_amt if in_pos else 0.0
        port_vals[i] = spot_val + cash + unreal

        if in_pos:
            tp_check = t1_bb_mid[i] if tp_mode == "bb_mid" else tp
            exit_p = None
            if lows[i] <= sl:
                exit_p = sl
            elif tp_check > ep and highs[i] >= tp_check:
                exit_p = tp_check; wins += 1
            elif i - entry_bar >= p["timeout_1h"]:
                exit_p = closes[i]
                if exit_p > ep: wins += 1
            if exit_p is not None:
                pnl = (exit_p - ep) * pos_amt
                fee = exit_p * pos_amt * TAKER_FEE
                cash += pnl - fee
                trades += 1
                in_pos = False
                # 현물 100% 재매수
                total_cash = cash
                spot_units = total_cash / closes[i] * (1 - TAKER_FEE)
                cash = 0.0

        if not in_pos and tier1_signal(i, t1_close, t1_bb_low, t1_rsi, t1_atr, t1_ema, p):
            ep = t1_close[i]
            total_now = spot_units * ep + cash
            # 현물 전량 매도
            cash += spot_units * ep * (1 - TAKER_FEE)
            spot_units = 0.0
            # 선물: 총 자산의 30% × 7x
            notional = total_now * p["pos_ratio"] * p["leverage"]
            pos_amt  = notional / ep
            cash -= total_now * p["pos_ratio"] * TAKER_FEE  # 진입 수수료
            sl  = calc_sl(ep, t1_atr[i], p["sl_atr_mult"])
            tp  = calc_tp(ep, t1_atr[i], tp_mode, t1_bb_mid[i])
            entry_bar = i
            in_pos = True

    mr = monthly_series(ts, port_vals)
    st = portfolio_stats(port_vals, mr)
    st["trades"] = trades
    st["win_rate"] = round(wins/trades*100, 1) if trades else 0.0
    st["scenario"] = "C_full_switch"
    return st, port_vals, ts


def run_futures_only(df: pd.DataFrame, p: dict) -> dict:
    """선물 전용: 현물 없이 Tier1 신호로만 거래."""
    closes   = df["close"].to_numpy(float)
    highs    = df["high"].to_numpy(float)
    lows     = df["low"].to_numpy(float)
    ts       = df["timestamp"].to_numpy()
    t1_close = df["t1_close"].to_numpy(float)
    t1_bb_low= df["t1_bb_lower"].to_numpy(float)
    t1_bb_mid= df["t1_bb_mid"].to_numpy(float)
    t1_rsi   = df["t1_rsi"].to_numpy(float)
    t1_atr   = df["t1_atr"].to_numpy(float)
    t1_ema   = df["t1_ema200"].to_numpy(float)
    n = len(closes)

    balance  = INITIAL_BAL
    peak     = INITIAL_BAL
    in_pos   = False
    ep = sl = tp = pos_amt = entry_bar = 0.0
    tp_mode  = p["tp_mode"]
    trades   = wins = 0

    port_vals = np.empty(n)

    for i in range(n):
        unreal = (closes[i] - ep) * pos_amt if in_pos else 0.0
        port_vals[i] = balance + unreal
        if port_vals[i] > peak:
            peak = port_vals[i]

        if in_pos:
            tp_check = t1_bb_mid[i] if tp_mode == "bb_mid" else tp
            exit_p = None
            if lows[i] <= sl:
                exit_p = sl
            elif tp_check > ep and highs[i] >= tp_check:
                exit_p = tp_check; wins += 1
            elif i - entry_bar >= p["timeout_1h"]:
                exit_p = closes[i]
                if exit_p > ep: wins += 1
            if exit_p is not None:
                pnl = (exit_p - ep) * pos_amt
                fee = exit_p * pos_amt * TAKER_FEE
                balance += pnl - fee
                trades += 1
                in_pos = False

        if not in_pos and balance > 0 and tier1_signal(i, t1_close, t1_bb_low, t1_rsi, t1_atr, t1_ema, p):
            ep = t1_close[i]
            notional = balance * p["pos_ratio"] * p["leverage"]
            pos_amt  = notional / ep
            balance -= notional * TAKER_FEE
            sl  = calc_sl(ep, t1_atr[i], p["sl_atr_mult"])
            tp  = calc_tp(ep, t1_atr[i], tp_mode, t1_bb_mid[i])
            entry_bar = i
            in_pos = True

    mr = monthly_series(ts, port_vals)
    st = portfolio_stats(port_vals, mr)
    st["trades"] = trades
    st["win_rate"] = round(wins/trades*100, 1) if trades else 0.0
    st["scenario"] = "FuturesOnly"
    return st, port_vals, ts


def run_buy_hold(df: pd.DataFrame) -> dict:
    closes = df["close"].to_numpy(float)
    ts     = df["timestamp"].to_numpy()
    vals   = INITIAL_BAL * closes / closes[0]
    mr = monthly_series(ts, vals)
    st = portfolio_stats(vals, mr)
    st["trades"]   = 0
    st["win_rate"] = 0.0
    st["scenario"] = "BuyHold"
    return st, vals, ts


def print_results(coin: str, results: list):
    print(f"\n{'='*70}")
    print(f"[{coin.upper()}] 시나리오 비교 ({START[:4]}~{END[:4]})")
    print(f"\n  {'시나리오':18} {'수익률':>10} {'MDD':>8} {'샤프':>7} {'Calmar':>8} {'거래':>6} {'승률':>7}")
    print(f"  {'-'*68}")
    for st, _, _ in results:
        label = st["scenario"]
        print(f"  {label:18} {st['return_pct']:>+9.1f}% {st['max_drawdown']:>7.1f}% "
              f"{st['sharpe']:>7.2f} {st['calmar']:>8.2f} "
              f"{st['trades']:>5}건 {st['win_rate']:>6.1f}%")


def save_results(coin: str, results: list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for st, vals, ts in results:
        # 월별 수익률
        df_v = pd.DataFrame({"ts": ts, "value": vals})
        df_v["ts"] = pd.to_datetime(df_v["ts"])
        df_v = df_v.set_index("ts")
        monthly = df_v["value"].resample("ME").last().ffill()
        monthly_ret = monthly.pct_change().fillna(0) * 100
        for month, ret in monthly_ret.items():
            rows.append({
                "scenario":   st["scenario"],
                "month":      str(month)[:7],
                "monthly_ret": round(ret, 2),
            })
    df_out = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, f"spot_hybrid_{coin}.csv")
    df_out.to_csv(path, index=False)

    # 요약
    summary_rows = []
    for st, _, _ in results:
        summary_rows.append({k: v for k, v in st.items() if k not in ("scenario",) or True})
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS_DIR, f"spot_hybrid_{coin}_summary.csv"), index=False
    )
    return path


def main():
    print("현물 + 선물 하이브리드 시나리오 비교 (4년치, 2022 하락장 포함)")
    print(f"코인: {[c.upper() for c in COINS]} | 기간: {START} ~ {END}\n")

    for coin in COINS:
        print(f"[{coin.upper()}] 데이터 준비 중...", flush=True)
        df = build_arrays(coin)
        p  = TIER1[coin]

        r_bh  = run_buy_hold(df)
        r_fut = run_futures_only(df, p)
        r_b   = run_scenario_b(df, p)

        results = [r_bh, r_fut, r_b]
        print_results(coin, results)

        path = save_results(coin, results)
        print(f"\n  저장: {path}")


if __name__ == "__main__":
    main()
