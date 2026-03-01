"""Dual-Tier 시그널 백테스트 — XRP, SOL, ETH, BTC, 2023~2025.

개념:
  Tier 1 (4h, 고품질): BB+RSI 과매도 → 30% 비중
  Tier 2 (1h, 중품질): BB+RSI 과매도 → 15% 비중
  - Tier 1 선점: Tier 2 활성 중 Tier 1 신호 → Tier 2 청산 후 Tier 1 진입
  - Tier 1 없는 코인(BTC): Tier 2 단독 운영

파라미터 (확정값):
  XRP Tier1(4h): RSI<30, sl×1.5, tp=bb_mid,  7x, 30%, EMA200=OFF
  XRP Tier2(1h): RSI<40, sl×1.0, tp=atr_3x,  7x, 15%, EMA200=ON
  SOL Tier1(4h): RSI<45, sl×1.0, tp=atr_2x,  7x, 30%, EMA200=ON
  SOL Tier2(1h): RSI<40, sl×2.0, tp=bb_mid,  7x, 15%, EMA200=ON
  ETH Tier1(4h): RSI<45, sl×2.0, tp=atr_2x,  7x, 30%, EMA200=ON
  ETH Tier2(1h): RSI<45, sl×2.0, tp=atr_2x,  7x, 15%, EMA200=ON
  BTC Tier2(1h): RSI<35, sl×1.0, tp=atr_3x,  3x, 10%, EMA200=ON  (Tier1 없음)
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
INITIAL_BAL = 1000.0
START       = "2023-01-01"
END         = "2025-12-31"

# ── 확정 파라미터 ────────────────────────────────────
@dataclass
class TierParams:
    rsi_thresh:   float
    sl_atr_mult:  float
    tp_mode:      str       # "bb_mid" | "atr_2x" | "atr_3x"
    leverage:     int
    pos_ratio:    float
    use_ema200:   bool
    timeout_bars: int       # 해당 타임프레임 기준 봉 수


_DISABLED_TIER1 = TierParams(0, 1.0, "atr_2x", 1, 0.0, True, 48)  # rsi_thresh=0 → 절대 발동 안 함

CONFIGS = {
    "xrp": {
        "tier1": TierParams(30,  1.5, "bb_mid",  7, 0.30, False, 48),
        "tier2": TierParams(40,  1.0, "atr_3x",  7, 0.15, True,  48),
    },
    "sol": {
        "tier1": TierParams(45,  1.0, "atr_2x",  7, 0.30, True,  48),
        "tier2": TierParams(40,  2.0, "bb_mid",  7, 0.15, True,  48),
    },
    "eth": {
        "tier1": TierParams(45,  2.0, "atr_2x",  7, 0.30, True,  48),
        "tier2": TierParams(45,  2.0, "atr_2x",  7, 0.15, True,  48),
    },
    "btc": {
        "tier1": _DISABLED_TIER1,  # 4h 손실 — Tier1 비활성
        "tier2": TierParams(35,  1.0, "atr_3x",  3, 0.10, True,  48),
    },
}

COINS = ["xrp", "sol", "eth", "btc"]


# ── 데이터 로드 ─────────────────────────────────────

def load_range(coin: str) -> pd.DataFrame:
    frames = []
    for y in range(2022, 2026):
        path = os.path.join(DATA_DIR, f"{coin}_1m_{y}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").set_index("timestamp")


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    r = df.resample(rule).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),   close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    return r.reset_index()


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

    tr = pd.concat([h - lo,
                    (h - c.shift(1)).abs(),
                    (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr    = tr.ewm(com=13, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()

    df = df.copy()
    df["bb_mid"]   = bb_mid
    df["bb_lower"] = bb_lower
    df["rsi"]      = rsi
    df["atr"]      = atr
    df["ema200"]   = ema200
    return df


# ── 메인 백테스트 루프 ────────────────────────────────

@dataclass
class Position:
    tier:          int
    entry_price:   float
    sl_price:      float
    tp_price:      float        # 0 = bb_mid 모드
    tp_mode:       str
    pos_amount:    float
    entry_bar:     int          # 1h 기준 인덱스
    timeout_1h:    int          # 1h 봉 수 기준 타임아웃


def run_dual_tier(coin: str) -> dict:
    cfg1: TierParams = CONFIGS[coin]["tier1"]
    cfg2: TierParams = CONFIGS[coin]["tier2"]

    # ── 데이터 준비 ──
    df_1m = load_range(coin)
    df_1h = resample(df_1m, "1h")
    df_4h = resample(df_1m, "4h")
    df_1h = add_indicators(df_1h)
    df_4h = add_indicators(df_4h)

    # 4h 지표를 1h 타임라인에 merge
    # - 진입 신호용 (close/rsi/bb_lower/atr/ema200): 4h 마감 시각에만 값 (NaN으로 신호 감지)
    # - 포지션 관리용 (bb_mid): forward-fill로 현재 4h bb_mid를 항상 참조 가능하게
    df_4h_idx = df_4h.set_index("timestamp")[
        ["close", "high", "low", "bb_mid", "bb_lower", "rsi", "atr", "ema200"]
    ].rename(columns=lambda c: f"t1_{c}")
    df = df_1h.join(df_4h_idx, on="timestamp", how="left")
    # bb_mid만 ffill — TP 체크 시 현재 4h 중심선을 매 1h 봉에서 사용
    df["t1_bb_mid"] = df["t1_bb_mid"].ffill()

    # 평가 구간
    s = pd.Timestamp(START, tz="UTC")
    e = pd.Timestamp(END + " 23:59:59", tz="UTC")
    df = df[(df["timestamp"] >= s) & (df["timestamp"] <= e)].reset_index(drop=True)

    # numpy 배열
    ts        = df["timestamp"].to_numpy()
    closes    = df["close"].to_numpy(float)
    highs     = df["high"].to_numpy(float)
    lows      = df["low"].to_numpy(float)
    bb_mid_1h = df["bb_mid"].to_numpy(float)
    bb_low_1h = df["bb_lower"].to_numpy(float)
    rsi_1h    = df["rsi"].to_numpy(float)
    atr_1h    = df["atr"].to_numpy(float)
    ema_1h    = df["ema200"].to_numpy(float)

    t1_close  = df["t1_close"].to_numpy(float)
    t1_high   = df["t1_high"].to_numpy(float)
    t1_low    = df["t1_low"].to_numpy(float)
    t1_bb_mid = df["t1_bb_mid"].to_numpy(float)
    t1_bb_low = df["t1_bb_lower"].to_numpy(float)
    t1_rsi    = df["t1_rsi"].to_numpy(float)
    t1_atr    = df["t1_atr"].to_numpy(float)
    t1_ema    = df["t1_ema200"].to_numpy(float)

    n = len(closes)

    balance      = INITIAL_BAL
    peak         = INITIAL_BAL
    max_dd       = 0.0
    position: Optional[Position] = None

    # 거래 기록
    trade_log = []
    monthly_pnl: dict = {}  # "YYYY-MM" → [pnl_pct, ...]

    def enter(tier: int, ep: float, atr_val: float, bb_mid_val: float,
              tp_mode: str, pos_ratio: float, lev: int, timeout_1h: int, bar: int):
        nonlocal balance, position
        notional   = balance * pos_ratio * lev
        pos_amount = notional / ep
        balance   -= notional * TAKER_FEE
        if tp_mode == "atr_2x":
            tp_price = ep + atr_val * 2.0
        elif tp_mode == "atr_3x":
            tp_price = ep + atr_val * 3.0
        else:
            tp_price = 0.0  # bb_mid: 동적
        sl_price = ep - atr_val * (cfg1.sl_atr_mult if tier == 1 else cfg2.sl_atr_mult)
        position = Position(
            tier=tier, entry_price=ep, sl_price=sl_price,
            tp_price=tp_price, tp_mode=tp_mode,
            pos_amount=pos_amount, entry_bar=bar, timeout_1h=timeout_1h,
        )

    def exit_pos(exit_price: float, bar: int, won: bool):
        nonlocal balance, position, peak, max_dd
        pos = position
        pnl  = (exit_price - pos.entry_price) * pos.pos_amount
        fee  = exit_price * pos.pos_amount * TAKER_FEE
        balance += pnl - fee
        pnl_pct  = (pnl - fee) / INITIAL_BAL * 100
        month    = str(ts[bar])[:7]
        monthly_pnl.setdefault(month, []).append(pnl_pct)
        trade_log.append({
            "exit_ts":   ts[bar],
            "tier":      pos.tier,
            "entry":     round(pos.entry_price, 6),
            "exit":      round(exit_price, 6),
            "pnl_pct":   round(pnl_pct, 3),
            "won":       won,
        })
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100
        if dd > max_dd:
            max_dd = dd
        position = None

    for i in range(1, n):
        # ── 포지션 관리 ──
        if position is not None:
            pos = position
            # TP 체크용 bb_mid 결정 (진입한 티어 기준)
            if pos.tp_mode == "bb_mid":
                tp_check = t1_bb_mid[i] if pos.tier == 1 else bb_mid_1h[i]
                tp_check = tp_check if not np.isnan(tp_check) else 0.0
            else:
                tp_check = pos.tp_price

            exited = False
            if lows[i] <= pos.sl_price:
                exit_pos(pos.sl_price, i, False); exited = True
            elif tp_check > pos.entry_price and highs[i] >= tp_check:
                exit_pos(tp_check, i, True); exited = True
            elif i - pos.entry_bar >= pos.timeout_1h:
                exit_pos(closes[i], i, closes[i] > pos.entry_price); exited = True

            if not exited:
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

        # ── 신호 체크: Tier 1 선점(Preemption) 로직 ──
        # Tier 1 신호 체크 (4h 캔들 마감 시각에만)
        tier1_signal = False
        if not np.isnan(t1_close[i]) and balance > 0:
            ok1 = (not np.isnan(t1_bb_low[i]) and not np.isnan(t1_rsi[i]) and
                   not np.isnan(t1_atr[i]) and not np.isnan(t1_ema[i]))
            if ok1:
                ema_pass1 = (not cfg1.use_ema200) or (t1_close[i] > t1_ema[i])
                if ema_pass1 and t1_close[i] < t1_bb_low[i] and t1_rsi[i] < cfg1.rsi_thresh:
                    tier1_signal = True

        if tier1_signal:
            # Tier 2 포지션 활성 중이면 즉시 시가 청산 후 Tier 1 진입
            if position is not None and position.tier == 2:
                exit_pos(t1_close[i], i, t1_close[i] > position.entry_price)
            if position is None:
                enter(1, t1_close[i], t1_atr[i], t1_bb_mid[i],
                      cfg1.tp_mode, cfg1.pos_ratio, cfg1.leverage,
                      cfg1.timeout_bars * 4, i)

        # Tier 2 신호 체크 (Tier 1 포지션 없을 때만)
        elif position is None and balance > 0:
            ok2 = (not np.isnan(bb_low_1h[i]) and not np.isnan(rsi_1h[i]) and
                   not np.isnan(atr_1h[i]) and not np.isnan(ema_1h[i]))
            if ok2:
                ema_pass2 = (not cfg2.use_ema200) or (closes[i] > ema_1h[i])
                if ema_pass2 and closes[i] < bb_low_1h[i] and rsi_1h[i] < cfg2.rsi_thresh:
                    enter(2, closes[i], atr_1h[i], bb_mid_1h[i],
                          cfg2.tp_mode, cfg2.pos_ratio, cfg2.leverage,
                          cfg2.timeout_bars, i)

    ret = (balance - INITIAL_BAL) / INITIAL_BAL * 100 if balance > 0 else -100.0
    trades = len(trade_log)
    wins   = sum(1 for t in trade_log if t["won"])
    win_rate = wins / trades * 100 if trades > 0 else 0.0
    calmar = ret / max_dd if max_dd > 0 else 0.0

    # 샤프 (월간 수익률 기준)
    monthly_rets = [sum(v) for v in monthly_pnl.values()]
    sharpe = 0.0
    if len(monthly_rets) > 1:
        mr = np.array(monthly_rets)
        sharpe = round(mr.mean() / (mr.std(ddof=1) + 1e-10) * np.sqrt(12), 2)

    # 티어별 집계
    t1_trades = [t for t in trade_log if t["tier"] == 1]
    t2_trades = [t for t in trade_log if t["tier"] == 2]

    # 월평균 거래수
    months = max(len(monthly_pnl), 1)
    trades_per_month = round(trades / months, 1)

    return {
        "coin":             coin.upper(),
        "return_pct":       round(ret, 2),
        "max_drawdown":     round(max_dd, 2),
        "sharpe":           sharpe,
        "calmar":           round(calmar, 2),
        "trades":           trades,
        "trades_per_month": trades_per_month,
        "win_rate":         round(win_rate, 1),
        "tier1_trades":     len(t1_trades),
        "tier1_win_rate":   round(sum(1 for t in t1_trades if t["won"]) / len(t1_trades) * 100, 1) if t1_trades else 0.0,
        "tier2_trades":     len(t2_trades),
        "tier2_win_rate":   round(sum(1 for t in t2_trades if t["won"]) / len(t2_trades) * 100, 1) if t2_trades else 0.0,
        "trade_log":        trade_log,
        "monthly_pnl":      monthly_pnl,
    }


def save_results(coin: str, result: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 거래 로그 CSV
    df_trades = pd.DataFrame(result["trade_log"])
    out = os.path.join(RESULTS_DIR, f"dual_tier_{coin.lower()}.csv")
    df_trades.to_csv(out, index=False)

    # 월별 요약 CSV
    rows = []
    for month, pnls in sorted(result["monthly_pnl"].items()):
        rows.append({
            "month":      month,
            "trades":     len(pnls),
            "total_pnl":  round(sum(pnls), 3),
            "avg_pnl":    round(sum(pnls) / len(pnls), 3),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, f"dual_tier_{coin.lower()}_monthly.csv"), index=False
    )
    return out


def print_monthly(coin: str, result: dict):
    print(f"\n[{coin.upper()}] 월별 수익 (단위: % of 초기자본)")
    print(f"  {'월':>8}  {'거래':>4}  {'합산PnL':>9}")
    print(f"  {'-'*28}")
    for month, pnls in sorted(result["monthly_pnl"].items()):
        sign = "+" if sum(pnls) >= 0 else ""
        print(f"  {month:>8}  {len(pnls):>4}건  {sign}{sum(pnls):>8.2f}%")


def main():
    print("Dual-Tier 시그널 백테스트 — XRP / SOL")
    print(f"기간: {START} ~ {END}\n")

    for coin in COINS:
        print(f"[{coin.upper()}] 처리 중...", flush=True)
        result = run_dual_tier(coin)

        print(f"\n{'='*60}")
        print(f"[{coin.upper()}] 결과")
        print(f"  총수익률  : {result['return_pct']:+.2f}%")
        print(f"  MDD       : {result['max_drawdown']:.1f}%")
        print(f"  샤프      : {result['sharpe']:.2f}")
        print(f"  Calmar    : {result['calmar']:.2f}")
        print(f"  총거래    : {result['trades']}건 ({result['trades_per_month']}/월)")
        print(f"  승률      : {result['win_rate']:.1f}%")
        print(f"  Tier1     : {result['tier1_trades']}건 (승률 {result['tier1_win_rate']:.1f}%)")
        print(f"  Tier2     : {result['tier2_trades']}건 (승률 {result['tier2_win_rate']:.1f}%)")

        print_monthly(coin, result)

        path = save_results(coin, result)
        print(f"\n  저장: {path}")


if __name__ == "__main__":
    main()
