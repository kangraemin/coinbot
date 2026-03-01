"""ICT + Elliott Wave 복합 전략 백테스트 (롱+숏, 기간별 비교)
=============================================================

ICT 구성요소:
  · Market Structure Break (BOS): 스윙 고/저점 돌파 → 추세 확인
  · Order Blocks (OB): W1 충격파 직전 마지막 역방향 캔들
  · Fair Value Gaps (FVG): W1 구간 내 3캔들 갭

Elliott Wave 필터:
  · W1: BOS 직전 스윙 저점→고점 (롱) / 고점→저점 (숏)
  · W2/W4: W1의 Fibonacci 되돌림 구간에서 OB/FVG 접촉 시 진입

분석 기간:
  · 3개월 (2025-11-28 ~)
  · 1년   (2025-02-28 ~)
  · 3년   (2023-02-28 ~)
"""

import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005

# ── 분석 기간 ────────────────────────────────────────────────────────────────
DATA_END = pd.Timestamp("2026-02-28", tz="UTC")
PERIODS  = {
    "3m": pd.Timestamp("2025-11-28", tz="UTC"),
    "1y": pd.Timestamp("2025-02-28", tz="UTC"),
    "3y": pd.Timestamp("2023-02-28", tz="UTC"),
}

# ── 파라미터 그리드 ──────────────────────────────────────────────────────────
SWING_WINDOWS  = [3, 5, 8]
RETRACE_LEVELS = [(0.382, 0.618), (0.382, 0.786), (0.500, 0.786)]
TP_EXTENSIONS  = [1.272, 1.618, 2.0]
SL_BUFFERS     = [0.002, 0.005]
LEVERAGES      = [3, 5, 7]
POS_RATIOS     = [0.10, 0.20]
SIGNAL_TYPES   = ["ob_only", "fvg_only", "both"]
ZONE_EXPIRY    = 60   # BOS 이후 유효 봉 수

COINS      = ["btc", "eth", "sol", "xrp"]
TIMEFRAMES = ["1H", "4H"]


# ── 데이터 로드 & 리샘플 ─────────────────────────────────────────────────────
def load_and_resample(coin: str, tf: str) -> pd.DataFrame:
    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.startswith(f"{coin}_1m_") and f.endswith(".parquet")
    ])
    if not files:
        raise FileNotFoundError(f"{coin} 데이터 없음")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    rule = tf.replace("H", "h")
    out = df.resample(rule).agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),    close=("close","last"),
        volume=("volume","sum"),
    ).dropna().reset_index()
    return out


# ── 스윙 감지 ────────────────────────────────────────────────────────────────
def detect_swings(highs: np.ndarray, lows: np.ndarray, window: int):
    n = len(highs)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        if (highs[i] > np.max(highs[i-window:i]) and
                highs[i] >= np.max(highs[i+1:i+window+1])):
            sh[i] = True
        if (lows[i] < np.min(lows[i-window:i]) and
                lows[i] <= np.min(lows[i+1:i+window+1])):
            sl[i] = True
    return sh, sl


# ── 존 빌드 (롱 + 숏) ───────────────────────────────────────────────────────
def build_all_zones(
    opens: np.ndarray,
    highs: np.ndarray,
    lows:  np.ndarray,
    closes: np.ndarray,
    window: int,
) -> list[dict]:
    n = len(closes)
    sh, sl  = detect_swings(highs, lows, window)
    sh_idx  = np.where(sh)[0]
    sl_idx  = np.where(sl)[0]

    # ── BOS 상향 (롱) ──────────────────────────────────────────────────
    bos_up = []
    last_up = 0
    for i in range(window * 2, n):
        post = sh_idx[(sh_idx > last_up) & (sh_idx < i)]
        if len(post) and closes[i] > highs[post[-1]]:
            bos_up.append(i)
            last_up = i

    # ── BOS 하향 (숏) ──────────────────────────────────────────────────
    bos_dn = []
    last_dn = 0
    for i in range(window * 2, n):
        post = sl_idx[(sl_idx > last_dn) & (sl_idx < i)]
        if len(post) and closes[i] < lows[post[-1]]:
            bos_dn.append(i)
            last_dn = i

    # ── 불리시 FVG: high[i-1] < low[i+1] ──────────────────────────────
    bull_fvgs = []
    for i in range(1, n - 1):
        if highs[i-1] < lows[i+1]:
            bull_fvgs.append((i, highs[i-1], lows[i+1]))

    # ── 베어리시 FVG: low[i-1] > high[i+1] ────────────────────────────
    bear_fvgs = []
    for i in range(1, n - 1):
        if lows[i-1] > highs[i+1]:
            bear_fvgs.append((i, highs[i+1], lows[i-1]))

    zones = []

    # ── 롱 존 (불리시 BOS 기준) ────────────────────────────────────────
    prev_up_map = {bos_up[k]: (bos_up[k-1] if k > 0 else 0)
                   for k in range(len(bos_up))}
    for bos_i in bos_up:
        pb = prev_up_map[bos_i]
        w1e_c = sh_idx[(sh_idx > pb) & (sh_idx < bos_i)]
        if not len(w1e_c): continue
        w1e_i = int(w1e_c[-1]);  w1e = highs[w1e_i]
        w1s_c = sl_idx[(sl_idx > pb) & (sl_idx < w1e_i)]
        if not len(w1s_c): continue
        w1s_i = int(w1s_c[-1]);  w1s = lows[w1s_i]
        w1sz  = w1e - w1s
        if w1sz <= 0: continue
        exp   = bos_i + ZONE_EXPIRY
        base  = dict(bos_i=bos_i, expire_i=exp,
                     w1_start=w1s, w1_end=w1e, w1_size=w1sz,
                     used=False, direction="long")
        # 불리시 OB
        for j in range(w1s_i, max(0, w1s_i-20)-1, -1):
            if closes[j] < opens[j]:
                zones.append({**base, "zone_lo": lows[j],
                              "zone_hi": highs[j], "type": "ob"})
                break
        # 불리시 FVG (W1 구간 내, 높은 것부터 3개)
        fw = [(fi,fl,fh) for fi,fl,fh in bull_fvgs if w1s_i<=fi<=w1e_i]
        fw.sort(key=lambda x: x[2], reverse=True)
        for fi,fl,fh in fw[:3]:
            zones.append({**base, "zone_lo": fl, "zone_hi": fh, "type": "fvg"})

    # ── 숏 존 (베어리시 BOS 기준) ──────────────────────────────────────
    prev_dn_map = {bos_dn[k]: (bos_dn[k-1] if k > 0 else 0)
                   for k in range(len(bos_dn))}
    for bos_i in bos_dn:
        pb = prev_dn_map[bos_i]
        w1e_c = sl_idx[(sl_idx > pb) & (sl_idx < bos_i)]  # W1 end = swing low
        if not len(w1e_c): continue
        w1e_i = int(w1e_c[-1]);  w1e = lows[w1e_i]
        w1s_c = sh_idx[(sh_idx > pb) & (sh_idx < w1e_i)]  # W1 start = swing high
        if not len(w1s_c): continue
        w1s_i = int(w1s_c[-1]);  w1s = highs[w1s_i]
        w1sz  = w1s - w1e   # 양수
        if w1sz <= 0: continue
        exp   = bos_i + ZONE_EXPIRY
        base  = dict(bos_i=bos_i, expire_i=exp,
                     w1_start=w1s, w1_end=w1e, w1_size=w1sz,
                     used=False, direction="short")
        # 베어리시 OB: w1_start(스윙 고점) 직전 마지막 양봉
        for j in range(w1s_i, max(0, w1s_i-20)-1, -1):
            if closes[j] > opens[j]:
                zones.append({**base, "zone_lo": lows[j],
                              "zone_hi": highs[j], "type": "ob"})
                break
        # 베어리시 FVG (W1 구간 내, 낮은 것부터 3개)
        fw = [(fi,fl,fh) for fi,fl,fh in bear_fvgs if w1e_i<=fi<=w1s_i]
        fw.sort(key=lambda x: x[1])
        for fi,fl,fh in fw[:3]:
            zones.append({**base, "zone_lo": fl, "zone_hi": fh, "type": "fvg"})

    return zones


# ── 단일 파라미터 백테스트 ──────────────────────────────────────────────────
def run_backtest(
    highs: np.ndarray,
    lows:  np.ndarray,
    closes: np.ndarray,
    zones_template: list[dict],
    retrace_min: float,
    retrace_max: float,
    tp_ext:      float,
    sl_buf:      float,
    leverage:    int,
    pos_ratio:   float,
    signal_type: str,
    initial_balance: float = 1000.0,
) -> dict:
    n = len(closes)
    type_filter = {"ob"} if signal_type == "ob_only" else \
                  {"fvg"} if signal_type == "fvg_only" else {"ob", "fvg"}

    zones = [dict(z) for z in zones_template if z["type"] in type_filter]
    zones.sort(key=lambda z: z["bos_i"])

    balance = initial_balance
    peak    = initial_balance
    max_dd  = 0.0
    trades = wins = long_trades = short_trades = 0

    in_pos    = False
    direction = "long"   # current position direction
    entry_price = tp_price = sl_price = pos_amt = 0.0

    zone_ptr = 0
    active: list[dict] = []

    for i in range(n):
        while zone_ptr < len(zones) and zones[zone_ptr]["bos_i"] <= i:
            active.append(zones[zone_ptr])
            zone_ptr += 1
        active = [z for z in active if not z["used"] and i < z["expire_i"]]

        if not in_pos:
            if balance <= 0:
                break
            for z in active:
                d = z["direction"]

                if d == "long":
                    # 가격이 존에 닿았는가 (위에서 아래로 접촉)
                    if lows[i] > z["zone_hi"] or closes[i] < z["zone_lo"]:
                        continue
                    if closes[i] >= z["w1_end"]:
                        continue
                    retrace = (z["w1_end"] - closes[i]) / z["w1_size"]

                else:  # short
                    # 가격이 존에 닿았는가 (아래에서 위로 접촉)
                    if highs[i] < z["zone_lo"] or closes[i] > z["zone_hi"]:
                        continue
                    if closes[i] <= z["w1_end"]:
                        continue
                    retrace = (closes[i] - z["w1_end"]) / z["w1_size"]

                if not (retrace_min <= retrace <= retrace_max):
                    continue

                # ── 진입 ─────────────────────────────────────────────
                entry_price = closes[i]
                notional    = balance * pos_ratio * leverage
                pos_amt     = notional / entry_price
                balance    -= notional * TAKER_FEE
                direction   = d
                z["used"]   = True

                if d == "long":
                    tp_price = entry_price + z["w1_size"] * tp_ext
                    sl_price = z["zone_lo"] * (1.0 - sl_buf)
                else:
                    tp_price = entry_price - z["w1_size"] * tp_ext
                    sl_price = z["zone_hi"] * (1.0 + sl_buf)

                in_pos = True
                break

        else:
            exit_price = None
            won = False

            if direction == "long":
                if lows[i] <= sl_price:
                    exit_price = sl_price
                elif highs[i] >= tp_price:
                    exit_price = tp_price;  won = True
            else:  # short
                if highs[i] >= sl_price:
                    exit_price = sl_price
                elif lows[i] <= tp_price:
                    exit_price = tp_price;  won = True

            if exit_price is not None:
                pnl = (exit_price - entry_price) * pos_amt * (1 if direction == "long" else -1)
                fee = abs(exit_price * pos_amt) * TAKER_FEE
                balance += pnl - fee
                trades  += 1
                if won: wins += 1
                if direction == "long":  long_trades  += 1
                else:                    short_trades += 1
                in_pos = False

            if balance > peak: peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd: max_dd = dd

    ret      = (balance - initial_balance) / initial_balance * 100 if balance > 0 else -100.0
    win_rate = wins / trades * 100 if trades > 0 else 0.0
    calmar   = ret / max_dd if max_dd > 0 else 0.0

    return dict(
        retrace_min=retrace_min, retrace_max=retrace_max,
        tp_ext=tp_ext, sl_buf=sl_buf, leverage=leverage,
        pos_ratio=pos_ratio, signal_type=signal_type,
        trades=trades, long_trades=long_trades, short_trades=short_trades,
        win_rate=round(win_rate, 1),
        return_pct=round(ret, 2),
        max_drawdown=round(max_dd, 2),
        calmar=round(calmar, 2),
        final_balance=round(balance, 2),
    )


# ── Worker ─────────────────────────────────────────────────────────────────
def worker(args: tuple) -> tuple[str, pd.DataFrame]:
    coin, tf, period_name = args
    print(f"[{coin.upper()} {tf} {period_name}] 시작...", flush=True)

    try:
        df_full = load_and_resample(coin, tf)
    except Exception as e:
        print(f"[{coin.upper()} {tf} {period_name}] 로드 실패: {e}", flush=True)
        return (f"{coin}_{tf}_{period_name}", pd.DataFrame())

    # 기간 필터
    start_ts = PERIODS[period_name]
    df = df_full[df_full["timestamp"] >= start_ts].reset_index(drop=True)
    if len(df) < 100:
        print(f"[{coin.upper()} {tf} {period_name}] 데이터 부족 ({len(df)}봉)", flush=True)
        return (f"{coin}_{tf}_{period_name}", pd.DataFrame())

    opens  = df["open"].to_numpy(dtype=float)
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    combos = list(itertools.product(
        RETRACE_LEVELS, TP_EXTENSIONS, SL_BUFFERS,
        LEVERAGES, POS_RATIOS, SIGNAL_TYPES,
    ))

    all_results = []
    for sw in SWING_WINDOWS:
        zones_tmpl = build_all_zones(opens, highs, lows, closes, sw)
        bos_count  = len({z["bos_i"] for z in zones_tmpl})
        for (rmin, rmax), tp_e, sl_b, lev, pos, sig in combos:
            r = run_backtest(highs, lows, closes, zones_tmpl,
                             rmin, rmax, tp_e, sl_b, lev, pos, sig)
            r.update(swing_window=sw, coin=coin,
                     timeframe=tf, period=period_name, n_bos=bos_count)
            all_results.append(r)

    df_res = pd.DataFrame(all_results).sort_values("calmar", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"ict_ew_{coin}_{tf}_{period_name}.csv")
    df_res.to_csv(out, index=False)

    best    = df_res.iloc[0]
    n_pos   = (df_res["return_pct"] > 0).sum()
    n_total = len(df_res)
    print(
        f"[{coin.upper()} {tf} {period_name}] 완료 | "
        f"봉수={len(df)} BOS={bos_count} | "
        f"수익>0: {n_pos}/{n_total} ({n_pos/n_total*100:.0f}%) | "
        f"Best: ret={best['return_pct']:+.1f}% win={best['win_rate']}% "
        f"MDD={best['max_drawdown']:.1f}% Calmar={best['calmar']:.2f} "
        f"L={best['long_trades']}건 S={best['short_trades']}건",
        flush=True,
    )
    return (f"{coin}_{tf}_{period_name}", df_res)


# ── 요약 출력 ───────────────────────────────────────────────────────────────
def print_summary(all_dfs: list[pd.DataFrame]):
    if not all_dfs:
        print("결과 없음"); return

    combined = pd.concat(all_dfs, ignore_index=True)

    # ── 1) 기간 × 코인 × 타임프레임: 최고 Calmar 비교 ─────────────────
    print("\n" + "=" * 90)
    print("ICT + Elliott Wave 분석 요약 (롱+숏)")
    print("=" * 90)

    print("\n▶ 기간별 최고 성과 (Calmar 기준)")
    pivot = []
    for period in ["3m", "1y", "3y"]:
        for tf in TIMEFRAMES:
            for coin in COINS:
                sub = combined[(combined["period"]==period) &
                               (combined["timeframe"]==tf) &
                               (combined["coin"]==coin)]
                if sub.empty: continue
                b = sub.iloc[0]
                pivot.append(dict(
                    기간=period, TF=tf, 코인=coin.upper(),
                    거래=int(b["trades"]),
                    롱=int(b["long_trades"]), 숏=int(b["short_trades"]),
                    승률=b["win_rate"],
                    수익률=b["return_pct"],
                    MDD=b["max_drawdown"],
                    Calmar=b["calmar"],
                    신호=b["signal_type"],
                    스윙W=int(b["swing_window"]),
                    TP=b["tp_ext"],
                ))
    df_pivot = pd.DataFrame(pivot)
    if not df_pivot.empty:
        print(df_pivot.to_string(index=False))

    # ── 2) 기간별 4H 코인 매트릭스 (한눈에 비교) ──────────────────────
    print("\n▶ 4H 기간별 성과 매트릭스 (Best Calmar | 수익률/MDD/Calmar)")
    header = f"{'코인':<6}" + "".join(f"{'3개월':>22}{'1년':>22}{'3년':>22}")
    print(header)
    print("-" * 72)
    for coin in COINS:
        row = f"{coin.upper():<6}"
        for period in ["3m", "1y", "3y"]:
            sub = combined[(combined["period"]==period) &
                           (combined["timeframe"]=="4H") &
                           (combined["coin"]==coin)]
            if sub.empty:
                row += f"{'N/A':>22}"
            else:
                b = sub.iloc[0]
                cell = f"{b['return_pct']:+.1f}% / {b['max_drawdown']:.1f}% / {b['calmar']:.2f}"
                row += f"{cell:>22}"
        print(row)

    # ── 3) 신호 유형별 성과 (기간 합산) ────────────────────────────────
    print("\n▶ 신호 유형별 중앙값 성과")
    sig_grp = (combined.groupby(["period", "signal_type"])
               .agg(trades=("trades","median"),
                    win_rate=("win_rate","median"),
                    return_pct=("return_pct","median"),
                    calmar=("calmar","median"))
               .round(2))
    print(sig_grp.to_string())

    # ── 4) 롱 vs 숏 비중 ────────────────────────────────────────────────
    print("\n▶ 기간별 롱/숏 거래 비중 (전체 평균)")
    ls_grp = (combined.groupby("period")
              .agg(avg_long=("long_trades","mean"),
                   avg_short=("short_trades","mean"),
                   avg_total=("trades","mean"))
              .round(1))
    ls_grp["롱비중(%)"] = (ls_grp["avg_long"] / ls_grp["avg_total"].replace(0,1) * 100).round(1)
    ls_grp["숏비중(%)"] = (ls_grp["avg_short"] / ls_grp["avg_total"].replace(0,1) * 100).round(1)
    print(ls_grp[["avg_long","avg_short","avg_total","롱비중(%)","숏비중(%)"]].to_string())

    # ── 5) Fibonacci 되돌림 구간 효과 ─────────────────────────────────
    print("\n▶ Fibonacci 되돌림 구간별 중앙값 성과 (전 기간)")
    fib_grp = (combined.assign(
        fib=combined["retrace_min"].astype(str) + "~" + combined["retrace_max"].astype(str)
    ).groupby("fib")
     .agg(win_rate=("win_rate","median"),
          return_pct=("return_pct","median"),
          calmar=("calmar","median"))
     .round(2))
    print(fib_grp.to_string())


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    tasks = list(itertools.product(COINS, TIMEFRAMES, PERIODS.keys()))
    n_per_sw  = (len(RETRACE_LEVELS) * len(TP_EXTENSIONS) * len(SL_BUFFERS) *
                 len(LEVERAGES) * len(POS_RATIOS) * len(SIGNAL_TYPES))
    total_per = n_per_sw * len(SWING_WINDOWS)
    total     = total_per * len(tasks)

    print("ICT + Elliott Wave 복합 지표 백테스트 (롱+숏)")
    print(f"코인: {[c.upper() for c in COINS]}  TF: {TIMEFRAMES}")
    print(f"기간: {list(PERIODS.keys())}  존 만료: BOS+{ZONE_EXPIRY}봉")
    print(f"조합: {total_per}개/태스크 × {len(tasks)}태스크 = {total:,}개\n")

    all_results: list[pd.DataFrame] = []

    with ProcessPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                key, df_res = fut.result()
                if not df_res.empty:
                    all_results.append(df_res)
            except Exception as e:
                print(f"  [{task}] 오류: {e}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
