"""ICT + Elliott Wave 듀얼 티어 전략 연구
==========================================

High Tier (4H, 확실한 신호):
  · 엄격한 Fibonacci (0.5~0.786), OB/FVG, 큰 비중
  · 월 2~3건 목표 → 자본의 30~50% 투입

Low Tier (1H, 잦은 거래):
  · 넓은 Fibonacci (0.382~0.618), OB+FVG, 작은 비중
  · 주 2~5건 목표 → 자본의 5~15% 투입

복합 운영:
  · 두 티어가 별도 자본 버킷으로 동시 운영
  · 1H 타임라인 기준, 4H 존은 타임스탬프 매핑
  · 롱/숏 모두 지원

파라미터 탐색:
  · alloc_high: 고확신 티어 자본 배분 비율
  · h_pos_ratio: 고확신 버킷 내 거래당 비중
  · l_pos_ratio: 자주 거래 버킷 내 거래당 비중
  · leverage, signal_type, Fibonacci 범위 등
"""

import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "market")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE   = 0.0005
ZONE_EXPIRY_4H = 30   # 4H 존 만료 (4H봉 기준)
ZONE_EXPIRY_1H = 60   # 1H 존 만료 (1H봉 기준)

# ── 파라미터 그리드 ──────────────────────────────────────────────────────────
# 자본 배분
ALLOC_HIGH     = [0.3, 0.5, 0.7]        # 고확신 버킷 비율

# High Tier (4H) — 엄격한 파라미터
H_SWING        = [3, 5, 8]
H_RETRACE      = [(0.382, 0.618), (0.500, 0.786)]
H_TP_EXT       = [1.272, 1.618, 2.0]
H_SL_BUF       = [0.002, 0.005]
H_LEV          = [5, 7]
H_POS_RATIO    = [0.30, 0.50]            # 버킷 내 거래당 비중
H_SIG          = ["ob_only", "fvg_only"]

# Low Tier (1H) — 넓은 파라미터
L_SWING        = [3, 5]
L_RETRACE      = [(0.382, 0.618), (0.382, 0.786)]
L_TP_EXT       = [1.272, 1.618]
L_SL_BUF       = [0.002]
L_LEV          = [3, 5]
L_POS_RATIO    = [0.10, 0.20]
L_SIG          = ["ob_only", "both"]

COINS = ["btc", "eth", "sol", "xrp"]


# ── 데이터 로드 ──────────────────────────────────────────────────────────────
def load_data(coin: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1H 및 4H 데이터프레임 반환"""
    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.startswith(f"{coin}_1m_") and f.endswith(".parquet")
    ])
    if not files:
        raise FileNotFoundError(f"{coin} 데이터 없음")
    df1m = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df1m.columns = [c.lower() for c in df1m.columns]
    df1m["timestamp"] = pd.to_datetime(df1m["timestamp"], utc=True)
    df1m = df1m.sort_values("timestamp").set_index("timestamp")

    def resample(rule):
        return df1m.resample(rule).agg(
            open=("open","first"), high=("high","max"),
            low=("low","min"),    close=("close","last"),
            volume=("volume","sum"),
        ).dropna().reset_index()

    return resample("1h"), resample("4h")


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


# ── 존 빌드 (롱+숏) ──────────────────────────────────────────────────────────
def build_zones(
    opens: np.ndarray, highs: np.ndarray,
    lows: np.ndarray,  closes: np.ndarray,
    window: int, zone_expiry: int,
) -> list[dict]:
    n = len(closes)
    sh, sl  = detect_swings(highs, lows, window)
    sh_idx  = np.where(sh)[0]
    sl_idx  = np.where(sl)[0]

    # BOS 상향
    bos_up = []; last_up = 0
    for i in range(window*2, n):
        post = sh_idx[(sh_idx > last_up) & (sh_idx < i)]
        if len(post) and closes[i] > highs[post[-1]]:
            bos_up.append(i); last_up = i

    # BOS 하향
    bos_dn = []; last_dn = 0
    for i in range(window*2, n):
        post = sl_idx[(sl_idx > last_dn) & (sl_idx < i)]
        if len(post) and closes[i] < lows[post[-1]]:
            bos_dn.append(i); last_dn = i

    # FVG
    bull_fvgs = [(i, highs[i-1], lows[i+1])
                 for i in range(1, n-1) if highs[i-1] < lows[i+1]]
    bear_fvgs = [(i, highs[i+1], lows[i-1])
                 for i in range(1, n-1) if lows[i-1] > highs[i+1]]

    zones = []

    # ── 불리시 존 ─────────────────────────────────────────────────────
    prev_up = {bos_up[k]: (bos_up[k-1] if k > 0 else 0) for k in range(len(bos_up))}
    for bos_i in bos_up:
        pb = prev_up[bos_i]
        w1e_c = sh_idx[(sh_idx > pb) & (sh_idx < bos_i)]
        if not len(w1e_c): continue
        w1e_i = int(w1e_c[-1]); w1e = highs[w1e_i]
        w1s_c = sl_idx[(sl_idx > pb) & (sl_idx < w1e_i)]
        if not len(w1s_c): continue
        w1s_i = int(w1s_c[-1]); w1s = lows[w1s_i]
        w1sz = w1e - w1s
        if w1sz <= 0: continue
        exp = bos_i + zone_expiry
        base = dict(bos_i=bos_i, expire_i=exp,
                    w1_start=w1s, w1_end=w1e, w1_size=w1sz,
                    used=False, direction="long")
        for j in range(w1s_i, max(0, w1s_i-20)-1, -1):
            if closes[j] < opens[j]:
                zones.append({**base, "zone_lo": lows[j], "zone_hi": highs[j], "type": "ob"})
                break
        fw = [(fi,fl,fh) for fi,fl,fh in bull_fvgs if w1s_i<=fi<=w1e_i]
        fw.sort(key=lambda x: x[2], reverse=True)
        for fi,fl,fh in fw[:3]:
            zones.append({**base, "zone_lo": fl, "zone_hi": fh, "type": "fvg"})

    # ── 베어리시 존 ───────────────────────────────────────────────────
    prev_dn = {bos_dn[k]: (bos_dn[k-1] if k > 0 else 0) for k in range(len(bos_dn))}
    for bos_i in bos_dn:
        pb = prev_dn[bos_i]
        w1e_c = sl_idx[(sl_idx > pb) & (sl_idx < bos_i)]
        if not len(w1e_c): continue
        w1e_i = int(w1e_c[-1]); w1e = lows[w1e_i]
        w1s_c = sh_idx[(sh_idx > pb) & (sh_idx < w1e_i)]
        if not len(w1s_c): continue
        w1s_i = int(w1s_c[-1]); w1s = highs[w1s_i]
        w1sz = w1s - w1e
        if w1sz <= 0: continue
        exp = bos_i + zone_expiry
        base = dict(bos_i=bos_i, expire_i=exp,
                    w1_start=w1s, w1_end=w1e, w1_size=w1sz,
                    used=False, direction="short")
        for j in range(w1s_i, max(0, w1s_i-20)-1, -1):
            if closes[j] > opens[j]:
                zones.append({**base, "zone_lo": lows[j], "zone_hi": highs[j], "type": "ob"})
                break
        fw = [(fi,fl,fh) for fi,fl,fh in bear_fvgs if w1e_i<=fi<=w1s_i]
        fw.sort(key=lambda x: x[1])
        for fi,fl,fh in fw[:3]:
            zones.append({**base, "zone_lo": fl, "zone_hi": fh, "type": "fvg"})

    return zones


def map_zones_to_1h(zones: list[dict], h4_ts: np.ndarray, h1_ts: np.ndarray) -> list[dict]:
    """4H 존 인덱스를 1H 인덱스로 변환 (타임스탬프 기준)"""
    n4 = len(h4_ts)
    for z in zones:
        bos_ts   = h4_ts[min(z["bos_i"],    n4-1)]
        exp_ts   = h4_ts[min(z["expire_i"],  n4-1)]
        z["bos_1h_i"]    = int(np.searchsorted(h1_ts, bos_ts))
        z["expire_1h_i"] = int(np.searchsorted(h1_ts, exp_ts))
    return zones


# ── 듀얼 티어 백테스트 ────────────────────────────────────────────────────────
def run_dual_backtest(
    h1_highs, h1_lows, h1_closes,    # 1H 가격 배열
    h4_zones_raw: list[dict],         # 4H 존 (1H 인덱스로 변환됨)
    h1_zones_raw: list[dict],         # 1H 존
    # High Tier 파라미터
    h_rmin, h_rmax, h_tp, h_sl_buf, h_lev, h_pos, h_sig,
    # Low Tier 파라미터
    l_rmin, l_rmax, l_tp, l_sl_buf, l_lev, l_pos, l_sig,
    # 자본 배분
    alloc_high: float,
    initial_balance: float = 1000.0,
) -> dict:
    n = len(h1_closes)

    # 타입 필터
    h_type = {"ob"} if h_sig == "ob_only" else \
             {"fvg"} if h_sig == "fvg_only" else {"ob", "fvg"}
    l_type = {"ob"} if l_sig == "ob_only" else \
             {"fvg"} if l_sig == "fvg_only" else {"ob", "fvg"}

    # 존 복사 + 필터
    h4z = [dict(z) for z in h4_zones_raw if z["type"] in h_type]
    h1z = [dict(z) for z in h1_zones_raw if z["type"] in l_type]
    h4z.sort(key=lambda z: z["bos_1h_i"])
    h1z.sort(key=lambda z: z["bos_i"])

    # 자본 버킷
    bal_h = initial_balance * alloc_high
    bal_l = initial_balance * (1.0 - alloc_high)
    peak  = initial_balance
    max_dd = 0.0

    h_trades = h_wins = h_long = h_short = 0
    l_trades = l_wins = l_long = l_short = 0

    # High tier 포지션
    in_h = False; h_dir = "long"
    h_entry = h_tp_p = h_sl_p = h_amt = 0.0

    # Low tier 포지션
    in_l = False; l_dir = "long"
    l_entry = l_tp_p = l_sl_p = l_amt = 0.0

    h4_ptr = 0; h4_active: list[dict] = []
    h1_ptr = 0; h1_active: list[dict] = []

    def check_entry(d, highs_i, lows_i, closes_i, z, rmin, rmax):
        if d == "long":
            if lows_i > z["zone_hi"] or closes_i < z["zone_lo"]: return False
            if closes_i >= z["w1_end"]: return False
            retrace = (z["w1_end"] - closes_i) / z["w1_size"]
        else:
            if highs_i < z["zone_lo"] or closes_i > z["zone_hi"]: return False
            if closes_i <= z["w1_end"]: return False
            retrace = (closes_i - z["w1_end"]) / z["w1_size"]
        return rmin <= retrace <= rmax

    for i in range(n):
        hi = h1_highs[i]; lo = h1_lows[i]; cl = h1_closes[i]

        # 4H 존 활성화/만료
        while h4_ptr < len(h4z) and h4z[h4_ptr]["bos_1h_i"] <= i:
            h4_active.append(h4z[h4_ptr]); h4_ptr += 1
        h4_active = [z for z in h4_active if not z["used"] and i < z["expire_1h_i"]]

        # 1H 존 활성화/만료
        while h1_ptr < len(h1z) and h1z[h1_ptr]["bos_i"] <= i:
            h1_active.append(h1z[h1_ptr]); h1_ptr += 1
        h1_active = [z for z in h1_active if not z["used"] and i < z["expire_i"]]

        # ── High Tier ────────────────────────────────────────────────
        if not in_h and bal_h > 0:
            for z in h4_active:
                d = z["direction"]
                if not check_entry(d, hi, lo, cl, z, h_rmin, h_rmax): continue
                h_entry  = cl
                notional = bal_h * h_pos * h_lev
                h_amt    = notional / h_entry
                bal_h   -= notional * TAKER_FEE
                h_dir    = d; z["used"] = True
                if d == "long":
                    h_tp_p = h_entry + z["w1_size"] * h_tp
                    h_sl_p = z["zone_lo"] * (1 - h_sl_buf)
                else:
                    h_tp_p = h_entry - z["w1_size"] * h_tp
                    h_sl_p = z["zone_hi"] * (1 + h_sl_buf)
                in_h = True; break
        elif in_h:
            ep = None; won = False
            if h_dir == "long":
                if lo <= h_sl_p: ep = h_sl_p
                elif hi >= h_tp_p: ep = h_tp_p; won = True
            else:
                if hi >= h_sl_p: ep = h_sl_p
                elif lo <= h_tp_p: ep = h_tp_p; won = True
            if ep is not None:
                pnl  = (ep - h_entry) * h_amt * (1 if h_dir=="long" else -1)
                fee  = abs(ep * h_amt) * TAKER_FEE
                bal_h += pnl - fee
                h_trades += 1
                if won: h_wins += 1
                if h_dir=="long": h_long += 1
                else: h_short += 1
                in_h = False

        # ── Low Tier ─────────────────────────────────────────────────
        if not in_l and bal_l > 0:
            for z in h1_active:
                d = z["direction"]
                if not check_entry(d, hi, lo, cl, z, l_rmin, l_rmax): continue
                l_entry  = cl
                notional = bal_l * l_pos * l_lev
                l_amt    = notional / l_entry
                bal_l   -= notional * TAKER_FEE
                l_dir    = d; z["used"] = True
                if d == "long":
                    l_tp_p = l_entry + z["w1_size"] * l_tp
                    l_sl_p = z["zone_lo"] * (1 - l_sl_buf)
                else:
                    l_tp_p = l_entry - z["w1_size"] * l_tp
                    l_sl_p = z["zone_hi"] * (1 + l_sl_buf)
                in_l = True; break
        elif in_l:
            ep = None; won = False
            if l_dir == "long":
                if lo <= l_sl_p: ep = l_sl_p
                elif hi >= l_tp_p: ep = l_tp_p; won = True
            else:
                if hi >= l_sl_p: ep = l_sl_p
                elif lo <= l_tp_p: ep = l_tp_p; won = True
            if ep is not None:
                pnl  = (ep - l_entry) * l_amt * (1 if l_dir=="long" else -1)
                fee  = abs(ep * l_amt) * TAKER_FEE
                bal_l += pnl - fee
                l_trades += 1
                if won: l_wins += 1
                if l_dir=="long": l_long += 1
                else: l_short += 1
                in_l = False

        # 복합 자산 MDD
        total = max(bal_h, 0) + max(bal_l, 0)
        if total > peak: peak = total
        dd = (peak - total) / peak * 100
        if dd > max_dd: max_dd = dd

    total_bal  = max(bal_h, 0) + max(bal_l, 0)
    tot_trades = h_trades + l_trades
    tot_wins   = h_wins + l_wins
    ret        = (total_bal - initial_balance) / initial_balance * 100
    win_rate   = tot_wins / tot_trades * 100 if tot_trades > 0 else 0.0
    calmar     = ret / max_dd if max_dd > 0 else 0.0

    months = n / (30 * 24)   # 1H 봉 수 → 개월 수
    h_per_m = h_trades / months if months > 0 else 0
    l_per_m = l_trades / months if months > 0 else 0

    return dict(
        alloc_high=alloc_high,
        h_swing=0, l_swing=0,   # worker에서 채움
        h_rmin=h_rmin, h_rmax=h_rmax, h_tp=h_tp,
        h_sl_buf=h_sl_buf, h_lev=h_lev, h_pos=h_pos, h_sig=h_sig,
        l_rmin=l_rmin, l_rmax=l_rmax, l_tp=l_tp,
        l_sl_buf=l_sl_buf, l_lev=l_lev, l_pos=l_pos, l_sig=l_sig,
        h_trades=h_trades, h_wins=h_wins,
        l_trades=l_trades, l_wins=l_wins,
        h_long=h_long, h_short=h_short,
        l_long=l_long, l_short=l_short,
        h_per_month=round(h_per_m, 1),
        l_per_month=round(l_per_m, 1),
        win_rate=round(win_rate, 1),
        return_pct=round(ret, 2),
        max_drawdown=round(max_dd, 2),
        calmar=round(calmar, 2),
        final_balance=round(total_bal, 2),
    )


# ── Worker ─────────────────────────────────────────────────────────────────
def worker(coin: str) -> pd.DataFrame:
    print(f"[{coin.upper()}] 시작...", flush=True)
    try:
        df_1h, df_4h = load_data(coin)
    except Exception as e:
        print(f"[{coin.upper()}] 로드 실패: {e}", flush=True)
        return pd.DataFrame()

    # numpy 배열
    h1_o = df_1h["open"].to_numpy(float)
    h1_h = df_1h["high"].to_numpy(float)
    h1_l = df_1h["low"].to_numpy(float)
    h1_c = df_1h["close"].to_numpy(float)
    h4_o = df_4h["open"].to_numpy(float)
    h4_h = df_4h["high"].to_numpy(float)
    h4_l = df_4h["low"].to_numpy(float)
    h4_c = df_4h["close"].to_numpy(float)

    # 타임스탬프 int64 (nanoseconds)
    h1_ts = df_1h["timestamp"].astype(np.int64).values
    h4_ts = df_4h["timestamp"].astype(np.int64).values

    # 파라미터 조합
    inner_h = list(itertools.product(H_RETRACE, H_TP_EXT, H_SL_BUF, H_LEV, H_POS_RATIO, H_SIG))
    inner_l = list(itertools.product(L_RETRACE, L_TP_EXT, L_SL_BUF, L_LEV, L_POS_RATIO, L_SIG))

    results = []
    for h_sw in H_SWING:
        h4_zones_raw = build_zones(h4_o, h4_h, h4_l, h4_c, h_sw, ZONE_EXPIRY_4H)
        map_zones_to_1h(h4_zones_raw, h4_ts, h1_ts)
        h4_bos = len({z["bos_i"] for z in h4_zones_raw})

        for l_sw in L_SWING:
            h1_zones_raw = build_zones(h1_o, h1_h, h1_l, h1_c, l_sw, ZONE_EXPIRY_1H)
            h1_bos = len({z["bos_i"] for z in h1_zones_raw})

            for alloc in ALLOC_HIGH:
                for (h_rmin, h_rmax), h_tp, h_sl, h_lev, h_pos, h_sig in inner_h:
                    for (l_rmin, l_rmax), l_tp, l_sl, l_lev, l_pos, l_sig in inner_l:
                        r = run_dual_backtest(
                            h1_h, h1_l, h1_c,
                            h4_zones_raw, h1_zones_raw,
                            h_rmin, h_rmax, h_tp, h_sl, h_lev, h_pos, h_sig,
                            l_rmin, l_rmax, l_tp, l_sl, l_lev, l_pos, l_sig,
                            alloc,
                        )
                        r["h_swing"] = h_sw
                        r["l_swing"] = l_sw
                        r["coin"]    = coin
                        r["h4_bos"]  = h4_bos
                        r["h1_bos"]  = h1_bos
                        results.append(r)

    df_res = pd.DataFrame(results).sort_values("calmar", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"ict_dual_{coin}.csv")
    df_res.to_csv(out, index=False)

    best = df_res.iloc[0]
    n_pos = (df_res["return_pct"] > 0).sum()
    print(
        f"[{coin.upper()}] 완료 | {len(results):,}조합 | 수익>0: {n_pos}/{len(results)} "
        f"({n_pos/len(results)*100:.0f}%) | "
        f"Best Calmar={best['calmar']:.2f} "
        f"ret={best['return_pct']:+.1f}% MDD={best['max_drawdown']:.1f}% "
        f"H={best['h_trades']}건({best['h_per_month']:.1f}/월) "
        f"L={best['l_trades']}건({best['l_per_month']:.1f}/월) "
        f"alloc={best['alloc_high']}",
        flush=True,
    )
    return df_res


# ── 요약 출력 ───────────────────────────────────────────────────────────────
def print_summary(dfs: dict[str, pd.DataFrame]):
    combined = pd.concat(list(dfs.values()), ignore_index=True)

    print("\n" + "=" * 90)
    print("듀얼 티어 전략 분석 요약")
    print("=" * 90)

    # 1) Calmar 상위 10
    print("\n▶ Calmar 상위 10개 조합")
    cols = ["coin", "alloc_high", "h_swing", "l_swing",
            "h_sig", "l_sig", "h_lev", "l_lev",
            "h_trades", "h_per_month", "l_trades", "l_per_month",
            "win_rate", "return_pct", "max_drawdown", "calmar"]
    print(combined.nlargest(10, "calmar")[cols].to_string(index=False))

    # 2) 자본 배분별 성과
    print("\n▶ 자본 배분(alloc_high)별 중앙값 성과")
    grp = (combined.groupby("alloc_high")
           .agg(h_per_m=("h_per_month","median"),
                l_per_m=("l_per_month","median"),
                win_rate=("win_rate","median"),
                return_pct=("return_pct","median"),
                max_dd=("max_drawdown","median"),
                calmar=("calmar","median"))
           .round(2))
    print(grp.to_string())

    # 3) 신호 조합별 성과
    print("\n▶ 신호 조합(H/L sig)별 중앙값 Calmar")
    sig_grp = (combined.groupby(["h_sig","l_sig"])
               .agg(return_pct=("return_pct","median"),
                    calmar=("calmar","median"),
                    win_rate=("win_rate","median"))
               .round(2).sort_values("calmar", ascending=False))
    print(sig_grp.to_string())

    # 4) 월 거래 빈도 필터 (H티어 1~4건/월 범위)
    print("\n▶ High Tier 월 1~4건 필터 후 Calmar 상위 10")
    filtered = combined[(combined["h_per_month"] >= 1.0) &
                        (combined["h_per_month"] <= 4.0)]
    if len(filtered) > 0:
        print(filtered.nlargest(10, "calmar")[cols].to_string(index=False))
    else:
        print("  (조건 맞는 조합 없음)")

    # 5) 코인별 최적 파라미터
    print("\n▶ 코인별 최고 Calmar (월 H거래 1~4건 필터)")
    for coin in COINS:
        sub = filtered[filtered["coin"] == coin] if len(filtered) > 0 else combined[combined["coin"]==coin]
        if sub.empty: continue
        b = sub.iloc[0]
        print(f"  {coin.upper()}: Calmar={b['calmar']:.2f} ret={b['return_pct']:+.1f}% "
              f"MDD={b['max_drawdown']:.1f}% | "
              f"High={b['h_trades']}건({b['h_per_month']:.1f}/월,alloc={b['alloc_high']}) "
              f"Low={b['l_trades']}건({b['l_per_month']:.1f}/월) | "
              f"H:{b['h_sig']} sw={b['h_swing']} ret={b['h_rmin']}~{b['h_rmax']} tp={b['h_tp']} lev={b['h_lev']}x pos={b['h_pos']} | "
              f"L:{b['l_sig']} sw={b['l_swing']} ret={b['l_rmin']}~{b['l_rmax']} tp={b['l_tp']} lev={b['l_lev']}x pos={b['l_pos']}")

    # 6) 단독 vs 듀얼 효과 (H only vs L only vs combined)
    print("\n▶ 전략 분리 효과 분석 (H only / L only 근사)")
    for coin in COINS:
        sub = combined[combined["coin"]==coin]
        if sub.empty: continue
        h_only = sub.sort_values("h_trades", ascending=False).head(1).iloc[0]
        l_only = sub.sort_values("l_trades", ascending=False).head(1).iloc[0]
        best   = sub.iloc[0]
        print(f"  {coin.upper()}: H-heavy(alloc=0.7) best ret={sub[sub['alloc_high']==0.7].iloc[0]['return_pct']:+.1f}% | "
              f"L-heavy(alloc=0.3) best ret={sub[sub['alloc_high']==0.3].iloc[0]['return_pct']:+.1f}% | "
              f"Balanced(0.5) best ret={sub[sub['alloc_high']==0.5].iloc[0]['return_pct']:+.1f}%")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    h_inner = (len(H_RETRACE) * len(H_TP_EXT) * len(H_SL_BUF) *
               len(H_LEV) * len(H_POS_RATIO) * len(H_SIG))
    l_inner = (len(L_RETRACE) * len(L_TP_EXT) * len(L_SL_BUF) *
               len(L_LEV) * len(L_POS_RATIO) * len(L_SIG))
    combos_per_coin = len(H_SWING) * len(L_SWING) * len(ALLOC_HIGH) * h_inner * l_inner

    print("ICT + Elliott Wave 듀얼 티어 전략 연구")
    print(f"코인: {[c.upper() for c in COINS]}")
    print(f"High Tier 파라미터: {h_inner}개 × Swing {H_SWING}")
    print(f"Low  Tier 파라미터: {l_inner}개 × Swing {L_SWING}")
    print(f"자본 배분 옵션: {ALLOC_HIGH}")
    print(f"코인당 조합: {combos_per_coin:,}개 | 총: {combos_per_coin * len(COINS):,}개\n")

    results_map: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=len(COINS)) as executor:
        futures = {executor.submit(worker, c): c for c in COINS}
        for fut in as_completed(futures):
            coin = futures[fut]
            try:
                df = fut.result()
                if not df.empty:
                    results_map[coin] = df
            except Exception as e:
                print(f"  [{coin}] 오류: {e}")

    print_summary(results_map)


if __name__ == "__main__":
    main()
