"""
코인별 최적 파라미터 탐색.

2017~2026 전체 히스토리 기반 트렌드 홀딩 전략 파라미터 최적화.
각 코인별로 B&H를 가장 많이 이기는 (tf, signal, lev, mode) 조합 탐색.

입력: data/market/{coin}_1h_full.parquet
출력: analysis/output/param_search_{coin}.csv
      analysis/output/param_search_summary.txt
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR   = Path(__file__).parent.parent / 'data' / 'market'
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 기간 정의 (데이터에서 B&H 동적 계산) ────────────────────────────
# coin별 데이터 시작일이 달라서 가능한 기간만 사용
PERIODS_ALL = {
    'bull_2017h2': ('2017-08-01', '2018-01-01'),
    'bear_2018':   ('2018-01-01', '2019-01-01'),
    'range_2019':  ('2019-01-01', '2020-01-01'),
    'bull_2020':   ('2020-01-01', '2021-01-01'),
    'bull_2021':   ('2021-01-01', '2022-01-01'),
    'bear_2022':   ('2022-01-01', '2023-01-01'),
    'bull_2023':   ('2023-01-01', '2024-01-01'),
    'range_2024h1':('2024-01-01', '2024-07-01'),
    'bull_2024h2': ('2024-07-01', '2025-01-01'),
    'bear_2025q1': ('2025-01-01', '2025-04-01'),
    'bull_2025q2': ('2025-04-01', '2025-06-01'),
    'bull_2025q3': ('2025-06-01', '2025-09-15'),
    'bear_2025q4': ('2025-09-15', '2025-12-01'),
    'bear_2026q1': ('2025-12-01', '2026-03-01'),
}

PERIOD_LABEL = {
    'bull_2017h2': '상승(17H2)',
    'bear_2018':   '하락(2018)',
    'range_2019':  '횡보(2019)',
    'bull_2020':   '상승(2020)',
    'bull_2021':   '상승(2021)',
    'bear_2022':   '하락(2022)',
    'bull_2023':   '회복(2023)',
    'range_2024h1':'횡보(24H1)',
    'bull_2024h2': '불장(24H2)',
    'bear_2025q1': '하락(25Q1)',
    'bull_2025q2': '반등(25Q2)',
    'bull_2025q3': '상승(25Q3)',
    'bear_2025q4': '하락(25Q4)',
    'bear_2026q1': '하락(26Q1)',
}

COINS      = ['btc', 'eth', 'xrp', 'sol']
TIMEFRAMES = ['1h', '4h', '1d']
LEVERAGES  = [1, 2, 3]
FEE_RATE   = 0.0005
SLIPPAGE   = 0.0003

RESAMPLE_MAP = {'1h': '1h', '4h': '4h', '1d': '1D'}


# ── 리샘플 ──────────────────────────────────────────────────────────
def resample_df(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = RESAMPLE_MAP[tf]
    df2 = df.set_index('timestamp').resample(rule).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).dropna(subset=['open'])
    return df2


# ── EMA ────────────────────────────────────────────────────────────
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    n = len(df)

    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))

    alpha = 2 / (period + 1)
    atr = np.empty(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]

    mid = (h + l) / 2
    bu = mid + mult * atr
    bl = mid - mult * atr
    fu = np.empty(n)
    fl = np.empty(n)
    fu[0], fl[0] = bu[0], bl[0]
    direction = np.ones(n, dtype=bool)

    for i in range(1, n):
        fl[i] = bl[i] if bl[i] > fl[i-1] or c[i-1] < fl[i-1] else fl[i-1]
        fu[i] = bu[i] if bu[i] < fu[i-1] or c[i-1] > fu[i-1] else fu[i-1]
        if c[i] > fu[i-1]:
            direction[i] = True
        elif c[i] < fl[i-1]:
            direction[i] = False
        else:
            direction[i] = direction[i-1]

    return pd.Series(direction, index=df.index)


def compute_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    c = df['close']
    sigs: dict[str, pd.Series] = {}

    # EMA 크로스
    for f, s in [(10, 30), (20, 50), (20, 100), (50, 200), (5, 20), (10, 50)]:
        sigs[f'ema{f}/{s}'] = _ema(c, f) > _ema(c, s)

    # Price vs EMA
    for p in [50, 100, 200]:
        sigs[f'c>ema{p}'] = c > _ema(c, p)

    # MACD
    for f, s, sg in [(12, 26, 9), (26, 52, 18)]:
        ml = _ema(c, f) - _ema(c, s)
        sigs[f'macd{f}/{s}'] = (ml - _ema(ml, sg)) > 0

    # Supertrend
    for m in [2.0, 3.0]:
        sigs[f'st{m}'] = _supertrend(df, 10, m)

    return sigs


# ── 백테스트 ─────────────────────────────────────────────────────────
def backtest(df: pd.DataFrame, sig: pd.Series, lev: int, with_short: bool) -> dict:
    opens  = df['open'].values.astype(float)
    closes = df['close'].values.astype(float)
    sv     = sig.values.astype(bool)
    n      = len(df)

    trades = []
    pos, ep = None, 0.0

    for i in range(1, n):
        prev = sv[i-1]
        if pos is None:
            if prev:
                ep, pos = opens[i] * (1 + SLIPPAGE), 'L'
            elif with_short:
                ep, pos = opens[i] * (1 - SLIPPAGE), 'S'
        elif pos == 'L' and not prev:
            xp = opens[i] * (1 - SLIPPAGE)
            trades.append((xp - ep) / ep)
            pos = None
            if with_short:
                ep, pos = xp, 'S'
        elif pos == 'S' and prev:
            xp = opens[i] * (1 + SLIPPAGE)
            trades.append((ep - xp) / ep)
            pos = None
            ep, pos = opens[i] * (1 + SLIPPAGE), 'L'

    if pos is not None:
        xp = closes[-1]
        pnl = (xp - ep) / ep if pos == 'L' else (ep - xp) / ep
        trades.append(pnl)

    if not trades:
        return dict(n=0, ret=0.0, win=0.0, dd=0.0)

    pnls = np.array([(p - FEE_RATE * 2) * lev * 100 for p in trades])
    cum  = np.cumsum(pnls)
    rm   = np.maximum.accumulate(cum)
    dd   = float(np.max(rm - cum)) if len(cum) > 0 else 0.0

    return dict(
        n=len(pnls),
        ret=round(float(pnls.sum()), 2),
        win=round(float((pnls > 0).mean()), 3),
        dd=round(dd, 2),
    )


# ── 메인 ──────────────────────────────────────────────────────────
all_summary = {}

for coin in COINS:
    parquet = DATA_DIR / f'{coin}_1h_full.parquet'
    if not parquet.exists():
        print(f'{coin.upper()}: 데이터 없음 — 건너뜀')
        continue

    print(f'\n{"="*60}')
    print(f'▶ {coin.upper()} 파라미터 탐색 시작')
    df_1h = pd.read_parquet(parquet)
    if 'timestamp' not in df_1h.columns:
        df_1h = df_1h.reset_index()
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], utc=True)
    print(f'  데이터: {len(df_1h):,}봉  ({df_1h["timestamp"].min().date()} ~ {df_1h["timestamp"].max().date()})')

    rows = []

    for tf in TIMEFRAMES:
        print(f'  [{tf}] 리샘플...', end=' ', flush=True)
        df_tf = resample_df(df_1h, tf)
        sigs_full = compute_signals(df_tf)
        print(f'{len(df_tf):,}봉, {len(sigs_full)}신호')

        for pkey, (start, end) in PERIODS_ALL.items():
            s_ts = pd.Timestamp(start, tz='UTC')
            e_ts = pd.Timestamp(end, tz='UTC')
            mask = (df_tf.index >= s_ts) & (df_tf.index < e_ts)
            if mask.sum() < 20:
                continue

            df_p = df_tf[mask]
            # B&H 수익률 동적 계산
            bh_ret = (df_p['close'].iloc[-1] / df_p['open'].iloc[0] - 1) * 100

            for sig_name, sig_full in sigs_full.items():
                sig_p = sig_full[mask]
                for lev in LEVERAGES:
                    for ws in [False, True]:
                        res  = backtest(df_p, sig_p, lev, ws)
                        beat = bool(res['ret'] > bh_ret)
                        rows.append({
                            'coin':     coin.upper(),
                            'tf':       tf,
                            'signal':   sig_name,
                            'lev':      lev,
                            'mode':     'L+S' if ws else 'L',
                            'period':   pkey,
                            'bh_ret':   round(bh_ret, 2),
                            'ret':      res['ret'],
                            'beat_bh':  beat,
                            'n_trades': res['n'],
                            'win_rate': res['win'],
                            'max_dd':   res['dd'],
                        })

    df_all = pd.DataFrame(rows)
    df_all.to_csv(OUTPUT_DIR / f'param_search_{coin}_all.csv', index=False)

    # 집계
    sum_rows = []
    for (tf, sig, lev, mode), g in df_all.groupby(['tf', 'signal', 'lev', 'mode']):
        valid    = g.dropna(subset=['beat_bh'])
        beat_cnt = int(valid['beat_bh'].sum())
        total_p  = len(valid)
        avg_ret  = round(float(g['ret'].mean()), 2)
        min_ret  = round(float(g['ret'].min()), 2)
        max_dd   = round(float(g['max_dd'].mean()), 2)
        avg_n    = round(float(g['n_trades'].mean()), 1)
        avg_bh   = round(float(g['bh_ret'].mean()), 2)
        beat_pct = round(beat_cnt / total_p * 100, 1) if total_p > 0 else 0.0

        period_rets = {row['period']: row['ret'] for _, row in g.iterrows()}
        period_bh   = {row['period']: row['bh_ret'] for _, row in g.iterrows()}

        sum_rows.append({
            'coin': coin.upper(), 'tf': tf, 'signal': sig,
            'lev': lev, 'mode': mode,
            'beat_cnt': beat_cnt, 'total_periods': total_p,
            'beat_pct': beat_pct,
            'avg_ret': avg_ret, 'min_ret': min_ret,
            'max_dd_avg': max_dd, 'avg_trades': avg_n, 'avg_bh': avg_bh,
            **{f'{p}_ret': period_rets.get(p) for p in PERIODS_ALL},
            **{f'{p}_bh':  period_bh.get(p)   for p in PERIODS_ALL},
        })

    df_sum = pd.DataFrame(sum_rows).sort_values(
        ['beat_cnt', 'avg_ret'], ascending=False
    )
    df_sum.to_csv(OUTPUT_DIR / f'param_search_{coin}.csv', index=False)
    all_summary[coin.upper()] = df_sum
    print(f'  저장: param_search_{coin}.csv ({len(df_sum)}개 조합)')


# ── 출력 ──────────────────────────────────────────────────────────
W = 74
lines = []

def p(s=''):
    print(s)
    lines.append(s)

p('=' * W)
p('코인별 최적 파라미터 — 트렌드 홀딩 전략 (2017~2026 전체 히스토리)')
p('(B&H를 가장 많이 이기는 전략, 신호 반전까지 포지션 유지)')
p('=' * W)

for coin_up, df_sum in all_summary.items():
    p(f'\n▶ {coin_up}  (총 {df_sum["total_periods"].iloc[0]}기간 분석)')
    p('-' * W)

    # TOP5 by beat_cnt
    top = df_sum.head(10)

    for rank, (_, r) in enumerate(top.iterrows(), 1):
        star = ' ★' if r['min_ret'] >= 0 else ''
        p(f"\n  [{rank}] {r['tf']} | {r['signal']} | {r['lev']}x {r['mode']}{star}")
        p(f"      B&H초과: {r['beat_cnt']}/{r['total_periods']}기간 ({r['beat_pct']:.0f}%)  "
          f"평균수익: {r['avg_ret']:+.1f}%  최소: {r['min_ret']:+.1f}%  "
          f"평균MDD: {r['max_dd_avg']:.1f}%  평균거래: {r['avg_trades']:.0f}건")

        for pkey, plabel in PERIOD_LABEL.items():
            ret = r.get(f'{pkey}_ret')
            bh  = r.get(f'{pkey}_bh')
            if ret is None or bh is None or pd.isna(ret):
                continue
            flag = '✓' if ret > bh else '✗'
            p(f"      {flag} {plabel:<12} 전략 {ret:>+7.1f}%  B&H {bh:>+7.1f}%  ({ret-bh:>+.1f}%p)")

    p()

    # 레버리지 1x 필터
    lev1 = df_sum[(df_sum['lev'] == 1)].head(5)
    p(f'  ── 레버리지 1x TOP5 ──')
    for _, r in lev1.iterrows():
        star = ' ★' if r['min_ret'] >= 0 else ''
        p(f"  {r['tf']} {r['signal']} {r['mode']}{star}  |  "
          f"B&H초과 {r['beat_cnt']}/{r['total_periods']} ({r['beat_pct']:.0f}%)  "
          f"avg {r['avg_ret']:+.1f}%  min {r['min_ret']:+.1f}%")

p()
p('=' * W)
p('★ = 모든 기간 손실 없음 (min_ret >= 0)')

summary_txt = OUTPUT_DIR / 'param_search_summary.txt'
summary_txt.write_text('\n'.join(lines), encoding='utf-8')
print(f'\n요약 저장: {summary_txt}')
