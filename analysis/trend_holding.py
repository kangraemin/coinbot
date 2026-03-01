"""
트렌드 홀딩 전략 백테스트.

기존 전략들이 B&H를 못 이긴 이유: 고정 TP/SL로 강세장 대부분 놓침.
이 스크립트는 신호 반전까지 포지션을 유지한다.
  - 롱 신호: 다음봉 시가 진입, 신호 반전 시까지 홀딩
  - 숏 모드: 롱 신호 없을 때 숏 진입
  - 타임프레임: 1h / 2h / 4h / 1d
  - 신호: EMA 크로스, Price>EMA, MACD, Supertrend
  - 레버리지: 1x / 2x / 3x
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.data_loader import load_1m, resample
from analysis.config import OUTPUT_DIR

# ── 분석 기간 ──────────────────────────────────────────────
PERIODS = {
    'bear_2026q1': ('2025-12-01', '2026-03-01'),
    'bear_2025q4': ('2025-09-15', '2025-12-01'),
    'bull_2025q3': ('2025-06-01', '2025-09-15'),
    'bull_2025q2': ('2025-04-01', '2025-06-01'),
    'bear_2025q1': ('2025-01-01', '2025-04-01'),
    'bull_2024h2': ('2024-07-01', '2025-01-01'),
    'bear_2024h1': ('2024-01-01', '2024-07-01'),
    'bull_2023':   ('2023-01-01', '2024-01-01'),
}

PERIOD_LABEL = {
    'bear_2026q1': '하락(26Q1)',
    'bear_2025q4': '하락(25Q4)',
    'bull_2025q3': '상승(25Q3)',
    'bull_2025q2': '반등(25Q2)',
    'bear_2025q1': '하락(25Q1)',
    'bull_2024h2': '불장(24H2)',
    'bear_2024h1': '횡보(24H1)',
    'bull_2023':   '회복(2023)',
}

BUY_HOLD = {
    'bear_2026q1': {'BTC': -26.8, 'ETH': -32.8, 'XRP': -39.4, 'SOL': -41.7},
    'bear_2025q4': {'BTC': -23.3, 'ETH': -28.4, 'XRP': -28.9, 'SOL': -44.5},
    'bull_2025q3': {'BTC':  +6.2, 'ETH': +66.3, 'XRP': +39.3, 'SOL': +53.1},
    'bull_2025q2': {'BTC': +26.8, 'ETH': +38.9, 'XRP':  +4.2, 'SOL': +25.7},
    'bear_2025q1': {'BTC': -11.8, 'ETH': -45.4, 'XRP':  +0.2, 'SOL': -34.4},
    'bull_2024h2': {'BTC': +47.4, 'ETH':  -2.4, 'XRP':+338.0, 'SOL': +29.1},
    'bear_2024h1': {'BTC': +43.5, 'ETH': +47.0, 'XRP': -22.7, 'SOL': +44.1},
    'bull_2023':   {'BTC':+155.2, 'ETH': +92.1, 'XRP': +81.6, 'SOL':+920.6},
}

COINS      = ['btc', 'eth', 'xrp', 'sol']
TIMEFRAMES = ['1h', '2h', '4h', '1d']
LEVERAGES  = [1, 2, 3]
FEE_RATE   = 0.0005
SLIPPAGE   = 0.0003
LOOKBACK   = 1500   # 전체 기간 로드 (~2021.10 ~)


# ── 인디케이터 ─────────────────────────────────────────────
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """True = 상승추세(롱), False = 하강추세(숏)"""
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


def compute_all_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    """전체 트렌드 신호. True=롱, False=숏/관망"""
    c = df['close']
    sigs: dict[str, pd.Series] = {}

    # EMA 크로스 (fast > slow = long)
    for f, s in [(20, 50), (50, 200), (10, 30), (20, 100), (5, 20)]:
        sigs[f'ema{f}/{s}'] = _ema(c, f) > _ema(c, s)

    # Price vs EMA
    for p in [50, 100, 200]:
        sigs[f'c>ema{p}'] = c > _ema(c, p)

    # MACD histogram > 0
    for f, s, sg in [(12, 26, 9), (26, 52, 18)]:
        ml = _ema(c, f) - _ema(c, s)
        sigs[f'macd{f}/{s}'] = (ml - _ema(ml, sg)) > 0

    # Supertrend
    for m in [2.0, 3.0, 4.0]:
        sigs[f'st{m}'] = _supertrend(df, 10, m)

    return sigs


# ── 백테스트 ───────────────────────────────────────────────
def _backtest(df: pd.DataFrame, sig: pd.Series, lev: int, with_short: bool) -> dict:
    """
    신호 반전 시 다음봉 시가에 포지션 전환.
    with_short=True: 롱 신호 없을 때 숏 진입.
    with_short=False: 롱 신호 없을 때 현금 보유.
    """
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

    # 미청산 포지션 → 마지막 종가로 청산
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


# ── 메인 ──────────────────────────────────────────────────
print('데이터 로드 중 (전체 기간)...')
rows = []

for coin in COINS:
    print(f'\n▶ {coin.upper()} 로드...')
    df_1m = load_1m(coin, lookback_days=LOOKBACK)

    for tf in TIMEFRAMES:
        print(f'  [{tf}] 리샘플 + 신호 계산...', end=' ', flush=True)
        df_tf = resample(df_1m, tf)

        if 'timestamp' in df_tf.columns:
            df_tf = df_tf.set_index('timestamp')
        if df_tf.index.tz is None:
            df_tf.index = df_tf.index.tz_localize('UTC')

        sigs_full = compute_all_signals(df_tf)
        print(f'{len(df_tf):,}봉, {len(sigs_full)}신호')

        for pkey, (start, end) in PERIODS.items():
            s_ts = pd.Timestamp(start, tz='UTC')
            e_ts = pd.Timestamp(end, tz='UTC')
            mask = (df_tf.index >= s_ts) & (df_tf.index < e_ts)

            if mask.sum() < 20:
                continue

            df_p   = df_tf[mask]
            bh_ret = BUY_HOLD.get(pkey, {}).get(coin.upper())

            for sig_name, sig_full in sigs_full.items():
                sig_p = sig_full[mask]

                for lev in LEVERAGES:
                    for ws in [False, True]:
                        res  = _backtest(df_p, sig_p, lev, ws)
                        beat = bool(res['ret'] > bh_ret) if bh_ret is not None else None
                        rows.append({
                            'coin':     coin.upper(),
                            'tf':       tf,
                            'signal':   sig_name,
                            'lev':      lev,
                            'mode':     'L+S' if ws else 'L',
                            'period':   pkey,
                            'ret':      res['ret'],
                            'bh_ret':   bh_ret,
                            'beat_bh':  beat,
                            'n_trades': res['n'],
                            'win_rate': res['win'],
                            'max_dd':   res['dd'],
                        })

print('\n결과 집계 중...')

df_all = pd.DataFrame(rows)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df_all.to_csv(OUTPUT_DIR / 'trend_holding_all.csv', index=False)

# ── 전략별 집계 ────────────────────────────────────────────
summary_rows = []

for (coin, tf, sig, lev, mode), g in df_all.groupby(['coin', 'tf', 'signal', 'lev', 'mode']):
    valid = g.dropna(subset=['beat_bh'])
    if len(valid) == 0:
        continue

    beat_cnt = int(valid['beat_bh'].sum())
    total_p  = len(valid)
    avg_ret  = round(float(g['ret'].mean()), 2)
    min_ret  = round(float(g['ret'].min()), 2)
    max_dd   = round(float(g['max_dd'].mean()), 2)
    avg_n    = round(float(g['n_trades'].mean()), 1)

    period_rets = {row['period']: row['ret'] for _, row in g.iterrows()}

    summary_rows.append({
        'coin': coin, 'tf': tf, 'signal': sig,
        'lev': lev, 'mode': mode,
        'beat_cnt': beat_cnt, 'total_periods': total_p,
        'avg_ret': avg_ret, 'min_ret': min_ret,
        'max_dd_avg': max_dd, 'avg_trades': avg_n,
        **{f'{p}_ret': period_rets.get(p) for p in PERIODS},
    })

df_sum = pd.DataFrame(summary_rows).sort_values(
    ['beat_cnt', 'avg_ret'], ascending=False
)
df_sum.to_csv(OUTPUT_DIR / 'trend_holding.csv', index=False)
print(f'저장: {OUTPUT_DIR}/trend_holding.csv\n')

# ── 출력 ──────────────────────────────────────────────────
W = 72
print('=' * W)
print('트렌드 홀딩 전략 — B&H 초과 기간 TOP')
print('(신호 반전까지 포지션 유지, 고정 TP/SL 없음)')
print('=' * W)

top = df_sum[df_sum['beat_cnt'] >= 4].head(30)
if top.empty:
    print('(4기간 이상 B&H 초과 전략 없음. beat_cnt >= 3으로 재시도)')
    top = df_sum[df_sum['beat_cnt'] >= 3].head(30)

for _, r in top.iterrows():
    label = f"{r['coin']} {r['tf']} | {r['signal']} | {r['lev']}x {r['mode']}"
    marker = ' ★' if r['min_ret'] >= 0 else ''
    print(f"\n  {label}{marker}")
    print(f"  B&H 이긴 기간: {r['beat_cnt']}/{r['total_periods']}  "
          f"평균수익: {r['avg_ret']:+.1f}%  최소수익: {r['min_ret']:+.1f}%  "
          f"평균거래: {r['avg_trades']:.0f}건  평균MDD: {r['max_dd_avg']:.1f}%")

    for pkey, plabel in PERIOD_LABEL.items():
        ret = r.get(f'{pkey}_ret')
        bh  = BUY_HOLD.get(pkey, {}).get(r['coin'])
        if ret is None or bh is None:
            continue
        flag = '✓' if ret > bh else '✗'
        print(f"    {flag} {plabel:<12} 전략 {ret:>+7.1f}%  B&H {bh:>+7.1f}%  ({ret-bh:>+.1f}%p)")

print()
print('=' * W)
print('레버리지 1x, 6기간 중 ≥5기간 B&H 초과')
print('=' * W)
top1x = df_sum[(df_sum['lev'] == 1) & (df_sum['beat_cnt'] >= 5)].head(20)
if top1x.empty:
    top1x = df_sum[(df_sum['lev'] == 1) & (df_sum['beat_cnt'] >= 4)].head(20)
for _, r in top1x.iterrows():
    marker = ' ★ (손실기간 없음)' if r['min_ret'] >= 0 else ''
    print(f"  {r['coin']} {r['tf']} {r['signal']} {r['mode']} | "
          f"이긴기간 {r['beat_cnt']}/{r['total_periods']} | "
          f"avg {r['avg_ret']:+.1f}% | min {r['min_ret']:+.1f}%{marker}")

print()
print('★ = 모든 기간 수익 (손실 없음)')
