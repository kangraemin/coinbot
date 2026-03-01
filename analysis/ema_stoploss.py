"""
EMA 트렌드 + 고정 손절 백테스트.

기존 EMA 트렌드 전략에 진입가 기준 -N% 손절 추가.
신호 반전 OR 손절 중 먼저 오는 것으로 청산.

손절 비율: -10%, -15%, -20%, -25%, -30% (없음 포함 비교)
레버리지: 2x (격리마진 청산 -47.5%)

실행: python3 analysis/ema_stoploss.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR   = Path(__file__).parent.parent / 'data' / 'market'
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEE_RATE = 0.0005
SLIPPAGE = 0.0003
LEV      = 2
LIQ_MOVE = 0.475   # 2x 청산

PERIODS = {
    'bull_2017h2':  ('2017-08-01', '2018-01-01'),
    'bear_2018':    ('2018-01-01', '2019-01-01'),
    'range_2019':   ('2019-01-01', '2020-01-01'),
    'bull_2020':    ('2020-01-01', '2021-01-01'),
    'bull_2021':    ('2021-01-01', '2022-01-01'),
    'bear_2022':    ('2022-01-01', '2023-01-01'),
    'bull_2023':    ('2023-01-01', '2024-01-01'),
    'range_2024h1': ('2024-01-01', '2024-07-01'),
    'bull_2024h2':  ('2024-07-01', '2025-01-01'),
    'bear_2025q1':  ('2025-01-01', '2025-04-01'),
    'bull_2025q2':  ('2025-04-01', '2025-06-01'),
    'bull_2025q3':  ('2025-06-01', '2025-09-15'),
    'bear_2025q4':  ('2025-09-15', '2025-12-01'),
    'bear_2026q1':  ('2025-12-01', '2026-03-01'),
}

PERIOD_LABEL = {
    'bull_2017h2':  '상승(17H2)',
    'bear_2018':    '하락(2018)',
    'range_2019':   '횡보(2019)',
    'bull_2020':    '상승(2020)',
    'bull_2021':    '상승(2021)',
    'bear_2022':    '하락(2022)',
    'bull_2023':    '회복(2023)',
    'range_2024h1': '횡보(24H1)',
    'bull_2024h2':  '불장(24H2)',
    'bear_2025q1':  '하락(25Q1)',
    'bull_2025q2':  '반등(25Q2)',
    'bull_2025q3':  '상승(25Q3)',
    'bear_2025q4':  '하락(25Q4)',
    'bear_2026q1':  '하락(26Q1)',
}

STOP_LOSSES = [None, 0.10, 0.15, 0.20, 0.25, 0.30]  # None = 손절 없음


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def load_4h(coin: str) -> pd.DataFrame:
    path = DATA_DIR / f'{coin}_1h_full.parquet'
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = (df.drop_duplicates('timestamp')
            .sort_values('timestamp')
            .set_index('timestamp'))
    return df.resample('4h').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).dropna(subset=['open'])


def ema_signal(df: pd.DataFrame, coin: str) -> pd.Series:
    c = df['close']
    if coin == 'btc':
        return _ema(c, 50) > _ema(c, 200)
    elif coin == 'eth':
        return c > _ema(c, 200)
    else:  # xrp
        return _ema(c, 20) > _ema(c, 100)


def backtest(df: pd.DataFrame, sig: pd.Series, sl: float | None) -> dict:
    """
    sl: 손절 비율 (예: 0.20 = 진입가 대비 -20% 하락 시 청산)
        None = 손절 없음 (기존 전략)
    """
    opens  = df['open'].values.astype(float)
    highs  = df['high'].values.astype(float)
    lows   = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)
    sv     = sig.values.astype(bool)
    n      = len(df)

    trades   = []
    pos      = None
    ep       = 0.0
    sl_price = 0.0
    liq      = False

    for i in range(1, n):
        if liq:
            break

        if pos == 'L':
            # 1순위: 강제청산 체크 (손절이 커버 못하는 갭다운 대비)
            if lows[i] <= ep * (1 - LIQ_MOVE):
                trades.append(-1.0)
                liq = True
                break

            # 2순위: 손절 체크 (캔들 저가 기준)
            if sl is not None and lows[i] <= sl_price:
                trades.append((sl_price - ep) / ep)
                pos = None
                continue

            # 3순위: 신호 반전 → 다음봉 시가 청산
            if not sv[i - 1]:
                xp = opens[i] * (1 - SLIPPAGE)
                trades.append((xp - ep) / ep)
                pos = None

        elif pos is None and sv[i - 1]:
            ep  = opens[i] * (1 + SLIPPAGE)
            pos = 'L'
            sl_price = ep * (1 - sl) if sl is not None else 0.0

    if pos is not None and not liq:
        trades.append((closes[-1] - ep) / ep)

    if not trades:
        return dict(n=0, ret=0.0, win=0.0, dd=0.0, liq=False)

    pnls    = np.array([(p - FEE_RATE * 2) * LEV * 100 for p in trades])
    cum     = np.cumsum(pnls)
    cum     = np.maximum(cum, -100.0)
    rm      = np.maximum.accumulate(cum)
    dd      = float(np.max(rm - cum))
    wiped   = liq or (len(pnls) > 0 and pnls[0] <= -199)

    return dict(
        n=len(pnls),
        ret=round(float(cum[-1]), 2),
        win=round(float((pnls > 0).mean()), 3),
        dd=round(dd, 2),
        liq=wiped,
    )


EMA_LABEL = {'btc': 'EMA50/200', 'eth': 'c>EMA200', 'xrp': 'EMA20/100'}
COINS     = ['btc', 'eth', 'xrp']

rows = []

for coin in COINS:
    print(f'▶ {coin.upper()} 로드...')
    df4   = load_4h(coin)
    sig_a = ema_signal(df4, coin)

    for pkey, (start, end) in PERIODS.items():
        s_ts = pd.Timestamp(start, tz='UTC')
        e_ts = pd.Timestamp(end, tz='UTC')
        mask = (df4.index >= s_ts) & (df4.index < e_ts)
        if mask.sum() < 20:
            continue

        df_p  = df4[mask]
        sig_p = sig_a[mask]
        bh    = round((df_p['close'].iloc[-1] / df_p['open'].iloc[0] - 1) * 100, 2)

        for sl in STOP_LOSSES:
            res = backtest(df_p, sig_p, sl)
            rows.append({
                'coin':   coin.upper(),
                'period': pkey,
                'label':  PERIOD_LABEL[pkey],
                'sl':     f'-{int(sl*100)}%' if sl else '없음',
                'sl_val': sl if sl else 999,
                'bh':     bh,
                'ret':    res['ret'],
                'n':      res['n'],
                'win':    res['win'],
                'dd':     res['dd'],
                'liq':    res['liq'],
                'beat':   int(res['ret'] > bh),
            })

df_all = pd.DataFrame(rows)
df_all.to_csv(OUTPUT_DIR / 'ema_stoploss.csv', index=False)

# ── 출력: 코인별 × 손절비율별 요약 ────────────────────────────────
W = 80
SL_LABELS = ['없음', '-10%', '-15%', '-20%', '-25%', '-30%']

for coin in COINS:
    sub = df_all[df_all['coin'] == coin.upper()]
    print(f'\n{"="*W}')
    print(f' {coin.upper()} EMA 트렌드 2x — 손절 비율별 비교 ({EMA_LABEL[coin]})')
    print(f'{"="*W}')
    print(f'{"손절":>5}  {"B&H초과":>8}  {"평균수익":>8}  {"최소수익":>8}  '
          f'{"평균MDD":>7}  {"평균거래":>7}  {"청산횟수":>7}')
    print(f'{"-"*W}')

    for sl_label in SL_LABELS:
        g = sub[sub['sl'] == sl_label]
        if g.empty:
            continue
        beat     = int(g['beat'].sum())
        total    = len(g)
        avg_ret  = g['ret'].mean()
        min_ret  = g['ret'].min()
        avg_dd   = g['dd'].mean()
        avg_n    = g['n'].mean()
        liq_cnt  = int(g['liq'].sum())
        marker   = ' ◀' if sl_label == '-20%' else ''
        print(f'{sl_label:>5}  {beat:>4}/{total:<3}  {avg_ret:>+8.1f}%  '
              f'{min_ret:>+8.1f}%  {avg_dd:>6.1f}%  {avg_n:>6.1f}건  '
              f'{liq_cnt:>5}회{marker}')

# ── 출력: BTC -20% 기간별 상세 ──────────────────────────────────
for coin in COINS:
    sub = df_all[(df_all['coin'] == coin.upper()) & (df_all['sl'] == '-20%')]
    sub_none = df_all[(df_all['coin'] == coin.upper()) & (df_all['sl'] == '없음')]

    print(f'\n{"="*W}')
    print(f' {coin.upper()} | 손절 없음 vs -20% 손절 비교')
    print(f'{"="*W}')
    print(f'{"기간":<13}  {"B&H":>7}  {"없음":>8}  {"없음 MDD":>8}  '
          f'{"−20%":>8}  {"−20% MDD":>8}  {"개선":>6}')
    print(f'{"-"*W}')

    for pkey in PERIODS:
        r0 = sub_none[sub_none['period'] == pkey]
        r1 = sub[sub['period'] == pkey]
        if r0.empty or r1.empty:
            continue
        r0, r1 = r0.iloc[0], r1.iloc[0]
        diff = r1['ret'] - r0['ret']
        flag = '🔴' if r0['liq'] else ('🟢' if diff > 5 else '  ')
        print(f'{r0["label"]:<13}  {r0["bh"]:>+7.1f}%  '
              f'{r0["ret"]:>+7.1f}%  {r0["dd"]:>7.1f}%  '
              f'{r1["ret"]:>+7.1f}%  {r1["dd"]:>7.1f}%  '
              f'{diff:>+5.1f}%p {flag}')

print(f'\n저장: {OUTPUT_DIR}/ema_stoploss.csv')
print('🔴 = 청산 발생 기간, 🟢 = -20% 손절이 오히려 더 좋은 기간')
