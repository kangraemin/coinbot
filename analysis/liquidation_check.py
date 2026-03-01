"""
청산 위험 시뮬레이션.

trend_holding.py의 상위 전략에 대해:
  - 각 거래 중 최대 역행 폭(MAE: Maximum Adverse Excursion) 계산
  - 실제 청산 발생 여부 시뮬레이션
  - 청산 반영 시 실제 수익률 재계산

바이낸스 선물 격리마진 기준:
  - 3x 레버리지: 진입가 대비 약 -32% 하락 시 청산
  - 2x 레버리지: 약 -49% 하락 시 청산
  - 1x 레버리지: 사실상 청산 없음
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.data_loader import load_1m, resample
from analysis.config import OUTPUT_DIR

# 청산 임계값 (격리마진, 유지증거금 ~0.5% 가정)
LIQ_THRESHOLD = {
    1: 0.99,   # 사실상 없음
    2: 0.505,  # ~49.5% 하락 시 청산
    3: 0.338,  # ~32% 하락 시 청산
}

PERIODS = {
    'bear_2026q1': ('2025-12-01', '2026-03-01'),
    'bear_2025q4': ('2025-09-15', '2025-12-01'),
    'bull_2025q3': ('2025-06-01', '2025-09-15'),
    'bull_2024h2': ('2024-07-01', '2025-01-01'),
    'bear_2024h1': ('2024-01-01', '2024-07-01'),
    'bull_2023':   ('2023-01-01', '2024-01-01'),
}
PERIOD_LABEL = {
    'bear_2026q1': '하락(26Q1)',
    'bear_2025q4': '하락(25Q4)',
    'bull_2025q3': '상승(25Q3)',
    'bull_2024h2': '불장(24H2)',
    'bear_2024h1': '횡보(24H1)',
    'bull_2023':   '회복(2023)',
}
BUY_HOLD = {
    'bear_2026q1': {'BTC': -26.8, 'ETH': -32.8},
    'bear_2025q4': {'BTC': -23.3, 'ETH': -28.4},
    'bull_2025q3': {'BTC':  +6.2, 'ETH': +66.3},
    'bull_2024h2': {'BTC': +47.4, 'ETH':  -2.4},
    'bear_2024h1': {'BTC': +43.5, 'ETH': +47.0},
    'bull_2023':   {'BTC':+155.2, 'ETH': +92.1},
}

FEE_RATE = 0.0005
SLIPPAGE = 0.0003
LOOKBACK = 1500

# 분석할 상위 전략 (trend_holding 결과 기준)
TOP_STRATEGIES = [
    ('btc', '1d', 'ema20/100', 3, False),
    ('eth', '4h', 'c>ema200',  2, False),
    ('btc', '1d', 'st3.0',     3, False),
    ('btc', '4h', 'ema50/200', 2, False),
    ('btc', '4h', 'ema20/50',  2, False),
    ('btc', '4h', 'ema50/200', 3, False),
    ('eth', '4h', 'c>ema200',  2, True),   # L+S
    ('btc', '1d', 'ema20/100', 2, False),  # 2x도 확인
    ('eth', '4h', 'c>ema200',  1, False),  # 1x
]


# ── 인디케이터 (trend_holding.py와 동일) ───────────────────
def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _supertrend(df, period=10, mult=3.0):
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    n = len(df)
    tr = np.empty(n); tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    alpha = 2/(period+1)
    atr = np.empty(n); atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha*tr[i] + (1-alpha)*atr[i-1]
    mid = (h+l)/2; bu = mid+mult*atr; bl = mid-mult*atr
    fu = np.empty(n); fl = np.empty(n); fu[0]=bu[0]; fl[0]=bl[0]
    direction = np.ones(n, dtype=bool)
    for i in range(1, n):
        fl[i] = bl[i] if bl[i]>fl[i-1] or c[i-1]<fl[i-1] else fl[i-1]
        fu[i] = bu[i] if bu[i]<fu[i-1] or c[i-1]>fu[i-1] else fu[i-1]
        if c[i]>fu[i-1]: direction[i]=True
        elif c[i]<fl[i-1]: direction[i]=False
        else: direction[i]=direction[i-1]
    return pd.Series(direction, index=df.index)

def get_signal(df, sig_name):
    c = df['close']
    if sig_name.startswith('ema'):
        parts = sig_name[3:].split('/')
        f, s = int(parts[0]), int(parts[1])
        return _ema(c, f) > _ema(c, s)
    elif sig_name.startswith('c>ema'):
        p = int(sig_name[5:])
        return c > _ema(c, p)
    elif sig_name.startswith('macd'):
        parts = sig_name[4:].split('/')
        f, s = int(parts[0]), int(parts[1])
        ml = _ema(c, f) - _ema(c, s)
        return (ml - _ema(ml, 9)) > 0
    elif sig_name.startswith('st'):
        m = float(sig_name[2:])
        return _supertrend(df, 10, m)
    raise ValueError(f'unknown signal: {sig_name}')


# ── 청산 포함 백테스트 ─────────────────────────────────────
def backtest_with_liq(df, sig, lev, with_short):
    """
    청산 시뮬레이션 포함 백테스트.
    각 거래마다: 보유 중 최저가가 청산가 아래로 내려가면 청산.
    """
    opens  = df['open'].values.astype(float)
    highs  = df['high'].values.astype(float)
    lows   = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)
    sv     = sig.values.astype(bool)
    n      = len(df)

    liq_pct = LIQ_THRESHOLD[lev]   # 진입가 대비 이 비율 아래면 청산

    trade_details = []
    pos, ep, ei = None, 0.0, 0
    peak_price = 0.0  # 롱 포지션 중 최고가 (MAE 계산용)
    min_price  = 999999.0  # 롱 포지션 중 최저가

    for i in range(1, n):
        prev = sv[i-1]

        if pos is None:
            if prev:
                ep = opens[i] * (1 + SLIPPAGE)
                ei, pos = i, 'L'
                peak_price = ep
                min_price  = ep
            elif with_short:
                ep = opens[i] * (1 - SLIPPAGE)
                ei, pos = i, 'S'
                peak_price = ep
                min_price  = ep

        elif pos == 'L':
            # 당일 최저가 추적
            min_price  = min(min_price, lows[i])
            peak_price = max(peak_price, highs[i])

            # 청산 체크: 최저가가 진입가의 (1 - liq_pct) 이하인지
            liq_price = ep * (1 - liq_pct)
            if lows[i] <= liq_price:
                # 청산 발생!
                mae_pct = (liq_price - ep) / ep * 100  # 항상 -liq_pct*100%
                trade_details.append({
                    'type': 'L', 'entry': ep, 'exit': liq_price,
                    'pnl_raw': (liq_price - ep) / ep,
                    'mae_pct': round((min_price - ep) / ep * 100, 2),
                    'mfe_pct': round((peak_price - ep) / ep * 100, 2),
                    'liquidated': True,
                    'hold_bars': i - ei,
                })
                pos = None
                if with_short:
                    ep = liq_price * (1 - SLIPPAGE)
                    ei, pos = i, 'S'
                    peak_price = ep; min_price = ep
            elif not prev:
                # 신호 반전 → 정상 청산
                xp = opens[i] * (1 - SLIPPAGE)
                trade_details.append({
                    'type': 'L', 'entry': ep, 'exit': xp,
                    'pnl_raw': (xp - ep) / ep,
                    'mae_pct': round((min_price - ep) / ep * 100, 2),
                    'mfe_pct': round((peak_price - ep) / ep * 100, 2),
                    'liquidated': False,
                    'hold_bars': i - ei,
                })
                pos = None
                if with_short:
                    ep = xp; ei, pos = i, 'S'
                    peak_price = ep; min_price = ep

        elif pos == 'S':
            peak_price = max(peak_price, highs[i])
            min_price  = min(min_price, lows[i])

            liq_price = ep * (1 + liq_pct)   # 숏은 상승 시 청산
            if highs[i] >= liq_price:
                trade_details.append({
                    'type': 'S', 'entry': ep, 'exit': liq_price,
                    'pnl_raw': (ep - liq_price) / ep,
                    'mae_pct': round((peak_price - ep) / ep * 100, 2),
                    'mfe_pct': round((ep - min_price) / ep * 100, 2),
                    'liquidated': True,
                    'hold_bars': i - ei,
                })
                pos = None
                if prev:
                    ep = liq_price * (1 + SLIPPAGE)
                    ei, pos = i, 'L'
                    peak_price = ep; min_price = ep
            elif prev:
                xp = opens[i] * (1 + SLIPPAGE)
                trade_details.append({
                    'type': 'S', 'entry': ep, 'exit': xp,
                    'pnl_raw': (ep - xp) / ep,
                    'mae_pct': round((peak_price - ep) / ep * 100, 2),
                    'mfe_pct': round((ep - min_price) / ep * 100, 2),
                    'liquidated': False,
                    'hold_bars': i - ei,
                })
                pos = None
                ep = xp * (1 + SLIPPAGE); ei, pos = i, 'L'
                peak_price = ep; min_price = ep

    # 미청산
    if pos is not None:
        xp = closes[-1]
        if pos == 'L':
            pnl_r = (xp - ep) / ep
            min_price = min(min_price, xp)
            trade_details.append({
                'type': 'L', 'entry': ep, 'exit': xp, 'pnl_raw': pnl_r,
                'mae_pct': round((min_price - ep) / ep * 100, 2),
                'mfe_pct': round((peak_price - ep) / ep * 100, 2),
                'liquidated': False, 'hold_bars': n-1-ei,
            })
        else:
            pnl_r = (ep - xp) / ep
            trade_details.append({
                'type': 'S', 'entry': ep, 'exit': xp, 'pnl_raw': pnl_r,
                'mae_pct': round((peak_price - ep) / ep * 100, 2),
                'mfe_pct': round((ep - min_price) / ep * 100, 2),
                'liquidated': False, 'hold_bars': n-1-ei,
            })

    if not trade_details:
        return dict(n=0, ret=0.0, liq_count=0, max_mae=0.0, min_mae=0.0)

    liq_count = sum(1 for t in trade_details if t['liquidated'])
    pnls = [(t['pnl_raw'] - FEE_RATE * 2) * lev * 100 for t in trade_details]
    maes = [t['mae_pct'] for t in trade_details if t['type'] == 'L']

    return dict(
        n=len(trade_details),
        ret=round(sum(pnls), 2),
        liq_count=liq_count,
        max_mae=round(min(maes) if maes else 0.0, 2),  # 가장 큰 역행 (음수)
        avg_mae=round(sum(maes)/len(maes) if maes else 0.0, 2),
        trades=trade_details,
    )


# ── 실행 ──────────────────────────────────────────────────
print('청산 위험 시뮬레이션\n')
print(f'  3x 격리마진 청산 기준: 진입가 대비 -{LIQ_THRESHOLD[3]*100:.0f}% 하락')
print(f'  2x 격리마진 청산 기준: 진입가 대비 -{LIQ_THRESHOLD[2]*100:.0f}% 하락')
print()

cache_1m = {}
cache_tf = {}

W = 74
for (coin, tf, sig_name, lev, ws) in TOP_STRATEGIES:
    # 데이터 로드 (캐시)
    if coin not in cache_1m:
        print(f'  {coin.upper()} 로드 중...')
        cache_1m[coin] = load_1m(coin, lookback_days=LOOKBACK)
    key_tf = (coin, tf)
    if key_tf not in cache_tf:
        df_tf = resample(cache_1m[coin], tf)
        if 'timestamp' in df_tf.columns:
            df_tf = df_tf.set_index('timestamp')
        if df_tf.index.tz is None:
            df_tf.index = df_tf.index.tz_localize('UTC')
        cache_tf[key_tf] = df_tf
    df_tf = cache_tf[key_tf]
    sig_full = get_signal(df_tf, sig_name)

    mode = 'L+S' if ws else 'L'
    print('=' * W)
    print(f'  {coin.upper()} {tf} | {sig_name} | {lev}x {mode}')
    print(f'  (3x 청산 기준: 진입가의 -{LIQ_THRESHOLD[3]*100:.0f}%, '
          f'2x 기준: -{LIQ_THRESHOLD[2]*100:.0f}%)')
    print('-' * W)
    print(f'  {"기간":<12} {"청산반영수익":>10} {"미반영수익":>10} {"청산횟수":>8} '
          f'{"최대역행":>9} {"B&H":>8} {"결과":>5}')
    print('-' * W)

    total_liq = 0
    beat_with_liq = 0
    beat_without_liq = 0
    total_valid = 0

    for pkey, (start, end) in PERIODS.items():
        s_ts = pd.Timestamp(start, tz='UTC')
        e_ts = pd.Timestamp(end, tz='UTC')
        mask = (df_tf.index >= s_ts) & (df_tf.index < e_ts)
        if mask.sum() < 20:
            continue

        df_p   = df_tf[mask]
        sig_p  = sig_full[mask]
        bh_ret = BUY_HOLD.get(pkey, {}).get(coin.upper())

        # 청산 포함 백테스트
        res_liq = backtest_with_liq(df_p, sig_p, lev, ws)

        # 청산 미포함 (기존)
        # 간단히 재계산
        opens  = df_p['open'].values.astype(float)
        closes = df_p['close'].values.astype(float)
        sv     = sig_p.values.astype(bool)
        n      = len(df_p)
        trades_noliq = []
        pos_n, ep_n = None, 0.0
        for i in range(1, n):
            prev = sv[i-1]
            if pos_n is None:
                if prev: ep_n, pos_n = opens[i]*(1+SLIPPAGE), 'L'
                elif ws: ep_n, pos_n = opens[i]*(1-SLIPPAGE), 'S'
            elif pos_n == 'L' and not prev:
                xp = opens[i]*(1-SLIPPAGE)
                trades_noliq.append((xp-ep_n)/ep_n)
                pos_n = None
                if ws: ep_n, pos_n = xp, 'S'
            elif pos_n == 'S' and prev:
                xp = opens[i]*(1+SLIPPAGE)
                trades_noliq.append((ep_n-xp)/ep_n)
                pos_n = None
                ep_n, pos_n = xp*(1+SLIPPAGE), 'L'
        if pos_n is not None:
            xp = closes[-1]
            trades_noliq.append((xp-ep_n)/ep_n if pos_n=='L' else (ep_n-xp)/ep_n)
        ret_noliq = round(sum((p-FEE_RATE*2)*lev*100 for p in trades_noliq), 2) if trades_noliq else 0.0

        total_liq += res_liq['liq_count']
        total_valid += 1
        flag_liq    = '✓' if (bh_ret is not None and res_liq['ret'] > bh_ret) else '✗'
        flag_noliq  = '✓' if (bh_ret is not None and ret_noliq > bh_ret) else '✗'
        if flag_liq == '✓':  beat_with_liq += 1
        if flag_noliq == '✓': beat_without_liq += 1

        liq_str = f'{res_liq["liq_count"]}회 ⚠' if res_liq['liq_count'] > 0 else '없음'
        mae_str = f'{res_liq["max_mae"]:+.1f}%'
        label   = PERIOD_LABEL[pkey]

        print(f'  {label:<12} {res_liq["ret"]:>+9.1f}%  {ret_noliq:>+9.1f}%  '
              f'{liq_str:>8}  {mae_str:>8}  {bh_ret:>+7.1f}%  '
              f'{flag_liq} ({flag_noliq})')

    print(f'\n  B&H 이긴 기간: 청산반영 {beat_with_liq}/{total_valid}  '
          f'미반영 {beat_without_liq}/{total_valid}  총청산: {total_liq}회\n')

print('=' * W)
print()
print('📌 정리:')
print(f'  - 청산가격 계산: 격리마진, 유지증거금 0.5% 가정')
print(f'  - 3x: 진입가 -32% → 청산 / 2x: 진입가 -49.5% → 청산')
print(f'  - "최대역행": 롱 포지션 보유 중 진입가 대비 최대 하락폭')
print(f'  - 최대역행이 청산기준보다 작으면 실전에서도 안전')
