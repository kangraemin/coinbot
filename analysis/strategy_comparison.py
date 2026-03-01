"""
EMA 트렌드 vs RSI+BB 역추세 전략 통계 비교.

전략 A (EMA 트렌드, 신호 반전까지 홀딩) — 2x 레버리지:
  BTC 4h: EMA50/200 크로스 → 롱 (롱 온리)  ← 청산 0회 검증
  ETH 4h: Price > EMA200   → 롱 (롱 온리)  ← 청산 1회(2017) 검증
  XRP 4h: EMA20/100 크로스 → 롱 (롱 온리)  ← 청산 0회 검증

전략 B (RSI+BB 역추세, 고정 TP/SL) — 확정 파라미터 3x 레버리지:
  BTC 4h: RSI<30 + 종가<BB하단 + 종가>EMA200 → 롱  TP=ATR×3, SL=ATR×2
           RSI>65 + 종가>BB상단 + 종가<EMA200 → 숏  TP=ATR×3, SL=ATR×2
  ETH 4h: RSI<25 + 종가<BB하단 + 종가>EMA200 → 롱  TP=ATR×2, SL=ATR×2
           RSI>65 + 종가>BB상단 + 종가<EMA200 → 숏  TP=ATR×2, SL=ATR×2
  XRP 4h: RSI<25 + 종가<BB하단 + 종가>EMA200 → 롱  TP=ATR×3, SL=ATR×2
           RSI>65 + 종가>BB상단 + 종가<EMA200 → 숏  TP=ATR×3, SL=ATR×2

2017~2026 전체 히스토리 14기간 비교.
실행: python3 analysis/strategy_comparison.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR   = Path(__file__).parent.parent / 'data' / 'market'
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEE_RATE = 0.0005   # 0.05%/side
SLIPPAGE = 0.0003   # 3bp/side

# 전략별 레버리지 분리
LEV_A    = 2        # EMA 트렌드: 2x (청산 0회 검증된 레버리지)
LEV_B    = 3        # RSI+BB: 3x (확정 파라미터)
LIQ_A    = 0.475    # 2x 청산: -47.5%
LIQ_B    = 0.317    # 3x 청산: -31.7%

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

# ── 인디케이터 ─────────────────────────────────────────────────────
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _bb(s: pd.Series, period: int = 20, num_std: float = 2.0):
    ma  = s.rolling(period).mean()
    std = s.rolling(period).std()
    return ma + num_std * std, ma - num_std * std   # upper, lower


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── 데이터 로드 (1h parquet → 4h 리샘플) ─────────────────────────
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


# ── 공통 요약 ──────────────────────────────────────────────────────
def _summarize(trades: list, lev: int) -> dict:
    if not trades:
        return dict(n=0, ret=0.0, win=0.0, dd=0.0)
    pnls = np.array([(p - FEE_RATE * 2) * lev * 100 for p in trades])
    cum  = np.cumsum(pnls)
    cum  = np.maximum(cum, -100.0)   # 계좌 소멸 이후 음수 방지
    rm   = np.maximum.accumulate(cum)
    dd   = float(np.max(rm - cum))
    return dict(
        n=len(pnls),
        ret=round(float(cum[-1]), 2),
        win=round(float((pnls > 0).mean()), 3),
        dd=round(dd, 2),
    )


# ── 전략 A: EMA 트렌드 홀딩 ──────────────────────────────────────
def backtest_trend(df: pd.DataFrame, sig: pd.Series) -> dict:
    """신호 True → 롱 진입, False → 청산. 2x 레버리지 청산가 터치 시 강제청산."""
    opens  = df['open'].values.astype(float)
    lows   = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)
    sv     = sig.values.astype(bool)
    n      = len(df)
    trades, pos, ep, liq = [], None, 0.0, False

    for i in range(1, n):
        if liq:
            break
        if pos == 'L' and lows[i] <= ep * (1 - LIQ_A):
            trades.append(-1.0)
            liq = True
            break

        prev = sv[i - 1]
        if pos is None and prev:
            ep, pos = opens[i] * (1 + SLIPPAGE), 'L'
        elif pos == 'L' and not prev:
            xp = opens[i] * (1 - SLIPPAGE)
            trades.append((xp - ep) / ep)
            pos = None

    if pos is not None and not liq:
        trades.append((closes[-1] - ep) / ep)
    return _summarize(trades, LEV_A)


# ── 전략 B: RSI+BB 역추세 (고정 TP/SL) ───────────────────────────
def backtest_rsi_bb(df: pd.DataFrame, inds: dict, coin: str) -> dict:
    """
    진입: 이전봉 RSI+BB 조건 충족 → 다음봉 시가 진입
    청산: 캔들 H/L로 TP 또는 SL 체크 (동시 터치 → SL 우선)
    """
    opens  = df['open'].values.astype(float)
    highs  = df['high'].values.astype(float)
    lows   = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)
    rsi_v  = inds['rsi14'].values
    bb_l_v = inds['bb_lower'].values
    bb_u_v = inds['bb_upper'].values
    ema_v  = inds['ema200'].values
    atr_v  = inds['atr14'].values
    n      = len(df)

    if coin == 'btc':
        rsi_long_th, rsi_short_th = 30, 65
        tp_l, sl_l = 3.0, 2.0
        tp_s, sl_s = 3.0, 2.0
        allow_short = True
    elif coin == 'eth':
        rsi_long_th, rsi_short_th = 25, 65
        tp_l, sl_l = 2.0, 2.0
        tp_s, sl_s = 2.0, 2.0
        allow_short = True
    else:  # xrp
        rsi_long_th, rsi_short_th = 25, 65
        tp_l, sl_l = 3.0, 2.0
        tp_s, sl_s = 3.0, 2.0
        allow_short = True

    trades, pos, ep = [], None, 0.0
    tp_p, sl_p = 0.0, 0.0

    for i in range(1, n):
        # ① TP/SL 체크 (보유 중인 포지션)
        if pos == 'L':
            hit_sl = lows[i] <= sl_p
            hit_tp = highs[i] >= tp_p
            if hit_sl and hit_tp:
                trades.append((sl_p - ep) / ep); pos = None  # SL 우선
            elif hit_sl:
                trades.append((sl_p - ep) / ep); pos = None
            elif hit_tp:
                trades.append((tp_p - ep) / ep); pos = None
        elif pos == 'S':
            hit_sl = highs[i] >= sl_p
            hit_tp = lows[i] <= tp_p
            if hit_sl and hit_tp:
                trades.append((ep - sl_p) / ep); pos = None  # SL 우선
            elif hit_sl:
                trades.append((ep - sl_p) / ep); pos = None
            elif hit_tp:
                trades.append((ep - tp_p) / ep); pos = None

        # ② 신규 진입 (이전봉 조건)
        if pos is not None:
            continue
        prev = i - 1
        rsi_ok    = not np.isnan(rsi_v[prev])
        bb_ok     = not np.isnan(bb_l_v[prev])
        ema_ok    = not np.isnan(ema_v[prev])
        atr_ok    = not np.isnan(atr_v[prev]) and atr_v[prev] > 0

        if rsi_ok and bb_ok and ema_ok and atr_ok:
            long_cond = (rsi_v[prev] < rsi_long_th and
                         closes[prev] < bb_l_v[prev] and
                         closes[prev] > ema_v[prev])
            if long_cond:
                ep   = opens[i] * (1 + SLIPPAGE)
                tp_p = ep + atr_v[prev] * tp_l
                sl_p = ep - atr_v[prev] * sl_l
                pos  = 'L'
                continue

            if allow_short:
                short_cond = (rsi_v[prev] > rsi_short_th and
                              closes[prev] > bb_u_v[prev] and
                              closes[prev] < ema_v[prev])
                if short_cond:
                    ep   = opens[i] * (1 - SLIPPAGE)
                    tp_p = ep - atr_v[prev] * tp_s
                    sl_p = ep + atr_v[prev] * sl_s
                    pos  = 'S'

    # 미청산 → 마지막 종가
    if pos is not None:
        if pos == 'L':
            trades.append((closes[-1] - ep) / ep)
        else:
            trades.append((ep - closes[-1]) / ep)
    return _summarize(trades, LEV_B)


# ── 코인별 전략 A 신호 정의 ────────────────────────────────────────
def ema_signal(df: pd.DataFrame, coin: str) -> pd.Series:
    c = df['close']
    if coin == 'btc':
        return _ema(c, 50) > _ema(c, 200)
    elif coin == 'eth':
        return c > _ema(c, 200)
    else:  # xrp
        return _ema(c, 20) > _ema(c, 100)


EMA_LABEL = {'btc': 'EMA50/200', 'eth': 'c>EMA200', 'xrp': 'EMA20/100'}
COINS     = ['btc', 'eth', 'xrp']


# ── 메인 ──────────────────────────────────────────────────────────
rows = []

for coin in COINS:
    print(f'\n▶ {coin.upper()} 4h 로드 및 지표 계산...')
    df4 = load_4h(coin)
    c   = df4['close']

    # 전략 A 신호 (전체 데이터에서 계산 → 슬라이싱)
    sig_a = ema_signal(df4, coin)

    # 전략 B 지표 (전체 데이터에서 계산 → 슬라이싱)
    bb_u, bb_l = _bb(c, 20, 2.0)
    inds_full = {
        'ema200':   _ema(c, 200),
        'rsi14':    _rsi(c, 14),
        'bb_upper': bb_u,
        'bb_lower': bb_l,
        'atr14':    _atr(df4, 14),
    }

    for pkey, (start, end) in PERIODS.items():
        s_ts = pd.Timestamp(start, tz='UTC')
        e_ts = pd.Timestamp(end, tz='UTC')
        mask = (df4.index >= s_ts) & (df4.index < e_ts)
        if mask.sum() < 20:
            continue

        df_p   = df4[mask]
        sig_p  = sig_a[mask]
        inds_p = {k: v[mask] for k, v in inds_full.items()}

        bh_ret = round((df_p['close'].iloc[-1] / df_p['open'].iloc[0] - 1) * 100, 2)
        res_a  = backtest_trend(df_p, sig_p)
        res_b  = backtest_rsi_bb(df_p, inds_p, coin)

        rows.append({
            'coin':   coin.upper(),
            'period': pkey,
            'label':  PERIOD_LABEL[pkey],
            'bh_ret': bh_ret,
            'a_ret':  res_a['ret'], 'a_n':  res_a['n'],
            'a_win':  res_a['win'], 'a_dd': res_a['dd'],
            'a_beat': int(res_a['ret'] > bh_ret),
            'b_ret':  res_b['ret'], 'b_n':  res_b['n'],
            'b_win':  res_b['win'], 'b_dd': res_b['dd'],
            'b_beat': int(res_b['ret'] > bh_ret),
        })

df_res = pd.DataFrame(rows)
df_res.to_csv(OUTPUT_DIR / 'strategy_comparison.csv', index=False)

# ── 출력 ──────────────────────────────────────────────────────────
W = 100
for coin in COINS:
    sub = df_res[df_res['coin'] == coin.upper()]
    print(f'\n{"="*W}')
    print(f' {coin.upper()} | A: EMA 트렌드 ({EMA_LABEL[coin]}) {LEV_A}x  vs  B: RSI+BB 역추세 {LEV_B}x')
    print(f'{"="*W}')
    print(f'{"기간":<13} {"B&H":>7}  '
          f'{"A수익":>8}{"":1} {"A거래":>5} {"A승률":>5} {"AMDD":>6}  '
          f'{"B수익":>8}{"":1} {"B거래":>5} {"B승률":>5} {"BMDD":>6}  {"승자":>4}')
    print(f'{"-"*W}')

    a_beat, b_beat = 0, 0
    for _, r in sub.iterrows():
        winner = 'A' if r['a_ret'] > r['b_ret'] else ('B' if r['b_ret'] > r['a_ret'] else '=')
        ab = '✓' if r['a_beat'] else '✗'
        bb_ = '✓' if r['b_beat'] else '✗'
        a_beat += r['a_beat']
        b_beat += r['b_beat']
        print(
            f'{r["label"]:<13} {r["bh_ret"]:>+7.1f}%  '
            f'{r["a_ret"]:>+7.1f}%{ab} {r["a_n"]:>5} {r["a_win"]:>5.0%} {r["a_dd"]:>5.1f}%  '
            f'{r["b_ret"]:>+7.1f}%{bb_} {r["b_n"]:>5} {r["b_win"]:>5.0%} {r["b_dd"]:>5.1f}%  '
            f'[{winner}]'
        )

    total = len(sub)
    print(f'{"-"*W}')
    print(f'  B&H 초과: A {a_beat}/{total}  B {b_beat}/{total}')
    print(f'  평균수익: A {sub["a_ret"].mean():+.1f}%  B {sub["b_ret"].mean():+.1f}%')
    print(f'  최소수익: A {sub["a_ret"].min():+.1f}%  B {sub["b_ret"].min():+.1f}%')
    print(f'  평균 MDD: A {sub["a_dd"].mean():.1f}%  B {sub["b_dd"].mean():.1f}%')
    print(f'  평균거래: A {sub["a_n"].mean():.1f}건  B {sub["b_n"].mean():.1f}건')

print(f'\n{"="*W}')
print(' 전체 요약 (3코인 합산)')
print(f'{"="*W}')
total_rows = len(df_res)
for col, label in [('a', 'EMA 트렌드 '), ('b', 'RSI+BB    ')]:
    beat  = int(df_res[f'{col}_beat'].sum())
    avg   = df_res[f'{col}_ret'].mean()
    mn    = df_res[f'{col}_ret'].min()
    dd    = df_res[f'{col}_dd'].mean()
    n_avg = df_res[f'{col}_n'].mean()
    print(f'  {label}: B&H 초과 {beat}/{total_rows} | 평균수익 {avg:+.1f}% | '
          f'최저수익 {mn:+.1f}% | 평균MDD {dd:.1f}% | 평균거래 {n_avg:.1f}건')

print(f'\n저장: {OUTPUT_DIR}/strategy_comparison.csv')
