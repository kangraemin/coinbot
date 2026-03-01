"""피보나치 + 통계 기반 롱/숏 전략 탐색.

핵심 아이디어:
  1. 스윙 고점/저점 자동 탐지 (ZigZag 방식)
  2. 피보나치 되돌림 레벨 계산 (0.236 / 0.382 / 0.5 / 0.618 / 0.786)
  3. 가격이 레벨 근처에 도달 시 통계적 확인 신호 대기:
       - RSI 다이버전스 (가격 신저 but RSI는 더 높은 저점)
       - 거래량 급증 (평균의 1.5배 이상)
       - Z-Score 과매도/과매수
       - 캔들 반전 패턴 (핀바, 도지)
  4. 상위 TF 추세 방향으로만 진입 (EMA200)

레버리지 1x, 비교: vs 바이앤홀드
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.config import DATA_DIR, OUTPUT_DIR, RESAMPLE_RULES

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "fibonacci_stat.csv"

PERIODS = {
    'recent_3m':   ('2025-12-02', '2026-03-02', '하락장(최근)'),
    'bear_2025q4': ('2025-10-01', '2026-01-01', '하락장(Q4)'),
    'bull_2025q3': ('2025-07-01', '2025-10-01', '상승(Q3)'),
    'bull_2024h2': ('2024-07-01', '2024-12-31', '불장(24H2)'),
    'bear_2024h1': ('2024-01-01', '2024-06-30', '횡보(24H1)'),
    'bull_2023':   ('2023-01-01', '2023-12-31', '회복(2023)'),
}
BUY_HOLD = {
    'recent_3m':   {'BTC': -26.8, 'ETH': -32.8},
    'bear_2025q4': {'BTC': -23.3, 'ETH': -28.4},
    'bull_2025q3': {'BTC':  +6.2, 'ETH': +66.3},
    'bull_2024h2': {'BTC': +47.4, 'ETH':  -2.4},
    'bear_2024h1': {'BTC': +43.5, 'ETH': +47.0},
    'bull_2023':   {'BTC':+155.2, 'ETH': +92.1},
}

COINS   = ['btc', 'eth']
TFS     = ['1h', '4h']
FEE     = 0.0005
SLIP    = 0.0003
TOUT    = 48
LEV     = 1
MIN_TR  = 8

FIB_LEVELS  = [0.236, 0.382, 0.500, 0.618, 0.786]
FIB_TOL     = 0.008   # ±0.8% 허용 오차
SWING_WIN   = 10      # 스윙 고점/저점 탐지 윈도우


# ── 데이터 ────────────────────────────────────────────────────────────────────

def load_period(coin, start, end):
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    files = [DATA_DIR / f"{coin}_1m_{y}.parquet"
             for y in range(s.year, e.year + 1)
             if (DATA_DIR / f"{coin}_1m_{y}.parquet").exists()]
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp')
    return df[['open','high','low','close','volume']][(df.index >= s) & (df.index < e)]

def resample_df(df, tf):
    rule = RESAMPLE_RULES.get(tf, tf)
    r = df.resample(rule).agg({'open':'first','high':'max',
                               'low':'min','close':'last','volume':'sum'}).dropna()
    return r.reset_index()


# ── 지표 계산 ─────────────────────────────────────────────────────────────────

def add_indicators(df):
    c = df['close']
    df['ema200']    = ta.trend.EMAIndicator(c, 200).ema_indicator()
    df['ema50']     = ta.trend.EMAIndicator(c, 50).ema_indicator()
    df['rsi']       = ta.momentum.RSIIndicator(c, 14).rsi()
    df['vol_ma']    = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['atr']       = ta.volatility.AverageTrueRange(
                          df['high'], df['low'], c, 14).average_true_range()
    # Z-Score
    df['zscore']    = (c - c.rolling(50).mean()) / c.rolling(50).std()
    # RSI 다이버전스용 — 가격 저점 vs RSI 저점 비교
    df['price_low_20']  = df['low'].rolling(20).min()
    df['rsi_at_low']    = df['rsi'].where(df['low'] == df['price_low_20'])
    return df


# ── 스윙 고점/저점 탐지 ───────────────────────────────────────────────────────

def find_swings(df, win=SWING_WIN):
    """로컬 고점/저점 탐지. 반환: swing_high, swing_low Series."""
    highs = df['high']
    lows  = df['low']

    swing_high = pd.Series(np.nan, index=df.index)
    swing_low  = pd.Series(np.nan, index=df.index)

    for i in range(win, len(df) - win):
        h = highs.iloc[i]
        l = lows.iloc[i]
        if h == highs.iloc[i-win:i+win+1].max():
            swing_high.iloc[i] = h
        if l == lows.iloc[i-win:i+win+1].min():
            swing_low.iloc[i] = l

    return swing_high, swing_low


# ── 피보나치 레벨 계산 ────────────────────────────────────────────────────────

def fib_levels_from_swing(high, low, direction='up'):
    """스윙 기준 피보나치 되돌림 레벨 계산.

    direction='up'  : 상승 후 되돌림 (롱 지지선)
    direction='down': 하락 후 되돌림 (숏 저항선)
    """
    diff = high - low
    if direction == 'up':
        # 상승 후 되돌림: high → low 방향 지지
        return {lvl: high - diff * lvl for lvl in FIB_LEVELS}
    else:
        # 하락 후 되돌림: low → high 방향 저항
        return {lvl: low + diff * lvl for lvl in FIB_LEVELS}


# ── 피보나치 터치 신호 생성 ───────────────────────────────────────────────────

def compute_fib_signals(df):
    """
    각 봉마다 최근 스윙 기준 피보나치 레벨 근접 여부 계산.
    반환: fib_long (bool), fib_short (bool), touched_level (float)
    """
    swing_high, swing_low = find_swings(df)

    n = len(df)
    fib_long  = np.zeros(n, dtype=bool)
    fib_short = np.zeros(n, dtype=bool)
    touched   = np.full(n, np.nan)

    # 가장 최근 스윙 고점/저점 추적
    last_sh = np.nan
    last_sh_idx = -1
    last_sl = np.nan
    last_sl_idx = -1

    for i in range(SWING_WIN * 2, n):
        # 스윙 업데이트
        if not np.isnan(swing_high.iloc[i - SWING_WIN]):
            last_sh = swing_high.iloc[i - SWING_WIN]
            last_sh_idx = i - SWING_WIN
        if not np.isnan(swing_low.iloc[i - SWING_WIN]):
            last_sl = swing_low.iloc[i - SWING_WIN]
            last_sl_idx = i - SWING_WIN

        if np.isnan(last_sh) or np.isnan(last_sl):
            continue

        price = df['close'].iloc[i]
        ema200_val = df['ema200'].iloc[i]

        # ── 롱: 상승 후 되돌림 (고점 → 현재까지 하락, 피보 지지에서 반등) ──
        # 조건: 직전 고점 > 직전 저점 + 현재 EMA200 위 또는 근접
        if last_sh > last_sl and last_sh_idx > last_sl_idx:
            levels = fib_levels_from_swing(last_sh, last_sl, 'up')
            for lvl, price_lvl in levels.items():
                # 0.618 이상 깊은 되돌림에서만 롱 (더 신뢰도 높음)
                if lvl >= 0.382:
                    tolerance = price_lvl * FIB_TOL
                    if abs(price - price_lvl) <= tolerance:
                        # 반등 확인: 현재봉이 양봉
                        if df['close'].iloc[i] > df['open'].iloc[i]:
                            fib_long[i] = True
                            touched[i] = lvl
                            break

        # ── 숏: 하락 후 되돌림 (저점 → 현재까지 반등, 피보 저항에서 하락) ──
        if last_sl < last_sh and last_sl_idx > last_sh_idx:
            levels = fib_levels_from_swing(last_sh, last_sl, 'down')
            for lvl, price_lvl in levels.items():
                if lvl >= 0.382:
                    tolerance = price_lvl * FIB_TOL
                    if abs(price - price_lvl) <= tolerance:
                        # 저항 확인: 현재봉이 음봉
                        if df['close'].iloc[i] < df['open'].iloc[i]:
                            fib_short[i] = True
                            touched[i] = lvl
                            break

    return (pd.Series(fib_long, index=df.index),
            pd.Series(fib_short, index=df.index),
            pd.Series(touched, index=df.index))


# ── 통계적 확인 필터들 ────────────────────────────────────────────────────────

def filt_rsi_oversold(df, sig, thresh=40):
    return sig & (df['rsi'] < thresh)

def filt_rsi_overbought(df, sig, thresh=60):
    return sig & (df['rsi'] > thresh)

def filt_vol_spike(df, sig, mult=1.5):
    return sig & (df['vol_ratio'] >= mult)

def filt_zscore_low(df, sig, thresh=-1.5):
    return sig & (df['zscore'] < thresh)

def filt_zscore_high(df, sig, thresh=1.5):
    return sig & (df['zscore'] > thresh)

def filt_ema200_above(df, sig):
    return sig & (df['close'] > df['ema200'])

def filt_ema200_below(df, sig):
    return sig & (df['close'] < df['ema200'])

def filt_deep_fib(df, sig, touched, min_level=0.618):
    """깊은 되돌림(0.618 이상)에서만."""
    return sig & (touched >= min_level)

def filt_rsi_divergence_long(df, sig, window=10):
    """불리시 다이버전스: 가격은 신저 but RSI는 이전 저점보다 높음."""
    price_lower = df['low'] < df['low'].shift(window)
    rsi_higher  = df['rsi'] > df['rsi'].shift(window)
    return sig & price_lower & rsi_higher

# 필터 조합
FILTER_SETS_LONG = [
    ('기본',          lambda df, s, t: s),
    ('RSI과매도',      lambda df, s, t: filt_rsi_oversold(df, s)),
    ('거래량스파이크',  lambda df, s, t: filt_vol_spike(df, s)),
    ('Z-Score낮음',   lambda df, s, t: filt_zscore_low(df, s)),
    ('깊은피보',       lambda df, s, t: filt_deep_fib(df, s, t)),
    ('RSI+거래량',    lambda df, s, t: filt_vol_spike(df, filt_rsi_oversold(df, s))),
    ('RSI+깊은피보',  lambda df, s, t: filt_deep_fib(df, filt_rsi_oversold(df, s), t)),
    ('Z+거래량',      lambda df, s, t: filt_vol_spike(df, filt_zscore_low(df, s))),
    ('RSI다이버전스',  lambda df, s, t: filt_rsi_divergence_long(df, s)),
    ('EMA200위+RSI',  lambda df, s, t: filt_rsi_oversold(df, filt_ema200_above(df, s))),
]

FILTER_SETS_SHORT = [
    ('기본',          lambda df, s, t: s),
    ('RSI과매수',      lambda df, s, t: filt_rsi_overbought(df, s)),
    ('거래량스파이크',  lambda df, s, t: filt_vol_spike(df, s)),
    ('Z-Score높음',   lambda df, s, t: filt_zscore_high(df, s)),
    ('깊은피보',       lambda df, s, t: filt_deep_fib(df, s, t)),
    ('RSI+거래량',    lambda df, s, t: filt_vol_spike(df, filt_rsi_overbought(df, s))),
    ('RSI+깊은피보',  lambda df, s, t: filt_deep_fib(df, filt_rsi_overbought(df, s), t)),
    ('Z+거래량',      lambda df, s, t: filt_vol_spike(df, filt_zscore_high(df, s))),
    ('EMA200아래+RSI', lambda df, s, t: filt_rsi_overbought(df, filt_ema200_below(df, s))),
]


# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────

def backtest(df, long_sig, short_sig, tp_pct, sl_pct):
    opens  = df['open'].to_numpy(float)
    highs  = df['high'].to_numpy(float)
    lows   = df['low'].to_numpy(float)
    closes = df['close'].to_numpy(float)

    la = long_sig.to_numpy(bool)
    sa = short_sig.to_numpy(bool)
    n  = len(df)

    trades = []
    i = 0
    while i < n - 1:
        is_l = la[i] if i < len(la) else False
        is_s = sa[i] if i < len(sa) else False
        if not is_l and not is_s:
            i += 1
            continue

        d  = 'long' if is_l else 'short'
        ei = i + 1
        ep = opens[ei] * (1 + SLIP if d == 'long' else 1 - SLIP)
        tp = ep * (1 + tp_pct if d == 'long' else 1 - tp_pct)
        sl = ep * (1 - sl_pct if d == 'long' else 1 + sl_pct)

        xp, xr, xi = None, None, ei
        for j in range(ei, min(ei + TOUT + 1, n)):
            if d == 'long':
                if lows[j]  <= sl: xp, xr, xi = sl, 'sl', j; break
                if highs[j] >= tp: xp, xr, xi = tp, 'tp', j; break
            else:
                if highs[j] >= sl: xp, xr, xi = sl, 'sl', j; break
                if lows[j]  <= tp: xp, xr, xi = tp, 'tp', j; break
            if j == min(ei + TOUT, n - 1):
                xp, xr, xi = closes[j], 'timeout', j; break

        if xp is None:
            xp, xr, xi = closes[min(ei + TOUT, n - 1)], 'timeout', min(ei + TOUT, n - 1)

        pnl = ((xp - ep)/ep if d == 'long' else (ep - xp)/ep) - FEE * 2
        trades.append({'d': d, 'pnl': pnl * LEV, 'r': xr})
        i = xi + 1

    if len(trades) < MIN_TR:
        return None

    p   = np.array([t['pnl'] for t in trades])
    cum = np.cumsum(p)
    dd  = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0
    sh  = float(p.mean() / p.std() * np.sqrt(len(p))) if p.std() > 0 else 0

    tdf = pd.DataFrame(trades)
    return {
        'total_trades': len(tdf),
        'long_trades':  int((tdf['d'] == 'long').sum()),
        'short_trades': int((tdf['d'] == 'short').sum()),
        'win_rate':     round((p > 0).sum() / len(p) * 100, 1),
        'total_return': round(float(p.sum() * 100), 2),
        'sharpe':       round(sh, 3),
        'max_dd':       round(dd * 100, 2),
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

TP_SL_LIST = [(0.03, 0.01), (0.05, 0.02), (0.04, 0.015), (0.06, 0.02)]

def run():
    all_rows = []

    for coin in COINS:
        print(f"\n{'='*60}  {coin.upper()}")

        pdata = {}
        for pkey, (s, e, _) in PERIODS.items():
            df1m = load_period(coin, s, e)
            if df1m is None:
                continue
            pdata[pkey] = {}
            for tf in TFS:
                df = resample_df(df1m, tf)
                if len(df) < 200:
                    continue
                pdata[pkey][tf] = add_indicators(df)

        for tf in TFS:
            print(f"  [{tf}] 피보나치 신호 계산 중...", end='', flush=True)

            # 기간별 피보나치 신호 미리 계산
            fib_cache = {}
            for pkey, tf_data in pdata.items():
                if tf not in tf_data:
                    continue
                df = tf_data[tf]
                try:
                    fl, fs, ft = compute_fib_signals(df)
                    fib_cache[pkey] = (fl, fs, ft, df)
                except Exception as e:
                    pass

            found = 0
            for fl_name, fl_fn in FILTER_SETS_LONG:
                for fs_name, fs_fn in FILTER_SETS_SHORT:
                    for tp_pct, sl_pct in TP_SL_LIST:
                        param = f"TP{int(tp_pct*100)}SL{int(sl_pct*100)}"
                        period_results = {}

                        for pkey, (fl_raw, fs_raw, ft, df) in fib_cache.items():
                            try:
                                long_sig  = fl_fn(df, fl_raw, ft).fillna(False)
                                short_sig = fs_fn(df, fs_raw, ft).fillna(False)
                            except Exception:
                                continue

                            if long_sig.sum() + short_sig.sum() < MIN_TR:
                                continue

                            r = backtest(df, long_sig, short_sig, tp_pct, sl_pct)
                            if r:
                                period_results[pkey] = r

                        if len(period_results) < 4:
                            continue

                        returns  = [r['total_return'] for r in period_results.values()]
                        sharpes  = [r['sharpe'] for r in period_results.values()]
                        positive = sum(1 for r in returns if r > 0)
                        beat_bh  = sum(1 for p, r in period_results.items()
                                       if r['total_return'] > BUY_HOLD.get(p, {}).get(coin.upper(), 9999))

                        if positive < 4:
                            continue

                        row = {
                            'coin': coin.upper(), 'tf': tf,
                            'long_filter': fl_name, 'short_filter': fs_name,
                            'params': param,
                            'positive': positive, 'beat_bh': beat_bh,
                            'n_periods': len(period_results),
                            'avg_ret':   round(np.mean(returns), 2),
                            'min_ret':   round(np.min(returns), 2),
                            'avg_sharpe': round(np.mean(sharpes), 3),
                            'total_trades': sum(r['total_trades'] for r in period_results.values()),
                        }
                        for pkey, (_, _, pl) in PERIODS.items():
                            r = period_results.get(pkey)
                            row[f'{pkey}_ret']  = r['total_return'] if r else None
                            row[f'{pkey}_wr']   = r['win_rate'] if r else None
                            row[f'{pkey}_tr']   = r['total_trades'] if r else None
                            row[f'{pkey}_long'] = r['long_trades'] if r else None
                            row[f'{pkey}_sht']  = r['short_trades'] if r else None
                        all_rows.append(row)
                        found += 1

            print(f" → {found}개")

    if not all_rows:
        print("\n조건 충족 없음 — 필터 완화 필요")
        return

    df_out = pd.DataFrame(all_rows).sort_values(['beat_bh','avg_sharpe'], ascending=False)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n\n{'='*80}")
    print(f"  ★ {len(df_out)}개 후보 (양수 ≥ 4/6, 레버리지 1x)")
    print(f"{'='*80}\n")

    show = ['coin','tf','long_filter','short_filter','params',
            'positive','beat_bh','n_periods','avg_ret','min_ret','avg_sharpe','total_trades']
    print(df_out[show].head(20).to_string(index=False))

    print(f"\n\n  [TOP 5 기간별 상세]")
    for _, row in df_out.head(5).iterrows():
        print(f"\n  {row['coin']} {row['tf']} | 롱필터:{row['long_filter']} | 숏필터:{row['short_filter']} | {row['params']}")
        print(f"  양수:{row['positive']}/{row['n_periods']}  B&H초과:{row['beat_bh']}기간  "
              f"평균:{row['avg_ret']:+.1f}%  최소:{row['min_ret']:+.1f}%  "
              f"Sharpe:{row['avg_sharpe']:.3f}  총거래:{row['total_trades']}건")
        for pkey, (_, _, pl) in PERIODS.items():
            ret = row.get(f'{pkey}_ret')
            wr  = row.get(f'{pkey}_wr')
            tr  = row.get(f'{pkey}_tr')
            lng = row.get(f'{pkey}_long')
            sht = row.get(f'{pkey}_sht')
            bh  = BUY_HOLD.get(pkey, {}).get(row['coin'])
            if ret is None or (isinstance(ret, float) and np.isnan(ret)):
                continue
            beat = '✓' if bh is not None and ret > bh else '✗'
            t = int(tr)  if tr  and not np.isnan(tr)  else 0
            l = int(lng) if lng and not np.isnan(lng) else 0
            s = int(sht) if sht and not np.isnan(sht) else 0
            print(f"    {beat} {pl:<14} {ret:+6.1f}%  vs B&H{bh:+.0f}%  "
                  f"승률{wr:.0f}%  롱{l}/숏{s}건")

    print(f"\n  저장: {OUT_CSV}")


if __name__ == '__main__':
    run()
