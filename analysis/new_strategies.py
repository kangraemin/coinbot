"""완전히 다른 계열의 전략 탐색.

기존에 테스트한 것들 (BB, RSI, MACD, EMA크로스, 스토캐스틱) 제외.

새로 탐색할 전략군:
  1. 터틀 트레이딩   — 20/55봉 신고가 돌파 (추세 추종)
  2. Ichimoku       — 구름 돌파 + Tenkan/Kijun 크로스
  3. VWAP 이탈 복귀 — VWAP 대비 과도한 이탈 후 회귀
  4. Z-Score        — 통계적 과매도 (평균에서 2σ 이탈)
  5. 캔들 클러스터   — 연속 음봉 후 반등 (단순 패턴)
  6. 고점/저점 돌파  — 직전 N봉 고점/저점 돌파
  7. 멀티TF 정배열  — 상위 TF 방향 + 하위 TF 진입
  8. 변동성 수축 돌파— ATR 수축 후 확장 (볼린저와 다름)
  9. 피벗 포인트    — 일간/주간 피벗 지지/저항 반등
 10. 거래량 역추세  — 극단적 거래량 + 반전 캔들

비교: 레버리지 1x 전략 vs 바이앤홀드
목표: 6개 기간 중 4개 이상 B&H 초과
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.config import DATA_DIR, OUTPUT_DIR, FEE_RATE, SLIPPAGE, TIMEOUT_BARS, RESAMPLE_RULES

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "new_strategies.csv"

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

COINS  = ['btc', 'eth']
TFS    = ['15m', '1h', '4h']
LEV    = 1
FEE    = 0.0005
SLIP   = 0.0003
TOUT   = 48
MIN_TR = 8


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
    return df[['open','high','low','close','volume']][(
        df.index >= s) & (df.index < e)]

def resample_df(df, tf):
    rule = RESAMPLE_RULES.get(tf, tf)
    r = df.resample(rule).agg({'open':'first','high':'max',
                               'low':'min','close':'last','volume':'sum'}).dropna()
    return r.reset_index()

def add_base(df):
    c = df['close']
    df['ema200'] = ta.trend.EMAIndicator(c, 200).ema_indicator()
    df['atr']    = ta.volatility.AverageTrueRange(
                       df['high'], df['low'], c, 14).average_true_range()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    return df


# ── 전략 1: 터틀 트레이딩 (Donchian 돌파) ────────────────────────────────────

def turtle_long(df, n=20):
    """n봉 신고가 돌파 → 추세 추종 롱."""
    highest = df['high'].shift(1).rolling(n).max()
    return df['high'] > highest

def turtle_short(df, n=20):
    """n봉 신저가 이탈 → 추세 추종 숏."""
    lowest = df['low'].shift(1).rolling(n).min()
    return df['low'] < lowest


# ── 전략 2: Ichimoku ──────────────────────────────────────────────────────────

def ichi_indicators(df):
    h, l, c = df['high'], df['low'], df['close']
    tenkan  = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    kijun   = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, senkou_a, senkou_b

def ichi_long(df):
    """가격이 구름(Kumo) 위 돌파 + Tenkan > Kijun."""
    tenkan, kijun, sa, sb = ichi_indicators(df)
    cloud_top = pd.concat([sa, sb], axis=1).max(axis=1)
    above_cloud = df['close'] > cloud_top
    tk_cross    = (tenkan.shift(1) <= kijun.shift(1)) & (tenkan > kijun)
    return above_cloud & tk_cross

def ichi_short(df):
    """가격이 구름 아래 돌파 + Tenkan < Kijun."""
    tenkan, kijun, sa, sb = ichi_indicators(df)
    cloud_bot = pd.concat([sa, sb], axis=1).min(axis=1)
    below_cloud = df['close'] < cloud_bot
    tk_cross    = (tenkan.shift(1) >= kijun.shift(1)) & (tenkan < kijun)
    return below_cloud & tk_cross


# ── 전략 3: VWAP 이탈 복귀 ───────────────────────────────────────────────────

def vwap_series(df):
    """누적 VWAP 계산 (일 단위 리셋은 생략, 전체 누적)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    cum_tp_vol = (tp * df['volume']).cumsum()
    cum_vol    = df['volume'].cumsum()
    return cum_tp_vol / cum_vol

def vwap_long(df, dev=0.03):
    """VWAP 대비 dev% 이상 하락 후 VWAP 회복."""
    vwap = vwap_series(df)
    far_below = df['close'].shift(1) < vwap.shift(1) * (1 - dev)
    recovering = df['close'] >= vwap * (1 - dev * 0.3)
    return far_below & recovering

def vwap_short(df, dev=0.03):
    """VWAP 대비 dev% 이상 상승 후 VWAP 회귀."""
    vwap = vwap_series(df)
    far_above = df['close'].shift(1) > vwap.shift(1) * (1 + dev)
    declining  = df['close'] <= vwap * (1 + dev * 0.3)
    return far_above & declining


# ── 전략 4: Z-Score 통계 이탈 ────────────────────────────────────────────────

def zscore(series, window=50):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    return (series - m) / s

def zscore_long(df, window=50, thresh=-2.0):
    """종가 Z-Score가 thresh 이하로 떨어진 후 회복."""
    z = zscore(df['close'], window)
    was_below = z.shift(1) < thresh
    recovering = z >= thresh * 0.5
    return was_below & recovering

def zscore_short(df, window=50, thresh=2.0):
    """종가 Z-Score가 thresh 이상 → 회귀."""
    z = zscore(df['close'], window)
    was_above = z.shift(1) > thresh
    declining  = z <= thresh * 0.5
    return was_above & declining


# ── 전략 5: 연속 음봉/양봉 후 반전 ──────────────────────────────────────────

def consec_long(df, n=3):
    """연속 n개 음봉 후 양봉 (반전)."""
    bearish = df['close'] < df['open']
    consec  = bearish.rolling(n).sum() == n
    reversal = df['close'] > df['open']
    return consec.shift(1) & reversal

def consec_short(df, n=3):
    """연속 n개 양봉 후 음봉 (반전)."""
    bullish  = df['close'] > df['open']
    consec   = bullish.rolling(n).sum() == n
    reversal = df['close'] < df['open']
    return consec.shift(1) & reversal


# ── 전략 6: 고점/저점 돌파 (단기 브레이크아웃) ──────────────────────────────

def breakout_long(df, n=10):
    """직전 n봉 고점 돌파 + 거래량 증가."""
    prev_high = df['high'].shift(1).rolling(n).max()
    vol_surge = df['volume'] > df['vol_ma'] * 1.5
    return (df['close'] > prev_high) & vol_surge

def breakout_short(df, n=10):
    """직전 n봉 저점 이탈 + 거래량 증가."""
    prev_low  = df['low'].shift(1).rolling(n).min()
    vol_surge = df['volume'] > df['vol_ma'] * 1.5
    return (df['close'] < prev_low) & vol_surge


# ── 전략 7: 멀티타임프레임 정배열 (1h 방향 + 15m 진입) ──────────────────────
# → 별도 처리 (상위 TF 데이터 필요)


# ── 전략 8: ATR 수축 후 확장 돌파 ────────────────────────────────────────────

def atr_squeeze_long(df, window=20, ratio=0.7):
    """ATR이 최근 평균의 ratio배 이하 (수축) → 상향 돌파."""
    atr_ma = df['atr'].rolling(window).mean()
    squeezed = df['atr'].shift(1) < atr_ma.shift(1) * ratio
    breakout  = df['close'] > df['close'].shift(1).rolling(5).max()
    return squeezed & breakout

def atr_squeeze_short(df, window=20, ratio=0.7):
    """ATR 수축 후 하향 이탈."""
    atr_ma = df['atr'].rolling(window).mean()
    squeezed  = df['atr'].shift(1) < atr_ma.shift(1) * ratio
    breakdown = df['close'] < df['close'].shift(1).rolling(5).min()
    return squeezed & breakdown


# ── 전략 9: 피벗 포인트 반등 ─────────────────────────────────────────────────

def pivot_long(df, n=24):
    """직전 n봉 피벗(고+저+종/3) 근처에서 반등."""
    pivot = (df['high'].shift(n) + df['low'].shift(n) + df['close'].shift(n)) / 3
    near_pivot = (df['low'] <= pivot * 1.005) & (df['low'] >= pivot * 0.995)
    bouncing   = df['close'] > df['open']
    return near_pivot & bouncing

def pivot_short(df, n=24):
    """피벗 근처에서 저항 후 하락."""
    pivot = (df['high'].shift(n) + df['low'].shift(n) + df['close'].shift(n)) / 3
    near_pivot = (df['high'] >= pivot * 0.995) & (df['high'] <= pivot * 1.005)
    declining  = df['close'] < df['open']
    return near_pivot & declining


# ── 전략 10: 극단적 거래량 + 핀바 반전 ──────────────────────────────────────

def vol_reversal_long(df):
    """극단 거래량(3σ) + 아래꼬리 긴 핀바 → 강한 매수세."""
    vol_extreme = df['volume'] > df['vol_ma'] + df['volume'].rolling(20).std() * 2.5
    body    = abs(df['close'] - df['open'])
    lo_tail = df['open'].combine(df['close'], min) - df['low']
    hi_tail = df['high'] - df['open'].combine(df['close'], max)
    pin_bar = (lo_tail > body * 2) & (lo_tail > hi_tail * 2)
    return vol_extreme & pin_bar

def vol_reversal_short(df):
    """극단 거래량 + 위꼬리 긴 핀바 → 강한 매도세."""
    vol_extreme = df['volume'] > df['vol_ma'] + df['volume'].rolling(20).std() * 2.5
    body    = abs(df['close'] - df['open'])
    lo_tail = df['open'].combine(df['close'], min) - df['low']
    hi_tail = df['high'] - df['open'].combine(df['close'], max)
    pin_bar = (hi_tail > body * 2) & (hi_tail > lo_tail * 2)
    return vol_extreme & pin_bar


# ── 전략 등록 ────────────────────────────────────────────────────────────────

STRATEGIES = {
    'turtle_20':        (turtle_long,          turtle_short,         'trend'),
    'ichimoku':         (ichi_long,             ichi_short,           'trend'),
    'vwap_revert':      (vwap_long,             vwap_short,           'mean'),
    'zscore_50':        (zscore_long,           zscore_short,         'mean'),
    'consec_3bar':      (consec_long,           consec_short,         'pattern'),
    'breakout_10':      (breakout_long,         breakout_short,       'trend'),
    'atr_squeeze':      (atr_squeeze_long,      atr_squeeze_short,    'volatility'),
    'pivot_24':         (pivot_long,            pivot_short,          'support'),
    'vol_pinbar':       (vol_reversal_long,     vol_reversal_short,   'pattern'),
}

TP_SL_LIST = [
    (0.03, 0.01),
    (0.05, 0.02),
    (0.04, 0.015),
    (0.06, 0.02),  # 추세 추종용 넓은 TP
]


# ── 백테스트 ─────────────────────────────────────────────────────────────────

def backtest(df, long_sig, short_sig, tp_pct, sl_pct, direction_filter=True):
    opens  = df['open'].to_numpy(float)
    highs  = df['high'].to_numpy(float)
    lows   = df['low'].to_numpy(float)
    closes = df['close'].to_numpy(float)
    ts     = df['timestamp'].to_numpy() if 'timestamp' in df.columns else df.index.to_numpy()

    ema200 = df['ema200'].to_numpy(float)
    la = long_sig.to_numpy(bool)
    sa = short_sig.to_numpy(bool)

    if direction_filter:
        # EMA200 방향 필터
        la = la & (closes > ema200)
        sa = sa & (closes < ema200)

    n = len(df)
    trades = []
    i = 0

    while i < n - 1:
        is_l = la[i] if i < len(la) else False
        is_s = sa[i] if i < len(sa) else False
        if not is_l and not is_s:
            i += 1
            continue

        d = 'long' if is_l else 'short'
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
            xp = closes[min(ei + TOUT, n - 1)]
            xr = 'timeout'
            xi = min(ei + TOUT, n - 1)

        pnl = ((xp - ep) / ep if d == 'long' else (ep - xp) / ep) - FEE * 2
        trades.append({'d': d, 'pnl': pnl * LEV, 'r': xr})
        i = xi + 1

    if len(trades) < MIN_TR:
        return None

    tdf = pd.DataFrame(trades)
    p   = tdf['pnl'].to_numpy()
    cum = np.cumsum(p)
    dd  = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0
    sh  = float(p.mean() / p.std() * np.sqrt(len(p))) if p.std() > 0 else 0

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

def run():
    all_rows = []

    for coin in COINS:
        print(f"\n{'='*60}  {coin.upper()}")

        # 기간별 데이터 미리 로드
        pdata = {}
        for pkey, (s, e, _) in PERIODS.items():
            df1m = load_period(coin, s, e)
            if df1m is None:
                continue
            pdata[pkey] = {}
            for tf in TFS:
                df = resample_df(df1m, tf)
                if len(df) < 100:
                    continue
                pdata[pkey][tf] = add_base(df)

        for tf in TFS:
            print(f"  [{tf}] ", end='', flush=True)
            found = 0

            for strat_name, (long_fn, short_fn, stype) in STRATEGIES.items():
                for tp_pct, sl_pct in TP_SL_LIST:
                    param = f"TP{int(tp_pct*100)}SL{int(sl_pct*100)}"
                    period_results = {}

                    for pkey, tf_data in pdata.items():
                        if tf not in tf_data:
                            continue
                        df = tf_data[tf]
                        try:
                            ls = long_fn(df).fillna(False)
                            ss = short_fn(df).fillna(False)
                        except Exception:
                            continue

                        r = backtest(df, ls, ss, tp_pct, sl_pct)
                        if r:
                            period_results[pkey] = r

                    if len(period_results) < 4:
                        continue

                    returns   = [r['total_return'] for r in period_results.values()]
                    sharpes   = [r['sharpe'] for r in period_results.values()]
                    positive  = sum(1 for r in returns if r > 0)
                    beat_bh   = sum(1 for p, r in period_results.items()
                                    if r['total_return'] > BUY_HOLD.get(p, {}).get(coin.upper(), 9999))

                    if positive < 4:
                        continue

                    row = {
                        'coin': coin.upper(), 'tf': tf,
                        'strategy': strat_name, 'type': stype, 'params': param,
                        'positive': positive, 'beat_bh': beat_bh,
                        'n_periods': len(period_results),
                        'avg_ret': round(np.mean(returns), 2),
                        'min_ret': round(np.min(returns), 2),
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

            print(f"→ {found}개")

    if not all_rows:
        print("\n조건 충족 없음")
        return

    df_out = pd.DataFrame(all_rows).sort_values(['beat_bh','avg_ret'], ascending=False)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n\n{'='*80}")
    print(f"  ★ {len(df_out)}개 후보 (양수기간 ≥ 4 / 6, 레버리지 1x)")
    print(f"{'='*80}\n")

    show = ['coin','tf','strategy','type','params','positive','beat_bh',
            'n_periods','avg_ret','min_ret','avg_sharpe','total_trades']
    print(df_out[show].head(25).to_string(index=False))

    print(f"\n\n  [TOP 5 — 기간별 상세]")
    for _, row in df_out.head(5).iterrows():
        print(f"\n  {row['coin']} {row['tf']} | {row['strategy']} ({row['type']}) | {row['params']}")
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
            t  = int(tr)  if tr  and not np.isnan(tr)  else 0
            l  = int(lng) if lng and not np.isnan(lng) else 0
            s  = int(sht) if sht and not np.isnan(sht) else 0
            print(f"    {beat} {pl:<14} {ret:+6.1f}%  vs B&H{bh:+.0f}%  "
                  f"승률{wr:.0f}%  롱{l}/숏{s}건")

    print(f"\n  저장: {OUT_CSV}")


if __name__ == '__main__':
    run()
