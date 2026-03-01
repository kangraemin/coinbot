"""롱/숏 혼합 전략 탐색 — EMA200 기준으로 방향 전환.

EMA200 위  → 롱 신호 탐색
EMA200 아래 → 숏 신호 탐색 (롱 신호의 역방향)

비교 기준: 바이앤홀드 (레버리지 없음) vs 전략 (레버리지 1x, 공정 비교)
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.config import DATA_DIR, OUTPUT_DIR, FEE_RATE, SLIPPAGE, TIMEOUT_BARS, RESAMPLE_RULES

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "long_short_results.csv"

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

COINS    = ['btc', 'eth']
TFS      = ['15m', '1h']
LEVERAGE = 1   # 공정 비교를 위해 레버리지 1x
FEE      = 0.0005
SLIP     = 0.0003
TIMEOUT  = 48
EMA_P    = 200
MIN_TRADES = 5


# ── 데이터 ────────────────────────────────────────────────────────────────────

def load_period(coin, start, end):
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    years = range(start_dt.year, end_dt.year + 1)
    files = [DATA_DIR / f"{coin}_1m_{yr}.parquet" for yr in years
             if (DATA_DIR / f"{coin}_1m_{yr}.parquet").exists()]
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp')
    df = df[['open','high','low','close','volume']]
    return df[(df.index >= start_dt) & (df.index < end_dt)]

def resample_df(df, tf):
    rule = RESAMPLE_RULES.get(tf, tf)
    r = df.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    return r.reset_index()

def add_indicators(df):
    c = df['close']
    df['ema200']     = ta.trend.EMAIndicator(c, 200).ema_indicator()
    df['ema50']      = ta.trend.EMAIndicator(c, 50).ema_indicator()
    df['ema21']      = ta.trend.EMAIndicator(c, 21).ema_indicator()
    df['ema9']       = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df['rsi']        = ta.momentum.RSIIndicator(c, 14).rsi()
    macd             = ta.trend.MACD(c)
    df['macd']       = macd.macd()
    df['macd_sig']   = macd.macd_signal()
    df['macd_hist']  = macd.macd_diff()
    bb               = ta.volatility.BollingerBands(c, 20, 2)
    df['bb_upper']   = bb.bollinger_hband()
    df['bb_lower']   = bb.bollinger_lband()
    df['bb_mid']     = bb.bollinger_mavg()
    df['bb_width']   = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['atr']        = ta.volatility.AverageTrueRange(df['high'], df['low'], c, 14).average_true_range()
    stoch            = ta.momentum.StochasticOscillator(df['high'], df['low'], c, 14, 3)
    df['stoch_k']    = stoch.stoch()
    df['stoch_d']    = stoch.stoch_signal()
    df['vol_ma20']   = df['volume'].rolling(20).mean()
    df['vol_ratio']  = df['volume'] / df['vol_ma20']
    return df


# ── 롱 신호 ──────────────────────────────────────────────────────────────────

def long_bb_bounce(df):
    return (df['close'].shift(1) < df['bb_lower'].shift(1)) & (df['close'] >= df['bb_lower'])

def long_rsi_bounce(df):
    return (df['rsi'].shift(1) < 35) & (df['rsi'] >= 35)

def long_macd_cross(df):
    return (df['macd'].shift(1) < df['macd_sig'].shift(1)) & (df['macd'] >= df['macd_sig'])

def long_ema_cross(df):
    return (df['ema9'].shift(1) < df['ema21'].shift(1)) & (df['ema9'] >= df['ema21'])

def long_stoch_oversold(df):
    return ((df['stoch_k'].shift(1) < df['stoch_d'].shift(1)) &
            (df['stoch_k'] >= df['stoch_d']) &
            (df['stoch_k'].shift(1) < 25))

def long_bb_squeeze(df):
    narrow   = df['bb_width'] < df['bb_width'].rolling(50).quantile(0.25)
    was_below = df['close'].shift(1) <= df['bb_mid'].shift(1)
    breakout  = df['close'] > df['bb_mid']
    return narrow.shift(1) & was_below & breakout

def long_ema50_pullback(df):
    return ((df['close'] > df['ema50']) &
            (df['low'] <= df['ema50'] * 1.002) &
            (df['close'] > df['open']))


# ── 숏 신호 (롱의 거울) ──────────────────────────────────────────────────────

def short_bb_reject(df):
    """BB 상단 이탈 후 회복 실패 → 숏."""
    return (df['close'].shift(1) > df['bb_upper'].shift(1)) & (df['close'] <= df['bb_upper'])

def short_rsi_drop(df):
    """RSI 과매수 이탈."""
    return (df['rsi'].shift(1) > 65) & (df['rsi'] <= 65)

def short_macd_cross(df):
    """MACD 데드크로스."""
    return (df['macd'].shift(1) > df['macd_sig'].shift(1)) & (df['macd'] <= df['macd_sig'])

def short_ema_cross(df):
    """EMA9 데드크로스."""
    return (df['ema9'].shift(1) > df['ema21'].shift(1)) & (df['ema9'] <= df['ema21'])

def short_stoch_overbought(df):
    """스토캐스틱 과매수 하락."""
    return ((df['stoch_k'].shift(1) > df['stoch_d'].shift(1)) &
            (df['stoch_k'] <= df['stoch_d']) &
            (df['stoch_k'].shift(1) > 75))

def short_bb_squeeze(df):
    """BB 스퀴즈 하향 돌파."""
    narrow    = df['bb_width'] < df['bb_width'].rolling(50).quantile(0.25)
    was_above = df['close'].shift(1) >= df['bb_mid'].shift(1)
    breakdown = df['close'] < df['bb_mid']
    return narrow.shift(1) & was_above & breakdown

def short_ema50_reject(df):
    """EMA50 저항 → 숏."""
    return ((df['close'] < df['ema50']) &
            (df['high'] >= df['ema50'] * 0.998) &
            (df['close'] < df['open']))


SIGNAL_PAIRS = {
    'bb_bounce':      (long_bb_bounce,      short_bb_reject),
    'rsi':            (long_rsi_bounce,      short_rsi_drop),
    'macd_cross':     (long_macd_cross,      short_macd_cross),
    'ema_cross':      (long_ema_cross,       short_ema_cross),
    'stoch':          (long_stoch_oversold,  short_stoch_overbought),
    'bb_squeeze':     (long_bb_squeeze,      short_bb_squeeze),
    'ema50_pullback': (long_ema50_pullback,  short_ema50_reject),
}


# ── 백테스트 엔진 (롱/숏 통합) ───────────────────────────────────────────────

def backtest_ls(df, long_sig, short_sig, tp_pct, sl_pct):
    """롱/숏 모두 처리하는 백테스트."""
    opens  = df['open'].to_numpy(float)
    highs  = df['high'].to_numpy(float)
    lows   = df['low'].to_numpy(float)
    closes = df['close'].to_numpy(float)
    ts     = df['timestamp'].to_numpy() if 'timestamp' in df.columns else df.index.to_numpy()

    long_arr  = long_sig.to_numpy(bool)
    short_arr = short_sig.to_numpy(bool)
    n = len(df)
    trades = []
    i = 0

    while i < n - 1:
        is_long  = long_arr[i]
        is_short = short_arr[i]

        if not is_long and not is_short:
            i += 1
            continue

        direction = 'long' if is_long else 'short'
        ei = i + 1
        ep = opens[ei] * (1 + SLIP) if direction == 'long' else opens[ei] * (1 - SLIP)
        tp = ep * (1 + tp_pct) if direction == 'long' else ep * (1 - tp_pct)
        sl = ep * (1 - sl_pct) if direction == 'long' else ep * (1 + sl_pct)

        exit_price = None
        exit_reason = None
        exit_idx = ei

        for j in range(ei, min(ei + TIMEOUT + 1, n)):
            if direction == 'long':
                if lows[j] <= sl:
                    exit_price, exit_reason, exit_idx = sl, 'sl', j; break
                if highs[j] >= tp:
                    exit_price, exit_reason, exit_idx = tp, 'tp', j; break
            else:
                if highs[j] >= sl:
                    exit_price, exit_reason, exit_idx = sl, 'sl', j; break
                if lows[j] <= tp:
                    exit_price, exit_reason, exit_idx = tp, 'tp', j; break
            if j == min(ei + TIMEOUT, n - 1):
                exit_price, exit_reason, exit_idx = closes[j], 'timeout', j; break

        if exit_price is None:
            exit_price = closes[min(ei + TIMEOUT, n - 1)]
            exit_reason = 'timeout'
            exit_idx = min(ei + TIMEOUT, n - 1)

        if direction == 'long':
            pnl = (exit_price - ep) / ep - FEE * 2
        else:
            pnl = (ep - exit_price) / ep - FEE * 2

        trades.append({'direction': direction, 'pnl': round(pnl * LEVERAGE, 6),
                       'exit_reason': exit_reason})
        i = exit_idx + 1

    if not trades:
        return None

    tdf = pd.DataFrame(trades)
    pnl = tdf['pnl'].to_numpy()
    wins = (pnl > 0).sum()
    cum  = np.cumsum(pnl)
    dd   = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) > 0 else 0
    sh   = float(pnl.mean() / pnl.std() * np.sqrt(len(pnl))) if pnl.std() > 0 else 0

    long_trades  = tdf[tdf['direction'] == 'long']
    short_trades = tdf[tdf['direction'] == 'short']

    return {
        'total_trades': len(tdf),
        'long_trades':  len(long_trades),
        'short_trades': len(short_trades),
        'win_rate':     round(wins / len(tdf) * 100, 1),
        'total_return': round(float(pnl.sum() * 100), 2),
        'avg_return':   round(float(pnl.mean() * 100), 3),
        'max_drawdown': round(dd * 100, 2),
        'sharpe':       round(sh, 3),
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def run():
    TP_SL_LIST = [(0.03, 0.01), (0.05, 0.02), (0.04, 0.015)]
    all_rows = []

    for coin in COINS:
        print(f"\n{'='*60}  {coin.upper()}")

        period_data = {}
        for pkey, (start, end, _) in PERIODS.items():
            df1m = load_period(coin, start, end)
            if df1m is None:
                continue
            period_data[pkey] = {}
            for tf in TFS:
                df = resample_df(df1m, tf)
                if len(df) < 200:
                    continue
                period_data[pkey][tf] = add_indicators(df)

        for tf in TFS:
            print(f"  [{tf}] 탐색 중...", end='', flush=True)
            found = 0

            for sig_name, (long_fn, short_fn) in SIGNAL_PAIRS.items():
                for tp_pct, sl_pct in TP_SL_LIST:
                    param = f"TP{int(tp_pct*100)}SL{int(sl_pct*100)}"

                    period_results = {}

                    for pkey, tf_data in period_data.items():
                        if tf not in tf_data:
                            continue
                        df = tf_data[tf]

                        try:
                            l_raw = long_fn(df).fillna(False)
                            s_raw = short_fn(df).fillna(False)
                        except Exception:
                            continue

                        # EMA200 방향 필터
                        above = df['close'] > df['ema200']
                        long_sig  = l_raw & above
                        short_sig = s_raw & ~above

                        # 거래량 필터
                        vol_ok = df['vol_ratio'] >= 1.3
                        long_sig  = long_sig & vol_ok
                        short_sig = short_sig & vol_ok

                        if (long_sig.sum() + short_sig.sum()) < MIN_TRADES:
                            continue

                        result = backtest_ls(df, long_sig, short_sig, tp_pct, sl_pct)
                        if result and result['total_trades'] >= MIN_TRADES:
                            period_results[pkey] = result

                    if len(period_results) < 4:
                        continue

                    returns = [r['total_return'] for r in period_results.values()]
                    sharpes = [r['sharpe'] for r in period_results.values()]
                    positive = sum(1 for r in returns if r > 0)

                    if positive < 4:
                        continue

                    bh_vals = [BUY_HOLD[p][coin.upper()] for p in period_results if p in BUY_HOLD]
                    beat_bh = sum(1 for p, r in period_results.items()
                                  if r['total_return'] > BUY_HOLD.get(p, {}).get(coin.upper(), 9999))

                    row = {
                        'coin': coin.upper(), 'tf': tf,
                        'signal': sig_name, 'params': param,
                        'positive_periods': positive,
                        'beat_bh': beat_bh,
                        'total_periods': len(period_results),
                        'avg_return': round(np.mean(returns), 2),
                        'min_return': round(np.min(returns), 2),
                        'avg_sharpe': round(np.mean(sharpes), 3),
                        'total_trades': sum(r['total_trades'] for r in period_results.values()),
                    }
                    for pkey, (_, _, plabel) in PERIODS.items():
                        r = period_results.get(pkey)
                        row[f'{pkey}_ret']    = r['total_return'] if r else None
                        row[f'{pkey}_wr']     = r['win_rate'] if r else None
                        row[f'{pkey}_trades'] = r['total_trades'] if r else None
                        row[f'{pkey}_long']   = r['long_trades'] if r else None
                        row[f'{pkey}_short']  = r['short_trades'] if r else None

                    all_rows.append(row)
                    found += 1

            print(f" → {found}개 후보")

    if not all_rows:
        print("\n조건 충족 전략 없음")
        return

    df_out = pd.DataFrame(all_rows).sort_values(['beat_bh', 'avg_return'], ascending=False)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\n\n{'='*80}")
    print(f"  ★ 발견: {len(df_out)}개 후보 (양수기간 ≥ 4, 레버리지 1x 기준)")
    print(f"{'='*80}")

    print("\n  [전체 랭킹 — B&H 초과 기간 순]")
    show = ['coin','tf','signal','params','positive_periods','beat_bh',
            'total_periods','avg_return','min_return','avg_sharpe','total_trades']
    print(df_out[show].head(20).to_string(index=False))

    print(f"\n\n  [TOP 8 — 기간별 상세 (레버리지 1x vs B&H 레버리지 없음)]")
    for _, row in df_out.head(8).iterrows():
        print(f"\n  {row['coin']} {row['tf']} | {row['signal']} | {row['params']}")
        print(f"  양수:{row['positive_periods']}/{row['total_periods']}  "
              f"B&H초과:{row['beat_bh']}기간  "
              f"평균수익:{row['avg_return']:+.1f}%  최소:{row['min_return']:+.1f}%  "
              f"Sharpe:{row['avg_sharpe']:.3f}  총거래:{row['total_trades']}건")
        for pkey, (_, _, plabel) in PERIODS.items():
            ret    = row.get(f'{pkey}_ret')
            wr     = row.get(f'{pkey}_wr')
            trades = row.get(f'{pkey}_trades')
            lng    = row.get(f'{pkey}_long')
            sht    = row.get(f'{pkey}_short')
            bh     = BUY_HOLD.get(pkey, {}).get(row['coin'])
            if ret is None or (isinstance(ret, float) and np.isnan(ret)):
                continue
            beat = '✓' if bh is not None and ret > bh else '✗'
            bh_str = f"B&H{bh:+.0f}%" if bh is not None else ""
            t = int(trades) if trades and not np.isnan(trades) else 0
            l = int(lng) if lng and not np.isnan(lng) else 0
            s = int(sht) if sht and not np.isnan(sht) else 0
            print(f"    {beat} {plabel:<14} 전략{ret:+6.1f}%  {bh_str:<10} "
                  f"승률{wr:.0f}%  롱{l}/숏{s}건")

    print(f"\n  저장: {OUT_CSV}")


if __name__ == '__main__':
    run()
