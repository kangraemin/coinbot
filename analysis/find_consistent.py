"""일관된 전략 탐색 — 여러 시장 국면에서 꾸준히 수익 나는 조합 찾기.

접근법:
  1. 베이스 신호 (BB, RSI, MACD, EMA, 캔들)
  2. 필터 조합: 거래량 스파이크 / 멀티TF 정배열 / ATR 기반 TP/SL
  3. 6개 기간 중 4개 이상 양수 수익인 조합만 추출
  4. 일관성 점수(consistency_score) 로 랭킹

필터 종류:
  - vol_spike   : 신호 봉 거래량 > 최근 20봉 평균의 N배
  - multi_tf    : 상위 TF(1h/4h)에서 EMA200 위 + 같은 방향 신호
  - atr_tp_sl   : ATR 기반 동적 TP/SL (고정 비율 대신)
  - rsi_confirm : RSI가 특정 범위일 때만 진입
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.config import DATA_DIR, OUTPUT_DIR, FEE_RATE, SLIPPAGE, TIMEOUT_BARS, RESAMPLE_RULES
from analysis.backtest_engine import run_backtest

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / "consistent_strategies.csv"

# ── 분석 기간 ─────────────────────────────────────────────────────────────────
PERIODS = {
    'recent_3m':   ('2025-12-02', '2026-03-02', '하락장(최근)'),
    'bear_2025q4': ('2025-10-01', '2026-01-01', '하락장(Q4)'),
    'bull_2025q3': ('2025-07-01', '2025-10-01', '상승(Q3)'),
    'bull_2024h2': ('2024-07-01', '2024-12-31', '불장(24H2)'),
    'bear_2024h1': ('2024-01-01', '2024-06-30', '횡보(24H1)'),
    'bull_2023':   ('2023-01-01', '2023-12-31', '회복(2023)'),
}

COINS    = ['btc', 'eth']
TFS      = ['15m', '1h']
LEVERAGE = 5
EMA_PERIOD = 200
MIN_TRADES = 8          # 최소 거래 수 (신뢰도 필터)
MIN_POSITIVE_PERIODS = 4  # 6기간 중 최소 양수 수익 기간


# ── 데이터 로더 ───────────────────────────────────────────────────────────────

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
    return r.reset_index().rename(columns={'timestamp':'timestamp'})


# ── 기술 지표 ─────────────────────────────────────────────────────────────────

def add_indicators(df):
    c = df['close']
    v = df['volume']
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(c, 14).rsi()
    # MACD
    macd = ta.trend.MACD(c)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    # EMA
    df['ema9']  = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df['ema21'] = ta.trend.EMAIndicator(c, 21).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(c, 50).ema_indicator()
    df['ema200']= ta.trend.EMAIndicator(c, 200).ema_indicator()
    # Bollinger
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid']   = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']  # 변동성
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], c, 14).average_true_range()
    # 거래량 MA
    df['vol_ma20'] = v.rolling(20).mean()
    df['vol_ratio'] = v / df['vol_ma20']
    # 스토캐스틱
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], c, 14, 3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    return df


# ── 신호 생성 함수들 ──────────────────────────────────────────────────────────

def sig_bb_bounce(df):
    """BB 하단 이탈 후 회복."""
    return (df['close'].shift(1) < df['bb_lower'].shift(1)) & (df['close'] >= df['bb_lower'])

def sig_rsi_oversold(df, thresh=35):
    """RSI 과매도 반등 — RSI가 thresh 이하로 내려갔다가 다시 올라오는 시점."""
    was_below = df['rsi'].shift(1) < thresh
    now_above = df['rsi'] >= thresh
    return was_below & now_above

def sig_macd_cross(df):
    """MACD 골든크로스."""
    return (df['macd'].shift(1) < df['macd_signal'].shift(1)) & (df['macd'] >= df['macd_signal'])

def sig_ema_cross(df):
    """EMA9 > EMA21 골든크로스."""
    return (df['ema9'].shift(1) < df['ema21'].shift(1)) & (df['ema9'] >= df['ema21'])

def sig_stoch_oversold(df):
    """스토캐스틱 과매도 반등 — K가 20 이하에서 D를 상향 돌파."""
    return (df['stoch_k'].shift(1) < df['stoch_d'].shift(1)) & \
           (df['stoch_k'] >= df['stoch_d']) & \
           (df['stoch_k'].shift(1) < 25)

def sig_bb_squeeze_breakout(df):
    """BB 밴드폭 수축(스퀴즈) 후 상향 돌파 — 변동성 폭발 직전."""
    narrow = df['bb_width'] < df['bb_width'].rolling(50).quantile(0.25)
    breakout = df['close'] > df['bb_mid']
    was_below = df['close'].shift(1) <= df['bb_mid'].shift(1)
    return narrow.shift(1) & was_below & breakout

def sig_ema_pullback(df):
    """EMA50 눌림목 — EMA50 위 추세에서 EMA50 터치 후 반등."""
    above_ema50 = df['close'] > df['ema50']
    touched = df['low'] <= df['ema50'] * 1.002  # EMA50 ±0.2% 이내 터치
    recovering = df['close'] > df['open']        # 양봉
    return above_ema50 & touched & recovering

BASE_SIGNALS = {
    'bb_bounce':          sig_bb_bounce,
    'rsi_bounce35':       sig_rsi_oversold,
    'macd_cross':         sig_macd_cross,
    'ema_cross_9_21':     sig_ema_cross,
    'stoch_oversold':     sig_stoch_oversold,
    'bb_squeeze_break':   sig_bb_squeeze_breakout,
    'ema50_pullback':     sig_ema_pullback,
}


# ── 필터 함수들 ───────────────────────────────────────────────────────────────

def filt_ema200(df, sig):
    """EMA200 위에 있을 때만."""
    return sig & (df['close'] > df['ema200'])

def filt_vol_spike(df, sig, mult=1.5):
    """거래량 스파이크 (평균의 mult배 이상)."""
    return sig & (df['vol_ratio'] >= mult)

def filt_rsi_range(df, sig, low=30, high=60):
    """RSI가 low~high 범위일 때만."""
    return sig & (df['rsi'] >= low) & (df['rsi'] <= high)

def filt_macd_positive(df, sig):
    """MACD 히스토그램이 증가 중 (모멘텀 개선)."""
    return sig & (df['macd_hist'] > df['macd_hist'].shift(1))

def filt_bb_not_too_wide(df, sig):
    """BB 밴드폭이 너무 넓지 않을 때 (과도한 변동성 회피) — 중간값 이하."""
    median_width = df['bb_width'].rolling(100).median()
    return sig & (df['bb_width'] <= median_width * 1.5)

FILTERS = {
    'ema200':        filt_ema200,
    'vol_spike':     filt_vol_spike,
    'rsi_range':     filt_rsi_range,
    'macd_pos':      filt_macd_positive,
    'bb_not_wide':   filt_bb_not_too_wide,
}

# 필터 조합 (단일 + 2중 조합만)
from itertools import combinations as _combs

FILTER_COMBOS = [
    ('ema200',),
    ('ema200', 'vol_spike'),
    ('ema200', 'rsi_range'),
    ('ema200', 'macd_pos'),
    ('ema200', 'bb_not_wide'),
    ('ema200', 'vol_spike', 'rsi_range'),
    ('ema200', 'vol_spike', 'macd_pos'),
    ('ema200', 'rsi_range', 'macd_pos'),
    ('ema200', 'vol_spike', 'bb_not_wide'),
]


# ── ATR 기반 백테스트 ─────────────────────────────────────────────────────────

def run_atr_backtest(df, signals, atr_tp_mult=3.0, atr_sl_mult=1.0):
    """ATR 기반 동적 TP/SL 백테스팅."""
    opens  = df['open'].to_numpy(float)
    highs  = df['high'].to_numpy(float)
    lows   = df['low'].to_numpy(float)
    closes = df['close'].to_numpy(float)
    atrs   = df['atr'].to_numpy(float)

    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].to_numpy()
    else:
        timestamps = df.index.to_numpy()

    sig = signals.to_numpy(bool)
    n = len(df)
    trades = []
    i = 0

    while i < n - 1:
        if not sig[i]:
            i += 1
            continue

        entry_idx = i + 1
        ep = opens[entry_idx] * (1 + SLIPPAGE)
        atr_val = atrs[i]
        if np.isnan(atr_val) or atr_val <= 0:
            i += 1
            continue

        tp_price = ep + atr_val * atr_tp_mult
        sl_price = ep - atr_val * atr_sl_mult

        exit_price = None
        exit_reason = None
        exit_idx = entry_idx

        for j in range(entry_idx, min(entry_idx + TIMEOUT_BARS + 1, n)):
            if lows[j] <= sl_price:
                exit_price, exit_reason, exit_idx = sl_price, 'sl', j
                break
            if highs[j] >= tp_price:
                exit_price, exit_reason, exit_idx = tp_price, 'tp', j
                break
            if j == min(entry_idx + TIMEOUT_BARS, n - 1):
                exit_price, exit_reason, exit_idx = closes[j], 'timeout', j
                break

        if exit_price is None:
            exit_price = closes[min(entry_idx + TIMEOUT_BARS, n - 1)]
            exit_reason = 'timeout'
            exit_idx = min(entry_idx + TIMEOUT_BARS, n - 1)

        pnl = (exit_price - ep) / ep - FEE_RATE * 2
        trades.append({
            'entry_time':    timestamps[entry_idx],
            'exit_time':     timestamps[exit_idx],
            'pnl_pct':       round(pnl, 6),
            'pnl_leveraged': round(pnl * LEVERAGE, 6),
            'exit_reason':   exit_reason,
        })
        i = exit_idx + 1

    if not trades:
        return pd.DataFrame(), {'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                                 'avg_return': 0, 'max_drawdown': 0, 'sharpe_ratio': 0}

    tdf = pd.DataFrame(trades)
    pnl = tdf['pnl_leveraged'].to_numpy()
    wins = (pnl > 0).sum()
    cum = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(cum) - cum))
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(len(pnl))) if pnl.std() > 0 else 0

    summary = {
        'total_trades': len(tdf),
        'win_rate':     round(wins / len(tdf), 4),
        'total_return': round(float(pnl.sum()), 4),
        'avg_return':   round(float(pnl.mean()), 4),
        'max_drawdown': round(dd, 4),
        'sharpe_ratio': round(sharpe, 4),
    }
    return tdf, summary


# ── 메인 탐색 ─────────────────────────────────────────────────────────────────

def run():
    all_rows = []

    # TP/SL 조합: 고정 + ATR
    FIXED_TP_SL = [(0.03, 0.01), (0.05, 0.02), (0.04, 0.015)]
    ATR_COMBOS  = [(3.0, 1.0), (4.0, 1.0), (3.0, 1.5)]

    total_combos = (len(COINS) * len(TFS) * len(BASE_SIGNALS) *
                    len(FILTER_COMBOS) * (len(FIXED_TP_SL) + len(ATR_COMBOS)))
    print(f"총 탐색 조합: ~{total_combos:,}개\n")

    for coin in COINS:
        print(f"\n{'='*60}")
        print(f"  {coin.upper()} 분석 중...")
        print(f"{'='*60}")

        # 기간별 데이터 미리 로드
        period_data = {}
        for pkey, (start, end, plabel) in PERIODS.items():
            df1m = load_period(coin, start, end)
            if df1m is None or len(df1m) < 1000:
                continue
            period_data[pkey] = {}
            for tf in TFS:
                df = resample_df(df1m, tf)
                if len(df) < 300:
                    continue
                df = add_indicators(df)
                period_data[pkey][tf] = df

        for tf in TFS:
            print(f"\n  [{tf}] 신호 × 필터 조합 탐색...", end='', flush=True)
            combo_count = 0

            for sig_name, sig_fn in BASE_SIGNALS.items():
                for filt_combo in FILTER_COMBOS:
                    filt_label = '+'.join(filt_combo)
                    combo_label = f"{sig_name}|{filt_label}"

                    for tp_sl_type, tp_sl_list in [('fixed', FIXED_TP_SL), ('atr', ATR_COMBOS)]:
                        for tp_sl in tp_sl_list:
                            if tp_sl_type == 'fixed':
                                tp_pct, sl_pct = tp_sl
                                param_label = f"TP{int(tp_pct*100)}SL{int(sl_pct*100)}"
                            else:
                                atr_tp, atr_sl = tp_sl
                                param_label = f"ATR{atr_tp}x{atr_sl}"

                            period_results = {}

                            for pkey, tf_data in period_data.items():
                                if tf not in tf_data:
                                    continue
                                df = tf_data[tf]

                                # 신호 생성
                                try:
                                    raw_sig = sig_fn(df).fillna(False)
                                except Exception:
                                    continue

                                # 필터 적용
                                sig = raw_sig.copy()
                                for fn in filt_combo:
                                    try:
                                        sig = FILTERS[fn](df, sig).fillna(False)
                                    except Exception:
                                        sig = pd.Series(False, index=df.index)
                                        break

                                if sig.sum() < MIN_TRADES:
                                    continue

                                # 백테스트
                                try:
                                    if tp_sl_type == 'fixed':
                                        _, summary = run_backtest(
                                            df, sig, tp_pct=tp_pct, sl_pct=sl_pct,
                                            leverage=LEVERAGE, fee_rate=FEE_RATE,
                                            slippage=SLIPPAGE, timeout_bars=TIMEOUT_BARS,
                                        )
                                    else:
                                        _, summary = run_atr_backtest(df, sig, atr_tp, atr_sl)
                                except Exception:
                                    continue

                                if summary['total_trades'] < MIN_TRADES:
                                    continue

                                period_results[pkey] = {
                                    'trades':  summary['total_trades'],
                                    'win_rate': round(summary['win_rate'] * 100, 1),
                                    'return':   round(summary['total_return'] * 100, 2),
                                    'sharpe':   summary['sharpe_ratio'],
                                }

                            if len(period_results) < MIN_POSITIVE_PERIODS:
                                continue

                            # 일관성 평가
                            positive_periods = sum(1 for r in period_results.values() if r['return'] > 0)
                            if positive_periods < MIN_POSITIVE_PERIODS:
                                continue

                            returns = [r['return'] for r in period_results.values()]
                            sharpes = [r['sharpe'] for r in period_results.values()]
                            avg_return = np.mean(returns)
                            avg_sharpe = np.mean(sharpes)
                            min_return = np.min(returns)
                            total_trades = sum(r['trades'] for r in period_results.values())

                            # 일관성 점수: 양수 기간 비율 × 평균 Sharpe
                            consistency = (positive_periods / len(period_results)) * avg_sharpe

                            row = {
                                'coin':             coin.upper(),
                                'timeframe':        tf,
                                'signal':           sig_name,
                                'filters':          filt_label,
                                'tp_sl_type':       tp_sl_type,
                                'params':           param_label,
                                'positive_periods': positive_periods,
                                'total_periods':    len(period_results),
                                'avg_return':       round(avg_return, 2),
                                'min_return':       round(min_return, 2),
                                'avg_sharpe':       round(avg_sharpe, 3),
                                'consistency':      round(consistency, 3),
                                'total_trades':     total_trades,
                            }
                            # 기간별 수익도 추가
                            for pkey in PERIODS:
                                if pkey in period_results:
                                    r = period_results[pkey]
                                    row[f'{pkey}_ret']    = r['return']
                                    row[f'{pkey}_wr']     = r['win_rate']
                                    row[f'{pkey}_trades'] = r['trades']
                                else:
                                    row[f'{pkey}_ret']    = None
                                    row[f'{pkey}_wr']     = None
                                    row[f'{pkey}_trades'] = None

                            all_rows.append(row)
                            combo_count += 1

            print(f" → {combo_count}개 후보 발견")

    if not all_rows:
        print("\n  6기간 중 4개 이상 양수 조합 없음")
        return

    df_out = pd.DataFrame(all_rows).sort_values('consistency', ascending=False)
    df_out.to_csv(OUT_CSV, index=False)

    # ── 결과 출력 ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(f"  ★ 일관된 전략 발견: {len(df_out)}개 후보 (양수 기간 ≥ {MIN_POSITIVE_PERIODS}/6)")
    print(f"{'='*90}")

    top = df_out.head(20)
    summary_cols = ['coin','timeframe','signal','filters','params',
                    'positive_periods','avg_return','min_return','avg_sharpe','consistency','total_trades']
    print("\n  [TOP 20 — 일관성 점수 순]")
    print(top[summary_cols].to_string(index=False))

    # 기간별 상세
    print(f"\n\n  [TOP 10 — 기간별 수익 상세]")
    period_cols = ['coin','tf','signal','filters','params']
    for _, row in df_out.head(10).iterrows():
        print(f"\n  {row['coin']} {row['timeframe']} | {row['signal']} | {row['filters']} | {row['params']}")
        print(f"  일관성:{row['consistency']:.3f}  양수:{row['positive_periods']}/{row['total_periods']}  "
              f"평균수익:{row['avg_return']:+.1f}%  최소수익:{row['min_return']:+.1f}%  "
              f"총거래:{row['total_trades']}건")
        for pkey, (_, _, plabel) in PERIODS.items():
            ret    = row.get(f'{pkey}_ret')
            wr     = row.get(f'{pkey}_wr')
            trades = row.get(f'{pkey}_trades')
            if ret is not None:
                flag = '✓' if ret > 0 else '✗'
                trades_int = int(trades) if trades is not None and not (isinstance(trades, float) and np.isnan(trades)) else 0
                wr_val = wr if wr is not None and not (isinstance(wr, float) and np.isnan(wr)) else 0.0
                print(f"    {flag} {plabel:<14} {ret:+6.1f}%  승률{wr_val:.0f}%  {trades_int}건")
            else:
                print(f"    — {plabel:<14} 데이터 없음")

    print(f"\n  결과 저장: {OUT_CSV}")


if __name__ == '__main__':
    run()
