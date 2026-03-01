"""기간별 분석 — 여러 시장 국면에서 전략 성과 + 거래량 지표 포함.

분석 기간:
  bear_2025q4 : 2025-10-01 ~ 2026-01-01  (최근 하락장)
  recent_3m   : 2025-12-02 ~ 2026-03-02  (직전 분석과 동일)
  bull_2025q3 : 2025-07-01 ~ 2025-10-01  (여름 상승장)
  bull_2024h2 : 2024-07-01 ~ 2024-12-31  (2024 하반기 불장)
  bear_2024h1 : 2024-01-01 ~ 2024-06-30  (2024 상반기)
  bull_2023   : 2023-01-01 ~ 2023-12-31  (2023 회복장)

집중 전략: EMA200 트렌드 필터 + TP=5% SL=2% (최적 파라미터)
타임프레임: 15m, 1h (핵심 TF만)
코인: BTC, ETH
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import ta

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.config import DATA_DIR, OUTPUT_DIR, FEE_RATE, SLIPPAGE, TIMEOUT_BARS, RESAMPLE_RULES
from analysis.signals import STRATEGIES
from analysis.backtest_engine import run_backtest

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "period_analysis.csv"

# ── 분석 설정 ─────────────────────────────────────────────────────────────────

PERIODS = {
    'recent_3m':   ('2025-12-02', '2026-03-02', '하락장 (최근 3개월)'),
    'bear_2025q4': ('2025-10-01', '2026-01-01', '하락장 (2025 Q4)'),
    'bull_2025q3': ('2025-07-01', '2025-10-01', '상승/횡보 (2025 Q3)'),
    'bull_2024h2': ('2024-07-01', '2024-12-31', '불장 (2024 하반기)'),
    'bear_2024h1': ('2024-01-01', '2024-06-30', '횡보/조정 (2024 상반기)'),
    'bull_2023':   ('2023-01-01', '2023-12-31', '회복장 (2023 전체)'),
}

COINS    = ['btc', 'eth']
TFS      = ['15m', '1h']          # 핵심 TF만
TP_PCT   = 0.050
SL_PCT   = 0.020
LEVERAGE = 5
EMA_PERIOD = 200


# ── 데이터 로더 (기간 지정 가능) ──────────────────────────────────────────────

def load_period(coin: str, start: str, end: str) -> pd.DataFrame:
    """지정 기간의 1분봉 데이터 로드."""
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    years = range(start_dt.year, end_dt.year + 1)
    files = []
    for yr in years:
        f = DATA_DIR / f"{coin}_1m_{yr}.parquet"
        if f.exists():
            files.append(f)

    if not files:
        raise FileNotFoundError(f"{coin} 데이터 없음: {start}~{end}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp')
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df[(df.index >= start_dt) & (df.index < end_dt)]


def resample_df(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = RESAMPLE_RULES.get(tf, tf)
    resampled = df.resample(rule).agg({
        'open': 'first', 'high': 'max',
        'low': 'min', 'close': 'last', 'volume': 'sum',
    }).dropna()
    return resampled.reset_index().rename(columns={'timestamp': 'timestamp'})


# ── 트렌드 필터 ───────────────────────────────────────────────────────────────

def apply_trend_filter(df: pd.DataFrame, signals: pd.Series) -> pd.Series:
    ema200 = ta.trend.EMAIndicator(df['close'], window=EMA_PERIOD).ema_indicator()
    return signals & (df['close'] > ema200)


# ── 거래량 지표 계산 ──────────────────────────────────────────────────────────

def volume_metrics(df: pd.DataFrame, signals: pd.Series) -> dict:
    """신호 발생 시점의 거래량 특성."""
    if signals.sum() == 0:
        return {'vol_ratio': None, 'avg_vol_usd_m': None, 'total_vol_usd_b': None}

    # 전체 평균 대비 신호 봉 거래량 비율
    avg_vol = df['volume'].mean()
    if 'timestamp' in df.columns:
        sig_vol = df.loc[signals.values, 'volume'].mean() if signals.sum() > 0 else 0
    else:
        sig_vol = df['volume'][signals].mean() if signals.sum() > 0 else 0

    vol_ratio = sig_vol / avg_vol if avg_vol > 0 else None

    # USDT 거래량 (volume × close 근사)
    vol_usd = df['volume'] * df['close']
    avg_vol_usd_m  = vol_usd.mean() / 1e6       # 봉당 평균 (백만 USD)
    total_vol_usd_b = vol_usd.sum() / 1e9        # 기간 합계 (십억 USD)

    return {
        'vol_ratio':      round(vol_ratio, 2) if vol_ratio else None,
        'avg_vol_usd_m':  round(avg_vol_usd_m, 2),
        'total_vol_usd_b': round(total_vol_usd_b, 1),
    }


# ── 시장 상태 요약 ────────────────────────────────────────────────────────────

def market_summary(df: pd.DataFrame) -> dict:
    """기간 내 시장 상태 요약."""
    if df.empty:
        return {}
    start_price = df['close'].iloc[0]
    end_price   = df['close'].iloc[-1]
    price_chg   = (end_price - start_price) / start_price * 100
    max_price   = df['high'].max()
    min_price   = df['low'].min()
    drawdown    = (min_price - max_price) / max_price * 100

    return {
        'start_price': round(start_price, 1),
        'end_price':   round(end_price, 1),
        'price_chg_pct': round(price_chg, 1),
        'max_drawdown_pct': round(drawdown, 1),
    }


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def run():
    all_rows = []

    for period_key, (start, end, period_label) in PERIODS.items():
        print(f"\n{'='*70}")
        print(f"  {period_label}  ({start} ~ {end})")
        print(f"{'='*70}")

        for coin in COINS:
            try:
                df_1m = load_period(coin, start, end)
            except FileNotFoundError as e:
                print(f"  [{coin.upper()}] 데이터 없음: {e}")
                continue

            # 1h 데이터로 시장 요약
            df_1h_raw = resample_df(df_1m, '1h')
            mkt = market_summary(df_1h_raw)
            print(f"\n  [{coin.upper()}]  {start_price_str(mkt)} → {end_price_str(mkt)}"
                  f"  ({mkt.get('price_chg_pct', '?'):+.1f}%)"
                  f"  MDD {mkt.get('max_drawdown_pct', '?'):.1f}%")

            for tf in TFS:
                df = resample_df(df_1m, tf)
                if len(df) < 300:
                    print(f"  [{tf}] 데이터 부족 ({len(df)}봉), 스킵")
                    continue

                for strat_name, module in STRATEGIES.items():
                    try:
                        raw_signals = module.detect(df)
                    except Exception as e:
                        continue

                    # 트렌드 필터 적용
                    signals = apply_trend_filter(df, raw_signals)

                    raw_cnt = int(raw_signals.sum())
                    filtered_cnt = int(signals.sum())

                    if filtered_cnt == 0:
                        continue

                    trades_df, summary = run_backtest(
                        df, signals,
                        tp_pct=TP_PCT, sl_pct=SL_PCT,
                        leverage=LEVERAGE,
                        fee_rate=FEE_RATE,
                        slippage=SLIPPAGE,
                        timeout_bars=TIMEOUT_BARS,
                    )

                    vol = volume_metrics(df, signals)

                    row = {
                        'period':           period_key,
                        'period_label':     period_label,
                        'coin':             coin.upper(),
                        'timeframe':        tf,
                        'strategy':         strat_name,
                        'start':            start,
                        'end':              end,
                        'price_chg_pct':    mkt.get('price_chg_pct'),
                        'market_mdd_pct':   mkt.get('max_drawdown_pct'),
                        'raw_signals':      raw_cnt,
                        'filtered_signals': filtered_cnt,
                        'total_trades':     summary['total_trades'],
                        'win_rate':         round(summary['win_rate'] * 100, 2),
                        'total_return':     round(summary['total_return'] * 100, 2),
                        'avg_return':       round(summary['avg_return'] * 100, 3),
                        'max_drawdown':     round(summary['max_drawdown'] * 100, 2),
                        'sharpe_ratio':     summary['sharpe_ratio'],
                        'vol_ratio':        vol['vol_ratio'],
                        'avg_vol_usd_m':    vol['avg_vol_usd_m'],
                        'total_vol_usd_b':  vol['total_vol_usd_b'],
                    }
                    all_rows.append(row)

    if not all_rows:
        print("\n결과 없음")
        return

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(CSV_PATH, index=False)

    # ── 전략별 기간 성과 피벗 ────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("  전략별 × 기간별 승률 피벗 (트렌드 필터 + TP=5% SL=2%)")
    print(f"{'='*90}")

    key_cols = ['coin', 'timeframe', 'strategy']
    for coin in COINS:
        for tf in TFS:
            sub = df_out[(df_out['coin'] == coin.upper()) & (df_out['timeframe'] == tf)]
            if sub.empty:
                continue
            pivot_wr = sub.pivot_table(
                index='strategy',
                columns='period',
                values='win_rate',
                aggfunc='first',
            ).round(1)
            pivot_tr = sub.pivot_table(
                index='strategy',
                columns='period',
                values='total_return',
                aggfunc='first',
            ).round(1)
            pivot_cnt = sub.pivot_table(
                index='strategy',
                columns='period',
                values='total_trades',
                aggfunc='first',
            )

            period_order = list(PERIODS.keys())
            wr_cols  = [c for c in period_order if c in pivot_wr.columns]
            tr_cols  = [c for c in period_order if c in pivot_tr.columns]

            print(f"\n  [{coin.upper()} {tf}] — 승률(%) / 총수익(%) / 거래수")
            print(f"  {'전략':<22} ", end='')
            for p in wr_cols:
                label = PERIODS[p][2][:10]
                print(f"  {label:<14}", end='')
            print()
            print(f"  {'-'*22} ", end='')
            for _ in wr_cols:
                print(f"  {'-'*14}", end='')
            print()

            for strat in pivot_wr.index:
                print(f"  {strat:<22} ", end='')
                for p in wr_cols:
                    wr  = pivot_wr.loc[strat, p] if p in pivot_wr.columns else None
                    tr  = pivot_tr.loc[strat, p] if p in pivot_tr.columns else None
                    raw_cnt_val = pivot_cnt.loc[strat, p] if p in pivot_cnt.columns else None
                    cnt = int(raw_cnt_val) if raw_cnt_val is not None and not pd.isna(raw_cnt_val) else 0
                    if wr is None:
                        print(f"  {'—':^14}", end='')
                    else:
                        cell = f"{wr:.0f}%/{tr:+.0f}%/{cnt}"
                        print(f"  {cell:<14}", end='')
                print()

    # ── 수익 양수 전체 랭킹 ──────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("  ★ 수익 양수 + Sharpe > 0 전체 랭킹 (거래수 ≥ 8)")
    print(f"{'='*90}")

    positive = df_out[
        (df_out['total_return'] > 0) &
        (df_out['sharpe_ratio'] > 0) &
        (df_out['total_trades'] >= 8)
    ].sort_values('sharpe_ratio', ascending=False)

    if positive.empty:
        print("  (없음)")
    else:
        show_cols = [
            'period_label', 'coin', 'timeframe', 'strategy',
            'total_trades', 'win_rate', 'total_return', 'sharpe_ratio',
            'price_chg_pct', 'vol_ratio', 'avg_vol_usd_m',
        ]
        print(positive[show_cols].to_string(index=False))

    print(f"\n  결과 저장: {CSV_PATH}")


def start_price_str(mkt):
    return f"${mkt.get('start_price', '?'):,.0f}"

def end_price_str(mkt):
    return f"${mkt.get('end_price', '?'):,.0f}"


if __name__ == '__main__':
    run()
