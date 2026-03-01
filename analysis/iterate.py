"""그리디 반복 탐색 — 좋은 롱 진입 포인트 찾을 때까지 자동 이터레이션.

이터레이션 순서:
  iter1: 베이스라인 (TP=2%, SL=1%, 트렌드 필터 없음)
  iter2: EMA200 트렌드 필터 추가 (가격 > EMA200 시에만 진입)
  iter3: TP/SL 파라미터 스윕 (트렌드 필터 포함)
  iter4: 복합 신호 + 트렌드 필터 + 최적 TP/SL

결과는 analysis/output/iteration_results.csv 에 누적 저장.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import ta

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.config import COIN_NAMES, TIMEFRAMES, FEE_RATE, SLIPPAGE, TIMEOUT_BARS, OUTPUT_DIR
from analysis.data_loader import load_all_timeframes
from analysis.signals import STRATEGIES
from analysis.backtest_engine import run_backtest

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "iteration_results.csv"

# ── 이터레이션 파라미터 ──────────────────────────────────────────────────────

# iter3 스윕 대상 TP/SL 조합 (R:R 2:1 이상만)
TP_SL_COMBOS = [
    (0.015, 0.005),   # 3:1
    (0.020, 0.007),   # ~2.9:1
    (0.020, 0.010),   # 2:1  (베이스라인)
    (0.030, 0.010),   # 3:1
    (0.030, 0.015),   # 2:1
    (0.040, 0.015),   # ~2.7:1
    (0.050, 0.020),   # 2.5:1
]

LEVERAGE = 5
EMA_PERIOD = 200  # 트렌드 필터용

# ── 트렌드 필터 ──────────────────────────────────────────────────────────────

def apply_trend_filter(df: pd.DataFrame, signals: pd.Series) -> pd.Series:
    """EMA200 위에 있을 때만 진입 허용."""
    close = df['close'] if 'close' in df.columns else df['close']
    ema200 = ta.trend.EMAIndicator(close, window=EMA_PERIOD).ema_indicator()
    above_ema = close > ema200
    return signals & above_ema


# ── 복합 신호 생성 ────────────────────────────────────────────────────────────

def get_combined_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    """단일 + 2중 복합 신호 딕셔너리 반환."""
    single = {}
    for name, module in STRATEGIES.items():
        try:
            sig = module.detect(df)
            if sig.sum() > 0:
                single[name] = sig
        except Exception:
            pass

    combined = dict(single)
    for (n1, s1), (n2, s2) in combinations(single.items(), 2):
        key = f"{n1}+{n2}"
        merged = s1 & s2
        if merged.sum() > 0:
            combined[key] = merged

    return combined


# ── 단일 이터레이션 실행 ──────────────────────────────────────────────────────

def run_iteration(
    label: str,
    use_trend_filter: bool,
    tp_pct: float,
    sl_pct: float,
    use_combined: bool,
    coins: list[str] = COIN_NAMES,
    timeframes: list[str] = None,
) -> list[dict]:
    """하나의 이터레이션 설정으로 전체 코인/TF/전략 백테스팅."""
    tfs = timeframes or TIMEFRAMES
    rows = []

    for coin in coins:
        try:
            tf_data = load_all_timeframes(coin)
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue

        for tf in tfs:
            if tf not in tf_data:
                continue
            df = tf_data[tf]

            if use_combined:
                signals_dict = get_combined_signals(df)
            else:
                signals_dict = {}
                for name, module in STRATEGIES.items():
                    try:
                        sig = module.detect(df)
                        if sig.sum() > 0:
                            signals_dict[name] = sig
                    except Exception:
                        pass

            for strat_name, signals in signals_dict.items():
                if use_trend_filter:
                    signals = apply_trend_filter(df, signals)

                if signals.sum() == 0:
                    continue

                trades_df, summary = run_backtest(
                    df, signals,
                    tp_pct=tp_pct, sl_pct=sl_pct,
                    leverage=LEVERAGE,
                    fee_rate=FEE_RATE,
                    slippage=SLIPPAGE,
                    timeout_bars=TIMEOUT_BARS,
                )

                rows.append({
                    'iteration':     label,
                    'coin':          coin.upper(),
                    'timeframe':     tf,
                    'strategy':      strat_name,
                    'trend_filter':  use_trend_filter,
                    'tp_pct':        tp_pct,
                    'sl_pct':        sl_pct,
                    'signal_count':  int(signals.sum()),
                    **summary,
                })

    return rows


# ── 결과 저장 / 출력 ──────────────────────────────────────────────────────────

def save_results(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # 퍼센트 컬럼 변환
    for col in ['win_rate', 'total_return', 'avg_return', 'max_drawdown']:
        if col in df.columns:
            df[col] = (df[col] * 100).round(2)

    if CSV_PATH.exists():
        existing = pd.read_csv(CSV_PATH)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(CSV_PATH, index=False)
    return df


def print_top(df: pd.DataFrame, n: int = 15, min_trades: int = 5) -> None:
    filtered = df[df['total_trades'] >= min_trades].copy()
    if filtered.empty:
        print("  (결과 없음 — 거래 수 부족)")
        return
    top = filtered.sort_values('sharpe_ratio', ascending=False).head(n)
    cols = ['iteration', 'coin', 'timeframe', 'strategy',
            'trend_filter', 'tp_pct', 'sl_pct',
            'total_trades', 'win_rate', 'total_return', 'sharpe_ratio']
    cols = [c for c in cols if c in top.columns]
    print(top[cols].to_string(index=False))


def separator(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    all_rows = []

    # ── iter1: 베이스라인 ──────────────────────────────────────────────────
    separator("ITER 1 / 4  —  베이스라인 (TP=2%, SL=1%, 트렌드 필터 없음)")
    rows = run_iteration(
        label='iter1_baseline',
        use_trend_filter=False,
        tp_pct=0.020, sl_pct=0.010,
        use_combined=False,
    )
    all_rows.extend(rows)
    df_all = save_results(rows)
    print(f"  완료 — {len(rows)}건")
    print_top(df_all[df_all['iteration'] == 'iter1_baseline'])

    # ── iter2: EMA200 트렌드 필터 ─────────────────────────────────────────
    separator("ITER 2 / 4  —  EMA200 트렌드 필터 추가 (TP=2%, SL=1%)")
    rows = run_iteration(
        label='iter2_trend_filter',
        use_trend_filter=True,
        tp_pct=0.020, sl_pct=0.010,
        use_combined=False,
    )
    all_rows.extend(rows)
    df_all = save_results(rows)
    print(f"  완료 — {len(rows)}건")
    print_top(df_all[df_all['iteration'] == 'iter2_trend_filter'])

    # ── iter3: TP/SL 파라미터 스윕 (트렌드 필터 포함) ─────────────────────
    separator("ITER 3 / 4  —  TP/SL 파라미터 스윕 (트렌드 필터 포함)")
    sweep_rows = []
    for tp, sl in TP_SL_COMBOS:
        label = f"iter3_tp{int(tp*1000)}sl{int(sl*1000)}"
        print(f"  TP={tp*100:.1f}% SL={sl*100:.1f}%  (R:R {tp/sl:.1f}:1) ...", end=' ', flush=True)
        rows = run_iteration(
            label=label,
            use_trend_filter=True,
            tp_pct=tp, sl_pct=sl,
            use_combined=False,
        )
        sweep_rows.extend(rows)
        print(f"{len(rows)}건")

    all_rows.extend(sweep_rows)
    df_all = save_results(sweep_rows)
    iter3_df = df_all[df_all['iteration'].str.startswith('iter3')]
    print("\n  [iter3 Top 15 — Sharpe 기준]")
    print_top(iter3_df)

    # 최적 TP/SL 찾기
    best_row = iter3_df[iter3_df['total_trades'] >= 5].sort_values('sharpe_ratio', ascending=False).iloc[0]
    best_tp = best_row['tp_pct'] / 100
    best_sl = best_row['sl_pct'] / 100
    print(f"\n  ★ 최적 TP/SL: TP={best_tp*100:.1f}% SL={best_sl*100:.1f}%"
          f"  (R:R {best_tp/best_sl:.1f}:1)"
          f"  — Sharpe {best_row['sharpe_ratio']:.3f}")

    # ── iter4: 복합 신호 + 트렌드 필터 + 최적 TP/SL ──────────────────────
    separator(f"ITER 4 / 4  —  복합 신호 + 트렌드 필터 + TP={best_tp*100:.1f}% SL={best_sl*100:.1f}%")
    rows = run_iteration(
        label='iter4_combined_best',
        use_trend_filter=True,
        tp_pct=best_tp, sl_pct=best_sl,
        use_combined=True,
    )
    all_rows.extend(rows)
    df_all = save_results(rows)
    iter4_df = df_all[df_all['iteration'] == 'iter4_combined_best']
    print(f"  완료 — {len(rows)}건")
    print("\n  [iter4 Top 15 — Sharpe 기준]")
    print_top(iter4_df)

    # ── 전체 최종 랭킹 ────────────────────────────────────────────────────
    separator("최종 랭킹  —  전체 이터레이션 통합 (Sharpe 상위 20)")
    df_final = pd.read_csv(CSV_PATH)
    print_top(df_final, n=20)

    # 양의 총수익 + Sharpe > 0 필터링
    positive = df_final[
        (df_final['total_return'] > 0) &
        (df_final['sharpe_ratio'] > 0) &
        (df_final['total_trades'] >= 5)
    ].sort_values('sharpe_ratio', ascending=False)

    if positive.empty:
        print("\n  ⚠  양의 수익 + Sharpe > 0 조합 없음 — 하락장 영향")
    else:
        separator(f"★ 수익 양수 조합 ({len(positive)}건)")
        print_top(positive, n=30, min_trades=5)

    print(f"\n  결과 저장: {CSV_PATH}")


if __name__ == '__main__':
    main()
