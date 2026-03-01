"""진입 포인트 탐색 & 백테스팅 CLI.

사용법:
  python -m analysis.main                          # 전체 (BTC+ETH, 모든 전략, 모든 TF)
  python -m analysis.main --symbol btc            # BTC만
  python -m analysis.main --strategy rsi_oversold  # 특정 전략
  python -m analysis.main --timeframe 1h           # 특정 타임프레임
  python -m analysis.main --leverage 7             # 레버리지 7x
  python -m analysis.main --report html            # HTML 리포트 생성
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.config import COIN_NAMES, TIMEFRAMES, TP_PCT, SL_PCT, FEE_RATE, SLIPPAGE, TIMEOUT_BARS
from analysis.data_loader import load_all_timeframes
from analysis.signals import STRATEGIES
from analysis.backtest_engine import run_backtest
from analysis.feature_analyzer import compare_combined_signals, compare_timeframes
from analysis.report import print_summary, print_comparison_table, generate_html_report


def parse_args():
    parser = argparse.ArgumentParser(
        description='BTC/ETH 롱 진입 포인트 탐색 & 백테스팅'
    )
    parser.add_argument('--symbol', type=str, default=None,
                        help='코인 (btc, eth). 기본: 전체')
    parser.add_argument('--strategy', type=str, default=None,
                        help=f'전략 ({", ".join(STRATEGIES.keys())}). 기본: 전체')
    parser.add_argument('--timeframe', type=str, default=None,
                        help=f'타임프레임 ({", ".join(TIMEFRAMES)}). 기본: 전체')
    parser.add_argument('--leverage', type=int, default=5,
                        help='레버리지 (기본: 5)')
    parser.add_argument('--report', type=str, default='terminal',
                        choices=['terminal', 'html'],
                        help='출력 형식 (terminal/html). 기본: terminal')
    parser.add_argument('--combined', action='store_true',
                        help='복합 신호 분석 포함')
    return parser.parse_args()


def run(args):
    coins = [args.symbol.lower()] if args.symbol else COIN_NAMES
    strategies = {args.strategy: STRATEGIES[args.strategy]} if args.strategy else STRATEGIES
    timeframes = [args.timeframe] if args.timeframe else TIMEFRAMES

    if args.strategy and args.strategy not in STRATEGIES:
        print(f"오류: 알 수 없는 전략 '{args.strategy}'. 사용 가능: {list(STRATEGIES.keys())}")
        sys.exit(1)

    all_results = []
    trades_by_label = {}

    for coin in coins:
        print(f"\n[{coin.upper()}] 데이터 로드 중...")
        try:
            tf_data = load_all_timeframes(coin)
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

        for tf in timeframes:
            if tf not in tf_data:
                print(f"  [{tf}] 데이터 없음, 스킵")
                continue

            df = tf_data[tf]
            print(f"  [{tf}] {len(df)}봉 로드됨")

            for strat_name, module in strategies.items():
                try:
                    signals = module.detect(df)
                except Exception as e:
                    print(f"  [{strat_name}] 신호 계산 실패: {e}")
                    continue

                sig_count = int(signals.sum())
                if sig_count == 0:
                    print(f"  [{strat_name}] 신호 없음")
                    continue

                trades_df, summary = run_backtest(
                    df, signals,
                    tp_pct=TP_PCT, sl_pct=SL_PCT,
                    leverage=args.leverage,
                    fee_rate=FEE_RATE,
                    slippage=SLIPPAGE,
                    timeout_bars=TIMEOUT_BARS,
                )

                label = f"{coin.upper()} {tf} {strat_name}"
                print_summary(coin, tf, strat_name, summary, trades_df)

                row = {
                    'coin':     coin.upper(),
                    'timeframe': tf,
                    'strategy': strat_name,
                    'signal_count': sig_count,
                    **summary,
                }
                all_results.append(row)
                if not trades_df.empty:
                    trades_by_label[label] = trades_df

        # 복합 신호 분석
        if args.combined:
            print(f"\n  [{coin.upper()}] 복합 신호 분석...")
            for tf in timeframes:
                if tf not in tf_data:
                    continue
                df = tf_data[tf]
                combined_df = compare_combined_signals(df, leverage=args.leverage)
                combined_df.insert(0, 'coin', coin.upper())
                combined_df.insert(1, 'timeframe', tf)
                print_comparison_table(
                    combined_df.to_dict('records'),
                    title=f"복합 신호 — {coin.upper()} {tf}"
                )

    # 전체 비교 테이블
    if all_results:
        print("\n")
        print_comparison_table(
            sorted(all_results, key=lambda x: x['win_rate'], reverse=True),
            title="전체 전략 비교 (승률 내림차순)"
        )

    # HTML 리포트
    if args.report == 'html':
        generate_html_report(all_results, trades_by_label)


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
