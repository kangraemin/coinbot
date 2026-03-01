"""리포트 모듈 — 터미널 출력 및 HTML 리포트 생성."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from analysis.config import OUTPUT_DIR


# ── 터미널 출력 ──────────────────────────────────────────────────────────────

def print_summary(
    coin: str,
    tf: str,
    strategy: str,
    summary: dict,
    trades_df: pd.DataFrame,
) -> None:
    """단일 전략 결과를 터미널에 출력."""
    total = summary['total_trades']
    if total == 0:
        print(f"  [{coin.upper()} {tf} {strategy}] 거래 없음")
        return

    wins = (trades_df['pnl_leveraged'] > 0).sum()
    losses = (trades_df['pnl_leveraged'] <= 0).sum()
    tp_cnt = (trades_df['exit_reason'] == 'tp').sum()
    sl_cnt = (trades_df['exit_reason'] == 'sl').sum()
    to_cnt = (trades_df['exit_reason'] == 'timeout').sum()

    print(f"\n  {'=' * 56}")
    print(f"  {coin.upper()} / {tf} / {strategy}")
    print(f"  {'=' * 56}")
    print(f"  거래 수    : {total}건  (TP {tp_cnt} / SL {sl_cnt} / 타임아웃 {to_cnt})")
    print(f"  승률       : {summary['win_rate'] * 100:.1f}%  ({wins}승 {losses}패)")
    print(f"  레버리지수익: {summary['total_return'] * 100:+.2f}%")
    print(f"  평균 수익   : {summary['avg_return'] * 100:+.3f}%")
    print(f"  최대 낙폭   : -{summary['max_drawdown'] * 100:.2f}%")
    print(f"  Sharpe     : {summary['sharpe_ratio']:.3f}")


def print_comparison_table(results: list[dict], title: str = "") -> None:
    """여러 전략/타임프레임 결과를 비교 테이블로 출력."""
    if not results:
        print("  결과 없음")
        return

    df = pd.DataFrame(results)

    # 퍼센트로 변환
    for col in ['win_rate', 'total_return', 'avg_return', 'max_drawdown']:
        if col in df.columns:
            df[col] = df[col] * 100

    if title:
        print(f"\n  {'=' * 70}")
        print(f"  {title}")
        print(f"  {'=' * 70}")

    # 컬럼 정렬
    col_order = [c for c in [
        'strategy', 'timeframe', 'coin', 'signal_count',
        'total_trades', 'win_rate', 'total_return',
        'avg_return', 'max_drawdown', 'sharpe_ratio'
    ] if c in df.columns]

    print(df[col_order].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


# ── HTML 리포트 ──────────────────────────────────────────────────────────────

def generate_html_report(
    all_results: list[dict],
    trades_by_strategy: dict[str, pd.DataFrame],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """matplotlib equity curve 포함 HTML 리포트 생성.

    Args:
        all_results: 전략별 요약 dict 리스트
        trades_by_strategy: {label: trades_df} 형태
        output_dir: 출력 디렉토리

    Returns:
        생성된 HTML 파일 경로
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import base64
    except ImportError:
        print("  [경고] matplotlib 미설치. pip install matplotlib")
        return _generate_simple_html(all_results, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    out_path = output_dir / f"report_{date_str}.html"

    # equity curve 이미지 생성
    chart_htmls = []
    for label, trades_df in trades_by_strategy.items():
        if trades_df.empty:
            continue
        img_b64 = _equity_curve_b64(trades_df, label, plt)
        chart_htmls.append(f"""
            <h3>{label}</h3>
            <img src="data:image/png;base64,{img_b64}" style="max-width:100%">
        """)

    # 요약 테이블 HTML
    if all_results:
        df_summary = pd.DataFrame(all_results)
        for col in ['win_rate', 'total_return', 'avg_return', 'max_drawdown']:
            if col in df_summary.columns:
                df_summary[col] = (df_summary[col] * 100).round(2)
        table_html = df_summary.to_html(index=False, classes='table', border=1)
    else:
        table_html = "<p>결과 없음</p>"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Coinbot 분석 리포트 {date_str}</title>
<style>
  body {{ font-family: monospace; padding: 20px; background: #1a1a1a; color: #e0e0e0; }}
  h1 {{ color: #4fc3f7; }}
  h2 {{ color: #81d4fa; border-bottom: 1px solid #333; }}
  h3 {{ color: #b0bec5; }}
  .table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  .table th {{ background: #333; padding: 8px; text-align: right; }}
  .table td {{ padding: 6px 8px; text-align: right; border-bottom: 1px solid #333; }}
  .table tr:hover {{ background: #2a2a2a; }}
</style>
</head>
<body>
<h1>Coinbot 진입 포인트 분석 리포트</h1>
<p>생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>전략별 요약</h2>
{table_html}

<h2>Equity Curve</h2>
{''.join(chart_htmls) if chart_htmls else '<p>차트 없음</p>'}
</body>
</html>"""

    out_path.write_text(html, encoding='utf-8')
    print(f"\n  HTML 리포트 저장: {out_path}")
    return out_path


def _equity_curve_b64(trades_df: pd.DataFrame, label: str, plt) -> str:
    """equity curve PNG를 base64 문자열로 반환."""
    import io, base64

    cumulative = trades_df['pnl_leveraged'].cumsum()

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.plot(cumulative.values, color='#4fc3f7', linewidth=1.5)
    ax.axhline(0, color='#555', linewidth=0.8, linestyle='--')
    ax.fill_between(range(len(cumulative)), cumulative.values, 0,
                    where=(cumulative.values >= 0), alpha=0.3, color='#4fc3f7')
    ax.fill_between(range(len(cumulative)), cumulative.values, 0,
                    where=(cumulative.values < 0), alpha=0.3, color='#ef5350')
    ax.set_title(f'Equity Curve — {label}', color='#e0e0e0')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_color('#444')

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=100, facecolor='#1a1a1a')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _generate_simple_html(all_results: list[dict], output_dir: Path) -> Path:
    """matplotlib 없이 간단한 HTML 생성."""
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    out_path = output_dir / f"report_{date_str}.html"

    if all_results:
        df = pd.DataFrame(all_results)
        table_html = df.to_html(index=False)
    else:
        table_html = "<p>결과 없음</p>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Coinbot Report</title></head>
<body><h1>Coinbot Report {date_str}</h1>{table_html}</body></html>"""
    out_path.write_text(html, encoding='utf-8')
    return out_path
