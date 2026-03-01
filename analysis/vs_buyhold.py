"""전략 vs 바이앤홀드 비교."""
import numpy as np
import pandas as pd

BUY_HOLD = {
    'recent_3m':   {'BTC': -26.8, 'ETH': -32.8},
    'bear_2025q4': {'BTC': -23.3, 'ETH': -28.4},
    'bull_2025q3': {'BTC':  +6.2, 'ETH': +66.3},
    'bull_2024h2': {'BTC': +47.4, 'ETH':  -2.4},
    'bear_2024h1': {'BTC': +43.5, 'ETH': +47.0},
    'bull_2023':   {'BTC':+155.2, 'ETH': +92.1},
}
PERIOD_LABEL = {
    'recent_3m':   '하락장(최근)',
    'bear_2025q4': '하락장(Q4)',
    'bull_2025q3': '상승(Q3)',
    'bull_2024h2': '불장(24H2)',
    'bear_2024h1': '횡보(24H1)',
    'bull_2023':   '회복(2023)',
}

df = pd.read_csv('analysis/output/consistent_strategies.csv')

rows = []
for _, row in df.iterrows():
    coin = row['coin']
    beat_count = 0
    total_periods = 0
    beat_details = []

    for pkey, label in PERIOD_LABEL.items():
        ret = row.get(f'{pkey}_ret')
        bh  = BUY_HOLD.get(pkey, {}).get(coin)
        if ret is None or (isinstance(ret, float) and np.isnan(ret)) or bh is None:
            continue
        total_periods += 1
        diff = ret - bh
        beat = bool(ret > bh)
        if beat:
            beat_count += 1
        beat_details.append((label, ret, bh, diff, beat))

    rows.append({
        'coin':          coin,
        'timeframe':     row['timeframe'],
        'signal':        row['signal'],
        'filters':       row['filters'],
        'params':        row['params'],
        'beat_count':    beat_count,
        'total_periods': total_periods,
        'avg_return':    row['avg_return'],
        'consistency':   row['consistency'],
        'total_trades':  row['total_trades'],
        '_details':      beat_details,
    })

result = pd.DataFrame(rows).sort_values(['beat_count', 'avg_return'], ascending=False)

save_cols = [c for c in result.columns if c != '_details']
result[save_cols].to_csv('analysis/output/vs_buyhold.csv', index=False)
print('저장 완료: analysis/output/vs_buyhold.csv\n')

print('=== 바이앤홀드 수익률 (레버리지 없음) ===')
print(f'{"기간":<15} {"BTC":>8} {"ETH":>8}')
print('-' * 35)
for pkey, label in PERIOD_LABEL.items():
    b = BUY_HOLD[pkey]
    print(f'{label:<15} {b["BTC"]:>+7.1f}%  {b["ETH"]:>+7.1f}%')

print()
print(f'※ 전략 수익은 레버리지 5x 적용 기준')
print()
print('=== 바이앤홀드를 이긴 전략 ===')

top = [r for _, r in result.iterrows() if r['beat_count'] >= 3]
for r in top[:15]:
    print(f"\n  {r['coin']} {r['timeframe']} | {r['signal']} | {r['filters']} | {r['params']}")
    print(f"  B&H 이긴 기간: {r['beat_count']}/{r['total_periods']}  "
          f"평균수익: {r['avg_return']:+.1f}%  총거래: {r['total_trades']}건")
    for label, ret, bh, diff, beat in r['_details']:
        flag = '✓' if beat else '✗'
        print(f"    {flag} {label:<14} 전략 {ret:+6.1f}%  vs  B&H {bh:+6.1f}%  (차이 {diff:+.1f}%p)")
