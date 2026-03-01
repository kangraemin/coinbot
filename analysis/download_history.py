"""
바이낸스에서 1시간봉 전체 히스토리 다운로드.
BTC: 2017-08, ETH: 2017-08, XRP: 2018-01, SOL: 2020-04부터.
저장: data/market/{coin}_1h_full.parquet
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / 'data' / 'market'
DATA_DIR.mkdir(parents=True, exist_ok=True)

COINS = {
    'btc': ('BTC/USDT', '2017-08-01'),
    'eth': ('ETH/USDT', '2017-08-01'),
    'xrp': ('XRP/USDT', '2018-01-01'),
    'sol': ('SOL/USDT', '2020-04-01'),
}

BATCH = 1000


def download(coin: str, symbol: str, since_str: str) -> pd.DataFrame:
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = int(datetime.strptime(since_str, '%Y-%m-%d')
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)

    all_rows = []
    cursor   = since_ms
    total    = 0

    while cursor < now_ms:
        try:
            rows = exchange.fetch_ohlcv(symbol, '1h', since=cursor, limit=BATCH)
        except Exception as e:
            print(f'  오류: {e} — 5초 후 재시도')
            time.sleep(5)
            continue

        if not rows:
            break

        all_rows.extend(rows)
        cursor = rows[-1][0] + 3_600_000   # 다음 봉
        total += len(rows)

        last_dt = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc)
        print(f'\r  {coin.upper()} {last_dt.strftime("%Y-%m-%d")} — {total:,}봉 수집', end='', flush=True)

        if len(rows) < BATCH:
            break

    print()
    df = pd.DataFrame(all_rows, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    return df


def main():
    for coin, (symbol, since) in COINS.items():
        out_path = DATA_DIR / f'{coin}_1h_full.parquet'
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            last_ts  = existing['timestamp'].max()
            print(f'{coin.upper()}: 기존 {len(existing):,}봉 (마지막: {last_ts.strftime("%Y-%m-%d")}) — 이어받기')
            since_ms = int((last_ts.timestamp() + 3600) * 1000)
            since_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')

            exchange = ccxt.binance({'enableRateLimit': True})
            new_rows = []
            cursor   = since_ms
            now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)

            while cursor < now_ms:
                try:
                    rows = exchange.fetch_ohlcv(symbol, '1h', since=cursor, limit=BATCH)
                except Exception as e:
                    print(f'  오류: {e} — 재시도')
                    time.sleep(5)
                    continue
                if not rows: break
                new_rows.extend(rows)
                cursor = rows[-1][0] + 3_600_000
                last_dt = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc)
                print(f'\r  → {last_dt.strftime("%Y-%m-%d")} ({len(new_rows):,}봉 추가)', end='', flush=True)
                if len(rows) < BATCH: break
            print()

            if new_rows:
                df_new = pd.DataFrame(new_rows, columns=['timestamp','open','high','low','close','volume'])
                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms', utc=True)
                df = pd.concat([existing, df_new]).drop_duplicates('timestamp').sort_values('timestamp')
                df.reset_index(drop=True).to_parquet(out_path, index=False)
                print(f'  저장: {len(df):,}봉 → {out_path.name}')
            else:
                print(f'  최신 상태 유지')
        else:
            print(f'{coin.upper()}: {since}부터 전체 다운로드')
            df = download(coin, symbol, since)
            df.reset_index(drop=True).to_parquet(out_path, index=False)
            print(f'  저장: {len(df):,}봉 → {out_path.name}')

    print('\n완료.')


if __name__ == '__main__':
    main()
