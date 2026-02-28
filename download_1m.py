"""Binance USDM BTC/USDT:USDT 1분봉 데이터 다운로드 스크립트.

사용법:
    python download_1m.py [--years 2]
"""

import argparse
import logging
import os
from datetime import datetime, timezone

import ccxt
import pandas as pd

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "1m"
DATA_DIR = "data"
LIMIT = 1000  # ccxt fetch_ohlcv 최대 한 번에 가져올 캔들 수


def fetch_ohlcv_range(
    exchange: ccxt.Exchange,
    since_ms: int,
    until_ms: int,
) -> list[list]:
    """since_ms ~ until_ms 구간의 OHLCV 데이터를 모두 가져온다."""
    all_candles: list[list] = []
    current = since_ms

    total_ms = until_ms - since_ms
    fetched_ms = 0

    while current < until_ms:
        try:
            candles = exchange.fetch_ohlcv(
                SYMBOL, TIMEFRAME, since=current, limit=LIMIT
            )
        except Exception as e:
            logger.error("fetch_ohlcv 실패: %s", e)
            break

        if not candles:
            break

        # until_ms 이전 데이터만 수집
        filtered = [c for c in candles if c[0] < until_ms]
        all_candles.extend(filtered)

        last_ts = candles[-1][0]
        fetched_ms = last_ts - since_ms

        pct = min(fetched_ms / total_ms * 100, 100) if total_ms > 0 else 100
        logger.info(
            "%.1f%% 완료 (%s)",
            pct,
            datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            ),
        )

        if len(candles) < LIMIT or last_ts >= until_ms:
            break

        current = last_ts + 60_000  # 1분 = 60,000ms

    return all_candles


def load_existing_parquet(path: str) -> pd.DataFrame | None:
    """기존 parquet 파일 로드. 없으면 None 반환."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning("기존 파일 읽기 실패 (%s): %s", path, e)
        return None


def download_year(exchange: ccxt.Exchange, year: int) -> None:
    """특정 연도의 1분봉 데이터를 다운로드/업데이트한다."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"btc_1m_{year}.parquet")

    now = datetime.now(tz=timezone.utc)
    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    # 미래 연도는 현재 시각까지만
    if year_end > now:
        year_end = now

    since_ms = int(year_start.timestamp() * 1000)
    until_ms = int(year_end.timestamp() * 1000)

    existing = load_existing_parquet(path)

    if existing is not None and not existing.empty:
        last_ts_ms = int(existing["timestamp"].max().timestamp() * 1000)
        # 마지막 데이터 이후부터 이어받기
        since_ms = last_ts_ms + 60_000
        logger.info(
            "[%d] 증분 업데이트: %s 부터",
            year,
            datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            ),
        )
    else:
        logger.info("[%d] 전체 다운로드 시작", year)

    if since_ms >= until_ms:
        logger.info("[%d] 이미 최신 상태", year)
        return

    candles = fetch_ohlcv_range(exchange, since_ms, until_ms)

    if not candles:
        logger.info("[%d] 새 데이터 없음", year)
        return

    df_new = pd.DataFrame(
        candles, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
    )
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp_ms"], unit="ms", utc=True)
    df_new = df_new.drop(columns=["timestamp_ms"])
    df_new = df_new.set_index("timestamp")

    if existing is not None and not existing.empty:
        existing_indexed = existing.set_index("timestamp") if "timestamp" in existing.columns else existing
        df_combined = pd.concat([existing_indexed, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        df_combined = df_combined.sort_index()
    else:
        df_combined = df_new

    df_combined.reset_index().to_parquet(path, index=False)
    logger.info(
        "[%d] 저장 완료: %s (%d 행)",
        year,
        path,
        len(df_combined),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binance USDM BTC/USDT 1분봉 데이터 다운로드"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="다운로드할 연 수 (기본: 2)",
    )
    args = parser.parse_args()

    exchange = ccxt.binanceusdm({"enableRateLimit": True})

    current_year = datetime.now(tz=timezone.utc).year
    target_years = [current_year - i for i in range(args.years - 1, -1, -1)]

    logger.info("다운로드 대상 연도: %s", target_years)

    for year in target_years:
        download_year(exchange, year)

    logger.info("모든 다운로드 완료")


if __name__ == "__main__":
    main()
