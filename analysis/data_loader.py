"""데이터 로더 — parquet 파일에서 OHLCV 로드 및 타임프레임 리샘플링."""

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from analysis.config import DATA_DIR, LOOKBACK_DAYS, RESAMPLE_RULES


def load_1m(coin: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """1분봉 parquet 파일 로드 후 최근 lookback_days 슬라이싱.

    Args:
        coin: 코인명 (예: 'btc', 'eth')
        lookback_days: 최근 몇 일치 데이터를 로드할지

    Returns:
        timestamp 인덱스를 가진 OHLCV DataFrame
    """
    files = sorted([
        f for f in DATA_DIR.iterdir()
        if f.name.startswith(f"{coin}_1m_") and f.name.endswith(".parquet")
    ])
    if not files:
        raise FileNotFoundError(f"{coin} 1분봉 데이터 없음: {DATA_DIR}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")

    # 최근 lookback_days 슬라이싱
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    df = df[df.index >= cutoff]

    return df[["open", "high", "low", "close", "volume"]]


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """1분봉 → 지정 타임프레임으로 리샘플링.

    Args:
        df: timestamp 인덱스를 가진 1분봉 DataFrame
        tf: 타임프레임 (예: '5m', '15m', '1h', '4h', '1d')

    Returns:
        리샘플링된 OHLCV DataFrame (timestamp 컬럼)
    """
    rule = RESAMPLE_RULES.get(tf, tf)
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return resampled.reset_index().rename(columns={"timestamp": "timestamp"})


def load_all_timeframes(coin: str, lookback_days: int = LOOKBACK_DAYS) -> dict[str, pd.DataFrame]:
    """코인의 모든 타임프레임 데이터 반환.

    Returns:
        {'5m': df, '15m': df, ...} 형태의 dict
    """
    df_1m = load_1m(coin, lookback_days)
    result = {}
    for tf in RESAMPLE_RULES:
        result[tf] = resample(df_1m, tf)
    return result
