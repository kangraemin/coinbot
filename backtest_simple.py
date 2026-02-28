"""1분봉 데이터 기반 백테스팅 CLI 엔진.

전략: prev_close 대비 --entry-pct% 하락 시 롱 진입
진입봉 내 체결 순서: low 먼저면 SL, high 먼저면 TP

사용법:
    python backtest_simple.py
    python backtest_simple.py --tp 2 --sl 1 --leverage 3
    python backtest_simple.py --from 2024-01-01 --to 2024-12-31
    python backtest_simple.py --pos 20
"""

import argparse
import logging
import os
from datetime import datetime, timezone

import pandas as pd

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATA_DIR = "data"

# 수수료: 0.05% taker × 레버리지 × 2 (진입+청산) → 단방향 레버리지 적용
TAKER_FEE_RATE = 0.0005


def load_data(from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """기간에 맞는 parquet 파일들을 로드해 합친다."""
    frames: list[pd.DataFrame] = []

    start_year = from_dt.year
    end_year = to_dt.year

    for year in range(start_year, end_year + 1):
        path = os.path.join(DATA_DIR, f"btc_1m_{year}.parquet")
        if not os.path.exists(path):
            logger.warning("파일 없음: %s", path)
            continue
        df = pd.read_parquet(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"{DATA_DIR}/ 에 parquet 파일이 없습니다.")

    df = pd.concat(frames, ignore_index=True)

    # timestamp 컬럼 처리
    if "timestamp" not in df.columns:
        raise ValueError("timestamp 컬럼이 없습니다.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 기간 필터
    from_ts = pd.Timestamp(from_dt)
    to_ts = pd.Timestamp(to_dt)
    df = df[(df["timestamp"] >= from_ts) & (df["timestamp"] < to_ts)]
    df = df.reset_index(drop=True)

    logger.info(
        "데이터 로드 완료: %d 행 (%s ~ %s)",
        len(df),
        df["timestamp"].iloc[0] if len(df) > 0 else "N/A",
        df["timestamp"].iloc[-1] if len(df) > 0 else "N/A",
    )
    return df


def run_backtest(
    df: pd.DataFrame,
    entry_pct: float,
    tp_pct: float,
    sl_pct: float,
    leverage: int,
    pos_ratio: float,
) -> dict:
    """백테스팅 실행 후 결과 dict 반환."""
    fee_rate = TAKER_FEE_RATE * leverage * 2  # 왕복 수수료율 (자본 기준)

    equity = 1.0  # 초기 자본 1.0 (복리)
    equity_curve: list[float] = [1.0]
    trades: list[dict] = []

    in_position = False
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    entry_idx = 0

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df["timestamp"].values

    for i in range(1, len(df)):
        prev_close = closes[i - 1]
        open_price = df["open"].values[i]
        high = highs[i]
        low = lows[i]
        ts = timestamps[i]

        if not in_position:
            # 진입 조건: prev_close 대비 entry_pct% 이상 하락
            entry_trigger = prev_close * (1 - entry_pct / 100)
            if low <= entry_trigger:
                entry_price = entry_trigger
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
                in_position = True
                entry_idx = i
                continue

        if in_position:
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

            if hit_tp and hit_sl:
                # 같은 봉에서 TP/SL 둘 다 도달: 열가 기준으로 어느 쪽이 먼저인지 판단
                # open이 entry_price에 가까우면 순서 불확실 → low가 sl에 먼저 닿는다고 가정
                # (보수적 처리)
                exit_price = sl_price
                outcome = "sl"
            elif hit_sl:
                exit_price = sl_price
                outcome = "sl"
            elif hit_tp:
                exit_price = tp_price
                outcome = "tp"
            else:
                continue

            # 수익률 계산 (레버리지 적용, 복리)
            price_return = (exit_price - entry_price) / entry_price
            trade_return = price_return * leverage - fee_rate
            trade_pnl = trade_return * pos_ratio / 100  # 자본 대비 실제 손익

            equity_before = equity
            equity *= 1 + trade_pnl
            equity_curve.append(equity)

            trades.append(
                {
                    "entry_ts": timestamps[entry_idx],
                    "exit_ts": ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "outcome": outcome,
                    "trade_return_pct": trade_return * 100,
                    "equity": equity,
                }
            )

            in_position = False

    return {
        "trades": trades,
        "equity_curve": equity_curve,
    }


def calc_stats(result: dict, df: pd.DataFrame) -> dict:
    """백테스트 결과 통계 계산."""
    trades = result["trades"]
    equity_curve = result["equity_curve"]

    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_tp_pct": 0.0,
            "avg_sl_pct": 0.0,
            "expectancy": 0.0,
            "total_return_pct": 0.0,
            "mdd_pct": 0.0,
            "max_consec_losses": 0,
        }

    wins = [t for t in trades if t["outcome"] == "tp"]
    losses = [t for t in trades if t["outcome"] == "sl"]

    win_rate = len(wins) / len(trades) * 100
    avg_tp = sum(t["trade_return_pct"] for t in wins) / len(wins) if wins else 0.0
    avg_sl = sum(t["trade_return_pct"] for t in losses) / len(losses) if losses else 0.0
    expectancy = (
        win_rate / 100 * avg_tp + (1 - win_rate / 100) * avg_sl
    )

    total_return_pct = (equity_curve[-1] - 1) * 100

    # MDD
    peak = equity_curve[0]
    mdd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > mdd:
            mdd = dd

    # 최대 연속 손실
    max_consec = 0
    cur_consec = 0
    for t in trades:
        if t["outcome"] == "sl":
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_tp_pct": avg_tp,
        "avg_sl_pct": avg_sl,
        "expectancy": expectancy,
        "total_return_pct": total_return_pct,
        "mdd_pct": mdd,
        "max_consec_losses": max_consec,
    }


def calc_monthly_table(trades: list[dict]) -> pd.DataFrame:
    """월별 수익표 생성."""
    if not trades:
        return pd.DataFrame()

    rows = []
    for t in trades:
        ts = pd.Timestamp(t["exit_ts"])
        rows.append(
            {
                "year": ts.year,
                "month": ts.month,
                "pnl_pct": t["trade_return_pct"],
            }
        )

    df_t = pd.DataFrame(rows)
    monthly = (
        df_t.groupby(["year", "month"])["pnl_pct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "수익(%)", "count": "거래수"})
        .reset_index()
    )
    monthly.columns = ["연도", "월", "수익(%)", "거래수"]
    monthly["수익(%)"] = monthly["수익(%)"].round(2)
    return monthly


def print_results(stats: dict, monthly: pd.DataFrame) -> None:
    """결과 출력."""
    print("\n" + "=" * 50)
    print("백테스팅 결과")
    print("=" * 50)
    print(f"총 거래수     : {stats['total_trades']}")
    print(f"승률          : {stats['win_rate']:.1f}%")
    print(f"평균 익절     : {stats['avg_tp_pct']:.2f}%")
    print(f"평균 손절     : {stats['avg_sl_pct']:.2f}%")
    print(f"기대값        : {stats['expectancy']:.2f}%")
    print("-" * 50)
    print(f"복리 총수익   : {stats['total_return_pct']:.2f}%")
    print(f"MDD           : {stats['mdd_pct']:.2f}%")
    print(f"최대연속손실  : {stats['max_consec_losses']}회")
    print("=" * 50)

    if not monthly.empty:
        print("\n월별 수익표:")
        print(monthly.to_string(index=False))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTC/USDT 1분봉 기반 단순 백테스팅"
    )
    parser.add_argument(
        "--entry-pct",
        type=float,
        default=1.0,
        dest="entry_pct",
        help="진입 조건: prev_close 대비 하락 %% (기본: 1.0)",
    )
    parser.add_argument(
        "--tp",
        type=float,
        default=1.5,
        help="익절 %% (기본: 1.5)",
    )
    parser.add_argument(
        "--sl",
        type=float,
        default=1.0,
        help="손절 %% (기본: 1.0)",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=3,
        help="레버리지 (기본: 3)",
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        type=str,
        default=None,
        help="시작 날짜 YYYY-MM-DD (기본: 2년 전)",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        type=str,
        default=None,
        help="종료 날짜 YYYY-MM-DD (기본: 오늘)",
    )
    parser.add_argument(
        "--pos",
        type=float,
        default=20.0,
        help="포지션 비율 %% (기본: 20)",
    )
    args = parser.parse_args()

    now = datetime.now(tz=timezone.utc)

    if args.from_date:
        from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    else:
        from_dt = datetime(now.year - 2, now.month, now.day, tzinfo=timezone.utc)

    if args.to_date:
        to_dt = datetime.strptime(args.to_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    else:
        to_dt = now

    logger.info(
        "파라미터 — entry: %.1f%%, TP: %.1f%%, SL: %.1f%%, 레버리지: %dx, 포지션: %.0f%%",
        args.entry_pct,
        args.tp,
        args.sl,
        args.leverage,
        args.pos,
    )

    df = load_data(from_dt, to_dt)

    if df.empty:
        logger.error("해당 기간 데이터 없음")
        return

    result = run_backtest(
        df,
        entry_pct=args.entry_pct,
        tp_pct=args.tp,
        sl_pct=args.sl,
        leverage=args.leverage,
        pos_ratio=args.pos,
    )

    stats = calc_stats(result, df)
    monthly = calc_monthly_table(result["trades"])
    print_results(stats, monthly)


if __name__ == "__main__":
    main()
