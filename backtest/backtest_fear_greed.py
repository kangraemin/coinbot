"""Fear & Greed Index 기반 매수 전략 백테스트 — 10,584건 그리드 서치.

전략:
  매수: F&G Index <= fg_buy_threshold → 다음 날 시가 매수
  매도 (3가지 모드):
    1. fg_sell: F&G >= fg_sell_threshold → 매도
    2. tp_sl: TP/SL % 기반 (TP만/SL만/둘다)
    3. hold_days: max 보유일 + TP/SL 오버레이

코인: BTC, ETH, XRP
데이터: 1h parquet → 1d resample + F&G daily merge
"""

import os
import sys
import itertools
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market")
SENTIMENT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sentiment")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE_RATE = 0.0005
INITIAL_BALANCE = 10000.0

# ── 파라미터 그리드 ──────────────────────────────────
FG_BUY_THRESHOLDS = [5, 10, 15, 20, 25, 30, 35]
FG_SELL_THRESHOLDS = [50, 60, 70, 80, 90]
TP_PCTS = [0.10, 0.20, 0.30]
SL_PCTS = [0.05, 0.10, 0.15]
HOLD_DAYS_LIST = [30, 60, 90, 180]
COOLDOWN_DAYS_LIST = [0, 3, 7, 14, 21, 30]
COINS = ["btc", "eth", "xrp"]


# ── 데이터 로딩 ──────────────────────────────────────

def fetch_fng(refresh: bool = False) -> pd.DataFrame:
    """Fear & Greed Index 데이터 fetch + 캐싱."""
    cache_path = os.path.join(SENTIMENT_DIR, "fng_daily.csv")
    if not refresh and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        print(f"  F&G 캐시 로드: {len(df)}일, {df['date'].min()} ~ {df['date'].max()}")
        return df

    print("  F&G API 호출 중...")
    resp = requests.get(
        "https://api.alternative.me/fng/",
        params={"limit": "0", "format": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["data"]

    rows = []
    for item in data:
        ts = int(item["timestamp"])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        rows.append({"date": dt, "fng": int(item["value"])})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    os.makedirs(SENTIMENT_DIR, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  F&G 저장: {len(df)}일, {df['date'].min()} ~ {df['date'].max()}")
    return df


def load_daily(coin: str) -> pd.DataFrame:
    """1h parquet → 1d resample."""
    path = os.path.join(DATA_DIR, f"{coin}_1h_full.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{coin} 1h 데이터 없음: {path}")

    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    daily = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    daily = daily.reset_index()
    daily["date"] = daily["timestamp"].dt.normalize()
    return daily


def merge_data(daily: pd.DataFrame, fng: pd.DataFrame) -> pd.DataFrame:
    """일봉 + F&G 병합."""
    daily = daily.copy()
    fng = fng.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    fng["date"] = pd.to_datetime(fng["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    merged = pd.merge_asof(
        daily.sort_values("date"),
        fng.sort_values("date"),
        on="date",
        direction="backward",
    )
    return merged.dropna(subset=["fng"]).reset_index(drop=True)


# ── 백테스트 엔진 ────────────────────────────────────

def run_backtest(df: pd.DataFrame, params: dict) -> dict:
    """단일 파라미터 조합 백테스트."""
    fg_buy = params["fg_buy"]
    sell_mode = params["sell_mode"]
    fg_sell = params.get("fg_sell")
    tp_pct = params.get("tp_pct")
    sl_pct = params.get("sl_pct")
    hold_days = params.get("hold_days")
    cooldown = params["cooldown"]

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    fng_arr = df["fng"].to_numpy(dtype=float)
    n = len(closes)

    balance = INITIAL_BALANCE
    peak_balance = INITIAL_BALANCE
    max_dd = 0.0
    trades = 0
    wins = 0
    total_hold = 0

    in_pos = False
    entry_price = 0.0
    entry_day = 0
    last_exit_day = -999

    for i in range(1, n):
        if not in_pos:
            if balance <= 0:
                break

            # cooldown 체크
            if i - last_exit_day <= cooldown:
                continue

            # 매수 신호: 전날 F&G <= threshold → 오늘 시가 매수
            if fng_arr[i - 1] <= fg_buy:
                entry_price = opens[i]
                if entry_price <= 0:
                    continue
                fee = balance * TAKER_FEE_RATE
                balance -= fee
                entry_day = i
                in_pos = True
        else:
            exit_price = None
            days_held = i - entry_day

            # SL 체크 (우선)
            if sl_pct is not None:
                sl_price = entry_price * (1 - sl_pct)
                if lows[i] <= sl_price:
                    exit_price = sl_price

            # TP 체크
            if exit_price is None and tp_pct is not None:
                tp_price = entry_price * (1 + tp_pct)
                if highs[i] >= tp_price:
                    exit_price = tp_price

            # 매도 모드별 추가 조건
            if exit_price is None:
                if sell_mode == "fg_sell":
                    if fng_arr[i] >= fg_sell:
                        exit_price = closes[i]
                elif sell_mode == "hold":
                    if days_held >= hold_days:
                        exit_price = closes[i]
                # tp_sl 모드: TP/SL만으로 청산 (위에서 이미 처리)

            if exit_price is not None:
                pnl_pct = (exit_price - entry_price) / entry_price
                balance = balance * (1 + pnl_pct)
                balance -= abs(balance * TAKER_FEE_RATE)

                trades += 1
                if pnl_pct > 0:
                    wins += 1
                total_hold += days_held

                peak_balance = max(peak_balance, balance)
                dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
                max_dd = max(max_dd, dd)

                in_pos = False
                last_exit_day = i

    # 미청산 포지션 → 마지막 종가로 청산
    if in_pos:
        exit_price = closes[-1]
        pnl_pct = (exit_price - entry_price) / entry_price
        balance = balance * (1 + pnl_pct)
        balance -= abs(balance * TAKER_FEE_RATE)
        trades += 1
        if pnl_pct > 0:
            wins += 1
        total_hold += (n - 1 - entry_day)
        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)

    total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0
    avg_hold = (total_hold / trades) if trades > 0 else 0

    # Buy & Hold 수익률
    bnh_return = (closes[-1] - opens[0]) / opens[0] * 100 if opens[0] > 0 else 0

    # CAGR
    days_total = n
    years = days_total / 365.25
    if years > 0 and balance > 0:
        cagr = ((balance / INITIAL_BALANCE) ** (1 / years) - 1) * 100
    else:
        cagr = 0

    return {
        **params,
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "mdd": round(max_dd * 100, 2),
        "trades": trades,
        "win_rate": round(win_rate, 1),
        "avg_hold_days": round(avg_hold, 1),
        "final_balance": round(balance, 2),
        "bnh_return": round(bnh_return, 2),
        "vs_bnh": round(total_return - bnh_return, 2),
    }


# ── 그리드 생성 ──────────────────────────────────────

def build_combos() -> list[dict]:
    """10,584건 파라미터 조합 생성."""
    combos = []
    for coin in COINS:
        for fg_buy in FG_BUY_THRESHOLDS:
            for cd in COOLDOWN_DAYS_LIST:
                base = {"coin": coin, "fg_buy": fg_buy, "cooldown": cd}

                # Mode 1: fg_sell
                for fg_sell in FG_SELL_THRESHOLDS:
                    combos.append({
                        **base,
                        "sell_mode": "fg_sell",
                        "fg_sell": fg_sell,
                        "tp_pct": None,
                        "sl_pct": None,
                        "hold_days": None,
                    })

                # Mode 2: tp_sl (TP만 / SL만 / 둘다)
                for tp in TP_PCTS:
                    combos.append({
                        **base,
                        "sell_mode": "tp_sl",
                        "fg_sell": None,
                        "tp_pct": tp,
                        "sl_pct": None,
                        "hold_days": None,
                    })
                for sl in SL_PCTS:
                    combos.append({
                        **base,
                        "sell_mode": "tp_sl",
                        "fg_sell": None,
                        "tp_pct": None,
                        "sl_pct": sl,
                        "hold_days": None,
                    })
                for tp in TP_PCTS:
                    for sl in SL_PCTS:
                        combos.append({
                            **base,
                            "sell_mode": "tp_sl",
                            "fg_sell": None,
                            "tp_pct": tp,
                            "sl_pct": sl,
                            "hold_days": None,
                        })

                # Mode 3: hold_days + TP/SL 오버레이
                for hd in HOLD_DAYS_LIST:
                    # 없이
                    combos.append({
                        **base,
                        "sell_mode": "hold",
                        "fg_sell": None,
                        "tp_pct": None,
                        "sl_pct": None,
                        "hold_days": hd,
                    })
                    # TP만
                    for tp in TP_PCTS:
                        combos.append({
                            **base,
                            "sell_mode": "hold",
                            "fg_sell": None,
                            "tp_pct": tp,
                            "sl_pct": None,
                            "hold_days": hd,
                        })
                    # SL만
                    for sl in SL_PCTS:
                        combos.append({
                            **base,
                            "sell_mode": "hold",
                            "fg_sell": None,
                            "tp_pct": None,
                            "sl_pct": sl,
                            "hold_days": hd,
                        })
                    # 둘다
                    for tp in TP_PCTS:
                        for sl in SL_PCTS:
                            combos.append({
                                **base,
                                "sell_mode": "hold",
                                "fg_sell": None,
                                "tp_pct": tp,
                                "sl_pct": sl,
                                "hold_days": hd,
                            })

    return combos


def _worker(args):
    """ProcessPoolExecutor 워커."""
    coin_data, params = args
    return run_backtest(coin_data, params)


# ── 메인 ─────────────────────────────────────────────

def main():
    refresh = "--refresh" in sys.argv

    print("=" * 60)
    print("Fear & Greed Index 매수 전략 백테스트")
    print("=" * 60)

    # 1. F&G 데이터
    fng = fetch_fng(refresh=refresh)

    # 2. 코인별 일봉 + F&G 병합
    coin_data = {}
    for coin in COINS:
        print(f"\n[{coin.upper()}] 데이터 로딩...")
        daily = load_daily(coin)
        merged = merge_data(daily, fng)
        coin_data[coin] = merged
        print(f"  일봉 {len(merged)}일, {merged['date'].min()} ~ {merged['date'].max()}")

    # 3. 그리드 생성
    combos = build_combos()
    print(f"\n총 조합: {len(combos):,}건")

    # 4. 병렬 실행
    tasks = [(coin_data[p["coin"]], p) for p in combos]
    results = []
    workers = os.cpu_count() or 4

    print(f"병렬 실행 (workers={workers})...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, t): i for i, t in enumerate(tasks)}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 1000 == 0 or done == len(futures):
                print(f"  진행: {done:,}/{len(futures):,} ({done/len(futures)*100:.0f}%)")
            try:
                results.append(future.result())
            except Exception as e:
                idx = futures[future]
                print(f"  ERROR combo #{idx}: {e}")
                results.append({**combos[idx], "total_return": None, "error": str(e)})

    # 5. 결과 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_results = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "fear_greed_grid.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path} ({len(df_results):,}행)")

    # 6. 요약 출력
    valid = df_results[df_results["total_return"].notna()].copy()
    if valid.empty:
        print("유효한 결과 없음")
        return

    print("\n" + "=" * 60)
    print("TOP 20 조합 (수익률 기준)")
    print("=" * 60)
    top = valid.nlargest(20, "total_return")
    cols = ["coin", "fg_buy", "sell_mode", "fg_sell", "tp_pct", "sl_pct",
            "hold_days", "cooldown", "total_return", "mdd", "trades",
            "win_rate", "cagr", "vs_bnh"]
    print(top[cols].to_string(index=False))

    # 모드별 평균
    print("\n" + "=" * 60)
    print("매도 모드별 평균 성과")
    print("=" * 60)
    mode_stats = valid.groupby("sell_mode").agg(
        avg_return=("total_return", "mean"),
        avg_mdd=("mdd", "mean"),
        avg_trades=("trades", "mean"),
        avg_winrate=("win_rate", "mean"),
        best_return=("total_return", "max"),
    ).round(2)
    print(mode_stats)

    # 코인별 평균
    print("\n" + "=" * 60)
    print("코인별 평균 성과")
    print("=" * 60)
    coin_stats = valid.groupby("coin").agg(
        avg_return=("total_return", "mean"),
        avg_mdd=("mdd", "mean"),
        avg_trades=("trades", "mean"),
        best_return=("total_return", "max"),
        bnh=("bnh_return", "first"),
    ).round(2)
    print(coin_stats)

    # fg_buy threshold별 평균
    print("\n" + "=" * 60)
    print("F&G 매수 threshold별 평균 수익률")
    print("=" * 60)
    fg_stats = valid.groupby("fg_buy").agg(
        avg_return=("total_return", "mean"),
        avg_trades=("trades", "mean"),
        avg_winrate=("win_rate", "mean"),
    ).round(2)
    print(fg_stats)


if __name__ == "__main__":
    main()
