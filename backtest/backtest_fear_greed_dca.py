"""Fear & Greed DCA 분할매수 전략 백테스트 v2 — ~505K건 그리드 서치.

전략:
  매수: F&G <= threshold → 매일 분할매수 (fixed/tiered/equal)
  추가 필터: funding rate, RSI(14), ETH/BTC 비율, 연속하락일
  매도: 가중 평균단가 기준 TP/SL/fg_sell/hold (hold+overlay 포함)
  레버리지: 1/3/7x

코인: BTC, ETH, XRP
"""

import os
import sys
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd

# 기존 코드 재사용
sys.path.insert(0, os.path.dirname(__file__))
from backtest_fear_greed import fetch_fng, load_daily, merge_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market")
SENTIMENT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sentiment")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
TAKER_FEE_RATE = 0.0005
INITIAL_BALANCE = 10000.0

# ── 파라미터 그리드 ──────────────────────────────────
FG_BUY_THRESHOLDS = [10, 15, 20, 25, 30]
BUY_PCTS = [0.05, 0.10, 0.20]
TIER_SCALES = ["conservative", "moderate", "aggressive"]
MAX_INVEST_PCTS = [0.70, 1.00]
LEVERAGES = [1, 3, 7]
FUNDING_FILTERS = ["off", "negative"]
RSI_FILTERS = ["off", "oversold"]       # RSI(14) < 30
ETHBTC_FILTERS = ["off", "btc_strong"]  # ETH/BTC < 20일 MA
CONSEC_DOWN_FILTERS = ["off", "3plus"]  # 3일+ 연속 하락
FG_SELL_THRESHOLDS = [50, 70, 90]
TP_PCTS = [0.10, 0.20, 0.30]
SL_PCTS = [0.05, 0.10, 0.15]
HOLD_DAYS_LIST = [30, 90, 180]
COINS = ["btc", "eth", "xrp"]

TIER_CONFIGS = {
    "conservative": {(21, 30): 0.03, (11, 20): 0.05, (0, 10): 0.10},
    "moderate":     {(21, 30): 0.05, (11, 20): 0.10, (0, 10): 0.20},
    "aggressive":   {(21, 30): 0.10, (11, 20): 0.20, (0, 10): 0.30},
}


# ── 데이터 로딩 ──────────────────────────────────────

def fetch_funding_rate(refresh: bool = False) -> pd.DataFrame:
    """Binance BTCUSDT funding rate 히스토리 fetch + 캐싱."""
    cache_path = os.path.join(SENTIMENT_DIR, "funding_rate_daily.csv")
    if not refresh and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        print(f"  Funding rate 캐시 로드: {len(df)}일")
        return df

    print("  Funding rate API 호출 중...")
    all_rates = []
    start_ts = int(datetime(2019, 9, 1).timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)

    while start_ts < end_ts:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": "BTCUSDT", "startTime": start_ts, "limit": 1000},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rates.extend(data)
        start_ts = data[-1]["fundingTime"] + 1
        if len(data) < 1000:
            break

    rows = []
    for r in all_rates:
        dt = datetime.fromtimestamp(r["fundingTime"] / 1000, tz=timezone.utc).date()
        rows.append({"date": dt, "funding_rate": float(r["fundingRate"])})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # 일별 평균
    daily = df.groupby("date")["funding_rate"].mean().reset_index()

    os.makedirs(SENTIMENT_DIR, exist_ok=True)
    daily.to_csv(cache_path, index=False)
    print(f"  Funding rate 저장: {len(daily)}일")
    return daily


def merge_all_data(daily: pd.DataFrame, fng: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """일봉 + F&G + funding rate 병합."""
    merged = merge_data(daily, fng)

    funding = funding.copy()
    funding["date"] = pd.to_datetime(funding["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None).astype("datetime64[ns]")

    merged = pd.merge_asof(
        merged.sort_values("date"),
        funding.sort_values("date"),
        on="date",
        direction="backward",
    )
    # funding_rate NaN → 0 (데이터 없는 초기 구간)
    merged["funding_rate"] = merged["funding_rate"].fillna(0)
    return merged.reset_index(drop=True)


# ── 백테스트 엔진 (최적화) ───────────────────────────

def compute_buy_log(opens, fng_arr, funding_arr, n, params):
    """매수 시점/금액 계산 → 매수 로그 반환.

    Returns: list of (day_idx, invest_amount) or None if no buys.
    """
    fg_buy = params["fg_buy"]
    buy_mode = params["buy_mode"]
    buy_pct = params.get("buy_pct", 0.10)
    tier_scale = params.get("tier_scale")
    max_invest = params["max_invest"]
    leverage = params["leverage"]
    funding_filter = params["funding_filter"]

    tier_cfg = TIER_CONFIGS.get(tier_scale) if tier_scale else None

    cash = INITIAL_BALANCE
    max_investable = INITIAL_BALANCE * max_invest
    total_invested = 0.0
    buys = []

    for i in range(1, n):
        if cash <= 0 or total_invested >= max_investable:
            continue

        # F&G 신호
        if fng_arr[i - 1] > fg_buy:
            continue

        # Funding rate 필터
        if funding_filter == "negative" and funding_arr[i - 1] >= 0:
            continue

        # 투입 비율 결정
        if buy_mode == "fixed":
            pct = buy_pct
        elif buy_mode == "tiered":
            fng_val = fng_arr[i - 1]
            pct = 0
            for (lo, hi), p in tier_cfg.items():
                if lo <= fng_val <= hi:
                    pct = p
                    break
            if fng_val < 0:
                pct = list(tier_cfg.values())[-1]  # 최고 비율
        elif buy_mode == "equal":
            pct = buy_pct  # 등분 비율
        else:
            continue

        invest = min(cash * pct, max_investable - total_invested)
        if invest <= 0:
            continue

        fee = invest * TAKER_FEE_RATE
        cash -= (invest + fee)
        total_invested += invest
        buys.append((i, invest * leverage, opens[i]))  # (day, notional, entry_price)

    return buys, cash


def run_sell_simulation(opens, highs, lows, closes, fng_arr, n, buys, remaining_cash, params):
    """매수 로그를 받아 매도 조건별 시뮬레이션."""
    sell_mode = params["sell_mode"]
    fg_sell = params.get("fg_sell")
    tp_pct = params.get("tp_pct")
    sl_pct = params.get("sl_pct")
    hold_days = params.get("hold_days")
    leverage = params["leverage"]

    if not buys:
        return {
            **params,
            "total_return": 0.0, "cagr": 0.0, "mdd": 0.0,
            "trades_buy": 0, "trades_sell": 0, "win_rate": 0.0,
            "avg_hold_days": 0.0, "final_balance": INITIAL_BALANCE,
            "bnh_return": 0.0, "vs_bnh": 0.0,
        }

    # 포지션 추적: 가중 평균단가
    total_notional = 0.0
    total_coins = 0.0
    first_buy_day = buys[0][0]
    buy_idx = 0
    in_position = False

    balance = remaining_cash
    peak_balance = INITIAL_BALANCE
    max_dd = 0.0
    sell_count = 0
    wins = 0
    total_hold = 0

    for i in range(first_buy_day, n):
        # 이번 날 매수 처리
        while buy_idx < len(buys) and buys[buy_idx][0] == i:
            _, notional, price = buys[buy_idx]
            if price > 0:
                coins = notional / price
                total_notional += notional
                total_coins += coins
                in_position = True
            buy_idx += 1

        if not in_position or total_coins <= 0:
            continue

        avg_entry = total_notional / total_coins
        exit_price = None

        # SL 체크 (우선)
        if sl_pct is not None:
            sl_price = avg_entry * (1 - sl_pct)
            if lows[i] <= sl_price:
                exit_price = sl_price

        # TP 체크
        if exit_price is None and tp_pct is not None:
            tp_price = avg_entry * (1 + tp_pct)
            if highs[i] >= tp_price:
                exit_price = tp_price

        # 모드별 추가 조건
        if exit_price is None:
            if sell_mode == "fg_sell" and fng_arr[i] >= fg_sell:
                exit_price = closes[i]
            elif sell_mode == "hold":
                days_since_first = i - first_buy_day
                if days_since_first >= hold_days:
                    exit_price = closes[i]

        if exit_price is not None:
            # 전체 청산
            pnl_per_coin = (exit_price - avg_entry) * leverage
            total_pnl = pnl_per_coin * total_coins / leverage  # 실제 PnL
            # 투입 원금 대비 수익
            invested_capital = total_notional / leverage  # 실제 투입금 (레버리지 전)
            pnl_ratio = (exit_price - avg_entry) / avg_entry * leverage
            balance += invested_capital * (1 + pnl_ratio)
            balance -= abs(balance * TAKER_FEE_RATE)

            sell_count += 1
            if pnl_ratio > 0:
                wins += 1
            total_hold += (i - first_buy_day)

            peak_balance = max(peak_balance, balance)
            dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
            max_dd = max(max_dd, dd)

            # 리셋
            total_notional = 0.0
            total_coins = 0.0
            in_position = False

            # 남은 매수 스킵 (청산 후 새 사이클)
            # DCA에서는 청산 후 추가 매수가 없으므로 종료
            break

    # 미청산 포지션
    if in_position and total_coins > 0:
        avg_entry = total_notional / total_coins
        exit_price = closes[-1]
        invested_capital = total_notional / leverage
        pnl_ratio = (exit_price - avg_entry) / avg_entry * leverage
        balance += invested_capital * (1 + pnl_ratio)
        balance -= abs(balance * TAKER_FEE_RATE)
        sell_count += 1
        if pnl_ratio > 0:
            wins += 1
        total_hold += (n - 1 - first_buy_day)
        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)

    total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    win_rate = (wins / sell_count * 100) if sell_count > 0 else 0
    avg_hold = (total_hold / sell_count) if sell_count > 0 else 0
    bnh_return = (closes[-1] - opens[0]) / opens[0] * 100 if opens[0] > 0 else 0

    years = n / 365.25
    if years > 0 and balance > 0:
        cagr = ((balance / INITIAL_BALANCE) ** (1 / years) - 1) * 100
    else:
        cagr = 0

    return {
        **params,
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "mdd": round(max_dd * 100, 2),
        "trades_buy": len(buys),
        "trades_sell": sell_count,
        "win_rate": round(win_rate, 1),
        "avg_hold_days": round(avg_hold, 1),
        "final_balance": round(balance, 2),
        "bnh_return": round(bnh_return, 2),
        "vs_bnh": round(total_return - bnh_return, 2),
    }


# ── 그리드 생성 ──────────────────────────────────────

def build_sell_combos():
    """매도 파라미터 조합 생성: 21가지."""
    combos = []
    # fg_sell
    for fg_sell in FG_SELL_THRESHOLDS:
        combos.append({"sell_mode": "fg_sell", "fg_sell": fg_sell,
                       "tp_pct": None, "sl_pct": None, "hold_days": None})
    # tp_sl: TP만
    for tp in TP_PCTS:
        combos.append({"sell_mode": "tp_sl", "fg_sell": None,
                       "tp_pct": tp, "sl_pct": None, "hold_days": None})
    # tp_sl: SL만
    for sl in SL_PCTS:
        combos.append({"sell_mode": "tp_sl", "fg_sell": None,
                       "tp_pct": None, "sl_pct": sl, "hold_days": None})
    # tp_sl: 둘다
    for tp in TP_PCTS:
        for sl in SL_PCTS:
            combos.append({"sell_mode": "tp_sl", "fg_sell": None,
                           "tp_pct": tp, "sl_pct": sl, "hold_days": None})
    # hold: 없이 (TP/SL 없이 순수 hold)
    for hd in HOLD_DAYS_LIST:
        combos.append({"sell_mode": "hold", "fg_sell": None,
                       "tp_pct": None, "sl_pct": None, "hold_days": hd})
    return combos


def build_buy_combos():
    """매수 파라미터 조합 생성."""
    combos = []
    for coin in COINS:
        for fg_buy in FG_BUY_THRESHOLDS:
            for max_inv in MAX_INVEST_PCTS:
                for lev in LEVERAGES:
                    for fund_f in FUNDING_FILTERS:
                        base = {
                            "coin": coin, "fg_buy": fg_buy,
                            "max_invest": max_inv, "leverage": lev,
                            "funding_filter": fund_f,
                        }
                        # fixed
                        for bp in BUY_PCTS:
                            combos.append({**base, "buy_mode": "fixed",
                                          "buy_pct": bp, "tier_scale": None})
                        # tiered
                        for ts in TIER_SCALES:
                            combos.append({**base, "buy_mode": "tiered",
                                          "buy_pct": None, "tier_scale": ts})
                        # equal
                        for bp in BUY_PCTS:
                            combos.append({**base, "buy_mode": "equal",
                                          "buy_pct": bp, "tier_scale": None})
    return combos


def _worker(args):
    """하나의 매수 조합에 대해 모든 매도 조합 실행."""
    coin_data, buy_params, sell_combos = args

    opens = coin_data["open"].to_numpy(dtype=float)
    highs = coin_data["high"].to_numpy(dtype=float)
    lows = coin_data["low"].to_numpy(dtype=float)
    closes = coin_data["close"].to_numpy(dtype=float)
    fng_arr = coin_data["fng"].to_numpy(dtype=float)
    funding_arr = coin_data["funding_rate"].to_numpy(dtype=float)
    n = len(closes)

    # 매수 로그 1회 계산
    buys, remaining_cash = compute_buy_log(
        opens, fng_arr, funding_arr, n, buy_params
    )

    # 각 매도 조건 순회
    results = []
    for sell_params in sell_combos:
        full_params = {**buy_params, **sell_params}
        result = run_sell_simulation(
            opens, highs, lows, closes, fng_arr, n,
            buys, remaining_cash, full_params
        )
        results.append(result)

    return results


# ── 메인 ─────────────────────────────────────────────

def main():
    refresh = "--refresh" in sys.argv

    print("=" * 60)
    print("Fear & Greed DCA 분할매수 전략 백테스트")
    print("=" * 60)

    # 1. 데이터 수집
    fng = fetch_fng(refresh=refresh)
    funding = fetch_funding_rate(refresh=refresh)

    # 2. 코인별 데이터 병합
    coin_data = {}
    for coin in COINS:
        print(f"\n[{coin.upper()}] 데이터 로딩...")
        daily = load_daily(coin)
        merged = merge_all_data(daily, fng, funding)
        coin_data[coin] = merged
        print(f"  일봉 {len(merged)}일, {merged['date'].min()} ~ {merged['date'].max()}")
        neg_days = (merged["funding_rate"] < 0).sum()
        print(f"  Funding < 0: {neg_days}일 ({neg_days/len(merged)*100:.1f}%)")

    # 3. 그리드 생성
    buy_combos = build_buy_combos()
    sell_combos = build_sell_combos()
    total = len(buy_combos) * len(sell_combos)
    print(f"\n매수 조합: {len(buy_combos):,}")
    print(f"매도 조합: {len(sell_combos)}")
    print(f"총 조합: {total:,}")

    # 4. 병렬 실행 (매수 조합 단위)
    tasks = [(coin_data[p["coin"]], p, sell_combos) for p in buy_combos]
    all_results = []
    workers = os.cpu_count() or 4

    print(f"\n병렬 실행 (workers={workers})...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, t): i for i, t in enumerate(tasks)}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0 or done == len(futures):
                pct = done / len(futures) * 100
                print(f"  매수조합 {done:,}/{len(futures):,} ({pct:.0f}%) → 결과 {done * len(sell_combos):,}건")
            try:
                all_results.extend(future.result())
            except Exception as e:
                idx = futures[future]
                print(f"  ERROR buy combo #{idx}: {e}")

    # 5. 결과 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(all_results)
    out_path = os.path.join(RESULTS_DIR, "fear_greed_dca_grid.csv")
    df.to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path} ({len(df):,}행)")

    # 6. 요약 출력
    valid = df[df["total_return"].notna()].copy()
    if valid.empty:
        print("유효한 결과 없음")
        return

    print("\n" + "=" * 60)
    print("TOP 20 조합 (수익률 기준)")
    print("=" * 60)
    top = valid.nlargest(20, "total_return")
    cols = ["coin", "fg_buy", "buy_mode", "buy_pct", "tier_scale",
            "max_invest", "leverage", "funding_filter",
            "sell_mode", "fg_sell", "tp_pct", "sl_pct", "hold_days",
            "total_return", "mdd", "trades_buy", "trades_sell",
            "win_rate", "cagr", "vs_bnh"]
    print(top[cols].to_string(index=False))

    # 매수 모드별
    print("\n" + "=" * 60)
    print("매수 모드별 평균 성과")
    print("=" * 60)
    print(valid.groupby("buy_mode").agg(
        avg_return=("total_return", "mean"),
        avg_mdd=("mdd", "mean"),
        avg_trades=("trades_buy", "mean"),
        best=("total_return", "max"),
    ).round(2))

    # 레버리지별
    print("\n" + "=" * 60)
    print("레버리지별 평균 성과")
    print("=" * 60)
    print(valid.groupby("leverage").agg(
        avg_return=("total_return", "mean"),
        avg_mdd=("mdd", "mean"),
        best=("total_return", "max"),
    ).round(2))

    # Funding 필터 효과
    print("\n" + "=" * 60)
    print("Funding 필터 효과")
    print("=" * 60)
    print(valid.groupby("funding_filter").agg(
        avg_return=("total_return", "mean"),
        avg_mdd=("mdd", "mean"),
        avg_trades=("trades_buy", "mean"),
        best=("total_return", "max"),
    ).round(2))

    # 코인별
    print("\n" + "=" * 60)
    print("코인별 평균 성과")
    print("=" * 60)
    print(valid.groupby("coin").agg(
        avg_return=("total_return", "mean"),
        avg_mdd=("mdd", "mean"),
        best=("total_return", "max"),
        bnh=("bnh_return", "first"),
    ).round(2))

    # fg_buy별
    print("\n" + "=" * 60)
    print("F&G 매수 threshold별")
    print("=" * 60)
    print(valid.groupby("fg_buy").agg(
        avg_return=("total_return", "mean"),
        avg_trades=("trades_buy", "mean"),
    ).round(2))


if __name__ == "__main__":
    main()
