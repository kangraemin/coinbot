"""
트렌드 신호 모니터.

백테스트에서 검증된 전략의 신호를 실시간 감지.
신호 변화(매수→매도, 매도→매수) 시:
  1. 텔레그램 알림
  2. 노션 DB 기록

실행: python3 analysis/signal_monitor.py
크론 예시 (4h봉 마감마다): 5 */4 * * * python3 /path/to/signal_monitor.py

── 전략 근거 (2017~2026 전체 히스토리 백테스트) ──────────────────────
BTC 1d EMA20/100:   12/14기간 B&H 초과 (86%), avg +114.9%, MDD 7.6%
BTC 4h EMA50/200:   12/14기간 B&H 초과 (86%), avg +175.8%
ETH 4h c>EMA200:    13/14기간 B&H 초과 (93%), avg +141.2%  ← 최고
ETH 4h EMA20/100:   13/14기간 B&H 초과 (93%), avg +130.1%
XRP 1d MACD26/52:   10/13기간 B&H 초과 (77%), avg +85.8%   (2x L)
SOL 4h c>EMA200:     7/11기간 B&H 초과 (64%), avg +389.4%
SOL 1d Supertrend:   7/11기간 B&H 초과 (64%), avg +532.1%
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 설정 ──────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
NOTION_TOKEN     = ""  # ~/.claude/.env 에서 로드
NOTION_DB_ID     = "316d4919-b6eb-8117-9ce3-cc9a384eb729"

# ~/.claude/.env 에서 NOTION_TOKEN 로드
_claude_env = Path.home() / ".claude" / ".env"
if _claude_env.exists():
    for line in _claude_env.read_text().splitlines():
        if line.startswith("NOTION_TOKEN="):
            NOTION_TOKEN = line.split("=", 1)[1].strip()
            break

# 신호 상태 저장 파일
STATE_FILE = Path(__file__).parent / "output" / "signal_state.json"

# ── 감시할 전략 ────────────────────────────────────────────
# 2017~2026 전체 히스토리 백테스트 기반 최적 파라미터
STRATEGIES = [
    # ── BTC ───────────────────────────────────────────────
    {
        "coin":     "BTC",
        "symbol":   "BTC/USDT",
        "tf":       "1d",
        "name":     "EMA20/100",
        "desc":     "일봉 EMA20이 EMA100 위로 → 매수 / 아래로 → 매도",
        "leverage": "3x",
        "backtest": "12/14기간 B&H 초과 (86%) | avg +114.9% | MDD 7.6%",
    },
    {
        "coin":     "BTC",
        "symbol":   "BTC/USDT",
        "tf":       "4h",
        "name":     "EMA50/200",
        "desc":     "4시간봉 EMA50이 EMA200 위로 → 매수 / 아래로 → 매도",
        "leverage": "2x",
        "backtest": "12/14기간 B&H 초과 (86%) | avg +175.8%",
    },
    # ── ETH ───────────────────────────────────────────────
    {
        "coin":     "ETH",
        "symbol":   "ETH/USDT",
        "tf":       "4h",
        "name":     "Price>EMA200",
        "desc":     "4시간봉 종가가 EMA200 위 → 매수 / 아래 → 매도",
        "leverage": "2x",
        "backtest": "13/14기간 B&H 초과 (93%) | avg +141.2%  ← ETH 최고 전략",
    },
    {
        "coin":     "ETH",
        "symbol":   "ETH/USDT",
        "tf":       "4h",
        "name":     "EMA20/100",
        "desc":     "4시간봉 EMA20이 EMA100 위로 → 매수 / 아래로 → 매도",
        "leverage": "2x",
        "backtest": "13/14기간 B&H 초과 (93%) | avg +130.1%",
    },
    # ── XRP ───────────────────────────────────────────────
    {
        "coin":     "XRP",
        "symbol":   "XRP/USDT",
        "tf":       "4h",
        "name":     "EMA20/100",
        "desc":     "4시간봉 EMA20이 EMA100 위로 → 매수 / 아래로 → 매도",
        "leverage": "2x",
        "backtest": "8/13기간 B&H 초과 (62%) | avg +74.8% | BTC/ETH 동일 계열",
    },
    {
        "coin":     "XRP",
        "symbol":   "XRP/USDT",
        "tf":       "4h",
        "name":     "EMA50/200",
        "desc":     "4시간봉 EMA50이 EMA200 위로 → 매수 / 아래로 → 매도",
        "leverage": "2x",
        "backtest": "8/13기간 B&H 초과 (62%) | avg +45.0% | BTC/ETH 동일 계열",
    },
    # ── SOL ───────────────────────────────────────────────
    {
        "coin":     "SOL",
        "symbol":   "SOL/USDT",
        "tf":       "4h",
        "name":     "Price>EMA200",
        "desc":     "4시간봉 종가가 EMA200 위 → 매수 / 아래 → 매도",
        "leverage": "2x",
        "backtest": "7/11기간 B&H 초과 (64%) | avg +389.4%",
    },
    {
        "coin":     "SOL",
        "symbol":   "SOL/USDT",
        "tf":       "1d",
        "name":     "Supertrend3",
        "desc":     "일봉 Supertrend(period=10, mult=3.0) 상승추세 → 매수 / 하강 → 매도",
        "leverage": "2x",
        "backtest": "7/11기간 B&H 초과 (64%) | avg +532.1%",
    },
]

TF_MAP = {"1d": "1d", "4h": "4h", "1h": "1h"}
CANDLE_LIMIT = 300  # MACD(26,52,18) + EMA200 계산에 충분한 봉 수


# ── 인디케이터 ─────────────────────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def supertrend_signal(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> bool:
    """True = 상승추세(롱), False = 하강추세"""
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(df)

    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))

    alpha = 2 / (period + 1)
    atr = np.empty(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]

    mid = (h + l) / 2
    bu = mid + mult * atr
    bl = mid - mult * atr
    fu = np.empty(n)
    fl = np.empty(n)
    fu[0], fl[0] = bu[0], bl[0]
    direction = np.ones(n, dtype=bool)

    for i in range(1, n):
        fl[i] = bl[i] if bl[i] > fl[i-1] or c[i-1] < fl[i-1] else fl[i-1]
        fu[i] = bu[i] if bu[i] < fu[i-1] or c[i-1] > fu[i-1] else fu[i-1]
        if c[i] > fu[i-1]:
            direction[i] = True
        elif c[i] < fl[i-1]:
            direction[i] = False
        else:
            direction[i] = direction[i-1]

    return bool(direction[-1])


def get_signal(df: pd.DataFrame, strategy_name: str) -> bool:
    """True = 롱(매수 유지), False = 청산(매도)"""
    c = df["close"]
    if strategy_name == "EMA20/100":
        return bool(ema(c, 20).iloc[-1] > ema(c, 100).iloc[-1])
    elif strategy_name == "EMA50/200":
        return bool(ema(c, 50).iloc[-1] > ema(c, 200).iloc[-1])
    elif strategy_name in ("Price>EMA200",):
        return bool(c.iloc[-1] > ema(c, 200).iloc[-1])
    elif strategy_name == "MACD26/52":
        ml = ema(c, 26) - ema(c, 52)
        hist = ml - ema(ml, 18)
        return bool(hist.iloc[-1] > 0)
    elif strategy_name == "Supertrend3":
        return supertrend_signal(df, period=10, mult=3.0)
    raise ValueError(f"unknown strategy: {strategy_name}")


# ── 데이터 로드 ────────────────────────────────────────────
def fetch_ohlcv(symbol: str, tf: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# ── 상태 저장/로드 ─────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


# ── 텔레그램 ───────────────────────────────────────────────
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ⚠ 텔레그램 설정 없음")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown",
    }, timeout=10)
    if resp.status_code == 200:
        print("  ✅ 텔레그램 전송")
    else:
        print(f"  ❌ 텔레그램 실패: {resp.text[:100]}")


# ── 노션 ───────────────────────────────────────────────────
def record_notion(s: dict, signal_type: str, price: float):
    if not NOTION_TOKEN:
        print("  ⚠ NOTION_TOKEN 없음")
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    title = f"{signal_type} {s['coin']} {s['tf']} {s['name']}"

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Title":    {"title":  [{"text": {"content": title}}]},
            "날짜":     {"date":   {"start": now}},
            "코인":     {"select": {"name": s["coin"]}},
            "신호":     {"select": {"name": signal_type}},
            "전략":     {"rich_text": [{"text": {"content": f"{s['name']} ({s['tf']}) — {s['desc']}"}}]},
            "타임프레임": {"select": {"name": s["tf"]}},
            "가격":     {"number": round(price, 2)},
        },
    }
    resp = requests.post(
        "https://api.notion.com/v1/pages",
        headers={
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=10,
    )
    if resp.status_code == 200:
        page_id = resp.json().get("id", "").replace("-", "")
        print(f"  ✅ 노션 기록 — https://notion.so/{page_id}")
    else:
        print(f"  ❌ 노션 실패: {resp.text[:150]}")


# ── 신호 발생 처리 ─────────────────────────────────────────
def on_signal_change(s: dict, new_signal: bool, price: float):
    signal_type = "📈 매수" if new_signal else "📉 매도"
    action      = "롱 진입" if new_signal else "청산 (매도)"

    print(f"\n  🔔 신호 변경: {signal_type}")

    msg = (
        f"{signal_type} *{s['coin']} {s['tf']} — {s['name']}*\n"
        f"현재가: ${price:,.2f}\n"
        f"액션: {action}\n"
        f"레버리지: {s['leverage']}\n"
        f"전략: {s['desc']}\n"
        f"백테스트: {s['backtest']}"
    )
    send_telegram(msg)
    record_notion(s, signal_type, price)


# ── 메인 ──────────────────────────────────────────────────
def main():
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== 신호 모니터 실행 [{now_str}] ===\n")

    state = load_state()
    changed = False

    for s in STRATEGIES:
        key = f"{s['coin']}_{s['tf']}_{s['name']}"
        print(f"▶ {s['coin']} {s['tf']} | {s['name']}")

        try:
            df = fetch_ohlcv(s["symbol"], TF_MAP[s["tf"]])
            current_signal = get_signal(df, s["name"])
            current_price  = float(df["close"].iloc[-1])

            prev_signal = state.get(key)
            signal_label = "📈 매수유지" if current_signal else "📉 관망/청산"
            print(f"  현재 신호: {signal_label} | 가격: ${current_price:,.2f}")

            if prev_signal is None:
                # 최초 실행 — 현재 상태 저장만
                print(f"  (최초 실행 — 상태 저장)")
                state[key] = current_signal
                changed = True
            elif current_signal != prev_signal:
                # 신호 변화!
                on_signal_change(s, current_signal, current_price)
                state[key] = current_signal
                changed = True
            else:
                print(f"  (변화 없음)")

        except Exception as e:
            print(f"  ❌ 오류: {e}")

    if changed:
        save_state(state)

    print(f"\n=== 완료 ===")


if __name__ == "__main__":
    main()
