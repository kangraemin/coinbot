"""
4H BB+RSI 양방향 평균회귀 신호 모니터.

백테스팅 검증 전략 (2022~2025, 3x×70%):
  BTC: +157.2% | MDD 25.2% | Calmar 6.24  (B&H +89.4%)
  ETH: +157.8% | MDD 23.9% | Calmar 6.61  (B&H -19.4%)
  XRP: +133.6% | MDD 22.3% | Calmar 5.99  (B&H +121.6%)

진입 조건:
  롱: close < BB_lower(20,2σ) AND RSI(14) < rsi_long  AND close > EMA(200)
  숏: close > BB_upper(20,2σ) AND RSI(14) > rsi_short AND close < EMA(200)

실행:
  python3 analysis/bb_rsi_signal.py

크론 예시 (4h봉 마감 5분 후):
  5 0,4,8,12,16,20 * * * /path/to/.venv/bin/python3 /path/to/analysis/bb_rsi_signal.py
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
NOTION_TOKEN     = ""
NOTION_DB_ID     = "316d4919-b6eb-8117-9ce3-cc9a384eb729"

# ~/.claude/.env 에서 NOTION_TOKEN 로드
_claude_env = Path.home() / ".claude" / ".env"
if _claude_env.exists():
    for line in _claude_env.read_text().splitlines():
        if line.startswith("NOTION_TOKEN="):
            NOTION_TOKEN = line.split("=", 1)[1].strip()
            break

# 신호 상태 저장 파일 (봉 단위 중복 알림 방지)
STATE_FILE = Path(__file__).parent / "output" / "bb_rsi_signal_state.json"
CANDLE_LIMIT = 250  # EMA(200) 웜업 포함 충분한 봉 수

# ── 코인별 전략 파라미터 ───────────────────────────────────
STRATEGIES = [
    {
        "coin":      "BTC",
        "symbol":    "BTC/USDT",
        "rsi_long":  30,
        "rsi_short": 65,
        "sl_mult":   2.0,
        "tp_mode":   "atr_3x",
        "leverage":  3,
        "pos_ratio": 0.70,
        "backtest":  "+157.2% | MDD 25.2% | Calmar 6.24 | B&H +89.4%",
    },
    {
        "coin":      "ETH",
        "symbol":    "ETH/USDT",
        "rsi_long":  25,
        "rsi_short": 65,
        "sl_mult":   2.0,
        "tp_mode":   "atr_2x",
        "leverage":  3,
        "pos_ratio": 0.70,
        "backtest":  "+157.8% | MDD 23.9% | Calmar 6.61 | B&H -19.4%",
    },
    {
        "coin":      "XRP",
        "symbol":    "XRP/USDT",
        "rsi_long":  25,
        "rsi_short": 65,
        "sl_mult":   2.0,
        "tp_mode":   "atr_3x",
        "leverage":  3,
        "pos_ratio": 0.70,
        "backtest":  "+133.6% | MDD 22.3% | Calmar 5.99 | B&H +121.6%",
    },
]


# ── 인디케이터 ─────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.Series:
    """BB(20,2σ), RSI(14), ATR(14), EMA(200) 계산 후 마지막 확정봉 값 반환."""
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)

    # Bollinger Bands (20, 2σ)
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std(ddof=1)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # RSI(14) Wilder EMA
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ATR(14) Wilder EMA
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean()

    # EMA(200)
    ema200 = close.ewm(span=200, adjust=False).mean()

    # 마지막 확정봉 (index -2: 현재 진행 중인 봉은 제외)
    i = -2
    return pd.Series({
        "timestamp": df["timestamp"].iloc[i],
        "close":     float(close.iloc[i]),
        "bb_upper":  float(bb_upper.iloc[i]),
        "bb_lower":  float(bb_lower.iloc[i]),
        "bb_mid":    float(bb_mid.iloc[i]),
        "rsi":       float(rsi.iloc[i]),
        "atr":       float(atr.iloc[i]),
        "ema200":    float(ema200.iloc[i]),
    })


# ── 데이터 로드 ────────────────────────────────────────────
def fetch_ohlcv(symbol: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, "4h", limit=limit)
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
def record_notion(s: dict, direction: str, ind: pd.Series):
    if not NOTION_TOKEN:
        print("  ⚠ NOTION_TOKEN 없음")
        return

    tp_mult_n = 3.0 if s["tp_mode"] == "atr_3x" else 2.0
    is_long   = direction == "📈 롱"
    rsi_thresh = s["rsi_long"] if is_long else s["rsi_short"]

    title = f"{direction} {s['coin']} 4h BB+RSI (RSI={ind['rsi']:.1f})"
    strategy_desc = (
        f"{'롱' if is_long else '숏'}: "
        f"{'close < BB_lower & RSI < ' + str(rsi_thresh) if is_long else 'close > BB_upper & RSI > ' + str(rsi_thresh)}"
        f" & EMA200 {'위' if is_long else '아래'} | "
        f"TP=ATR×{tp_mult_n:.0f}, SL=ATR×{s['sl_mult']:.0f}"
    )

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Title":      {"title":     [{"text": {"content": title}}]},
            "날짜":        {"date":      {"start": datetime.now(timezone.utc).strftime("%Y-%m-%d")}},
            "코인":        {"select":    {"name": s["coin"]}},
            "신호":        {"select":    {"name": direction}},
            "전략":        {"rich_text": [{"text": {"content": strategy_desc}}]},
            "타임프레임":   {"select":    {"name": "4h"}},
            "가격":        {"number":    round(ind["close"], 4)},
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
def on_signal(s: dict, direction: str, ind: pd.Series):
    is_long   = direction == "📈 롱"
    tp_mult_n = 3.0 if s["tp_mode"] == "atr_3x" else 2.0
    atr_val   = ind["atr"]

    tp_price = ind["close"] + atr_val * tp_mult_n if is_long else ind["close"] - atr_val * tp_mult_n
    sl_price = ind["close"] - atr_val * s["sl_mult"] if is_long else ind["close"] + atr_val * s["sl_mult"]
    rr       = tp_mult_n / s["sl_mult"]

    bb_key  = "bb_lower" if is_long else "bb_upper"
    bb_val  = ind[bb_key]

    msg = (
        f"{direction} *{s['coin']} 4H BB+RSI 진입 신호*\n"
        f"현재가: ${ind['close']:,.4f}\n"
        f"RSI: {ind['rsi']:.1f} | ATR: {atr_val:.4f}\n"
        f"BB_{'lower' if is_long else 'upper'}: ${bb_val:,.4f}\n"
        f"EMA200: ${ind['ema200']:,.4f}\n"
        f"TP: ${tp_price:,.4f} (ATR×{tp_mult_n:.0f})\n"
        f"SL: ${sl_price:,.4f} (ATR×{s['sl_mult']:.0f})\n"
        f"R:R = 1:{rr:.1f} | {s['leverage']}x × {s['pos_ratio']*100:.0f}%\n"
        f"백테스트: {s['backtest']}"
    )
    print(f"\n  🔔 신호: {direction}")
    send_telegram(msg)
    record_notion(s, direction, ind)


# ── 메인 ──────────────────────────────────────────────────
def main():
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== 4H BB+RSI 신호 모니터 [{now_str}] ===\n")

    state   = load_state()
    changed = False

    for s in STRATEGIES:
        print(f"▶ {s['coin']} 4h | BB+RSI 양방향 (롱RSI<{s['rsi_long']}, 숏RSI>{s['rsi_short']})")

        try:
            df      = fetch_ohlcv(s["symbol"])
            ind     = compute_indicators(df)
            bar_ts  = str(ind["timestamp"])

            print(
                f"  봉: {bar_ts} | 가격: ${ind['close']:,.4f} | "
                f"RSI: {ind['rsi']:.1f} | EMA200: ${ind['ema200']:,.4f}"
            )
            print(
                f"  BB_lower: ${ind['bb_lower']:,.4f} | BB_upper: ${ind['bb_upper']:,.4f}"
            )

            # 같은 봉은 재처리하지 않음
            if state.get(s["coin"]) == bar_ts:
                print(f"  (이미 처리된 봉 — 건너뜀)")
                continue

            # 진입 조건 체크
            long_cond = (
                ind["close"] < ind["bb_lower"] and
                ind["rsi"]   < s["rsi_long"]   and
                ind["close"] > ind["ema200"]
            )
            short_cond = (
                ind["close"] > ind["bb_upper"] and
                ind["rsi"]   > s["rsi_short"]  and
                ind["close"] < ind["ema200"]
            )

            if long_cond:
                on_signal(s, "📈 롱", ind)
            elif short_cond:
                on_signal(s, "📉 숏", ind)
            else:
                print(f"  (신호 없음)")

            # 처리 완료 — 봉 timestamp 기록
            state[s["coin"]] = bar_ts
            changed = True

        except Exception as e:
            print(f"  ❌ 오류: {e}")

    if changed:
        save_state(state)

    print(f"\n=== 완료 ===")


if __name__ == "__main__":
    main()
