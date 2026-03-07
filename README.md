# coinbot

> Binance USDM Futures algorithmic trading bot — 4H BB+RSI dual-direction mean-reversion strategy.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Exchange](https://img.shields.io/badge/Exchange-Binance%20Futures-F0B90B)](https://binance.com)
[![Strategy](https://img.shields.io/badge/Strategy-BB%2BRSI%204H-22c55e)](.)
[![Live](https://img.shields.io/badge/Status-Live%20on%20Oracle%20Cloud-brightgreen)](.)
[![Tests](https://img.shields.io/badge/Tests-pytest-blue)](.)

---

## Strategy

**4H Bollinger Band + RSI mean-reversion — both long and short.**

The final strategy is the result of systematically evaluating 37 different approaches across 9 years of data (2017–2026) before arriving at this configuration.

| Signal | Condition |
|--------|-----------|
| **Long** | `close < BB_lower(20, 2σ)` AND `RSI(14) < threshold` AND `close > EMA(200)` |
| **Short** | `close > BB_upper(20, 2σ)` AND `RSI(14) > threshold` AND `close < EMA(200)` |
| **Take Profit** | `entry ± ATR(14) × tp_mult` via limit order |
| **Stop Loss** | `entry ∓ ATR(14) × sl_mult` via `STOP_MARKET` (Binance algo order) |
| **Timeout** | Force-close after 192h (48 × 4H bars) |

### Confirmed Parameters (2022–2026 backtest)

| Symbol | RSI Long | RSI Short | SL | TP | Leverage | Return | MDD | Calmar |
|--------|----------|-----------|----|----|----------|--------|-----|--------|
| BTC/USDT | < 30 | > 65 | 2.0× ATR | 3.0× ATR | 3x | +157.2% | 40% | 3.93 |
| ETH/USDT | < 25 | > 65 | 2.0× ATR | 2.0× ATR | 3x | +157.8% | 27% | 5.85 |
| XRP/USDT | < 25 | > 65 | 2.0× ATR | 3.0× ATR | 3x | +133.6% | 27% | 4.96 |

- **Position size**: 30% of available balance per coin (3 simultaneous max)
- **EMA200**: strict trend filter — long only above EMA200, short only below

---

## Research

The parameters above weren't guessed. Here's what it took to find them.

### Strategies Evaluated

33 active backtest scripts (+ 5 archived). Every approach was implemented, run, and measured before being accepted or rejected.

| Category | What Was Tested |
|----------|----------------|
| **Mean reversion** | BB+RSI (v1, v2, dual-direction), volume filters, robustness validation |
| **Trend following** | EMA200 crossover, MACD 1D, Golden Cross (multiple variants) |
| **Regime switching** | Per-coin EMA200 regime, BTC-global regime for all coins |
| **Signal quality** | Dual-tier (4H high-quality + 1H medium-quality), adaptive thresholds |
| **Portfolio** | Multi-coin correlation, dynamic vs fixed capital, asset rotation |
| **Timeframe** | 1H / 2H / 4H / 1D full comparison; 1m/3m/5m/15m intraday grid |
| **Validation** | Walk-forward analysis, year-by-year consistency, out-of-sample testing |

### Grid Search

- **5,000+ parameter combinations** tested across BTC, ETH, XRP, SOL
- **Grid axes**: RSI threshold × SL multiplier × TP multiplier × leverage × position ratio × EMA200 on/off
- **164 result files** generated (CSV + markdown summaries)
- **9 years of data**: 2017–2026, 1H candles resampled to target timeframes

### Key Findings

**Timeframe: 4H is definitively better**

Every coin, every metric. The gap widens as you go shorter.

| Timeframe | XRP return | ETH return | Noise level |
|-----------|-----------|-----------|-------------|
| 1H | +122.6% | +5.8% | High |
| 4H | **+407.9%** | **+85.5%** | Low |

**EMA200 filter is non-negotiable**

Removing it breaks the strategy entirely.

| Coin | With EMA200 | Without EMA200 |
|------|-------------|----------------|
| ETH | positive | **−45.6%** |
| SOL | positive | **−63.9%** |

**Dual-tier signal (4H + 1H) tested and rejected**

The intuition was sound: use 4H for big positions, 1H for smaller ones.
In practice, XRP 1H signals showed only **25% win rate** — adding them hurt more than helped regardless of position size. 4H standalone wins.

**Regime switching: promising but not yet live**

BTC/ETH switching to EMA50 trend-following in bull markets showed +25,000%+ on 9-year backtests. Not deployed because:
1. The bull market gains are concentrated in specific windows (high overfitting risk)
2. XRP regime switching loses −65% over 9 years — per-coin behavior diverges too much
3. Walk-forward validation pending before live deployment

**Walk-forward validation**

Strategy parameters were validated on rolling out-of-sample windows. Results are consistent across years — not a backtest artifact from a single lucky period.

---

## Architecture

```
main.py (coinbot service)
 ├── data_loop (× 3 symbols)    # WebSocket 4H candle feed
 ├── strategy_loop              # 30s tick — evaluates all symbols
 │    └── _handle_symbol()
 │         ├── Phase A: compute indicators → check signal → market entry
 │         └── Phase C: monitor TP/SL fill → timeout force-close
 ├── risk_loop                  # daily loss limit (−5%)
 ├── daily_report_loop          # 07:00 KST Telegram summary
 └── heartbeat_loop             # 1h status ping

fng_alert_service.py (coinbot-fng-alert service)
 └── F&G index polling → Telegram alert with trend chart & weekly avg

bot_listener.py (coinbot-listener service)
 └── Telegram /status command → real-time position & balance summary
```

### Order Flow

```
4H candle closes (confirmed bar only — no repainting)
    ↓
Compute: BB(20,2σ) / RSI(14) / ATR(14) / EMA(200)
    ↓
Signal? long / short / none
    ↓ signal
Market order → immediate fill
    ↓
TP  → limit order (reduceOnly)
SL  → STOP_MARKET closePosition=True  ← Binance algo order
    ↓
Fill detected or 192h timeout
    ↓
Close → SQLite journal + Telegram alert
```

**Restart recovery**: On bot restart, open positions and algo SL orders are restored via `fapiPrivateGetOpenAlgoOrders`. If SL is missing, it's re-registered using ATR back-calculated from entry price.

---

## Project Structure

```
coinbot/
├── main.py                 # entrypoint — asyncio event loop
├── config.py               # all strategy parameters in one place
├── fng_alert_service.py    # Fear & Greed index alert (standalone service)
├── bot_listener.py         # Telegram /status command listener (standalone service)
├── bot/
│   ├── strategy.py         # signal detection, order execution, state management
│   ├── exchange.py         # Binance USDM futures via ccxt.pro WebSocket
│   ├── risk.py             # daily loss limit watchdog
│   ├── journal.py          # SQLite trade log (data/trades.db)
│   ├── report.py           # Telegram notifications
│   └── fng_alert.py        # Fear & Greed index fetch + formatting
├── analysis/               # standalone analysis & research modules
│   ├── backtest_engine.py  # reusable backtest engine
│   ├── data_loader.py      # parquet data loading
│   ├── param_search.py     # parameter grid search
│   ├── signals/            # signal modules (BB, EMA, RSI, MACD, etc.)
│   └── output/             # generated results (not tracked)
├── backtest/               # 33 research scripts
│   ├── backtest_fullperiod_grid.py     # 2017–2026 full-period grid search
│   ├── backtest_bb_rsi_v2.py          # 4H BB+RSI core strategy
│   ├── backtest_switch_btcregime.py   # BTC-global regime switching
│   ├── backtest_dual_tier.py          # 4H+1H signal quality tiering
│   ├── backtest_wf_all.py             # walk-forward validation
│   ├── _archive/                      # deprecated versions
│   └── ...
├── tools/
│   ├── check_state.py     # position state inspector
│   ├── close_all.py       # emergency close all positions
│   └── download_1m.py     # historical 1m candle downloader
├── data/
│   ├── market/             # parquet candle data (2017–2026, not tracked)
│   ├── trades.db           # live trade history (not tracked)
│   └── results/            # 164 backtest output files
├── tests/                  # 5 test modules (19 tests)
│   ├── conftest.py
│   ├── test_strategy.py
│   ├── test_risk.py
│   ├── test_journal.py
│   ├── test_report.py
│   └── test_listener.py
├── coinbot-fng-alert.service   # systemd unit for F&G alert
└── coinbot-listener.service    # systemd unit for status listener
```

---

## Setup

**Requirements**: Python 3.10+, Binance Futures account with API access.

```bash
git clone https://github.com/kangraemin/coinbot
cd coinbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in your credentials
python main.py
```

```env
# .env.example
TESTNET=false
REAL_API_KEY=
REAL_API_SECRET=
TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
```

```bash
# run tests
pytest tests/ -v
```

---

## Risk

| Control | Setting |
|---------|---------|
| Daily loss limit | −5% (halts new entries) |
| Position size | 30% balance per coin |
| Max simultaneous | 3 positions |
| Timeout | 192h forced close |
| Leverage | 3x isolated |

**Disclaimer**: Futures trading with leverage carries significant risk of total loss. This is personal research code, not financial advice.
