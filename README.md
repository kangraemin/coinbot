# coinbot

BTC/USDT 선물 자동매매 봇 + 백테스팅 프레임워크

---

## 프로젝트 구조

```
coinbot/
├── main.py           # 봇 진입점 (WebSocket 이벤트 루프)
├── strategy.py       # 매매 전략 (ADX 레짐 + BB/RSI)
├── config.py         # 전체 파라미터 설정
├── exchange.py       # Binance USDM ccxt 연결
├── risk.py           # 리스크 관리 (타임아웃, 일일 손실 한도)
├── journal.py        # SQLite 매매 일지
├── report.py         # Telegram 알림
├── download_1m.py    # 1분봉 데이터 다운로드
├── data/             # 다운로드된 parquet 데이터 (gitignore)
└── backtest/         # 백테스팅 스크립트
    ├── backtest.py        # V1: ADX 평균회귀 (15분봉)
    ├── backtest_v2.py     # V2: EMA 추세추종
    ├── backtest_v3.py     # V3: RSI 다이버전스 + 볼륨
    ├── backtest_v4.py     # V4: STARS 멀티레짐
    ├── backtest_simple.py # 단순 하락 진입 전략 (1분봉)
    └── backtest_grid.py   # 파라미터 그리드 서치 (1분봉)
```

---

## 봇 실행

### 환경 설정

```bash
cp .env.example .env
# .env에 Binance API 키 및 Telegram 토큰 입력
```

### 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 실행

```bash
python main.py
```

---

## 백테스팅

### 1. 데이터 다운로드

```bash
# BTC 5년치
python download_1m.py --symbol BTC/USDT:USDT --years 5

# 멀티코인
python download_1m.py --symbol ETH/USDT:USDT --years 5
python download_1m.py --symbol SOL/USDT:USDT --years 5
```

### 2. 단순 전략 백테스트

```bash
cd backtest
python backtest_simple.py --entry 1.5 --tp 3.0 --sl 1.5 --leverage 10
```

### 3. 파라미터 그리드 서치

```bash
cd backtest
python backtest_grid.py --coin btc --output ../grid_results_btc.csv
python backtest_grid.py --coin eth --output ../grid_results_eth.csv
```

---

## 백테스팅 결과 요약 (5년, BTC)

> 분석 기간: 2022-01-01 ~ 2026-02-28 | 전략: prev_close 대비 N% 하락 시 롱 진입

| 구분 | entry% | TP% | SL% | Lev | 수익률 | MDD | Sharpe | 거래수 |
|------|--------|-----|-----|-----|--------|-----|--------|--------|
| 공격형 | 1.5% | 3.0% | 1.5% | 10x | +325.73% | 41.5% | 7.84 | 135건 |
| 균형형 | 1.5% | 3.0% | 0.5% | 7x | +161.56% | 22.7% | 7.11 | 145건 |
| 안전형 | 1.0% | 1.5% | 0.5% | 7x | +101.37% | 14.6% | 6.95 | 454건 |

전체 분석 보고서: [`analysis_report.md`](analysis_report.md)

---

## 봇 전략

**레짐 필터**: ADX(14) < 25 (횡보장)에서만 진입

**진입 조건**:
- BB 하단 터치 (과매도)
- RSI < 35
- RSI 상승 전환

**청산**:
- TP: BB 중심선
- SL: ATR × 1.5
- 미체결 타임아웃: 10분

---

## 라이선스

Private
