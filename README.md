# coinbot

Binance USDM 선물 자동매매 봇 — **1분봉 하락 역추세 롱 전략**

BTC / ETH / SOL / XRP 동시 운영 | 7x 레버리지 | Oracle Cloud 24/7

---

## 전략 개요

```
매 분봉 종료 시:
  entry_price = prev_close × (1 - 1.5%)
  → 리밋 롱 주문

체결 시:
  TP = entry × (1 + tp_pct%)   ← 리밋 매도
  SL = entry × (1 - sl_pct%)   ← STOP_MARKET
```

**핵심 아이디어**: 직전 봉 종가 대비 1.5% 이상 하락한 가격에 리밋 매수 대기.
단기 과매도 구간에서 반등을 노리는 역추세 전략.

---

## 현재 운영 파라미터

> 5년치 1분봉 데이터 그리드 서치 결과 기반 코인별 개별 최적화

| 코인 | entry | TP | SL | 레버리지 | 포지션 비율 |
|------|-------|----|----|---------|-----------|
| BTC  | -1.5% | +3.0% | -1.5% | 7x | 20% |
| ETH  | -1.5% | +2.0% | -0.5% | 7x | 20% |
| SOL  | -1.5% | +3.0% | -0.5% | 7x | 20% |
| XRP  | -1.5% | +3.0% | -0.5% | 7x | 20% |

- **BTC**: SL을 1.5%로 확대 → 노이즈 손절 방지, 승률 26.9% → 45.2%
- **ETH**: TP를 2.0%로 낮춤 → 빠른 익절, MDD 17.9% → 12.5%

---

## 백테스트 결과

### 분석 조건

- 데이터: Binance USDM 1분봉 5년치 (2022-01-01 ~ 2026-02-28)
- 수수료: 0.05% (maker/taker 평균)
- 그리드 서치: entry(5) × TP(4) × SL(5) × leverage(4) × pos(3) = **1,200 조합**
- 타임프레임 비교: 1m / 3m / 5m / 15m 병렬 서치 → **1분봉이 모든 코인에서 최우수**

---

### BTC — 5년 그리드 서치 요약

| 구분 | entry | TP | SL | Lev | 거래수 | 승률 | 수익률 | MDD | Sharpe |
|------|-------|----|----|-----|--------|------|--------|-----|--------|
| 수익률 1위 | 1.5% | 3.0% | 1.5% | 10x | 135 | 45.2% | **+325.7%** | 41.5% | 7.84 |
| Sharpe 1위 | 1.5% | 3.0% | 0.5% | 10x | 145 | 26.9% | +277.4% | 31.0% | **8.96** |
| 균형형 ✅  | 1.5% | 3.0% | 0.5% | 7x  | 145 | 26.9% | +161.6% | 22.7% | 7.11 |
| 안전형     | 1.0% | 1.5% | 0.5% | 7x  | 454 | 34.1% | +101.4% | 14.6% | 6.95 |

#### 파라미터별 핵심 인사이트

**entry_pct** — 가장 중요한 변수

| entry | 평균 수익률 | 평균 거래수 |
|-------|------------|------------|
| 0.3%  | -92.9%     | 4,533건    |
| 0.5%  | -82.6%     | 2,000건    |
| 0.8%  | -17.0%     | 709건      |
| 1.0%  | +5.3%      | 415건      |
| **1.5%** | **+38.5%** | **143건** |

> entry가 낮을수록 수수료 마찰로 손실 폭발. **1.5%가 임계점.**

**sl_pct** — entry 1.5% 한정

| SL   | 평균 수익률 | 평균 승률 |
|------|------------|----------|
| 0.5% | +49.1%     | 36.2%    |
| 1.5% | **+59.8%** | **57.3%** |
| 2.0% | +17.3%     | 58.9%    |

> SL 1.5% — 승률과 수익률 모두 균형. SL 2.0%는 한 번 손절 시 손실이 커 역효과.

---

### 멀티코인 비교 (동일 파라미터: entry 1.5% / TP 3.0% / SL 0.5% / 10x / 30%)

| 코인 | 거래수 | 승률 | 수익률 | MDD | Sharpe |
|------|--------|------|--------|-----|--------|
| BTC  | 145 | 26.9% | +277% | 31.0% | 8.96 |
| ETH  | 365 | 22.5% | +447% | 35.4% | 12.65 |
| SOL  | 1,021 | 22.4% | +11,091% ⚠️ | 49.7% | 223 |
| XRP  | 694  | 24.6% | +11,637% ⚠️ | 34.9% | 334 |

> SOL/XRP의 1만%+ 수익은 극단적 변동성으로 인한 in-sample 과적합 포함 가능성 있음.

### 코인별 최적 조합

| 코인 | entry | TP | SL | Lev | 거래수 | 수익률 | MDD | Sharpe |
|------|-------|----|----|-----|--------|--------|-----|--------|
| BTC  | 1.5%  | 3.0% | 1.5% | 7x | 135 | +193.6% | 30.9% | 6.27 |
| ETH  | 1.5%  | 2.0% | 0.5% | 10x | 371 | +550.3% | 25.1% | 21.9 |
| SOL  | 1.5%  | 3.0% | 0.5% | 7x | 1,021 | +3,196% | 37.7% | 84.7 |
| XRP  | 1.5%  | 3.0% | 0.5% | 7x | 694 | +3,138% | 25.6% | 122 |

### 타임프레임별 비교 (BTC, 최상위 조합 기준)

| TF  | 수익률 | 승률 | Sharpe | 비고 |
|-----|--------|------|--------|------|
| **1m**  | **+325.7%** | **45.2%** | **7.84** | ✅ 채택 |
| 15m | +256.6% | 27.8% | — | 거래수 적음 |
| 3m  | +209.0% | — | — | — |
| 5m  | +187.2% | — | — | — |

> 1분봉이 모든 지표에서 우수. 더 긴 타임프레임은 거래 빈도가 낮아 통계 신뢰도 하락.

---

## 아키텍처

```
main.py
 ├── data_loop(symbol × 4)     # WebSocket 1분봉 실시간 수신
 ├── strategy_loop             # 10초마다 4코인 순회
 │    └── _handle_symbol()
 │         ├── Flow A: 신규 진입 주문 (중복 주문 자동 감지/취소)
 │         ├── Flow B: 대기 주문 체결 확인 → TP/SL 설정
 │         └── Flow C: 포지션 종료 감지 → 일지 기록
 ├── risk_loop                 # 일일 손실 한도 (-5%) 감시
 ├── daily_report_loop         # 매일 오전 7시(KST) Telegram 리포트
 └── heartbeat_loop            # 1시간마다 봇 상태 알림
```

### 주문 흐름

```
prev_close 갱신
    ↓
entry_price = prev_close × 0.985
    ↓
리밋 매수 주문 ──► 미체결 대기
    ↓ 체결
TP: entry × 1.03  (리밋 매도)          ← BTC/SOL/XRP
SL: entry × 0.995 (STOP_MARKET)        ← ETH/SOL/XRP
    (BTC: SL entry × 0.985)
    ↓ 어느 쪽이든 체결
포지션 종료 감지 → journal.close_trade()
```

### 중복 주문 방지

재시작 또는 비정상 상황에서 기존 주문이 남아있을 경우 자동 처리:
- 주문 1개 → 상태 복원 후 재주문 생략
- 주문 2개 이상 → 초과분 자동 취소 후 첫 번째 주문 유지

---

## 프로젝트 구조

```
coinbot/
├── main.py             # 진입점 — asyncio 이벤트 루프, WebSocket
├── strategy.py         # 전략 로직 (진입/TP/SL/포지션 관리)
├── config.py           # 전체 파라미터 (코인별 SYMBOL_PARAMS)
├── exchange.py         # Binance USDM ccxt.pro 연결
├── risk.py             # 일일 손실 한도 / 미체결 주문 정리
├── journal.py          # SQLite 매매 일지 (trades.db)
├── report.py           # Telegram 알림 (진입/체결/종료/일보)
├── download_1m.py      # 1분봉 과거 데이터 다운로드
├── check_state.py      # 현재 포지션/잔고 확인 유틸
├── close_all.py        # 전체 포지션 즉시 청산 유틸
├── backtest/
│   ├── backtest_simple.py    # 단순 하락 진입 전략 백테스트
│   ├── backtest_grid.py      # 파라미터 그리드 서치 (1분봉)
│   └── backtest_grid_tf.py   # 타임프레임별 그리드 서치 (병렬)
├── results/
│   ├── analysis_report.md   # BTC 5년 그리드 서치 분석 보고서
│   └── grid_*.csv           # 코인별 × 타임프레임별 결과 CSV
└── tests/
    ├── conftest.py
    └── test_strategy.py
```

---

## 설치 및 실행

### 환경 설정

```bash
cp .env.example .env
# .env에 Binance API 키 및 Telegram 토큰 입력
```

```env
TESTNET=false
REAL_API_KEY=your_api_key
REAL_API_SECRET=your_api_secret
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
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

중복 실행 방지를 위해 `/tmp/coinbot.pid` PID 파일을 사용합니다.

---

## 백테스팅

### 1. 데이터 다운로드

```bash
python download_1m.py --symbol BTC/USDT:USDT --years 5
python download_1m.py --symbol ETH/USDT:USDT --years 5
python download_1m.py --symbol SOL/USDT:USDT --years 5
python download_1m.py --symbol XRP/USDT:USDT --years 5
```

### 2. 단일 파라미터 백테스트

```bash
cd backtest
python backtest_simple.py --coin btc --entry 1.5 --tp 3.0 --sl 1.5 --leverage 7
```

### 3. 파라미터 그리드 서치

```bash
cd backtest
python backtest_grid.py --coin btc --years 5 --output ../results/grid_btc.csv
```

### 4. 타임프레임 비교 (병렬)

```bash
cd backtest
python backtest_grid_tf.py  # 4코인 × 4타임프레임 병렬 실행
```

---

## 배포 (Oracle Cloud)

- 서버: Oracle Cloud Free Tier AMD (Ubuntu)
- 서비스: `systemd` — `coinbot.service`
- **`git push`하면 자동으로 Oracle에 배포됩니다** (`.git/hooks/post-push`)

```bash
# 로그 확인
ssh ubuntu@<oracle-ip> "journalctl -u coinbot -f"

# 서비스 상태 확인
ssh ubuntu@<oracle-ip> "systemctl status coinbot"

# 수동 재시작
ssh ubuntu@<oracle-ip> "sudo systemctl restart coinbot"
```

---

## 리스크 관리

| 항목 | 설정 | 비고 |
|------|------|------|
| 일일 최대 손실 | -5% | 초과 시 신규 진입 중단 |
| 코인당 포지션 비율 | 20% | 잔고 × 20% × 7x |
| 최대 동시 포지션 | 4개 | 코인당 1개 |
| 미체결 주문 정리 | 재시작 시 | non-reduceOnly 주문 전체 취소 |
| 주문 갱신 임계값 | 0.5% | prev_close 0.5% 이상 변동 시 재주문 |

---

## 주의사항

1. **백테스트 ≠ 실전**: 슬리피지, 호가 스프레드 미반영
2. **승률 26~45%**: 대부분의 거래가 손절됨 — 롱런을 위한 멘탈 관리 필수
3. **SOL/XRP 수익률**: in-sample 과적합 가능성 — 실전 포지션 비율 축소 권장
4. **레버리지 위험**: 7x 운용 시 -14% 하락으로 원금 전액 손실 가능

---

*Private*
