# Architecture

## 개요

바이낸스 선물 자동매매 봇. ccxt.pro + asyncio 기반 이벤트 드리븐 아키텍처.
GCP e2-micro (RAM 1GB) 환경을 고려해 메모리 최소화 설계.

## 구조

```
main.py
  └─ asyncio.gather()
       ├─ data_loop()     # WebSocket 1개 → 캔들 공유 버퍼 업데이트
       ├─ strategy_loop() # 공유 버퍼 감시 → 조건 충족 시 주문
       └─ risk_loop()     # 미체결 주문 10분 후 자동 취소
```

## 파일 구조

```
coinbot/
├── main.py          # 진입점, asyncio.gather, 재연결 로직
├── config.py        # 모든 상수/파라미터 (심볼, 레버리지, 지표 설정)
├── exchange.py      # ccxt.pro 초기화, 격리마진/레버리지 세팅 유틸
├── strategy.py      # 전략 A (스나이퍼 봇) 진입/익절 로직
├── risk.py          # 미체결 주문 관리, 일일 손실 한도 체크
├── journal.py       # SQLite 매매 일지 저장/조회
├── report.py        # 일일 리포트 생성 + Telegram 발송
├── .env             # API 키 (git 제외)
├── .env.example     # 키 템플릿
└── requirements.txt
```

## 데이터 흐름

```
[Binance Testnet WebSocket]
        │
        ▼
  data_loop() ──→ shared_state (전역 dict)
                    ├─ candles: deque(maxlen=200)  # 최근 200개 캔들
                    ├─ last_price: float
                    └─ indicators: dict            # BB, RSI, EMA, ATR, Volume

        │
        ▼
  strategy_loop() ──→ 조건 체크 ──→ [주문 실행] ──→ journal.py 기록
                                                  └─ Telegram 알림
        │
        ▼
  risk_loop() ──→ 미체결 주문 감시 ──→ 10분 초과 시 취소
```

## 전략 A: 트렌드 필터 평균회귀

**타임프레임:** 15분봉
**대상:** BTCUSDT (config에서 변경 가능)
**방향:** 롱 온리 (숏은 ENABLE_SHORT=True로 활성화 가능한 구조)

### 진입 조건 (5개 모두 충족)
1. 가격 > EMA 200 — 매크로 상승장 필터
2. 가격 < BB 하단 (20, 2σ) — 과매도 구간
3. RSI(14) < 35 — 모멘텀 약화
4. 현재 거래량 > 20봉 평균 × 1.2 — 신호 신뢰도
5. RSI 반등 시작 (RSI[0] > RSI[1]) — 바닥 확인

### 익절/손절 (ATR 기반 동적)
- 익절: 진입가 + ATR(14) × 2.0
- 손절: 진입가 - ATR(14) × 1.0

### 리스크 관리
- 동시 포지션: 최대 1개
- 레버리지: 3배 격리(Isolated)
- 1회 거래금액: config.py의 TRADE_AMOUNT_USDT
- 일일 손실 한도: -5% 도달 시 당일 자동 중단
- 미체결 지정가 10분 초과 시 자동 취소

## 메모리 최적화 (e2-micro 대응)

- WebSocket 연결 1개만 유지
- 캔들 버퍼: `deque(maxlen=200)` — 고정 크기
- pandas DataFrame 재생성 없이 deque로 지표 업데이트
- 단일 프로세스 (멀티프로세싱 금지)

## 예외 처리

- WebSocket 끊김: 5초 대기 후 재연결 (무한 재시도)
- API Rate Limit: ccxt 내장 rateLimit 사용
- 주문 실패: 로그 기록 + Telegram 알림, 봇 계속 실행
