# coinbot - Development Guide

프로젝트 전체 규칙의 허브. 모든 에이전트(`~/.claude/agents/`)가 이 문서를 먼저 읽는다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 앱 이름 | coinbot |
| 플랫폼 | Python CLI (GCP e2-micro 배포) |
| 언어 | Python 3.11+ |
| 거래소 | 바이낸스 Futures Testnet → 실거래 전환 예정 |
| 전략 | 트렌드 필터 평균회귀 (EMA200 + BB + RSI + ATR + Volume) |
| 방향 | 롱 온리 (숏 추가 가능한 구조) |
| 레버리지 | 3배 격리(Isolated) |

---

## 2. 상세 가이드

| 문서 | 내용 |
|------|------|
| [Architecture](docs/ARCHITECTURE.md) | 아키텍처, 파일 구조, 전략, 데이터 흐름 |
| [Coding Conventions](docs/CODING_CONVENTIONS.md) | 네이밍, 비동기 규칙, 주문 규칙, 에러 처리 |
| [Testing](docs/TESTING.md) | 테스트 방법, Testnet 실행, 완료 조건 |
| [Git Rules](~/.claude/rules/git-rules.md) | 커밋, 푸시, PR 규칙 (글로벌) |

---

## 3. 개발 Phase

### Phase 1: 뼈대 & 연결
- config.py, exchange.py 작성
- ccxt.pro WebSocket 연결 + 캔들 수신
- shared_state 구조 + 재연결 로직
- main.py 비동기 루프 뼈대

### Phase 2: 전략 & 주문
- 초기 과거 캔들 100개 로드 + 지표 초기화
- strategy.py: 5개 조건 체크 + 지정가 주문 실행
- ATR 기반 익절/손절 주문 자동 배치

### Phase 3: 리스크 & 매매 일지
- risk.py: 미체결 10분 취소, 일일 손실 한도
- journal.py: SQLite 매매 기록
- report.py: 일일 요약 + Telegram 알림
- .env.example, requirements.txt 정리

---

## 4. 기술 스택

```
ccxt[pro]       # 바이낸스 WebSocket + REST API
pandas          # 데이터 처리
pandas_ta       # 기술 지표 (BB, RSI, EMA, ATR)
python-dotenv   # 환경변수
aiohttp         # Telegram 비동기 HTTP
sqlite3         # 매매 일지 (표준 라이브러리)
```

---

## 5. Git 컨벤션

`~/.claude/rules/git-rules.md` 참조.

- `main` 브랜치: 안정 버전
- `feature/<phase명>` 브랜치로 각 Phase 작업
- Phase 완료 시 main 머지
