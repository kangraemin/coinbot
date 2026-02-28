# Coding Conventions

## 언어 & 버전

- Python 3.11+
- 타입 힌트 적극 사용 (`def foo(x: float) -> None`)

## 네이밍

- 변수/함수: `snake_case`
- 상수: `UPPER_SNAKE_CASE` (config.py에만 정의)
- 클래스: `PascalCase` (최소화, 필요 시만)

## 비동기 규칙

- 모든 I/O 작업은 `async/await` 사용
- `time.sleep()` 금지 → `asyncio.sleep()` 사용
- 블로킹 코드 금지 (requests 등)

## 주문 규칙

- 시장가(Market) 주문 코드 금지
- 모든 주문은 지정가(Limit/postOnly)
- 주문 전 반드시 격리마진 + 레버리지 세팅 호출

## 설정값

- 하드코딩 금지. 모든 파라미터는 `config.py`에서 가져옴
- API 키는 `.env`에서만 로드 (`python-dotenv`)

## 로깅

- `print()` 금지 → `logging` 모듈 사용
- 포맷: `%(asctime)s [%(levelname)s] %(message)s`
- 레벨: INFO (일반), WARNING (주문 실패), ERROR (재연결 등)

## 에러 처리

- 봇을 죽이는 예외 최소화 — try/except로 감싸고 로그 후 계속 실행
- WebSocket 재연결: 5초 sleep 후 재시도
- 주문 실패: 로그 + Telegram 알림, 포지션 상태 재확인

## 파일별 책임

각 파일은 단일 책임. 파일 간 순환 참조 금지.
의존성 방향: `main → strategy/risk → exchange/journal → config`
