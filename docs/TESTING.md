# Testing

## 환경

- **Testnet 우선**: 바이낸스 Futures Testnet에서 실행 검증
- 유닛 테스트: pytest (지표 계산, 조건 로직 검증)

## 테스트 실행

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## Testnet 실행

```bash
# .env에 TESTNET=true 설정 후
python main.py
```

## 검증 항목

### Phase 1 완료 조건
- [ ] ccxt.pro로 Testnet WebSocket 연결 성공
- [ ] 캔들 데이터 수신 및 shared_state 업데이트 확인
- [ ] 재연결 로직 동작 확인 (네트워크 강제 차단 후 복구)
- [ ] 격리마진 + 3배 레버리지 세팅 API 호출 성공

### Phase 2 완료 조건
- [ ] EMA200/50, BB, RSI, ATR, Volume 지표 계산 정확도 검증
- [ ] 5개 진입 조건 체크 로직 단위 테스트
- [ ] Testnet에서 지정가 매수 주문 생성 확인
- [ ] 익절/손절 주문 자동 배치 확인

### Phase 3 완료 조건
- [ ] 미체결 주문 10분 후 자동 취소 동작 확인
- [ ] 일일 손실 -5% 도달 시 거래 중단 확인
- [ ] SQLite 매매 일지 기록 확인
- [ ] Telegram 알림 수신 확인
- [ ] 24시간 무중단 실행 안정성 확인

## 수수료 검증

Testnet 수수료율 (실거래와 동일):
- Maker(지정가): 0.02%
- 매매 일지에 수수료 차감 후 실수익 기록 확인
