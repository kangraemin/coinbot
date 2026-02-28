"""1분봉 하락 진입 전략 — prev_close -1.5% 리밋 롱, TP +3%, SL -0.5%.

동작:
  - 매 분봉 종료 시 prev_close 갱신
  - 진입 트리거 = prev_close * (1 - ENTRY_DROP_PCT%)
  - 리밋 매수 주문 → 체결 시 TP(리밋 매도) + SL(스탑마켓) 설정
  - 포지션 종료 감지 → 저널 업데이트
"""

import asyncio
import logging

import config as cfg
import journal
import report

logger = logging.getLogger(__name__)

# ── 심볼별 포지션 상태 ────────────────────────────────
_pos: dict[str, dict] = {
    symbol: {
        "has_position": False,
        "trade_id": None,
        "entry_price": 0.0,
        "amount": 0.0,
        "entry_order_id": None,   # 대기 중인 진입 주문
        "tp_order_id": None,
        "sl_order_id": None,
        "last_prev_close": 0.0,   # 마지막으로 주문 낸 기준 prev_close
    }
    for symbol in cfg.SYMBOLS
}


async def _get_total_balance(exchange) -> float:
    """USDT 총 잔액 반환."""
    try:
        data = await exchange.fetch_balance()
        return float(data.get("total", {}).get("USDT", 0) or 0)
    except Exception as e:
        logger.error("잔액 조회 실패: %s", e)
        return 0.0


async def _cancel_safe(exchange, order_id: str, symbol: str) -> None:
    """주문 취소 — 이미 취소/체결된 경우 무시."""
    try:
        await exchange.cancel_order(order_id, symbol)
        logger.debug("[%s] 주문 취소: %s", symbol, order_id)
    except Exception:
        pass


def _sym_params(symbol: str) -> dict:
    """심볼별 파라미터 반환 (없으면 전역 기본값)."""
    p = cfg.SYMBOL_PARAMS.get(symbol, {})
    return {
        "entry_pct": p.get("entry_pct", cfg.ENTRY_DROP_PCT),
        "tp_pct":    p.get("tp_pct",    cfg.TP_PCT),
        "sl_pct":    p.get("sl_pct",    cfg.SL_PCT),
    }


async def _place_entry(exchange, symbol: str, prev_close: float) -> None:
    """진입 리밋 주문 발행."""
    params = _sym_params(symbol)
    entry_price = prev_close * (1 - params["entry_pct"] / 100)

    balance = await _get_total_balance(exchange)
    if balance <= 0:
        logger.warning("[%s] 잔액 없음, 주문 건너뜀", symbol)
        return

    raw_amount = balance * cfg.POSITION_RATIO * cfg.LEVERAGE / entry_price
    try:
        amount = float(exchange.amount_to_precision(symbol, raw_amount))
    except Exception:
        amount = round(raw_amount, 6)

    if amount * entry_price < 10:
        logger.warning("[%s] 최소 주문금액 미달 (%.2f USDT notional)", symbol, amount * entry_price)
        return

    tp_price_approx = entry_price * (1 + params["tp_pct"] / 100)
    sl_price_approx = entry_price * (1 - params["sl_pct"] / 100)

    try:
        order = await exchange.create_order(
            symbol, "limit", "buy", amount, entry_price,
            {"timeInForce": "GTC"},
        )
        _pos[symbol]["entry_order_id"] = order["id"]
        _pos[symbol]["last_prev_close"] = prev_close
        logger.info(
            "[%s] 진입 주문 — %.6f @ %.4f (prev_close=%.4f, -%.1f%%)",
            symbol, amount, entry_price, prev_close, params["entry_pct"],
        )
        await report.send_entry_order_alert(
            symbol, entry_price, amount, tp_price_approx, sl_price_approx
        )
    except Exception as e:
        logger.error("[%s] 진입 주문 실패: %s", symbol, e)


async def _place_tp_sl(exchange, symbol: str, entry_price: float, amount: float) -> None:
    """TP 리밋 매도 + SL 스탑마켓 주문 발행."""
    params = _sym_params(symbol)
    tp_price = entry_price * (1 + params["tp_pct"] / 100)
    sl_price = entry_price * (1 - params["sl_pct"] / 100)

    try:
        tp_price = float(exchange.price_to_precision(symbol, tp_price))
        sl_price = float(exchange.price_to_precision(symbol, sl_price))
    except Exception:
        tp_price = round(tp_price, 4)
        sl_price = round(sl_price, 4)

    try:
        tp_order = await exchange.create_order(
            symbol, "limit", "sell", amount, tp_price,
            {"reduceOnly": True, "timeInForce": "GTC"},
        )
        _pos[symbol]["tp_order_id"] = tp_order["id"]
        logger.info("[%s] TP 주문 — @ %.4f", symbol, tp_price)
    except Exception as e:
        logger.error("[%s] TP 주문 실패: %s", symbol, e)

    try:
        sl_order = await exchange.create_order(
            symbol, "STOP_MARKET", "sell", amount, None,
            {"stopPrice": sl_price, "reduceOnly": True},
        )
        _pos[symbol]["sl_order_id"] = sl_order["id"]
        logger.info("[%s] SL 주문 — @ %.4f", symbol, sl_price)
    except Exception as e:
        logger.error("[%s] SL 주문 실패: %s", symbol, e)

    await report.send_trade_alert(
        "long", entry_price, amount, tp_price, sl_price, symbol=symbol
    )


async def _handle_symbol(exchange, symbol: str, shared_state: dict) -> None:
    """단일 심볼의 전략 상태를 처리한다."""
    state = _pos[symbol]
    sym = shared_state.get(symbol, {})
    prev_close = sym.get("prev_close", 0.0)

    logger.info("[%s] handle — pos=%s entry_oid=%s",
                symbol, state["has_position"], state["entry_order_id"])

    if prev_close <= 0:
        return

    # ── C: 포지션 보유 중 → 종료 여부 확인 ──────────────
    if state["has_position"]:
        try:
            positions = await exchange.fetch_positions([symbol])
            open_pos = next(
                (p for p in positions if abs(float(p.get("contracts") or 0)) > 1e-8),
                None,
            )

            if open_pos is None:
                logger.info("[%s] 포지션 종료 감지", symbol)

                for oid in (state["tp_order_id"], state["sl_order_id"]):
                    if oid:
                        await _cancel_safe(exchange, oid, symbol)

                if state["trade_id"]:
                    exit_price = sym.get("last_price", state["entry_price"])
                    pnl = (exit_price - state["entry_price"]) * state["amount"] * cfg.LEVERAGE
                    fee = state["entry_price"] * state["amount"] * 0.0005 * 2
                    journal.close_trade(state["trade_id"], exit_price, fee, pnl)
                    await report.send_close_alert(
                        state["entry_price"], exit_price, pnl, fee, symbol=symbol
                    )

                state.update({
                    "has_position": False, "trade_id": None,
                    "entry_price": 0.0, "amount": 0.0,
                    "tp_order_id": None, "sl_order_id": None,
                    "entry_order_id": None, "last_prev_close": 0.0,
                })
        except Exception as e:
            logger.error("[%s] 포지션 조회 오류: %s", symbol, e)
        return

    # ── B: 진입 주문 대기 중 ──────────────────────────────
    if state["entry_order_id"]:
        try:
            order = await exchange.fetch_order(state["entry_order_id"], symbol)
            status = order.get("status", "")

            if status == "closed":
                filled_price = float(order.get("average") or order.get("price") or 0)
                filled_amount = float(order.get("filled") or order.get("amount") or 0)

                if filled_price <= 0 or filled_amount <= 0:
                    logger.error("[%s] 체결 정보 이상 — 건너뜀", symbol)
                    state["entry_order_id"] = None
                    return

                trade_id = journal.record_trade(
                    "long", symbol, filled_price, filled_amount, order["id"]
                )
                state.update({
                    "has_position": True, "trade_id": trade_id,
                    "entry_price": filled_price, "amount": filled_amount,
                    "entry_order_id": None,
                })
                logger.info("[%s] 진입 체결 — %.6f @ %.4f", symbol, filled_amount, filled_price)
                await _place_tp_sl(exchange, symbol, filled_price, filled_amount)

            elif status == "open":
                # prev_close가 0.5% 이상 변하면 주문 갱신
                lpc = state["last_prev_close"]
                if lpc > 0 and abs(prev_close - lpc) / lpc > 0.005:
                    logger.debug("[%s] prev_close 변경 (%.4f→%.4f), 주문 갱신", symbol, lpc, prev_close)
                    await _cancel_safe(exchange, state["entry_order_id"], symbol)
                    state["entry_order_id"] = None
                    state["last_prev_close"] = 0.0
                    await _place_entry(exchange, symbol, prev_close)

            else:
                # canceled / expired
                state["entry_order_id"] = None
                state["last_prev_close"] = 0.0

        except Exception as e:
            logger.error("[%s] 주문 조회 오류: %s", symbol, e)
        return

    # ── A: 신규 진입 주문 ─────────────────────────────────
    await _place_entry(exchange, symbol, prev_close)


async def _restore_state(exchange) -> None:
    """재시작 시 기존 진입 주문 취소 + 포지션만 복원."""
    for symbol in cfg.SYMBOLS:
        state = _pos[symbol]
        try:
            # 기존 포지션 복원 (포지션이 있으면 새 진입 주문 안 냄)
            positions = await exchange.fetch_positions([symbol])
            open_pos = next(
                (p for p in positions if abs(float(p.get("contracts") or 0)) > 1e-8),
                None,
            )
            if open_pos:
                state["has_position"] = True
                state["entry_price"] = float(open_pos.get("entryPrice") or 0)
                state["amount"] = abs(float(open_pos.get("contracts") or 0))
                logger.info("[%s] 포지션 복원 — %.6f @ %.4f", symbol, state["amount"], state["entry_price"])
                continue

            # 기존 진입 주문(non-reduceOnly) 전부 취소 — 재시작 시 항상 새로 주문
            orders = await exchange.fetch_open_orders(symbol)
            for o in orders:
                if not o.get("reduceOnly", False):
                    await _cancel_safe(exchange, o["id"], symbol)
                    logger.info("[%s] 재시작 — 기존 진입 주문 취소: %s", symbol, o["id"])
        except Exception as e:
            logger.error("[%s] 상태 초기화 오류: %s", symbol, e)


async def strategy_loop(exchange, shared_state: dict) -> None:
    """전략 메인 루프 — 10초마다 모든 심볼 순회."""
    logger.info("전략 루프 시작 — %s", cfg.SYMBOLS)
    await asyncio.sleep(5)  # 초기 캔들 로드 대기
    await _restore_state(exchange)  # 재시작 시 기존 상태 복원

    while True:
        try:
            if not shared_state.get("trading_halted", False):
                for symbol in cfg.SYMBOLS:
                    await _handle_symbol(exchange, symbol, shared_state)
            else:
                logger.debug("거래 중단 상태 — 전략 루프 대기")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("전략 루프 오류: %s", e)

        await asyncio.sleep(10)
