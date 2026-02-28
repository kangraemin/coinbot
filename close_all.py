"""모든 포지션/주문 정리 스크립트 — 봇 재시작 전 실행."""

import asyncio
import logging

from exchange import create_exchange
import config as cfg

logging.basicConfig(format=cfg.LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    exchange = create_exchange()
    try:
        # 잔액 확인
        bal = await exchange.fetch_balance()
        total = float(bal.get("total", {}).get("USDT", 0))
        free = float(bal.get("free", {}).get("USDT", 0))
        logger.info("잔액 — total: %.2f USDT, free: %.2f USDT", total, free)

        # 1) 모든 심볼 미체결 주문 취소
        for symbol in cfg.SYMBOLS:
            try:
                orders = await exchange.fetch_open_orders(symbol)
                for o in orders:
                    await exchange.cancel_order(o["id"], symbol)
                    logger.info("[%s] 주문 취소: %s (%s)", symbol, o["id"], o.get("type"))
                if orders:
                    logger.info("[%s] %d개 주문 취소 완료", symbol, len(orders))
                else:
                    logger.info("[%s] 미체결 주문 없음", symbol)
            except Exception as e:
                logger.error("[%s] 주문 취소 오류: %s", symbol, e)

        # 2) 모든 포지션 시장가 청산
        for symbol in cfg.SYMBOLS:
            try:
                positions = await exchange.fetch_positions([symbol])
                for pos in positions:
                    contracts = float(pos.get("contracts") or 0)
                    if abs(contracts) < 1e-8:
                        continue
                    side = pos.get("side", "")
                    close_side = "sell" if side == "long" else "buy"
                    order = await exchange.create_order(
                        symbol, "market", close_side, abs(contracts),
                        None, {"reduceOnly": True}
                    )
                    logger.info("[%s] 포지션 청산 — %s %.6f @ market (order: %s)",
                                symbol, side, abs(contracts), order["id"])
            except Exception as e:
                logger.error("[%s] 포지션 청산 오류: %s", symbol, e)

        # 최종 잔액 확인
        await asyncio.sleep(2)
        bal2 = await exchange.fetch_balance()
        total2 = float(bal2.get("total", {}).get("USDT", 0))
        free2 = float(bal2.get("free", {}).get("USDT", 0))
        logger.info("정리 후 잔액 — total: %.2f USDT, free: %.2f USDT", total2, free2)

    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
