"""잔액 + 포지션 + 주문 상태 점검."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.exchange import create_exchange
import config as cfg


async def main():
    exchange = create_exchange()
    try:
        bal = await exchange.fetch_balance()
        usdt = bal.get("USDT", {})
        print(f"USDT — total: {usdt.get('total', 0):.2f}  free: {usdt.get('free', 0):.2f}  used: {usdt.get('used', 0):.2f}")

        for symbol in cfg.SYMBOLS:
            positions = await exchange.fetch_positions([symbol])
            for p in positions:
                c = float(p.get("contracts") or 0)
                if abs(c) > 1e-8:
                    print(f"  [{symbol}] POSITION: {p.get('side')} {c} @ {p.get('entryPrice')} | margin={p.get('initialMargin')} notional={p.get('notional')}")

            orders = await exchange.fetch_open_orders(symbol)
            for o in orders:
                print(f"  [{symbol}] ORDER: {o.get('type')} {o.get('side')} {o.get('amount')} @ {o.get('price')} id={o['id']}")
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
