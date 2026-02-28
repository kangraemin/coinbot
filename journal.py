"""매매 일지 — SQLite 기록/조회."""

import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = "trades.db"


def init_db() -> None:
    """매매 기록 테이블을 생성한다."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            side TEXT NOT NULL,
            symbol TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            amount REAL NOT NULL,
            fee REAL DEFAULT 0,
            pnl REAL DEFAULT 0,
            order_id TEXT,
            tp_order_id TEXT,
            sl_order_id TEXT,
            status TEXT DEFAULT 'open'
        )
        """
    )
    conn.commit()
    conn.close()
    logger.info("매매 일지 DB 초기화 완료")


def record_trade(
    side: str,
    symbol: str,
    entry_price: float,
    amount: float,
    order_id: str,
    tp_order_id: str | None = None,
    sl_order_id: str | None = None,
) -> int:
    """매매 기록을 저장하고 row ID를 반환한다."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """
        INSERT INTO trades (timestamp, side, symbol, entry_price, amount, order_id, tp_order_id, sl_order_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            side,
            symbol,
            entry_price,
            amount,
            order_id,
            tp_order_id,
            sl_order_id,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    logger.info("매매 기록 저장 — ID: %d, %s %.6f @ %.2f", row_id, side, amount, entry_price)
    return row_id


def close_trade(
    trade_id: int,
    exit_price: float,
    fee: float,
    pnl: float,
) -> None:
    """매매를 종료 처리한다."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        UPDATE trades SET exit_price = ?, fee = ?, pnl = ?, status = 'closed'
        WHERE id = ?
        """,
        (exit_price, fee, pnl, trade_id),
    )
    conn.commit()
    conn.close()
    logger.info("매매 종료 — ID: %d, 종가: %.2f, PnL: %.2f", trade_id, exit_price, pnl)


def get_daily_trades() -> list[dict]:
    """오늘(UTC) 매매 기록을 조회한다."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM trades WHERE timestamp LIKE ?",
        (f"{today}%",),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_daily_pnl() -> float:
    """오늘(UTC) 총 PnL을 반환한다."""
    trades = get_daily_trades()
    return sum(t.get("pnl", 0) for t in trades)


def get_open_trade(symbol: str | None = None) -> dict | None:
    """현재 열린 매매 기록을 반환한다. symbol 지정 시 해당 심볼만 조회."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if symbol:
        row = conn.execute(
            "SELECT * FROM trades WHERE status = 'open' AND symbol = ? ORDER BY id DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    conn.close()
    return dict(row) if row else None


# 모듈 로드 시 DB 초기화
init_db()
