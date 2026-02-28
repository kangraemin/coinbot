import sqlite3
import pytest

from bot import journal


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Redirect journal DB to a temp file and initialize tables."""
    db_file = str(tmp_path / "test_trades.db")
    monkeypatch.setattr(journal, "DB_PATH", db_file)
    journal.init_db()
    return db_file


def test_record_trade(tmp_db):
    """record_trade inserts a row with status='open'."""
    trade_id = journal.record_trade("long", "BTC/USDT:USDT", 50000.0, 0.1, "ord1")

    assert trade_id is not None and trade_id > 0

    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT side, symbol, entry_price, amount, order_id, status FROM trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "long"
    assert row[1] == "BTC/USDT:USDT"
    assert row[2] == 50000.0
    assert row[3] == 0.1
    assert row[4] == "ord1"
    assert row[5] == "open"


def test_close_trade(tmp_db):
    """close_trade updates status to 'closed' and sets exit_price/pnl."""
    trade_id = journal.record_trade("long", "BTC/USDT:USDT", 50000.0, 0.1, "ord1")
    journal.close_trade(trade_id, 51000.0, 0.5, 100.0)

    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT status, exit_price, fee, pnl FROM trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "closed"
    assert row[1] == 51000.0
    assert row[2] == 0.5
    assert row[3] == 100.0
