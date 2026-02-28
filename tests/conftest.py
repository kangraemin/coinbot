import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def exchange():
    exc = AsyncMock()
    exc.amount_to_precision = MagicMock(side_effect=lambda sym, amt: amt)
    exc.price_to_precision = MagicMock(side_effect=lambda sym, price: price)
    exc.create_order = AsyncMock(return_value={"id": "order123"})
    exc.fetch_open_orders = AsyncMock(return_value=[])
    return exc


@pytest.fixture
def shared_state():
    return {}
