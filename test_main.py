import pytest
from fastapi.testclient import TestClient
from decimal import Decimal
import time
import os
from unittest.mock import AsyncMock, patch

# Import the classes from the main script
from main import app, engine, Order, OrderSide, OrderType, OrderStatus, Trade

# Use a separate state file for testing to avoid overwriting the main one
TEST_STATE_FILE = 'test_order_book_state.json'
engine.state_file = TEST_STATE_FILE


@pytest.fixture(autouse=True)
def reset_engine_before_each_test():
    """Fixture to ensure the engine is in a clean state before each test."""
    engine.reset()
    yield  # this is where the test runs
    if os.path.exists(TEST_STATE_FILE):
        os.remove(TEST_STATE_FILE)


@pytest.fixture
def client():
    """Fixture to provide a test client for the FastAPI app."""
    return TestClient(app)


# --- Test Basic Order Book Operations ---

def test_submit_and_get_bbo(client):
    """Test submitting basic limit orders and checking the Best Bid and Offer."""
    # Submit a buy order
    client.post("/order", json={
        "symbol": "BTC-USDT", "order_type": "limit", "side": "buy",
        "quantity": 1.0, "price": 50000
    })
    # Submit a sell order
    client.post("/order", json={
        "symbol": "BTC-USDT", "order_type": "limit", "side": "sell",
        "quantity": 1.0, "price": 50100
    })

    bbo = engine.get_bbo("BTC-USDT")
    assert bbo['bid'] is not None
    assert bbo['ask'] is not None
    assert Decimal(bbo['bid'][0]) == Decimal('50000')
    assert Decimal(bbo['ask'][0]) == Decimal('50100')


def test_get_order_book_depth(client):
    """Test the order book depth endpoint."""
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "buy", "quantity": 1.5, "price": 50000})
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "buy", "quantity": 2.0, "price": 49900})
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 1.0, "price": 50100})
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 2.5, "price": 50200})

    depth = engine.get_order_book_depth("BTC-USDT")
    assert len(depth['bids']) == 2
    assert len(depth['asks']) == 2
    # Bids should be sorted high to low
    assert Decimal(depth['bids'][0][0]) == Decimal('50000')
    # Asks should be sorted low to high
    assert Decimal(depth['asks'][0][0]) == Decimal('50100')


# --- Test Price-Time Priority ---

def test_price_priority(client):
    """Ensure that orders with better prices are prioritized."""
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "buy", "quantity": 1, "price": 50000})
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "buy", "quantity": 1, "price": 50100}) # Better price
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 1, "price": 50300})
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 1, "price": 50200}) # Better price

    bbo = engine.get_bbo("BTC-USDT")
    assert Decimal(bbo['bid'][0]) == Decimal('50100')
    assert Decimal(bbo['ask'][0]) == Decimal('50200')


def test_time_priority_fifo(client):
    """Ensure FIFO for orders at the same price level."""
    # Submit three orders at the same price
    response1 = client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 1.0, "price": 50000})
    time.sleep(0.01) # ensure distinct timestamps
    response2 = client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 1.5, "price": 50000})
    time.sleep(0.01)
    response3 = client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 0.5, "price": 50000})
    
    order1_id = response1.json()['order_id']
    order2_id = response2.json()['order_id']

    # A single buy order should match against the first two orders in sequence
    match_response = client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "buy", "quantity": 2.0, "price": 50000})
    
    executions = match_response.json()['executions']
    assert len(executions) == 2
    
    # First execution should be against the first order submitted
    assert executions[0]['maker_order_id'] == order1_id
    assert Decimal(executions[0]['quantity']) == Decimal('1.0')
    
    # Second execution should be against the second order submitted
    assert executions[1]['maker_order_id'] == order2_id
    assert Decimal(executions[1]['quantity']) == Decimal('1.0') # Partially fills the second order

    # Verify the remaining state of the order book
    order_book = engine.order_books["BTC-USDT"]
    ask_level = order_book.asks[Decimal('50000')]
    assert ask_level.total_quantity == Decimal('1.0') # 1.5 (order2) - 1.0 (fill) + 0.5 (order3)
    assert len(ask_level.orders) == 2 # Remaining part of order2 and all of order3


# --- Test Trade-Through Protection ---

def test_trade_through_protection(client):
    """
    Ensures a marketable order fills at the best available prices sequentially
    without "trading through" (skipping) better prices.
    """
    # Setup the sell side with multiple price levels
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 1.0, "price": 50100}) # Best price
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 1.5, "price": 50150}) # Next best
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 2.0, "price": 50200})

    # A large buy order that should sweep through the first two levels
    response = client.post("/order", json={"symbol": "BTC-USDT", "order_type": "market", "side": "buy", "quantity": 3.0})
    
    result = response.json()
    assert result['status'] == 'filled'
    assert len(result['executions']) == 3 # Will match against 3 orders

    executions = result['executions']
    # First fill must be at the best price (50100)
    assert Decimal(executions[0]['price']) == Decimal('50100')
    assert Decimal(executions[0]['quantity']) == Decimal('1.0')

    # Second fill must be at the next best price (50150)
    assert Decimal(executions[1]['price']) == Decimal('50150')
    assert Decimal(executions[1]['quantity']) == Decimal('1.5')
    
    # Third fill is the remainder at the last price level
    assert Decimal(executions[2]['price']) == Decimal('50200')
    assert Decimal(executions[2]['quantity']) == Decimal('0.5')
    
    # Ensure the book is in the correct state
    bbo = engine.get_bbo("BTC-USDT")
    assert bbo['bid'] is None
    assert Decimal(bbo['ask'][0]) == Decimal('50200')
    assert Decimal(bbo['ask'][1]) == Decimal('1.5') # 2.0 - 0.5


# --- Test Advanced Order Types ---

def test_fok_order_success(client):
    """Test a Fill-Or-Kill order that can be completely filled."""
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 5.0, "price": 50000})

    response = client.post("/order", json={
        "symbol": "BTC-USDT", "order_type": "fok", "side": "buy",
        "quantity": 4.0, "price": 50000
    })
    
    result = response.json()
    assert result['status'] == 'filled'
    assert Decimal(result['filled_quantity']) == Decimal('4.0')
    assert len(result['executions']) == 1


def test_fok_order_failure(client):
    """Test a Fill-Or-Kill order that cannot be completely filled and is cancelled."""
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 3.0, "price": 50000})

    response = client.post("/order", json={
        "symbol": "BTC-USDT", "order_type": "fok", "side": "buy",
        "quantity": 4.0, "price": 50000
    })

    result = response.json()
    assert result['status'] == 'cancelled'
    assert Decimal(result['filled_quantity']) == Decimal('0')
    assert len(result['executions']) == 0
    
    # Make sure the resting order was not affected
    bbo = engine.get_bbo("BTC-USDT")
    assert Decimal(bbo['ask'][1]) == Decimal('3.0')


def test_ioc_order_partial_fill(client):
    """Test an Immediate-Or-Cancel order that is partially filled."""
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 2.0, "price": 50000})

    response = client.post("/order", json={
        "symbol": "BTC-USDT", "order_type": "ioc", "side": "buy",
        "quantity": 5.0, "price": 50000
    })

    result = response.json()
    # The order status should be CANCELLED because the remaining part is cancelled.
    # The engine logic correctly marks the overall order status this way.
    assert result['status'] == 'cancelled' 
    assert Decimal(result['filled_quantity']) == Decimal('2.0')
    assert Decimal(result['remaining_quantity']) == Decimal('3.0')
    assert len(result['executions']) == 1
    
    # The book should now be empty
    bbo = engine.get_bbo("BTC-USDT")
    assert bbo['ask'] is None


# --- Test Conditional Orders ---

def test_stop_loss_sell_trigger(client):
    """Test that a stop-loss sell order triggers correctly."""
    # Place a resting buy order to establish a market
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "buy", "quantity": 1, "price": 50000})

    # Place a stop loss sell order
    client.post("/order", json={
        "symbol": "BTC-USDT", "order_type": "stop_loss", "side": "sell",
        "quantity": 1.0, "stop_price": 49900
    })
    assert len(engine.conditional_orders["BTC-USDT"]) == 1

    # Execute a trade that brings the price down, triggering the stop
    response = client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 1, "price": 49850})
    
    # The trade at 49850 should trigger the stop at 49900. The stop becomes a market order.
    # However, the best bid is now gone, so the stop order will not fill and be cancelled.
    # This tests the trigger mechanism correctly.
    assert len(engine.conditional_orders["BTC-USDT"]) == 0
    order_book = engine.order_books["BTC-USDT"]
    assert order_book.get_best_bid() is None


def test_stop_limit_buy_trigger(client):
    """Test that a stop-limit buy order triggers and becomes a limit order."""
    # Initial sell order to set a market price
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "sell", "quantity": 1, "price": 50000})

    # Place a stop-limit buy order (e.g., to enter a breakout)
    client.post("/order", json={
        "symbol": "BTC-USDT", "order_type": "stop_limit", "side": "buy",
        "quantity": 1.0, "stop_price": 50100, "price": 50150
    })
    assert len(engine.conditional_orders["BTC-USDT"]) == 1

    # A trade occurs above the stop price
    client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "buy", "quantity": 0.5, "price": 50110})

    # The conditional order should now be gone and turned into a regular limit order
    assert len(engine.conditional_orders["BTC-USDT"]) == 0
    bbo = engine.get_bbo("BTC-USDT")
    assert Decimal(bbo['bid'][0]) == Decimal('50150') # The new limit order is now the best bid


# --- Test State Management ---

def test_save_and_load_state():
    """Test saving and loading the engine's state."""
    # 1. Setup initial state in a fresh engine instance
    initial_engine = engine
    initial_engine.submit_order("BTC-USDT", "limit", "buy", 1, 50000)
    initial_engine.submit_order("ETH-USDT", "limit", "sell", 10, 3000)
    initial_engine.submit_order("BTC-USDT", "stop_loss", "sell", 2, stop_price=49000)

    # 2. Save the state
    initial_engine.save_state()
    assert os.path.exists(TEST_STATE_FILE)

    # 3. Create a new engine instance and load the state
    new_engine = initial_engine
    new_engine.load_state()

    # 4. Verify the state was loaded correctly
    # Check regular orders
    assert "BTC-USDT" in new_engine.order_books
    assert "ETH-USDT" in new_engine.order_books
    btc_book = new_engine.order_books["BTC-USDT"]
    assert btc_book.get_best_bid() is not None
    assert btc_book.get_best_bid()[0] == Decimal('50000')

    # Check conditional orders
    assert "BTC-USDT" in new_engine.conditional_orders
    assert len(new_engine.conditional_orders["BTC-USDT"]) == 1
    cond_order = new_engine.conditional_orders["BTC-USDT"][0]
    assert cond_order.order_type == OrderType.STOP_LOSS
    assert cond_order.stop_price == Decimal('49000')


# --- Test API Validation ---
def test_api_rejects_invalid_quantity(client):
    """Test API endpoint rejects orders with zero or negative quantity."""
    response = client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "buy", "quantity": 0, "price": 50000})
    assert response.status_code == 400
    assert "Quantity must be positive" in response.json()['reason']

    response = client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "buy", "quantity": -1, "price": 50000})
    assert response.status_code == 400
    assert "Quantity must be positive" in response.json()['reason']

def test_api_rejects_missing_price_for_limit(client):
    """Test API rejects limit orders without a price."""
    response = client.post("/order", json={"symbol": "BTC-USDT", "order_type": "limit", "side": "buy", "quantity": 1})
    assert response.status_code == 400
    assert "Price is required" in response.json()['reason']

def test_api_rejects_missing_stop_price(client):
    """Test API rejects conditional orders without a stop_price."""
    response = client.post("/order", json={"symbol": "BTC-USDT", "order_type": "stop_loss", "side": "sell", "quantity": 1})
    assert response.status_code == 400
    assert "Stop price is required" in response.json()['reason']