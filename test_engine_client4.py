import pytest
import requests
import websockets
import json
import asyncio
from async_timeout import timeout
import os
from decimal import Decimal

# --- Test Configuration ---
BASE_URL = "http://127.0.0.1:8000"
ORDER_URL = f"{BASE_URL}/order"
RESET_URL = f"{BASE_URL}/reset"
TRADES_WS_URL = "ws://127.0.0.1:8000/ws/trades"
MARKET_DATA_WS_URL_TEMPLATE = "ws://127.0.0.1:8000/ws/marketdata/{symbol}"
STATE_FILE = "order_book_state.json"

# --- Helper Functions ---

def post_order(payload: dict):
    """Helper to send a POST request to the /order endpoint."""
    return requests.post(ORDER_URL, json=payload)

async def listen_for_message(ws, expected_count=1):
    """
    Helper to listen for a specific number of messages on a websocket.
    It will wait for the expected number of messages to arrive before timing out.
    """
    messages = []
    try:
        async with timeout(2.0):
            for _ in range(expected_count):
                message = await ws.recv()
                messages.append(json.loads(message))
    except asyncio.TimeoutError:
        pytest.fail(f"WebSocket timed out waiting for {expected_count} message(s). Received {len(messages)}.")
    return messages[0] if expected_count == 1 else messages

def cleanup_state_file():
    """Ensure the state file is removed before and after test runs."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

# --- Fixtures ---

@pytest.fixture(scope="module", autouse=True)
def manage_state_file():
    """Fixture to clean up the state file before and after the test module runs."""
    cleanup_state_file()
    yield
    cleanup_state_file()

@pytest.fixture(scope="function", autouse=True)
def reset_engine_state():
    """
    Fixture that calls the /reset endpoint before each test function.
    This ensures that every test starts with a clean, empty order book.
    'autouse=True' means it's automatically applied to all tests.
    """
    response = requests.post(RESET_URL)
    assert response.status_code == 200
    assert response.json()['status'] == 'success'
    yield


# --- Test Classes ---

class TestAPIValidation:
    """Tests for the API input validation layer."""

    def test_invalid_json_payload(self):
        headers = {'Content-Type': 'application/json'}
        response = requests.post(ORDER_URL, data="{not a valid json}", headers=headers)
        assert response.status_code == 400
        assert response.json()['reason'] == 'Invalid JSON payload'

    def test_missing_required_fields(self):
        payload = {'symbol': 'BTC-USDT', 'side': 'buy', 'quantity': 1}
        response = post_order(payload)
        assert response.status_code == 400
        assert response.json()['reason'] == 'Missing required fields'

    def test_negative_quantity(self):
        payload = {'symbol': 'BTC-USDT', 'order_type': 'limit', 'side': 'buy', 'quantity': -1, 'price': 50000}
        response = post_order(payload)
        assert response.status_code == 400
        assert response.json()['reason'] == 'Quantity must be positive'

    def test_negative_price(self):
        payload = {'symbol': 'BTC-USDT', 'order_type': 'limit', 'side': 'buy', 'quantity': 1, 'price': -50000}
        response = post_order(payload)
        assert response.status_code == 400
        assert response.json()['reason'] == 'Price must be positive'
        
    def test_negative_stop_price(self):
        payload = {'symbol': 'BTC-USDT', 'order_type': 'stop_loss', 'side': 'sell', 'quantity': 1, 'stop_price': -45000}
        response = post_order(payload)
        assert response.status_code == 400
        assert response.json()['reason'] == 'Stop price must be positive'

    def test_limit_order_missing_price(self):
        payload = {'symbol': 'BTC-USDT', 'order_type': 'limit', 'side': 'buy', 'quantity': 1}
        response = post_order(payload)
        assert response.status_code == 400
        assert response.json()['reason'] == 'Price is required for this order type'

    def test_stop_order_missing_stop_price(self):
        payload = {'symbol': 'BTC-USDT', 'order_type': 'stop_loss', 'side': 'sell', 'quantity': 1, 'price': 45000}
        response = post_order(payload)
        assert response.status_code == 400
        assert response.json()['reason'] == 'Stop price is required for this order type'
        
    def test_invalid_numeric_value(self):
        payload = {'symbol': 'BTC-USDT', 'order_type': 'limit', 'side': 'buy', 'quantity': 'one', 'price': '50k'}
        response = post_order(payload)
        assert response.status_code == 400
        assert 'Invalid numeric value' in response.json()['reason']


@pytest.mark.asyncio
class TestMatchingEngineLogic:
    """Tests for the core matching logic, order types, and priority."""

    async def test_limit_order_placement_and_price_time_priority(self):
        """
        Tests price-time priority by placing buy orders and then a single sell order
        to match against them, verifying the execution order via WebSocket trade feed.
        """
        symbol = "ETH-USDT"
        
        async with websockets.connect(TRADES_WS_URL) as trade_ws, \
             websockets.connect(MARKET_DATA_WS_URL_TEMPLATE.format(symbol=symbol)) as md_ws:

            # # 1. Consume the initial empty depth snapshot sent on connection.
            # initial_depth = await listen_for_message(md_ws)
            # assert initial_depth['bids'] == []
            
            # 2. Place first order (highest price, first in time).
            order1 = {'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 1.0, 'price': 3000}
            res1 = post_order(order1)
            assert res1.status_code == 200
            order1_id = res1.json()['order_id']

            # 3. Place second order (lower price).
            order2 = {'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 0.5, 'price': 2999}
            post_order(order2)

            # 4. Place third order (same price as first, second in time).
            order3 = {'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 0.7, 'price': 3000}
            res3 = post_order(order3)
            assert res3.status_code == 200
            order3_id = res3.json()['order_id']
            
            # 5. Listen for the 3 market data updates pushed by the server. The last one contains the final book state.
            depth_updates = await listen_for_message(md_ws, expected_count=3)
            final_depth_before_match = depth_updates[-1]
            
            # 6. Verify the book state: bids should be sorted by price (desc), and quantities aggregated.
            assert Decimal(final_depth_before_match['bids'][0][0]) == Decimal('3000')
            assert Decimal(final_depth_before_match['bids'][0][1]) == Decimal('1.0') + Decimal('0.7')
            assert Decimal(final_depth_before_match['bids'][1][0]) == Decimal('2999')

            # 7. Place a taker sell order to test priority.
            taker_order = {'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 1.5, 'price': 3000}
            taker_res = post_order(taker_order)
            assert taker_res.status_code == 200
            assert len(taker_res.json()['executions']) == 2

            # 8. Listen for two trade executions on the trade feed.
            trades = await listen_for_message(trade_ws, expected_count=2)
            trades.sort(key=lambda x: x['timestamp']) # Sort by timestamp to ensure chronological order.

            # 9. Verify trades: first trade matches order1 (time priority), second trade partially matches order3.
            assert Decimal(trades[0]['quantity']) == Decimal('1.0')
            assert trades[0]['maker_order_id'] == order1_id
            
            assert Decimal(trades[1]['quantity']) == Decimal('0.5')
            assert trades[1]['maker_order_id'] == order3_id
            
    async def test_market_order_execution(self):
        """Test a market order consuming multiple price levels."""
        symbol = "SOL-USDT"
        
        # Setup book with two price levels on the ask side.
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 10, 'price': 150})
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 15, 'price': 151})
        
        async with websockets.connect(TRADES_WS_URL) as trade_ws:
            # Market buy order that will cross both levels.
            market_order = {'symbol': symbol, 'order_type': 'market', 'side': 'buy', 'quantity': 20}
            res = post_order(market_order)
            assert res.json()['status'] == 'filled'
            assert Decimal(res.json()['filled_quantity']) == Decimal('20')
            
            # Expect two trades from the two fills.
            trades = await listen_for_message(trade_ws, expected_count=2)
            trades.sort(key=lambda x: x['price']) # Sort by price to make assertion deterministic.
            
            assert Decimal(trades[0]['price']) == Decimal('150')
            assert Decimal(trades[0]['quantity']) == Decimal('10')
            assert Decimal(trades[1]['price']) == Decimal('151')
            assert Decimal(trades[1]['quantity']) == Decimal('10')

    async def test_fok_order_success_and_failure(self):
        """Test Fill-Or-Kill orders for both success and failure cases."""
        symbol = "ADA-USDT"
        
        # Setup book
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 100, 'price': 0.45})
        
        # FOK that CAN be filled entirely.
        fok_success = {'symbol': symbol, 'order_type': 'fok', 'side': 'buy', 'quantity': 75, 'price': 0.45}
        res_success = post_order(fok_success)
        assert res_success.json()['status'] == 'filled'
        assert len(res_success.json()['executions']) == 1

        # FOK that CANNOT be filled (quantity too high). Remaining book quantity is 25.
        fok_fail_qty = {'symbol': symbol, 'order_type': 'fok', 'side': 'buy', 'quantity': 100, 'price': 0.45}
        res_fail = post_order(fok_fail_qty)
        assert res_fail.json()['status'] == 'cancelled'
        assert len(res_fail.json()['executions']) == 0
        
    async def test_ioc_order(self):
        """Test Immediate-Or-Cancel order that gets partially filled."""
        symbol = "XRP-USDT"
        
        # Setup book
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 50, 'price': 0.5})

        # IOC that can only be partially filled.
        ioc_order = {'symbol': symbol, 'order_type': 'ioc', 'side': 'buy', 'quantity': 100, 'price': 0.5}
        res = post_order(ioc_order)
        # Status is 'cancelled' because the remaining quantity > 0 is cancelled.
        assert res.json()['status'] == 'cancelled'
        assert Decimal(res.json()['filled_quantity']) == Decimal('50')
        assert Decimal(res.json()['remaining_quantity']) == Decimal('50')
        assert len(res.json()['executions']) == 1

@pytest.mark.asyncio
class TestConditionalOrders:
    """Tests for stop-loss, stop-limit, and take-profit orders."""

    async def test_stop_loss_trigger(self):
        """Test if a stop-loss market order triggers correctly when price drops."""
        symbol = "LINK-USDT"

        # Establish a market price.
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 10, 'price': 17.9})
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 10, 'price': 18.1})
        
        # Place a stop-loss sell order with a trigger below the current best bid.
        stop_order = {'symbol': symbol, 'order_type': 'stop_loss', 'side': 'sell', 'quantity': 5, 'stop_price': 17.95}
        res_stop = post_order(stop_order)
        assert res_stop.json()['status'] == 'new'
        assert 'Order is conditional' in res_stop.json()['reason']
        stop_order_id = res_stop.json()['order_id']
        
        async with websockets.connect(TRADES_WS_URL) as trade_ws:
            # Execute a market sell to bring the last traded price down, triggering the stop.
            trigger_order = {'symbol': symbol, 'order_type': 'market', 'side': 'sell', 'quantity': 1}
            post_order(trigger_order)

            # Expect 2 trades: 1) the trigger trade, 2) the triggered stop-loss market order.
            trades = await listen_for_message(trade_ws, expected_count=2)
            
            # Find the trade corresponding to our triggered stop order.
            triggered_trade = next((t for t in trades if t['taker_order_id'] == stop_order_id), None)
            assert triggered_trade is not None, "Triggered stop-loss trade not found in websocket feed"

            assert Decimal(triggered_trade['price']) == Decimal('17.9')
            assert Decimal(triggered_trade['quantity']) == Decimal('5')
            assert triggered_trade['aggressor_side'] == 'sell'

    async def test_take_profit_limit_sell_trigger(self):
        """Test if a take-profit-limit sell order triggers and is placed on the book."""
        symbol = "AVAX-USDT"
        
        # 1. Place resting orders to create a market.
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 10, 'price': 36.5})
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 15, 'price': 37.0}) # Initial ask
        maker_order_at_trigger_price = {'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 10, 'price': 37.5}
        post_order(maker_order_at_trigger_price) # A bid at the trigger price

        # 2. Place a take-profit limit sell order.
        tp_order = {
            'symbol': symbol, 'order_type': 'take_profit_limit', 'side': 'sell',
            'quantity': 5, 'price': 36.5, 'stop_price': 37.0 # Trigger at 37.0, place limit at 36.5
        }
        res_tp = post_order(tp_order)
        assert res_tp.json()['status'] == 'new'

        async with websockets.connect(MARKET_DATA_WS_URL_TEMPLATE.format(symbol=symbol)) as md_ws:
            # 3. Consume the initial state after all setup orders are placed.
            await listen_for_message(md_ws)

            # 4. Execute a market sell order to drop the price and trigger the take-profit.
            trigger_order = {'symbol': symbol, 'order_type': 'market', 'side': 'sell', 'quantity': 1}
            post_order(trigger_order)
            
            # 5. The trigger causes a trade and activates the conditional order, which is placed on the book.
            # This results in two market data updates. We listen for the final state.
            depth_updates = await listen_for_message(md_ws, expected_count=1)
            
            asks = depth_updates['asks']
            # 6. Verify that our new limit order is now resting on the book at its specified price.
            ask_level = next((level for level in asks if Decimal(level[0]) == Decimal('37.0')), None)
            assert ask_level is not None, "Price level 37.0 not found in asks."
            assert Decimal(ask_level[1]) == Decimal('5')

    async def test_take_profit_market_trigger(self):
        """Test if a take-profit-market sell order triggers and executes correctly."""
        symbol = "ATOM-USDT"

        # 1. Establish a market.
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 10, 'price': 10.0})
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 10, 'price': 10.5})

        # 2. Place the take-profit market sell order.
        tp_market_order = {
            'symbol': symbol, 'order_type': 'take_profit_market', 'side': 'sell',
            'quantity': 5, 'stop_price': 10.5 # Trigger when price hits 10.5
        }
        res_tp = post_order(tp_market_order)
        assert res_tp.json()['status'] == 'new'
        tp_order_id = res_tp.json()['order_id']

        async with websockets.connect(TRADES_WS_URL) as trade_ws:
            # 3. Execute a trade to push the last traded price up to the trigger price.
            trigger_order = {'symbol': symbol, 'order_type': 'market', 'side': 'buy', 'quantity': 1}
            post_order(trigger_order)

            # 4. Expect two trades: the trigger trade and the executed take-profit market order.
            trades = await listen_for_message(trade_ws, expected_count=2)

            triggered_trade = next(
                (t for t in trades if t['taker_order_id'] == tp_order_id),
                None
            )
            assert triggered_trade is not None, "Take-profit order did not execute."

            # 5. Verify the details of the triggered trade. It should execute against the best bid.
            assert triggered_trade['aggressor_side'] == 'sell'
            assert Decimal(triggered_trade['quantity']) == Decimal('5')
            assert Decimal(triggered_trade['price']) == Decimal('10.0')

@pytest.mark.asyncio
class TestWebSocketFeeds:
    """Tests dedicated to WebSocket functionality."""

    async def test_trade_feed_subscription(self):
        """Verify that a simple trade is broadcast correctly."""
        symbol = "DOGE-USDT"
        async with websockets.connect(TRADES_WS_URL) as ws:
            post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 1000, 'price': 0.15})
            post_order({'symbol': symbol, 'order_type': 'market', 'side': 'sell', 'quantity': 500})
            
            trade_msg = await listen_for_message(ws)
            
            assert trade_msg['symbol'] == symbol
            assert Decimal(trade_msg['price']) == Decimal('0.15')
            assert Decimal(trade_msg['quantity']) == Decimal('500')

    # async def test_market_data_feed_and_updates(self):
    #     """Verify initial depth and subsequent pushed updates are correct."""
    #     symbol = "MATIC-USDT"
    #     async with websockets.connect(MARKET_DATA_WS_URL_TEMPLATE.format(symbol=symbol)) as ws:
            
    #         # 1. Add an order and listen for the pushed update.
    #         post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 100, 'price': 0.75})
    #         depth_update1 = await listen_for_message(ws, expected_count=1)
    #         print("-------------------------\n\n\n\n\n\n\n\n\n------------------")
    #         print(depth_update1)
    #         print("-------------------------\n\n\n\n\n\n\n\n\n------------------")
    #         assert Decimal(depth_update1['bids'][0][0]) == Decimal('0.75')
    #         assert Decimal(depth_update1['bids'][0][1]) == Decimal('100')

    #         # 2. Add another order at the same level and check for quantity aggregation in the new update.
    #         post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 50, 'price': 0.75})
    #         depth_update2 = await listen_for_message(ws)
    #         assert Decimal(depth_update2['bids'][0][0]) == Decimal('0.75')
    #         assert Decimal(depth_update2['bids'][0][1]) == Decimal('150')

    #         # 3. Match the entire level and check that the book is empty again.
    #         post_order({'symbol': symbol, 'order_type': 'market', 'side': 'sell', 'quantity': 150})
    #         depth_update3 = await listen_for_message(ws)
    #         assert depth_update3['bids'] == []