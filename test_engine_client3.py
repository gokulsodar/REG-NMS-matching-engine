import pytest
import requests
import websockets
import json
import asyncio
from asyncio import timeout
import time
import os
from decimal import Decimal

# --- Test Configuration ---
BASE_URL = "http://127.0.0.1:8000"
ORDER_URL = f"{BASE_URL}/order"
RESET_URL = f"{BASE_URL}/reset"  # <-- ADDED
TRADES_WS_URL = "ws://127.0.0.1:8000/ws/trades"
MARKET_DATA_WS_URL_TEMPLATE = "ws://127.0.0.1:8000/ws/marketdata/{symbol}"
STATE_FILE = "order_book_state.json"

# --- Helper Functions ---

def post_order(payload: dict):
    """Helper to send a POST request to the /order endpoint."""
    return requests.post(ORDER_URL, json=payload)

async def listen_for_message(ws, expected_count=1):
    """Helper to listen for a specific number of messages on a websocket."""
    messages = []
    try:
        async with timeout(2):  # 2-second timeout to prevent tests from hanging
            for _ in range(expected_count):
                message = await ws.recv()
                messages.append(json.loads(message))
    except TimeoutError:
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

# NEW FIXTURE to reset the engine state before each test
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

# @pytest.mark.asyncio
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
        1. Place a buy order.
        2. Place another buy order at a lower price.
        3. Place a third buy order at the same price as the first.
        4. Place a sell order that should match the first and third orders in sequence.
        """
        symbol = "ETH-USDT"
        
        async with websockets.connect(TRADES_WS_URL) as trade_ws, \
             websockets.connect(MARKET_DATA_WS_URL_TEMPLATE.format(symbol=symbol)) as md_ws:

            # Initial depth snapshot
            initial_depth = await listen_for_message(md_ws)
            assert initial_depth['bids'] == []
            
            # 1. First order (highest price, first in time)
            order1 = {'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 1.0, 'price': 3000}
            res1 = post_order(order1)
            assert res1.status_code == 200
            order1_id = res1.json()['order_id']

            # 2. Second order (lower price)
            order2 = {'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 0.5, 'price': 2999}
            post_order(order2)

            # 3. Third order (same price as first, second in time)
            order3 = {'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 0.7, 'price': 3000}
            res3 = post_order(order3)
            assert res3.status_code == 200
            order3_id = res3.json()['order_id']
            
            # Check market data update
            depth_update = await listen_for_message(md_ws, expected_count=3)
            best_bid = depth_update[-1]['bids'][0]

            assert Decimal(best_bid[0]) == Decimal('3000')
            assert Decimal(best_bid[1]) == Decimal('1.0') + Decimal('0.7')

            # 4. Taker sell order to test priority
            taker_order = {'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 1.5, 'price': 3000}
            taker_res = post_order(taker_order)
            assert taker_res.status_code == 200
            assert taker_res.json()['status'] == 'filled'

            # Listen for two trade executions
            trades = await listen_for_message(trade_ws, expected_count=2)
            
            # First trade should match order1 (time priority)
            assert Decimal(trades[0]['quantity']) == Decimal('1.0')
            assert trades[0]['maker_order_id'] == order1_id
            
            # Second trade should match order3
            assert Decimal(trades[1]['quantity']) == Decimal('0.5')
            assert trades[1]['maker_order_id'] == order3_id
            
    async def test_market_order_execution(self):
        """Test a market order consuming multiple price levels."""
        symbol = "SOL-USDT"
        
        # Setup book
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 10, 'price': 150})
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 15, 'price': 151})
        
        async with websockets.connect(TRADES_WS_URL) as trade_ws:
            # Market buy order that will cross two levels
            market_order = {'symbol': symbol, 'order_type': 'market', 'side': 'buy', 'quantity': 20}
            res = post_order(market_order)
            assert res.json()['status'] == 'filled'
            assert Decimal(res.json()['filled_quantity']) == Decimal('20')
            
            # Expect two trades
            trades = await listen_for_message(trade_ws, expected_count=2)
            assert Decimal(trades[0]['price']) == Decimal('150')
            assert Decimal(trades[0]['quantity']) == Decimal('10')
            assert Decimal(trades[1]['price']) == Decimal('151')
            assert Decimal(trades[1]['quantity']) == Decimal('10')

    async def test_fok_order_success_and_failure(self):
        """Test Fill-Or-Kill orders."""
        symbol = "ADA-USDT"
        
        # Setup book
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 100, 'price': 0.45})
        
        # FOK that CAN be filled
        fok_success = {'symbol': symbol, 'order_type': 'fok', 'side': 'buy', 'quantity': 75, 'price': 0.45}
        res_success = post_order(fok_success)
        assert res_success.json()['status'] == 'filled'
        assert len(res_success.json()['executions']) == 1

        # FOK that CANNOT be filled (quantity too high)
        fok_fail_qty = {'symbol': symbol, 'order_type': 'fok', 'side': 'buy', 'quantity': 100, 'price': 0.45}
        res_fail = post_order(fok_fail_qty)
        # Note: The remaining quantity on the book is 25.
        assert res_fail.json()['status'] == 'cancelled'
        assert len(res_fail.json()['executions']) == 0
        
    async def test_ioc_order(self):
        """Test Immediate-Or-Cancel orders."""
        symbol = "XRP-USDT"
        
        # Setup book
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 50, 'price': 0.5})

        # IOC that can be partially filled
        ioc_order = {'symbol': symbol, 'order_type': 'ioc', 'side': 'buy', 'quantity': 100, 'price': 0.5}
        res = post_order(ioc_order)
        assert res.json()['status'] == 'cancelled' # Cancelled because remaining > 0
        assert Decimal(res.json()['filled_quantity']) == Decimal('50')
        assert Decimal(res.json()['remaining_quantity']) == Decimal('50')
        assert len(res.json()['executions']) == 1

@pytest.mark.asyncio
class TestConditionalOrders:
    """Tests for stop-loss, stop-limit, and take-profit orders."""

    async def test_stop_loss_trigger(self):
        """Test if a stop-loss order triggers correctly."""
        symbol = "LINK-USDT"

        # Initial market price is around 18
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 10, 'price': 17.9})
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 10, 'price': 18.1})
        
        # Place a stop-loss sell order
        stop_order = {'symbol': symbol, 'order_type': 'stop_loss', 'side': 'sell', 'quantity': 5, 'stop_price': 17.95}
        res_stop = post_order(stop_order)
        assert res_stop.json()['status'] == 'new'
        assert 'Order is conditional' in res_stop.json()['reason']
        stop_order_id = res_stop.json()['order_id']
        
        async with websockets.connect(TRADES_WS_URL) as trade_ws:
            # Create a trade that brings the price down and triggers the stop
            trigger_order = {'symbol': symbol, 'order_type': 'market', 'side': 'sell', 'quantity': 1}
            post_order(trigger_order)

            # 1st trade: the trigger trade itself
            # 2nd trade: the triggered stop-loss market order
            trades = await listen_for_message(trade_ws, expected_count=2)
            # sort triggered trade by timestamp
    
            from datetime import datetime
            trades.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]))
            triggered_trade = trades[1]
            # print("-----------------------------------\n\n\n\n\n\n\n\n\n\n\n----------------------------")
            # print(trades)
            # print("-----------------------------------\n\n\n\n\n\n\n\n\n\n\n----------------------------")
            assert Decimal(triggered_trade['price']) == Decimal('17.9')
            assert Decimal(triggered_trade['quantity']) == Decimal('5')
            assert triggered_trade['aggressor_side'] == 'sell'
            assert triggered_trade['taker_order_id'] == stop_order_id

    async def test_take_profit_limit_sell_trigger_corrected(self):
        """
        Test if a take-profit-limit sell order triggers correctly when the last
        traded price rises to meet the trigger price.
        """
        symbol = "AVAX-USDT"
        
        # 1. Place resting orders to create a clear bid/ask spread.
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 10, 'price': 35.0})
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 10, 'price': 36.0})

        # 2. Add an order at the intended trigger price to ensure a trade can occur there.
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 10, 'price': 36.5})

        # 3. Place a take-profit limit sell order with a trigger price above the current market.
        # When the last trade price rises to (or above) the stop_price, a limit sell
        # order will be placed at the specified 'price'.
        tp_order = {
            'symbol': symbol,
            'order_type': 'take_profit_limit',
            'side': 'sell',
            'quantity': 5,
            'price': 36.5,      # The limit price for the triggered order.
            'stop_price': 36.5  # The price that triggers the order.
        }
        res_tp = post_order(tp_order)
        assert res_tp.json()['status'] == 'new'
        tp_order_id = res_tp.json()['order_id']

        async with websockets.connect(MARKET_DATA_WS_URL_TEMPLATE.format(symbol=symbol)) as md_ws:
            # Consume initial state messages from the websocket to have a clean slate.
            await listen_for_message(md_ws, expected_count=1)

            # 4. Execute a market buy order large enough to clear the 36.0 level and
            # trade at 36.5, which will activate our take-profit order.
            trigger_order = {'symbol': symbol, 'order_type': 'market', 'side': 'buy', 'quantity': 11}
            post_order(trigger_order)
            
            # 5. The take-profit order should now be triggered and on the book.
            
            depth_update = await listen_for_message(md_ws) 
            
            asks = depth_update['asks']
            
            # Find the ask price level for our triggered order.
            ask_level = next((level for level in asks if Decimal(level[0]) == Decimal('36.5')), None)

            # Assert that the price level exists.
            assert ask_level is not None, "Price level 36.5 not found in asks."

            # The total quantity at this level should be the remainder of the original order
            # (10 - 1 = 9) plus the quantity of our newly triggered take-profit order (5).
            expected_quantity = Decimal('14')
            actual_quantity = Decimal(ask_level[1])
            assert actual_quantity == expected_quantity
    
    async def test_take_profit_market_trigger(self):
        """
        Test if a take-profit-market sell order triggers correctly when the
        last traded price rises to the trigger price.
        """
        symbol = "ATOM-USDT"

        # 1. Establish a market with a clear price. The last traded price will
        #    be implicitly between the bid and ask.
        #    We need a bid for the triggered market sell to execute against.
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 10, 'price': 10.0})
        post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'sell', 'quantity': 10, 'price': 10.5})

        # 2. Place the take-profit market sell order. For a sell order, the
        #    trigger price (stop_price) must be *above* the current market price.
        tp_market_order = {
            'symbol': symbol,
            'order_type': 'take_profit_market',
            'side': 'sell',
            'quantity': 5,
            'stop_price': 10.5  # Trigger when the price reaches or exceeds this
        }
        res_tp = post_order(tp_market_order)
        assert res_tp.json()['status'] == 'new'
        assert 'Order is conditional' in res_tp.json()['reason']
        tp_order_id = res_tp.json()['order_id']

        async with websockets.connect(TRADES_WS_URL) as trade_ws:
            # 3. Execute a trade that pushes the last traded price up to the
            #    trigger price. A market buy that consumes the ask at 10.5 will do this.
            trigger_order = {'symbol': symbol, 'order_type': 'market', 'side': 'buy', 'quantity': 1}
            post_order(trigger_order)

            # 4. We expect two trades on the WebSocket feed:
            #    - The first is from the market buy order that we just sent.
            #    - The second is from our take-profit order being triggered and
            #      executing as a market order.
            trades = await listen_for_message(trade_ws, expected_count=2)

            # The trades can arrive out of order, so find our triggered trade
            # by matching the taker_order_id.
            triggered_trade = next(
                (t for t in trades if t['taker_order_id'] == tp_order_id),
                None
            )

            assert triggered_trade is not None, "Take-profit order did not execute."

            # 5. Verify the details of the triggered trade.
            assert triggered_trade['aggressor_side'] == 'sell'
            assert Decimal(triggered_trade['quantity']) == Decimal('5')
            # It executed against the best bid, which was at 10.0
            assert Decimal(triggered_trade['price']) == Decimal('10.0')
            
@pytest.mark.asyncio
class TestWebSocketFeeds:
    """Tests dedicated to WebSocket functionality."""

    async def test_trade_feed_subscription(self):
        """Verify that trades are broadcast correctly."""
        symbol = "DOGE-USDT"
        async with websockets.connect(TRADES_WS_URL) as ws:
            post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 1000, 'price': 0.15})
            post_order({'symbol': symbol, 'order_type': 'market', 'side': 'sell', 'quantity': 500})
            
            trade_msg = await listen_for_message(ws)
            
            assert trade_msg['symbol'] == symbol
            assert Decimal(trade_msg['price']) == Decimal('0.15')
            assert Decimal(trade_msg['quantity']) == Decimal('500')

    async def test_market_data_feed_and_updates(self):
        """Verify initial depth and subsequent updates."""
        symbol = "MATIC-USDT"
        async with websockets.connect(MARKET_DATA_WS_URL_TEMPLATE.format(symbol=symbol)) as ws:
            # 1. Server should send initial (empty) depth on connect
            initial_depth = await listen_for_message(ws)
            assert initial_depth['symbol'] == symbol
            assert initial_depth['bids'] == []
            assert initial_depth['asks'] == []

            # 2. Add an order and expect an update
            post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 100, 'price': 0.75})
            depth_update1 = await listen_for_message(ws)
            assert Decimal(depth_update1['bids'][0][0]) == Decimal('0.75')
            assert Decimal(depth_update1['bids'][0][1]) == Decimal('100')

            # 3. Add another order at the same level and expect aggregation
            post_order({'symbol': symbol, 'order_type': 'limit', 'side': 'buy', 'quantity': 50, 'price': 0.75})
            depth_update2 = await listen_for_message(ws)
            assert Decimal(depth_update2['bids'][0][0]) == Decimal('0.75')
            assert Decimal(depth_update2['bids'][0][1]) == Decimal('150')

            # 4. Match the order and expect the level to be removed
            post_order({'symbol': symbol, 'order_type': 'market', 'side': 'sell', 'quantity': 150})
            depth_update3 = await listen_for_message(ws)
            assert depth_update3['bids'] == []