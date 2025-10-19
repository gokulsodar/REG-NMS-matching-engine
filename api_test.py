import requests
import websockets
import json
import asyncio
from asyncio import TimeoutError

# --- Configuration ---
# Ensure your matching engine server is running at this address
BASE_URL = "http://127.0.0.1:8000"
ORDER_URL = f"{BASE_URL}/order"
RESET_URL = f"{BASE_URL}/reset"
TRADES_WS_URL = "ws://127.0.0.1:8000/ws/trades"
MARKET_DATA_WS_URL_TEMPLATE = "ws://127.0.0.1:8000/ws/marketdata/{symbol}"
TEST_SYMBOL = "BTC-USDT"

# --- Health Check Functions ---

def test_http_endpoints():
    """Tests the basic functionality of the /reset and /order HTTP endpoints."""
    print("--- Testing HTTP Endpoints ---")
    
    # 1. Test the /reset endpoint
    try:
        print(f"[*] Testing POST {RESET_URL}...")
        response = requests.post(RESET_URL)
        if response.status_code == 200 and response.json().get('status') == 'success':
            print(f"[SUCCESS] Reset endpoint is working. Response: {response.json()}")
        else:
            print(f"[FAILURE] Reset endpoint failed. Status: {response.status_code}, Response: {response.text}")
            return # Stop if reset fails
    except requests.exceptions.RequestException as e:
        print(f"[FAILURE] Could not connect to the reset endpoint: {e}")
        return

    # 2. Test the /order endpoint with a valid order
    try:
        print(f"[*] Testing POST {ORDER_URL} with a valid order...")
        order_payload = {
            "symbol": TEST_SYMBOL,
            "order_type": "limit",
            "side": "buy",
            "quantity": 0.1,
            "price": 50000
        }
        response = requests.post(ORDER_URL, json=order_payload)
        if response.status_code == 200 and response.json().get('status') == 'new':
            print(f"[SUCCESS] Order endpoint accepted a valid order. Response: {response.json()}")
        else:
            print(f"[FAILURE] Order endpoint failed for a valid order. Status: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[FAILURE] Could not connect to the order endpoint: {e}")

async def test_websocket_endpoints():
    """Tests the connectivity and basic message flow of the WebSocket endpoints."""
    print("\n--- Testing WebSocket Endpoints ---")
    
    # 1. Test the Market Data WebSocket Feed
    market_data_url = MARKET_DATA_WS_URL_TEMPLATE.format(symbol=TEST_SYMBOL)
    print(f"[*] Testing Market Data feed at {market_data_url}...")
    try:
        async with websockets.connect(market_data_url) as ws:
            # The server should send an initial depth snapshot upon connection
            message = await asyncio.wait_for(ws.recv(), timeout=3.0)
            data = json.loads(message)
            if data.get('symbol') == TEST_SYMBOL:
                print(f"[SUCCESS] Market data feed connected and received initial message: {data}")
            else:
                print(f"[FAILURE] Market data feed received unexpected data: {data}")
    except TimeoutError:
        print("[FAILURE] Market data feed timed out waiting for the initial message.")
    except Exception as e:
        print(f"[FAILURE] An error occurred with the market data feed: {e}")

    # 2. Test the Trades WebSocket Feed
    print(f"[*] Testing Trades feed at {TRADES_WS_URL}...")
    try:
        async with websockets.connect(TRADES_WS_URL) as ws:
            print("[*]   Connected to trade feed. Placing orders to generate a trade...")
            # Place orders that will immediately match and create a trade
            requests.post(ORDER_URL, json={"symbol": "GENERIC-COIN", "order_type": "limit", "side": "buy", "quantity": 10, "price": 100})
            requests.post(ORDER_URL, json={"symbol": "GENERIC-COIN", "order_type": "market", "side": "sell", "quantity": 5})
            
            # Wait for the trade message from the WebSocket
            message = await asyncio.wait_for(ws.recv(), timeout=3.0)
            data = json.loads(message)
            if data.get('symbol') == "GENERIC-COIN" and data.get('quantity') == '5.0':
                 print(f"[SUCCESS] Trade feed received a trade execution message: {data}")
            else:
                print(f"[FAILURE] Trade feed received unexpected data: {data}")

    except TimeoutError:
        print("[FAILURE] Trade feed timed out waiting for a trade message.")
    except Exception as e:
        print(f"[FAILURE] An error occurred with the trade feed: {e}")


if __name__ == "__main__":
    # First, run the synchronous HTTP tests
    test_http_endpoints()
    
    # Then, run the asynchronous WebSocket tests
    asyncio.run(test_websocket_endpoints())