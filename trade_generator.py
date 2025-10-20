import requests
import time
import json

# The base URL of the matching engine API
BASE_URL = "http://127.0.0.1:8000"

def submit_order(order_payload: dict):
    """
    Submits an order to the matching engine and prints the response.
    """
    url = f"{BASE_URL}/order"
    headers = {'Content-Type': 'application/json'}
    try:
        print(f"\n--> Submitting Order: {order_payload}")
        response = requests.post(url, headers=headers, data=json.dumps(order_payload))
        response.raise_for_status()  # Raise an exception for bad status codes
        
        response_data = response.json()
        print(f"<-- Response: {response_data}")
        return response_data
    except requests.exceptions.RequestException as e:
        print(f"!!! Error submitting order: {e}")
        return None

def run_trading_scenario():
    """
    Executes a sequence of orders to demonstrate matching and trading.
    """
    print("--- Starting Trading Scenario ---")
    
    # Let's check the initial state from the pre-populated orders
    print("\nInitial state is based on pre-populated orders:")
    print("  - BUY 1.5 @ 50000")
    print("  - BUY 2.0 @ 49900")
    print("  - SELL 1.0 @ 50100")
    print("  - SELL 2.5 @ 50200")
    print("-" * 30)
    time.sleep(2)

    # 1. Place a limit buy order that doesn't immediately match
    # This will be added to the bid side of the order book.
    submit_order({
        "symbol": "BTC-USDT",
        "order_type": "limit",
        "side": "buy",
        "quantity": 0.5,
        "price": 50050
    })
    time.sleep(2) # Pause to observe the market data update

    # 2. Place a limit sell order that will partially match the new best bid
    # This will execute a trade.
    submit_order({
        "symbol": "BTC-USDT",
        "order_type": "limit",
        "side": "sell",
        "quantity": 1.0,
        "price": 50050
    })
    time.sleep(2)

    # 3. Place a market buy order that will cross the spread
    # This will consume the resting sell order at 50100.
    submit_order({
        "symbol": "BTC-USDT",
        "order_type": "market",
        "side": "buy",
        "quantity": 0.75
    })
    time.sleep(2)

    # 4. Place a large market sell order to clear several bid levels
    # This will generate multiple trades against the resting buy orders.
    submit_order({
        "symbol": "BTC-USDT",
        "order_type": "market",
        "side": "sell",
        "quantity": 3.0
    })
    
    print("\n--- Trading Scenario Finished ---")

if __name__ == "__main__":
    # Before starting, you can optionally reset the engine state
    # to ensure a clean run every time.
    try:
        print("Resetting the matching engine...")
        requests.post(f"{BASE_URL}/reset")
        time.sleep(2) # Give the server a moment to reset and re-populate
    except requests.exceptions.RequestException as e:
        print(f"Could not reset engine. Is the server running? Error: {e}")

    run_trading_scenario()