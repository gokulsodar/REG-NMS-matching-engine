import asyncio
import websockets
import json
from datetime import datetime

async def subscribe_to_trades():
    """
    Connects to the global trade feed WebSocket and prints live trade executions.
    """
    uri = "ws://127.0.0.1:8000/ws/trades"
    print(f"Connecting to trade feed at {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Successfully connected to the global trade feed.")
            while True:
                try:
                    # Wait for a trade execution message from the server
                    message = await websocket.recv()
                    trade_data = json.loads(message)

                    # Pretty-print the received trade data for clarity
                    print("\n" + "!"*50)
                    print("!!! TRADE EXECUTED !!!")
                    print("!"*50)
                    print(f"  Trade ID:        {trade_data['trade_id']}")
                    print(f"  Symbol:          {trade_data['symbol']}")
                    
                    # Format timestamp for better readability
                    ts = datetime.fromisoformat(trade_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    print(f"  Timestamp:       {ts}")
                    
                    print(f"  Aggressor Side:  {trade_data['aggressor_side'].upper()}")
                    print(f"  Price:           {trade_data['price']}")
                    print(f"  Quantity:        {trade_data['quantity']}")
                    print("-" * 50)
                    print(f"  Taker Order ID:  {trade_data['taker_order_id']}")
                    print(f"  Maker Order ID:  {trade_data['maker_order_id']}")
                    print(f"  Taker Fee:       {trade_data['taker_fee']}")
                    print(f"  Maker Fee:       {trade_data['maker_fee']}")
                    print("!"*50 + "\n")


                except websockets.ConnectionClosed:
                    print("Connection to trade feed closed.")
                    break
                except Exception as e:
                    print(f"An error occurred while processing a trade message: {e}")
                    break
    except Exception as e:
        print(f"Failed to connect to the trade feed WebSocket server: {e}")
        print("Please ensure the matching engine server is running.")

if __name__ == "__main__":
    # Run the trade subscriber indefinitely
    asyncio.run(subscribe_to_trades())