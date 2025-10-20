import asyncio
import websockets
import json

async def subscribe_to_market_data(symbol: str):
    """
    Connects to the market data WebSocket feed for a given symbol and prints updates.
    """
    uri = f"ws://127.0.0.1:8000/ws/marketdata/{symbol}"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Successfully connected to {symbol} market data feed.")
            while True:
                try:
                    # Wait for a message from the server
                    message = await websocket.recv()
                    data = json.loads(message)

                    # Pretty-print the received market data
                    print("\n" + "="*50)
                    print(f"Received Market Data Update for {symbol}")
                    print("="*50)

                    print(json.dumps(data))
                    
                    print("="*50 + "\n")

                except websockets.ConnectionClosed:
                    print(f"Connection to {symbol} feed closed.")
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break
    except Exception as e:
        print(f"Failed to connect to WebSocket server: {e}")
        print("Please ensure the matching engine server is running.")

if __name__ == "__main__":
    # Run the subscriber for the BTC-USDT symbol
    asyncio.run(subscribe_to_market_data("BTC-USDT"))