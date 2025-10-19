import asyncio
import websockets
import json
import time
from datetime import datetime
from typing import Dict, Any

class ActiveMarketDataClient:
    def __init__(self, base_url: str = "ws://127.0.0.1:8000"):
        self.base_url = base_url
        self.websocket = None
        self.is_connected = False
        
    async def connect(self, symbol: str):
        """Connect to the WebSocket endpoint"""
        uri = f"{self.base_url}/ws/marketdata/{symbol}"
        try:
            self.websocket = await websockets.connect(uri)
            self.is_connected = True
            print(f"✅ Connected to {symbol} market data at {datetime.now()}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def get_market_data(self) -> Dict[str, Any]:
        """Get current market data snapshot"""
        if not self.is_connected or not self.websocket:
            raise ConnectionError("Not connected to WebSocket")
        
        try:
            # Receive the latest data (your server sends data on connection and potentially on receives)
            await self.websocket.send("")  # Send empty message to trigger response if needed
            
            # Try to receive data with timeout
            data = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
            return json.loads(data)
        except asyncio.TimeoutError:
            raise TimeoutError("No data received within timeout period")
        except Exception as e:
            raise ConnectionError(f"Error receiving data: {e}")
    
    async def continuous_updates(self, symbol: str, update_interval: float = 1.0):
        """
        Continuously poll for market data updates every second
        
        Args:
            symbol: Trading symbol
            update_interval: Time between updates in seconds
        """
        if not await self.connect(symbol):
            return
        
        update_count = 0
        last_update_time = time.time()
        
        try:
            while self.is_connected:
                try:
                    # Get market data
                    market_data = await self.get_market_data()
                    update_count += 1
                    
                    # Display the data
                    self.display_market_data(market_data, update_count, symbol)
                    
                    # Wait for the next update
                    elapsed = time.time() - last_update_time
                    sleep_time = max(0, update_interval - elapsed)
                    await asyncio.sleep(sleep_time)
                    last_update_time = time.time()
                    
                except (TimeoutError, ConnectionError) as e:
                    print(f"⚠️  Error getting data: {e}")
                    print("🔄 Attempting to reconnect...")
                    if not await self.connect(symbol):
                        await asyncio.sleep(5)  # Wait before retry
                        
        except KeyboardInterrupt:
            print("\n🛑 Stopping market data updates...")
        finally:
            await self.disconnect()
    
    def display_market_data(self, data: Dict[str, Any], update_count: int, symbol: str):
        """Display market data in a formatted way"""
        print(f"\n{'='*60}")
        print(f"📊 Update #{update_count} | {symbol} | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        if 'bids' in data and 'asks' in data:
            bids = data['bids']
            asks = data['asks']
            
            print("💰 Bids (Buyers):")
            for i, bid in enumerate(bids[:5]):  # Top 5 bids
                price = float(bid[0]) if isinstance(bid[0], str) else bid[0]
                quantity = float(bid[1]) if isinstance(bid[1], str) else bid[1]
                print(f"   {i+1:2d}. {price:12.6f} | {quantity:12.6f}")
            
            print("\n💸 Asks (Sellers):")
            for i, ask in enumerate(asks[:5]):  # Top 5 asks
                price = float(ask[0]) if isinstance(ask[0], str) else ask[0]
                quantity = float(ask[1]) if isinstance(ask[1], str) else ask[1]
                print(f"   {i+1:2d}. {price:12.6f} | {quantity:12.6f}")
            
            # Calculate spread if we have both bids and asks
            if bids and asks:
                best_bid = float(bids[0][0]) if isinstance(bids[0][0], str) else bids[0][0]
                best_ask = float(asks[0][0]) if isinstance(asks[0][0], str) else asks[0][0]
                spread = best_ask - best_bid
                spread_percent = (spread / best_bid) * 100
                print(f"\n📐 Spread: {spread:.6f} ({spread_percent:.4f}%)")
        
        # Show other available data
        other_keys = [k for k in data.keys() if k not in ['bids', 'asks']]
        if other_keys:
            print(f"\n📋 Other data: {', '.join(other_keys)}")
            for key in other_keys:
                if key != 'bids' and key != 'asks':
                    print(f"   {key}: {data[key]}")
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            print("🔌 Disconnected from market data feed")

# Alternative simple polling script
async def simple_polling_client(symbol: str, interval: float = 1.0):
    """
    Simple version that creates a new connection for each poll
    This is less efficient but more robust for testing
    """
    uri = f"ws://127.0.0.1:8000/ws/marketdata/{symbol}"
    poll_count = 0
    
    try:
        while True:
            poll_count += 1
            try:
                async with websockets.connect(uri) as websocket:
                    # Get initial data
                    data = await websocket.recv()
                    market_data = json.loads(data)
                    
                    # Display
                    print(f"\n🔄 Poll #{poll_count} | {datetime.now().strftime('%H:%M:%S')}")
                    print(f"Symbol: {symbol}")
                    
                    if 'bids' in market_data and 'asks' in market_data:
                        bids = market_data['bids'][:3]  # Top 3
                        asks = market_data['asks'][:3]  # Top 3
                        
                        print("Bids:", [f"{float(b[0]):.2f}" for b in bids])
                        print("Asks:", [f"{float(a[0]):.2f}" for a in asks])
                    
            except Exception as e:
                print(f"❌ Poll #{poll_count} failed: {e}")
            
            await asyncio.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped after {poll_count} polls")

# Usage examples
async def main():
    symbol = "BTCUSDT"  # Change to your symbol
    
    print("Choose polling mode:")
    print("1. Continuous connection (faster)")
    print("2. New connection each poll (more robust)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Method 1: Continuous connection
        client = ActiveMarketDataClient()
        await client.continuous_updates(symbol, update_interval=1.0)
    else:
        # Method 2: New connection each time
        await simple_polling_client(symbol, interval=1.0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")