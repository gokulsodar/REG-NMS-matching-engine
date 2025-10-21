"""
Market Data WebSocket Client
Connects to the matching engine's market data feed for multiple symbols
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketDataClient:
    """Client for connecting to market data WebSocket feeds"""
    
    def __init__(self, base_url: str = "ws://127.0.0.1:8000"):
        self.base_url = base_url
        self.connections = {}
        self.running = False
    
    async def connect_symbol(self, symbol: str):
        """Connect to market data feed for a specific symbol"""
        url = f"{self.base_url}/ws/marketdata/{symbol}"
        
        try:
            async with websockets.connect(url) as websocket:
                logger.info(f"Connected to market data feed for {symbol}")
                self.connections[symbol] = websocket
                
                while self.running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        self.handle_market_data(symbol, data)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning(f"Connection closed for {symbol}")
                        break
                    except Exception as e:
                        logger.error(f"Error receiving data for {symbol}: {e}")
                        break
                        
        except Exception as e:
            logger.error(f"Failed to connect to {symbol}: {e}")
    
    def handle_market_data(self, symbol: str, data: dict):
        """Process incoming market data"""
        timestamp = data.get('timestamp', '')
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Market Data Update - {symbol}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"{'-'*60}")
        
        if bids:
            logger.info("Top Bids (Price | Quantity):")
            for price, qty in bids[:5]:  # Show top 5 bids
                logger.info(f"  ${price:>12} | {qty:>10}")
        else:
            logger.info("No bids available")
        
        logger.info(f"{'-'*60}")
        
        if asks:
            logger.info("Top Asks (Price | Quantity):")
            for price, qty in asks[:5]:  # Show top 5 asks
                logger.info(f"  ${price:>12} | {qty:>10}")
        else:
            logger.info("No asks available")
        
        # Calculate spread if both bid and ask exist
        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            logger.info(f"{'-'*60}")
            logger.info(f"Spread: ${spread:.2f} ({spread_pct:.4f}%)")
        
        logger.info(f"{'='*60}\n")
    
    async def start(self, symbols: list):
        """Start monitoring multiple symbols"""
        self.running = True
        logger.info(f"Starting market data client for symbols: {', '.join(symbols)}")
        
        # Create tasks for each symbol
        tasks = [self.connect_symbol(symbol) for symbol in symbols]
        
        # Run all connections concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop(self):
        """Stop the client"""
        self.running = False
        logger.info("Stopping market data client...")


async def main():
    """Main function to run the market data client"""
    # Define the symbols to monitor
    symbols = ["BTC-USDT", "ETH-USDT", "ADA-USDT"]
    
    # Create client instance
    client = MarketDataClient()
    
    try:
        # Start monitoring
        await client.start(symbols)
    except KeyboardInterrupt:
        logger.info("\nReceived interrupt signal")
        client.stop()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        client.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")