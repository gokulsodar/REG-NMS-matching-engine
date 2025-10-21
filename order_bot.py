"""
Intelligent Order Generator
Creates continuous orders that maximize trade execution by placing orders
within the spread and near best bid/ask prices
"""

import asyncio
import aiohttp
import random
import logging
from datetime import datetime
from typing import Dict, Optional, List
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderGenerator:
    """Generates intelligent orders that maximize trade execution"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.order_endpoint = f"{base_url}/order"
        self.market_state: Dict[str, Dict] = {}
        self.running = False
        self.order_count = 0
        self.trade_count = 0
        
    async def submit_order(self, symbol: str, order_type: str, side: str, 
                          quantity: float, price: Optional[float] = None,
                          stop_price: Optional[float] = None) -> Dict:
        """Submit an order to the matching engine"""
        payload = {
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "quantity": quantity
        }
        
        if price is not None:
            payload["price"] = price
        if stop_price is not None:
            payload["stop_price"] = stop_price
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.order_endpoint, json=payload) as response:
                    result = await response.json()
                    self.order_count += 1
                    
                    # Count successful trades
                    if result.get('executions'):
                        self.trade_count += len(result['executions'])
                        logger.info(f"✓ Order filled: {symbol} {side} {quantity} @ {price} - {len(result['executions'])} trades")
                    elif result.get('status') == 'new' or result.get('status') == 'partially_filled':
                        logger.info(f"⊙ Order resting: {symbol} {side} {quantity} @ {price}")
                    
                    return result
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return {"status": "error", "reason": str(e)}
    
    def generate_aggressive_price(self, symbol: str, side: str, 
                                  best_bid: float, best_ask: float) -> float:
        """
        Generate prices that are likely to execute immediately
        Places orders within or crossing the spread
        """
        spread = best_ask - best_bid
        
        if side == "buy":
            # For buys: place between best_bid and best_ask (likely to execute)
            # Occasionally cross the spread to guarantee execution
            if random.random() < 0.6:  # 60% cross the spread
                price = best_ask + random.uniform(0, spread * 0.5)
            else:  # 40% place within spread
                price = random.uniform(best_bid + spread * 0.2, best_ask)
        else:  # sell
            # For sells: place between best_bid and best_ask
            if random.random() < 0.6:  # 60% cross the spread
                price = best_bid - random.uniform(0, spread * 0.5)
            else:  # 40% place within spread
                price = random.uniform(best_bid, best_ask - spread * 0.2)
        
        return round(price, 2)
    
    def generate_passive_price(self, symbol: str, side: str,
                               best_bid: float, best_ask: float) -> float:
        """
        Generate prices for limit orders that add liquidity
        These orders rest on the book but can still execute if market moves
        """
        spread = best_ask - best_bid
        
        if side == "buy":
            # Place buy orders slightly below best bid or at best bid
            if random.random() < 0.4:  # 40% at best bid (might execute)
                price = best_bid
            else:  # 60% below best bid
                price = best_bid - random.uniform(0, spread * 2)
        else:  # sell
            # Place sell orders slightly above best ask or at best ask
            if random.random() < 0.4:  # 40% at best ask (might execute)
                price = best_ask
            else:  # 60% above best ask
                price = best_ask + random.uniform(0, spread * 2)
        
        return round(price, 2)
    
    def generate_quantity(self, symbol: str) -> float:
        """Generate realistic order quantities"""
        if "BTC" in symbol:
            return round(random.uniform(0.01, 2.5), 2)
        elif "ETH" in symbol:
            return round(random.uniform(0.1, 15.0), 1)
        elif "ADA" in symbol:
            return round(random.uniform(100, 10000), 0)
        else:
            return round(random.uniform(1, 100), 2)
    
    async def get_market_state(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current market state to make intelligent order decisions
        In production, this would come from WebSocket market data feed
        """
        # For this implementation, we'll estimate based on known initial prices
        initial_prices = {
            "BTC-USDT": {"bid": 51450, "ask": 52100},
            "ETH-USDT": {"bid": 3445, "ask": 3510},
            "ADA-USDT": {"bid": 0.645, "ask": 0.655}
        }
        
        if symbol in initial_prices:
            # Add some randomness to simulate market movement
            base = initial_prices[symbol]
            volatility = 0.02  # 2% volatility
            
            bid = base["bid"] * (1 + random.uniform(-volatility, volatility))
            ask = base["ask"] * (1 + random.uniform(-volatility, volatility))
            
            # Ensure ask > bid
            if ask <= bid:
                ask = bid * 1.001
            
            return {"bid": round(bid, 2), "ask": round(ask, 2)}
        
        return None
    
    async def create_market_making_orders(self, symbol: str):
        """
        Create a pair of orders (buy and sell) around the current price
        This strategy ensures high execution probability
        """
        market_state = await self.get_market_state(symbol)
        if not market_state:
            return
        
        best_bid = market_state["bid"]
        best_ask = market_state["ask"]
        mid_price = (best_bid + best_ask) / 2
        
        # Create buy and sell orders
        buy_qty = self.generate_quantity(symbol)
        sell_qty = self.generate_quantity(symbol)
        
        # Mix of aggressive and passive orders
        strategy = random.choice(["aggressive", "passive", "mixed"])
        
        if strategy == "aggressive":
            # Both orders likely to execute
            buy_price = self.generate_aggressive_price(symbol, "buy", best_bid, best_ask)
            sell_price = self.generate_aggressive_price(symbol, "sell", best_bid, best_ask)
            
            await self.submit_order(symbol, "limit", "buy", buy_qty, buy_price)
            await asyncio.sleep(random.uniform(0.1, 0.5))
            await self.submit_order(symbol, "limit", "sell", sell_qty, sell_price)
            
        elif strategy == "passive":
            # Orders that add liquidity but might still execute
            buy_price = self.generate_passive_price(symbol, "buy", best_bid, best_ask)
            sell_price = self.generate_passive_price(symbol, "sell", best_bid, best_ask)
            
            await self.submit_order(symbol, "limit", "buy", buy_qty, buy_price)
            await asyncio.sleep(random.uniform(0.1, 0.5))
            await self.submit_order(symbol, "limit", "sell", sell_qty, sell_price)
            
        else:  # mixed
            # One aggressive, one passive
            if random.random() < 0.5:
                buy_price = self.generate_aggressive_price(symbol, "buy", best_bid, best_ask)
                sell_price = self.generate_passive_price(symbol, "sell", best_bid, best_ask)
            else:
                buy_price = self.generate_passive_price(symbol, "buy", best_bid, best_ask)
                sell_price = self.generate_aggressive_price(symbol, "sell", best_bid, best_ask)
            
            await self.submit_order(symbol, "limit", "buy", buy_qty, buy_price)
            await asyncio.sleep(random.uniform(0.1, 0.5))
            await self.submit_order(symbol, "limit", "sell", sell_qty, sell_price)
    
    async def create_market_order(self, symbol: str):
        """Create market orders that will execute immediately"""
        side = random.choice(["buy", "sell"])
        quantity = self.generate_quantity(symbol)
        
        await self.submit_order(symbol, "market", side, quantity)
    
    async def create_ioc_order(self, symbol: str):
        """Create IOC orders that execute immediately or cancel"""
        market_state = await self.get_market_state(symbol)
        if not market_state:
            return
        
        best_bid = market_state["bid"]
        best_ask = market_state["ask"]
        
        side = random.choice(["buy", "sell"])
        quantity = self.generate_quantity(symbol)
        
        # Price likely to match
        if side == "buy":
            price = self.generate_aggressive_price(symbol, "buy", best_bid, best_ask)
        else:
            price = self.generate_aggressive_price(symbol, "sell", best_bid, best_ask)
        
        await self.submit_order(symbol, "ioc", side, quantity, price)
    
    async def generate_orders_for_symbol(self, symbol: str):
        """Generate continuous orders for a specific symbol"""
        while self.running:
            try:
                # Choose order strategy with weighted probability
                strategy = random.choices(
                    ["market_making", "market", "ioc"],
                    weights=[0.6, 0.3, 0.1],  # 60% market making, 30% market, 10% IOC
                    k=1
                )[0]
                
                if strategy == "market_making":
                    await self.create_market_making_orders(symbol)
                elif strategy == "market":
                    await self.create_market_order(symbol)
                elif strategy == "ioc":
                    await self.create_ioc_order(symbol)
                
                # Wait between order batches
                await asyncio.sleep(random.uniform(0.5, 2.0))
                
                # Log statistics periodically
                if self.order_count % 50 == 0:
                    execution_rate = (self.trade_count / self.order_count * 100) if self.order_count > 0 else 0
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Statistics: {self.order_count} orders, {self.trade_count} trades")
                    logger.info(f"Execution rate: {execution_rate:.1f}%")
                    logger.info(f"{'='*60}\n")
                
            except Exception as e:
                logger.error(f"Error generating orders for {symbol}: {e}")
                await asyncio.sleep(1)
    
    async def start(self, symbols: List[str]):
        """Start generating orders for multiple symbols"""
        self.running = True
        logger.info(f"Starting order generator for: {', '.join(symbols)}")
        logger.info("Strategy: Intelligent order placement to maximize trade execution\n")
        
        # Create tasks for each symbol
        tasks = [self.generate_orders_for_symbol(symbol) for symbol in symbols]
        
        # Run all generators concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop(self):
        """Stop the order generator"""
        self.running = False
        logger.info("\nStopping order generator...")
        logger.info(f"Final stats: {self.order_count} orders, {self.trade_count} trades")
        if self.order_count > 0:
            logger.info(f"Overall execution rate: {self.trade_count / self.order_count * 100:.1f}%")


async def main():
    """Main function to run the order generator"""
    symbols = ["BTC-USDT", "ETH-USDT", "ADA-USDT"]
    
    generator = OrderGenerator()
    
    try:
        await generator.start(symbols)
    except KeyboardInterrupt:
        logger.info("\nReceived interrupt signal")
        generator.stop()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        generator.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")