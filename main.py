"""
High-Performance Cryptocurrency Matching Engine
Implements REG NMS-inspired price-time priority with internal order protection
"""

import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass, field
from collections import deque, defaultdict
from decimal import Decimal
import logging
import json
from enum import Enum
import heapq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"  # Immediate-Or-Cancel
    FOK = "fok"  # Fill-Or-Kill


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"                    # DO we really need cancelled and rejected??
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal]
    timestamp: float
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: Decimal = Decimal('0')
    
    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity
    
    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'price': str(self.price) if self.price else None,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'filled_quantity': str(self.filled_quantity)
        }


@dataclass
class Trade:
    """Represents a trade execution"""
    trade_id: str
    symbol: str
    price: Decimal
    quantity: Decimal
    timestamp: float
    aggressor_side: OrderSide
    maker_order_id: str
    taker_order_id: str
    
    def to_dict(self) -> dict:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'price': str(self.price),
            'quantity': str(self.quantity),
            'timestamp': datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            'aggressor_side': self.aggressor_side.value,
            'maker_order_id': self.maker_order_id,
            'taker_order_id': self.taker_order_id
        }


class PriceLevel:
    """Represents orders at a specific price level with FIFO queue"""
    def __init__(self, price: Decimal):
        self.price = price
        self.orders: deque[Order] = deque()     ## TIME BASED SORTING IS NOT DONE ON ORDERS AT A PRICE LEVEL
        self.total_quantity = Decimal('0')
    
    def add_order(self, order: Order):
        """Add order to this price level"""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity
    
    def remove_order(self, order_id: str) -> bool:
        """Remove order from this price level"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                self.total_quantity -= order.remaining_quantity
                del self.orders[i]
                return True
        return False
    
    def is_empty(self) -> bool:
        return len(self.orders) == 0


class OrderBook:
    """Order book for a single trading pair with price-time priority"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Bids: max heap (highest price first)
        self.bids: Dict[Decimal, PriceLevel] = {}
        self.bid_prices: List[Decimal] = []  # Max heap for bids
        
        # Asks: min heap (lowest price first)
        self.asks: Dict[Decimal, PriceLevel] = {}
        self.ask_prices: List[Decimal] = []  # Min heap for asks
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
    
    def add_order(self, order: Order):
        """Add order to the book"""
        if order.side == OrderSide.BUY:
            if order.price not in self.bids:
                self.bids[order.price] = PriceLevel(order.price)
                heapq.heappush(self.bid_prices, -order.price)  # Negative for max heap
            self.bids[order.price].add_order(order)
        else:
            if order.price not in self.asks:
                self.asks[order.price] = PriceLevel(order.price)
                heapq.heappush(self.ask_prices, order.price)
            self.asks[order.price].add_order(order)
        
        self.orders[order.order_id] = order
        logger.info(f"Added {order.side.value} order {order.order_id} at {order.price}")
    
    def remove_order(self, order_id: str) -> bool:
        """Remove order from the book"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.side == OrderSide.BUY:
            if order.price in self.bids:
                self.bids[order.price].remove_order(order_id)
                if self.bids[order.price].is_empty():
                    del self.bids[order.price]
        else:
            if order.price in self.asks:
                self.asks[order.price].remove_order(order_id)
                if self.asks[order.price].is_empty():
                    del self.asks[order.price]
        
        del self.orders[order_id]
        logger.info(f"Removed order {order_id}")
        return True
    
    def get_best_bid(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best bid (highest buy price)"""
        while self.bid_prices and -self.bid_prices[0] not in self.bids:
            heapq.heappop(self.bid_prices)
        
        if self.bid_prices:
            price = -self.bid_prices[0]
            return (price, self.bids[price].total_quantity)
        return None
    
    def get_best_ask(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best ask (lowest sell price)"""
        while self.ask_prices and self.ask_prices[0] not in self.asks:
            heapq.heappop(self.ask_prices)
        
        if self.ask_prices:
            price = self.ask_prices[0]
            return (price, self.asks[price].total_quantity)
        return None
    
    def get_bbo(self) -> Dict:
        """Get Best Bid and Offer"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        return {
            'symbol': self.symbol,
            'bid': [str(best_bid[0]), str(best_bid[1])] if best_bid else None,
            'ask': [str(best_ask[0]), str(best_ask[1])] if best_ask else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_depth(self, levels: int = 10) -> Dict:
        """Get order book depth"""
        bids = []
        asks = []
        
        # Get top bid levels
        sorted_bids = sorted(self.bids.keys(), reverse=True)[:levels]
        for price in sorted_bids:
            level = self.bids[price]
            bids.append([str(price), str(level.total_quantity)])
        
        # Get top ask levels
        sorted_asks = sorted(self.asks.keys())[:levels]
        for price in sorted_asks:
            level = self.asks[price]
            asks.append([str(price), str(level.total_quantity)])
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': self.symbol,
            'bids': bids,
            'asks': asks
        }


class MatchingEngine:
    """Core matching engine with REG NMS-inspired principles"""
    def __init__(self):
        self.order_books: Dict[str, OrderBook] = {}
        self.trades: List[Trade] = []
        self.trade_callbacks: List = []
        self.market_data_callbacks: List = []
        
    def get_or_create_order_book(self, symbol: str) -> OrderBook:
        """Get or create order book for symbol"""
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol)
        return self.order_books[symbol]
    
    def submit_order(self, symbol: str, order_type: str, side: str, 
                     quantity: float, price: Optional[float] = None) -> Dict:
        """
        Submit a new order to the matching engine
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            order_type: "market", "limit", "ioc", "fok"
            side: "buy" or "sell"
            quantity: Order quantity
            price: Limit price (required for limit orders)
        
        Returns:
            Dict with order status and execution details
        """
        try:
            # Validate inputs
            order_type_enum = OrderType(order_type.lower())
            side_enum = OrderSide(side.lower())
            qty = Decimal(str(quantity))
            
            if qty <= 0:
                raise ValueError("Quantity must be positive")
            
            if order_type_enum in [OrderType.LIMIT, OrderType.IOC, OrderType.FOK]:
                if price is None:
                    raise ValueError(f"{order_type} orders require a price")
                price_decimal = Decimal(str(price))
                if price_decimal <= 0:
                    raise ValueError("Price must be positive")
            else:
                price_decimal = None
            
            # Create order
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                order_type=order_type_enum,
                side=side_enum,
                quantity=qty,
                price=price_decimal,
                timestamp=time.time()
            )
            
            logger.info(f"Received {order_type} {side} order: {order.order_id}")
            
            # Process order
            result = self._process_order(order)
            
            return result
            
        except ValueError as e:
            logger.error(f"Invalid order parameters: {e}")
            return {
                'status': 'rejected',
                'reason': str(e)
            }
        except Exception as e:
            logger.error(f"Error processing order: {e}", exc_info=True)
            return {
                'status': 'error',
                'reason': str(e)
            }
    
    def _process_order(self, order: Order) -> Dict:
        """Process order through matching engine"""
        order_book = self.get_or_create_order_book(order.symbol)
        executions = []
        
        if order.order_type == OrderType.MARKET:
            executions = self._match_market_order(order, order_book)
        elif order.order_type == OrderType.IOC:
            executions = self._match_ioc_order(order, order_book)
        elif order.order_type == OrderType.FOK:
            executions = self._match_fok_order(order, order_book)
        else:  # LIMIT
            executions = self._match_limit_order(order, order_book)
        
        # Broadcast market data update
        self._broadcast_market_data(order_book)
        
        return {
            'status': order.status.value,
            'order_id': order.order_id,
            'filled_quantity': str(order.filled_quantity),
            'remaining_quantity': str(order.remaining_quantity),
            'executions': [trade.to_dict() for trade in executions]
        }
    
    def _match_market_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """Match market order at best available prices"""
        trades = self._match_aggressive_order(order, order_book, allow_partial=True)
        
        if order.remaining_quantity > 0:
            order.status = OrderStatus.CANCELLED
            logger.warning(f"Market order {order.order_id} partially filled, "
                         f"cancelling remaining {order.remaining_quantity}")
        else:
            order.status = OrderStatus.FILLED
        
        return trades
    
    def _match_ioc_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """Match IOC order - fill immediately or cancel"""
        trades = self._match_aggressive_order(order, order_book, allow_partial=True)
        
        if order.remaining_quantity > 0:
            order.status = OrderStatus.CANCELLED
            logger.info(f"IOC order {order.order_id} partially filled, "
                       f"cancelling remaining {order.remaining_quantity}")
        else:
            order.status = OrderStatus.FILLED
        
        return trades
    
    def _match_fok_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """Match FOK order - fill completely or cancel entirely"""
        # Check if order can be fully filled
        if not self._can_fill_quantity(order, order_book, order.quantity):
            order.status = OrderStatus.CANCELLED
            logger.info(f"FOK order {order.order_id} cannot be fully filled, cancelling")
            return []
        
        # Execute full fill
        trades = self._match_aggressive_order(order, order_book, allow_partial=False)
        order.status = OrderStatus.FILLED
        
        return trades
    
    def _match_limit_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """Match limit order with price-time priority"""
        trades = self._match_aggressive_order(order, order_book, allow_partial=True)
        
        if order.remaining_quantity > 0:
            # Add to book if not fully filled
            order_book.add_order(order)
            order.status = OrderStatus.PARTIALLY_FILLED if order.filled_quantity > 0 else OrderStatus.NEW
            logger.info(f"Limit order {order.order_id} resting on book with "
                       f"{order.remaining_quantity} remaining")
        else:
            order.status = OrderStatus.FILLED
        
        return trades
    
    def _match_aggressive_order(self, order: Order, order_book: OrderBook, 
                                allow_partial: bool) -> List[Trade]:
        """Match an aggressive (marketable) order against the book"""
        trades = []
        
        while order.remaining_quantity > 0:
            # Get best counter price
            if order.side == OrderSide.BUY:
                best_level = order_book.get_best_ask()
                price_levels = order_book.asks
            else:
                best_level = order_book.get_best_bid()
                price_levels = order_book.bids
            
            if not best_level:
                break  # No liquidity
            
            best_price, _ = best_level
            
            # Check if order is marketable
            if order.price is not None:
                if order.side == OrderSide.BUY and best_price > order.price:
                    break  # Can't buy at higher price
                if order.side == OrderSide.SELL and best_price < order.price:
                    break  # Can't sell at lower price
            
            # Match at this price level (FIFO)
            level = price_levels[best_price]
            
            while level.orders and order.remaining_quantity > 0:
                maker_order = level.orders[0]
                
                # Calculate fill quantity
                fill_qty = min(order.remaining_quantity, maker_order.remaining_quantity)
                
                # Execute trade
                trade = self._execute_trade(order, maker_order, best_price, fill_qty)
                trades.append(trade)
                
                # Update order quantities
                order.filled_quantity += fill_qty
                maker_order.filled_quantity += fill_qty
                
                # Remove fully filled maker order
                if maker_order.remaining_quantity == 0:
                    maker_order.status = OrderStatus.FILLED
                    level.orders.popleft()
                    level.total_quantity -= fill_qty
                    del order_book.orders[maker_order.order_id]
                    logger.info(f"Maker order {maker_order.order_id} fully filled")
                else:
                    level.total_quantity -= fill_qty
            
            # Clean up empty price level
            if level.is_empty():
                del price_levels[best_price]
        
        return trades
    
    def _can_fill_quantity(self, order: Order, order_book: OrderBook, 
                          quantity: Decimal) -> bool:
        """Check if quantity can be filled at acceptable prices"""
        available = Decimal('0')
        
        if order.side == OrderSide.BUY:
            for price in sorted(order_book.asks.keys()):
                if order.price is not None and price > order.price:
                    break
                available += order_book.asks[price].total_quantity
                if available >= quantity:
                    return True
        else:
            for price in sorted(order_book.bids.keys(), reverse=True):
                if order.price is not None and price < order.price:
                    break
                available += order_book.bids[price].total_quantity
                if available >= quantity:
                    return True
        
        return False
    
    def _execute_trade(self, taker: Order, maker: Order, price: Decimal, 
                      quantity: Decimal) -> Trade:
        """Execute a trade between two orders"""
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=taker.symbol,
            price=price,
            quantity=quantity,
            timestamp=time.time(),
            aggressor_side=taker.side,
            maker_order_id=maker.order_id,
            taker_order_id=taker.order_id
        )
        
        self.trades.append(trade)
        logger.info(f"Trade executed: {trade.trade_id} - {quantity} @ {price}")
        
        # Broadcast trade
        self._broadcast_trade(trade)
        
        return trade
    
    def _broadcast_trade(self, trade: Trade):
        """Broadcast trade to subscribers"""
        for callback in self.trade_callbacks:
            try:
                callback(trade.to_dict())
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
    
    def _broadcast_market_data(self, order_book: OrderBook):
        """Broadcast market data update"""
        data = order_book.get_depth()
        for callback in self.market_data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in market data callback: {e}")
    
    def subscribe_trades(self, callback):
        """Subscribe to trade feed"""
        self.trade_callbacks.append(callback)
    
    def subscribe_market_data(self, callback):
        """Subscribe to market data feed"""
        self.market_data_callbacks.append(callback)
    
    def get_order_book_depth(self, symbol: str, levels: int = 10) -> Dict:
        """Get current order book depth"""
        if symbol not in self.order_books:
            return {'error': 'Symbol not found'}
        return self.order_books[symbol].get_depth(levels)
    
    def get_bbo(self, symbol: str) -> Dict:
        """Get Best Bid and Offer"""
        if symbol not in self.order_books:
            return {'error': 'Symbol not found'}
        return self.order_books[symbol].get_bbo()


# Example usage
if __name__ == "__main__":
    # Create matching engine
    engine = MatchingEngine()
    
    # Subscribe to feeds
    def print_trade(trade):
        print(f"TRADE: {json.dumps(trade, indent=2)}")
    
    def print_market_data(data):
        print(f"MARKET DATA: {json.dumps(data, indent=2)}")
    
    engine.subscribe_trades(print_trade)
    engine.subscribe_market_data(print_market_data)
    
    # Submit sample orders
    print("\n=== Submitting Limit Orders ===")
    result1 = engine.submit_order("BTC-USDT", "limit", "buy", 1.5, 50000)
    print(f"Order 1: {json.dumps(result1, indent=2)}")
    
    result2 = engine.submit_order("BTC-USDT", "limit", "buy", 2.0, 49900)
    print(f"Order 2: {json.dumps(result2, indent=2)}")
    
    result3 = engine.submit_order("BTC-USDT", "limit", "sell", 1.0, 50100)
    print(f"Order 3: {json.dumps(result3, indent=2)}")
    
    result4 = engine.submit_order("BTC-USDT", "limit", "sell", 2.5, 50200)
    print(f"Order 4: {json.dumps(result4, indent=2)}")
    
    # Check BBO
    print("\n=== BBO ===")
    print(json.dumps(engine.get_bbo("BTC-USDT"), indent=2))
    
    # Submit market order that crosses
    print("\n=== Market Order ===")
    result5 = engine.submit_order("BTC-USDT", "market", "buy", 1.5)
    print(f"Market Order: {json.dumps(result5, indent=2)}")
    
    # Check updated BBO
    print("\n=== Updated BBO ===")
    print(json.dumps(engine.get_bbo("BTC-USDT"), indent=2))
    
    # IOC order
    print("\n=== IOC Order ===")
    result6 = engine.submit_order("BTC-USDT", "ioc", "sell", 3.0, 49900)
    print(f"IOC Order: {json.dumps(result6, indent=2)}")
    
    # FOK order (should fail - not enough liquidity)
    print("\n=== FOK Order (Insufficient Liquidity) ===")
    result7 = engine.submit_order("BTC-USDT", "fok", "buy", 5.0, 51000)
    print(f"FOK Order: {json.dumps(result7, indent=2)}")