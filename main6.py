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
import os

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
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
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
    stop_price: Optional[Decimal] = None

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
            'stop_price': str(self.stop_price) if self.stop_price else None,
            'status': self.status.value,
            'filled_quantity': str(self.filled_quantity)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Order':
        return cls(
            order_id=data['order_id'],
            symbol=data['symbol'],
            order_type=OrderType(data['order_type']),
            side=OrderSide(data['side']),
            quantity=Decimal(data['quantity']),
            price=Decimal(data['price']) if data['price'] else None,
            timestamp=data['timestamp'],
            status=OrderStatus(data['status']),
            filled_quantity=Decimal(data['filled_quantity']),
            stop_price=Decimal(data['stop_price']) if data['stop_price'] else None
        )


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
    maker_fee: Decimal = Decimal('0')
    taker_fee: Decimal = Decimal('0')

    def to_dict(self) -> dict:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'price': str(self.price),
            'quantity': str(self.quantity),
            'timestamp': datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            'aggressor_side': self.aggressor_side.value,
            'maker_order_id': self.maker_order_id,
            'taker_order_id': self.taker_order_id,
            'maker_fee': str(self.maker_fee),
            'taker_fee': str(self.taker_fee)
        }


from typing import Optional
from collections.abc import Iterator

@dataclass
class Node:
    """Represents a node in the doubly linked list"""
    order: 'Order'
    prev: Optional['Node'] = None
    next: Optional['Node'] = None

class OrderList:
    """A doubly linked list implementation for orders"""
    def __init__(self):
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self._size: int = 0
    
    def append(self, order: 'Order') -> Node:
        node = Node(order)
        if not self.head:
            self.head = self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
        self._size += 1
        return node
    
    def remove_node(self, node: Node):
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
            
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        
        self._size -= 1
    
    def __len__(self) -> int:
        return self._size
    
    def __iter__(self) -> Iterator[Order]:
        current = self.head
        while current:
            yield current.order
            current = current.next

class PriceLevel:
    """Represents orders at a specific price level with FIFO queue"""
    def __init__(self, price: Decimal):
        self.price = price
        self.orders = OrderList()
        self.order_map: dict[str, Node] = {}
        self.total_quantity = Decimal('0')

    def add_order(self, order: Order):
        """Add order to this price level"""
        node = self.orders.append(order)
        self.order_map[order.order_id] = node
        self.total_quantity += order.remaining_quantity

    def remove_order(self, order_id: str) -> bool:
        """Remove order from this price level - O(1) complexity"""
        if order_id not in self.order_map:
            return False
            
        node = self.order_map[order_id]
        self.total_quantity -= node.order.remaining_quantity
        self.orders.remove_node(node)
        del self.order_map[order_id]
        return True

    def is_empty(self) -> bool:
        return len(self.orders) == 0

from sortedcontainers import SortedDict
from typing import Dict, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone

class OrderBook:
    """Order book for a single trading pair with price-time priority"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids = SortedDict()  # Sorted in descending order for bids
        self.asks = SortedDict()  # Sorted in ascending order for asks
        self.orders: Dict[str, Order] = {}

    def add_order(self, order: Order):
        """Add order to the book"""
        if order.side == OrderSide.BUY:
            if order.price not in self.bids:
                self.bids[order.price] = PriceLevel(order.price)
            self.bids[order.price].add_order(order)
        else:
            if order.price not in self.asks:
                self.asks[order.price] = PriceLevel(order.price)
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
        if self.bids:
            price = self.bids.peekitem(-1)[0]  # Get highest bid price
            return (price, self.bids[price].total_quantity)
        return None

    def get_best_ask(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best ask (lowest sell price)"""
        if self.asks:
            price = self.asks.peekitem(0)[0]  # Get lowest ask price
            return (price, self.asks[price].total_quantity)
        return None

    def get_bbo(self) -> Dict:
        """Get Best Bid and Offer"""
        bbo_update_start = time.perf_counter()
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        bbo_update_latency = (time.perf_counter() - bbo_update_start) * 1000
        logger.info(f"BBO update latency for {self.symbol}: {bbo_update_latency:.4f} ms")
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
        
        # Get top N bids (highest prices)
        for price in self.bids.keys()[-levels:]:    # change here for order
            level = self.bids[price]
            bids.insert(0, [str(price), str(level.total_quantity)])
            
        # Get top N asks (lowest prices)
        for price in self.asks.keys()[:levels]:
            level = self.asks[price]
            asks.append([str(price), str(level.total_quantity)])

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': self.symbol,
            'bids': bids,
            'asks': asks
        }
    

import asyncio


class MatchingEngine:
    """Core matching engine with REG NMS-inspired principles"""
    def __init__(self, state_file: str = 'order_book_state.json'):
        self.order_books: Dict[str, OrderBook] = {}
        self.trades: List[Trade] = []
        self.trade_callbacks: List = []
        self.market_data_callbacks: List = []
        self.conditional_orders: Dict[str, List[Order]] = defaultdict(list)
        self.last_trade_price: Dict[str, Decimal] = {}
        self.maker_fee_rate = Decimal('0.001')  # 0.1%
        self.taker_fee_rate = Decimal('0.002')  # 0.2%
        self.state_file = state_file
        self.load_state()

    def reset(self):
        """Clears all order books, trades, and conditional orders."""
        self.order_books: Dict[str, OrderBook] = {}
        self.trades: List[Trade] = []
        self.conditional_orders: Dict[str, List[Order]] = defaultdict(list)
        self.last_trade_price: Dict[str, Decimal] = {}
        os.remove(self.state_file) if os.path.exists(self.state_file) else None
        logger.info("Matching engine state has been reset.")

    def get_or_create_order_book(self, symbol: str) -> OrderBook:
        """Get or create order book for symbol"""
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol)
        return self.order_books[symbol]

    def submit_order(self, symbol: str, order_type: str, side: str,
                     quantity: float, price: Optional[float] = None,
                     stop_price: Optional[float] = None) -> Dict:
        """
        Submit a new order to the matching engine.
        This function assumes that input parameters have been pre-validated.
        """
        start_time = time.perf_counter()
        try:
            # Convert inputs to the required internal types
            order_type_enum = OrderType(order_type.lower())
            side_enum = OrderSide(side.lower())
            qty = Decimal(str(quantity))

            price_decimal = Decimal(str(price)) if price is not None else None
            stop_price_decimal = Decimal(str(stop_price)) if stop_price is not None else None

            # Create the order object
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                order_type=order_type_enum,
                side=side_enum,
                quantity=qty,
                price=price_decimal,
                timestamp=time.time(),
                stop_price=stop_price_decimal
            )

            logger.info(f"Received {order_type} {side} order: {order.order_id}")
            
            # Pass the validated and created order to the processing logic
            result = self._process_order(order)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.info(f"Order processing latency for {order.order_id}: {processing_time:.4f} ms")
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing order: {e}", exc_info=True)
            return {'status': 'error', 'reason': 'An internal error occurred during order processing.'}

    def _process_order(self, order: Order) -> Dict:
        """Process order through matching engine"""
        conditional_types = [OrderType.STOP_LOSS, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_MARKET, OrderType.TAKE_PROFIT_LIMIT]
        if order.order_type in conditional_types:
            return self._add_conditional_order(order)
        order_book = self.get_or_create_order_book(order.symbol)
        executions = []
        if order.order_type == OrderType.MARKET:
            executions = self._match_market_order(order, order_book)
        elif order.order_type == OrderType.IOC:
            executions = self._match_ioc_order(order, order_book)
        elif order.order_type == OrderType.FOK:
            executions = self._match_fok_order(order, order_book)
        else:
            executions = self._match_limit_order(order, order_book)
        if order.symbol in market_data_clients:
            asyncio.create_task(self._broadcast_market_data(order_book))
        
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
            logger.warning(f"Market order {order.order_id} partially filled, cancelling remaining {order.remaining_quantity}")
        else:
            order.status = OrderStatus.FILLED
        return trades

    def _match_ioc_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """Match IOC order - fill immediately or cancel"""
        trades = self._match_aggressive_order(order, order_book, allow_partial=True)
        if order.remaining_quantity > 0:
            order.status = OrderStatus.CANCELLED
            logger.info(f"IOC order {order.order_id} partially filled, cancelling remaining {order.remaining_quantity}")
        else:
            order.status = OrderStatus.FILLED
        return trades

    def _match_fok_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """Match FOK order - fill completely or cancel entirely"""
        if not self._can_fill_quantity(order, order_book, order.quantity):
            order.status = OrderStatus.CANCELLED
            logger.info(f"FOK order {order.order_id} cannot be fully filled, cancelling")
            return []
        trades = self._match_aggressive_order(order, order_book, allow_partial=False)
        order.status = OrderStatus.FILLED
        return trades

    def _match_limit_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """Match limit order with price-time priority"""
        trades = self._match_aggressive_order(order, order_book, allow_partial=True)
        if order.remaining_quantity > 0:
            order_book.add_order(order)
            order.status = OrderStatus.PARTIALLY_FILLED if order.filled_quantity > 0 else OrderStatus.NEW
            logger.info(f"Limit order {order.order_id} resting on book with {order.remaining_quantity} remaining")
        else:
            order.status = OrderStatus.FILLED
        return trades

    def _match_aggressive_order(self, order: Order, order_book: OrderBook, allow_partial: bool) -> List[Trade]:
        """
        Match an aggressive (marketable) order against the book.
        MODIFIED: This function is updated to work with the new OrderBook
        structure which uses SortedDict for price levels and a linked list for orders.
        """
        trades = []
        while order.remaining_quantity > 0:
            if order.side == OrderSide.BUY:
                if not order_book.asks:
                    break
                best_price = order_book.asks.peekitem(0)[0]
                if order.price is not None and best_price > order.price:
                    break
                level = order_book.asks[best_price]
            else: # SELL side
                if not order_book.bids:
                    break
                best_price = order_book.bids.peekitem(-1)[0]
                if order.price is not None and best_price < order.price:
                    break
                level = order_book.bids[best_price]

            # Iterate through orders at the best price level (FIFO)
            while len(level.orders) > 0 and order.remaining_quantity > 0:
                # Get the first order from the linked list at this price level
                maker_order = level.orders.head.order
                
                fill_qty = min(order.remaining_quantity, maker_order.remaining_quantity)
                trade = self._execute_trade(order, maker_order, best_price, fill_qty)
                trades.append(trade)

                # Update quantities on orders and the price level
                order.filled_quantity += fill_qty
                maker_order.filled_quantity += fill_qty
                level.total_quantity -= fill_qty

                if maker_order.remaining_quantity == 0:
                    maker_order.status = OrderStatus.FILLED
                    # Use the new OrderBook method which handles all removal logic internally
                    order_book.remove_order(maker_order.order_id)
                    logger.info(f"Maker order {maker_order.order_id} fully filled")
        return trades

    def _can_fill_quantity(self, order: Order, order_book: OrderBook, quantity: Decimal) -> bool:
        """
        Check if quantity can be filled at acceptable prices.
        MODIFIED: This function is updated to iterate efficiently over the
        new SortedDict keys for bids and asks.
        """
        available = Decimal('0')
        if order.side == OrderSide.BUY:
            # order_book.asks.keys() is already sorted from low to high price
            for price in order_book.asks.keys():
                if order.price is not None and price > order.price:
                    break
                available += order_book.asks[price].total_quantity
                if available >= quantity:
                    return True
        else: # SELL side
            # Use reversed() for bids to go from high to low price
            for price in reversed(order_book.bids.keys()):
                if order.price is not None and price < order.price:
                    break
                available += order_book.bids[price].total_quantity
                if available >= quantity:
                    return True
        return False

    def _execute_trade(self, taker: Order, maker: Order, price: Decimal, quantity: Decimal) -> Trade:
        """Execute a trade between two orders"""
        trade_start_time = time.perf_counter()
        maker_fee = quantity * price * self.maker_fee_rate
        taker_fee = quantity * price * self.taker_fee_rate
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=taker.symbol,
            price=price,
            quantity=quantity,
            timestamp=time.time(),
            aggressor_side=taker.side,
            maker_order_id=maker.order_id,
            taker_order_id=taker.order_id,
            maker_fee=maker_fee,
            taker_fee=taker_fee
        )
        self.trades.append(trade)
        logger.info(f"Trade executed: {trade.trade_id} - {quantity} @ {price}")
        self.last_trade_price[taker.symbol] = price
        if trade_clients:
            asyncio.create_task(self._broadcast_trade(trade))
        trade_generation_latency = (time.perf_counter() - trade_start_time) * 1000
        logger.info(f"Trade data generation latency for {trade.trade_id}: {trade_generation_latency:.4f} ms")
        self._check_conditional_orders(taker.symbol, price)
        return trade

    async def _broadcast_trade(self, trade: Trade):
        """Broadcast trade to subscribers"""
        for callback in self.trade_callbacks:
            try:
                await callback(trade.to_dict())
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")

    async def _broadcast_market_data(self, order_book: OrderBook):
        """Broadcast market data update"""
        data = {
            'depth': order_book.get_depth(),
            'bbo': order_book.get_bbo()
        }
        for callback in self.market_data_callbacks:
            try:
                await callback(data)
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

    def _add_conditional_order(self, order: Order) -> Dict:
        """Add a conditional order to the holding list."""
        self.conditional_orders[order.symbol].append(order)
        logger.info(f"Accepted conditional {order.order_type.value} order {order.order_id}, waiting for trigger.")
        return {
            'status': order.status.value,
            'order_id': order.order_id,
            'reason': 'Order is conditional and is waiting for trigger price.'
        }

    def _check_conditional_orders(self, symbol: str, current_price: Decimal):
        """Check and trigger any conditional orders."""
        triggered_orders = []
        remaining_orders = []
        for order in self.conditional_orders[symbol]:
            should_trigger = False
            if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
                if order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_trigger = True
                elif order.side == OrderSide.BUY and current_price >= order.stop_price:
                    should_trigger = True
            elif order.order_type in [OrderType.TAKE_PROFIT_MARKET, OrderType.TAKE_PROFIT_LIMIT]:
                if order.side == OrderSide.SELL and current_price >= order.stop_price:
                    should_trigger = True
                elif order.side == OrderSide.BUY and current_price <= order.stop_price:
                    should_trigger = True
            if should_trigger:
                triggered_orders.append(order)
            else:
                remaining_orders.append(order)
        self.conditional_orders[symbol] = remaining_orders
        for order in triggered_orders:
            logger.info(f"Triggering conditional order {order.order_id} ({order.order_type.value})")
            if order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT_MARKET]:
                order.order_type = OrderType.MARKET
            elif order.order_type in [OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
                order.order_type = OrderType.LIMIT
            order.timestamp = time.time()
            self._process_order(order)

    def save_state(self):
        """Save the current state of the order books to a file."""
        state = {
            'order_books': {
                symbol: [order.to_dict() for order in book.orders.values()]
                for symbol, book in self.order_books.items()
            },
            'conditional_orders': {
                symbol: [order.to_dict() for order in orders]
                for symbol, orders in self.conditional_orders.items()
            }
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Order book state saved to {self.state_file}")

    def load_state(self):
        """Load the order book state from a file."""
        if not os.path.exists(self.state_file):
            logger.info("No state file found, starting with a fresh order book.")
            return
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        for symbol, orders_data in state.get('order_books', {}).items():
            order_book = self.get_or_create_order_book(symbol)
            for order_data in orders_data:
                order = Order.from_dict(order_data)
                order_book.add_order(order)
        for symbol, orders_data in state.get('conditional_orders', {}).items():
            for order_data in orders_data:
                order = Order.from_dict(order_data)
                self.conditional_orders[symbol].append(order)
        logger.info(f"Order book state loaded from {self.state_file}")


# Add these imports at the top of your file
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
import uvicorn
import atexit
# --- API and Server Implementation ---

# Create the matching engine instance
engine = MatchingEngine()

# Register the save_state function to be called on exit
atexit.register(engine.save_state)

# Create Flask app and Sock for WebSockets
app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket Client Management ---
trade_clients = []
market_data_clients = defaultdict(list)


# --- API Endpoints ---

@app.post('/reset')
async def reset_engine_api():
    """
    Resets the entire matching engine, clearing all order books,
    trades, and conditional orders for all symbols.
    """
    try:
        engine.reset()
        return JSONResponse(
            content={'status': 'success', 'message': 'Matching engine reset successfully.'},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Failed to reset engine: {e}", exc_info=True)
        return JSONResponse(
            content={'status': 'error', 'reason': 'An internal error occurred during reset.'},
            status_code=500
        )


@app.post('/order')
async def submit_order_api(request: Request):
    """
    REST API endpoint for submitting a new order.
    Expects a JSON payload with order details.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(content={'status': 'rejected', 'reason': 'Invalid JSON payload'}, status_code=400)

    required_fields = ['symbol', 'order_type', 'side', 'quantity']
    if not all(field in data for field in required_fields):
        return JSONResponse(content={'status': 'rejected', 'reason': 'Missing required fields'}, status_code=400)

    try:
        quantity = float(data['quantity'])
        if quantity <= 0:
            return JSONResponse(content={'status': 'rejected', 'reason': 'Quantity must be positive'}, status_code=400)

        price = data.get('price')
        if price is not None:
            price = float(price)
            if price <= 0:
                return JSONResponse(content={'status': 'rejected', 'reason': 'Price must be positive'}, status_code=400)

        stop_price = data.get('stop_price')
        if stop_price is not None:
            stop_price = float(stop_price)
            if stop_price <= 0:
                return JSONResponse(content={'status': 'rejected', 'reason': 'Stop price must be positive'}, status_code=400)
    except (ValueError, TypeError):
        return JSONResponse(content={'status': 'rejected', 'reason': 'Invalid numeric value for quantity, price, or stop_price'}, status_code=400)


    order_type_lower = data['order_type'].lower()
    limit_based_types = ['limit', 'ioc', 'fok', 'stop_limit', 'take_profit_limit']
    if order_type_lower in limit_based_types and 'price' not in data:
        return JSONResponse(content={'status': 'rejected', 'reason': 'Price is required for this order type'}, status_code=400)

    conditional_types = ['stop_loss', 'stop_limit', 'take_profit_market', 'take_profit_limit']
    if order_type_lower in conditional_types and 'stop_price' not in data:
        return JSONResponse(content={'status': 'rejected', 'reason': 'Stop price is required for this order type'}, status_code=400)

    try:
        result = engine.submit_order(
            symbol=data['symbol'],
            order_type=data['order_type'],
            side=data['side'],
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"API Error processing order: {e}", exc_info=True)
        return JSONResponse(content={'status': 'error', 'reason': str(e)}, status_code=500)

@app.websocket("/ws/trades")
async def trade_feed(websocket: WebSocket):
    """
    WebSocket endpoint for the real-time trade execution feed.
    """
    await websocket.accept()
    logger.info("Trade feed client connected.")
    trade_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        logger.info(f"Trade feed client disconnected: {e}")
    finally:
        trade_clients.remove(websocket)


@app.websocket("/ws/marketdata/{symbol}")
async def market_data_feed(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time market data (order book depth).
    """
    await websocket.accept()
    logger.info(f"Market data client connected for symbol: {symbol}")
    market_data_clients[symbol].append(websocket)
    try:
        order_book = engine.get_or_create_order_book(symbol)
        await websocket.send_json(order_book.get_depth())
        while True:
            await websocket.receive_text()
    except Exception as e:
        logger.info(f"Market data client for {symbol} disconnected: {e}")
    finally:
        market_data_clients[symbol].remove(websocket)


# --- Callback Functions for Broadcasting Data ---

async def broadcast_trade_update(trade_data: dict):
    """
    Callback to send trade data to all connected WebSocket clients.
    """
    payload = json.dumps(trade_data)
    disconnected_clients = []
    for client in trade_clients:
        try:
            await client.send_text(payload)
        except Exception as e:
            logger.error(f"Failed to send trade update to client: {e}")
            disconnected_clients.append(client)
    for client in disconnected_clients:
        trade_clients.remove(client)


async def broadcast_market_data_update(market_data: dict):
    """
    Callback to send market data updates to subscribed WebSocket clients.
    """
    symbol = market_data.get('symbol')
    if symbol and symbol in market_data_clients:
        payload = json.dumps(market_data)
        disconnected_clients = []
        for client in market_data_clients[symbol]:
            try:
                await client.send_text(payload)
            except Exception as e:
                logger.error(f"Failed to send market data to client for {symbol}: {e}")
                disconnected_clients.append(client)
        for client in disconnected_clients:
            market_data_clients[symbol].remove(client)
# --- Main Application Runner ---

if __name__ == "__main__":
    engine.subscribe_trades(broadcast_trade_update)
    engine.subscribe_market_data(broadcast_market_data_update)

    def populate_book():
        logger.info("Pre-populating order book for BTC-USDT...")
        engine.submit_order("BTC-USDT", "limit", "buy", 1.5, 50000)
        engine.submit_order("BTC-USDT", "limit", "buy", 2.0, 49900)
        engine.submit_order("BTC-USDT", "limit", "sell", 1.0, 50100)
        engine.submit_order("BTC-USDT", "limit", "sell", 2.5, 50200)
        logger.info("Order book populated.")
    import threading
    threading.Thread(target=populate_book, daemon=True).start()
    logger.info("Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
