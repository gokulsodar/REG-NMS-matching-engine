"""
Real-Time Trading Dashboard
Displays live market data, trades, and analytics with interactive plots
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from collections import deque
import threading
import time

import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingDashboard:
    """Real-time trading dashboard with plots and market data"""
    
    def __init__(self, root, symbols=None, base_url="ws://127.0.0.1:8000"):
        self.root = root
        self.root.title("Crypto Trading Dashboard")
        self.root.geometry("1600x900")
        
        self.symbols = symbols or ["BTC-USDT", "ETH-USDT", "ADA-USDT"]
        self.base_url = base_url
        self.current_symbol = self.symbols[0]
        
        # Data storage
        self.market_data = {symbol: {} for symbol in self.symbols}
        self.trades = {symbol: deque(maxlen=100) for symbol in self.symbols}
        self.price_history = {symbol: deque(maxlen=50) for symbol in self.symbols}
        self.time_history = {symbol: deque(maxlen=50) for symbol in self.symbols}
        
        # WebSocket connections
        self.running = False
        self.ws_tasks = []
        
        # Build UI
        self.setup_ui()
        
        # Start async event loop in separate thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.start_async_loop, daemon=True)
        self.thread.start()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Top frame for symbol selection and BBO
        top_frame = tk.Frame(self.root, bg="#1e1e1e", height=100)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Symbol selector
        tk.Label(top_frame, text="Symbol:", bg="#1e1e1e", fg="white", 
                font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.symbol_var = tk.StringVar(value=self.current_symbol)
        symbol_dropdown = ttk.Combobox(top_frame, textvariable=self.symbol_var, 
                                      values=self.symbols, state="readonly", 
                                      font=("Arial", 11), width=15)
        symbol_dropdown.pack(side=tk.LEFT, padx=5)
        symbol_dropdown.bind("<<ComboboxSelected>>", self.on_symbol_change)
        
        # Fee information frame
        fee_frame = tk.LabelFrame(top_frame, text="Fee Structure", bg="#1e1e1e", 
                                 fg="white", font=("Arial", 10, "bold"))
        fee_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH)
        
        tk.Label(fee_frame, text="Maker: 0.1%", bg="#1e1e1e", fg="#00ccff", 
                font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        tk.Label(fee_frame, text="Taker: 0.2%", bg="#1e1e1e", fg="#ff9900", 
                font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        # Middle frame - Charts
        chart_frame = tk.Frame(self.root, bg="#2e2e2e")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Price chart
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 6), 
                                                       facecolor="#2e2e2e")
        self.fig.tight_layout(pad=3)
        
        # Configure price chart
        self.ax1.set_facecolor("#1e1e1e")
        self.ax1.set_title("Price History", color="white", fontsize=12, fontweight="bold")
        self.ax1.set_xlabel("Time", color="white")
        self.ax1.set_ylabel("Price (USDT)", color="white")
        self.ax1.tick_params(colors="white")
        self.ax1.grid(True, alpha=0.2)
        
        # Configure volume chart
        self.ax2.set_facecolor("#1e1e1e")
        self.ax2.set_title("Trade Volume", color="white", fontsize=12, fontweight="bold")
        self.ax2.set_xlabel("Time", color="white")
        self.ax2.set_ylabel("Volume", color="white")
        self.ax2.tick_params(colors="white")
        self.ax2.grid(True, alpha=0.2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bottom frame - Order book and trades
        bottom_frame = tk.Frame(self.root, bg="#2e2e2e")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Order book
        orderbook_frame = tk.LabelFrame(bottom_frame, text="Order Book Depth", 
                                       bg="#1e1e1e", fg="white", 
                                       font=("Arial", 11, "bold"))
        orderbook_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.orderbook_text = scrolledtext.ScrolledText(orderbook_frame, 
                                                        bg="#1e1e1e", fg="white", 
                                                        font=("Courier", 9), 
                                                        height=12, wrap=tk.WORD)
        self.orderbook_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Recent trades
        trades_frame = tk.LabelFrame(bottom_frame, text="Recent Trades", 
                                    bg="#1e1e1e", fg="white", 
                                    font=("Arial", 11, "bold"))
        trades_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.trades_text = scrolledtext.ScrolledText(trades_frame, 
                                                     bg="#1e1e1e", fg="white", 
                                                     font=("Courier", 9), 
                                                     height=12, wrap=tk.WORD)
        self.trades_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Start animation
        self.anim = FuncAnimation(self.fig, self.update_charts, interval=1000, 
                                 blit=False, cache_frame_data=False)
    
    def on_symbol_change(self, event):
        """Handle symbol selection change"""
        self.current_symbol = self.symbol_var.get()
        self.update_display()
    
    def update_display(self):
        """Update all display elements"""
        self.update_orderbook()
        self.update_trades_display()
    
    def update_orderbook(self):
        """Update order book display"""
        self.orderbook_text.delete(1.0, tk.END)
        data = self.market_data.get(self.current_symbol, {})
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        self.orderbook_text.insert(tk.END, f"{'='*60}\n", "header")
        self.orderbook_text.insert(tk.END, f"Order Book - {self.current_symbol}\n", "header")
        self.orderbook_text.insert(tk.END, f"{'='*60}\n\n", "header")
        
        # Display asks (reversed order - highest first)
        self.orderbook_text.insert(tk.END, "ASKS (Sell Orders)\n", "header")
        self.orderbook_text.insert(tk.END, f"{'-'*60}\n")
        self.orderbook_text.insert(tk.END, f"{'Price':>15} {'Quantity':>15} {'Total':>15}\n")
        self.orderbook_text.insert(tk.END, f"{'-'*60}\n")
        
        for price, qty in asks[:10]:
            total = float(price) * float(qty)
            line = f"{float(price):>15,.2f} {float(qty):>15,.4f} {total:>15,.2f}\n"
            self.orderbook_text.insert(tk.END, line, "ask")
        
        self.orderbook_text.insert(tk.END, f"\n{'-'*60}\n\n")
        
        # Display bids
        self.orderbook_text.insert(tk.END, "BIDS (Buy Orders)\n", "header")
        self.orderbook_text.insert(tk.END, f"{'-'*60}\n")
        self.orderbook_text.insert(tk.END, f"{'Price':>15} {'Quantity':>15} {'Total':>15}\n")
        self.orderbook_text.insert(tk.END, f"{'-'*60}\n")
        
        for price, qty in reversed(bids[:10]):
            total = float(price) * float(qty)
            line = f"{float(price):>15,.2f} {float(qty):>15,.4f} {total:>15,.2f}\n"
            self.orderbook_text.insert(tk.END, line, "bid")
        
        # Configure tags
        self.orderbook_text.tag_config("header", foreground="#00ccff", font=("Courier", 9, "bold"))
        self.orderbook_text.tag_config("ask", foreground="#ff6666")
        self.orderbook_text.tag_config("bid", foreground="#66ff66")
    
    def update_trades_display(self):
        """Update recent trades display"""
        self.trades_text.delete(1.0, tk.END)
        trades = self.trades.get(self.current_symbol, [])
        
        self.trades_text.insert(tk.END, f"{'='*80}\n", "header")
        self.trades_text.insert(tk.END, f"Recent Trades - {self.current_symbol}\n", "header")
        self.trades_text.insert(tk.END, f"{'='*80}\n\n", "header")
        self.trades_text.insert(tk.END, f"{'Time':>10} {'Side':>6} {'Price':>12} {'Qty':>10} {'Maker Fee':>12} {'Taker Fee':>12}\n")
        self.trades_text.insert(tk.END, f"{'-'*80}\n")
        
        for trade in reversed(list(trades)[-20:]):  # Show last 20 trades
            timestamp = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            time_str = timestamp.strftime("%H:%M:%S")
            side = trade['aggressor_side'].upper()
            price = float(trade['price'])
            qty = float(trade['quantity'])
            maker_fee = float(trade['maker_fee'])
            taker_fee = float(trade['taker_fee'])
            
            # Calculate total costs
            trade_value = price * qty
            maker_total = trade_value + maker_fee
            taker_total = trade_value + taker_fee
            
            line = f"{time_str:>10} {side:>6} {price:>12,.2f} {qty:>10,.4f} {maker_fee:>12,.4f} {taker_fee:>12,.4f}\n"
            
            if side == "BUY":
                self.trades_text.insert(tk.END, line, "buy")
            else:
                self.trades_text.insert(tk.END, line, "sell")
        
        # Configure tags
        self.trades_text.tag_config("header", foreground="#00ccff", font=("Courier", 9, "bold"))
        self.trades_text.tag_config("buy", foreground="#66ff66")
        self.trades_text.tag_config("sell", foreground="#ff6666")
    
    def update_charts(self, frame):
        """Update price and volume charts"""
        if not self.price_history[self.current_symbol]:
            return
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Get data
        times = list(self.time_history[self.current_symbol])
        prices = list(self.price_history[self.current_symbol])
        trades = list(self.trades[self.current_symbol])
        
        # Plot price
        self.ax1.plot(times, prices, color="#00ff00", linewidth=2, marker='o', markersize=4)
        self.ax1.set_facecolor("#1e1e1e")
        self.ax1.set_title(f"Price History - {self.current_symbol}", color="white", 
                          fontsize=12, fontweight="bold")
        self.ax1.set_xlabel("Time", color="white")
        self.ax1.set_ylabel("Price (USDT)", color="white")
        self.ax1.tick_params(colors="white")
        self.ax1.grid(True, alpha=0.2, color="gray")
        
        # Format x-axis
        if times:
            self.ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            self.fig.autofmt_xdate()
        
        # Plot volume
        if trades:
            trade_times = [datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')) 
                          for t in trades[-20:]]
            volumes = [float(t['quantity']) for t in trades[-20:]]
            colors = ['#66ff66' if t['aggressor_side'] == 'buy' else '#ff6666' 
                     for t in trades[-20:]]
            
            self.ax2.bar(trade_times, volumes, color=colors, alpha=0.7, width=0.001)
            self.ax2.set_facecolor("#1e1e1e")
            self.ax2.set_title("Trade Volume", color="white", fontsize=12, fontweight="bold")
            self.ax2.set_xlabel("Time", color="white")
            self.ax2.set_ylabel("Volume", color="white")
            self.ax2.tick_params(colors="white")
            self.ax2.grid(True, alpha=0.2, color="gray")
            
            if trade_times:
                self.ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        
        self.fig.tight_layout()
    
    async def connect_market_data(self, symbol):
        """Connect to market data WebSocket"""
        url = f"{self.base_url}/ws/marketdata/{symbol}"
        
        try:
            async with websockets.connect(url) as ws:
                logger.info(f"Connected to market data: {symbol}")
                while self.running:
                    try:
                        message = await ws.recv()
                        data = json.loads(message)
                        self.market_data[symbol] = data
                        
                        # Update price history
                        if data.get('asks'):
                            price = float(data['asks'][0][0])
                            self.price_history[symbol].append(price)
                            self.time_history[symbol].append(datetime.now())
                        
                        # Update UI if current symbol
                        if symbol == self.current_symbol:
                            self.root.after(0, self.update_display)
                            
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        logger.error(f"Market data error for {symbol}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to connect market data {symbol}: {e}")
    
    async def connect_trades(self):
        """Connect to trades WebSocket"""
        url = f"{self.base_url}/ws/trades"
        
        try:
            async with websockets.connect(url) as ws:
                logger.info("Connected to trade feed")
                while self.running:
                    try:
                        message = await ws.recv()
                        trade = json.loads(message)
                        symbol = trade['symbol']
                        self.trades[symbol].append(trade)
                        
                        # Update UI if current symbol
                        if symbol == self.current_symbol:
                            self.root.after(0, self.update_trades_display)
                            
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        logger.error(f"Trade feed error: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to connect to trade feed: {e}")
    
    def start_async_loop(self):
        """Start the asyncio event loop"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.run_websockets())
    
    async def run_websockets(self):
        """Run all WebSocket connections"""
        self.running = True
        
        # Create tasks for market data feeds
        tasks = [self.connect_market_data(symbol) for symbol in self.symbols]
        
        # Add trade feed task
        tasks.append(self.connect_trades())
        
        # Run all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        time.sleep(0.5)
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    app = TradingDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()