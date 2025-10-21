import requests
import json
from decimal import Decimal

def get_user_input():
    """Get order details from user with input validation"""
    try:
        # Get symbol
        print("\nAvailable trading pairs: BTC-USDT, ETH-USDT, ADA-USDT")
        symbol = input("Enter trading pair (e.g. BTC-USDT): ").strip().upper()
        
        # Get order type
        print("\nAvailable order types:")
        print("1. market   2. limit    3. ioc      4. fok")
        print("5. stop_loss            6. stop_limit")
        print("7. take_profit_market   8. take_profit_limit")
        order_type = input("Enter order type (1-8): ").strip()
        
        # Map numeric input to order type
        order_types = {
            "1": "market",
            "2": "limit",
            "3": "ioc",
            "4": "fok",
            "5": "stop_loss",
            "6": "stop_limit",
            "7": "take_profit_market",
            "8": "take_profit_limit"
        }
        order_type = order_types.get(order_type)
        if not order_type:
            raise ValueError("Invalid order type")

        # Get side
        side = input("\nEnter side (buy/sell): ").strip().lower()
        if side not in ["buy", "sell"]:
            raise ValueError("Invalid side")

        # Get quantity
        quantity = float(input("\nEnter quantity: ").strip())
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        # Get price for limit orders
        price = None
        if order_type in ["limit", "ioc", "fok", "stop_limit", "take_profit_limit"]:
            price = float(input("\nEnter price: ").strip())
            if price <= 0:
                raise ValueError("Price must be positive")

        # Get stop price for conditional orders
        stop_price = None
        if order_type in ["stop_loss", "stop_limit", "take_profit_market", "take_profit_limit"]:
            stop_price = float(input("\nEnter stop price: ").strip())
            if stop_price <= 0:
                raise ValueError("Stop price must be positive")

        return {
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "quantity": quantity,
            "price": price,
            "stop_price": stop_price
        }

    except ValueError as e:
        print(f"\nError: {str(e)}")
        return None
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return None

def submit_order(order_data):
    """Submit order to the matching engine API"""
    try:
        # Remove None values from the order data
        order_data = {k: v for k, v in order_data.items() if v is not None}
        
        # Send POST request to the order endpoint
        response = requests.post(
            "http://127.0.0.1:8000/order",
            json=order_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Pretty print the response
        print("\nOrder Response:")
        print(json.dumps(response.json(), indent=2))
        
        return response.status_code == 200
    
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the matching engine. Is it running?")
        return False
    except Exception as e:
        print(f"\nError submitting order: {str(e)}")
        return False

def main():
    while True:
        print("\n=== Cryptocurrency Order Entry ===")
        order_data = get_user_input()
        
        if order_data:
            if submit_order(order_data):
                print("\nOrder submitted successfully!")
            else:
                print("\nFailed to submit order.")
        
        # Ask if user wants to place another order
        again = input("\nSubmit another order? (y/n): ").strip().lower()
        if again != 'y':
            break

if __name__ == "__main__":
    main()