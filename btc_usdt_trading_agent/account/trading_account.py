import datetime

class TradingAccount:
    """
    Simulates a cryptocurrency trading account for managing USDT and BTC balances,
    and recording transaction history.
    """
    FEE_PERCENTAGE = 0.001  # 0.1% transaction fee

    def __init__(self, initial_usdt_balance: float = 1000.0):
        """
        Initializes the trading account.

        Args:
            initial_usdt_balance: The starting balance in USDT. Defaults to 1000.0.
        """
        if initial_usdt_balance < 0:
            raise ValueError("Initial USDT balance cannot be negative.")
        
        self.usdt_balance: float = initial_usdt_balance
        self.btc_balance: float = 0.0
        self.transaction_history: list[dict] = []

    def get_balance(self) -> dict:
        """
        Retrieves the current USDT and BTC balances.

        Returns:
            A dictionary with "usdt_balance" and "btc_balance".
        """
        return {"usdt_balance": self.usdt_balance, "btc_balance": self.btc_balance}

    def record_transaction(self, timestamp: str, type: str, symbol: str, 
                           amount_crypto: float, price: float, total_usdt: float, 
                           reason: str, fee: float = 0.0) -> None:
        """
        Adds a new transaction to the transaction history.

        Args:
            timestamp: ISO 8601 formatted string of the transaction time.
            type: "BUY" or "SELL".
            symbol: Trading symbol (e.g., "BTCUSDT").
            amount_crypto: Amount of cryptocurrency (BTC) bought or sold.
            price: Price per unit of cryptocurrency in USDT.
            total_usdt: Total USDT value of the transaction (gross for buy, net for sell before fee).
            reason: Justification for the trade.
            fee: Transaction fee in USDT.
        """
        transaction = {
            "timestamp": timestamp,
            "type": type,
            "symbol": symbol,
            "amount_crypto": amount_crypto,
            "price": price,
            "total_usdt_value_before_fee": total_usdt,
            "reason": reason,
            "fee_usdt": fee,
        }
        self.transaction_history.append(transaction)

    def execute_buy_order(self, symbol: str, usdt_amount_to_spend: float, 
                          current_btc_price: float, reason: str, timestamp: str) -> dict:
        """
        Executes a buy order for BTC using USDT.
        """
        if usdt_amount_to_spend <= 0:
            return {"success": False, "message": "USDT amount to spend must be positive."}
        if current_btc_price <= 0:
            return {"success": False, "message": "BTC price must be positive."}

        fee = usdt_amount_to_spend * self.FEE_PERCENTAGE
        total_usdt_deducted = usdt_amount_to_spend + fee

        if total_usdt_deducted > self.usdt_balance:
            return {"success": False, "message": "Insufficient USDT balance to cover amount and fee."}

        btc_to_buy = usdt_amount_to_spend / current_btc_price
        
        self.usdt_balance -= total_usdt_deducted
        self.btc_balance += btc_to_buy
        
        self.record_transaction(
            timestamp=timestamp,
            type="BUY",
            symbol=symbol,
            amount_crypto=btc_to_buy,
            price=current_btc_price,
            total_usdt=usdt_amount_to_spend,
            reason=reason,
            fee=fee
        )
        
        return {
            "success": True, 
            "message": "Buy order executed successfully.", 
            "btc_bought": btc_to_buy, 
            "usdt_spent_before_fee": usdt_amount_to_spend,
            "fee_usdt": fee,
            "total_usdt_deducted": total_usdt_deducted
        }

    def execute_sell_order(self, symbol: str, btc_amount_to_sell: float, 
                           current_btc_price: float, reason: str, timestamp: str) -> dict:
        """
        Executes a sell order for BTC, receiving USDT.
        """
        if btc_amount_to_sell <= 0:
            return {"success": False, "message": "BTC amount to sell must be positive."}
        if current_btc_price <= 0:
            return {"success": False, "message": "BTC price must be positive."}
        if btc_amount_to_sell > self.btc_balance:
            return {"success": False, "message": "Insufficient BTC balance."}

        usdt_value_of_btc_sold = btc_amount_to_sell * current_btc_price
        fee = usdt_value_of_btc_sold * self.FEE_PERCENTAGE
        usdt_received_after_fee = usdt_value_of_btc_sold - fee
        
        self.btc_balance -= btc_amount_to_sell
        self.usdt_balance += usdt_received_after_fee
        
        self.record_transaction(
            timestamp=timestamp,
            type="SELL",
            symbol=symbol,
            amount_crypto=btc_amount_to_sell,
            price=current_btc_price,
            total_usdt=usdt_value_of_btc_sold,
            reason=reason,
            fee=fee
        )
        
        return {
            "success": True,
            "message": "Sell order executed successfully.",
            "btc_sold": btc_amount_to_sell,
            "usdt_value_before_fee": usdt_value_of_btc_sold,
            "fee_usdt": fee,
            "usdt_received_after_fee": usdt_received_after_fee
        }
```
