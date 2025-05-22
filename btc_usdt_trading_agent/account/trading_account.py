import datetime
import uuid
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class OpenPosition(BaseModel):
    """
    Represents an open trading position with details including stop-loss and take-profit levels.
    """
    position_id: str = Field(default_factory=lambda: f"pos_{uuid.uuid4().hex[:8]}")
    symbol: str
    amount_crypto: float # Amount of BTC in this position
    entry_price: float   # Price at which BTC was bought
    timestamp: str       # ISO 8601 timestamp of when the position was opened
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    status: Literal["OPEN", "CLOSED"] = "OPEN"
    closure_reason: Optional[str] = None # e.g., "manual", "stop-loss", "take-profit"
    exit_price: Optional[float] = None
    closed_timestamp: Optional[str] = None

class TradingAccount:
    """
    Simulates a cryptocurrency trading account for managing USDT and BTC balances,
    recording transaction history, and tracking open positions with SL/TP.
    """
    FEE_PERCENTAGE = 0.001  # 0.1% transaction fee
    DEFAULT_STOP_LOSS_PCT = 0.05  # 5% below entry price
    DEFAULT_TAKE_PROFIT_PCT = 0.10 # 10% above entry price

    def __init__(self, initial_usdt_balance: float = 1000.0):
        if initial_usdt_balance < 0:
            raise ValueError("Initial USDT balance cannot be negative.")
        
        self.usdt_balance: float = initial_usdt_balance
        self.btc_balance: float = 0.0 # Total BTC held, derived from open positions
        self.transaction_history: list[dict] = []
        self.open_positions: List[OpenPosition] = []

    def _calculate_total_btc_in_open_positions(self) -> float:
        """Calculates total BTC held in all 'OPEN' positions."""
        return sum(pos.amount_crypto for pos in self.open_positions if pos.status == "OPEN")

    def get_balance(self) -> dict:
        """Retrieves the current USDT and total BTC balances."""
        # Ensure btc_balance is consistent if direct manipulation occurred elsewhere (should not happen with this design)
        self.btc_balance = self._calculate_total_btc_in_open_positions()
        return {"usdt_balance": self.usdt_balance, "btc_balance": self.btc_balance}

    def record_transaction(self, timestamp: str, type: str, symbol: str, 
                           amount_crypto: float, price: float, total_usdt: float, 
                           reason: str, fee: float = 0.0, position_id: Optional[str] = None) -> None:
        """Adds a new transaction to the transaction history."""
        transaction = {
            "timestamp": timestamp,
            "type": type,
            "symbol": symbol,
            "amount_crypto": amount_crypto,
            "price": price,
            "total_usdt_value_before_fee": total_usdt,
            "reason": reason, # Can include "manual", "stop-loss", "take-profit" for sells
            "fee_usdt": fee,
            "position_id": position_id
        }
        self.transaction_history.append(transaction)

    def execute_buy_order(self, symbol: str, usdt_amount_to_spend: float, 
                          current_btc_price: float, reason: str, timestamp: str,
                          stop_loss_price: Optional[float] = None, 
                          take_profit_price: Optional[float] = None) -> dict:
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
        
        # Calculate default SL/TP if not provided
        actual_stop_loss = stop_loss_price if stop_loss_price is not None else current_btc_price * (1 - self.DEFAULT_STOP_LOSS_PCT)
        actual_take_profit = take_profit_price if take_profit_price is not None else current_btc_price * (1 + self.DEFAULT_TAKE_PROFIT_PCT)

        new_position = OpenPosition(
            symbol=symbol,
            amount_crypto=btc_to_buy,
            entry_price=current_btc_price,
            timestamp=timestamp,
            stop_loss_price=actual_stop_loss,
            take_profit_price=actual_take_profit
        )
        self.open_positions.append(new_position)
        self.btc_balance = self._calculate_total_btc_in_open_positions() # Update total BTC balance

        self.record_transaction(
            timestamp=timestamp, type="BUY", symbol=symbol,
            amount_crypto=btc_to_buy, price=current_btc_price,
            total_usdt=usdt_amount_to_spend, reason=reason, fee=fee,
            position_id=new_position.position_id
        )
        
        return {
            "success": True, 
            "message": "Buy order executed successfully, new position opened.", 
            "position_id": new_position.position_id,
            "btc_bought": btc_to_buy, 
            "usdt_spent_before_fee": usdt_amount_to_spend,
            "fee_usdt": fee,
            "total_usdt_deducted": total_usdt_deducted,
            "entry_price": current_btc_price,
            "stop_loss_price": actual_stop_loss,
            "take_profit_price": actual_take_profit
        }

    def execute_sell_order(self, symbol: str, btc_amount_to_sell: float, # This amount might be ignored if position_id is given
                           current_btc_price: float, reason: str, timestamp: str, 
                           position_id_to_close: Optional[str] = None) -> dict:
        if current_btc_price <= 0:
            return {"success": False, "message": "BTC price must be positive."}

        position_to_close: Optional[OpenPosition] = None
        btc_actually_sold = 0.0

        if position_id_to_close:
            found_pos = next((p for p in self.open_positions if p.position_id == position_id_to_close and p.status == "OPEN"), None)
            if not found_pos:
                return {"success": False, "message": f"Open position with ID '{position_id_to_close}' not found."}
            position_to_close = found_pos
            btc_actually_sold = position_to_close.amount_crypto # Assume full close
        elif btc_amount_to_sell > 0 : # Manual sell without specific position ID, try to close oldest matching amount
            # Simplified for now: find oldest position that can be fully closed by this amount.
            # This logic can be complex (e.g. partial closes, multiple positions).
            # For this iteration: if btc_amount_to_sell is specified, we try to find an exact match in oldest positions.
            # If not, we'll use the "close oldest open position" logic below.
            # This part needs careful thought for a real system.
            # Revised simplification: If no position_id, close the oldest open position IF btc_amount_to_sell matches its amount.
            # If btc_amount_to_sell is very different, it's ambiguous.
            # For now, let's prioritize closing by ID. If no ID, and btc_amount_to_sell > 0, try to find *any* open pos.
            
            # Simplest manual sell: close the oldest open position. The btc_amount_to_sell arg is used if provided and matches.
            # If not, the position's amount is used.
            oldest_open_position = next((p for p in sorted(self.open_positions, key=lambda x: x.timestamp) if p.status == "OPEN"), None)
            if not oldest_open_position:
                 return {"success": False, "message": "No open positions to sell."}
            position_to_close = oldest_open_position
            # If btc_amount_to_sell was specified and is different from the position's amount, it's ambiguous.
            # For this version, we'll assume if position_id_to_close is None, the LLM intends to sell *a* position,
            # and we pick the oldest. The amount from that position is used.
            btc_actually_sold = position_to_close.amount_crypto
            if btc_amount_to_sell > 0 and abs(btc_amount_to_sell - btc_actually_sold) > 1e-9: # Comparing floats
                 # This indicates LLM might want to sell a specific amount not tied to one full position.
                 # For now, we only support closing a full position with this simplified manual sell.
                 return {"success": False, "message": f"Manual sell for {btc_amount_to_sell} BTC does not match oldest open position amount ({btc_actually_sold:.8f} BTC). Only full position closure supported for manual sells without ID."}

        else: # No position_id and btc_amount_to_sell is 0 or negative
             return {"success": False, "message": "BTC amount to sell must be positive or a position_id provided."}


        if not position_to_close or btc_actually_sold <= 0: # Should be caught by above, but as safeguard
            return {"success": False, "message": "No valid position or amount to sell."}
        
        if btc_actually_sold > self._calculate_total_btc_in_open_positions(): # Redundant if using position logic, but safe
            return {"success": False, "message": "Insufficient total BTC balance for the operation."}


        usdt_value_of_btc_sold = btc_actually_sold * current_btc_price
        fee = usdt_value_of_btc_sold * self.FEE_PERCENTAGE
        usdt_received_after_fee = usdt_value_of_btc_sold - fee
        
        # Update position status
        position_to_close.status = "CLOSED"
        position_to_close.closure_reason = reason 
        position_to_close.exit_price = current_btc_price
        position_to_close.closed_timestamp = timestamp
        
        self.usdt_balance += usdt_received_after_fee
        self.btc_balance = self._calculate_total_btc_in_open_positions() # Recalculate based on remaining open positions

        self.record_transaction(
            timestamp=timestamp, type="SELL", symbol=symbol,
            amount_crypto=btc_actually_sold, price=current_btc_price,
            total_usdt=usdt_value_of_btc_sold, reason=reason, fee=fee,
            position_id=position_to_close.position_id
        )
        
        return {
            "success": True,
            "message": f"Sell order executed successfully. Position {position_to_close.position_id} closed.",
            "position_id_closed": position_to_close.position_id,
            "btc_sold": btc_actually_sold,
            "usdt_value_before_fee": usdt_value_of_btc_sold,
            "fee_usdt": fee,
            "usdt_received_after_fee": usdt_received_after_fee,
            "exit_price": current_btc_price
        }

```
