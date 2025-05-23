import datetime
import uuid
from typing import Optional, List, Literal, Dict, Any # Added Dict, Any
from pydantic import BaseModel, Field
import logging 
import json # Added
import os   # Added

logger = logging.getLogger("TradingAgentRunner.TradingAccount")

class OpenPosition(BaseModel):
    """
    Represents an open trading position with details including stop-loss and take-profit levels.
    """
    position_id: str = Field(default_factory=lambda: f"pos_{uuid.uuid4().hex[:8]}")
    symbol: str
    amount_crypto: float 
    entry_price: float   
    timestamp: str       
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    status: Literal["OPEN", "CLOSED"] = "OPEN"
    closure_reason: Optional[str] = None 
    exit_price: Optional[float] = None
    closed_timestamp: Optional[str] = None

class TradingAccount:
    """
    Simulates a cryptocurrency trading account for managing USDT and BTC balances,
    recording transaction history, and tracking open positions with SL/TP.
    """
    FEE_PERCENTAGE = 0.001 
    DEFAULT_STOP_LOSS_PCT = 0.05  
    DEFAULT_TAKE_PROFIT_PCT = 0.10 

    def __init__(self, initial_usdt_balance: float = 1000.0):
        if initial_usdt_balance < 0:
            logger.error(f"Attempted to initialize TradingAccount with negative balance: {initial_usdt_balance}")
            raise ValueError("Initial USDT balance cannot be negative.")
        
        self.usdt_balance: float = initial_usdt_balance
        self.btc_balance: float = 0.0 
        self.transaction_history: list[dict] = []
        self.open_positions: List[OpenPosition] = []
        logger.debug(f"TradingAccount initialized with USDT: {initial_usdt_balance:.2f}, BTC: {self.btc_balance:.8f}")

    def _calculate_total_btc_in_open_positions(self) -> float:
        total_btc = sum(pos.amount_crypto for pos in self.open_positions if pos.status == "OPEN")
        logger.debug(f"Calculated total BTC in open positions: {total_btc:.8f}")
        return total_btc

    def get_balance(self) -> dict:
        self.btc_balance = self._calculate_total_btc_in_open_positions()
        balance_info = {"usdt_balance": self.usdt_balance, "btc_balance": self.btc_balance}
        logger.debug(f"Balance requested: {balance_info}")
        return balance_info

    def record_transaction(self, timestamp: str, type: str, symbol: str, 
                           amount_crypto: float, price: float, total_usdt: float, 
                           reason: str, fee: float = 0.0, position_id: Optional[str] = None) -> None:
        transaction = {
            "timestamp": timestamp, "type": type, "symbol": symbol,
            "amount_crypto": amount_crypto, "price": price,
            "total_usdt_value_before_fee": total_usdt, "reason": reason, 
            "fee_usdt": fee, "position_id": position_id
        }
        self.transaction_history.append(transaction)
        logger.debug(f"Transaction recorded: {type} {amount_crypto:.8f} {symbol} at {price:.2f}, PosID: {position_id}, Reason: {reason}, Fee: {fee:.4f}")

    def execute_buy_order(self, symbol: str, usdt_amount_to_spend: float, 
                          current_btc_price: float, reason: str, timestamp: str,
                          stop_loss_price: Optional[float] = None, 
                          take_profit_price: Optional[float] = None) -> dict:
        
        logger.info(f"Attempting BUY: {usdt_amount_to_spend=:.2f}, {current_btc_price=:.2f}, SL={stop_loss_price}, TP={take_profit_price}, Reason='{reason}'")

        if usdt_amount_to_spend <= 0:
            result = {"success": False, "message": "USDT amount to spend must be positive."}
            logger.warning(f"BUY failed: {result['message']}")
            return result
        if current_btc_price <= 0:
            result = {"success": False, "message": "BTC price must be positive."}
            logger.warning(f"BUY failed: {result['message']}")
            return result

        fee = usdt_amount_to_spend * self.FEE_PERCENTAGE
        total_usdt_deducted = usdt_amount_to_spend + fee

        if total_usdt_deducted > self.usdt_balance:
            result = {"success": False, "message": "Insufficient USDT balance to cover amount and fee."}
            logger.warning(f"BUY failed: {result['message']} (Needed: {total_usdt_deducted:.2f}, Available: {self.usdt_balance:.2f})")
            return result

        btc_to_buy = usdt_amount_to_spend / current_btc_price
        self.usdt_balance -= total_usdt_deducted
        
        actual_stop_loss = stop_loss_price if stop_loss_price is not None else current_btc_price * (1 - self.DEFAULT_STOP_LOSS_PCT)
        actual_take_profit = take_profit_price if take_profit_price is not None else current_btc_price * (1 + self.DEFAULT_TAKE_PROFIT_PCT)
        logger.debug(f"Calculated SL: {actual_stop_loss:.2f}, TP: {actual_take_profit:.2f} for buy at {current_btc_price:.2f}")

        new_position = OpenPosition(
            symbol=symbol, amount_crypto=btc_to_buy, entry_price=current_btc_price,
            timestamp=timestamp, stop_loss_price=actual_stop_loss, take_profit_price=actual_take_profit
        )
        self.open_positions.append(new_position)
        self.btc_balance = self._calculate_total_btc_in_open_positions() 

        self.record_transaction(
            timestamp=timestamp, type="BUY", symbol=symbol, amount_crypto=btc_to_buy, 
            price=current_btc_price, total_usdt=usdt_amount_to_spend, reason=reason, 
            fee=fee, position_id=new_position.position_id
        )
        
        logger.info(f"BUY successful for position {new_position.position_id}: Bought {btc_to_buy:.8f} BTC at {current_btc_price:.2f}. SL: {actual_stop_loss:.2f}, TP: {actual_take_profit:.2f}")
        return {
            "success": True, "message": "Buy order executed successfully, new position opened.", 
            "position_id": new_position.position_id, "btc_bought": btc_to_buy, 
            "usdt_spent_before_fee": usdt_amount_to_spend, "fee_usdt": fee,
            "total_usdt_deducted": total_usdt_deducted, "entry_price": current_btc_price,
            "stop_loss_price": actual_stop_loss, "take_profit_price": actual_take_profit
        }

    def execute_sell_order(self, symbol: str, btc_amount_to_sell: float, 
                           current_btc_price: float, reason: str, timestamp: str, 
                           position_id_to_close: Optional[str] = None) -> dict:
        
        log_pos_id = f"'{position_id_to_close}'" if position_id_to_close else "oldest available"
        logger.info(f"Attempting SELL for position {log_pos_id} or amount '{btc_amount_to_sell:.8f}'. Reason: {reason}, Price: {current_btc_price:.2f}")

        if current_btc_price <= 0:
            result = {"success": False, "message": "BTC price must be positive."}
            logger.warning(f"SELL failed: {result['message']}")
            return result

        position_to_close: Optional[OpenPosition] = None
        btc_actually_sold = 0.0

        if position_id_to_close:
            found_pos = next((p for p in self.open_positions if p.position_id == position_id_to_close and p.status == "OPEN"), None)
            if not found_pos:
                result = {"success": False, "message": f"Open position with ID '{position_id_to_close}' not found."}
                logger.warning(f"SELL failed: {result['message']}")
                return result
            position_to_close = found_pos
            btc_actually_sold = position_to_close.amount_crypto
        elif btc_amount_to_sell > 0:
            oldest_open_position = next((p for p in sorted(self.open_positions, key=lambda x: x.timestamp) if p.status == "OPEN"), None)
            if not oldest_open_position:
                result = {"success": False, "message": "No open positions to sell."}
                logger.warning(f"SELL failed: {result['message']}")
                return result
            position_to_close = oldest_open_position
            btc_actually_sold = position_to_close.amount_crypto
            if abs(btc_amount_to_sell - btc_actually_sold) > 1e-9: 
                result = {"success": False, "message": f"Manual sell for {btc_amount_to_sell:.8f} BTC does not match oldest open position amount ({btc_actually_sold:.8f} BTC). Only full position closure supported for manual sells without ID."}
                logger.warning(f"SELL failed: {result['message']}")
                return result
        else:
            result = {"success": False, "message": "BTC amount to sell must be positive or a position_id provided."}
            logger.warning(f"SELL failed: {result['message']}")
            return result

        if not position_to_close: 
            result = {"success": False, "message": "No valid position identified for selling."}
            logger.warning(f"SELL failed: {result['message']}")
            return result
        
        usdt_value_of_btc_sold = btc_actually_sold * current_btc_price
        fee = usdt_value_of_btc_sold * self.FEE_PERCENTAGE
        usdt_received_after_fee = usdt_value_of_btc_sold - fee
        
        position_to_close.status = "CLOSED"
        position_to_close.closure_reason = reason 
        position_to_close.exit_price = current_btc_price
        position_to_close.closed_timestamp = timestamp
        logger.debug(f"Position {position_to_close.position_id} marked as CLOSED. Reason: {reason}, Exit Price: {current_btc_price:.2f}")
        
        self.usdt_balance += usdt_received_after_fee
        self.btc_balance = self._calculate_total_btc_in_open_positions() 

        self.record_transaction(
            timestamp=timestamp, type="SELL", symbol=symbol, amount_crypto=btc_actually_sold, 
            price=current_btc_price, total_usdt=usdt_value_of_btc_sold, reason=reason, 
            fee=fee, position_id=position_to_close.position_id
        )
        
        logger.info(f"SELL successful for position {position_to_close.position_id}: Sold {btc_actually_sold:.8f} BTC at {current_btc_price:.2f}. Reason: {reason}")
        return {
            "success": True, "message": f"Sell order executed successfully. Position {position_to_close.position_id} closed.",
            "position_id_closed": position_to_close.position_id, "btc_sold": btc_actually_sold,
            "usdt_value_before_fee": usdt_value_of_btc_sold, "fee_usdt": fee,
            "usdt_received_after_fee": usdt_received_after_fee, "exit_price": current_btc_price
        }

    def save_state(self, filepath: str) -> bool:
        """Saves the current state of the trading account to a JSON file."""
        logger.debug(f"Attempting to save account state to {filepath}")
        state_data = {
            "usdt_balance": self.usdt_balance,
            "open_positions": [pos.model_dump() for pos in self.open_positions],
            "transaction_history": self.transaction_history
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=4)
            logger.info(f"TradingAccount state saved successfully to {filepath}")
            return True
        except IOError as e:
            logger.error(f"Failed to save TradingAccount state to {filepath} due to IOError: {e}", exc_info=True)
            return False
        except Exception as e: # Catch any other unexpected errors during saving
            logger.error(f"An unexpected error occurred while saving TradingAccount state to {filepath}: {e}", exc_info=True)
            return False

    def load_state(self, filepath: str) -> bool:
        """Loads the trading account state from a JSON file."""
        logger.debug(f"Attempting to load account state from {filepath}")
        if not os.path.exists(filepath):
            logger.warning(f"TradingAccount state file {filepath} not found. Initializing with current/default state.")
            return False
        
        try:
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            
            self.usdt_balance = loaded_data.get("usdt_balance", self.usdt_balance) 
            self.transaction_history = loaded_data.get("transaction_history", [])
            
            loaded_positions_data = loaded_data.get("open_positions", [])
            self.open_positions = [] 
            for pos_data in loaded_positions_data:
                try:
                    self.open_positions.append(OpenPosition(**pos_data))
                except Exception as e: 
                    logger.warning(f"Could not load an open position due to data error: {pos_data}. Error: {e}", exc_info=True)

            self.btc_balance = self._calculate_total_btc_in_open_positions() 

            logger.info(f"TradingAccount state successfully loaded from {filepath}. Balance: USDT {self.usdt_balance:.2f}, BTC {self.btc_balance:.8f}")
            logger.debug(f"Loaded {len(self.open_positions)} positions and {len(self.transaction_history)} transactions.")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load TradingAccount state from {filepath} due to JSON decoding error: {e}", exc_info=True)
            return False
        except IOError as e:
            logger.error(f"Failed to load TradingAccount state from {filepath} due to IOError: {e}", exc_info=True)
            return False
        except Exception as e: 
            logger.error(f"An unexpected error occurred while loading TradingAccount state from {filepath}: {e}", exc_info=True)
            return False
```
