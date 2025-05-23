import pytest
import datetime
import uuid # For checking position_id format, not generating it here.
from btc_usdt_trading_agent.account.trading_account import TradingAccount, OpenPosition

# Define a fixed timestamp for predictable tests
TEST_TIMESTAMP_STR = datetime.datetime.now(datetime.timezone.utc).isoformat()
FEE = TradingAccount.FEE_PERCENTAGE
DEFAULT_SL_PCT = TradingAccount.DEFAULT_STOP_LOSS_PCT
DEFAULT_TP_PCT = TradingAccount.DEFAULT_TAKE_PROFIT_PCT

@pytest.fixture
def account():
    """Returns a TradingAccount instance with default initial balance."""
    return TradingAccount(initial_usdt_balance=1000.0)

@pytest.fixture
def account_with_open_position(account: TradingAccount):
    """Returns an account with one open position."""
    buy_result = account.execute_buy_order(
        symbol="BTCUSDT", 
        usdt_amount_to_spend=500.0, 
        current_btc_price=50000.0, 
        reason="Initial position for testing sells", 
        timestamp=TEST_TIMESTAMP_STR
    )
    assert buy_result["success"]
    assert len(account.open_positions) == 1
    return account, buy_result["position_id"], buy_result["btc_bought"], buy_result["entry_price"]


def test_initialization_default(account: TradingAccount):
    assert account.usdt_balance == 1000.0
    assert account.btc_balance == 0.0 # Derived from empty open_positions
    assert account.open_positions == []
    assert account.transaction_history == []

def test_initialization_custom_balance():
    acc = TradingAccount(initial_usdt_balance=500.0)
    assert acc.usdt_balance == 500.0
    assert acc.btc_balance == 0.0
    assert acc.open_positions == []

def test_initialization_negative_balance():
    with pytest.raises(ValueError, match="Initial USDT balance cannot be negative."):
        TradingAccount(initial_usdt_balance=-100.0)

def test_get_balance_no_positions(account: TradingAccount):
    assert account.get_balance() == {"usdt_balance": 1000.0, "btc_balance": 0.0}

def test_execute_buy_order_success_with_sl_tp(account: TradingAccount):
    initial_usdt = account.usdt_balance
    usdt_to_spend = 100.0
    btc_price = 50000.0
    sl_price = 47000.0
    tp_price = 55000.0
    reason = "Test buy with SL/TP"
    
    result = account.execute_buy_order(
        "BTCUSDT", usdt_to_spend, btc_price, reason, TEST_TIMESTAMP_STR,
        stop_loss_price=sl_price, take_profit_price=tp_price
    )

    assert result["success"] is True
    assert result["message"] == "Buy order executed successfully, new position opened."
    assert "position_id" in result
    position_id = result["position_id"]
    assert isinstance(position_id, str)

    expected_fee = usdt_to_spend * FEE
    expected_total_deducted = usdt_to_spend + expected_fee
    expected_btc_bought = usdt_to_spend / btc_price

    assert result["btc_bought"] == pytest.approx(expected_btc_bought)
    assert result["entry_price"] == pytest.approx(btc_price)
    assert result["stop_loss_price"] == pytest.approx(sl_price)
    assert result["take_profit_price"] == pytest.approx(tp_price)
    
    assert account.usdt_balance == pytest.approx(initial_usdt - expected_total_deducted)
    assert account.get_balance()["btc_balance"] == pytest.approx(expected_btc_bought) # Check derived balance
    
    assert len(account.open_positions) == 1
    open_pos = account.open_positions[0]
    assert open_pos.position_id == position_id
    assert open_pos.symbol == "BTCUSDT"
    assert open_pos.amount_crypto == pytest.approx(expected_btc_bought)
    assert open_pos.entry_price == pytest.approx(btc_price)
    assert open_pos.stop_loss_price == pytest.approx(sl_price)
    assert open_pos.take_profit_price == pytest.approx(tp_price)
    assert open_pos.status == "OPEN"
    
    assert len(account.transaction_history) == 1
    tx = account.transaction_history[0]
    assert tx["type"] == "BUY"
    assert tx["position_id"] == position_id

def test_execute_buy_order_default_sl_tp_calculation(account: TradingAccount):
    usdt_to_spend = 100.0
    btc_price = 50000.0
    
    result = account.execute_buy_order(
        "BTCUSDT", usdt_to_spend, btc_price, "Test default SL/TP", TEST_TIMESTAMP_STR
    )
    assert result["success"]
    position_id = result["position_id"]
    
    expected_sl = btc_price * (1 - DEFAULT_SL_PCT)
    expected_tp = btc_price * (1 + DEFAULT_TP_PCT)
    
    assert result["stop_loss_price"] == pytest.approx(expected_sl)
    assert result["take_profit_price"] == pytest.approx(expected_tp)

    open_pos = next(p for p in account.open_positions if p.position_id == position_id)
    assert open_pos.stop_loss_price == pytest.approx(expected_sl)
    assert open_pos.take_profit_price == pytest.approx(expected_tp)

def test_execute_sell_order_specific_position_id(account_with_open_position):
    account, pos_id_to_sell, initial_pos_btc, entry_price = account_with_open_position
    initial_usdt = account.usdt_balance
    
    sell_price = 52000.0
    reason = "Closing specific position"
    
    result = account.execute_sell_order(
        "BTCUSDT", btc_amount_to_sell=0, # Amount ignored when position_id is provided
        current_btc_price=sell_price, 
        reason=reason, 
        timestamp=TEST_TIMESTAMP_STR + "_sell",
        position_id_to_close=pos_id_to_sell
    )

    assert result["success"] is True
    assert result["message"] == f"Sell order executed successfully. Position {pos_id_to_sell} closed."
    assert result["position_id_closed"] == pos_id_to_sell
    assert result["btc_sold"] == pytest.approx(initial_pos_btc)
    assert result["exit_price"] == pytest.approx(sell_price)

    expected_usdt_value = initial_pos_btc * sell_price
    expected_fee = expected_usdt_value * FEE
    expected_usdt_received = expected_usdt_value - expected_fee
    
    assert account.usdt_balance == pytest.approx(initial_usdt + expected_usdt_received)
    assert account.get_balance()["btc_balance"] == pytest.approx(0) # Only one position was open
    
    closed_pos = next(p for p in account.open_positions if p.position_id == pos_id_to_sell)
    assert closed_pos.status == "CLOSED"
    assert closed_pos.closure_reason == reason
    assert closed_pos.exit_price == pytest.approx(sell_price)
    assert closed_pos.closed_timestamp == TEST_TIMESTAMP_STR + "_sell"
    
    assert len(account.transaction_history) == 2 # Initial buy + this sell
    sell_tx = account.transaction_history[-1]
    assert sell_tx["type"] == "SELL"
    assert sell_tx["position_id"] == pos_id_to_sell
    assert sell_tx["reason"] == reason

def test_execute_sell_order_manual_oldest_position(account: TradingAccount):
    # Create two positions
    buy1_res = account.execute_buy_order("BTCUSDT", 100, 50000, "Buy 1", TEST_TIMESTAMP_STR + "_1")
    pos1_id = buy1_res["position_id"]
    pos1_btc = buy1_res["btc_bought"]
    
    # Make sure timestamps are different for "oldest"
    import time; time.sleep(0.001) 
    buy2_res = account.execute_buy_order("BTCUSDT", 100, 51000, "Buy 2", datetime.datetime.now(datetime.timezone.utc).isoformat())

    assert len(account.open_positions) == 2
    initial_total_btc = account.get_balance()["btc_balance"]

    # Sell oldest (pos1) by providing its exact amount
    sell_price = 52000.0
    result = account.execute_sell_order(
        "BTCUSDT", btc_amount_to_sell=pos1_btc, 
        current_btc_price=sell_price, 
        reason="Manual sell oldest", 
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    assert result["success"]
    assert result["position_id_closed"] == pos1_id
    
    pos1_obj = next(p for p in account.open_positions if p.position_id == pos1_id)
    assert pos1_obj.status == "CLOSED"
    assert account.get_balance()["btc_balance"] == pytest.approx(initial_total_btc - pos1_btc)

def test_execute_sell_order_manual_amount_mismatch(account_with_open_position):
    account, pos_id, pos_btc, _ = account_with_open_position
    result = account.execute_sell_order(
        "BTCUSDT", btc_amount_to_sell=pos_btc + 0.001, # Different amount
        current_btc_price=52000.0, 
        reason="Manual sell mismatch", 
        timestamp=TEST_TIMESTAMP_STR
    )
    assert not result["success"]
    assert "does not match oldest open position amount" in result["message"]

def test_execute_sell_order_non_existent_id(account: TradingAccount):
    result = account.execute_sell_order("BTCUSDT", 0, 50000, "Sell non-existent", TEST_TIMESTAMP_STR, "fake_id")
    assert not result["success"]
    assert "Open position with ID 'fake_id' not found" in result["message"]

def test_execute_sell_order_no_open_positions_manual(account: TradingAccount):
    assert len(account.open_positions) == 0
    result = account.execute_sell_order("BTCUSDT", 0.1, 50000, "Sell no open", TEST_TIMESTAMP_STR)
    assert not result["success"]
    assert "No open positions to sell" in result["message"]

def test_execute_sell_order_already_closed_position(account_with_open_position):
    account, pos_id, _, _ = account_with_open_position
    # Close it once
    account.execute_sell_order("BTCUSDT", 0, 52000, "First close", TEST_TIMESTAMP_STR, position_id_to_close=pos_id)
    
    # Attempt to close again
    result = account.execute_sell_order("BTCUSDT", 0, 52000, "Second close attempt", TEST_TIMESTAMP_STR + "_2", position_id_to_close=pos_id)
    assert not result["success"]
    assert f"Open position with ID '{pos_id}' not found" in result["message"] # Because it looks for "OPEN" status

def test_get_balance_consistency_after_trades(account: TradingAccount):
    res1 = account.execute_buy_order("BTCUSDT", 100, 50000, "b1", TEST_TIMESTAMP_STR + "_1")
    btc1 = res1["btc_bought"]
    pos1_id = res1["position_id"]
    
    res2 = account.execute_buy_order("BTCUSDT", 150, 51000, "b2", TEST_TIMESTAMP_STR + "_2")
    btc2 = res2["btc_bought"]

    assert account.get_balance()["btc_balance"] == pytest.approx(btc1 + btc2)
    
    account.execute_sell_order("BTCUSDT", 0, 52000, "s1", TEST_TIMESTAMP_STR + "_3", position_id_to_close=pos1_id)
    assert account.get_balance()["btc_balance"] == pytest.approx(btc2)

def test_execute_buy_order_insufficient_funds(account: TradingAccount):
    initial_usdt = account.usdt_balance
    usdt_to_spend = initial_usdt + 1.0 # Ensure it's more than balance, even after fee
    
    result = account.execute_buy_order("BTCUSDT", usdt_to_spend, 50000.0, "Test buy fail", TEST_TIMESTAMP_STR)
    assert not result["success"]
    assert "Insufficient USDT balance" in result["message"]
    assert len(account.open_positions) == 0
    assert account.usdt_balance == initial_usdt

# Original insufficient BTC test is implicitly covered by "no open positions" or "non-existent ID"
# If we want to test "trying to sell more BTC than available via a specific position ID"
# that's not possible with current logic as it sells the full position amount.
# The `btc_amount_to_sell` is only used for the ambiguous manual sell case.

```
