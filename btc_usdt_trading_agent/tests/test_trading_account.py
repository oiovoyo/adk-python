import pytest
import datetime
from btc_usdt_trading_agent.account.trading_account import TradingAccount

# Define a fixed timestamp for predictable tests
TEST_TIMESTAMP = datetime.datetime.now(datetime.timezone.utc).isoformat()
FEE = TradingAccount.FEE_PERCENTAGE

@pytest.fixture
def account():
    """Returns a TradingAccount instance with default initial balance."""
    return TradingAccount(initial_usdt_balance=1000.0)

@pytest.fixture
def account_for_btc_tests():
    """Returns a TradingAccount instance with some BTC for selling tests."""
    acc = TradingAccount(initial_usdt_balance=1000.0)
    # Pre-populate with some BTC without fees for simpler sell test setup
    acc.btc_balance = 1.0 
    return acc

def test_initialization_default():
    """Test account initialization with default USDT balance."""
    acc = TradingAccount()
    assert acc.usdt_balance == 1000.0
    assert acc.btc_balance == 0.0
    assert acc.transaction_history == []

def test_initialization_custom_balance():
    """Test account initialization with a custom USDT balance."""
    acc = TradingAccount(initial_usdt_balance=500.0)
    assert acc.usdt_balance == 500.0
    assert acc.btc_balance == 0.0

def test_initialization_negative_balance():
    """Test initialization with negative USDT balance raises ValueError."""
    with pytest.raises(ValueError, match="Initial USDT balance cannot be negative."):
        TradingAccount(initial_usdt_balance=-100.0)

def test_get_balance(account: TradingAccount):
    """Test get_balance returns correct balances."""
    account.usdt_balance = 500.0
    account.btc_balance = 0.5
    assert account.get_balance() == {"usdt_balance": 500.0, "btc_balance": 0.5}

# --- Test execute_buy_order ---
def test_execute_buy_order_success(account: TradingAccount):
    initial_usdt = account.usdt_balance
    usdt_to_spend = 100.0
    btc_price = 50000.0
    reason = "Test buy"
    
    expected_fee = usdt_to_spend * FEE
    expected_total_deducted = usdt_to_spend + expected_fee
    expected_btc_bought = usdt_to_spend / btc_price

    result = account.execute_buy_order("BTCUSDT", usdt_to_spend, btc_price, reason, TEST_TIMESTAMP)

    assert result["success"] is True
    assert result["message"] == "Buy order executed successfully."
    assert result["btc_bought"] == pytest.approx(expected_btc_bought)
    assert result["usdt_spent_before_fee"] == pytest.approx(usdt_to_spend)
    assert result["fee_usdt"] == pytest.approx(expected_fee)
    assert result["total_usdt_deducted"] == pytest.approx(expected_total_deducted)
    
    assert account.usdt_balance == pytest.approx(initial_usdt - expected_total_deducted)
    assert account.btc_balance == pytest.approx(expected_btc_bought)
    
    assert len(account.transaction_history) == 1
    tx = account.transaction_history[0]
    assert tx["type"] == "BUY"
    assert tx["symbol"] == "BTCUSDT"
    assert tx["amount_crypto"] == pytest.approx(expected_btc_bought)
    assert tx["price"] == pytest.approx(btc_price)
    assert tx["total_usdt_value_before_fee"] == pytest.approx(usdt_to_spend)
    assert tx["reason"] == reason
    assert tx["timestamp"] == TEST_TIMESTAMP
    assert tx["fee_usdt"] == pytest.approx(expected_fee)

def test_execute_buy_order_insufficient_funds(account: TradingAccount):
    initial_usdt = account.usdt_balance
    initial_btc = account.btc_balance
    # usdt_to_spend = initial_usdt + 100.0 # More than available (considering fee makes it even more)
    # Recalculate to ensure it's truly insufficient even if initial_usdt is small
    # If usdt_to_spend is just initial_usdt, fee will make it insufficient
    # If usdt_to_spend = initial_usdt / (1 + FEE) it would be exact.
    # So, usdt_to_spend = initial_usdt should fail if FEE > 0
    if FEE > 0:
         usdt_to_spend_for_fail = initial_usdt # This amount, after adding fee, will exceed balance
         if usdt_to_spend_for_fail * (1 + FEE) <= initial_usdt : # handle case where initial_usdt is very small
             usdt_to_spend_for_fail = initial_usdt + 1.0 # Ensure it's definitely more
    else: # if fee is zero, need to spend more than balance
         usdt_to_spend_for_fail = initial_usdt + 1.0


    btc_price = 50000.0
    
    result = account.execute_buy_order("BTCUSDT", usdt_to_spend_for_fail, btc_price, "Test buy fail", TEST_TIMESTAMP)

    assert result["success"] is False
    assert "Insufficient USDT balance" in result["message"]
    assert account.usdt_balance == initial_usdt # Balance unchanged
    assert account.btc_balance == initial_btc   # Balance unchanged
    assert len(account.transaction_history) == 0 # No transaction recorded

def test_execute_buy_order_zero_amount(account: TradingAccount):
    result = account.execute_buy_order("BTCUSDT", 0, 50000.0, "Test buy zero", TEST_TIMESTAMP)
    assert result["success"] is False
    assert result["message"] == "USDT amount to spend must be positive."
    assert len(account.transaction_history) == 0

def test_execute_buy_order_negative_amount(account: TradingAccount):
    result = account.execute_buy_order("BTCUSDT", -100, 50000.0, "Test buy negative", TEST_TIMESTAMP)
    assert result["success"] is False
    assert result["message"] == "USDT amount to spend must be positive."
    assert len(account.transaction_history) == 0

def test_execute_buy_order_zero_price(account: TradingAccount):
    result = account.execute_buy_order("BTCUSDT", 100, 0, "Test buy zero price", TEST_TIMESTAMP)
    assert result["success"] is False
    assert result["message"] == "BTC price must be positive."
    assert len(account.transaction_history) == 0

# --- Test execute_sell_order ---
def test_execute_sell_order_success(account_for_btc_tests: TradingAccount):
    account = account_for_btc_tests # Use pre-populated account
    initial_usdt = account.usdt_balance
    initial_btc = account.btc_balance # Should be 1.0
    
    btc_to_sell = 0.5
    btc_price = 52000.0
    reason = "Test sell"

    expected_usdt_value = btc_to_sell * btc_price
    expected_fee = expected_usdt_value * FEE
    expected_usdt_received = expected_usdt_value - expected_fee
    
    result = account.execute_sell_order("BTCUSDT", btc_to_sell, btc_price, reason, TEST_TIMESTAMP)

    assert result["success"] is True
    assert result["message"] == "Sell order executed successfully."
    assert result["btc_sold"] == pytest.approx(btc_to_sell)
    assert result["usdt_value_before_fee"] == pytest.approx(expected_usdt_value)
    assert result["fee_usdt"] == pytest.approx(expected_fee)
    assert result["usdt_received_after_fee"] == pytest.approx(expected_usdt_received)

    assert account.btc_balance == pytest.approx(initial_btc - btc_to_sell)
    assert account.usdt_balance == pytest.approx(initial_usdt + expected_usdt_received)
    
    assert len(account.transaction_history) == 1
    tx = account.transaction_history[0]
    assert tx["type"] == "SELL"
    assert tx["symbol"] == "BTCUSDT"
    assert tx["amount_crypto"] == pytest.approx(btc_to_sell)
    assert tx["price"] == pytest.approx(btc_price)
    assert tx["total_usdt_value_before_fee"] == pytest.approx(expected_usdt_value)
    assert tx["reason"] == reason
    assert tx["timestamp"] == TEST_TIMESTAMP
    assert tx["fee_usdt"] == pytest.approx(expected_fee)

def test_execute_sell_order_insufficient_btc(account: TradingAccount): # Uses default account with 0 BTC
    initial_usdt = account.usdt_balance
    initial_btc = account.btc_balance # Should be 0
    
    result = account.execute_sell_order("BTCUSDT", 0.1, 52000.0, "Test sell fail", TEST_TIMESTAMP)

    assert result["success"] is False
    assert result["message"] == "Insufficient BTC balance."
    assert account.usdt_balance == initial_usdt
    assert account.btc_balance == initial_btc
    assert len(account.transaction_history) == 0

def test_execute_sell_order_zero_amount(account_for_btc_tests: TradingAccount):
    result = account_for_btc_tests.execute_sell_order("BTCUSDT", 0, 52000.0, "Test sell zero", TEST_TIMESTAMP)
    assert result["success"] is False
    assert result["message"] == "BTC amount to sell must be positive."
    assert len(account_for_btc_tests.transaction_history) == 0

def test_execute_sell_order_negative_amount(account_for_btc_tests: TradingAccount):
    result = account_for_btc_tests.execute_sell_order("BTCUSDT", -0.1, 52000.0, "Test sell negative", TEST_TIMESTAMP)
    assert result["success"] is False
    assert result["message"] == "BTC amount to sell must be positive."
    assert len(account_for_btc_tests.transaction_history) == 0

def test_execute_sell_order_zero_price(account_for_btc_tests: TradingAccount):
    result = account_for_btc_tests.execute_sell_order("BTCUSDT", 0.1, 0, "Test sell zero price", TEST_TIMESTAMP)
    assert result["success"] is False
    assert result["message"] == "BTC price must be positive."
    assert len(account_for_btc_tests.transaction_history) == 0
    
# --- Test multiple transactions and fee accuracy ---
def test_multiple_transactions(account: TradingAccount):
    # 1. Buy BTC
    buy1_usdt = 200.0
    buy1_price = 50000.0
    buy1_fee = buy1_usdt * FEE
    buy1_total_deducted = buy1_usdt + buy1_fee
    buy1_btc_bought = buy1_usdt / buy1_price
    
    account.execute_buy_order("BTCUSDT", buy1_usdt, buy1_price, "Buy 1", TEST_TIMESTAMP + "_1")
    
    assert account.usdt_balance == pytest.approx(1000.0 - buy1_total_deducted)
    assert account.btc_balance == pytest.approx(buy1_btc_bought)
    
    # 2. Buy more BTC
    buy2_usdt = 300.0
    buy2_price = 51000.0 # Different price
    buy2_fee = buy2_usdt * FEE
    buy2_total_deducted = buy2_usdt + buy2_fee
    buy2_btc_bought = buy2_usdt / buy2_price
    
    current_usdt_before_buy2 = account.usdt_balance
    account.execute_buy_order("BTCUSDT", buy2_usdt, buy2_price, "Buy 2", TEST_TIMESTAMP + "_2")
    
    assert account.usdt_balance == pytest.approx(current_usdt_before_buy2 - buy2_total_deducted)
    assert account.btc_balance == pytest.approx(buy1_btc_bought + buy2_btc_bought)
    
    # 3. Sell some BTC
    sell1_btc = buy1_btc_bought / 2
    sell1_price = 52000.0
    sell1_usdt_value = sell1_btc * sell1_price
    sell1_fee = sell1_usdt_value * FEE
    sell1_usdt_received = sell1_usdt_value - sell1_fee
    
    current_usdt_before_sell1 = account.usdt_balance
    current_btc_before_sell1 = account.btc_balance
    account.execute_sell_order("BTCUSDT", sell1_btc, sell1_price, "Sell 1", TEST_TIMESTAMP + "_3")
    
    assert account.usdt_balance == pytest.approx(current_usdt_before_sell1 + sell1_usdt_received)
    assert account.btc_balance == pytest.approx(current_btc_before_sell1 - sell1_btc)
    
    assert len(account.transaction_history) == 3
    assert account.transaction_history[0]["type"] == "BUY"
    assert account.transaction_history[0]["reason"] == "Buy 1"
    assert account.transaction_history[1]["type"] == "BUY"
    assert account.transaction_history[1]["reason"] == "Buy 2"
    assert account.transaction_history[2]["type"] == "SELL"
    assert account.transaction_history[2]["reason"] == "Sell 1"

    # Check total fees paid (sum of fees from individual transactions)
    total_fees_recorded = sum(tx["fee_usdt"] for tx in account.transaction_history)
    assert total_fees_recorded == pytest.approx(buy1_fee + buy2_fee + sell1_fee)

def test_record_transaction_directly(account: TradingAccount):
    """Test direct call to record_transaction."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    account.record_transaction(
        timestamp=ts, type="DEPOSIT", symbol="USDT",
        amount_crypto=100.0, price=1.0, total_usdt=100.0, # price not relevant for deposit
        reason="Initial deposit check", fee=0.0
    )
    assert len(account.transaction_history) == 1
    tx = account.transaction_history[0]
    assert tx["type"] == "DEPOSIT"
    assert tx["reason"] == "Initial deposit check"
```
