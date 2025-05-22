import pytest
import pytest_asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock # For mocking super()._run_async_impl

from google_adk.agents import InvocationContext, RunConfig, Event, Part, Content
from google_adk.sessions import Session # Using real Session for simplicity in mock_ctx

from btc_usdt_trading_agent.account.trading_account import TradingAccount
from btc_usdt_trading_agent.agent import TradingAgent, TradingDecision
from btc_usdt_trading_agent.tools.binance_data_tool import BinanceDataTool

TEST_SYMBOL = "BTCUSDT"
FEE = TradingAccount.FEE_PERCENTAGE

@pytest.fixture
def trading_account():
    """Fixture for a TradingAccount instance with 1000 USDT initial balance."""
    return TradingAccount(initial_usdt_balance=1000.0)

@pytest.fixture
def trading_agent(trading_account: TradingAccount):
    """Fixture for a TradingAgent instance."""
    return TradingAgent(trading_account=trading_account)

@pytest.fixture
def mock_invocation_context(trading_agent: TradingAgent):
    """Fixture for a mock InvocationContext."""
    # Using real Session and RunConfig for simplicity as they are mostly data holders
    mock_session = Session(app_name="test_app", user_id="test_user", id="test_session")
    ctx = InvocationContext(
        invocation_id="test_inv_id",
        agent=trading_agent, # Agent needs to be set for some internal LlmAgent logic
        session=mock_session,
        run_config=RunConfig()
    )
    return ctx

async def get_agent_events(agent: TradingAgent, request: Any = None) -> list[Event]:
    """Helper to collect all events from an agent run."""
    events = []
    async for event in agent.run_async(request=request): # request can be None for default trigger
        events.append(event)
    return events

@pytest_asyncio.fixture
async def mock_binance_tool_methods(trading_agent: TradingAgent, monkeypatch):
    """Mocks methods of the BinanceDataTool instance within the agent."""
    mock_ticker = MagicMock(return_value={"symbol": TEST_SYMBOL, "price": "60000.00"})
    mock_klines = MagicMock(return_value={"symbol": TEST_SYMBOL, "klines": [{"open_time": "t1", "close": "c1"}]})
    
    monkeypatch.setattr(trading_agent.binance_data_tool, "get_ticker_price", mock_ticker)
    monkeypatch.setattr(trading_agent.binance_data_tool, "get_candlestick_data", mock_klines)
    return mock_ticker, mock_klines


@pytest.mark.asyncio
async def test_agent_buy_decision_successful(
    trading_agent: TradingAgent, 
    trading_account: TradingAccount, 
    mock_invocation_context: InvocationContext,
    monkeypatch,
    mock_binance_tool_methods # Ensure tools are mocked for pre-trade price fetch
):
    """Test agent's handling of a BUY decision from LLM."""
    initial_usdt = trading_account.usdt_balance
    buy_usdt_amount = 200.0
    btc_price_for_trade = 60000.0 # Price agent will use for execution
    
    # Mock the get_ticker_price call that happens *inside* _run_async_impl before trade
    mock_ticker_price_execution, _ = mock_binance_tool_methods
    mock_ticker_price_execution.return_value = {"symbol": TEST_SYMBOL, "price": str(btc_price_for_trade)}

    # Mock super()._run_async_impl to yield a BUY decision
    buy_decision = TradingDecision(
        action="BUY", 
        amount_usdt_to_spend=buy_usdt_amount, 
        amount_btc_to_sell=0.0, 
        reason="Test buy reason"
    )
    
    async def mock_super_run_impl(*args, **kwargs):
        # Simulate LLM making a decision (final event)
        yield Event(
            event_type="llm_response", 
            data=buy_decision.model_dump(), # LLM often returns dict
            response_type="llm_response", # Match agent's check
            response_data=buy_decision # Agent expects this to be TradingDecision instance
        )
    
    monkeypatch.setattr(TradingAgent, "_run_async_impl", mock_super_run_impl, raising=False)
    # Note: The above `raising=False` on `monkeypatch.setattr` is for the superclass method.
    # Actually, a better way to mock the LLM's final decision event within our overridden _run_async_impl
    # is to mock the `super()._run_async_impl` call *inside* our agent's `_run_async_impl`.
    # For this test, let's directly mock the agent's _run_llm_async which is simpler
    # if we only care about the final decision processing.
    # Or, even simpler, we assume the agent's _run_async_impl is structured to call a process_decision method.
    # Given the current structure of _run_async_impl, we need to mock the event stream it consumes.
    
    # Let's refine the mocking:
    # We want to test the logic *after* the LLM has provided a decision.
    # The TradingAgent._run_async_impl itself processes the stream from super()._run_async_impl
    # So, we mock what `super()._run_async_impl` would yield.
    
    mock_llm_event_stream = AsyncMock()
    mock_llm_event_stream.return_value.__aiter__.return_value = [
        Event(
            event_type="llm_response", 
            data=Content(parts=[Part(text=buy_decision.model_dump_json())]), # Simulate raw LLM output
            response_type="llm_response", # This is what LlmAgent yields
            response_data=buy_decision # This is what our agent's _run_async_impl expects after parsing
        )
    ]
    monkeypatch.setattr(trading_agent, "_run_llm_with_tools_async", mock_llm_event_stream)


    # Run the agent (this will now call the overridden _run_async_impl)
    events = await get_agent_events(trading_agent, request=mock_invocation_context) # Pass context as request

    expected_fee = buy_usdt_amount * FEE
    expected_btc_bought = buy_usdt_amount / btc_price_for_trade
    
    assert trading_account.usdt_balance == pytest.approx(initial_usdt - buy_usdt_amount - expected_fee)
    assert trading_account.btc_balance == pytest.approx(expected_btc_bought)
    assert len(trading_account.transaction_history) == 1
    assert trading_account.transaction_history[0]["type"] == "BUY"
    assert trading_account.transaction_history[0]["amount_crypto"] == pytest.approx(expected_btc_bought)

    summary_event = events[-1]
    assert summary_event.response_type == "agent_action_summary"
    assert summary_event.response_data["llm_decision"]["action"] == "BUY"
    assert summary_event.response_data["trade_executed"] is True
    assert "Buy order executed successfully" in summary_event.response_data["action_result"]


@pytest.mark.asyncio
async def test_agent_sell_decision_successful(
    trading_agent: TradingAgent, 
    trading_account: TradingAccount, 
    mock_invocation_context: InvocationContext,
    monkeypatch,
    mock_binance_tool_methods
):
    # Pre-populate account with BTC
    trading_account.btc_balance = 0.1
    trading_account.usdt_balance = 1000.0 # Reset USDT for clarity
    initial_btc = trading_account.btc_balance
    initial_usdt = trading_account.usdt_balance

    sell_btc_amount = 0.05
    btc_price_for_trade = 62000.0
    
    mock_ticker_price_execution, _ = mock_binance_tool_methods
    mock_ticker_price_execution.return_value = {"symbol": TEST_SYMBOL, "price": str(btc_price_for_trade)}

    sell_decision = TradingDecision(
        action="SELL", 
        amount_usdt_to_spend=0.0, 
        amount_btc_to_sell=sell_btc_amount, 
        reason="Test sell reason"
    )
    
    mock_llm_event_stream = AsyncMock()
    mock_llm_event_stream.return_value.__aiter__.return_value = [
        Event(event_type="llm_response", data=Content(parts=[Part(text=sell_decision.model_dump_json())]), response_type="llm_response", response_data=sell_decision)
    ]
    monkeypatch.setattr(trading_agent, "_run_llm_with_tools_async", mock_llm_event_stream)

    events = await get_agent_events(trading_agent, request=mock_invocation_context)

    expected_usdt_value = sell_btc_amount * btc_price_for_trade
    expected_fee = expected_usdt_value * FEE
    
    assert trading_account.btc_balance == pytest.approx(initial_btc - sell_btc_amount)
    assert trading_account.usdt_balance == pytest.approx(initial_usdt + expected_usdt_value - expected_fee)
    assert len(trading_account.transaction_history) == 1
    assert trading_account.transaction_history[0]["type"] == "SELL"

    summary_event = events[-1]
    assert summary_event.response_type == "agent_action_summary"
    assert summary_event.response_data["llm_decision"]["action"] == "SELL"
    assert summary_event.response_data["trade_executed"] is True
    assert "Sell order executed successfully" in summary_event.response_data["action_result"]


@pytest.mark.asyncio
async def test_agent_hold_decision(
    trading_agent: TradingAgent, 
    trading_account: TradingAccount, 
    mock_invocation_context: InvocationContext,
    monkeypatch
):
    initial_usdt = trading_account.usdt_balance
    initial_btc = trading_account.btc_balance

    hold_decision = TradingDecision(
        action="HOLD", 
        amount_usdt_to_spend=0.0, 
        amount_btc_to_sell=0.0, 
        reason="Market unclear"
    )

    mock_llm_event_stream = AsyncMock()
    mock_llm_event_stream.return_value.__aiter__.return_value = [
         Event(event_type="llm_response", data=Content(parts=[Part(text=hold_decision.model_dump_json())]), response_type="llm_response", response_data=hold_decision)
    ]
    monkeypatch.setattr(trading_agent, "_run_llm_with_tools_async", mock_llm_event_stream)

    events = await get_agent_events(trading_agent, request=mock_invocation_context)

    assert trading_account.usdt_balance == initial_usdt
    assert trading_account.btc_balance == initial_btc
    assert len(trading_account.transaction_history) == 0 # No trade transaction

    summary_event = events[-1]
    assert summary_event.response_type == "agent_action_summary"
    assert summary_event.response_data["llm_decision"]["action"] == "HOLD"
    assert summary_event.response_data["trade_executed"] is False
    assert "Holding position" in summary_event.response_data["action_result"]


@pytest.mark.asyncio
async def test_agent_llm_invalid_decision_format(
    trading_agent: TradingAgent, 
    mock_invocation_context: InvocationContext,
    monkeypatch
):
    # Simulate LLM returning something that's not a valid TradingDecision
    invalid_llm_output = {"action": "BUY", "amount_usdt_to_spend": "a lot", "reason": 123} # Invalid types

    mock_llm_event_stream = AsyncMock()
    # The LlmAgent would try to parse this into TradingDecision. If it fails,
    # our overridden _run_async_impl expects response_data to be None or not a TradingDecision.
    # The LlmAgent's own parsing (part of super()._run_async_impl) would handle this.
    # So we mock the event stream *after* LlmAgent's parsing attempt.
    # If LlmAgent yields an llm_response with response_data that's not TradingDecision, our agent handles it.
    
    # Let's simulate the LlmAgent yielding a response_data that is not a TradingDecision instance
    # due to pydantic validation error within the LlmAgent itself.
    # Our agent's _run_async_impl has a check: `if isinstance(event.response_data, TradingDecision):`
    # If the LlmAgent's schema validation fails, it might yield a different response_data or error.
    # For this test, we'll assume the LlmAgent successfully parses but our agent's logic for some reason
    # gets a raw dict that fails *its own* Pydantic conversion (if it were to re-parse).
    # More simply, we test the path where `llm_decision_response` is None after the loop in _run_async_impl.

    async def mock_super_run_impl_empty(*args, **kwargs):
        # Simulate LLM flow that doesn't yield a final usable TradingDecision object
        # For example, max_steps reached or LLM consistently fails to provide valid JSON.
        yield Event(event_type="llm_response", data=None, response_type="llm_response", response_data=None)
        # Or, yield Event(event_type="llm_response", data=invalid_llm_output, response_type="llm_response", response_data=invalid_llm_output)
        # and let the agent's isinstance check fail.
        
    # To test the specific part in _run_async_impl where it handles non-TradingDecision response_data:
    async def mock_super_run_impl_invalid_data(*args, **kwargs):
        yield Event(
            event_type="llm_response", 
            data=invalid_llm_output, # Raw output
            response_type="llm_response", 
            response_data=invalid_llm_output # Simulate LlmAgent passed raw dict
        )

    monkeypatch.setattr(TradingAgent, "_run_llm_with_tools_async", mock_super_run_impl_invalid_data)
    
    events = await get_agent_events(trading_agent, request=mock_invocation_context)

    error_event = events[-1] # Should be the agent_error from the agent's own logic
    assert error_event.response_type == "agent_error"
    assert "LLM output did not conform to TradingDecision schema" in error_event.response_data["error"]


@pytest.mark.asyncio
async def test_agent_trade_execution_price_fetch_fails(
    trading_agent: TradingAgent, 
    trading_account: TradingAccount, 
    mock_invocation_context: InvocationContext,
    monkeypatch,
    mock_binance_tool_methods # bring in the mocked tools
):
    initial_usdt = trading_account.usdt_balance
    initial_btc = trading_account.btc_balance

    buy_decision = TradingDecision(
        action="BUY", 
        amount_usdt_to_spend=200.0, 
        amount_btc_to_sell=0.0, 
        reason="Test buy reason, price fetch will fail"
    )

    # Mock the initial LLM decision successfully
    mock_llm_event_stream = AsyncMock()
    mock_llm_event_stream.return_value.__aiter__.return_value = [
        Event(event_type="llm_response", data=Content(parts=[Part(text=buy_decision.model_dump_json())]), response_type="llm_response", response_data=buy_decision)
    ]
    monkeypatch.setattr(trading_agent, "_run_llm_with_tools_async", mock_llm_event_stream)

    # Mock BinanceDataTool.get_ticker_price (called by agent before trade) to return an error
    mock_ticker_price_execution, _ = mock_binance_tool_methods
    mock_ticker_price_execution.return_value = {"error": "API is down"}

    events = await get_agent_events(trading_agent, request=mock_invocation_context)

    # Balances should not change
    assert trading_account.usdt_balance == initial_usdt
    assert trading_account.btc_balance == initial_btc
    assert len(trading_account.transaction_history) == 0

    summary_event = events[-1]
    assert summary_event.response_type == "agent_action_summary" # Still yields summary
    assert summary_event.response_data["llm_decision"]["action"] == "BUY"
    assert summary_event.response_data["trade_executed"] is False
    assert "Could not execute BUY: Failed to fetch current price - API is down" in summary_event.response_data["action_result"]

```
