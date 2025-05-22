import pytest
import pytest_asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch 

from google_adk.agents import InvocationContext, RunConfig, Event, Part, Content, FunctionCall
from google_adk.sessions import Session 

from btc_usdt_trading_agent.account.trading_account import TradingAccount
from btc_usdt_trading_agent.agent import TradingAgent, TradingDecision
from btc_usdt_trading_agent.tools.binance_data_tool import BinanceDataTool
from btc_usdt_trading_agent.tools import news_fetch_tool # To mock get_crypto_news

TEST_SYMBOL = "BTCUSDT"
FEE = TradingAccount.FEE_PERCENTAGE
AGENT_MODULE_PATH = "btc_usdt_trading_agent.agent" # For patching imported functions

@pytest.fixture
def trading_account():
    """Fixture for a TradingAccount instance with 1000 USDT initial balance."""
    return TradingAccount(initial_usdt_balance=1000.0)

@pytest.fixture
def trading_agent_instance(trading_account: TradingAccount): # Renamed to avoid conflict with class
    """Fixture for a TradingAgent instance."""
    return TradingAgent(trading_account=trading_account)

@pytest.fixture
def mock_invocation_context(trading_agent_instance: TradingAgent): # Updated to use renamed fixture
    """Fixture for a mock InvocationContext."""
    mock_session = Session(app_name="test_app", user_id="test_user", id="test_session")
    ctx = InvocationContext(
        invocation_id="test_inv_id",
        agent=trading_agent_instance, 
        session=mock_session,
        run_config=RunConfig()
    )
    return ctx

async def get_agent_events(agent: TradingAgent, request: Any = None) -> list[Event]:
    """Helper to collect all events from an agent run."""
    events = []
    # Ensure the request object has an instruction field if agent.instruction is not directly used by run_async
    # For LlmAgent, the instruction in the request can override self.instruction
    llm_request = request if isinstance(request, LlmRequest) else LlmRequest(instruction=agent.instruction)
    
    async for event in agent.run_async(request=llm_request): 
        events.append(event)
    return events

@pytest_asyncio.fixture
async def mock_binance_tool_methods_on_agent(trading_agent_instance: TradingAgent, monkeypatch):
    """Mocks methods of the BinanceDataTool instance within the agent."""
    mock_ticker = MagicMock(return_value={"symbol": TEST_SYMBOL, "price": "60000.00"})
    mock_klines = MagicMock(return_value={"symbol": TEST_SYMBOL, "klines": [{"open_time": "t1", "close": "c1"}]})
    
    monkeypatch.setattr(trading_agent_instance.binance_data_tool, "get_ticker_price", mock_ticker)
    monkeypatch.setattr(trading_agent_instance.binance_data_tool, "get_candlestick_data", mock_klines)
    return mock_ticker, mock_klines


def test_agent_can_be_configured_with_news_tool(trading_agent_instance: TradingAgent):
    """Verify that get_cryptocurrency_news is one of the agent's tools and instruction is updated."""
    assert len(trading_agent_instance.tools) == 3
    news_tool_registered = any(
        tool.name == "get_cryptocurrency_news" for tool in trading_agent_instance.tools
    )
    assert news_tool_registered, "News tool not found in agent's tools list."

    # Check if the function itself is correctly assigned to the tool
    news_function_tool = next(
        tool for tool in trading_agent_instance.tools if tool.name == "get_cryptocurrency_news"
    )
    assert news_function_tool.func == news_fetch_tool.get_crypto_news

    # Check the instruction template stored in the agent
    assert "get_cryptocurrency_news" in trading_agent_instance._original_instruction
    assert "news articles" in trading_agent_instance._original_instruction


@pytest.mark.asyncio
async def test_agent_llm_uses_news_tool_then_holds(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_invocation_context: InvocationContext, # Use the fixture
    monkeypatch,
    mock_binance_tool_methods_on_agent # Ensure other tools are also benignly mocked
):
    """
    Test a scenario where the LLM first calls the news tool, then makes a HOLD decision.
    This tests the multi-turn tool use flow.
    """
    initial_usdt = trading_account.usdt_balance
    initial_btc = trading_account.btc_balance
    news_query = "bitcoin positive sentiment"

    # 1. Mock the `get_crypto_news` function itself (which is wrapped by the FunctionTool)
    mock_news_results = {"query": news_query, "news_items": [{"title": "BTC News", "snippet": "BTC is great", "source_url": "http://news.com/btc"}]}
    # Patch the function in the module where the agent imports it from
    mocked_news_func_on_module = AsyncMock(return_value=mock_news_results)
    monkeypatch.setattr(f"{AGENT_MODULE_PATH}.get_crypto_news", mocked_news_func_on_module)
    
    # 2. Mock the LlmAgent's `_run_llm_with_tools_async` to simulate multi-turn conversation
    # This mock will control the sequence of events yielded by the LLM part of the agent.
    
    # First LLM response: request to call the news tool
    news_tool_call = FunctionCall(name="get_cryptocurrency_news", args={"query": news_query})
    llm_response_event_1_tool_call = Event(
        event_type="llm_response",
        data=Content(parts=[Part(function_call=news_tool_call)]),
        response_type="llm_response", # LlmAgent yields this after LLM call
        response_data=Content(parts=[Part(function_call=news_tool_call)]) # Parsed content
    )

    # Second LLM response (after tool result): final HOLD decision
    hold_decision = TradingDecision(
        action="HOLD", 
        amount_usdt_to_spend=0.0, 
        amount_btc_to_sell=0.0, 
        reason="Holding after reviewing positive Bitcoin news and stable market data."
    )
    llm_response_event_2_final_decision = Event(
        event_type="llm_response",
        data=Content(parts=[Part(text=hold_decision.model_dump_json())]), # LLM outputs JSON text
        response_type="llm_response",
        response_data=hold_decision # Agent's _run_async_impl expects this to be TradingDecision
    )

    # Configure the mock for _run_llm_with_tools_async
    mock_llm_flow = AsyncMock()
    # Simulate two calls to the LLM:
    # 1st call: LLM decides to use the news tool.
    # 2nd call (after tool result is processed by LlmAgent): LLM makes final HOLD decision.
    # The LlmAgent's `_run_async_impl` calls `_run_llm_with_tools_async` in a loop.
    # We need it to yield the tool call first, then the final decision.
    # The actual tool call processing is handled by LlmAgent's `_process_tool_calls_async`.
    
    # We are mocking `_run_llm_with_tools_async` which is called by `super()._run_async_impl`
    # The `super()._run_async_impl` itself has the loop for tool use.
    # So, mocking `super()._run_async_impl` to control the *entire* event stream is better.

    async def mock_super_run_flow(*args, **kwargs):
        # 1. LLM requests to call the news tool
        yield llm_response_event_1_tool_call
        
        # 2. Agent (LlmAgent part) processes tool call and yields FunctionCallEvent
        # This part is implicitly handled by the actual LlmAgent if we let it run.
        # For this test, we need to ensure our mocked get_crypto_news is called.
        # The LlmAgent will execute the tool.
        
        # 3. LlmAgent then yields the FunctionToolResultEvent
        # We need to create this event as if the tool (our mocked get_crypto_news) ran.
        tool_result_content = Content(parts=[Part(text=str(mock_news_results))]) # Tool output usually stringified
        yield Event(
            event_type="tool_code_execution_result", 
            data=tool_result_content, # Or directly the result dict
            response_type="tool_code_execution_result",
            response_data=mock_news_results # The actual result from the tool
        )
        
        # 4. LLM makes final decision after seeing tool results
        yield llm_response_event_2_final_decision

    # Patch the method within the superclass that LlmAgent's _run_async_impl calls.
    # This is tricky because _run_async_impl calls other internal methods.
    # Let's simplify: patch `_run_llm_with_tools_async` and assume it's called twice.
    mock_llm_tool_flow_mock = AsyncMock(side_effect=[
        # First call to LLM -> requests tool
        [llm_response_event_1_tool_call], 
        # Second call to LLM (after tool result) -> final decision
        [llm_response_event_2_final_decision]  
    ])
    # This needs to be an async generator.
    async def multi_turn_llm_mock(*args, **kwargs):
        yield llm_response_event_1_tool_call
        # LlmAgent processes this, calls the tool (our mocked_news_func_on_module)
        # Then LlmAgent calls LLM again with tool results.
        # The *next* call to _run_llm_with_tools_async should yield the final decision.
        # This still requires knowing how many times _run_llm_with_tools_async is called.
        
    # A more direct way to test the tool use is to ensure the FunctionTool's func is called.
    # The LlmAgent's `_process_tool_calls_async` would call our mocked_news_func_on_module.
    # The final LLM decision (HOLD) would be a separate mock.

    # Let's use the approach of mocking `super()._run_async_impl` to control the *full* event stream
    # that our agent's `_run_async_impl` consumes.
    @patch.object(LlmAgent, '_run_async_impl', new_callable=AsyncMock) # Patch on the LlmAgent class
    async def run_test_with_patched_super(mock_super_run_impl_method):
        mock_super_run_impl_method.return_value.__aiter__.return_value = mock_super_run_flow()
        
        # Prepare the initial request for the agent
        current_balances = trading_agent_instance.trading_account.get_balance()
        formatted_instruction = trading_agent_instance._original_instruction.format(
            usdt_balance=f"{current_balances['usdt_balance']:.2f}",
            btc_balance=f"{current_balances['btc_balance']:.8f}"
        )
        llm_request = LlmRequest(instruction=formatted_instruction)


        events = await get_agent_events(trading_agent_instance, request=llm_request)
        
        # Assertions
        mocked_news_func_on_module.assert_called_once_with(query=news_query, num_results=3) # Default num_results
        
        assert trading_account.usdt_balance == initial_usdt
        assert trading_account.btc_balance == initial_btc
        assert len(trading_account.transaction_history) == 0 

        # Check for the agent_action_summary
        summary_event = events[-1]
        assert summary_event.response_type == "agent_action_summary"
        assert summary_event.response_data["llm_decision"]["action"] == "HOLD"
        assert "Holding after reviewing positive Bitcoin news" in summary_event.response_data["llm_decision"]["reason"]
        assert summary_event.response_data["trade_executed"] is False
        
        # Check for earlier events if needed (optional here)
        # print(f"Events: {[e.event_type for e in events]}") # Debugging
        # Example: find the tool call event if LlmAgent yields it explicitly
        # function_call_event = next((e for e in events if e.event_type == "function_call" and e.data.name == "get_cryptocurrency_news"), None)
        # assert function_call_event is not None 
        # tool_result_event = next((e for e in events if e.event_type == "function_tool_result" and "BTC is great" in str(e.data)), None)
        # assert tool_result_event is not None

    await run_test_with_patched_super()


# Keep other tests from previous steps (BUY, SELL, HOLD direct, errors)
# These tests from previous subtask (test_agent.py for TradingAgent before news tool) are adapted:

@pytest.mark.asyncio
async def test_agent_buy_decision_successful_direct_llm(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_invocation_context: InvocationContext,
    monkeypatch,
    mock_binance_tool_methods_on_agent 
):
    initial_usdt = trading_account.usdt_balance
    buy_usdt_amount = 200.0
    btc_price_for_trade = 60000.0 
    
    mock_ticker_price_execution, _ = mock_binance_tool_methods_on_agent
    mock_ticker_price_execution.return_value = {"symbol": TEST_SYMBOL, "price": str(btc_price_for_trade)}

    buy_decision = TradingDecision(
        action="BUY", 
        amount_usdt_to_spend=buy_usdt_amount, 
        amount_btc_to_sell=0.0, 
        reason="Test buy reason"
    )
    
    async def mock_super_run_impl_direct_decision(*args, **kwargs):
        # Simulate LLM making a decision (final event after any tool use)
        yield Event(event_type="llm_response", data=Content(parts=[Part(text=buy_decision.model_dump_json())]), response_type="llm_response", response_data=buy_decision)

    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_direct_decision)


    current_balances = trading_agent_instance.trading_account.get_balance()
    formatted_instruction = trading_agent_instance._original_instruction.format(
        usdt_balance=f"{current_balances['usdt_balance']:.2f}",
        btc_balance=f"{current_balances['btc_balance']:.8f}"
    )
    llm_request = LlmRequest(instruction=formatted_instruction)
    events = await get_agent_events(trading_agent_instance, request=llm_request) 

    expected_fee = buy_usdt_amount * FEE
    expected_btc_bought = buy_usdt_amount / btc_price_for_trade
    
    assert trading_account.usdt_balance == pytest.approx(initial_usdt - buy_usdt_amount - expected_fee)
    assert trading_account.btc_balance == pytest.approx(expected_btc_bought)
    
    summary_event = events[-1]
    assert summary_event.response_type == "agent_action_summary"
    assert summary_event.response_data["llm_decision"]["action"] == "BUY"


@pytest.mark.asyncio
async def test_agent_sell_decision_successful_direct_llm(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_invocation_context: InvocationContext,
    monkeypatch,
    mock_binance_tool_methods_on_agent
):
    trading_account.btc_balance = 0.1
    trading_account.usdt_balance = 1000.0 
    initial_btc = trading_account.btc_balance
    initial_usdt = trading_account.usdt_balance
    sell_btc_amount = 0.05
    btc_price_for_trade = 62000.0
    
    mock_ticker_price_execution, _ = mock_binance_tool_methods_on_agent
    mock_ticker_price_execution.return_value = {"symbol": TEST_SYMBOL, "price": str(btc_price_for_trade)}

    sell_decision = TradingDecision(action="SELL", amount_usdt_to_spend=0.0, amount_btc_to_sell=sell_btc_amount, reason="Test sell reason")
    
    async def mock_super_run_impl_direct_decision(*args, **kwargs):
        yield Event(event_type="llm_response", data=Content(parts=[Part(text=sell_decision.model_dump_json())]), response_type="llm_response", response_data=sell_decision)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_direct_decision)

    current_balances = trading_agent_instance.trading_account.get_balance()
    formatted_instruction = trading_agent_instance._original_instruction.format(
        usdt_balance=f"{current_balances['usdt_balance']:.2f}",
        btc_balance=f"{current_balances['btc_balance']:.8f}"
    )
    llm_request = LlmRequest(instruction=formatted_instruction)
    events = await get_agent_events(trading_agent_instance, request=llm_request)

    expected_usdt_value = sell_btc_amount * btc_price_for_trade
    expected_fee = expected_usdt_value * FEE
    
    assert trading_account.btc_balance == pytest.approx(initial_btc - sell_btc_amount)
    assert trading_account.usdt_balance == pytest.approx(initial_usdt + expected_usdt_value - expected_fee)

    summary_event = events[-1]
    assert summary_event.response_type == "agent_action_summary"
    assert summary_event.response_data["llm_decision"]["action"] == "SELL"

@pytest.mark.asyncio
async def test_agent_hold_decision_direct_llm(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_invocation_context: InvocationContext,
    monkeypatch
):
    initial_usdt = trading_account.usdt_balance
    initial_btc = trading_account.btc_balance
    hold_decision = TradingDecision(action="HOLD", reason="Market unclear")

    async def mock_super_run_impl_direct_decision(*args, **kwargs):
        yield Event(event_type="llm_response", data=Content(parts=[Part(text=hold_decision.model_dump_json())]), response_type="llm_response", response_data=hold_decision)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_direct_decision)

    current_balances = trading_agent_instance.trading_account.get_balance()
    formatted_instruction = trading_agent_instance._original_instruction.format(
        usdt_balance=f"{current_balances['usdt_balance']:.2f}",
        btc_balance=f"{current_balances['btc_balance']:.8f}"
    )
    llm_request = LlmRequest(instruction=formatted_instruction)
    events = await get_agent_events(trading_agent_instance, request=llm_request)

    assert trading_account.usdt_balance == initial_usdt
    assert trading_account.btc_balance == initial_btc
    summary_event = events[-1]
    assert summary_event.response_type == "agent_action_summary"
    assert summary_event.response_data["llm_decision"]["action"] == "HOLD"

@pytest.mark.asyncio
async def test_agent_llm_invalid_decision_format_direct_llm(
    trading_agent_instance: TradingAgent, 
    mock_invocation_context: InvocationContext,
    monkeypatch
):
    invalid_llm_output_dict = {"action": "BUY", "amount_usdt_to_spend": "a lot", "reason": 123}

    async def mock_super_run_impl_invalid_data(*args, **kwargs):
        # Simulate LlmAgent yielding raw dict that fails Pydantic parsing in TradingAgent's _run_async_impl
        yield Event(event_type="llm_response", data=invalid_llm_output_dict, response_type="llm_response", response_data=invalid_llm_output_dict)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_invalid_data)
    
    current_balances = trading_agent_instance.trading_account.get_balance()
    formatted_instruction = trading_agent_instance._original_instruction.format(
        usdt_balance=f"{current_balances['usdt_balance']:.2f}",
        btc_balance=f"{current_balances['btc_balance']:.8f}"
    )
    llm_request = LlmRequest(instruction=formatted_instruction)
    events = await get_agent_events(trading_agent_instance, request=llm_request)

    error_event = events[-1] 
    assert error_event.response_type == "agent_error"
    assert "LLM output did not conform to TradingDecision schema" in error_event.response_data["error"]

@pytest.mark.asyncio
async def test_agent_trade_execution_price_fetch_fails_direct_llm(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_invocation_context: InvocationContext,
    monkeypatch,
    mock_binance_tool_methods_on_agent
):
    buy_decision = TradingDecision(action="BUY", amount_usdt_to_spend=200.0, reason="Test buy, price fetch will fail")
    
    async def mock_super_run_impl_direct_decision(*args, **kwargs):
         yield Event(event_type="llm_response", data=Content(parts=[Part(text=buy_decision.model_dump_json())]), response_type="llm_response", response_data=buy_decision)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_direct_decision)

    mock_ticker_price_execution, _ = mock_binance_tool_methods_on_agent
    mock_ticker_price_execution.return_value = {"error": "API is down"}

    current_balances = trading_agent_instance.trading_account.get_balance()
    formatted_instruction = trading_agent_instance._original_instruction.format(
        usdt_balance=f"{current_balances['usdt_balance']:.2f}",
        btc_balance=f"{current_balances['btc_balance']:.8f}"
    )
    llm_request = LlmRequest(instruction=formatted_instruction)
    events = await get_agent_events(trading_agent_instance, request=llm_request)

    summary_event = events[-1]
    assert summary_event.response_type == "agent_action_summary"
    assert summary_event.response_data["trade_executed"] is False
    assert "Could not execute BUY: Failed to fetch current price - API is down" in summary_event.response_data["action_result"]

```
