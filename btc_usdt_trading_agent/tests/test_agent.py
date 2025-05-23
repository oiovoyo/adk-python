import pytest
import pytest_asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch 

# Corrected ADK Imports
from google.adk.agents import LlmAgent 
from google.adk.models import LlmRequest 
from google.adk.events import Event 
from google.adk.runners import InvocationContext, RunConfig 
from google.adk.sessions import Session 
from google.genai.types import Content, Part, FunctionCall 


from btc_usdt_trading_agent.account.trading_account import TradingAccount
# Import local AgentEventContent and TradingDecision from agent.py
from btc_usdt_trading_agent.agent import TradingAgent, TradingDecision, AgentEventContent 
from btc_usdt_trading_agent.tools.binance_data_tool import BinanceDataTool
from btc_usdt_trading_agent.tools import news_fetch_tool

TEST_SYMBOL = "BTCUSDT"
FEE = TradingAccount.FEE_PERCENTAGE
AGENT_MODULE_PATH = "btc_usdt_trading_agent.agent" 

@pytest.fixture
def trading_account():
    return TradingAccount(initial_usdt_balance=1000.0)

@pytest.fixture
def trading_agent_instance(trading_account: TradingAccount):
    return TradingAgent(trading_account=trading_account)

@pytest.fixture
def mock_llm_request(trading_agent_instance: TradingAgent):
    return LlmRequest(instruction=trading_agent_instance._original_instruction)


async def get_agent_events(agent: TradingAgent, request: LlmRequest) -> list[Event]:
    events = []
    async for event in agent.run_async(request=request): 
        events.append(event)
    return events

@pytest_asyncio.fixture
async def mock_binance_tool_methods_on_agent(trading_agent_instance: TradingAgent, monkeypatch):
    mock_ticker = MagicMock(return_value={"symbol": TEST_SYMBOL, "price": "60000.00"})
    mock_klines = MagicMock(return_value={"symbol": TEST_SYMBOL, "klines": [{"open_time": "t1", "close": "c1"}]})
    
    monkeypatch.setattr(trading_agent_instance.binance_data_tool, "get_ticker_price", mock_ticker)
    monkeypatch.setattr(trading_agent_instance.binance_data_tool, "get_candlestick_data", mock_klines)
    return mock_ticker, mock_klines


def test_agent_tool_configuration(trading_agent_instance: TradingAgent):
    """Verify all five tools are correctly configured and instruction is updated."""
    assert len(trading_agent_instance.tools) == 5
    
    expected_tools = {
        "get_ticker_price": trading_agent_instance.binance_data_tool.get_ticker_price,
        "get_candlestick_data": trading_agent_instance.binance_data_tool.get_candlestick_data,
        "get_cryptocurrency_news": news_fetch_tool.get_crypto_news,
        "get_current_account_position": trading_agent_instance._get_current_account_position,
        "get_recent_trade_history": trading_agent_instance._get_recent_trade_history,
    }

    for tool_name, tool_func in expected_tools.items():
        registered_tool = next((t for t in trading_agent_instance.tools if t.name == tool_name), None)
        assert registered_tool is not None, f"{tool_name} not found in agent's tools list."
        assert registered_tool.func == tool_func, f"Function for {tool_name} does not match."

    instruction = trading_agent_instance._original_instruction
    assert "get_current_account_position()" in instruction
    assert "get_recent_trade_history(num_trades: int = 5)" in instruction
    assert "get_cryptocurrency_news" in instruction
    assert "get_ticker_price" in instruction
    assert "get_candlestick_data" in instruction
    assert "{usdt_balance}" not in instruction 
    assert "{btc_balance}" not in instruction  
    assert "Before making any trading decision, you should first understand your current financial position" in instruction


@pytest.mark.asyncio
async def test_agent_llm_uses_account_tools_then_buys(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_llm_request: LlmRequest, 
    monkeypatch,
    mock_binance_tool_methods_on_agent 
):
    initial_usdt = trading_account.usdt_balance
    initial_btc = trading_account.btc_balance
    num_trades_history_req = 2
    buy_usdt_amount = 150.0
    btc_price_for_trade = 61000.0

    mock_ticker_exec, _ = mock_binance_tool_methods_on_agent
    mock_ticker_exec.return_value = {"symbol": TEST_SYMBOL, "price": str(btc_price_for_trade)}
    
    account_pos_result = trading_agent_instance._get_current_account_position() 
    trading_account.record_transaction(datetime.datetime.now(datetime.timezone.utc).isoformat(), "BUY", "BTCUSDT", 0.001, 50000, 50, "dummy tx for history", 0.05)
    history_result = trading_agent_instance._get_recent_trade_history(num_trades=num_trades_history_req)
    
    pos_tool_call = FunctionCall(name="get_current_account_position", args={})
    llm_event_1_pos_call_content = Content(parts=[Part(function_call=pos_tool_call)])
    
    hist_tool_call = FunctionCall(name="get_recent_trade_history", args={"num_trades": num_trades_history_req})
    llm_event_2_hist_call_content = Content(parts=[Part(function_call=hist_tool_call)])

    buy_decision = TradingDecision(
        action="BUY", amount_usdt_to_spend=buy_usdt_amount, 
        reason="Analyzed account (has USDT) and history (no recent buys), market looks good.",
        suggested_stop_loss_percentage=0.05, suggested_take_profit_percentage=0.10
    )
    llm_event_3_final_decision_content = buy_decision 

    async def mock_super_run_flow(*args, **kwargs):
        yield Event(event_type="llm_response", content=llm_event_1_pos_call_content, author="model")
        yield Event(event_type="tool_code_execution_result", content=Content(parts=[Part(text=str(account_pos_result))]), author="tool")
        yield Event(event_type="llm_response", content=llm_event_2_hist_call_content, author="model")
        yield Event(event_type="tool_code_execution_result", content=Content(parts=[Part(text=str(history_result))]), author="tool")
        yield Event(event_type="llm_response", content=llm_event_3_final_decision_content, author="model", is_final_response=True)

    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_flow)
    
    events = await get_agent_events(trading_agent_instance, request=mock_llm_request)
    
    expected_fee = buy_usdt_amount * FEE
    expected_btc_bought = buy_usdt_amount / btc_price_for_trade
    
    assert trading_account.usdt_balance == pytest.approx(initial_usdt - buy_usdt_amount - expected_fee)
    assert trading_account.btc_balance == pytest.approx(initial_btc + expected_btc_bought) 
    assert len(trading_account.transaction_history) == 2 

    summary_event = events[-1] 
    assert isinstance(summary_event.content, AgentEventContent) # Check it's our local AgentEventContent
    assert summary_event.content.response_type == "agent_action_summary"
    summary_data = summary_event.content.data
    assert summary_data["llm_decision"]["action"] == "BUY"
    assert summary_data["trade_executed"] is True

@pytest.mark.asyncio
async def test_agent_direct_llm_decision_flow(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_llm_request: LlmRequest,
    monkeypatch,
    mock_binance_tool_methods_on_agent 
):
    buy_usdt_amount = 100.0
    btc_price_for_trade = 60000.0
    mock_ticker_exec, _ = mock_binance_tool_methods_on_agent
    mock_ticker_exec.return_value = {"symbol": TEST_SYMBOL, "price": str(btc_price_for_trade)}

    buy_decision = TradingDecision(action="BUY", amount_usdt_to_spend=buy_usdt_amount, reason="Direct buy")
    
    async def mock_super_run_impl_direct_decision(*args, **kwargs):
        yield Event(event_type="llm_response", content=buy_decision, author="model", is_final_response=True)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_direct_decision)

    events = await get_agent_events(trading_agent_instance, request=mock_llm_request) 

    summary_event = events[-1]
    assert isinstance(summary_event.content, AgentEventContent)
    assert summary_event.content.response_type == "agent_action_summary"
    assert summary_event.content.data["llm_decision"]["action"] == "BUY"
    assert summary_event.content.data["trade_executed"] is True


@pytest.mark.asyncio
async def test_agent_rm_trade_event_structure(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_llm_request: LlmRequest,
    monkeypatch,
    mock_binance_tool_methods_on_agent
):
    buy_res = trading_account.execute_buy_order("BTCUSDT", 200, 60000, "test_sl", "ts1", stop_loss_price=59000)
    
    mock_ticker_exec, _ = mock_binance_tool_methods_on_agent
    mock_ticker_exec.return_value = {"symbol": TEST_SYMBOL, "price": "58000.00"} 

    hold_decision = TradingDecision(action="HOLD", reason="Waiting for market clarity")
    async def mock_super_run_impl_hold(*args, **kwargs):
        yield Event(event_type="llm_response", content=hold_decision, author="model", is_final_response=True)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_hold)

    events = await get_agent_events(trading_agent_instance, request=mock_llm_request)
    
    rm_event_found = False
    for event in events:
        if isinstance(event.content, AgentEventContent) and event.content.response_type == "risk_management_trade_executed":
            rm_event_found = True
            assert "message" in event.content.data
            assert "trade_details" in event.content.data
            assert event.content.data["trade_details"]["success"] is True
            break
    assert rm_event_found, "Risk management trade executed event not found or not in correct format."

# Test for SELL decision (adapted)
@pytest.mark.asyncio
async def test_agent_sell_decision_successful_direct_llm(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_llm_request: LlmRequest,
    monkeypatch,
    mock_binance_tool_methods_on_agent
):
    trading_account.execute_buy_order("BTCUSDT", 500, 50000, "initial buy for sell test", "ts_init_sell") # Ensure BTC
    trading_account.btc_balance = trading_account._calculate_total_btc_in_open_positions() # Update manually for fixture state
    initial_btc = trading_account.btc_balance
    initial_usdt = trading_account.usdt_balance
    
    sell_btc_amount = initial_btc / 2 
    if sell_btc_amount == 0 and initial_btc > 0 : sell_btc_amount = initial_btc # ensure selling something if available
    assert sell_btc_amount > 0, "Test setup error: No BTC to sell"

    btc_price_for_trade = 62000.0
    
    mock_ticker_price_execution, _ = mock_binance_tool_methods_on_agent
    mock_ticker_price_execution.return_value = {"symbol": TEST_SYMBOL, "price": str(btc_price_for_trade)}

    sell_decision = TradingDecision(action="SELL", amount_btc_to_sell=sell_btc_amount, reason="Test sell direct")
    
    async def mock_super_run_impl_direct_decision(*args, **kwargs):
        yield Event(event_type="llm_response", content=sell_decision, author="model", is_final_response=True)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_direct_decision)

    events = await get_agent_events(trading_agent_instance, request=mock_llm_request)

    summary_event = events[-1]
    assert isinstance(summary_event.content, AgentEventContent)
    assert summary_event.content.response_type == "agent_action_summary"
    assert summary_event.content.data["llm_decision"]["action"] == "SELL"
    assert summary_event.content.data["trade_executed"] is True
    assert trading_account.btc_balance < initial_btc


# Test for HOLD decision (adapted)
@pytest.mark.asyncio
async def test_agent_hold_decision_direct_llm(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_llm_request: LlmRequest,
    monkeypatch
):
    initial_usdt = trading_account.usdt_balance
    initial_btc = trading_account.btc_balance
    hold_decision = TradingDecision(action="HOLD", reason="Market unclear direct")

    async def mock_super_run_impl_direct_decision(*args, **kwargs):
        yield Event(event_type="llm_response", content=hold_decision, author="model", is_final_response=True)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_direct_decision)

    events = await get_agent_events(trading_agent_instance, request=mock_llm_request)

    assert trading_account.usdt_balance == initial_usdt
    assert trading_account.btc_balance == initial_btc
    summary_event = events[-1]
    assert isinstance(summary_event.content, AgentEventContent)
    assert summary_event.content.response_type == "agent_action_summary"
    assert summary_event.content.data["llm_decision"]["action"] == "HOLD"

# Test for invalid LLM decision format (adapted)
@pytest.mark.asyncio
async def test_agent_llm_invalid_decision_format_direct_llm(
    trading_agent_instance: TradingAgent, 
    mock_llm_request: LlmRequest,
    monkeypatch
):
    invalid_llm_output_content = Content(parts=[Part(text='{"action": "BUY", "amount_usdt_to_spend": "a lot", "reason": 123}')]) # Not TradingDecision

    async def mock_super_run_impl_invalid_data(*args, **kwargs):
        # Simulate LlmAgent yielding raw content that is not a TradingDecision instance
        yield Event(event_type="llm_response", content=invalid_llm_output_content, author="model", is_final_response=True)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_invalid_data)
    
    events = await get_agent_events(trading_agent_instance, request=mock_llm_request)

    # The LlmAgent itself should handle this if output_schema is used.
    # If it results in an error within _run_async_impl, it should be caught by the agent.
    # Our agent yields AgentEventContent(response_type="agent_error", ...)
    error_event_found = False
    for event in events:
        if isinstance(event.content, AgentEventContent) and event.content.response_type == "agent_error":
            error_event_found = True
            assert "LLM output did not conform to TradingDecision schema" in event.content.data["error"] or \
                   "LLM did not produce a final TradingDecision" in event.content.data["error"]
            break
    assert error_event_found, "Agent error for invalid decision format not found or not in correct format."


# Test for trade execution price fetch failure (adapted)
@pytest.mark.asyncio
async def test_agent_trade_execution_price_fetch_fails_direct_llm(
    trading_agent_instance: TradingAgent, 
    trading_account: TradingAccount, 
    mock_llm_request: LlmRequest,
    monkeypatch,
    mock_binance_tool_methods_on_agent
):
    buy_decision = TradingDecision(action="BUY", amount_usdt_to_spend=200.0, reason="Test buy, price fetch will fail")
    
    async def mock_super_run_impl_direct_decision(*args, **kwargs):
         yield Event(event_type="llm_response", content=buy_decision, author="model", is_final_response=True)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_direct_decision)

    mock_ticker_price_execution, _ = mock_binance_tool_methods_on_agent
    mock_ticker_price_execution.return_value = {"error": "API is down for trade exec"}

    events = await get_agent_events(trading_agent_instance, request=mock_llm_request)

    summary_event = events[-1]
    assert isinstance(summary_event.content, AgentEventContent)
    assert summary_event.content.response_type == "agent_action_summary"
    assert summary_event.content.data["trade_executed"] is False
    assert "Could not execute BUY: Failed to fetch current price for trade - API is down for trade exec" in summary_event.content.data["action_result"]
