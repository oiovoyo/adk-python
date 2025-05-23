import pytest
import pytest_asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch 

# ADK Imports
from google_adk.agents import LlmAgent # Updated
from google_adk.models import LlmRequest # Updated
from google_adk.events import Event # Updated
from google_adk.runners import InvocationContext, RunConfig # InvocationContext might not be needed if LlmRequest is used
from google_adk.sessions import Session 
from google.genai.types import Content, Part, FunctionCall # Updated

# Project Imports
from btc_usdt_trading_agent.account.trading_account import TradingAccount
# Import local BaseAgentResponse and TradingDecision from agent.py
from btc_usdt_trading_agent.agent import TradingAgent, TradingDecision, BaseAgentResponse 
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

# mock_invocation_context might not be needed if we directly create LlmRequest for agent.run_async
@pytest.fixture
def mock_llm_request(trading_agent_instance: TradingAgent):
    # The instruction will be formatted specifically within each test
    # to reflect the dynamic nature (or static nature after recent changes)
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
    
    # Actual results from the agent's own methods (which access the real trading_account)
    account_pos_result = trading_agent_instance._get_current_account_position() 
    trading_account.record_transaction(datetime.datetime.now(datetime.timezone.utc).isoformat(), "BUY", "BTCUSDT", 0.001, 50000, 50, "dummy tx for history", 0.05)
    history_result = trading_agent_instance._get_recent_trade_history(num_trades=num_trades_history_req)
    
    # LLM sequence
    pos_tool_call = FunctionCall(name="get_current_account_position", args={})
    llm_event_1_pos_call_content = Content(parts=[Part(function_call=pos_tool_call)])
    
    hist_tool_call = FunctionCall(name="get_recent_trade_history", args={"num_trades": num_trades_history_req})
    llm_event_2_hist_call_content = Content(parts=[Part(function_call=hist_tool_call)])

    buy_decision = TradingDecision(
        action="BUY", amount_usdt_to_spend=buy_usdt_amount, 
        reason="Analyzed account (has USDT) and history (no recent buys), market looks good.",
        suggested_stop_loss_percentage=0.05, suggested_take_profit_percentage=0.10
    )
    # The LlmAgent with output_schema=TradingDecision will directly place TradingDecision in event.content
    llm_event_3_final_decision_content = buy_decision 

    async def mock_super_run_flow(*args, **kwargs):
        # 1. LLM requests get_current_account_position
        yield Event(event_type="llm_response", content=llm_event_1_pos_call_content, author="model")
        # 2. LlmAgent executes it, yields tool_code_execution_result
        yield Event(event_type="tool_code_execution_result", content=Content(parts=[Part(text=str(account_pos_result))]), author="tool")
        # 3. LLM requests get_recent_trade_history
        yield Event(event_type="llm_response", content=llm_event_2_hist_call_content, author="model")
        # 4. LlmAgent executes it, yields tool_code_execution_result
        yield Event(event_type="tool_code_execution_result", content=Content(parts=[Part(text=str(history_result))]), author="tool")
        # 5. LLM makes final BUY decision
        yield Event(event_type="llm_response", content=llm_event_3_final_decision_content, author="model", is_final_response=True)

    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_flow)
    
    events = await get_agent_events(trading_agent_instance, request=mock_llm_request)
    
    expected_fee = buy_usdt_amount * FEE
    expected_btc_bought = buy_usdt_amount / btc_price_for_trade
    
    assert trading_account.usdt_balance == pytest.approx(initial_usdt - buy_usdt_amount - expected_fee)
    assert trading_account.btc_balance == pytest.approx(initial_btc + expected_btc_bought)
    assert len(trading_account.transaction_history) == 2 

    summary_event = events[-1] # Last event should be agent_action_summary
    assert isinstance(summary_event.content, BaseAgentResponse) # Check it's our local BaseAgentResponse
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
    """Tests agent processing when LLM makes a direct decision without prior tool calls in this cycle."""
    buy_usdt_amount = 100.0
    btc_price_for_trade = 60000.0
    mock_ticker_exec, _ = mock_binance_tool_methods_on_agent
    mock_ticker_exec.return_value = {"symbol": TEST_SYMBOL, "price": str(btc_price_for_trade)}

    buy_decision = TradingDecision(action="BUY", amount_usdt_to_spend=buy_usdt_amount, reason="Direct buy")
    
    async def mock_super_run_impl_direct_decision(*args, **kwargs):
        # Simulate LlmAgent yielding the TradingDecision directly as event.content
        yield Event(event_type="llm_response", content=buy_decision, author="model", is_final_response=True)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_direct_decision)

    events = await get_agent_events(trading_agent_instance, request=mock_llm_request) 

    summary_event = events[-1]
    assert isinstance(summary_event.content, BaseAgentResponse)
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
    """Tests that risk_management_trade_executed events use the local BaseAgentResponse."""
    # Setup an open position that will be closed by SL
    buy_res = trading_account.execute_buy_order("BTCUSDT", 200, 60000, "test_sl", "ts1", stop_loss_price=59000)
    
    # Mock price to trigger SL
    mock_ticker_exec, _ = mock_binance_tool_methods_on_agent
    mock_ticker_exec.return_value = {"symbol": TEST_SYMBOL, "price": "58000.00"} # Triggers SL

    # Mock LLM to HOLD so only RM trade occurs
    hold_decision = TradingDecision(action="HOLD", reason="Waiting for market clarity")
    async def mock_super_run_impl_hold(*args, **kwargs):
        yield Event(event_type="llm_response", content=hold_decision, author="model", is_final_response=True)
    monkeypatch.setattr(LlmAgent, "_run_async_impl", mock_super_run_impl_hold)

    events = await get_agent_events(trading_agent_instance, request=mock_llm_request)
    
    rm_event_found = False
    for event in events:
        if isinstance(event.content, BaseAgentResponse) and event.content.response_type == "risk_management_trade_executed":
            rm_event_found = True
            assert "message" in event.content.data
            assert "trade_details" in event.content.data
            assert event.content.data["trade_details"]["success"] is True
            break
    assert rm_event_found, "Risk management trade executed event not found or not in correct format."

# Other tests (sell, hold, errors) would be adapted similarly to use the new LlmRequest and
# ensure that when they mock `LlmAgent._run_async_impl` to yield a final decision,
# that decision is directly in event.content as a TradingDecision instance.
# And when asserting on agent's custom yielded events, check for local BaseAgentResponse.
```
