import pytest
import pytest_asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

# Corrected ADK Imports
from google.adk.events import Event
from google.genai.types import Content, Part # For constructing mock events if needed

# Module to be tested
from btc_usdt_trading_agent import main_runner 
# For patching classes instantiated within main_runner.main and importing local AgentEventContent
from btc_usdt_trading_agent.agent import AgentEventContent, TradingDecision 
# Note: LlmRequest, RunConfig, Session, InMemoryRunner are used by main_runner,
# so their paths should be correct in main_runner.py. Here we mock instances.

@pytest.fixture(autouse=True)
def ensure_logging_setup_for_tests(caplog):
    """Ensure that logging is set up for each test to capture messages."""
    main_runner.setup_logging(log_level=logging.DEBUG, log_file=None) 

@pytest.mark.asyncio
async def test_main_runner_executes_cycles_and_logs(caplog, monkeypatch):
    """
    Test the main() function execution flow, mocking key components and checking logs.
    """
    monkeypatch.setattr(f"{main_runner.MAIN_RUNNER_PATH}.num_cycles", 1) 

    mock_trading_account_instance = MagicMock()
    mock_trading_account_instance.get_balance.return_value = {"usdt_balance": 1000.0, "btc_balance": 0.0}
    mock_trading_account_instance.load_state.return_value = False # Simulate no existing state
    mock_trading_account_instance.save_state.return_value = True
    monkeypatch.setattr(f"{main_runner.MAIN_RUNNER_PATH}.TradingAccount", MagicMock(return_value=mock_trading_account_instance))

    mock_trading_agent_instance = MagicMock()
    monkeypatch.setattr(f"{main_runner.MAIN_RUNNER_PATH}.TradingAgent", MagicMock(return_value=mock_trading_agent_instance))

    mock_runner_instance = MagicMock()
    
    # Mock events yielded by runner.run_async
    # Event content should be AgentEventContent for custom agent events
    mock_event_rm_trade_content = AgentEventContent(
        response_type="risk_management_trade_executed", 
        data={"message": "SL triggered for POS123", "trade_details": {"success": True, "btc_sold": 0.1}}
    )
    mock_event_rm_trade = Event(event_type="llm_response", author="agent", content=mock_event_rm_trade_content) # LlmAgent yields llm_response for its own content

    mock_event_agent_summary_content = AgentEventContent(
        response_type="agent_action_summary", 
        data={
            "llm_decision": {"action": "HOLD", "reason": "Market volatile"}, 
            "trade_executed": False,
            "action_result": "Holding position.",
            "final_balances": {"usdt_balance": 950.0, "btc_balance": 0.05}
        }
    )
    mock_event_agent_summary = Event(event_type="llm_response", author="agent", content=mock_event_agent_summary_content, is_final_response=True) # LlmAgent yields llm_response
    
    async def mock_run_async_stream(*args, **kwargs):
        yield mock_event_rm_trade
        yield mock_event_agent_summary

    mock_runner_constructor = MagicMock(return_value=mock_runner_instance)
    mock_runner_instance.run_async = mock_run_async_stream 
    monkeypatch.setattr(f"{main_runner.MAIN_RUNNER_PATH}.InMemoryRunner", mock_runner_constructor)

    mock_history_add_record = MagicMock()
    mock_history_display = MagicMock()
    mock_interaction_history_instance = MagicMock()
    mock_interaction_history_instance.add_record = mock_history_add_record
    mock_interaction_history_instance.display_history = mock_history_display
    mock_interaction_history_instance.load_history.return_value = False # Simulate no existing history
    mock_interaction_history_instance.save_history.return_value = True
    monkeypatch.setattr(f"{main_runner.MAIN_RUNNER_PATH}.InteractionHistory", MagicMock(return_value=mock_interaction_history_instance))

    await main_runner.main()

    log_text = caplog.text
    assert "Main function started. Initializing components..." in log_text
    assert "New TradingAccount initialized." in log_text # Because load_state returns False
    assert "TradingAgent initialized." in log_text
    assert "InMemoryRunner initialized." in log_text
    assert "New InteractionHistory initialized." in log_text # Because load_history returns False
    assert "Starting trading simulation for 1 cycles." in log_text
    assert "--- Starting Trading Cycle 1/1 ---" in log_text
    assert "Cycle 1: User trigger sent to agent." in log_text
    
    assert "Risk Management trade: SL triggered for POS123" in log_text
    assert "Cycle 1: Agent Action Summary: HOLD - Market volatile" in log_text
    
    assert "Cycle 1: State and history saved." in log_text
    assert "--- End of Trading Cycle 1/1 ---" in log_text
    assert "Trading simulation loop finished or interrupted." in log_text
    assert "Final trading account state saved." in log_text
    assert "Final interaction history saved." in log_text
    assert "Main function completed." in log_text
    
    mock_trading_account_instance.load_state.assert_called_once_with(main_runner.ACCOUNT_STATE_FILE)
    mock_interaction_history_instance.load_history.assert_called_once_with(main_runner.INTERACTION_HISTORY_FILE)
    
    # Called after cycle + in finally block
    assert mock_trading_account_instance.save_state.call_count == 2
    mock_trading_account_instance.save_state.assert_called_with(main_runner.ACCOUNT_STATE_FILE)
    assert mock_interaction_history_instance.save_history.call_count == 2
    mock_interaction_history_instance.save_history.assert_called_with(main_runner.INTERACTION_HISTORY_FILE)

    mock_history_display.assert_called_once()


def test_setup_logging_configures_handlers(caplog):
    app_logger = logging.getLogger("TradingAgentRunner")
    
    main_runner.setup_logging(log_level=logging.DEBUG, log_file=None)
    assert len(app_logger.handlers) >= 1 
    assert any(isinstance(h, logging.StreamHandler) for h in app_logger.handlers)
    
    with patch("logging.FileHandler", MagicMock()) as mock_file_handler_class:
        main_runner.setup_logging(log_level=logging.INFO, log_file="test_dummy.log")
        assert len(app_logger.handlers) >= 2 
        mock_file_handler_class.assert_called_once_with("test_dummy.log", mode='a')
        
    assert "Logging setup complete." in caplog.text

```
