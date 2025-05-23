import pytest
import pytest_asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from google_adk.agents import Event, BaseAgentResponse # LlmRequest, Content, Part not directly used in this test's mocks but good to remember
from google_adk.models.llm_request import Content, Part # For constructing mock events if needed

# Module to be tested
from btc_usdt_trading_agent import main_runner 
# For patching classes instantiated within main_runner.main
MAIN_RUNNER_PATH = "btc_usdt_trading_agent.main_runner" 

@pytest.fixture(autouse=True)
def ensure_logging_setup_for_tests(caplog):
    """Ensure that logging is set up for each test to capture messages."""
    # Call setup_logging with a low level to capture all messages during tests
    main_runner.setup_logging(log_level=logging.DEBUG, log_file=None) # No file for tests

@pytest.mark.asyncio
async def test_main_runner_executes_cycles_and_logs(caplog, monkeypatch):
    """
    Test the main() function execution flow, mocking key components and checking logs.
    """
    # 1. Patch constants in main_runner
    monkeypatch.setattr(f"{MAIN_RUNNER_PATH}.num_cycles", 1) # Run for only 1 cycle

    # 2. Mock external classes instantiated in main()
    mock_trading_account_instance = MagicMock()
    mock_trading_account_instance.get_balance.return_value = {"usdt_balance": 1000.0, "btc_balance": 0.0}
    monkeypatch.setattr(f"{MAIN_RUNNER_PATH}.TradingAccount", MagicMock(return_value=mock_trading_account_instance))

    mock_trading_agent_instance = MagicMock()
    # The agent instance itself is not heavily used by main_runner's direct logic, 
    # mostly passed to InMemoryRunner. So, a simple MagicMock is often enough.
    monkeypatch.setattr(f"{MAIN_RUNNER_PATH}.TradingAgent", MagicMock(return_value=mock_trading_agent_instance))

    # 3. Mock InMemoryRunner and its run_async method
    mock_runner_instance = MagicMock() # Use MagicMock for the instance
    
    # Define the mock events that run_async will yield
    # These need to be structured as BaseAgentResponse wrapped in Event.content for LlmAgent
    mock_event_rm_trade = Event(
        event_type="risk_management_trade_executed", # This is a custom event type from TradingAgent
        author="agent",
        content=BaseAgentResponse(
            response_type="risk_management_trade_executed", 
            data={"message": "SL triggered for POS123", "trade_details": {"success": True, "btc_sold": 0.1}}
        )
    )
    mock_event_agent_summary = Event(
        event_type="final_response", # LlmAgent yields "final_response"
        author="agent",
        content=BaseAgentResponse(
            response_type="agent_action_summary", 
            data={
                "llm_decision": {"action": "HOLD", "reason": "Market volatile"}, 
                "trade_executed": False,
                "action_result": "Holding position.",
                "final_balances": {"usdt_balance": 950.0, "btc_balance": 0.05}
            }
        ),
        is_final_response=True
    )
    
    async def mock_run_async_stream(*args, **kwargs):
        yield mock_event_rm_trade
        yield mock_event_agent_summary

    # Patch the run_async method of the instance that will be created by InMemoryRunner
    # We mock the InMemoryRunner class, then its instance's run_async
    mock_runner_constructor = MagicMock(return_value=mock_runner_instance)
    mock_runner_instance.run_async = mock_run_async_stream # Assign the async generator
    monkeypatch.setattr(f"{MAIN_RUNNER_PATH}.InMemoryRunner", mock_runner_constructor)


    # 4. Mock InteractionHistory methods
    mock_history_add_record = MagicMock()
    mock_history_display = MagicMock()
    mock_interaction_history_instance = MagicMock()
    mock_interaction_history_instance.add_record = mock_history_add_record
    mock_interaction_history_instance.display_history = mock_history_display
    monkeypatch.setattr(f"{MAIN_RUNNER_PATH}.InteractionHistory", MagicMock(return_value=mock_interaction_history_instance))

    # 5. Run main()
    await main_runner.main()

    # 6. Assertions
    # Check if InMemoryRunner was instantiated and run_async called
    mock_runner_constructor.assert_called_once_with(agent=mock_trading_agent_instance)
    # mock_runner_instance.run_async.assert_called() # run_async is an async gen, direct assert_called might not work as expected.
    # Instead, check effects like logs or history calls.

    # Check log messages
    log_text = caplog.text
    assert "Main function started. Initializing components..." in log_text
    assert "TradingAccount initialized. Initial balance:" in log_text
    assert "TradingAgent initialized." in log_text
    assert "InMemoryRunner initialized." in log_text
    assert "InteractionHistory initialized with max_turns=20" in log_text # Default max_turns
    assert "Starting trading simulation for 1 cycles." in log_text # Patched num_cycles
    assert "--- Starting Trading Cycle 1/1 ---" in log_text
    assert "Cycle 1: User trigger sent to agent." in log_text
    
    # Check logs for specific events processed
    assert "Risk Management trade executed: SL triggered for POS123" in log_text
    assert "Cycle 1: Agent Action Summary Received." in log_text
    assert "LLM Decision: HOLD - Market volatile" in log_text # From mocked summary
    assert "Trade Outcome: Holding position." in log_text
    
    assert "Cycle 1: Account balance:" in log_text # Logged after processing events
    assert "--- End of Trading Cycle 1/1 ---" in log_text
    assert "Trading simulation completed. Final Account Balance:" in log_text
    assert "Main function completed." in log_text

    # Check InteractionHistory calls
    # Example: Check that add_record was called for the user trigger
    mock_history_add_record.assert_any_call(
        record_type="USER_TRIGGER",
        data={"message": "Evaluate market conditions and current account status, then decide on a trading action for BTC/USDT.", "cycle": 1}
    )
    # Check that add_record was called for the agent summary
    # This requires inspecting call_args_list for more complex data
    agent_summary_recorded = False
    for call in mock_history_add_record.call_args_list:
        if call.kwargs.get('record_type') == "AGENT_ACTION_SUMMARY":
            if call.kwargs.get('data', {}).get('llm_decision', {}).get('action') == "HOLD":
                agent_summary_recorded = True
                break
    assert agent_summary_recorded, "Agent action summary for HOLD decision was not recorded in history."
    
    mock_history_display.assert_called_once()


def test_setup_logging_configures_handlers(caplog):
    """Test that setup_logging configures console and optionally file handlers."""
    # Get the specific logger instance that setup_logging configures
    app_logger = logging.getLogger("TradingAgentRunner")
    
    # Test with console only
    main_runner.setup_logging(log_level=logging.DEBUG, log_file=None)
    assert len(app_logger.handlers) >= 1 # Should have at least console handler
    assert any(isinstance(h, logging.StreamHandler) for h in app_logger.handlers)
    
    # Test with file handler (mocking FileHandler to avoid actual file I/O)
    with patch("logging.FileHandler", MagicMock()) as mock_file_handler_class:
        main_runner.setup_logging(log_level=logging.INFO, log_file="test_dummy.log")
        assert len(app_logger.handlers) >= 2 # Console + File
        mock_file_handler_class.assert_called_once_with("test_dummy.log", mode='a')
        
    # Check if initial setup message is logged
    assert "Logging setup complete." in caplog.text
```
