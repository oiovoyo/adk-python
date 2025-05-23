import logging
import asyncio
import datetime 
from typing import Optional, List, Dict, Any 

# ADK Imports
from google_adk.runners import InMemoryRunner
from google_adk.agents import BaseAgentResponse, Event # LlmRequest is not directly used by runner.run_async
from google_adk.models.llm_request import Content, Part # Corrected import for Content/Part

# Project Imports
from btc_usdt_trading_agent.account.trading_account import TradingAccount
from btc_usdt_trading_agent.agent import TradingAgent, TradingDecision


# Instantiate a logger for use within this file.
logger = logging.getLogger("TradingAgentRunner.Main")

def setup_logging(log_level=logging.INFO, log_file: Optional[str] = "trading_agent_run.log"):
    """
    Configures logging for the application.
    """
    app_logger = logging.getLogger("TradingAgentRunner")
    app_logger.setLevel(log_level)
    
    if app_logger.hasHandlers():
        app_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    app_logger.addHandler(ch)
    
    if log_file:
        try:
            fh = logging.FileHandler(log_file, mode='a') 
            fh.setFormatter(formatter)
            app_logger.addHandler(fh)
        except Exception as e:
            console_only_logger = logging.getLogger("TradingAgentRunner.SetupError")
            if not console_only_logger.hasHandlers(): 
                error_ch = logging.StreamHandler()
                error_ch.setFormatter(formatter)
                console_only_logger.addHandler(error_ch)
                console_only_logger.setLevel(logging.ERROR)
            console_only_logger.error(f"Failed to set up file handler for {log_file}: {e}")

    app_logger.info("Logging setup complete. Initial log level: %s.", logging.getLevelName(app_logger.level))


class InteractionHistory:
    """
    Manages a history of agent interactions with a sliding window.
    """
    def __init__(self, max_turns: int = 20):
        if max_turns <= 0:
            raise ValueError("max_turns must be a positive integer.")
        self.max_turns: int = max_turns
        self.history: List[Dict[str, Any]] = []
        self._logger = logging.getLogger("TradingAgentRunner.InteractionHistory")

    def add_record(self, record_type: str, data: dict, timestamp: Optional[str] = None) -> None:
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        record = {
            "timestamp": timestamp,
            "type": record_type,
            "details": data
        }
        self.history.append(record)
        self._logger.debug(f"Added record: {record_type} - {data.get('short_summary', data if isinstance(data, dict) else str(data))}")

        while len(self.history) > self.max_turns:
            removed_record = self.history.pop(0) 
            self._logger.debug(f"History limit ({self.max_turns}) exceeded. Removed oldest record: {removed_record['type']} from {removed_record['timestamp']}")

    def get_recent_history(self) -> List[Dict[str, Any]]:
        return self.history[:] 

    def display_history(self) -> None:
        if not self.history:
            logger.info("Interaction history is empty.") # Use logger
            return
        
        logger.info("\n--- Interaction History ---")
        for record in self.history:
            log_message = f"[{record['timestamp']}] [{record['type']}]\n"
            for key, value in record['details'].items():
                log_message += f"  {key}: {value}\n"
            log_message += ("-" * 20)
            logger.info(log_message) # Use logger for display as well
        logger.info("--- End of History ---\n")


async def main():
    """
    Main entry point for the trading agent runner.
    Orchestrates the agent's lifecycle and interaction.
    """
    logger.info("Main function started. Initializing components...")
    
    trading_account = TradingAccount(initial_usdt_balance=1000.0)
    logger.info(f"TradingAccount initialized. Initial balance: {trading_account.get_balance()}")
    
    trading_agent = TradingAgent(trading_account=trading_account)
    logger.info("TradingAgent initialized.")

    # InMemoryRunner does not take app_name directly in constructor.
    # It's more of a conceptual grouping if used with a broader system.
    runner = InMemoryRunner(agent=trading_agent)
    logger.info("InMemoryRunner initialized.")

    history_manager = InteractionHistory(max_turns=20)
    logger.info(f"InteractionHistory initialized with max_turns={history_manager.max_turns}.")

    num_cycles = 3 # Configurable number of trading cycles
    logger.info(f"Starting trading simulation for {num_cycles} cycles.")

    for i in range(num_cycles):
        cycle_number = i + 1
        logger.info(f"--- Starting Trading Cycle {cycle_number}/{num_cycles} ---")

        user_trigger_text = "Evaluate market conditions and current account status, then decide on a trading action for BTC/USDT."
        user_message = Content(parts=[Part(text=user_trigger_text)], role="user")
        
        history_manager.add_record(
            record_type="USER_TRIGGER", 
            data={"message": user_trigger_text, "cycle": cycle_number}
        )
        logger.info(f"Cycle {cycle_number}: User trigger sent to agent.")

        try:
            session_id = f"session_cycle_{cycle_number}" # New session for each cycle for fresh context
            user_id = "test_user"
            
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
                logger.debug(f"Cycle {cycle_number}: Raw Event - Type: {event.event_type}, Author: {event.author}, Data: {event.data}")
                history_manager.add_record(
                    record_type=f"AGENT_EVENT_{event.event_type.upper()}",
                    data={
                        "author": event.author, 
                        "content_parts": [part.to_dict() for part in event.content.parts] if event.content else None,
                        "is_final": event.is_final_response()
                    }
                )

                if event.response_type == "risk_management_trade_executed":
                    log_data = event.data if isinstance(event.data, dict) else {"raw_data": str(event.data)}
                    logger.info(f"Cycle {cycle_number}: Risk Management trade executed: {log_data.get('message', 'N/A')}")
                    history_manager.add_record("RISK_MGMT_TRADE_EXECUTED", log_data)
                
                elif event.response_type == "risk_management_trade_error":
                    log_data = event.data if isinstance(event.data, dict) else {"raw_data": str(event.data)}
                    logger.error(f"Cycle {cycle_number}: Risk Management trade error: {log_data.get('message', 'N/A')}")
                    history_manager.add_record("RISK_MGMT_TRADE_ERROR", log_data)

                elif event.is_final_response() and isinstance(event.content, BaseAgentResponse):
                    if event.content.response_type == "agent_action_summary":
                        summary_data = event.content.data # This is the dict from TradingAgent
                        llm_decision = summary_data.get("llm_decision", {})
                        action_result = summary_data.get("action_result", "No action result provided.")
                        
                        logger.info(f"Cycle {cycle_number}: Agent Action Summary Received.")
                        logger.info(f"  LLM Decision: {llm_decision.get('action')} - {llm_decision.get('reason')}")
                        if llm_decision.get('action') == "BUY":
                            logger.info(f"    Amount USDT to Spend: {llm_decision.get('amount_usdt_to_spend')}")
                            logger.info(f"    Suggested SL %: {llm_decision.get('suggested_stop_loss_percentage')}")
                            logger.info(f"    Suggested TP %: {llm_decision.get('suggested_take_profit_percentage')}")
                        elif llm_decision.get('action') == "SELL":
                             logger.info(f"    Amount BTC to Sell: {llm_decision.get('amount_btc_to_sell')}")
                        logger.info(f"  Trade Outcome: {action_result}")
                        
                        history_manager.add_record("AGENT_ACTION_SUMMARY", summary_data)
                    elif event.content.response_type == "agent_error":
                        error_data = event.content.data if isinstance(event.content.data, dict) else {"raw_error": str(event.content.data)}
                        logger.error(f"Cycle {cycle_number}: Agent reported an error: {error_data.get('error', 'Unknown agent error')}")
                        history_manager.add_record("AGENT_ERROR", error_data)
                elif event.is_final_response(): # Catch other final responses not matching BaseAgentResponse
                    logger.info(f"Cycle {cycle_number}: Agent final response (unstructured): {event.content}")
                    history_manager.add_record("AGENT_FINAL_UNSTRUCTURED", {"content": str(event.content)})


        except Exception as e:
            logger.error(f"Cycle {cycle_number}: An error occurred during agent run: {e}", exc_info=True)
            history_manager.add_record("CYCLE_ERROR", {"error": str(e), "cycle": cycle_number})

        current_cycle_balance = trading_account.get_balance()
        logger.info(f"Cycle {cycle_number}: Account balance: USDT: {current_cycle_balance['usdt_balance']:.2f}, BTC: {current_cycle_balance['btc_balance']:.8f}")
        logger.info(f"--- End of Trading Cycle {cycle_number}/{num_cycles} ---")
        
        if i < num_cycles - 1: # Don't sleep after the last cycle
            await asyncio.sleep(2) # Simulate time passing

    final_balance = trading_account.get_balance()
    logger.info(f"Trading simulation completed. Final Account Balance: USDT: {final_balance['usdt_balance']:.2f}, BTC: {final_balance['btc_balance']:.8f}")
    
    history_manager.display_history()
    logger.info("Main function completed.")


if __name__ == "__main__":
    setup_logging(log_level=logging.INFO) # INFO for normal run, DEBUG for more verbosity
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Main runner interrupted by user (KeyboardInterrupt). Exiting.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
```
