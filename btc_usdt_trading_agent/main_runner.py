import logging
import asyncio
import datetime 
import json 
import os   
from typing import Optional, List, Dict, Any 

# ADK Imports
from google_adk.runners import InMemoryRunner
from google_adk.events import Event # Updated
# from google_adk.agents import BaseAgentResponse # Removed
from google.genai.types import Content, Part # Updated

# Project Imports
from btc_usdt_trading_agent.account.trading_account import TradingAccount
# Import local AgentEventContent and TradingDecision from agent.py
from btc_usdt_trading_agent.agent import TradingAgent, TradingDecision, AgentEventContent 


logger = logging.getLogger("TradingAgentRunner.Main")
ACCOUNT_STATE_FILE = "trading_account_state.json"
INTERACTION_HISTORY_FILE = "interaction_history.log.json"

def setup_logging(log_level=logging.INFO, log_file: Optional[str] = "trading_agent_run.log"):
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
    def __init__(self, max_turns: int = 20):
        if max_turns <= 0:
            raise ValueError("max_turns must be a positive integer.")
        self.max_turns: int = max_turns
        self.history: List[Dict[str, Any]] = []
        self._logger = logging.getLogger("TradingAgentRunner.InteractionHistory")

    def add_record(self, record_type: str, data: dict, timestamp: Optional[str] = None) -> None:
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record = {"timestamp": timestamp, "type": record_type, "details": data}
        self.history.append(record)
        self._logger.debug(f"Added record: {record_type} - {data.get('short_summary', data if isinstance(data, dict) else str(data))}")
        while len(self.history) > self.max_turns:
            removed_record = self.history.pop(0) 
            self._logger.debug(f"History limit ({self.max_turns}) exceeded. Removed: {removed_record['type']} from {removed_record['timestamp']}")

    def get_recent_history(self) -> List[Dict[str, Any]]:
        return self.history[:] 

    def display_history(self) -> None:
        if not self.history:
            self._logger.info("Interaction history is empty.")
            return
        self._logger.info("\n--- Interaction History ---")
        for record in self.history:
            log_message = f"[{record['timestamp']}] [{record['type']}]\n"
            # Ensure details is a dict before iterating
            details_data = record.get('details', {})
            if isinstance(details_data, dict):
                for key, value in details_data.items():
                    log_message += f"  {key}: {value}\n"
            else: # If details is not a dict (e.g. direct string from some error)
                log_message += f"  data: {details_data}\n"
            log_message += ("-" * 20)
            self._logger.info(log_message)
        self._logger.info("--- End of History ---\n")

    def save_history(self, filepath: str) -> bool:
        self._logger.debug(f"Attempting to save interaction history to {filepath}")
        try:
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=4)
            self._logger.info(f"Interaction history saved to {filepath}. {len(self.history)} records.")
            return True
        except Exception as e:
            self._logger.error(f"Error saving history to {filepath}: {e}", exc_info=True)
            return False

    def load_history(self, filepath: str) -> bool:
        self._logger.debug(f"Attempting to load interaction history from {filepath}")
        if not os.path.exists(filepath):
            self._logger.warning(f"History file {filepath} not found.")
            return False
        try:
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            if not isinstance(loaded_data, list):
                self._logger.error(f"History data in {filepath} is not a list.")
                return False
            self.history = loaded_data
            while len(self.history) > self.max_turns:
                self.history.pop(0)
            self._logger.info(f"History loaded from {filepath}. {len(self.history)} records (max_turns={self.max_turns}).")
            return True
        except Exception as e:
            self._logger.error(f"Error loading history from {filepath}: {e}", exc_info=True)
            return False


async def main():
    logger.info("Main function started. Initializing components...")
    
    trading_account = TradingAccount()
    if trading_account.load_state(ACCOUNT_STATE_FILE):
        logger.info(f"TradingAccount state loaded. Balance: {trading_account.get_balance()}")
    else:
        logger.info(f"New TradingAccount initialized. Balance: {trading_account.get_balance()}")
    
    trading_agent = TradingAgent(trading_account=trading_account)
    logger.info("TradingAgent initialized.")

    runner = InMemoryRunner(agent=trading_agent)
    logger.info("InMemoryRunner initialized.")

    history_manager = InteractionHistory(max_turns=20)
    if history_manager.load_history(INTERACTION_HISTORY_FILE):
        logger.info("InteractionHistory loaded.")
    else:
        logger.info("New InteractionHistory initialized.")

    num_cycles = 1 
    logger.info(f"Starting trading simulation for {num_cycles} cycles.")

    try:
        for i in range(num_cycles):
            cycle_number = i + 1
            logger.info(f"--- Starting Trading Cycle {cycle_number}/{num_cycles} ---")
            user_trigger_text = "Evaluate market and decide action for BTC/USDT."
            # Correct usage of Content and Part from google.genai.types
            user_message = Content(parts=[Part(text=user_trigger_text)], role="user") 
            
            history_manager.add_record("USER_TRIGGER", {"message": user_trigger_text, "cycle": cycle_number})
            logger.info(f"Cycle {cycle_number}: User trigger sent.")

            try:
                session_id = f"session_cycle_{cycle_number}"
                user_id = "test_user"
                
                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
                    logger.debug(f"Cycle {cycle_number}: Raw Event - Type: {event.event_type}, Author: {event.author}")
                    
                    event_data_for_history = {"author": event.author, "is_final": event.is_final_response()}
                    
                    if isinstance(event.content, AgentEventContent): # Check against local AgentEventContent
                        agent_event_content = event.content
                        event_data_for_history["response_type"] = agent_event_content.response_type
                        event_data_for_history["data_summary"] = str(agent_event_content.data)[:200]

                        if agent_event_content.response_type == "risk_management_trade_executed":
                            logger.info(f"Cycle {cycle_number}: Risk Management trade: {agent_event_content.data.get('message', 'N/A')}")
                        elif agent_event_content.response_type == "risk_management_trade_error":
                            logger.error(f"Cycle {cycle_number}: Risk Management error: {agent_event_content.data.get('message', 'N/A')}")
                        elif agent_event_content.response_type == "agent_action_summary" and event.is_final_response():
                            summary_data = agent_event_content.data
                            llm_decision_data = summary_data.get("llm_decision", {})
                            logger.info(f"Cycle {cycle_number}: Agent Action Summary: {llm_decision_data.get('action')} - {llm_decision_data.get('reason')}")
                        elif agent_event_content.response_type == "agent_error":
                            logger.error(f"Cycle {cycle_number}: Agent error: {agent_event_content.data.get('error', 'Unknown')}")
                    
                    elif isinstance(event.content, TradingDecision) and event.is_final_response():
                        logger.info(f"Cycle {cycle_number}: Direct LLM TradingDecision received (should be wrapped by agent): {event.content}")
                        event_data_for_history["llm_decision_direct"] = event.content.model_dump()
                    
                    elif event.content and event.content.parts: 
                        event_data_for_history["content_parts_summary"] = [str(part)[:200] for part in event.content.parts]
                    
                    history_manager.add_record(f"AGENT_EVENT_{event.event_type.upper()}", event_data_for_history)

            except Exception as e:
                logger.error(f"Cycle {cycle_number}: Error during agent run: {e}", exc_info=True)
                history_manager.add_record("CYCLE_ERROR", {"error": str(e), "cycle": cycle_number})

            trading_account.save_state(ACCOUNT_STATE_FILE)
            history_manager.save_history(INTERACTION_HISTORY_FILE)
            logger.info(f"Cycle {cycle_number}: State and history saved. Account: {trading_account.get_balance()}")
            logger.info(f"--- End of Trading Cycle {cycle_number}/{num_cycles} ---")
            
            if i < num_cycles - 1: 
                await asyncio.sleep(1) 
    finally:
        logger.info("Trading simulation loop finished or interrupted.")
        if trading_account.save_state(ACCOUNT_STATE_FILE): logger.info("Final trading account state saved.")
        else: logger.error("Failed to save final trading account state.")
        
        if history_manager.save_history(INTERACTION_HISTORY_FILE): logger.info("Final interaction history saved.")
        else: logger.error("Failed to save final interaction history.")
        
        history_manager.display_history()
        logger.info("Main function completed.")


if __name__ == "__main__":
    setup_logging(log_level=logging.INFO) 
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Main runner interrupted by user (KeyboardInterrupt). Exiting.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
```
