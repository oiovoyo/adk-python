import datetime
from typing import Literal, AsyncGenerator, Any, Optional, List 
import logging # Added

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from pydantic import BaseModel, Field

from btc_usdt_trading_agent.tools.binance_data_tool import BinanceDataTool
from btc_usdt_trading_agent.account.trading_account import TradingAccount
from btc_usdt_trading_agent.tools.news_fetch_tool import get_crypto_news

logger = logging.getLogger("TradingAgentRunner.TradingAgent") # Added

class TradingDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"] = Field(description="The trading action to take: BUY, SELL, or HOLD.")
    amount_usdt_to_spend: float = Field(default=0.0, description="Amount of USDT to spend if action is BUY.")
    amount_btc_to_sell: float = Field(default=0.0, description="Amount of BTC to sell if action is SELL.")
    reason: str = Field(description="Concise reasoning for the trading decision.")
    suggested_stop_loss_percentage: Optional[float] = Field(default=None, description="Suggested stop-loss as a percentage (e.g., 0.05 for 5%). For BUY only.")
    suggested_take_profit_percentage: Optional[float] = Field(default=None, description="Suggested take-profit as a percentage (e.g., 0.10 for 10%). For BUY only.")


class TradingAgent(LlmAgent[TradingDecision, BaseAgentResponse[TradingDecision]]):
    def __init__(self, trading_account: TradingAccount, model_name: str = "gemini-1.5-flash"):
        self.trading_account = trading_account
        self.binance_data_tool = BinanceDataTool() 
        logger.debug(f"TradingAgent initialized with model: {model_name}. TradingAccount balance: USDT {trading_account.usdt_balance:.2f}, BTC {trading_account.btc_balance:.8f}")


        tools = [
            FunctionTool(func=self.binance_data_tool.get_ticker_price, name="get_ticker_price", description="Fetches latest BTC/USDT price."),
            FunctionTool(func=self.binance_data_tool.get_candlestick_data, name="get_candlestick_data", description="Fetches candlestick data."),
            FunctionTool(func=get_crypto_news, name="get_cryptocurrency_news", description="Fetches crypto news."),
            FunctionTool(func=self._get_current_account_position, name="get_current_account_position", description="Fetches current account balances."),
            FunctionTool(func=self._get_recent_trade_history, name="get_recent_trade_history", description="Fetches recent trade history.")
        ]

        self._original_instruction = """\
You are a cryptocurrency trading analyst. Your goal is to maximize the USDT value of a simulated trading account.

**Process:**
1.  **Assess Current Standing:** Use `get_current_account_position()` and `get_recent_trade_history(num_trades: int = 5)` to understand your current financial position and past actions.
2.  **Analyze Market:** Use `get_ticker_price(symbol: str)`, `get_candlestick_data(...)`, and `get_cryptocurrency_news(...)` to gather market intelligence.
3.  **Decide Action:** Based on all available data, decide whether to BUY BTC, SELL BTC, or HOLD.

**Decision Guidelines:**
*   **BUY BTC:**
    *   Specify `amount_usdt_to_spend`.
    *   Ensure `amount_btc_to_sell` is 0.
    *   **Optionally, suggest risk management levels:**
        *   `suggested_stop_loss_percentage`: e.g., 0.05 for 5% below your intended entry price.
        *   `suggested_take_profit_percentage`: e.g., 0.10 for 10% above your intended entry price.
*   **SELL BTC:**
    *   Specify `amount_btc_to_sell`.
    *   Ensure `amount_usdt_to_spend` is 0.
*   **HOLD:** Set both amounts to 0.

**Reasoning:** Provide clear, concise reasoning based on all data.

**Output Format (JSON):**
```json
{{
  "action": "BUY" | "SELL" | "HOLD",
  "amount_usdt_to_spend": float,
  "amount_btc_to_sell": float,
  "reason": "Your concise analysis and reasoning.",
  "suggested_stop_loss_percentage": float (optional, for BUY only),
  "suggested_take_profit_percentage": float (optional, for BUY only)
}}
```
The system will automatically check for stop-loss/take-profit triggers on open positions before asking for your new decision.
"""
        super().__init__(
            instruction=self._original_instruction,
            tools=tools,
            model=model_name,
            output_schema=TradingDecision,
            response_type=BaseAgentResponse[TradingDecision]
        )

    def _check_and_trigger_risk_management_orders(self, current_price: float) -> List[dict]:
        actions_taken: List[dict] = []
        open_positions_to_check = [pos for pos in self.trading_account.open_positions if pos.status == "OPEN"]
        logger.debug(f"Risk Management: Checking {len(open_positions_to_check)} open positions against current price: {current_price:.2f}")

        for position in open_positions_to_check:
            action_taken_for_this_position = False
            current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

            if position.stop_loss_price is not None and current_price <= position.stop_loss_price:
                logger.info(f"Stop-loss condition met for position {position.position_id} (Entry: {position.entry_price:.2f}, SL: {position.stop_loss_price:.2f}) at current price {current_price:.2f}. Attempting to sell.")
                sell_outcome = self.trading_account.execute_sell_order(
                    symbol=position.symbol,
                    btc_amount_to_sell=position.amount_crypto,
                    current_btc_price=current_price, 
                    reason="Stop-loss triggered",
                    timestamp=current_time,
                    position_id_to_close=position.position_id
                )
                actions_taken.append(sell_outcome)
                if sell_outcome.get("success"):
                    logger.info(f"Risk management SALE (Stop-Loss) for position {sell_outcome.get('position_id_closed')} successful. Sold {sell_outcome.get('btc_sold'):.8f} BTC.")
                else:
                    logger.warning(f"Risk management SALE (Stop-Loss) attempt failed for position {position.position_id}. Reason: {sell_outcome.get('message')}")
                action_taken_for_this_position = True 

            if not action_taken_for_this_position and \
               position.take_profit_price is not None and \
               current_price >= position.take_profit_price:
                logger.info(f"Take-profit condition met for position {position.position_id} (Entry: {position.entry_price:.2f}, TP: {position.take_profit_price:.2f}) at current price {current_price:.2f}. Attempting to sell.")
                sell_outcome = self.trading_account.execute_sell_order(
                    symbol=position.symbol,
                    btc_amount_to_sell=position.amount_crypto,
                    current_btc_price=current_price, 
                    reason="Take-profit triggered",
                    timestamp=current_time,
                    position_id_to_close=position.position_id
                )
                actions_taken.append(sell_outcome)
                if sell_outcome.get("success"):
                    logger.info(f"Risk management SALE (Take-Profit) for position {sell_outcome.get('position_id_closed')} successful. Sold {sell_outcome.get('btc_sold'):.8f} BTC.")
                else:
                    logger.warning(f"Risk management SALE (Take-Profit) attempt failed for position {position.position_id}. Reason: {sell_outcome.get('message')}")
        
        if not actions_taken:
            logger.debug("Risk Management: No stop-loss or take-profit orders triggered.")
        return actions_taken

    async def _run_async_impl(
        self,
        request: LlmRequest, 
        execution_context: Any | None = None
    ) -> AsyncGenerator[BaseAgentResponse[TradingDecision], None]:
        
        logger.debug("Agent _run_async_impl started.")
        current_btc_price: Optional[float] = None
        ticker_data = self.binance_data_tool.get_ticker_price(symbol="BTCUSDT") 
        if "error" not in ticker_data and ticker_data.get("price"):
            try:
                current_btc_price = float(ticker_data["price"])
                logger.info(f"Current BTC/USDT price for pre-LLM risk check: {current_btc_price:.2f}")
            except ValueError as e:
                logger.warning(f"Invalid price format from get_ticker_price: {ticker_data.get('price')}. Error: {e}")
                yield BaseAgentResponse(response_type="agent_error", response_data={"error": f"Invalid price format: {ticker_data.get('price')}"}, raw_request=request)
        else:
            logger.warning(f"Failed to fetch current BTC price for risk management: {ticker_data.get('error', 'Unknown error')}")
            yield BaseAgentResponse(response_type="agent_error", response_data={"error": f"Price fetch failed: {ticker_data.get('error')}"}, raw_request=request)

        if current_btc_price is not None:
            risk_actions = self._check_and_trigger_risk_management_orders(current_btc_price)
            for action_result in risk_actions: # This loop will execute if any SL/TP trades occurred
                yield BaseAgentResponse(
                    response_type="risk_management_trade_executed" if action_result.get("success") else "risk_management_trade_error",
                    response_data=action_result, raw_request=request
                )
        
        logger.debug("Proceeding to LLM for trading decision.")
        llm_decision_response: TradingDecision | None = None
        async for event in super()._run_async_impl(request, execution_context):
            yield event 
            if event.response_type == "llm_response" and event.response_data:
                if isinstance(event.response_data, TradingDecision):
                     llm_decision_response = event.response_data
                elif isinstance(event.response_data, dict) and "action" in event.response_data: 
                    try:
                        llm_decision_response = TradingDecision(**event.response_data)
                    except Exception as pydantic_error:
                        logger.warning(f"LLM output failed Pydantic validation: {pydantic_error}. Raw: {event.response_data}")
                        yield BaseAgentResponse(response_type="agent_error", response_data={"error": f"LLM output parsing error: {pydantic_error}", "raw_output": event.response_data}, raw_request=request)
                        llm_decision_response = None 
                        break 

        if llm_decision_response:
            decision = llm_decision_response
            logger.info(f"LLM decided action: {decision.action}. Reason: {decision.reason}")
            action_result_message = f"LLM Decision: {decision.action}. Reason: {decision.reason}"
            trade_executed = False
            current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            trade_outcome_details = {}

            if decision.action == "BUY":
                logger.debug(f"Processing BUY decision. USDT to spend: {decision.amount_usdt_to_spend}. SL%: {decision.suggested_stop_loss_percentage}, TP%: {decision.suggested_take_profit_percentage}")
                if decision.amount_usdt_to_spend > 0:
                    price_data_for_trade = self.binance_data_tool.get_ticker_price(symbol="BTCUSDT")
                    if "error" not in price_data_for_trade and price_data_for_trade.get("price"):
                        current_price_for_trade = float(price_data_for_trade["price"])
                        logger.debug(f"Fetched price for BUY execution: {current_price_for_trade:.2f}")
                        
                        stop_loss_price: Optional[float] = None
                        take_profit_price: Optional[float] = None

                        if decision.suggested_stop_loss_percentage is not None and decision.suggested_stop_loss_percentage > 0:
                            stop_loss_price = current_price_for_trade * (1 - decision.suggested_stop_loss_percentage)
                            logger.debug(f"Calculated SL price: {stop_loss_price:.2f} based on suggestion {decision.suggested_stop_loss_percentage*100:.2f}%")
                        if decision.suggested_take_profit_percentage is not None and decision.suggested_take_profit_percentage > 0:
                            take_profit_price = current_price_for_trade * (1 + decision.suggested_take_profit_percentage)
                            logger.debug(f"Calculated TP price: {take_profit_price:.2f} based on suggestion {decision.suggested_take_profit_percentage*100:.2f}%")
                        
                        buy_outcome = self.trading_account.execute_buy_order(
                            symbol="BTCUSDT", usdt_amount_to_spend=decision.amount_usdt_to_spend,
                            current_btc_price=current_price_for_trade, reason=decision.reason,
                            timestamp=current_time, stop_loss_price=stop_loss_price, take_profit_price=take_profit_price
                        )
                        action_result_message += f"\nBuy Order Outcome: {buy_outcome['message']}"
                        trade_executed = buy_outcome["success"]
                        trade_outcome_details = buy_outcome
                        if trade_executed:
                             logger.info(f"New BUY position {buy_outcome.get('position_id')} opened. Entry: {buy_outcome.get('entry_price'):.2f}, SL: {buy_outcome.get('stop_loss_price'):.2f}, TP: {buy_outcome.get('take_profit_price'):.2f}")
                    else:
                        action_result_message += f"\nCould not execute BUY: Failed to fetch current price for trade - {price_data_for_trade.get('error', 'Unknown error')}"
                        logger.warning(f"BUY execution failed: Price fetch error - {price_data_for_trade.get('error', 'Unknown error')}")
                else:
                    action_result_message += "\nBuy action chosen, but amount_usdt_to_spend was zero. No trade executed."
                    logger.info("BUY action with zero amount_usdt_to_spend. No trade.")
            
            elif decision.action == "SELL":
                logger.debug(f"Processing SELL decision. BTC to sell: {decision.amount_btc_to_sell}")
                if decision.amount_btc_to_sell > 0: 
                    price_data_for_trade = self.binance_data_tool.get_ticker_price(symbol="BTCUSDT")
                    if "error" not in price_data_for_trade and price_data_for_trade.get("price"):
                        current_price_for_trade = float(price_data_for_trade["price"])
                        logger.debug(f"Fetched price for SELL execution: {current_price_for_trade:.2f}")
                        sell_outcome = self.trading_account.execute_sell_order(
                            symbol="BTCUSDT", btc_amount_to_sell=decision.amount_btc_to_sell, 
                            current_btc_price=current_price_for_trade, reason=decision.reason, timestamp=current_time
                        )
                        action_result_message += f"\nSell Order Outcome: {sell_outcome['message']}"
                        trade_executed = sell_outcome["success"]
                        trade_outcome_details = sell_outcome
                    else:
                        action_result_message += f"\nCould not execute SELL: Failed to fetch current price for trade - {price_data_for_trade.get('error', 'Unknown error')}"
                        logger.warning(f"SELL execution failed: Price fetch error - {price_data_for_trade.get('error', 'Unknown error')}")
                else:
                    action_result_message += "\nSell action chosen, but amount_btc_to_sell was zero. No trade executed."
                    logger.info("SELL action with zero amount_btc_to_sell. No trade.")
            
            elif decision.action == "HOLD":
                action_result_message += "\nHolding position. No trade executed."
                logger.info("HOLD decision processed. No trade executed.")

            final_balances = self.trading_account.get_balance()
            action_result_message += f"\nFinal Balances: USDT: {final_balances['usdt_balance']:.2f}, BTC: {final_balances['btc_balance']:.8f}"
            
            yield BaseAgentResponse(
                response_type="agent_action_summary",
                response_data={
                    "llm_decision": decision.model_dump(), "action_result": action_result_message,
                    "trade_executed": trade_executed, "trade_details": trade_outcome_details,
                    "final_balances": final_balances
                },
                raw_request=request, raw_response=llm_decision_response 
            )
        elif request.agent_id: 
             logger.warning("LLM did not produce a final TradingDecision or was interrupted.")
             yield BaseAgentResponse(response_type="agent_error", response_data={"error": "LLM did not produce final decision."}, raw_request=request, raw_response=None)

    def _get_current_account_position(self) -> dict:
        logger.debug("Fetching current account position.")
        return self.trading_account.get_balance()

    def _get_recent_trade_history(self, num_trades: int = 5) -> dict:
        logger.debug(f"Fetching recent trade history for last {num_trades} trades.")
        if num_trades <= 0:
            logger.debug("num_trades is non-positive, returning empty history.")
            history_slice = []
        else:
            history_slice = self.trading_account.transaction_history[-num_trades:]
        
        return {"trade_history": history_slice, "trades_returned": len(history_slice)}
