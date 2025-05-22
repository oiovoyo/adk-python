import datetime
from typing import Literal, AsyncGenerator, Any, Optional, List # Added Optional, List

from google_adk.agents import LlmAgent, BaseAgentResponse, LlmRequest
from google_adk.tools import FunctionTool
from pydantic import BaseModel, Field

from btc_usdt_trading_agent.tools.binance_data_tool import BinanceDataTool
from btc_usdt_trading_agent.account.trading_account import TradingAccount
from btc_usdt_trading_agent.tools.news_fetch_tool import get_crypto_news


class TradingDecision(BaseModel):
    """
    Pydantic model for the structured output expected from the LLM
    representing a trading decision.
    """
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        description="The trading action to take: BUY, SELL, or HOLD."
    )
    amount_usdt_to_spend: float = Field(
        default=0.0, 
        description="Amount of USDT to spend if action is BUY. Must be greater than 0 for BUY action."
    )
    amount_btc_to_sell: float = Field(
        default=0.0, 
        description="Amount of BTC to sell if action is SELL. Must be greater than 0 for SELL action."
    )
    reason: str = Field(
        description="Concise reasoning for the trading decision based on market data analysis."
    )
    # New fields for SL/TP suggestions
    suggested_stop_loss_percentage: Optional[float] = Field(
        default=None, 
        description="Suggested stop-loss as a percentage below entry price (e.g., 0.05 for 5%). Only for BUY action."
    )
    suggested_take_profit_percentage: Optional[float] = Field(
        default=None, 
        description="Suggested take-profit as a percentage above entry price (e.g., 0.10 for 10%). Only for BUY action."
    )


class TradingAgent(LlmAgent[TradingDecision, BaseAgentResponse[TradingDecision]]):
    """
    A cryptocurrency trading agent that analyzes market data and makes trading
    decisions (BUY, SELL, HOLD) for BTC/USDT to maximize USDT value.
    It also manages open positions with stop-loss and take-profit levels.
    """

    def __init__(self, trading_account: TradingAccount, model_name: str = "gemini-1.5-flash"):
        self.trading_account = trading_account
        self.binance_data_tool = BinanceDataTool() 

        tools = [
            FunctionTool(func=self.binance_data_tool.get_ticker_price, name="get_ticker_price", description="Fetches the latest price for a cryptocurrency symbol. Example: {\"symbol\": \"BTCUSDT\"}"),
            FunctionTool(func=self.binance_data_tool.get_candlestick_data, name="get_candlestick_data", description="Fetches recent candlestick (k-line) data. Example: {\"symbol\": \"BTCUSDT\", \"interval\": \"1h\", \"limit\": 24}"),
            FunctionTool(func=get_crypto_news, name="get_cryptocurrency_news", description="Fetches recent news articles. Example: {\"query\": \"bitcoin price sentiment\", \"num_results\": 3}"),
            FunctionTool(func=self._get_current_account_position, name="get_current_account_position", description="Fetches current USDT and BTC balances."),
            FunctionTool(func=self._get_recent_trade_history, name="get_recent_trade_history", description="Fetches the last N trades. Example: {\"num_trades\": 5}")
        ]

        self._original_instruction = """\
You are a cryptocurrency trading analyst. Your goal is to maximize the USDT value of a simulated trading account.

**Process:**
1.  **Assess Current Standing:** Use `get_current_account_position()` and `get_recent_trade_history(num_trades: int = 5)` to understand your current financial position and past actions.
2.  **Analyze Market:** Use `get_ticker_price(symbol: str)`, `get_candlestick_data(...)`, and `get_cryptocurrency_news(...)` to gather market intelligence.
3.  **Decide Action:** Based on all available data, decide whether to BUY BTC, SELL BTC, or HOLD.

**Decision Guidelines:**
*   **BUY BTC:**
    *   Specify `amount_usdt_to_spend` (e.g., 10-50% of available USDT if strong opportunity).
    *   Ensure `amount_btc_to_sell` is 0.
    *   **Optionally, suggest risk management levels:**
        *   `suggested_stop_loss_percentage`: e.g., 0.05 for 5% below your intended entry price.
        *   `suggested_take_profit_percentage`: e.g., 0.10 for 10% above your intended entry price.
        *   If not provided, default percentages will be applied.
*   **SELL BTC:**
    *   Specify `amount_btc_to_sell` (e.g., a portion of available BTC).
    *   Ensure `amount_usdt_to_spend` is 0.
    *   (Stop-loss/take-profit are set at time of BUY).
*   **HOLD:**
    *   Set both `amount_usdt_to_spend` and `amount_btc_to_sell` to 0. Appropriate if no clear opportunity or high uncertainty.

**Reasoning:**
You MUST provide a clear and concise reason for your decision, integrating analysis from all data sources (account status, market data, news).

**Output Format (JSON):**
```json
{{
  "action": "BUY" | "SELL" | "HOLD",
  "amount_usdt_to_spend": float,
  "amount_btc_to_sell": float,
  "reason": "Your concise analysis and reasoning.",
  "suggested_stop_loss_percentage": float (optional, for BUY only, e.g., 0.05),
  "suggested_take_profit_percentage": float (optional, for BUY only, e.g., 0.10)
}}
```
Example BUY: {{"action": "BUY", "amount_usdt_to_spend": 200.0, "amount_btc_to_sell": 0.0, "reason": "Price at support, positive news.", "suggested_stop_loss_percentage": 0.03, "suggested_take_profit_percentage": 0.08}}
Example SELL: {{"action": "SELL", "amount_usdt_to_spend": 0.0, "amount_btc_to_sell": 0.01, "reason": "Reached resistance, negative sentiment from news."}}
Example HOLD: {{"action": "HOLD", "amount_usdt_to_spend": 0.0, "amount_btc_to_sell": 0.0, "reason": "Market consolidating, awaiting clearer signals."}}

The system will automatically check for stop-loss/take-profit triggers on open positions before asking for your new decision.
"""
        super().__init__(
            instruction=self._original_instruction,
            tools=tools,
            model=model_name,
            output_schema=TradingDecision,
            response_type=BaseAgentResponse[TradingDecision]
        )

    async def _run_async_impl(
        self,
        request: LlmRequest, # The initial request that starts this agent run
        execution_context: Any | None = None
    ) -> AsyncGenerator[BaseAgentResponse[TradingDecision], None]:
        
        # 1. Fetch Current Price for Risk Management
        current_btc_price: Optional[float] = None
        ticker_data = self.binance_data_tool.get_ticker_price(symbol="BTCUSDT") # Synchronous call
        if "error" not in ticker_data and ticker_data.get("price"):
            try:
                current_btc_price = float(ticker_data["price"])
            except ValueError:
                yield BaseAgentResponse(
                    response_type="agent_error",
                    response_data={"error": f"Invalid price format from get_ticker_price: {ticker_data.get('price')}"},
                    raw_request=request
                )
        else:
            yield BaseAgentResponse(
                response_type="agent_error",
                response_data={"error": f"Failed to fetch current BTC price for risk management: {ticker_data.get('error', 'Unknown error')}"},
                raw_request=request
            )

        # 2. Call Risk Management Check if price is available
        if current_btc_price is not None:
            risk_actions = self._check_and_trigger_risk_management_orders(current_btc_price)
            for action_result in risk_actions:
                if action_result.get("success"):
                    yield BaseAgentResponse(
                        response_type="risk_management_trade_executed", # Distinguishable event type
                        response_data={
                            "message": "Automatic risk management trade executed.",
                            "trade_details": action_result
                        },
                        raw_request=request # Associate with the initial request
                    )
                else: # If SL/TP sell order failed for some reason (e.g. insufficient funds if logic error)
                     yield BaseAgentResponse(
                        response_type="risk_management_trade_error",
                        response_data={
                            "message": "Automatic risk management trade failed.",
                            "error_details": action_result
                        },
                        raw_request=request
                    )
        
        # 3. Proceed with existing LLM interaction loop (tools, decision making)
        # The agent's instruction is static (self._original_instruction).
        # LlmAgent's super()._run_async_impl handles tool calls based on LLM's requests.
        llm_decision_response: TradingDecision | None = None
        async for event in super()._run_async_impl(request, execution_context):
            yield event # Yield all events from the parent class (tool calls, intermediate LLM responses)
            if event.response_type == "llm_response" and event.response_data:
                # Check if the response_data is already the parsed Pydantic model or a raw dict
                if isinstance(event.response_data, TradingDecision):
                     llm_decision_response = event.response_data
                elif isinstance(event.response_data, dict) and "action" in event.response_data: # If it's a dict from LLM
                    try:
                        llm_decision_response = TradingDecision(**event.response_data)
                    except Exception as pydantic_error:
                        yield BaseAgentResponse(
                            response_type="agent_error",
                            response_data={"error": f"LLM output did not conform to TradingDecision schema after tool use: {pydantic_error}", "raw_output": event.response_data},
                            raw_request=request,
                            raw_response=None 
                        )
                        llm_decision_response = None 
                        break # Stop further processing if LLM output is invalid

        # 4. Process the LLM's final decision
        if llm_decision_response:
            decision = llm_decision_response
            action_result_message = f"LLM Decision: {decision.action}. Reason: {decision.reason}"
            trade_executed = False
            current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            trade_outcome_details = {}

            if decision.action == "BUY":
                if decision.amount_usdt_to_spend > 0:
                    price_data_for_trade = self.binance_data_tool.get_ticker_price(symbol="BTCUSDT")
                    if "error" not in price_data_for_trade and price_data_for_trade.get("price"):
                        current_price_for_trade = float(price_data_for_trade["price"])
                        
                        stop_loss_price: Optional[float] = None
                        take_profit_price: Optional[float] = None

                        if decision.suggested_stop_loss_percentage is not None and decision.suggested_stop_loss_percentage > 0:
                            stop_loss_price = current_price_for_trade * (1 - decision.suggested_stop_loss_percentage)
                        if decision.suggested_take_profit_percentage is not None and decision.suggested_take_profit_percentage > 0:
                            take_profit_price = current_price_for_trade * (1 + decision.suggested_take_profit_percentage)
                        
                        buy_outcome = self.trading_account.execute_buy_order(
                            symbol="BTCUSDT",
                            usdt_amount_to_spend=decision.amount_usdt_to_spend,
                            current_btc_price=current_price_for_trade,
                            reason=decision.reason,
                            timestamp=current_time,
                            stop_loss_price=stop_loss_price,      # Pass calculated SL
                            take_profit_price=take_profit_price   # Pass calculated TP
                        )
                        action_result_message += f"\nBuy Order Outcome: {buy_outcome['message']}"
                        trade_executed = buy_outcome["success"]
                        trade_outcome_details = buy_outcome
                    else:
                        action_result_message += f"\nCould not execute BUY: Failed to fetch current price for trade - {price_data_for_trade.get('error', 'Unknown error')}"
                else:
                    action_result_message += "\nBuy action chosen, but amount_usdt_to_spend was zero. No trade executed."
            
            elif decision.action == "SELL":
                if decision.amount_btc_to_sell > 0: # For manual sell, LLM decides amount
                    price_data_for_trade = self.binance_data_tool.get_ticker_price(symbol="BTCUSDT")
                    if "error" not in price_data_for_trade and price_data_for_trade.get("price"):
                        current_price_for_trade = float(price_data_for_trade["price"])
                        # For SELL, LLM doesn't suggest SL/TP. It might specify a position_id to close,
                        # but current TradingDecision doesn't include that. So, it's a market sell of oldest.
                        sell_outcome = self.trading_account.execute_sell_order(
                            symbol="BTCUSDT",
                            btc_amount_to_sell=decision.amount_btc_to_sell, # LLM specified amount
                            current_btc_price=current_price_for_trade,
                            reason=decision.reason,
                            timestamp=current_time
                            # position_id_to_close is not part of TradingDecision yet
                        )
                        action_result_message += f"\nSell Order Outcome: {sell_outcome['message']}"
                        trade_executed = sell_outcome["success"]
                        trade_outcome_details = sell_outcome
                    else:
                        action_result_message += f"\nCould not execute SELL: Failed to fetch current price for trade - {price_data_for_trade.get('error', 'Unknown error')}"
                else:
                    action_result_message += "\nSell action chosen, but amount_btc_to_sell was zero. No trade executed."
            
            elif decision.action == "HOLD":
                action_result_message += "\nHolding position. No trade executed."

            final_balances = self.trading_account.get_balance()
            action_result_message += f"\nFinal Balances: USDT: {final_balances['usdt_balance']:.2f}, BTC: {final_balances['btc_balance']:.8f}"
            
            yield BaseAgentResponse(
                response_type="agent_action_summary",
                response_data={
                    "llm_decision": decision.model_dump(),
                    "action_result": action_result_message,
                    "trade_executed": trade_executed,
                    "trade_details": trade_outcome_details, # Include details of the executed trade
                    "final_balances": final_balances
                },
                raw_request=request,
                raw_response=llm_decision_response 
            )
        elif request.agent_id: # Ensure we yield something if no LLM decision was made but it's not an error from super()
             yield BaseAgentResponse(
                response_type="agent_error",
                response_data={"error": "LLM did not produce a final TradingDecision or was interrupted."},
                raw_request=request,
                raw_response=None 
            )


    def _get_current_account_position(self) -> dict:
        """
        Retrieves the current USDT and BTC balances from the trading account.
        This method is intended to be wrapped as a FunctionTool.
        """
        return self.trading_account.get_balance()

    def _get_recent_trade_history(self, num_trades: int = 5) -> dict:
        """
        Retrieves the most recent trades from the transaction history.
        This method is intended to be wrapped as a FunctionTool.
        """
        if num_trades <= 0:
            history_slice = []
        else:
            history_slice = self.trading_account.transaction_history[-num_trades:]
        
        return {
            "trade_history": history_slice,
            "trades_returned": len(history_slice)
        }

    def _check_and_trigger_risk_management_orders(self, current_price: float) -> List[dict]:
        """
        Checks all open positions and triggers sales if stop-loss or take-profit prices are met.
        """
        actions_taken: List[dict] = []
        open_positions_to_check = [pos for pos in self.trading_account.open_positions if pos.status == "OPEN"]

        for position in open_positions_to_check:
            action_taken_for_this_position = False
            current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

            if position.stop_loss_price is not None and current_price <= position.stop_loss_price:
                sell_outcome = self.trading_account.execute_sell_order(
                    symbol=position.symbol,
                    btc_amount_to_sell=position.amount_crypto,
                    current_btc_price=current_price, # Use current_price for SL/TP execution
                    reason="Stop-loss triggered",
                    timestamp=current_time,
                    position_id_to_close=position.position_id
                )
                actions_taken.append(sell_outcome)
                action_taken_for_this_position = True 

            if not action_taken_for_this_position and \
               position.take_profit_price is not None and \
               current_price >= position.take_profit_price:
                sell_outcome = self.trading_account.execute_sell_order(
                    symbol=position.symbol,
                    btc_amount_to_sell=position.amount_crypto,
                    current_btc_price=current_price, # Use current_price for SL/TP execution
                    reason="Take-profit triggered",
                    timestamp=current_time,
                    position_id_to_close=position.position_id
                )
                actions_taken.append(sell_outcome)
        
        return actions_taken

```
