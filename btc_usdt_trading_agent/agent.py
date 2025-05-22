import datetime
from typing import Literal, AsyncGenerator, Any

from google_adk.agents import LlmAgent, BaseAgentResponse, LlmRequest
from google_adk.tools import FunctionTool
from pydantic import BaseModel, Field

from btc_usdt_trading_agent.tools.binance_data_tool import BinanceDataTool
from btc_usdt_trading_agent.account.trading_account import TradingAccount


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


class TradingAgent(LlmAgent[TradingDecision, BaseAgentResponse[TradingDecision]]):
    """
    A cryptocurrency trading agent that analyzes market data and makes trading
    decisions (BUY, SELL, HOLD) for BTC/USDT to maximize USDT value.
    """

    def __init__(self, trading_account: TradingAccount, model_name: str = "gemini-1.5-flash"):
        """
        Initializes the TradingAgent.

        Args:
            trading_account: An instance of TradingAccount to manage funds and execute trades.
            model_name: The name of the LLM model to use (e.g., "gemini-1.5-flash").
        """
        self.trading_account = trading_account
        self.binance_data_tool = BinanceDataTool() # API key/secret handled by tool if needed

        tools = [
            FunctionTool(
                func=self.binance_data_tool.get_ticker_price,
                name="get_ticker_price",
                description="Fetches the latest price for a cryptocurrency symbol from Binance. Example: {\"symbol\": \"BTCUSDT\"}"
            ),
            FunctionTool(
                func=self.binance_data_tool.get_candlestick_data,
                name="get_candlestick_data",
                description="Fetches recent candlestick (k-line) data for a cryptocurrency symbol from Binance. Example: {\"symbol\": \"BTCUSDT\", \"interval\": \"1h\", \"limit\": 24}"
            )
        ]

        instruction = """\
You are a cryptocurrency trading analyst. Your goal is to maximize the USDT value of a simulated trading account.

Current Account Balance:
USDT: {usdt_balance}
BTC: {btc_balance}

You will be provided with the current BTC/USDT ticker price and recent candlestick data using your available tools.
Based on your analysis of the provided data and the current account balance, you must decide whether to BUY BTC, SELL BTC, or HOLD.

Decision Guidelines:
1.  If you decide to BUY BTC:
    *   Specify the amount of USDT to spend (e.g., 'amount_usdt_to_spend: 200').
    *   Try to use a reasonable portion of available USDT (e.g., 10% to 50%) if you identify a strong buying opportunity.
    *   Ensure `amount_btc_to_sell` is 0.
2.  If you decide to SELL BTC:
    *   Specify the amount of BTC to sell (e.g., 'amount_btc_to_sell: 0.01').
    *   Try to sell a reasonable portion of available BTC if you identify a strong selling opportunity.
    *   Ensure `amount_usdt_to_spend` is 0.
3.  If you decide to HOLD:
    *   Both `amount_usdt_to_spend` and `amount_btc_to_sell` should be 0.
    *   This is appropriate if no clear opportunity is present or the market is too volatile without clear direction.

Reasoning:
You MUST provide a clear and concise reason for your decision, based on your analysis of the market data (candlestick patterns, trends, support/resistance levels, significant price movements).

Output Format:
Your final decision MUST be structured as a JSON object conforming to the following schema:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "amount_usdt_to_spend": float (default 0.0),
  "amount_btc_to_sell": float (default 0.0),
  "reason": "Your concise analysis and reasoning for the decision."
}}

Example Decisions:
- Buying: {{"action": "BUY", "amount_usdt_to_spend": 200.0, "amount_btc_to_sell": 0.0, "reason": "Price bounced off support at 60000 and shows upward momentum based on candlestick analysis."}}
- Selling: {{"action": "SELL", "amount_usdt_to_spend": 0.0, "amount_btc_to_sell": 0.01, "reason": "Candlestick pattern (e.g., bearish engulfing) indicates a potential reversal after a strong rally to resistance at 65000."}}
- Holding: {{"action": "HOLD", "amount_usdt_to_spend": 0.0, "amount_btc_to_sell": 0.0, "reason": "Market is consolidating within a tight range (61000-62000), no clear entry or exit signal."}}

The system will execute valid BUY or SELL trades on your behalf using a simulated account and will inform you of the outcome. Focus on maximizing the overall USDT value of the account.
Always use your tools to get the latest market data before making a decision.
"""
        # The instruction will be formatted with current balances before each LLM call
        # in the _run_async_impl or a suitable callback.

        super().__init__(
            instruction=instruction, # This will be dynamically formatted
            tools=tools,
            model=model_name,
            output_schema=TradingDecision,
            response_type=BaseAgentResponse[TradingDecision]
        )

    async def _run_async_impl(
        self,
        request: LlmRequest,
        execution_context: Any | None = None
    ) -> AsyncGenerator[BaseAgentResponse[TradingDecision], None]:
        """
        Core logic for the TradingAgent.
        This method is called by the LlmAgent's default flow.
        It will format the instruction with current balance, let the LLM make a decision,
        then process that decision.
        """
        
        # 1. Format instruction with current balance
        current_balances = self.trading_account.get_balance()
        formatted_instruction = self.instruction.format(
            usdt_balance=current_balances["usdt_balance"],
            btc_balance=current_balances["btc_balance"]
        )
        # Update the instruction in the request for the LLM.
        # Note: LlmAgent's internal request might need specific handling
        # or use a before_model_callback to achieve this.
        # For simplicity, we'll assume the LlmAgent uses the latest self.instruction.
        # A more robust way is to modify the LlmRequest directly if possible, or use a callback.
        
        # Store the original instruction template
        original_instruction_template = self.instruction
        self.instruction = formatted_instruction # Temporarily set for this run

        # 2. Let the LlmAgent's default flow handle tool use and LLM interaction
        # The LlmAgent's `run_async` will call `_run_llm_async` which uses `self.instruction`.
        llm_decision_response: TradingDecision | None = None
        
        async for event in super()._run_async_impl(request, execution_context):
            yield event # Yield all events from the parent class (tool calls, LLM responses)
            if event.response_type == "llm_response" and event.response_data:
                # Assuming the final LLM response data is the TradingDecision
                # This might need adjustment based on how LlmAgent structures its final output event
                if isinstance(event.response_data, TradingDecision):
                     llm_decision_response = event.response_data
                elif isinstance(event.response_data, dict) and "action" in event.response_data : # If it's a dict from LLM
                    try:
                        llm_decision_response = TradingDecision(**event.response_data)
                    except Exception:
                        # LLM output was not a valid TradingDecision
                        # This case should ideally be handled by LlmAgent's retry/error mechanisms
                        # or by yielding a specific error event.
                        # For now, we'll just note it and proceed without action.
                        yield BaseAgentResponse(
                            response_type="agent_error",
                            response_data={"error": "LLM output did not conform to TradingDecision schema after tool use.", "raw_output": event.response_data},
                            raw_request=request,
                            raw_response=None 
                        )
                        llm_decision_response = None # Ensure it's None
                        break 

        # Restore original instruction template
        self.instruction = original_instruction_template

        # 3. Process the LLM's final decision
        if llm_decision_response:
            decision = llm_decision_response
            action_result_message = f"LLM Decision: {decision.action}. Reason: {decision.reason}"
            trade_executed = False
            
            current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

            if decision.action == "BUY":
                if decision.amount_usdt_to_spend > 0:
                    # For executing a buy, we need the current price.
                    # The LLM might have fetched it, but it could be slightly stale.
                    # For simulation, we could assume the LLM provides the price it used,
                    # or the agent re-fetches. Let's assume re-fetching for accuracy.
                    ticker_data = self.binance_data_tool.get_ticker_price(symbol="BTCUSDT")
                    if "error" not in ticker_data:
                        current_btc_price = float(ticker_data["price"])
                        buy_outcome = self.trading_account.execute_buy_order(
                            symbol="BTCUSDT",
                            usdt_amount_to_spend=decision.amount_usdt_to_spend,
                            current_btc_price=current_btc_price,
                            reason=decision.reason,
                            timestamp=current_time
                        )
                        action_result_message += f"\nBuy Order Outcome: {buy_outcome['message']}"
                        trade_executed = buy_outcome["success"]
                    else:
                        action_result_message += f"\nCould not execute BUY: Failed to fetch current price - {ticker_data['error']}"
                else:
                    action_result_message += "\nBuy action chosen, but amount_usdt_to_spend was zero. No trade executed."
            
            elif decision.action == "SELL":
                if decision.amount_btc_to_sell > 0:
                    ticker_data = self.binance_data_tool.get_ticker_price(symbol="BTCUSDT")
                    if "error" not in ticker_data:
                        current_btc_price = float(ticker_data["price"])
                        sell_outcome = self.trading_account.execute_sell_order(
                            symbol="BTCUSDT",
                            btc_amount_to_sell=decision.amount_btc_to_sell,
                            current_btc_price=current_btc_price,
                            reason=decision.reason,
                            timestamp=current_time
                        )
                        action_result_message += f"\nSell Order Outcome: {sell_outcome['message']}"
                        trade_executed = sell_outcome["success"]
                    else:
                        action_result_message += f"\nCould not execute SELL: Failed to fetch current price - {ticker_data['error']}"
                else:
                    action_result_message += "\nSell action chosen, but amount_btc_to_sell was zero. No trade executed."
            
            elif decision.action == "HOLD":
                action_result_message += "\nHolding position. No trade executed."

            # Yield a final agent response summarizing the action taken
            final_balances = self.trading_account.get_balance()
            action_result_message += f"\nFinal Balances: USDT: {final_balances['usdt_balance']:.2f}, BTC: {final_balances['btc_balance']:.8f}"
            
            yield BaseAgentResponse(
                response_type="agent_action_summary", # Custom type to indicate agent's action summary
                response_data={
                    "llm_decision": decision.model_dump(),
                    "action_result": action_result_message,
                    "trade_executed": trade_executed,
                    "final_balances": final_balances
                },
                raw_request=request, # The initial request that triggered this run
                raw_response=llm_decision_response # The raw LLM decision object
            )
        else:
            # This case occurs if the LLM flow completed without a parsable TradingDecision
            # (e.g., if max_steps reached, or LLM failed to output JSON after retries)
            yield BaseAgentResponse(
                response_type="agent_error",
                response_data={"error": "LLM did not produce a final TradingDecision."},
                raw_request=request,
                raw_response=None 
            )

    # To make the dynamic instruction formatting more robust within LlmAgent's flow,
    # one would typically use `before_llm_request_callback` or override `_create_llm_request`.
    # For example:
    #
    # def before_llm_request_callback(self, request: LlmRequest) -> LlmRequest:
    #     current_balances = self.trading_account.get_balance()
    #     # Assuming self.instruction is the template string store elsewhere or the initial one
    #     base_instruction = super().instruction # Or load template
    #     request.instruction = base_instruction.format(
    #         usdt_balance=current_balances["usdt_balance"],
    #         btc_balance=current_balances["btc_balance"]
    #     )
    #     return request

    # However, for this subtask, the _run_async_impl override demonstrates the intent.
    # The LlmAgent's instruction is typically set at init. Modifying it per-run
    # as done above is a simplification. A callback is cleaner.
    # Let's assume the LlmAgent will pick up the modified `self.instruction` for now.

```
