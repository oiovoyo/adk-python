import datetime
from typing import Literal, AsyncGenerator, Any

from google_adk.agents import LlmAgent, BaseAgentResponse, LlmRequest
from google_adk.tools import FunctionTool
from pydantic import BaseModel, Field

from btc_usdt_trading_agent.tools.binance_data_tool import BinanceDataTool
from btc_usdt_trading_agent.account.trading_account import TradingAccount
from btc_usdt_trading_agent.tools.news_fetch_tool import get_crypto_news # Import the new tool


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
        self.binance_data_tool = BinanceDataTool() 

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
            ),
            FunctionTool( # New NewsFetchTool
                func=get_crypto_news, # Directly use the imported async function
                name="get_cryptocurrency_news",
                description="Fetches recent news articles related to a specific cryptocurrency or market trend. Use queries like 'bitcoin price sentiment', 'ethereum new projects', or 'overall crypto market trends'. Example: {\"query\": \"bitcoin price sentiment\", \"num_results\": 3}"
            )
        ]

        # Store the instruction template. It will be formatted dynamically.
        self._original_instruction = """\
You are a cryptocurrency trading analyst. Your goal is to maximize the USDT value of a simulated trading account.

Current Account Balance:
USDT: {usdt_balance}
BTC: {btc_balance}

You have access to the following tools to gather market data:
1. `get_ticker_price(symbol: str)`: Fetches the latest price for a cryptocurrency symbol.
2. `get_candlestick_data(symbol: str, interval: str = '1h', limit: int = 24)`: Fetches recent candlestick (k-line) data.
3. `get_cryptocurrency_news(query: str, num_results: int = 3)`: Fetches recent news articles. Use this to understand market sentiment or find news that might impact prices (e.g., query 'bitcoin price sentiment' or 'ethereum new regulations').

Based on your analysis of all available data (ticker price, candlestick patterns, and relevant news), you must decide whether to BUY BTC, SELL BTC, or HOLD.

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
You MUST provide a clear and concise reason for your decision, based on your analysis of the market data (candlestick patterns, trends, support/resistance levels, significant price movements, and news sentiment).

Output Format:
Your final decision MUST be structured as a JSON object conforming to the following schema:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "amount_usdt_to_spend": float (default 0.0),
  "amount_btc_to_sell": float (default 0.0),
  "reason": "Your concise analysis and reasoning for the decision."
}}

Example Decisions:
- Buying: {{"action": "BUY", "amount_usdt_to_spend": 200.0, "amount_btc_to_sell": 0.0, "reason": "Price bounced off support at 60000, RSI is oversold, and recent positive news about Bitcoin adoption suggests upward potential."}}
- Selling: {{"action": "SELL", "amount_usdt_to_spend": 0.0, "amount_btc_to_sell": 0.01, "reason": "Bearish engulfing pattern on 1h chart at resistance 65000, coupled with negative regulatory news."}}
- Holding: {{"action": "HOLD", "amount_usdt_to_spend": 0.0, "amount_btc_to_sell": 0.0, "reason": "Market is consolidating (61000-62000), news is mixed, awaiting clearer signal."}}

Always use your tools to get the latest market data before making a decision. Consider all available data (price, candlesticks, news) for a comprehensive analysis.
"""
        # The instruction passed to super() will be formatted dynamically in _run_async_impl or a callback.
        # For now, pass the template; LlmAgent expects an instruction string.
        super().__init__(
            instruction=self._original_instruction, # Will be formatted before each LLM call
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
        Formats instruction with current balance, lets LLM make a decision, then processes it.
        """
        current_balances = self.trading_account.get_balance()
        formatted_instruction = self._original_instruction.format(
            usdt_balance=f"{current_balances['usdt_balance']:.2f}", # Format for readability
            btc_balance=f"{current_balances['btc_balance']:.8f}"  # Format for readability
        )
        
        # Create a new LlmRequest with the formatted instruction for this specific run.
        # This is a cleaner way than modifying self.instruction directly.
        # The LlmAgent's run_async flow should ideally use the instruction from the request object.
        # We assume that the LlmAgent uses request.instruction if provided,
        # or falls back to self.instruction. If `request` is directly mutated, it works.
        # If not, LlmAgent might need a specific way to update instruction per run.
        
        # For simplicity in this context, we'll rely on the LlmAgent using its `self.instruction`.
        # We'll set it for this run and then restore it.
        # A more robust ADK pattern would use a callback like `before_llm_request_callback`
        # to modify the instruction in the LlmRequest just before the API call.
        
        original_agent_instruction = self.instruction # Save agent's base instruction
        self.instruction = formatted_instruction     # Set dynamically formatted one for this run

        llm_decision_response: TradingDecision | None = None
        
        # The super()._run_async_impl will use the tools and the (now dynamically formatted) self.instruction
        async for event in super()._run_async_impl(request, execution_context):
            yield event
            if event.response_type == "llm_response" and event.response_data:
                if isinstance(event.response_data, TradingDecision):
                     llm_decision_response = event.response_data
                elif isinstance(event.response_data, dict) and "action" in event.response_data:
                    try:
                        llm_decision_response = TradingDecision(**event.response_data)
                    except Exception:
                        yield BaseAgentResponse(
                            response_type="agent_error",
                            response_data={"error": "LLM output did not conform to TradingDecision schema after tool use.", "raw_output": event.response_data},
                            raw_request=request,
                            raw_response=None 
                        )
                        llm_decision_response = None
                        break 

        self.instruction = original_agent_instruction # Restore original instruction template

        if llm_decision_response:
            decision = llm_decision_response
            action_result_message = f"LLM Decision: {decision.action}. Reason: {decision.reason}"
            trade_executed = False
            current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

            if decision.action == "BUY":
                if decision.amount_usdt_to_spend > 0:
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

            final_balances = self.trading_account.get_balance()
            action_result_message += f"\nFinal Balances: USDT: {final_balances['usdt_balance']:.2f}, BTC: {final_balances['btc_balance']:.8f}"
            
            yield BaseAgentResponse(
                response_type="agent_action_summary",
                response_data={
                    "llm_decision": decision.model_dump(),
                    "action_result": action_result_message,
                    "trade_executed": trade_executed,
                    "final_balances": final_balances
                },
                raw_request=request,
                raw_response=llm_decision_response
            )
        else:
            yield BaseAgentResponse(
                response_type="agent_error",
                response_data={"error": "LLM did not produce a final TradingDecision."},
                raw_request=request,
                raw_response=None 
            )

    # A more robust way for dynamic instruction formatting using ADK's LlmAgent:
    # def before_llm_request_callback(self, request: LlmRequest) -> LlmRequest:
    #     """Dynamically formats the instruction with current account balances."""
    #     current_balances = self.trading_account.get_balance()
    #     # Ensure self._original_instruction holds the template with placeholders
    #     request.instruction = self._original_instruction.format(
    #         usdt_balance=f"{current_balances['usdt_balance']:.2f}",
    #         btc_balance=f"{current_balances['btc_balance']:.8f}"
    #     )
    #     return request

```
