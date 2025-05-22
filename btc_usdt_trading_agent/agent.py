import datetime
from typing import Literal, AsyncGenerator, Any

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
            FunctionTool( 
                func=get_crypto_news, 
                name="get_cryptocurrency_news",
                description="Fetches recent news articles related to a specific cryptocurrency or market trend. Use queries like 'bitcoin price sentiment', 'ethereum new projects', or 'overall crypto market trends'. Example: {\"query\": \"bitcoin price sentiment\", \"num_results\": 3}"
            ),
            FunctionTool( # New Account Position Tool
                func=self._get_current_account_position,
                name="get_current_account_position",
                description="Fetches the current balances of USDT and BTC in your trading account."
            ),
            FunctionTool( # New Trade History Tool
                func=self._get_recent_trade_history,
                name="get_recent_trade_history",
                description="Fetches the last N simulated trade transactions. You can specify N (e.g., num_trades=5)."
            )
        ]

        self._original_instruction = """\
You are a cryptocurrency trading analyst. Your goal is to maximize the USDT value of a simulated trading account.

Before making any trading decision, you should first understand your current financial position and recent trading activity. Use these tools:
1. `get_current_account_position()`: Fetches your current USDT and BTC balances.
2. `get_recent_trade_history(num_trades: int = 5)`: Fetches your last N trades.

Then, analyze the market using:
3. `get_ticker_price(symbol: str)`: Fetches the latest price for a cryptocurrency symbol.
4. `get_candlestick_data(symbol: str, interval: str = '1h', limit: int = 24)`: Fetches recent candlestick (k-line) data.
5. `get_cryptocurrency_news(query: str, num_results: int = 3)`: Fetches recent news articles. Use this to understand market sentiment or find news that might impact prices (e.g., query 'bitcoin price sentiment' or 'ethereum new regulations').

Based on your analysis of all available data (your current position, past trades, ticker price, candlestick patterns, and relevant news), you must decide whether to BUY BTC, SELL BTC, or HOLD.

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
You MUST provide a clear and concise reason for your decision, based on your analysis of all available data.

Output Format:
Your final decision MUST be structured as a JSON object conforming to the following schema:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "amount_usdt_to_spend": float (default 0.0),
  "amount_btc_to_sell": float (default 0.0),
  "reason": "Your concise analysis and reasoning for the decision."
}}

Example Decisions:
- Buying: {{"action": "BUY", "amount_usdt_to_spend": 200.0, "amount_btc_to_sell": 0.0, "reason": "Account has 1000 USDT. Bitcoin price is at support 60000, RSI is oversold, and recent positive news about Bitcoin adoption suggests upward potential."}}
- Selling: {{"action": "SELL", "amount_usdt_to_spend": 0.0, "amount_btc_to_sell": 0.01, "reason": "Account has 0.5 BTC. Bearish engulfing pattern on 1h chart at resistance 65000, coupled with negative regulatory news."}}
- Holding: {{"action": "HOLD", "amount_usdt_to_spend": 0.0, "amount_btc_to_sell": 0.0, "reason": "Market is consolidating (61000-62000), news is mixed, current BTC holding is minimal, awaiting clearer signal."}}

Always use your tools to get the latest market data and account status before making a decision.
"""
        super().__init__(
            instruction=self._original_instruction, # This is now the static template
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
        The instruction is now static; LLM uses tools to get dynamic info.
        """
        # The dynamic instruction formatting block is removed.
        # self.instruction is already set to self._original_instruction in __init__.
        # If LlmAgent requires the instruction to be part of the request object for each run,
        # then the calling code (or a callback like before_llm_request_callback)
        # would be responsible for populating request.instruction.
        # For this implementation, we assume LlmAgent uses self.instruction.

        llm_decision_response: TradingDecision | None = None
        
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
        # No need to restore self.instruction as it wasn't changed for this specific run.

        if llm_decision_response:
            decision = llm_decision_response
            action_result_message = f"LLM Decision: {decision.action}. Reason: {decision.reason}"
            trade_executed = False
            current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

            if decision.action == "BUY":
                if decision.amount_usdt_to_spend > 0:
                    # LLM should have used tools to get price, but for safety in simulation, re-fetch.
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

        Args:
            num_trades: The maximum number of recent trades to retrieve.
                        Defaults to 5. If non-positive, no trades are returned.

        Returns:
            A dictionary containing the list of recent trades and the count.
        """
        if num_trades <= 0:
            history_slice = []
        else:
            history_slice = self.trading_account.transaction_history[-num_trades:]
        
        return {
            "trade_history": history_slice,
            "trades_returned": len(history_slice)
        }

    # Example of using before_llm_request_callback for dynamic instructions
    # if self.instruction modification in _run_async_impl is not preferred/supported.
    # def before_llm_request_callback(self, request: LlmRequest) -> LlmRequest:
    #     """Dynamically formats the instruction if needed (not used in current setup)."""
    #     # This would be the place to inject dynamic info into request.instruction
    #     # if self.instruction passed to super() was a static template not containing balances.
    #     # However, current _original_instruction is now static and doesn't need balance formatting.
    #     return request
```
