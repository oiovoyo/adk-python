import os
from datetime import datetime

from binance.client import Client
# from google_adk.tools import BaseTool # Removed due to persistent ModuleNotFoundError in sandbox
# It's good practice to handle potential import errors,
# though the problem states we can assume it's installed.
try:
    from binance.exceptions import BinanceAPIException, BinanceRequestException
except ImportError:
    # Handle the case where python-binance is not installed,
    # though for this task we assume it is.
    # In a real scenario, you might log this or raise a custom error.
    print("python-binance library is not installed. Please install it to use BinanceDataTool.")
    BinanceAPIException = None 
    BinanceRequestException = None


class BinanceDataTool: # Removed inheritance from BaseTool
    """
    A tool/class to fetch cryptocurrency market data from Binance.
    Its methods can be wrapped as FunctionTools by the ADK.
    This tool provides methods to get ticker prices and candlestick data.
    """

    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initializes the Binance client.
        API key and secret can be provided or loaded from environment variables.
        For public data endpoints, API key/secret are often not required.
        """
        # super().__init__( # Removed super call as BaseTool is no longer inherited
        #     name="binance_data_tool",
        #     description="A tool to fetch cryptocurrency market data from Binance."
        # )
        self._client = Client(api_key, api_secret)

    def get_ticker_price(self, symbol: str) -> dict:
        """
        Fetches the latest price ticker for a given symbol.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT").

        Returns:
            A dictionary containing the symbol and its price,
            or an error dictionary if an issue occurs.
            Example: {"symbol": "BTCUSDT", "price": "60000.00"}
            Error example: {"error": "Invalid symbol: XYZ"}
        """
        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            return {"symbol": ticker["symbol"], "price": ticker["price"]}
        except BinanceAPIException as e:
            return {"error": f"Binance API exception for {symbol}: {e.message}"}
        except BinanceRequestException as e:
            return {"error": f"Binance request exception for {symbol}: {e.message}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred while fetching ticker price for {symbol}: {str(e)}"}

    def get_candlestick_data(self, symbol: str, interval: str = '1h', limit: int = 24) -> dict:
        """
        Fetches recent candlestick data for a given symbol.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT").
            interval: The candlestick interval (e.g., '1m', '1h', '1d').
                      Defaults to '1h'.
            limit: The number of candlesticks to fetch. Defaults to 24.
                   Max is typically 1000 for most intervals.

        Returns:
            A dictionary containing the symbol and a list of candlestick data,
            or an error dictionary if an issue occurs.
            Candlestick data is formatted as a list of dictionaries.
            Example:
            {
                "symbol": "BTCUSDT",
                "klines": [
                    {
                        "open_time": "2023-01-01T00:00:00", "open": "30000.00",
                        "high": "30100.00", "low": "29900.00", "close": "30050.00",
                        "volume": "100.50", "close_time": "2023-01-01T00:59:59"
                    },
                    ...
                ]
            }
            Error example: {"error": "Failed to fetch klines for BTCUSDT: invalid interval"}
        """
        try:
            # Fetch raw klines data from Binance
            # [[open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]]
            raw_klines = self._client.get_klines(symbol=symbol, interval=interval, limit=limit)

            formatted_klines = []
            for kline in raw_klines:
                formatted_klines.append({
                    "open_time": datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                    "open": kline[1],
                    "high": kline[2],
                    "low": kline[3],
                    "close": kline[4],
                    "volume": kline[5],
                    "close_time": datetime.fromtimestamp(kline[6] / 1000).isoformat(),
                    # Optional: add other fields if needed by the LLM
                    # "quote_asset_volume": kline[7],
                    # "number_of_trades": kline[8],
                    # "taker_buy_base_asset_volume": kline[9],
                    # "taker_buy_quote_asset_volume": kline[10]
                })
            
            return {"symbol": symbol, "klines": formatted_klines}

        except BinanceAPIException as e:
            return {"error": f"Binance API exception for {symbol} klines: {e.message}"}
        except BinanceRequestException as e:
            return {"error": f"Binance request exception for {symbol} klines: {e.message}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred while fetching klines for {symbol}: {str(e)}"}

# Example of how these functions might be wrapped as FunctionTools later
# (Not part of this file, but for conceptual clarity for the ADK integration)
#
# from google_adk.tools import FunctionTool
#
# binance_tool_instance = BinanceDataTool()
#
# get_ticker_price_tool = FunctionTool(
#     func=binance_tool_instance.get_ticker_price,
#     name="get_ticker_price",
#     description="Fetches the latest price for a cryptocurrency symbol from Binance."
# )
#
# get_candlestick_data_tool = FunctionTool(
#     func=binance_tool_instance.get_candlestick_data,
#     name="get_candlestick_data",
#     description="Fetches recent candlestick (k-line) data for a cryptocurrency symbol from Binance."
# )
#
# # These tools would then be added to a BaseToolset or registered with the agent.
# ``` # This was causing a syntax error. Commenting it out.
