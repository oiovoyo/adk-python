import os
from datetime import datetime
import logging # Added

from binance.client import Client
try:
    from binance.exceptions import BinanceAPIException, BinanceRequestException
except ImportError:
    BinanceAPIException = None 
    BinanceRequestException = None

logger = logging.getLogger("TradingAgentRunner.BinanceDataTool") # Added

class BinanceDataTool:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self._client = Client(api_key, api_secret)
        logger.debug("BinanceDataTool initialized.") # Added as per implied requirement

    def get_ticker_price(self, symbol: str) -> dict:
        logger.debug(f"Fetching ticker price for symbol: {symbol}")
        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            logger.debug(f"Successfully fetched ticker for {symbol}: Price {ticker['price']}")
            return {"symbol": ticker["symbol"], "price": ticker["price"]}
        except BinanceAPIException as e:
            logger.warning(f"Binance API exception for {symbol} ticker: {e.message}")
            return {"error": f"Binance API exception for {symbol}: {e.message}"}
        except BinanceRequestException as e:
            logger.warning(f"Binance request exception for {symbol} ticker: {e.message}")
            return {"error": f"Binance request exception for {symbol}: {e.message}"}
        except Exception as e:
            logger.warning(f"Failed to fetch ticker price for {symbol}: {e}", exc_info=True) # Added exc_info
            return {"error": f"An unexpected error occurred while fetching ticker price for {symbol}: {str(e)}"}

    def get_candlestick_data(self, symbol: str, interval: str = '1h', limit: int = 24) -> dict:
        logger.debug(f"Fetching candlestick data for symbol: {symbol}, interval: {interval}, limit: {limit}")
        try:
            raw_klines = self._client.get_klines(symbol=symbol, interval=interval, limit=limit)
            logger.debug(f"Successfully fetched {len(raw_klines)} klines for {symbol} with interval {interval}.")
            
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
                })
            
            return {"symbol": symbol, "klines": formatted_klines}

        except BinanceAPIException as e:
            logger.warning(f"Binance API exception for {symbol} klines (interval {interval}): {e.message}")
            return {"error": f"Binance API exception for {symbol} klines: {e.message}"}
        except BinanceRequestException as e:
            logger.warning(f"Binance request exception for {symbol} klines (interval {interval}): {e.message}")
            return {"error": f"Binance request exception for {symbol} klines: {e.message}"}
        except Exception as e:
            logger.warning(f"Failed to fetch candlestick data for {symbol} (interval {interval}): {e}", exc_info=True) # Added exc_info
            return {"error": f"An unexpected error occurred while fetching klines for {symbol}: {str(e)}"}

```
