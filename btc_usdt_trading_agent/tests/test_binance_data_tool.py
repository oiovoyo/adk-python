import pytest
from binance.client import Client # For monkeypatching Client.ping and other methods
from binance.exceptions import BinanceAPIException, BinanceRequestException

from btc_usdt_trading_agent.tools.binance_data_tool import BinanceDataTool

# No BINANCE_API_URL needed if we monkeypatch client methods directly

@pytest.fixture
def tool(monkeypatch):
    """
    Fixture to create an instance of BinanceDataTool.
    Client.ping is monkeypatched to avoid real network calls during instantiation.
    """
    def mock_ping_dummy(*args, **kwargs):
        return {} 
    monkeypatch.setattr(Client, "ping", mock_ping_dummy)
    
    tool_instance = BinanceDataTool(api_key="test_key", api_secret="test_secret")
    return tool_instance

# --- Test Cases for get_ticker_price ---

def test_get_ticker_price_success(tool: BinanceDataTool, monkeypatch):
    """Test successful fetching of a ticker price."""
    symbol = "BTCUSDT"
    expected_raw_response = {"symbol": "BTCUSDT", "price": "65000.50"}
    
    def mock_get_symbol_ticker(sym): # Renamed arg to avoid confusion
        if sym == symbol:
            return expected_raw_response
        raise ValueError("Mock not configured for this symbol") # Should not happen
    monkeypatch.setattr(tool._client, "get_symbol_ticker", mock_get_symbol_ticker)
    
    result = tool.get_ticker_price(symbol=symbol)
    assert result == {"symbol": "BTCUSDT", "price": "65000.50"}

def test_get_ticker_price_api_error(tool: BinanceDataTool, monkeypatch):
    """Test API error (e.g., invalid symbol) for get_ticker_price."""
    symbol_to_test = "INVALIDUSDT"
    
    def mock_get_symbol_ticker_api_error(sym):
        if sym == symbol_to_test:
            # Correctly simulate how BinanceAPIException gets its message
            # The 'text' field usually contains the JSON string with 'msg' and 'code'
            error_json_text = '{"code": -1121, "msg": "Invalid symbol error from mock"}' 
            raise BinanceAPIException(response=None, status_code=400, text=error_json_text)
        raise ValueError("Mock not configured for this symbol")
    monkeypatch.setattr(tool._client, "get_symbol_ticker", mock_get_symbol_ticker_api_error)
    
    result = tool.get_ticker_price(symbol=symbol_to_test)
    assert "error" in result
    # BinanceAPIException populates its .message from the 'msg' in the JSON text
    assert result["error"] == f"Binance API exception for {symbol_to_test}: Invalid symbol error from mock"

def test_get_ticker_price_request_error(tool: BinanceDataTool, monkeypatch):
    """Test request error (e.g., network issue) for get_ticker_price."""
    symbol_to_test = "BTCUSDT"
    
    def mock_get_symbol_ticker_request_error(sym):
        if sym == symbol_to_test:
            raise BinanceRequestException("Simulated network error")
        raise ValueError("Mock not configured for this symbol")
    monkeypatch.setattr(tool._client, "get_symbol_ticker", mock_get_symbol_ticker_request_error)
        
    result = tool.get_ticker_price(symbol=symbol_to_test)
    assert "error" in result
    assert result["error"] == f"Binance request exception for {symbol_to_test}: Simulated network error"

def test_get_ticker_price_unexpected_error(tool: BinanceDataTool, monkeypatch):
    """Test unexpected non-Binance error during get_ticker_price."""
    symbol_to_test = "BTCUSDT"
    def mock_get_symbol_ticker_unexpected(*args, **kwargs):
        raise ValueError("Something broke unexpectedly internally")
    monkeypatch.setattr(tool._client, "get_symbol_ticker", mock_get_symbol_ticker_unexpected)
    
    result = tool.get_ticker_price(symbol=symbol_to_test)
    assert "error" in result
    assert result["error"] == f"An unexpected error occurred while fetching ticker price for {symbol_to_test}: Something broke unexpectedly internally"

# --- Test Cases for get_candlestick_data ---

def test_get_candlestick_data_success(tool: BinanceDataTool, monkeypatch):
    """Test successful fetching and transformation of candlestick data."""
    symbol_to_test = "ETHUSDT"
    interval_to_test = "1h"
    limit_to_test = 2
    
    mock_raw_klines_response = [
        [1672531200000, "1200.00", "1205.00", "1198.00", "1202.50", "1500.50", 1672534799999, "1800600.25", 100, "750.00", "900300.00", "0"],
        [1672534800000, "1202.50", "1210.00", "1201.00", "1208.00", "1800.75", 1672538399999, "2172906.00", 120, "900.00", "1087200.00", "0"]
    ]
    def mock_get_klines(symbol, interval, limit):
        if symbol == symbol_to_test and interval == interval_to_test and limit == limit_to_test:
            return mock_raw_klines_response
        raise ValueError("Mock not configured for these kline parameters")
    monkeypatch.setattr(tool._client, "get_klines", mock_get_klines)

    result = tool.get_candlestick_data(symbol=symbol_to_test, interval=interval_to_test, limit=limit_to_test)

    assert result["symbol"] == symbol_to_test
    assert len(result["klines"]) == 2
    kline1 = result["klines"][0]
    assert kline1["open_time"] == "2023-01-01T00:00:00"
    assert kline1["open"] == "1200.00"
    # ... (other assertions for kline1 can be added if needed)
    assert kline1["close_time"] == "2023-01-01T00:59:59.999000"
    kline2 = result["klines"][1]
    assert kline2["open_time"] == "2023-01-01T01:00:00"
    assert kline2["close"] == "1208.00"

def test_get_candlestick_data_empty(tool: BinanceDataTool, monkeypatch):
    """Test fetching candlestick data when API returns an empty list."""
    symbol_to_test = "ADAUSDT"
    def mock_get_klines_empty(symbol, interval, limit):
        if symbol == symbol_to_test:
            return []
        raise ValueError("Mock not configured for these kline parameters")
    monkeypatch.setattr(tool._client, "get_klines", mock_get_klines_empty)
    
    result = tool.get_candlestick_data(symbol=symbol_to_test, interval="1d", limit=5)
    assert result["symbol"] == symbol_to_test
    assert result["klines"] == []

def test_get_candlestick_data_api_error(tool: BinanceDataTool, monkeypatch):
    """Test API error for get_candlestick_data."""
    symbol_to_test = "BTCUSDT"
    def mock_get_klines_api_error(symbol, interval, limit):
        if symbol == symbol_to_test:
            error_json_text = '{"code": -1003, "msg": "Klines API error from mock"}'
            raise BinanceAPIException(response=None, status_code=400, text=error_json_text)
        raise ValueError("Mock not configured for these kline parameters")
    monkeypatch.setattr(tool._client, "get_klines", mock_get_klines_api_error)

    result = tool.get_candlestick_data(symbol=symbol_to_test, interval="1w", limit=100)
    assert "error" in result
    assert result["error"] == f"Binance API exception for {symbol_to_test} klines: Klines API error from mock"

def test_get_candlestick_data_request_error(tool: BinanceDataTool, monkeypatch):
    """Test request error for get_candlestick_data."""
    symbol_to_test = "LINKUSDT"
    def mock_get_klines_request_error(symbol, interval, limit):
        if symbol == symbol_to_test:
            raise BinanceRequestException("Simulated klines network error")
        raise ValueError("Mock not configured for these kline parameters")
    monkeypatch.setattr(tool._client, "get_klines", mock_get_klines_request_error)

    result = tool.get_candlestick_data(symbol=symbol_to_test, interval="1h", limit=24)
    assert "error" in result
    assert result["error"] == f"Binance request exception for {symbol_to_test} klines: Simulated klines network error"

def test_get_candlestick_data_unexpected_error(tool: BinanceDataTool, monkeypatch):
    """Test unexpected non-Binance error during get_candlestick_data."""
    symbol_to_test = "ETHUSDT"
    def mock_get_klines_unexpected(*args, **kwargs):
        raise TypeError("Unexpected kline data format internally")
    monkeypatch.setattr(tool._client, "get_klines", mock_get_klines_unexpected)

    result = tool.get_candlestick_data(symbol=symbol_to_test) # Uses default interval and limit
    assert "error" in result
    assert result["error"] == f"An unexpected error occurred while fetching klines for {symbol_to_test}: Unexpected kline data format internally"

if __name__ == "__main__":
    pytest.main()
