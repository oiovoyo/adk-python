import pytest
from unittest.mock import patch, AsyncMock
from btc_usdt_trading_agent.tools.news_fetch_tool import get_crypto_news

# Since get_crypto_news is in news_fetch_tool.py, we patch CRAWLER and CRAWL4AI_AVAILABLE there.
NEWS_FETCH_TOOL_PATH = 'btc_usdt_trading_agent.tools.news_fetch_tool'

@pytest.mark.asyncio
async def test_get_crypto_news_success():
    """Scenario 1: Successful News Fetching with title/snippet fallbacks."""
    mock_docs = [
        {"url": "http://example.com/news1", "title": "Bitcoin Hits New High", "description": "Bitcoin reached a new all-time high today...", "content": "Longer content of news 1..."},
        {"url": "http://example.com/news2", "title": "Ethereum Update Soon", "content": "Longer content of news 2, used for snippet..."}, # No description
        {"url": "http://example.com/news3", "content": "A news item with no title just content, also for snippet..."}, # No title, no description
        # This one should be limited by num_results=2 if not for the title fallback test
        {"url": "http://example.com/news4", "title": "Fourth News", "description": "Description for fourth news.", "content": "Content for fourth news."},
    ]
    
    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.run.return_value = mock_docs
    
    with patch(f'{NEWS_FETCH_TOOL_PATH}.Crawler', return_value=mock_crawler_instance) as mock_crawler_class:
        with patch(f'{NEWS_FETCH_TOOL_PATH}.CRAWL4AI_AVAILABLE', True):
            result = await get_crypto_news(query="bitcoin", num_results=2) # Request 2 results

            assert "query" in result
            assert result["query"] == "bitcoin"
            assert "news_items" in result
            assert len(result["news_items"]) == 2 # Should honor num_results

            # Check first item (full details)
            item1 = result["news_items"][0]
            assert item1["title"] == "Bitcoin Hits New High"
            assert item1["snippet"] == "Bitcoin reached a new all-time high today..."
            assert item1["source_url"] == "http://example.com/news1"

            # Check second item (snippet from content)
            item2 = result["news_items"][1]
            assert item2["title"] == "Ethereum Update Soon"
            assert item2["snippet"].startswith("Longer content of news 2, used for snippet...")
            assert item2["source_url"] == "http://example.com/news2"
            
            mock_crawler_instance.run.assert_called_once_with(
                query="bitcoin cryptocurrency news",
                max_pages=2 * 2, # num_results * 2
                output_format='documents'
            )

@pytest.mark.asyncio
async def test_get_crypto_news_title_snippet_fallbacks_and_filtering():
    """Scenario 5: Detailed filtering (no URL) and snippet/title generation logic."""
    mock_docs = [
        {"url": "http://example.com/full", "title": "Full Item", "description": "Full description.", "content": "Full content."},
        {"url": None, "title": "No URL Item", "description": "This item has no URL.", "content": "Content for no URL."}, # Should be filtered out
        {"url": "http://example.com/no_desc", "title": "No Description Item", "content": "Content for no description, used for snippet."},
        {"url": "http://example.com/no_title_no_desc", "content": "Content for no title and no description, used for both."},
        {"url": "http://example.com/only_url_title", "title": "Only URL and Title"}, # No content, no description
    ]
    
    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.run.return_value = mock_docs
    
    with patch(f'{NEWS_FETCH_TOOL_PATH}.Crawler', return_value=mock_crawler_instance):
        with patch(f'{NEWS_FETCH_TOOL_PATH}.CRAWL4AI_AVAILABLE', True):
            result = await get_crypto_news(query="fallbacks", num_results=3) # Ask for 3, expect 3 valid ones

            assert "news_items" in result
            assert len(result["news_items"]) == 3 # Item with no URL is filtered out

            # Item 1 (Full)
            assert result["news_items"][0]["title"] == "Full Item"
            assert result["news_items"][0]["snippet"] == "Full description."
            assert result["news_items"][0]["source_url"] == "http://example.com/full"

            # Item 2 (No Description)
            assert result["news_items"][1]["title"] == "No Description Item"
            assert result["news_items"][1]["snippet"].startswith("Content for no description, used for snippet.")
            assert result["news_items"][1]["source_url"] == "http://example.com/no_desc"
            
            # Item 3 (No Title, No Description)
            assert result["news_items"][2]["title"].startswith("Content for no title and no description, used for both.") # Title from content
            assert result["news_items"][2]["snippet"].startswith("Content for no title and no description, used for both.") # Snippet from content
            assert result["news_items"][2]["source_url"] == "http://example.com/no_title_no_desc"

            # Item with "Only URL and Title" (no content, no desc) will have "No snippet available."
            # but it won't be in the top 3 due to the order and num_results=3.
            # To test it specifically, we could ask for num_results=4 or change order.

@pytest.mark.asyncio
async def test_get_crypto_news_no_documents_returned():
    """Scenario 2: crawl4ai returns no documents."""
    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.run.return_value = [] # Empty list
    
    with patch(f'{NEWS_FETCH_TOOL_PATH}.Crawler', return_value=mock_crawler_instance):
        with patch(f'{NEWS_FETCH_TOOL_PATH}.CRAWL4AI_AVAILABLE', True):
            result = await get_crypto_news(query="unknowncoin")
            assert "error" in result
            assert result["error"] == "No relevant news found for the query."

@pytest.mark.asyncio
async def test_get_crypto_news_crawl_exception():
    """Scenario 3: crawl4ai throws an exception."""
    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.run.side_effect = Exception("Crawl failed spectacularly")
    
    with patch(f'{NEWS_FETCH_TOOL_PATH}.Crawler', return_value=mock_crawler_instance):
        with patch(f'{NEWS_FETCH_TOOL_PATH}.CRAWL4AI_AVAILABLE', True):
            result = await get_crypto_news(query="bitcoin")
            assert "error" in result
            assert "Failed to fetch or process news: Crawl failed spectacularly" in result["error"]

@pytest.mark.asyncio
async def test_get_crypto_news_crawl4ai_not_installed():
    """Scenario 4: crawl4ai not installed."""
    with patch(f'{NEWS_FETCH_TOOL_PATH}.CRAWL4AI_AVAILABLE', False):
        result = await get_crypto_news(query="bitcoin")
        assert "error" in result
        assert result["error"] == "crawl4ai library is not available. News fetching is disabled."

@pytest.mark.asyncio
async def test_get_crypto_news_no_suitable_items_after_filtering():
    """Test case where documents are returned but none are suitable (e.g., all missing URLs)."""
    mock_docs = [
        {"title": "News without URL", "description": "This is a valid news item but has no URL.", "content": "Content here."},
        {"url": None, "title": "Another News without URL", "content": "More content."}
    ]
    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.run.return_value = mock_docs
    
    with patch(f'{NEWS_FETCH_TOOL_PATH}.Crawler', return_value=mock_crawler_instance):
        with patch(f'{NEWS_FETCH_TOOL_PATH}.CRAWL4AI_AVAILABLE', True):
            result = await get_crypto_news(query="bitcoin", num_results=2)
            assert "error" in result
            assert result["error"] == "Could not extract suitable news items from crawled pages."

@pytest.mark.asyncio
async def test_get_crypto_news_title_fallback_from_url_if_no_content_title():
    """Test title fallback to URL if content and title are missing."""
    mock_docs = [
        {"url": "http://example.com/url_only_news"} # No title, no description, no content
    ]
    mock_crawler_instance = AsyncMock()
    mock_crawler_instance.run.return_value = mock_docs
    
    with patch(f'{NEWS_FETCH_TOOL_PATH}.Crawler', return_value=mock_crawler_instance):
        with patch(f'{NEWS_FETCH_TOOL_PATH}.CRAWL4AI_AVAILABLE', True):
            result = await get_crypto_news(query="bitcoin", num_results=1)
            assert "news_items" in result
            assert len(result["news_items"]) == 1
            item = result["news_items"][0]
            assert item["title"] == "http://example.com/url_only_news" # Title falls back to URL
            assert item["snippet"] == "No snippet available."
            assert item["source_url"] == "http://example.com/url_only_news"

```
