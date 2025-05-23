import asyncio
import logging # Added

logger = logging.getLogger("TradingAgentRunner.NewsFetchTool") # Added

try:
    from crawl4ai import Crawler
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    class Crawler: # Placeholder
        async def run(self, **kwargs):
            logger.warning("crawl4ai.Crawler placeholder used as library is not available.")
            return []

async def get_crypto_news(query: str, num_results: int = 3) -> dict:
    """
    Fetches cryptocurrency news for a given query using crawl4ai.
    """
    logger.debug(f"Fetching crypto news for query: '{query}', num_results: {num_results}")
    if not CRAWL4AI_AVAILABLE:
        logger.warning("crawl4ai library not available. Cannot fetch news.")
        return {"error": "crawl4ai library is not available. News fetching is disabled."}

    try:
        crawler = Crawler()
        logger.debug(f"crawl4ai Crawler instantiated. Executing run for query: '{query} cryptocurrency news'")
        results = await crawler.run(
            query=f"{query} cryptocurrency news", 
            max_pages=num_results * 2, 
            output_format='documents'
        )

        if not results:
            logger.info(f"No news documents found by crawl4ai for query: '{query}' (effective query: '{query} cryptocurrency news')")
            return {"error": "No relevant news found for the query."}
        
        logger.debug(f"crawl4ai returned {len(results)} documents for query: '{query}'")

        news_items = []
        for doc_index, doc in enumerate(results):
            if len(news_items) >= num_results:
                logger.debug(f"Reached num_results limit ({num_results}). Stopping processing of further documents.")
                break

            title = doc.get('title', None)
            content = doc.get('content', None) # Get content for potential fallbacks
            description = doc.get('description', None)
            source_url = doc.get('url', None)

            if not title and content:
                 title = content[:80] + "..." 
                 logger.debug(f"Doc {doc_index}: Title missing, derived from content: '{title}'")
            elif not title:
                title = source_url if source_url else "Source URL Missing"
                logger.debug(f"Doc {doc_index}: Title missing, no content, using URL as title: '{title}'")
            
            snippet = description
            if not snippet and content:
                snippet = content[:200].strip().replace('\n', ' ') + "..."
                logger.debug(f"Doc {doc_index}: Snippet missing, derived from content: '{snippet}'")
            elif not snippet:
                snippet = "No snippet available."
                logger.debug(f"Doc {doc_index}: Snippet missing, no content for fallback.")
            
            if source_url: 
                news_items.append({
                    "title": title,
                    "snippet": snippet,
                    "source_url": source_url
                })
                logger.debug(f"Doc {doc_index}: Added news item: '{title}' from {source_url}")
            else:
                logger.debug(f"Doc {doc_index}: Skipped item due to missing source_url. Title was: '{title}'")
        
        if not news_items: 
             logger.warning(f"Could not extract any suitable news items with URLs from {len(results)} crawled pages for query: '{query}'.")
             return {"error": "Could not extract suitable news items from crawled pages."}

        logger.info(f"Successfully fetched and processed {len(news_items)} news items for query: '{query}'.")
        return {
            "query": query,
            "news_items": news_items
        }

    except Exception as e:
        logger.error(f"Error fetching/processing news for query '{query}': {e}", exc_info=True) 
        return {"error": f"Failed to fetch or process news: {str(e)}"}

```
