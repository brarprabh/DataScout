import logging
from typing import List
from tavily import TavilyClient

from app.core.config import settings
from app.schemas.rag import WebSource

logger = logging.getLogger(__name__)


class WebSearchService:
    def __init__(self):
        self.client = None
        if settings.TAVILY_API_KEY:
            self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)

    def search(self, query: str, max_results: int = 4) -> List[WebSource]:
        """
        Executes a web search via Tavily and returns a clean list of WebSource objects.
        """
        if not self.client:
            logger.warning("TAVILY_API_KEY is not configured. Web fallback returned 0 results.")
            return []

        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic"
            )
            sources = []
            for item in response.get("results", []):
                sources.append(
                    WebSource(
                        title=item.get("title", "Untitled Source"),
                        url=item.get("url", ""),
                        snippet=item.get("content", "").strip()
                    )
                )
            return sources
        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            return []


web_search_service = WebSearchService()
