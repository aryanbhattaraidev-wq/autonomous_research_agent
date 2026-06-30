"""
app/tools/web_search.py
Web search via Tavily API with graceful fallback.
"""
from __future__ import annotations
import os
import logging
from typing import Optional
from tavily import TavilyClient

from app.models.schemas import SearchResult

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int | None = 3) -> list[SearchResult]:
    """
    Search the web using the Tavily API.
    Returns a list of SearchResult objects.
    Falls back to an empty list if the API key is missing.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — returning empty search results.")
        return []

    max_results = max_results 

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=True,
        )
        results: list[SearchResult] = []
        for r in response.get("results", []):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                content=r.get("raw_content", "") or r.get("content", ""),
            ))
        logger.info("search_web: query=%r → %d results", query, len(results))
        return results

    except Exception as exc:
        logger.error("search_web error: %s", exc)
        return []


def multi_search(queries: list[str], max_results_per_query: int = 3) -> list[SearchResult]:
    """Run multiple queries and deduplicate by URL."""
    seen_urls: set[str] = set()
    all_results: list[SearchResult] = []
    for query in queries:
        for result in search_web(query, max_results=max_results_per_query):
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                all_results.append(result)
    return all_results