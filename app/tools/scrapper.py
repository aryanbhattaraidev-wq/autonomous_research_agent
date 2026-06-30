"""
app/tools/scraper.py
Light-weight web scraper to enrich SearchResults with full page text.
"""
from __future__ import annotations
import logging
import re

import requests
from bs4 import BeautifulSoup

from app.models.schemas import SearchResult

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchAgent/1.0; +https://github.com/)"
    )
}
TIMEOUT = 10          # seconds
MAX_CHARS = 8_000     # truncate very long pages


def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text[:MAX_CHARS]


def scrape_url(url: str) -> str:
    """Fetch a URL and return cleaned plain text (best-effort)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return _clean_text(resp.text)
    except Exception as exc:
        logger.warning("scrape_url failed for %s: %s", url, exc)
        return ""


def enrich_results(results: list[SearchResult]) -> list[SearchResult]:
    """
    For each SearchResult that lacks full content, scrape the URL.
    Returns the same list (mutated in-place) for convenience.
    """
    for result in results:
        if not result.content or len(result.content) < 200:
            scraped = scrape_url(result.url)
            if scraped:
                result.content = scraped
    return results