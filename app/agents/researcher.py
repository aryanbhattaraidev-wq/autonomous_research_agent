"""
app/agents/researcher.py
Researching agent - executes the research plan using web search + scraping.
"""
from __future__ import annotations
import logging

from app.config import get_settings
from app.graph.state import ResearchState
from app.models.schemas import ResearchPlan
from app.tools.scraper import enrich_results
from app.tools.web_search import multi_search

logger = logging.getLogger(__name__)


def researcher_node(state: ResearchState) -> dict:
    """LangGraph node. Input: plan. Output: raw_documents, status."""
    plan_data = state.get("plan")
    if not plan_data:
        return {"error": "researcher: no plan available", "raw_documents": [], "status": "noting"}

    plan = ResearchPlan(**plan_data)
    logger.info("[Researcher] Executing %d search queries", len(plan.search_queries))

    results = multi_search(
        queries=plan.search_queries,
        max_results_per_query=get_settings().max_results_per_query,
    )
    results = enrich_results(results)
    logger.info("[Researcher] Collected %d documents", len(results))

    return {"raw_documents": [r.model_dump() for r in results], "status": "summarizing"}
