"""
app/agents/researcher.py
Researching Agent - Executes the research plan using web scraping and web search
"""

from __future__ import annotations
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
import logging

from app.models.schemas import ResearchState, SearchResult, ResearchPlan
from app.tools.web_search import multi_search
from app.tools.scrapper import enrich_results

logger = logging.getLogger(__name__)

def researcher_node(state: ResearchState) -> dict:
    """
    Langgraph node.
    Input: state["plan"]
    Output: state["raw_documents"] + state["status"]
    """
    plan_data = state.get("plan")
    if not plan_data:
        return {**state, "error": "No plan data available", "status": "done"}

    plan = ResearchPlan(**plan_data)
    logger.info("[Researcher] Executing %d search queries", len(plan.search_queries))

    results: list[SearchResult] = multi_search(
        queries=plan.search_queries,
        max_results_per_query=3,
    )

    results = enrich_results(results)

    logger.info("[Researcher] Collected %d documents", len(results))

    raw_docs = [r.model_dump() for r in results]
    return {**state, "raw_docs": raw_docs, "status": "summarizing"}