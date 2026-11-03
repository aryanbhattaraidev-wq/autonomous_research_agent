"""
app/agents/planner.py
Planner agent - turns a user topic into a structured research plan.
"""
from __future__ import annotations
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import ResearchState
from app.llm import get_llm, invoke_structured
from app.models.schemas import ResearchPlan

logger = logging.getLogger(__name__)
_PROMPT = (Path(__file__).parent.parent / "prompts" / "planner.txt").read_text(encoding="utf-8")


def planner_node(state: ResearchState) -> dict:
    """LangGraph node. Input: topic. Output: plan, status."""
    topic = state.get("topic", "")
    logger.info("[Planner] Planning research for: %r", topic)

    try:
        plan = invoke_structured(
            get_llm(),
            [SystemMessage(content=_PROMPT), HumanMessage(content=f"research topic: {topic}")],
            ResearchPlan,
        )
        logger.info("[Planner] Generated %d queries", len(plan.search_queries))
        return {"plan": plan.model_dump(), "status": "researching"}
    except Exception as exc:
        logger.error("[Planner] Error: %s - using fallback plan", exc)
        fallback = ResearchPlan(
            topic=topic,
            subtopics=[topic],
            search_queries=[topic, f"{topic} explained", f"{topic} overview"],
            learning_objectives=[f"Understand {topic}"],
        )
        return {"plan": fallback.model_dump(), "status": "researching", "error": f"planner: {exc}"}


# Backwards-compatible alias for the old name.
ResearchNode = planner_node
