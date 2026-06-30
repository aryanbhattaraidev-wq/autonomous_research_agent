"""
app/agents/planner.py
Planner agent - turns a user topic into a structured research plan.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.models.schemas import ResearchState, ResearchPlan

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

logger = logging.getLogger(__name__)
_PROMPT = (Path(__file__).parent.parent / "prompts" / "planner.txt").read_text()

def ResearchNode(state: ResearchState) -> dict:
    """
    Langgraph node.
    Input: state["topic"]
    Output: state["plan"], state["status"]
    """
    topic: str = state.get("topic", "")
    logger.info("[Planner] Planning research for: %r", topic)

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        api_key=os.getenv("GROQ_API_KEY")
    ).with_structured_output(ResearchPlan)

    messages = [
        SystemMessage(content=_PROMPT),
        HumanMessage(content=f"research topic: {topic}")        
    ]
    
    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        plan = ResearchPlan(**data)
        logger.info("[Planner] Generated %d queries", len(plan.search_queries))
        return {**state, "plan": plan.model_dump(), "status": "researching"}

    except Exception as exc:
        logger.error("[Planner] Error: %s", exc)
        fallback = ResearchPlan(
            topic=topic,
            subtopics=[topic],
            search_queries=[topic, f"{topic} explained", f"{topic} overview"],
            learning_objectives=[f"Understand {topic}"],
        )
        return {**state, "plan": fallback.model_dump(), "status": "researching", "error": str(exc)}