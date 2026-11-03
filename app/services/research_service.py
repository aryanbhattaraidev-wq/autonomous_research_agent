"""
app/services/research_service.py
Runs the LangGraph pipeline - blocking or as a stream of progress events.
"""
from __future__ import annotations
import logging
from collections.abc import Iterator

from app.graph.builder import get_graph
from app.graph.state import ResearchState

logger = logging.getLogger(__name__)

STEP_LABELS = {
    "planner": "Planning research",
    "researcher": "Searching the web",
    "summarizer": "Summarising findings",
    "notes": "Writing notes",
    "memory": "Storing to memory",
}


def run_research(topic: str) -> ResearchState:
    """Run the full pipeline and return the final state."""
    return get_graph().invoke({"topic": topic, "status": "planning"})


def stream_research(topic: str) -> Iterator[tuple[str, ResearchState]]:
    """
    Yield (node_name, accumulated_state) after each node finishes.
    The last yielded state is the final result.
    """
    state: ResearchState = {"topic": topic, "status": "planning"}
    for update in get_graph().stream(state, stream_mode="updates"):
        for node, delta in update.items():
            state = {**state, **(delta or {})}
            yield node, state
