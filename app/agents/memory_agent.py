"""
app/agents/memory_agent.py
Final graph node - embeds the finished note into the vector store so it can
be retrieved for follow-up questions.
"""
from __future__ import annotations
import logging

from app.graph.state import ResearchState
from app.memory.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def memory_node(state: ResearchState) -> dict:
    """LangGraph node. Input: notes, summaries. Output: status=done."""
    note = state.get("notes") or {}
    topic = note.get("topic") or state.get("topic", "")
    # Index the full report (note + sources) so source URLs are retrievable too.
    text = state.get("report") or note.get("markdown", "")
    sources = list(dict.fromkeys(u for s in state.get("summaries", []) for u in s.get("sources", [])))
    try:
        get_vector_store().add_note(topic, text, sources)
    except Exception as exc:  # memory failure must never lose the research result
        logger.error("[Memory] could not store note: %s", exc)
        return {"status": "done", "error": (state.get("error", "") + f" memory: {exc}").strip()}
    return {"status": "done"}
