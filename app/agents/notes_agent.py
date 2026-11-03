"""
app/agents/notes_agent.py
Notes agent - converts summaries into polished markdown notes + flashcards.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import ResearchState
from app.llm import get_llm, invoke_structured
from app.memory.notes_store import save_note
from app.models.schemas import ResearchNote, StructuredSummary

logger = logging.getLogger(__name__)
_PROMPT = (Path(__file__).parent.parent / "prompts" / "notes.txt").read_text(encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_summary(summaries: list[StructuredSummary]) -> str:
    """Flatten multiple summaries into a single text block for the LLM."""
    parts = []
    for s in summaries:
        parts.append(
            f"=== {s.topic} ===\n"
            f"Summary: {s.summary}\n"
            f"Key ideas: {'; '.join(s.key_ideas)}\n"
            f"Terms: {'; '.join(s.important_terms)}\n"
            f"Examples: {'; '.join(s.examples)}\n"
            f"Limitations: {'; '.join(s.limitations)}\n"
            f"Sources: {'; '.join(s.sources)}"
        )
    return "\n\n".join(parts)


def notes_node(state: ResearchState) -> dict:
    """LangGraph node. Input: summaries, plan. Output: notes, report, status."""
    plan = state.get("plan") or {}
    topic = state.get("topic") or plan.get("topic", "research")
    summaries_data = state.get("summaries", [])

    if not summaries_data:
        logger.warning("[Notes] No summaries - generating placeholder note")
        note = ResearchNote(
            topic=topic,
            markdown=f"# {topic}\n\nNo research data was collected. Check your Tavily / LLM API keys and the server logs.",
            created_at=_now(),
        )
        return {"notes": note.model_dump(), "report": note.markdown, "status": "storing"}

    summaries = [StructuredSummary(**s) for s in summaries_data]
    summary_text = _build_summary(summaries)

    try:
        note = invoke_structured(
            get_llm(),
            [SystemMessage(content=_PROMPT),
             HumanMessage(content=f'Main topic: "{topic}"\n\nResearch summaries:\n{summary_text}')],
            ResearchNote,
        )
        note.created_at = _now()
    except Exception as exc:
        logger.error("[Notes] Error: %s - using raw-summary fallback", exc)
        note = ResearchNote(topic=topic, markdown=f"# {topic}\n\n{summary_text}", created_at=_now())

    note.topic = topic  # keep the user's topic as the canonical key
    save_note(topic, note.markdown)

    sources = list(dict.fromkeys(url for s in summaries for url in s.sources))
    sources_block = "\n".join(f"- {src}" for src in sources) if sources else "_No sources._"
    report = note.markdown + f"\n\n---\n\n## Sources\n{sources_block}"

    logger.info("[Notes] Note created for topic=%r", topic)
    return {"notes": note.model_dump(), "report": report, "status": "storing"}
