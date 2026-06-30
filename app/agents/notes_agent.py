"""
app/agents/notes_agent.py
Notes agent - converts summaries into polished markdown notes + flashcards
"""

from __future__ import annotations
import json
import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timezone

from app.models.schemas import StructuredSummary, ResearchState, ResearchNote

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)
_PROMPT = (Path(__file__).parent.parent / "prompts" / "notes.txt").read_text()

load_dotenv()


def _build_summary(summaries: list[StructuredSummary]) -> str:
    """Flatten multiple summaries into a single text block for the LLM."""
    parts: list[str] = []
    for s in summaries:
        part = (
            f"=== {s.topic} ===\n"
            f"Summary: {s.summary}\n"
            f"Key ideas: {'; '.join(s.key_ideas)}\n"
            f"Terms: {'; '.join(s.important_terms)}\n"
            f"Examples: {'; '.join(s.examples)}\n"
            f"Limitations: {'; '.join(s.limitations)}\n"
            f"Sources: {'; '.join(s.sources)}"
        )
        parts.append(part)
    return "\n\n".join(parts)


def notes_node(state: ResearchState) -> dict:
    """
    Langgraph node.
    Input: state["summaries"], state["plan"] 
    Output: state["notes"] + state["report"] + state["status"]
    """
    summaries_data = state.get("summaries", [])
    plan_data = state.get("plan", {})
    topic = plan_data.get("topic", state.get("topic", "research"))

    if not summaries_data:
        logger.warning("[Notes] No summaries — generating placeholder note")
        note = ResearchNote(
            topic=topic,
            markdown=f"# {topic}\n\nNo research data was collected. Please check your API keys.",
            flashcards=[],
            key_concepts=[],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return {**state, "notes": note.model_dump(), "report": note.markdown, "status": "done"}

    summaries = [StructuredSummary(**s) for s in summaries_data]
    summary_text = _build_summary(summaries)

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        api_key=os.getenv("GROQ_API_KEY")
    ).with_structured_output(ResearchNote)

    messages = [
       SystemMessage(content=_PROMPT), 
       HumanMessage(content=f'Main topic: "{topic}"\n\nResearch summaries:\n{summary_text}') 
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        note = ResearchNote(
            **data,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.error("[Notes] Error: %s", exc)
        note = ResearchNote(
            topic=topic,
            markdown=f"# {topic}\n\n{summary_text}",
            flashcards=[],
            key_concepts=[],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    _save_note(note)

    # Build final report: note markdown + sources section
    all_sources = list({url for s in summaries for url in s.sources})
    sources_block = "\n".join(f"- {src}" for src in all_sources) if all_sources else "_No sources._"
    report = note.markdown + f"\n\n---\n\n## Sources\n{sources_block}"

    logger.info("[Notes] Note created for topic=%r", topic)
    return {**state, "notes": note.model_dump(), "report": report, "status": "done"}


def _save_note(note: ResearchNote) -> None:
    """Save note as a markdown file in the notes directory."""
    notes_dir = Path(os.getenv("NOTES_DIR", "./notes"))
    notes_dir.mkdir(parents=True, exist_ok=True)
    filename = note.topic.lower().replace(" ", "_")[:60] + ".md"
    filepath = notes_dir / filename
    try:
        filepath.write_text(note.markdown, encoding="utf-8")
        logger.info("[Notes] Saved note to %s", filepath)
    except Exception as exc:
        logger.warning("[Notes] Could not save note: %s", exc)
