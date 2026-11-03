"""
app/services/note_service.py
Note listing/retrieval and follow-up Q&A (RAG over the vector store).
"""
from __future__ import annotations
import logging
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm import get_llm
from app.memory.notes_store import get_note, list_notes, slugify  # re-exported for the API layer
from app.memory.vector_store import get_vector_store

logger = logging.getLogger(__name__)
_PROMPT = (Path(__file__).parent.parent / "prompts" / "followup.txt").read_text(encoding="utf-8")

__all__ = ["get_note", "list_notes", "list_vector_topics", "answer_follow_up"]


def list_vector_topics() -> list[str]:
    return get_vector_store().list_topics()


def answer_follow_up(
    topic: str, question: str, conversation_history: list[dict[str, str]] | None = None
) -> tuple[str, list[str]]:
    """Answer `question` using chunks retrieved for `topic`. Returns (answer, sources)."""
    store = get_vector_store()
    # The UI may pass a note slug (from the sidebar) - map it back to the stored topic name.
    topic = next((t for t in store.list_topics() if slugify(t) == slugify(topic)), topic)
    hits = store.search(question, topic=topic, k=5) or store.search(question, k=5)
    if not hits:
        return "I don't have any stored research for that yet. Run a research session first.", []

    context = "\n\n---\n\n".join(h["chunk"] for h in hits)
    sources = list(dict.fromkeys(u for h in hits for u in h.get("sources", [])))

    messages = [SystemMessage(content=_PROMPT)]
    # Prior turns (the UI includes the current question as the last user message; drop it).
    history = (conversation_history or [])[:-1] if conversation_history else []
    for turn in history[-6:]:
        cls = HumanMessage if turn.get("role") == "user" else AIMessage
        messages.append(cls(content=turn.get("content", "")))
    messages.append(HumanMessage(content=f"Research context:\n{context}\n\nQuestion: {question}"))

    answer = get_llm().invoke(messages).content
    if isinstance(answer, list):
        answer = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in answer)
    return answer.strip(), sources
