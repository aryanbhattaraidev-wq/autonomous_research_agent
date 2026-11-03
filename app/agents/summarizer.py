"""
app/agents/summarizer.py
Summarizer agent - converts raw documents into structured summaries.
"""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import ResearchState
from app.llm import get_llm, invoke_structured
from app.models.schemas import SearchResult, StructuredSummary

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent.parent / "prompts" / "summarize.txt").read_text(encoding="utf-8")
_MAX_CONTENT_CHARS = 6_000  # per document fed to the LLM
_MAX_DOCS_PER_CALL = 3      # batch size to avoid context overflow
_MAX_WORKERS = 4


def _summarize_batch(llm, label: str, docs: list[SearchResult]) -> StructuredSummary | None:
    """Summarize a batch of documents into one StructuredSummary."""
    content_block = ""
    sources: list[str] = []
    for doc in docs:
        text = doc.content or doc.snippet
        content_block += f"\n\n=== {doc.title} ===\nURL: {doc.url}\n{text[:_MAX_CONTENT_CHARS]}"
        sources.append(doc.url)

    messages = [
        SystemMessage(content=_PROMPT),
        HumanMessage(content=f'Subtopic/Query: "{label}"\n\nResearch content:\n{content_block}'),
    ]
    try:
        summary = invoke_structured(llm, messages, StructuredSummary)
        if not summary.sources:
            summary.sources = sources
        return summary
    except Exception as exc:
        logger.error("[Summarizer] batch error: %s", exc)
        return None


def summarizer_node(state: ResearchState) -> dict:
    """LangGraph node. Input: raw_documents, plan. Output: summaries, status."""
    raw_docs = state.get("raw_documents", [])
    plan = state.get("plan") or {}
    topic = state.get("topic") or plan.get("topic", "research")
    queries = plan.get("search_queries") or [topic]

    if not raw_docs:
        logger.warning("[Summarizer] No documents to summarize.")
        return {"summaries": [], "status": "noting"}

    docs = [SearchResult(**d) for d in raw_docs]
    llm = get_llm(fast=True)

    batches = [(queries[(i // _MAX_DOCS_PER_CALL) % len(queries)] if queries else topic, docs[i : i + _MAX_DOCS_PER_CALL])
               for i in range(0, len(docs), _MAX_DOCS_PER_CALL)]
    # Label each batch with the topic itself so summaries stay on-topic.
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        results = list(pool.map(lambda b: _summarize_batch(llm, f"{topic} - {b[0]}", b[1]), batches))

    summaries = [s for s in results if s]
    logger.info("[Summarizer] Produced %d summaries from %d batches", len(summaries), len(batches))
    update: dict = {"summaries": [s.model_dump() for s in summaries], "status": "noting"}
    if not summaries:
        update["error"] = "summarizer: no summaries produced"
    return update
