"""
app/agents/summarizer.py
Summarizer agent - converts raw documents into structured summaries.
"""

from __future__ import annotations
import json
import os
import logging
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.schemas import SearchResult, StructuredSummary, ResearchState

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent.parent / "prompts" / "summarize.txt").read_text()
_MAX_CONTENT_CHARS = 6_000  # per document fed to LLM
_MAX_DOCS_PER_CALL = 3      # batch size to avoid context overflow

def _summarize_batch(
        llm: ChatGroq,
        topic: str,
        docs: list[SearchResult]
) -> StructuredSummary | None:
    """Summarize a batch of documents into one StructuredSummary"""
    content_block = ""
    sources: list[str] = []
    for doc in docs:
        text = doc.content or doc.snippet
        content_block += f"\n\n=== {doc.title} ===\nURL: {doc.url}\n{text[:_MAX_CONTENT_CHARS]}"
        sources.append(doc.url)

    messages = [
        SystemMessage(content=_PROMPT),
        HumanMessage(
            content=(
                f'Subtopic/Query: "{topic}"\n\n'
                f"Research content:\n{content_block}"
            )
        )
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        data.setdefault("sources", sources)
        return StructuredSummary(**data)
    except Exception as exc:
        logger.error("[Summarizer] batch error: %s", exc)
        return None


def summarizer_node(state: ResearchState) -> dict:
    """
    Langgraph node.
    Input: state["raw_documents"], state["plan"] 
    Output: state["summaries"], state["status"]
    """
    raw_docs = state.get("raw_documents", [])
    plan_data = state.get("plan", {})
    topic = plan_data.get("topic", state.get("topic", "research"))
    search_queries = plan_data.get("search_queries", [topic])

    if not raw_docs:
        logger.warning("[Summarizer] No documents to summarize.")
        return {**state, "summaries": [], "status": "noting"}

    docs = [SearchResult(**d) for d in raw_docs]

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        api_key=os.getenv("GROQ_API_KEY")
    ).with_structured_output(StructuredSummary)

    summaries: list[StructuredSummary] = []

    for i in range(0, len(docs), _MAX_CONTENT_CHARS):
        batch = docs[i: i + _MAX_CONTENT_CHARS]
        query_labels = search_queries[i // _MAX_DOCS_PER_CALL] if (i // _MAX_DOCS_PER_CALL) < len(search_queries) else topic
        summary = _summarize_batch(llm, query_labels, batch)
        if summary:
            summaries.append(summary)

    logger.info("[Summarize] Produced %d summaries", len(summaries))

    return {
        **state, 
        "summaries": [s.model_dump() for s in summaries], 
        "status": "noting"
    }
