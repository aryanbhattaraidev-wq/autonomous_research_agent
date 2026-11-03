"""
app/api/routes.py
FastAPI router - REST API for the research agent.
"""
from __future__ import annotations
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.models.schemas import (
    FollowUpRequest,
    FollowUpResponse,
    ResearchRequest,
    ResearchResponse,
)
from app.services.note_service import answer_follow_up, get_note, list_notes, list_vector_topics
from app.services.research_service import STEP_LABELS, run_research, stream_research

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_response(topic: str, state: dict) -> ResearchResponse:
    notes = state.get("notes") or {}
    summaries = state.get("summaries", [])
    sources = list(dict.fromkeys(u for s in summaries for u in s.get("sources", [])))
    return ResearchResponse(
        topic=topic,
        report=state.get("report", ""),
        notes_markdown=notes.get("markdown", ""),
        flashcards=notes.get("flashcards", []),
        key_concepts=notes.get("key_concepts", []),
        sources=sources,
        summaries=summaries,
        status=state.get("status", "done"),
    )


# -- Research ---------------------------------------------------------------

@router.post("/research", response_model=ResearchResponse)
def start_research(req: ResearchRequest):
    """Run the full research pipeline and return the finished report."""
    try:
        state = run_research(req.topic)
    except Exception as exc:
        logger.exception("research failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return _to_response(req.topic, state)


@router.post("/research/stream")
def start_research_stream(req: ResearchRequest):
    """
    Server-sent events: one `progress` event per finished node, then a final
    `result` event with the full ResearchResponse (or an `error` event).
    """
    def events():
        final: dict = {}
        try:
            for node, state in stream_research(req.topic):
                final = state
                payload = {"node": node, "label": STEP_LABELS.get(node, node)}
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
            result = _to_response(req.topic, final).model_dump()
            yield f"event: result\ndata: {json.dumps(result)}\n\n"
        except Exception as exc:
            logger.exception("research stream failed")
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# -- Follow-up Q&A ----------------------------------------------------------

@router.post("/followup", response_model=FollowUpResponse)
async def follow_up(req: FollowUpRequest):
    """Answer a follow-up question using RAG over stored research."""
    try:
        answer, sources = await run_in_threadpool(
            answer_follow_up, req.topic, req.question, req.conversation_history
        )
    except Exception as exc:
        logger.exception("follow-up failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return FollowUpResponse(answer=answer, sources_used=sources)


# -- Notes ------------------------------------------------------------------

@router.get("/notes", response_model=list[str])
def list_all_notes():
    """Return the names (slugs) of all saved notes."""
    return list_notes()


@router.get("/notes/{topic}", response_class=PlainTextResponse)
def get_note_by_topic(topic: str):
    """Return the markdown note for a topic or slug."""
    note = get_note(topic)
    if note is None:
        raise HTTPException(status_code=404, detail=f"Note not found for topic: {topic!r}")
    return note


# -- Memory / vector DB -----------------------------------------------------

@router.get("/memory/topics", response_model=list[str])
def get_memory_topics():
    """Return topics stored in the vector database."""
    try:
        return list_vector_topics()
    except Exception:
        logger.exception("could not list vector topics")
        return []


@router.get("/health")
def health():
    return {"status": "ok"}
