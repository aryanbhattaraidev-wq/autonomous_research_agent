"""
app/graph/state.py
Shared state threaded through the LangGraph research graph.

A TypedDict (not a Pydantic model) so LangGraph merges the partial dicts
returned by each node. Nested objects are stored as plain dicts
(`model_dump()` output) so the state is JSON-serialisable / checkpointable.
"""
from __future__ import annotations
from typing import Literal, TypedDict

Status = Literal["idle", "planning", "researching", "summarizing", "noting", "storing", "done", "error"]


class ResearchState(TypedDict, total=False):
    topic: str
    plan: dict               # ResearchPlan.model_dump()
    raw_documents: list[dict]  # SearchResult.model_dump() items
    summaries: list[dict]    # StructuredSummary.model_dump() items
    notes: dict              # ResearchNote.model_dump()
    report: str
    error: str
    status: Status
