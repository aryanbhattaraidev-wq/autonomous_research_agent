"""
Pydantic schemas for the research agent system.
"""
from pydantic import BaseModel, Field
from typing import Any, Literal

# planner
class ResearchPlan(BaseModel):
    """Structured research plan produced by the research agent."""
    topic: str
    subtopics: list[str] = Field(description="High-level subtopics to investigate.")
    search_queries: list[str] = Field(description="Concrete web-search queries.")
    learning_objectives: list[str] = Field(description="What we aim to learn.")


# search/scrape
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    content: str = ""

# summarizer
class StructuredSummary(BaseModel):
    topic: str
    summary: str
    key_ideas: list[str]
    important_terms: list[str]
    examples: list[str]
    limitations: list[str]
    sources: list[str]
    
# notes
class ResearchNote(BaseModel):
    topic: str
    markdown: str
    flashcards: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {question, answer} dicts"
    )
    key_concepts: list[str] = Field(
        default_factory=list
    )
    created_at: str = ""


# graph state
class ResearchState(BaseModel):
    """Shared state threaded throughtout the langgraph research graph."""
    topic: str = ""
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    plan: ResearchPlan | None = None

    raw_documents: list[SearchResult] = Field(default_factory=list)

    summaries: list[StructuredSummary] = Field(default_factory=list)

    notes: ResearchNote | None = None

    report: str = ""

    retrieved_context: str = ""
    follow_up_answer: str = ""

    error: str = ""
    status: Literal["idle", "planning", "researching", "summarizing", "nothing", "done"] = "idle"



# API request/response 
class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)

class FollowUpRequest(BaseModel):
    topic: str
    question: str
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

class ResearchResponse(BaseModel):
    topic: str
    report: str
    notes_markdown: str
    flashcards: list[dict[str, str]]
    key_concepts: list[str]
    sources: list[str]
    summaries: list[dict[str, Any]]
    status: str

class FollowUpResponse(BaseModel):
    answer: str
    sources_used: list[str] = Field(default_factory=list)

