"""
app/graph/builder.py
Assembles the LangGraph research pipeline.

    planner -> researcher -> summarizer -> notes -> memory -> END
                   |                         ^
                   +-- (no documents) -------+   skip summarizer, go straight to notes
"""
from __future__ import annotations
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents.memory_agent import memory_node
from app.agents.notes_agent import notes_node
from app.agents.planner import planner_node
from app.agents.researcher import researcher_node
from app.agents.summarizer import summarizer_node
from app.graph.state import ResearchState


def _after_research(state: ResearchState) -> str:
    return "summarizer" if state.get("raw_documents") else "notes"


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("summarizer", summarizer_node)
    g.add_node("notes", notes_node)
    g.add_node("memory", memory_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_conditional_edges("researcher", _after_research, {"summarizer": "summarizer", "notes": "notes"})
    g.add_edge("summarizer", "notes")
    g.add_edge("notes", "memory")
    g.add_edge("memory", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
