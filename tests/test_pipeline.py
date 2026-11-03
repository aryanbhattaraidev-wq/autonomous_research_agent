"""Graph + API tests with the LLM, Tavily and vector store stubbed out."""
import json

import pytest
from fastapi.testclient import TestClient

from app.llm import _extract_json
from app.memory import notes_store
from app.memory.vector_store import chunk_markdown
from app.models.schemas import ResearchNote, ResearchPlan, SearchResult, StructuredSummary


def test_extract_json_handles_think_and_fences():
    raw = '<think>hmm {not json}</think>\n```json\n{"a": 1}\n```'
    assert _extract_json(raw) == {"a": 1}


def test_chunk_markdown_splits_on_headings():
    md = "# Title\n\nintro paragraph here, long enough\n\n## Part\n\nanother paragraph that is long enough"
    assert len(chunk_markdown(md)) == 2


def test_slug_roundtrip():
    notes_store.save_note("Diffusion Models!", "# x")
    assert notes_store.list_notes() == ["diffusion_models"]
    assert notes_store.get_note("Diffusion Models!") == "# x"
    assert notes_store.get_note("diffusion_models") == "# x"
    assert notes_store.get_note("nope") is None


class FakeStore:
    def __init__(self):
        self.added = []

    def add_note(self, topic, markdown, sources=None):
        self.added.append((topic, markdown, sources))
        return 1

    def list_topics(self):
        return [t for t, *_ in self.added]

    def search(self, query, topic=None, k=5):
        return [{"chunk": m, "sources": s or []} for t, m, s in self.added]


@pytest.fixture
def stubbed(monkeypatch):
    store = FakeStore()

    def fake_invoke(llm, messages, schema):
        if schema is ResearchPlan:
            return ResearchPlan(topic="t", subtopics=["a"], search_queries=["q1", "q2"], learning_objectives=["o"])
        if schema is StructuredSummary:
            return StructuredSummary(topic="t", summary="s", key_ideas=["k"], important_terms=[],
                                     examples=[], limitations=[], sources=["http://x"])
        return ResearchNote(topic="t", markdown="# Note\n\nbody body body body body",
                            flashcards=[{"question": "q", "answer": "a"}], key_concepts=["c"])

    docs = [SearchResult(title="T", url="http://x", snippet="s", content="c" * 300)]
    monkeypatch.setattr("app.llm.get_llm", lambda *a, **k: object())
    for mod in ("planner", "summarizer", "notes_agent"):
        monkeypatch.setattr(f"app.agents.{mod}.get_llm", lambda *a, **k: object())
        monkeypatch.setattr(f"app.agents.{mod}.invoke_structured", fake_invoke)
    monkeypatch.setattr("app.agents.researcher.multi_search", lambda **k: list(docs))
    monkeypatch.setattr("app.agents.researcher.enrich_results", lambda r: r)
    monkeypatch.setattr("app.agents.memory_agent.get_vector_store", lambda: store)
    monkeypatch.setattr("app.services.note_service.get_vector_store", lambda: store)
    monkeypatch.setattr("app.main.get_vector_store", lambda: store)
    monkeypatch.setattr("app.api.routes.list_vector_topics", lambda: store.list_topics())
    return store


def test_full_graph(stubbed):
    from app.graph.builder import build_graph

    state = build_graph().invoke({"topic": "my topic"})
    assert state["status"] == "done"
    assert state["summaries"][0]["sources"] == ["http://x"]
    assert "## Sources" in state["report"]
    assert stubbed.added and stubbed.added[0][0] == "my topic"
    assert notes_store.get_note("my topic")


def test_graph_without_documents_skips_summarizer(stubbed, monkeypatch):
    from app.graph.builder import build_graph

    monkeypatch.setattr("app.agents.researcher.multi_search", lambda **k: [])
    state = build_graph().invoke({"topic": "empty"})
    assert state["status"] == "done"
    assert "No research data" in state["report"]


def test_api_endpoints(stubbed, monkeypatch):
    from app.main import app

    monkeypatch.setattr("app.services.note_service.get_llm",
                        lambda *a, **k: type("L", (), {"invoke": lambda self, m: type("R", (), {"content": "answer!"})()})())
    with TestClient(app) as c:
        assert c.get("/api/v1/health").json() == {"status": "ok"}
        r = c.post("/api/v1/research", json={"topic": "my topic"})
        assert r.status_code == 200 and r.json()["flashcards"]
        assert c.get("/api/v1/notes").json() == ["my_topic"]
        assert c.get("/api/v1/notes/my_topic").status_code == 200
        assert c.get("/api/v1/notes/zzz").status_code == 404
        assert c.get("/api/v1/memory/topics").json() == ["my topic"]
        f = c.post("/api/v1/followup", json={"topic": "my topic", "question": "why?",
                                             "conversation_history": [{"role": "user", "content": "why?"}]})
        assert f.json()["answer"] == "answer!"
        with c.stream("POST", "/api/v1/research/stream", json={"topic": "my topic"}) as s:
            body = "".join(s.iter_text())
        assert body.count("event: progress") == 5 and "event: result" in body
        json.loads(body.split("event: result\ndata: ")[1].split("\n\n")[0])
