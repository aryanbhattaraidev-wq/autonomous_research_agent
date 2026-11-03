# Architecture

The Autonomous Research Agent takes a topic, plans a research strategy, searches
and scrapes the web, summarises what it finds, writes study notes (with
flashcards), stores them in long-term memory, and then answers follow-up
questions from that memory.

It is a **LangGraph** pipeline exposed through a **FastAPI** backend, with a
**Streamlit** frontend.

```
 Streamlit UI  --HTTP/SSE-->  FastAPI (app/api)  -->  services  -->  LangGraph graph
                                                                        |
                          planner -> researcher -> summarizer -> notes -> memory
                                         |                          ^
                                         +-- no documents found ----+
```

## Why LangGraph (and not the OpenAI Agents SDK)

| | LangGraph (chosen) | OpenAI Agents SDK |
|---|---|---|
| Workflow shape | Explicit nodes and edges: a known pipeline with a few conditional branches | LLM-driven loop with tools and handoffs |
| Control | Deterministic stage order, typed shared state | The model decides what to do next |
| Providers | Any (OpenRouter, Groq, ...) through LangChain | OpenAI-first; other providers need an adapter |
| Progress | `graph.stream()` emits one event per node, which drives the UI progress bar | Event stream is per agent/tool step |
| Cost / latency | Each stage uses a known number of LLM calls | Variable; the loop can wander |

This project is a fixed research pipeline, so explicit control and cheap,
predictable runs matter more than agent autonomy. The Agents SDK would be a
better fit if the agent had to freely choose among many tools in an open-ended loop.

## Layout

```
app/
  main.py               FastAPI app, CORS, lifespan (warms the vector store)
  config.py             Settings from env/.env (keys, model names, paths)
  llm.py                get_llm() factory + invoke_structured() JSON helper
  api/routes.py         REST + SSE endpoints
  services/
    research_service.py run_research() / stream_research() - drive the graph
    note_service.py     note listing + answer_follow_up() (RAG)
  graph/
    state.py            ResearchState (TypedDict) shared by all nodes
    builder.py          Builds and compiles the StateGraph
  agents/               One module per graph node (see below)
  tools/
    web_search.py       Tavily search + URL de-duplication
    scraper.py          requests + BeautifulSoup page-text extraction
  memory/
    notes_store.py      Markdown notes on disk (shared slug function)
    vector_store.py     Embedded Qdrant + fastembed embeddings
  models/schemas.py     Pydantic models (plan, summary, note, API payloads)
  prompts/*.txt         System prompts (planner, summarize, notes, followup)
frontend/streamlit_app.py
tests/                  pytest (LLM, Tavily and vector store stubbed)
```

## The graph

State (`app/graph/state.py`) is a `TypedDict`. Each node returns only the keys it
changes and LangGraph merges them. Nested objects are stored as plain dicts
(`model_dump()`), so state stays JSON-serialisable.

| Node | Reads | Writes | What it does |
|---|---|---|---|
| `planner` | `topic` | `plan`, `status=researching` | LLM produces subtopics, 4-8 search queries and learning objectives. On any failure it falls back to a simple 3-query plan. |
| `researcher` | `plan` | `raw_documents`, `status=summarizing` | Runs every query through Tavily (dedupes by URL), then scrapes pages that came back with little content. |
| `summarizer` | `raw_documents`, `plan` | `summaries`, `status=noting` | Batches 3 documents per LLM call (6,000 chars each) and summarises batches in parallel with the cheaper "fast" model. |
| `notes` | `summaries`, `topic` | `notes`, `report`, `status=storing` | LLM writes markdown notes, flashcards and key concepts; saves `notes/<slug>.md`; appends a Sources section to form `report`. |
| `memory` | `report`, `notes`, `summaries` | `status=done` | Chunks the report by heading, embeds it and upserts it into Qdrant (replacing older chunks for that topic). |

**Conditional edge:** after `researcher`, if no documents were found the graph
skips `summarizer` and goes to `notes`, which produces a clear "no data
collected" placeholder instead of failing.

**Failure policy:** the planner, notes and memory nodes degrade gracefully (a
fallback plan, a raw-summary note, or a recorded `error` field) so that one bad
LLM reply doesn't lose the whole run. Errors are collected in `state["error"]`.

## LLM layer (`app/llm.py`)

- **Primary: OpenRouter** through `langchain-openai`'s `ChatOpenAI` with
  `base_url=https://openrouter.ai/api/v1`. `OPENROUTER_MODEL` is used for
  planning and note-writing; `OPENROUTER_FAST_MODEL` for summarising.
- **Fallback: Groq** (`GROQ_MODEL`) if no OpenRouter key is set.
- `invoke_structured()` asks for JSON (the prompts require JSON-only replies),
  strips `<think>` blocks and code fences, extracts the JSON object and validates
  it against a Pydantic schema. It does not rely on native tool-calling, so it
  behaves the same on every model.

## Memory and follow-up Q&A

1. After a run, the `memory` node chunks the report (split on `#`/`##`/`###`
   headings, packed to about 900 characters) and embeds each chunk with
   `BAAI/bge-small-en-v1.5` (fastembed, runs locally; no API key).
2. Chunks are stored in an on-disk Qdrant collection (`qdrant_data/`) with the
   payload `{topic, chunk, sources}`.
3. `POST /followup` retrieves the top 5 chunks for the topic (falling back to a
   global search), then asks the LLM to answer *only* from that context, with the
   last few conversation turns included. Source URLs of the retrieved chunks are returned.

Qdrant's embedded mode allows **one process per data directory**, so run a
single backend instance (or switch to a Qdrant server for multiple workers).

## HTTP API (`/api/v1`)

| Method / path | Purpose |
|---|---|
| `POST /research` | Run the pipeline, return the full result (blocking) |
| `POST /research/stream` | Server-sent events: `progress` per node, then `result` (or `error`) |
| `POST /followup` | RAG answer: `{topic, question, conversation_history}` -> `{answer, sources_used}` |
| `GET /notes`, `GET /notes/{topic}` | List saved note slugs / fetch markdown (accepts a slug or the original topic) |
| `GET /memory/topics` | Topics present in the vector store |
| `GET /health` | Liveness |

## Frontend

Streamlit tabs: **New Research** (consumes the SSE stream for a real progress
bar), **Report & Notes** (report, key concepts, flashcards, summaries, sources,
download), **Follow-up Q&A** (chat backed by `/followup`). The sidebar lists saved
notes and vector-store topics. All dynamic strings are HTML-escaped.

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | - | Primary LLM provider |
| `OPENROUTER_MODEL` | `anthropic/claude-sonnet-4.5` | Planner, notes, follow-up |
| `OPENROUTER_FAST_MODEL` | `google/gemini-2.5-flash` | Summarizer |
| `GROQ_API_KEY`, `GROQ_MODEL` | -, `qwen/qwen3-32b` | Fallback provider |
| `TAVILY_API_KEY` | - | Web search (without it, searches return nothing) |
| `NOTES_DIR` | `./notes` | Markdown notes |
| `QDRANT_PATH` | `./qdrant_data` | Embedded vector DB |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `API_BASE` | `http://localhost:8000/api/v1` | Used by the Streamlit app |

## Running

```bash
uv sync
uv run uvicorn app.main:app --port 8000          # backend
uv run streamlit run frontend/streamlit_app.py   # UI (second terminal)
uv run pytest                                    # tests
```

The first start downloads the embedding model (about 100 MB).

## Extension points

- **New stage** (e.g. a fact-checker): write `app/agents/x.py` returning a partial
  state dict, then add the node and edges in `graph/builder.py`.
- **Loop / retry**: add a conditional edge from `summarizer` back to `planner`
  when coverage is poor.
- **Persistence / resume**: pass a LangGraph checkpointer to `compile()`.
- **Different models per stage**: extend `get_llm()` with another setting.
