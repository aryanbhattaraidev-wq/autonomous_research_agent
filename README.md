# Autonomous Research Agent

Give it a topic. It plans the research, searches and scrapes the web, summarises
the findings, writes study notes with flashcards, stores them in a vector
database, and answers follow-up questions from that memory.

Built with LangGraph, OpenRouter, FastAPI, Qdrant and Streamlit.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how it works.

## Setup

1. Create `.env` in the project root:
   ```
   OPENROUTER_API_KEY=...
   TAVILY_API_KEY=...
   # optional: OPENROUTER_MODEL, OPENROUTER_FAST_MODEL, GROQ_API_KEY (fallback)
   ```
2. Install and run:
   ```bash
   uv sync
   uv run uvicorn app.main:app --port 8000
   uv run streamlit run frontend/streamlit_app.py   # in a second terminal
   ```
3. Tests: `uv run pytest`
