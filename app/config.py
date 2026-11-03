"""
app/config.py
Central configuration. Loads .env once and exposes typed settings.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-sonnet-4.5"
    openrouter_fast_model: str = "google/gemini-2.5-flash"
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3-32b"
    tavily_api_key: str = ""
    notes_dir: Path = ROOT / "notes"
    qdrant_path: Path = ROOT / "qdrant_data"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_results_per_query: int = 3


def get_settings() -> Settings:
    """Read settings from the environment (re-read each call so tests can monkeypatch)."""
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", Settings.openrouter_model),
        openrouter_fast_model=os.getenv("OPENROUTER_FAST_MODEL", Settings.openrouter_fast_model),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", Settings.groq_model),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        notes_dir=Path(os.getenv("NOTES_DIR", str(ROOT / "notes"))),
        qdrant_path=Path(os.getenv("QDRANT_PATH", str(ROOT / "qdrant_data"))),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
    )
