import pytest


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point storage at a temp dir and clear real API keys for every test."""
    monkeypatch.setenv("NOTES_DIR", str(tmp_path / "notes"))
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    for key in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
