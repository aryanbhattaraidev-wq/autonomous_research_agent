"""
app/memory/notes_store.py
File-based storage for markdown notes. One shared slug function so saving
and loading always agree on filenames.
"""
from __future__ import annotations
import logging
import re

from app.config import get_settings

logger = logging.getLogger(__name__)


def slugify(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return slug[:60] or "note"


def save_note(topic: str, markdown: str) -> str:
    """Write the note to NOTES_DIR/<slug>.md and return the slug."""
    notes_dir = get_settings().notes_dir
    notes_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    (notes_dir / f"{slug}.md").write_text(markdown, encoding="utf-8")
    logger.info("[Notes] Saved note to %s", notes_dir / f"{slug}.md")
    return slug


def list_notes() -> list[str]:
    notes_dir = get_settings().notes_dir
    if not notes_dir.exists():
        return []
    return sorted(p.stem for p in notes_dir.glob("*.md"))


def get_note(name: str) -> str | None:
    """Look a note up by slug or by original topic text."""
    path = get_settings().notes_dir / f"{slugify(name)}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None
