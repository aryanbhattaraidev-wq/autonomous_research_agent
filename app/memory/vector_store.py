"""
app/memory/vector_store.py
Long-term memory: notes are chunked, embedded locally (fastembed) and stored
in an embedded, on-disk Qdrant collection. Used for follow-up Q&A (RAG).
"""
from __future__ import annotations
import logging
import re
import threading
import uuid
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION = "research_notes"
_lock = threading.Lock()


class VectorStore:
    def __init__(self) -> None:
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient, models

        s = get_settings()
        s.qdrant_path.mkdir(parents=True, exist_ok=True)
        self._models = models
        self._embedder = TextEmbedding(model_name=s.embedding_model)
        self._client = QdrantClient(path=str(s.qdrant_path))
        dim = len(next(iter(self._embedder.embed(["dimension probe"]))))
        if not self._client.collection_exists(COLLECTION):
            self._client.create_collection(
                COLLECTION,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._embedder.embed(texts)]

    def add_note(self, topic: str, markdown: str, sources: list[str] | None = None) -> int:
        """Replace any existing chunks for `topic` and store the new ones."""
        chunks = chunk_markdown(markdown)
        if not chunks:
            return 0
        m = self._models
        with _lock:
            self.delete_topic(topic)
            vectors = self._embed(chunks)
            self._client.upsert(
                COLLECTION,
                points=[
                    m.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={"topic": topic, "chunk": chunk, "sources": sources or []},
                    )
                    for chunk, vec in zip(chunks, vectors)
                ],
            )
        logger.info("[Memory] Stored %d chunks for %r", len(chunks), topic)
        return len(chunks)

    def delete_topic(self, topic: str) -> None:
        m = self._models
        self._client.delete(
            COLLECTION,
            points_selector=m.FilterSelector(
                filter=m.Filter(must=[m.FieldCondition(key="topic", match=m.MatchValue(value=topic))])
            ),
        )

    def search(self, query: str, topic: str | None = None, k: int = 5) -> list[dict]:
        m = self._models
        flt = (
            m.Filter(must=[m.FieldCondition(key="topic", match=m.MatchValue(value=topic))])
            if topic
            else None
        )
        vec = self._embed([query])[0]
        hits = self._client.query_points(COLLECTION, query=vec, query_filter=flt, limit=k).points
        return [{"score": h.score, **h.payload} for h in hits]

    def list_topics(self) -> list[str]:
        topics: set[str] = set()
        offset = None
        while True:
            points, offset = self._client.scroll(
                COLLECTION, limit=256, offset=offset, with_payload=["topic"], with_vectors=False
            )
            topics.update(p.payload["topic"] for p in points if p.payload)
            if offset is None:
                break
        return sorted(topics)


def chunk_markdown(markdown: str, max_chars: int = 900) -> list[str]:
    """Split on markdown headings, then pack paragraphs up to ~max_chars."""
    sections = re.split(r"\n(?=#{1,3} )", markdown.strip())
    chunks: list[str] = []
    for section in sections:
        buf = ""
        for para in re.split(r"\n{2,}", section):
            if buf and len(buf) + len(para) > max_chars:
                chunks.append(buf.strip())
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append(buf.strip())
    return [c for c in chunks if len(c) > 20]


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()
