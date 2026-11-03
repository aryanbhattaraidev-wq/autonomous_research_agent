"""
app/main.py
FastAPI application entry point.
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.memory.vector_store import get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up the vector store connection
    logger.info("Initialising vector store ...")
    try:
        get_vector_store()
    except Exception:
        logger.exception("Vector store unavailable - follow-up Q&A will fail until fixed")
    get_settings().notes_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Research Agent API ready 🚀")
    yield
    logger.info("Shutting down …")


app = FastAPI(
    title="Autonomous Research Agent",
    description=(
        "An AI-powered research agent built with LangGraph. "
        "Automatically plans, searches, summarises, and stores research notes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
