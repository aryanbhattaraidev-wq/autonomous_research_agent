"""
app/llm.py
LLM factory + a single helper for getting validated structured output.

Primary provider: OpenRouter (OpenAI-compatible). Fallback: Groq.
"""
from __future__ import annotations
import json
import logging
import re
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def get_llm(fast: bool = False, temperature: float = 0.2) -> BaseChatModel:
    """Return a chat model. `fast=True` picks the cheaper model (used for summarising)."""
    s = get_settings()
    if s.openrouter_api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.openrouter_fast_model if fast else s.openrouter_model,
            api_key=s.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            timeout=120,
            max_retries=2,
        )
    if s.groq_api_key:
        from langchain_groq import ChatGroq

        return ChatGroq(model=s.groq_model, api_key=s.groq_api_key, temperature=temperature)
    raise RuntimeError("No LLM configured: set OPENROUTER_API_KEY (or GROQ_API_KEY) in .env")


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of an LLM reply (handles <think> blocks and code fences)."""
    text = _THINK_RE.sub("", text).strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


def invoke_structured(llm: BaseChatModel, messages: list[BaseMessage], schema: type[T]) -> T:
    """
    Call the LLM and return a validated `schema` instance.

    The prompts already instruct the model to reply with JSON only, so we parse
    the raw text ourselves. This works identically on every provider/model
    (no dependence on native tool-calling support) and tolerates <think> blocks.
    """
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return schema(**_extract_json(content))
