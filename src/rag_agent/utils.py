"""Utility helpers for graph nodes and formatting."""

from __future__ import annotations

import functools
import logging
import os
import time
import unicodedata
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


def time_node(name: str) -> Callable[[F], F]:
    """Log basic timing information for LangGraph nodes."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.info("Node %s completed in %.3fs", name, elapsed)

        return wrapper  # type: ignore[return-value]

    return decorator


def strip_code_fence(text: str) -> str:
    """Remove common markdown code fences from model-generated SQL/Cypher."""
    value = text.strip()
    for prefix in ("```sql", "```cypher", "```json", "```"):
        if value.lower().startswith(prefix):
            value = value[len(prefix):].strip()
            if value.endswith("```"):
                value = value[:-3].strip()
            break
    return value


def truncate_text(text: str, limit: int = 8000) -> str:
    """Trim text to a prompt-safe size while preserving a useful prefix."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def detect_answer_language(text: str) -> str:
    """Detect the response language for prompt control.

    Use langdetect when installed, and keep a deterministic heuristic fallback
    so local/test environments do not fail if the optional detector is absent.
    """
    stripped = text.strip()
    if not stripped:
        return "English"
    try:
        from langdetect import LangDetectException, detect

        detected = detect(stripped)
        if detected.startswith("zh"):
            return "Chinese"
        return "English"
    except (ImportError, LangDetectException):
        logger.debug("Falling back to heuristic language detection", exc_info=True)

    ascii_letters = sum(1 for char in text if char.isascii() and char.isalpha())
    non_ascii = sum(1 for char in text if not char.isascii() and not char.isspace())
    if ascii_letters >= max(non_ascii, 1):
        return "English"
    return "Chinese"


def language_instruction(text: str) -> str:
    """Return a strict language instruction for final answer prompts."""
    language = detect_answer_language(text)
    if language == "English":
        return (
            "Required answer language: English. Use English only. Do not answer in German, Chinese, or any other language. "
            "Use plain ASCII punctuation only."
        )
    return "Required answer language: Chinese. Use Chinese only unless the user explicitly asks for another language."


def sanitize_answer_for_query(answer: str, query: str) -> str:
    """Normalize final answers for the detected user language.

    By default, preserve non-ASCII title text and only normalize punctuation
    that commonly causes CLI/display issues. Set ASCII_SANITIZE_ANSWERS=true
    to opt into legacy ASCII-only behavior.
    """
    if detect_answer_language(query) != "English":
        return answer

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u00bf": "",
        "\u00a1": "",
    }
    normalized = answer
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    if os.getenv("ASCII_SANITIZE_ANSWERS", "false").lower() == "true":
        normalized = unicodedata.normalize("NFKD", normalized)
        normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized
