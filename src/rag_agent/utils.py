"""Utility helpers for graph nodes and formatting."""

from __future__ import annotations

import functools
import time
import unicodedata
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


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
                print(f"[node:{name}] completed in {elapsed:.3f}s")

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

    This intentionally keeps the heuristic simple: if the question is mostly
    ASCII letters, force English; otherwise use Chinese for the current product
    workflow.
    """
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

    For English prompts, force plain ASCII output to avoid PowerShell mojibake
    from curly punctuation or accented title text.
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
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized
