"""State for the movie tool selection graph."""

from __future__ import annotations

from typing_extensions import TypedDict


class ToolSelectionState(TypedDict, total=False):
    """Input/output state for one-shot tool selection."""

    question: str
    sql_table_names: list[str]
    graph_table_names: list[str]
    chat_history_messages: list[str]
    user_memories_text: str
    session_memories_text: str
    previous_tool: str
    previous_answer: str
    tool: str
    query: str
    confidence: float
    reasoning: str
    error: str | None
