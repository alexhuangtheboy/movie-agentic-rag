"""State for the movie tool selection graph."""

from __future__ import annotations

from typing_extensions import TypedDict


class ToolSelectionState(TypedDict, total=False):
    """Input/output state for one-shot tool selection."""

    question: str
    sql_table_names: list[str]
    graph_table_names: list[str]
    tool: str
    query: str
    confidence: float
    reasoning: str
    error: str | None
