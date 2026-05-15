"""LangGraph state definitions for movie workflows."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class AddDataState(TypedDict, total=False):
    """State for vector ingestion."""

    movies_data: list[dict[str, Any]]
    processed_count: int
    error_count: int
    result: dict[str, Any]
    message: str


class QueryState(TypedDict, total=False):
    """State for vector-only movie RAG query."""

    query: str
    top_k: int
    similar_movies: list[dict[str, Any]]
    context: str
    answer: str
    reasoning: str
    success: bool


class MovieAgentState(QueryState, total=False):
    """State for the global movie agentic query graph."""

    sql_table_names: list[str]
    graph_table_names: list[str]
    chat_history_messages: list[str]
    target_reply_message: str | None
    user_id: str | None
    chat_id: str | None
    user_memories: list[dict[str, Any]]
    session_memories: list[dict[str, Any]]
    memory_error: str | None
    suggested_tools: str
    routing_query: str
    routing_confidence: float
    routing_reasoning: str
    sql_refinement_attempts: int
    graph_refinement_attempts: int
    raw_sql_result: str
    raw_graph_result: str
