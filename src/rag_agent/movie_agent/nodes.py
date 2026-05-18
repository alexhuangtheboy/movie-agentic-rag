"""Node functions for movie vector RAG workflows."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from rag_agent.context import Context, format_memories_for_prompt
from rag_agent.context.base import DIRECT_RESPONSE_SYSTEM_PROMPT
from rag_agent.llm import get_chat_model
from rag_agent.movie_agent.formatters import convert_movies_to_toon
from rag_agent.movie_agent.prompts import build_direct_response_prompt, build_vector_answer_prompt
from rag_agent.movie_agent.states import AddDataState, QueryState
from rag_agent.movie_agent.tools import format_context_from_movies, get_movie_vector_service
from rag_agent.utils import (
    detect_answer_language,
    language_instruction,
    sanitize_answer_for_query,
    time_node,
    truncate_text,
)

logger = logging.getLogger(__name__)


def _looks_english(text: str) -> bool:
    return detect_answer_language(text) == "English"


def _no_context_answer(query: str) -> str:
    if _looks_english(query):
        return (
            "I could not find matching movie plot context in the vector database. "
            "The vector table may be empty, the stored embeddings may not match the current embedding model, "
            "or no records are close enough to the requested themes."
        )
    return "我没有在向量数据库中找到匹配的电影剧情上下文。可能是向量表为空、已存 embedding 与当前模型不匹配，或没有足够相近的记录。"


def _chat_history_section(state: QueryState) -> str:
    messages = state.get("chat_history_messages") or []
    if not messages:
        return "No prior chat history."
    history_lines = [f"- {message}" for message in messages if message]
    return "\n".join(history_lines) if history_lines else "No prior chat history."


@time_node("store_movie_embeddings")
def store_movie_embeddings_node(state: AddDataState) -> AddDataState:
    """Store movie payloads into pgvector."""
    movies_data = state.get("movies_data") or []
    try:
        result = get_movie_vector_service().add_movies(movies_data)
        return {
            **state,
            "processed_count": result["processed_count"],
            "error_count": 0,
            "result": result,
            "message": "Movie embeddings stored successfully.",
        }
    except Exception as exc:
        logger.exception("Failed to store movie embeddings")
        return {
            **state,
            "processed_count": 0,
            "error_count": len(movies_data),
            "result": {"error": str(exc)},
            "message": f"Failed to store movie embeddings: {exc}",
        }


@time_node("validate_query")
def validate_query_node(state: QueryState) -> QueryState:
    """Validate query input."""
    query = (state.get("query") or "").strip()
    if not query:
        return {**state, "success": False, "answer": "Please provide a movie question.", "reasoning": "Missing query"}
    return {**state, "query": query, "top_k": state.get("top_k") or 5}


@time_node("search_vector_db")
def search_vector_db_node(state: QueryState) -> QueryState:
    """Search pgvector for semantically similar movie content."""
    if state.get("success") is False and state.get("answer") and not state.get("query"):
        return state
    query = state.get("query", "")
    top_k = int(state.get("top_k") or 5)
    try:
        similar_movies = get_movie_vector_service().search(query, top_k)
        if not similar_movies:
            return {
                **state,
                "similar_movies": [],
                "success": False,
                "reasoning": "Vector search returned no similar movie chunks.",
                "answer": _no_context_answer(query),
            }
        return {
            **state,
            "similar_movies": similar_movies,
            "success": True,
            "reasoning": f"Found {len(similar_movies)} semantically similar movie chunks.",
        }
    except Exception as exc:
        logger.exception("Vector search failed")
        return {
            **state,
            "similar_movies": [],
            "success": False,
            "reasoning": f"Vector search failed: {exc}",
            "answer": f"Vector search failed: {exc}",
        }


@time_node("format_context")
def format_context_node(state: QueryState) -> QueryState:
    """Format vector search results for answer generation."""
    similar_movies = state.get("similar_movies") or []
    context = format_context_from_movies(similar_movies)
    toon = convert_movies_to_toon(
        [
            {
                **(item.get("metadata") or {}),
                "movie_id": item.get("movie_id"),
                "description": item.get("content"),
                "score": item.get("score"),
            }
            for item in similar_movies
        ]
    )
    return {**state, "context": f"{toon}\n\nDetailed chunks:\n{context}"}


@time_node("generate_answer")
def generate_answer_node(state: QueryState, runtime: Runtime[Context]) -> QueryState:
    """Generate the final answer using vector context and optional memories."""
    query = state.get("query", "")
    if not state.get("similar_movies"):
        return {
            **state,
            "answer": state.get("answer") or _no_context_answer(query),
            "success": False,
            "reasoning": state.get("reasoning") or "No vector context available for answer generation.",
        }
    context = truncate_text(state.get("context", ""))
    memory_texts = format_memories_for_prompt(state)
    answer_language_instruction = language_instruction(query)
    prompt = build_vector_answer_prompt(
        system_prompt=runtime.context.system_prompt,
        user_memories_text=memory_texts.get("user_memories_text", ""),
        session_memories_text=memory_texts.get("session_memories_text", ""),
        movie_context=context,
        question=query,
        language_instruction_text=answer_language_instruction,
    )
    try:
        llm = get_chat_model(runtime.context.model)
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = sanitize_answer_for_query(str(response.content), query)
        return {**state, "answer": answer, "success": True, "reasoning": "Generated answer from vector context."}
    except Exception as exc:
        logger.exception("Failed to generate answer")
        return {**state, "answer": f"Failed to generate answer: {exc}", "success": False, "reasoning": str(exc)}


@time_node("generate_direct_response")
def generate_direct_response_node(state: QueryState, runtime: Runtime[Context]) -> QueryState:
    """Generate a direct answer without movie vector/SQL/graph retrieval context."""
    query = state.get("query", "")
    memory_texts = format_memories_for_prompt(state)
    answer_language_instruction = language_instruction(query)
    prompt = build_direct_response_prompt(
        direct_system_prompt=DIRECT_RESPONSE_SYSTEM_PROMPT,
        user_memories_text=memory_texts.get("user_memories_text", ""),
        session_memories_text=memory_texts.get("session_memories_text", ""),
        chat_history_text=_chat_history_section(state),
        question=query,
        language_instruction_text=answer_language_instruction,
    )
    try:
        llm = get_chat_model(runtime.context.model)
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = sanitize_answer_for_query(str(response.content), query)
        return {
            **state,
            "answer": answer,
            "success": True,
            "reasoning": "Generated direct response without movie retrieval.",
        }
    except Exception as exc:
        logger.exception("Failed to generate direct response")
        return {**state, "answer": f"Failed to generate direct response: {exc}", "success": False, "reasoning": str(exc)}
