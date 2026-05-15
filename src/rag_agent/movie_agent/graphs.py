"""Movie vector RAG LangGraph workflows."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag_agent.context import Context
from rag_agent.movie_agent.nodes import (
    format_context_node,
    generate_answer_node,
    search_vector_db_node,
    store_movie_embeddings_node,
    validate_query_node,
)
from rag_agent.movie_agent.states import AddDataState, QueryState

movie_rag_add_data_builder = StateGraph(AddDataState)
movie_rag_add_data_builder.add_node("store_movie_embeddings", store_movie_embeddings_node)
movie_rag_add_data_builder.add_edge(START, "store_movie_embeddings")
movie_rag_add_data_builder.add_edge("store_movie_embeddings", END)
movie_rag_add_data_graph = movie_rag_add_data_builder.compile(name="Movie RAG Add Data")

movie_rag_query_builder = StateGraph(QueryState, context_schema=Context)
movie_rag_query_builder.add_node("validate_query", validate_query_node)
movie_rag_query_builder.add_node("search_vector_db", search_vector_db_node)
movie_rag_query_builder.add_node("format_context", format_context_node)
movie_rag_query_builder.add_node("generate_answer", generate_answer_node)
movie_rag_query_builder.add_edge(START, "validate_query")
movie_rag_query_builder.add_edge("validate_query", "search_vector_db")
movie_rag_query_builder.add_edge("search_vector_db", "format_context")
movie_rag_query_builder.add_edge("format_context", "generate_answer")
movie_rag_query_builder.add_edge("generate_answer", END)
movie_rag_query_graph = movie_rag_query_builder.compile(name="Movie RAG Query")

__all__ = ["movie_rag_add_data_graph", "movie_rag_query_graph"]
