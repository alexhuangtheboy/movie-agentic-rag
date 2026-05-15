"""Central configuration constants for the movie Agentic RAG backend."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_MODEL = "qwen-plus"
DEFAULT_LLM_ENDPOINT = "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_ENDPOINT = DEFAULT_LLM_ENDPOINT

MOVIE_SQL_TABLES = [
    "title_content",
    "movie_directors",
    "movie_writers",
    "movie_ratings",
]

MOVIE_GRAPH_SCHEMAS = ["movie_graph"]

FINAL_ANSWER_NODES = {"generate_answer", "refine_sql_result", "refine_graph_result"}


def _normalized_base_url(value: str) -> str:
    """Normalize OpenAI-compatible base URLs without adding duplicate suffixes."""
    value = value.rstrip("/")
    return value


def get_llm_model() -> str:
    """Return configured LLM model name."""
    return os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)


def get_llm_endpoint() -> str:
    """Return configured LLM OpenAI-compatible endpoint."""
    return _normalized_base_url(os.getenv("LLM_ENDPOINT", DEFAULT_LLM_ENDPOINT))


def get_llm_api_key() -> str:
    """Return configured LLM API key."""
    return os.getenv("LLM_API_KEY", "dummy-key")


def get_embedding_model() -> str:
    """Return configured embedding model name."""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_embedding_endpoint() -> str:
    """Return configured embedding OpenAI-compatible endpoint."""
    return _normalized_base_url(os.getenv("EMBEDDING_ENDPOINT", DEFAULT_EMBEDDING_ENDPOINT))


def get_embedding_api_key() -> str:
    """Return configured embedding API key."""
    return os.getenv("EMBEDDING_API_KEY", get_llm_api_key())


def get_movie_vector_table() -> str:
    """Return pgvector table name for movie embeddings."""
    return os.getenv("MOVIE_VECTOR_TABLE", "movie_plot_embeddings")
