"""Database connection string helpers."""

from __future__ import annotations

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()


def get_movie_sql_database_url() -> str:
    """Return SQL Supabase URL for structured movie data."""
    url = os.getenv("MOVIE_SQL_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("MOVIE_SQL_DATABASE_URL is required")
    return url


def get_memory_database_url() -> str:
    """Return Postgres URL for user/session memory tables."""
    return os.getenv("MEMORY_DATABASE_URL") or get_movie_sql_database_url()


def get_checkpoint_database_url() -> str:
    """Return Postgres URL for LangGraph checkpoint tables."""
    url = os.getenv("CHECKPOINT_DATABASE_URL") or get_movie_sql_database_url()
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def get_movie_vector_database_url() -> str:
    """Build the Vector Supabase URL for pgvector retrieval."""
    explicit = os.getenv("MOVIE_VECTOR_DATABASE_URL")
    if explicit:
        return explicit

    host = os.getenv("MOVIE_VECTOR_DB_HOST")
    port = os.getenv("MOVIE_VECTOR_DB_PORT", "5432")
    user = os.getenv("MOVIE_VECTOR_DB_USER")
    password = os.getenv("MOVIE_VECTOR_DB_PW")
    db_name = os.getenv("MOVIE_VECTOR_DB_NAME", "postgres")

    if not all([host, user, password]):
        raise RuntimeError("MOVIE_VECTOR_DB_HOST, MOVIE_VECTOR_DB_USER and MOVIE_VECTOR_DB_PW are required")

    return f"postgresql+psycopg://{quote_plus(user or '')}:{quote_plus(password or '')}@{host}:{port}/{db_name}"
