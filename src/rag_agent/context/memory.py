"""User and session memory helpers backed by SQL Supabase."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine, text

from rag_agent.context.base import Context
from rag_agent.database.connections import get_memory_database_url

logger = logging.getLogger(__name__)


def _memory_engine():
    timeout = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10"))
    return create_engine(get_memory_database_url(), connect_args={"connect_timeout": timeout})


def _memory_auto_create_tables() -> bool:
    return os.getenv("MEMORY_AUTO_CREATE_TABLES", "false").lower() == "true"


def _memory_max_items(kind: str) -> int:
    env_name = "USER_MEMORY_MAX_ITEMS" if kind == "user" else "SESSION_MEMORY_MAX_ITEMS"
    return int(os.getenv(env_name, "200" if kind == "user" else "100"))


@dataclass
class MemoryContext(Context):
    """Runtime context with optional user/session identity."""

    user_id: str | None = None
    chat_id: str | None = None


def ensure_memory_tables() -> None:
    """Create memory tables if they do not exist."""
    engine = _memory_engine()
    ddl = """
    CREATE TABLE IF NOT EXISTS user_memories (
        memory_id text PRIMARY KEY,
        user_id text NOT NULL,
        content text NOT NULL,
        metadata jsonb DEFAULT '{}'::jsonb,
        created_at timestamptz DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_user_memories_user_id ON user_memories(user_id);

    CREATE TABLE IF NOT EXISTS session_memories (
        memory_id text PRIMARY KEY,
        chat_id text NOT NULL,
        content text NOT NULL,
        metadata jsonb DEFAULT '{}'::jsonb,
        created_at timestamptz DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_session_memories_chat_id ON session_memories(chat_id);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _maybe_ensure_memory_tables() -> None:
    if _memory_auto_create_tables():
        ensure_memory_tables()


def _prune_memories(table: str, key: str, value: str, max_items: int) -> None:
    """Keep only the newest max_items rows for one memory owner."""
    if max_items <= 0:
        return
    engine = _memory_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                f"DELETE FROM {table} WHERE {key} = :value AND memory_id NOT IN ("
                f"SELECT memory_id FROM {table} WHERE {key} = :value "
                "ORDER BY created_at DESC LIMIT :max_items)"
            ),
            {"value": value, "max_items": max_items},
        )


def add_user_memory(user_id: str, content: str, metadata: dict[str, Any] | None = None) -> str:
    """Store one user memory and return its ID."""
    _maybe_ensure_memory_tables()
    memory_id = str(uuid.uuid4())
    engine = _memory_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO user_memories(memory_id, user_id, content, metadata) "
                "VALUES (:memory_id, :user_id, :content, CAST(:metadata AS jsonb))"
            ),
            {
                "memory_id": memory_id,
                "user_id": user_id,
                "content": content,
                "metadata": json.dumps(metadata or {}),
            },
        )
    _prune_memories("user_memories", "user_id", user_id, _memory_max_items("user"))
    return memory_id


def add_session_memory(chat_id: str, content: str, metadata: dict[str, Any] | None = None) -> str:
    """Store one session memory and return its ID."""
    _maybe_ensure_memory_tables()
    memory_id = str(uuid.uuid4())
    engine = _memory_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO session_memories(memory_id, chat_id, content, metadata) "
                "VALUES (:memory_id, :chat_id, :content, CAST(:metadata AS jsonb))"
            ),
            {
                "memory_id": memory_id,
                "chat_id": chat_id,
                "content": content,
                "metadata": json.dumps(metadata or {}),
            },
        )
    _prune_memories("session_memories", "chat_id", chat_id, _memory_max_items("session"))
    return memory_id


def _fetch_memories(table: str, key: str, value: str, limit: int = 8) -> list[dict[str, Any]]:
    engine = _memory_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT memory_id, content, metadata, created_at FROM {table} "
                f"WHERE {key} = :value ORDER BY created_at DESC LIMIT :limit"
            ),
            {"value": value, "limit": limit},
        ).mappings().all()
    return [dict(row) for row in rows]


def retrieve_memories_for_state(state: dict[str, Any]) -> dict[str, Any]:
    """Attach user and session memories to a graph state when IDs are available."""
    try:
        updated = dict(state)
        user_id = state.get("user_id")
        chat_id = state.get("chat_id")
        if user_id:
            updated["user_memories"] = _fetch_memories("user_memories", "user_id", user_id)
        if chat_id:
            updated["session_memories"] = _fetch_memories("session_memories", "chat_id", chat_id)
        return updated
    except Exception as exc:
        logger.exception("Failed to retrieve memories")
        updated = dict(state)
        updated["memory_error"] = str(exc)
        updated.setdefault("user_memories", [])
        updated.setdefault("session_memories", [])
        return updated


def format_memories_for_prompt(state: dict[str, Any]) -> dict[str, str]:
    """Format retrieved memories for prompts."""
    user_memories = state.get("user_memories") or []
    session_memories = state.get("session_memories") or []
    user_text = "\n".join(f"- {item.get('content')}" for item in user_memories if item.get("content"))
    session_text = "\n".join(f"- {item.get('content')}" for item in session_memories if item.get("content"))
    return {"user_memories_text": user_text, "session_memories_text": session_text}
