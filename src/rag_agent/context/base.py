"""LangGraph runtime context."""

from __future__ import annotations

from dataclasses import dataclass

from rag_agent.constants import get_llm_model


@dataclass
class Context:
    """Runtime configuration passed into LangGraph nodes."""

    model: str = get_llm_model()
    top_k: int = 5
    num_ctx: int = 8192
    system_prompt: str = (
        "You are a movie Agentic RAG assistant. Answer using the retrieved SQL, graph, "
        "vector, and memory context. Be concise, mention uncertainty, and do not invent "
        "database facts that are not present in tool results. Answer in the same language "
        "as the user's question."
    )
