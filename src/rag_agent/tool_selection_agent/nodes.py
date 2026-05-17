"""Nodes for movie tool selection."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from rag_agent.database.db_prompts import get_graph_schema_prompts, get_sql_schema_prompts
from rag_agent.llm import get_chat_model
from rag_agent.tool_selection_agent.prompts import (
    get_tool_selection_system_prompt,
    get_tool_selection_user_prompt,
)
from rag_agent.tool_selection_agent.states import ToolSelectionState
from rag_agent.utils import strip_code_fence, time_node


@time_node("initialize_tool_selection")
def initialize_tool_selection_node(state: ToolSelectionState) -> ToolSelectionState:
    """Initialize router defaults."""
    return {
        **state,
        "question": (state.get("question") or "").strip(),
        "sql_table_names": state.get("sql_table_names") or [],
        "graph_table_names": state.get("graph_table_names") or [],
        "chat_history_messages": state.get("chat_history_messages") or [],
        "user_memories_text": state.get("user_memories_text") or "",
        "session_memories_text": state.get("session_memories_text") or "",
        "previous_tool": state.get("previous_tool") or "",
        "previous_answer": state.get("previous_answer") or "",
        "tool": state.get("tool") or "",
        "query": state.get("query") or "",
        "confidence": float(state.get("confidence") or 0.0),
        "reasoning": state.get("reasoning") or "",
        "error": None,
    }


def _fallback_route(question: str) -> dict[str, object]:
    lowered = question.lower()
    if any(token in lowered for token in ["actor", "acted", "co-star", "costar", "合作", "演员", "出演"]):
        return {"tool": "graph query", "query": "", "reasoning": "Relationship wording suggests graph retrieval.", "confidence": 0.45}
    if any(token in lowered for token in ["rating", "评分", "director", "导演", "genre", "类型", "year", "年份", "votes"]):
        return {"tool": "sql query", "query": "", "reasoning": "Structured filters suggest SQL retrieval.", "confidence": 0.45}
    return {"tool": "rag", "query": "", "reasoning": "Semantic movie search is the safest fallback.", "confidence": 0.4}


@time_node("select_tool")
def select_tool_node(state: ToolSelectionState) -> ToolSelectionState:
    """Ask the LLM to select a retrieval tool and generate SQL/Cypher when needed."""
    question = state.get("question", "")
    if not question:
        return {**state, "tool": "direct response", "query": "", "reasoning": "No question provided.", "confidence": 0.0}

    sql_schema = get_sql_schema_prompts(state.get("sql_table_names") or [])
    graph_schema = get_graph_schema_prompts(state.get("graph_table_names") or [])
    llm = get_chat_model()
    try:
        response = llm.bind(response_format={"type": "json_object"}).invoke(
            [
                SystemMessage(content=get_tool_selection_system_prompt()),
                HumanMessage(
                    content=get_tool_selection_user_prompt(
                        sql_schema,
                        graph_schema,
                        question,
                        chat_history_messages=state.get("chat_history_messages") or [],
                        user_memories_text=state.get("user_memories_text") or "",
                        session_memories_text=state.get("session_memories_text") or "",
                        previous_tool=state.get("previous_tool") or "",
                        previous_answer=state.get("previous_answer") or "",
                    )
                ),
            ]
        )
        payload = json.loads(strip_code_fence(str(response.content)))
        tool = payload.get("tool") or "rag"
        query = payload.get("query") or ""
        if tool in {"rag", "direct response"}:
            query = ""
        return {
            **state,
            "tool": tool,
            "query": query,
            "reasoning": payload.get("reasoning") or "",
            "confidence": float(payload.get("confidence") or 0.0),
        }
    except Exception as exc:
        fallback = _fallback_route(question)
        return {**state, **fallback, "error": str(exc)}
