"""Nodes for movie tool selection."""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from rag_agent.database.db_prompts import get_graph_schema_prompts, get_sql_schema_prompts
from rag_agent.llm import get_chat_model
from rag_agent.tool_selection_agent.prompts import (
    get_tool_selection_system_prompt,
    get_tool_selection_user_prompt,
)
from rag_agent.tool_selection_agent.states import ToolSelectionState
from rag_agent.utils import strip_code_fence, time_node

logger = logging.getLogger(__name__)

ACTOR_CAST_LOOKUP_PATTERNS = [
    re.compile(r'\bwho\s+(?:acted|acts|starred|stars?)\s+in\s+["\']?(.+?)["\']?\s*[?.!]?$', re.IGNORECASE),
    re.compile(r'\b(?:actors?|cast)\s+(?:in|of|for)\s+["\']?(.+?)["\']?\s*[?.!]?$', re.IGNORECASE),
]


class ToolSelectionPayload(BaseModel):
    """Validated JSON payload returned by the routing LLM."""

    tool: Literal["sql query", "graph query", "rag", "direct response"] = "rag"
    query: str = ""
    reasoning: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


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
    if any(token in lowered for token in ["actor", "cast", "acted", "starred", "stars", "co-star", "costar", "合作", "演员", "出演"]):
        return {"tool": "graph query", "query": "", "reasoning": "Relationship wording suggests graph retrieval.", "confidence": 0.45}
    if any(token in lowered for token in ["rating", "评分", "director", "导演", "genre", "类型", "year", "年份", "votes"]):
        return {"tool": "sql query", "query": "", "reasoning": "Structured filters suggest SQL retrieval.", "confidence": 0.45}
    return {"tool": "rag", "query": "", "reasoning": "Semantic movie search is the safest fallback.", "confidence": 0.4}


def _actor_cast_movie_title(question: str) -> str:
    """Extract a likely movie title from direct actor/cast lookup wording."""
    for pattern in ACTOR_CAST_LOOKUP_PATTERNS:
        match = pattern.search(question)
        if not match:
            continue
        title = match.group(1).strip().strip("\"'")
        lowered = title.lower()
        if not title or lowered.startswith(("movies ", "films ")):
            return ""
        return title
    return ""


def _actor_cast_lookup_cypher(title: str) -> str:
    title_literal = json.dumps(title)
    return f"""MATCH (a:Actor)-[r:ACTED_IN]->(m:Movie)
WHERE toLower(m.primaryTitle) CONTAINS toLower({title_literal})
RETURN a.primaryName AS actor, m.primaryTitle AS movie, r.characters AS characters, r.primaryProfession AS primaryProfession
LIMIT 20"""


def _fallback_with_error(question: str, exc: Exception, raw_output: str = "") -> dict[str, object]:
    fallback = _fallback_route(question)
    logger.warning(
        "Tool selection fallback. question=%r raw_output=%r fallback_tool=%r",
        question,
        raw_output,
        fallback.get("tool"),
        exc_info=True,
    )
    return {
        **fallback,
        "error": str(exc),
        "reasoning": f"{fallback.get('reasoning')} Router fallback after error: {exc}",
    }


@time_node("select_tool")
def select_tool_node(state: ToolSelectionState) -> ToolSelectionState:
    """Ask the LLM to select a retrieval tool and generate SQL/Cypher when needed."""
    question = state.get("question", "")
    if not question:
        return {**state, "tool": "direct response", "query": "", "reasoning": "No question provided.", "confidence": 0.0}

    sql_schema = get_sql_schema_prompts(state.get("sql_table_names") or [])
    graph_schema = get_graph_schema_prompts(state.get("graph_table_names") or [])
    llm = get_chat_model()
    raw_output = ""
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
        raw_output = strip_code_fence(str(response.content))
        payload = ToolSelectionPayload.model_validate(json.loads(raw_output))
        tool = payload.tool
        query = payload.query
        actor_cast_title = _actor_cast_movie_title(question)
        if actor_cast_title and tool != "graph query":
            tool = "graph query"
            query = _actor_cast_lookup_cypher(actor_cast_title)
            reasoning = "Actor/cast lookup must use the ACTED_IN graph relationship."
            confidence = max(payload.confidence, 0.9)
        else:
            reasoning = payload.reasoning
            confidence = payload.confidence
        if tool in {"rag", "direct response"}:
            query = ""
        return {
            **state,
            "tool": tool,
            "query": query,
            "reasoning": reasoning,
            "confidence": confidence,
        }
    except (json.JSONDecodeError, ValidationError) as exc:
        return {**state, **_fallback_with_error(question, exc, raw_output)}
    except Exception as exc:
        return {**state, **_fallback_with_error(question, exc, raw_output)}
