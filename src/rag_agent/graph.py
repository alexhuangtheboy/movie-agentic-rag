"""Global movie Agentic RAG workflow."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from rag_agent.constants import MOVIE_GRAPH_SCHEMAS, MOVIE_SQL_TABLES
from rag_agent.context import MemoryContext, format_memories_for_prompt, retrieve_memories_for_state
from rag_agent.database.connections import get_checkpoint_database_url
from rag_agent.database.db_prompts import get_graph_schema_prompts, get_sql_schema_prompts
from rag_agent.llm import get_chat_model
from rag_agent.movie_agent.nodes import (
    format_context_node,
    generate_answer_node,
    search_vector_db_node,
    validate_query_node,
)
from rag_agent.movie_agent.states import MovieAgentState
from rag_agent.movie_agent.tools import execute_movie_graph_query, execute_movie_sql_query
from rag_agent.tool_selection_agent import tool_selection_graph
from rag_agent.utils import (
    language_instruction,
    sanitize_answer_for_query,
    strip_code_fence,
    time_node,
    truncate_text,
)


def _with_defaults(state: MovieAgentState) -> MovieAgentState:
    updated = dict(state)
    updated.setdefault("top_k", 5)
    updated.setdefault("sql_table_names", MOVIE_SQL_TABLES)
    updated.setdefault("graph_table_names", MOVIE_GRAPH_SCHEMAS)
    updated.setdefault("similar_movies", [])
    updated.setdefault("user_memories", [])
    updated.setdefault("session_memories", [])
    updated.setdefault("sql_refinement_attempts", 0)
    updated.setdefault("graph_refinement_attempts", 0)
    updated.setdefault("reasoning", "")
    return updated  # type: ignore[return-value]


def _reset_transient_query_state(state: MovieAgentState) -> MovieAgentState:
    """Clear values restored from a previous checkpoint for a new user query."""
    updated = dict(state)
    updated.update(
        {
            "answer": "",
            "reasoning": "",
            "success": None,
            "suggested_tools": "",
            "routing_query": "",
            "routing_confidence": 0.0,
            "routing_reasoning": "",
            "sql_refinement_attempts": 0,
            "graph_refinement_attempts": 0,
            "raw_sql_result": "",
            "raw_graph_result": "",
            "similar_movies": [],
            "context": "",
        }
    )
    return updated  # type: ignore[return-value]


@time_node("initialize_state")
def initialize_state_node(state: MovieAgentState) -> MovieAgentState:
    """Initialize global graph state."""
    query = (state.get("query") or "").strip()
    if not query:
        return {
            **_with_defaults(state),
            "success": False,
            "answer": "Please provide a movie question.",
            "reasoning": "Missing query.",
        }
    return {**_with_defaults(_reset_transient_query_state(state)), "query": query}


@time_node("retrieve_memories")
def retrieve_memories_node(state: MovieAgentState) -> MovieAgentState:
    """Attach user and session memory from SQL Supabase."""
    return _with_defaults(retrieve_memories_for_state(state))  # type: ignore[arg-type]


@time_node("call_tool_selection")
def call_tool_selection_node(state: MovieAgentState) -> MovieAgentState:
    """Route the query to SQL, graph, or vector RAG."""
    routing_input = {
        "question": state.get("query", ""),
        "sql_table_names": state.get("sql_table_names", MOVIE_SQL_TABLES),
        "graph_table_names": state.get("graph_table_names", MOVIE_GRAPH_SCHEMAS),
    }
    result = tool_selection_graph.invoke(routing_input)
    return {
        **state,
        "suggested_tools": result.get("tool", "rag"),
        "routing_query": result.get("query", ""),
        "routing_confidence": float(result.get("confidence") or 0.0),
        "routing_reasoning": result.get("reasoning", ""),
        "reasoning": f"Router selected {result.get('tool', 'rag')}: {result.get('reasoning', '')}",
    }


@time_node("execute_sql_query")
def execute_sql_query_node(state: MovieAgentState) -> MovieAgentState:
    """Execute routed SQL query."""
    query = (state.get("routing_query") or "").strip()
    if not query:
        return {**state, "success": False, "answer": "No SQL query generated.", "reasoning": "Empty SQL route."}
    result = execute_movie_sql_query.invoke({"query": query})
    has_error = isinstance(result, str) and result.startswith("Error")
    has_no_results = isinstance(result, str) and "No rows returned" in result
    return {
        **state,
        "raw_sql_result": str(result),
        "answer": str(result),
        "success": not has_error and not has_no_results,
        "reasoning": f"Executed SQL query. {'No rows returned.' if has_no_results else ''}",
    }


@time_node("execute_graph_query")
def execute_graph_query_node(state: MovieAgentState) -> MovieAgentState:
    """Execute routed Neo4j Cypher query."""
    query = (state.get("routing_query") or "").strip()
    if not query:
        return {**state, "success": False, "answer": "No Cypher query generated.", "reasoning": "Empty graph route."}
    result = execute_movie_graph_query.invoke({"query": query})
    has_error = isinstance(result, str) and result.startswith("Error")
    has_no_results = isinstance(result, str) and "No results returned" in result
    return {
        **state,
        "raw_graph_result": str(result),
        "answer": str(result),
        "success": not has_error and not has_no_results,
        "reasoning": f"Executed graph query. {'No results returned.' if has_no_results else ''}",
    }


def _memory_sections(state: MovieAgentState) -> str:
    memory_texts = format_memories_for_prompt(state)
    return (
        f"User memories:\n{memory_texts.get('user_memories_text', '')}\n\n"
        f"Session memories:\n{memory_texts.get('session_memories_text', '')}"
    )


@time_node("refine_sql_query")
def refine_sql_query_node(state: MovieAgentState, runtime: Runtime[MemoryContext]) -> MovieAgentState:
    """Relax SQL conditions when SQL returns no rows."""
    attempts = int(state.get("sql_refinement_attempts") or 0)
    prompt = f"""A movie SQL query returned no rows. Rewrite it to preserve the user's core intent while relaxing filters.

Relaxation attempt {attempts + 1} of 3:
1. Prefer exact title/person matches -> ILIKE fuzzy matches.
2. Widen numeric thresholds such as "averageRating", "numVotes", "runtimeMinutes", and "startYear" ranges.
3. If still too strict, remove secondary genre/person filters while keeping the main intent.

Original question: {state.get("query", "")}

Current SQL:
{state.get("routing_query", "")}

SQL schema:
{get_sql_schema_prompts(state.get("sql_table_names") or MOVIE_SQL_TABLES)}

Return only executable read-only SQL. Double quote every camelCase PostgreSQL column such as "primaryTitle", "startYear", "averageRating", and "numVotes". Include LIMIT 10 unless the query is aggregate.
"""
    llm = get_chat_model(runtime.context.model)
    response = llm.invoke([HumanMessage(content=prompt)])
    refined = strip_code_fence(str(response.content))
    return {
        **state,
        "routing_query": refined,
        "sql_refinement_attempts": attempts + 1,
        "reasoning": f"Refined SQL query attempt {attempts + 1}/3.",
    }


@time_node("refine_graph_query")
def refine_graph_query_node(state: MovieAgentState, runtime: Runtime[MemoryContext]) -> MovieAgentState:
    """Relax Cypher query when graph returns no results."""
    attempts = int(state.get("graph_refinement_attempts") or 0)
    prompt = f"""A Neo4j Cypher query returned no results. Rewrite it to preserve the user's relationship intent while relaxing filters.

Relaxation attempt {attempts + 1} of 3:
1. Use CONTAINS/toLower for actor and movie names instead of exact equality.
2. Remove optional relationship-property filters such as characters or primaryProfession.
3. Broaden path or relationship constraints while staying within Actor-Movie-ACTED_IN schema.

Original question: {state.get("query", "")}

Current Cypher:
{state.get("routing_query", "")}

Graph schema:
{get_graph_schema_prompts(state.get("graph_table_names") or MOVIE_GRAPH_SCHEMAS)}

Return only executable read-only Neo4j Cypher. Include LIMIT 20 unless the query is aggregate.
"""
    llm = get_chat_model(runtime.context.model)
    response = llm.invoke([HumanMessage(content=prompt)])
    refined = strip_code_fence(str(response.content))
    return {
        **state,
        "routing_query": refined,
        "graph_refinement_attempts": attempts + 1,
        "reasoning": f"Refined graph query attempt {attempts + 1}/3.",
    }


@time_node("refine_sql_result")
def refine_sql_result_node(state: MovieAgentState, runtime: Runtime[MemoryContext]) -> MovieAgentState:
    """Turn raw SQL output into a user-facing answer."""
    answer_language_instruction = language_instruction(state.get("query", ""))
    prompt = f"""{runtime.context.system_prompt}

{_memory_sections(state)}

User question: {state.get("query", "")}

SQL query:
{state.get("routing_query", "")}

SQL result:
{truncate_text(state.get("raw_sql_result") or state.get("answer", ""))}

{answer_language_instruction}
Summarize only facts present in the SQL result.
"""
    try:
        llm = get_chat_model(runtime.context.model)
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = sanitize_answer_for_query(str(response.content), state.get("query", ""))
        return {**state, "answer": answer, "success": True, "reasoning": "Refined SQL result into final answer."}
    except Exception as exc:
        return {**state, "success": True, "reasoning": f"SQL result refinement failed: {exc}"}


@time_node("refine_graph_result")
def refine_graph_result_node(state: MovieAgentState, runtime: Runtime[MemoryContext]) -> MovieAgentState:
    """Turn raw graph output into a user-facing answer."""
    answer_language_instruction = language_instruction(state.get("query", ""))
    prompt = f"""{runtime.context.system_prompt}

{_memory_sections(state)}

User question: {state.get("query", "")}

Cypher query:
{state.get("routing_query", "")}

Graph result:
{truncate_text(state.get("raw_graph_result") or state.get("answer", ""))}

{answer_language_instruction}
Explain relationships clearly and only use facts present in the graph result.
"""
    try:
        llm = get_chat_model(runtime.context.model)
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = sanitize_answer_for_query(str(response.content), state.get("query", ""))
        return {**state, "answer": answer, "success": True, "reasoning": "Refined graph result into final answer."}
    except Exception as exc:
        return {**state, "success": True, "reasoning": f"Graph result refinement failed: {exc}"}


def route_decision(state: MovieAgentState) -> Literal["execute_sql_query", "execute_graph_query", "rag_fallback"]:
    """Route after tool selection."""
    tool = (state.get("suggested_tools") or "rag").lower()
    if tool in {"sql query", "sql"}:
        return "execute_sql_query"
    if tool in {"graph query", "graph"}:
        return "execute_graph_query"
    return "rag_fallback"


def route_after_sql(state: MovieAgentState) -> Literal["refine_sql_result", "refine_sql_query", "rag_fallback"]:
    """Route after SQL execution."""
    if state.get("success"):
        return "refine_sql_result"
    answer = state.get("answer", "")
    attempts = int(state.get("sql_refinement_attempts") or 0)
    if isinstance(answer, str) and "No rows returned" in answer and attempts < 3:
        return "refine_sql_query"
    return "rag_fallback"


def route_after_graph(state: MovieAgentState) -> Literal["refine_graph_result", "refine_graph_query", "rag_fallback"]:
    """Route after graph execution."""
    if state.get("success"):
        return "refine_graph_result"
    answer = state.get("answer", "")
    attempts = int(state.get("graph_refinement_attempts") or 0)
    if isinstance(answer, str) and "No results returned" in answer and attempts < 3:
        return "refine_graph_query"
    return "rag_fallback"


movie_agent_query_builder = StateGraph(MovieAgentState, context_schema=MemoryContext)
movie_agent_query_builder.add_node("initialize_state", initialize_state_node)
movie_agent_query_builder.add_node("retrieve_memories", retrieve_memories_node)
movie_agent_query_builder.add_node("call_tool_selection", call_tool_selection_node)
movie_agent_query_builder.add_node("execute_sql_query", execute_sql_query_node)
movie_agent_query_builder.add_node("refine_sql_query", refine_sql_query_node)
movie_agent_query_builder.add_node("refine_sql_result", refine_sql_result_node)
movie_agent_query_builder.add_node("execute_graph_query", execute_graph_query_node)
movie_agent_query_builder.add_node("refine_graph_query", refine_graph_query_node)
movie_agent_query_builder.add_node("refine_graph_result", refine_graph_result_node)
movie_agent_query_builder.add_node("validate_query", validate_query_node)
movie_agent_query_builder.add_node("search_vector_db", search_vector_db_node)
movie_agent_query_builder.add_node("format_context", format_context_node)
movie_agent_query_builder.add_node("generate_answer", generate_answer_node)

movie_agent_query_builder.add_edge(START, "initialize_state")
movie_agent_query_builder.add_edge("initialize_state", "retrieve_memories")
movie_agent_query_builder.add_edge("retrieve_memories", "call_tool_selection")
movie_agent_query_builder.add_conditional_edges(
    "call_tool_selection",
    route_decision,
    {
        "execute_sql_query": "execute_sql_query",
        "execute_graph_query": "execute_graph_query",
        "rag_fallback": "validate_query",
    },
)
movie_agent_query_builder.add_conditional_edges(
    "execute_sql_query",
    route_after_sql,
    {
        "refine_sql_result": "refine_sql_result",
        "refine_sql_query": "refine_sql_query",
        "rag_fallback": "validate_query",
    },
)
movie_agent_query_builder.add_edge("refine_sql_query", "execute_sql_query")
movie_agent_query_builder.add_edge("refine_sql_result", END)
movie_agent_query_builder.add_conditional_edges(
    "execute_graph_query",
    route_after_graph,
    {
        "refine_graph_result": "refine_graph_result",
        "refine_graph_query": "refine_graph_query",
        "rag_fallback": "validate_query",
    },
)
movie_agent_query_builder.add_edge("refine_graph_query", "execute_graph_query")
movie_agent_query_builder.add_edge("refine_graph_result", END)
movie_agent_query_builder.add_edge("validate_query", "search_vector_db")
movie_agent_query_builder.add_edge("search_vector_db", "format_context")
movie_agent_query_builder.add_edge("format_context", "generate_answer")
movie_agent_query_builder.add_edge("generate_answer", END)

try:
    checkpointer_cm = PostgresSaver.from_conn_string(get_checkpoint_database_url())
    checkpointer = checkpointer_cm.__enter__() if hasattr(checkpointer_cm, "__enter__") else checkpointer_cm
    checkpointer.setup()
    movie_agent_query_graph = movie_agent_query_builder.compile(
        name="movie_agent_query",
        checkpointer=checkpointer,
    )
except Exception as exc:
    print(f"Warning: PostgresSaver unavailable, compiling without checkpointing: {exc}")
    movie_agent_query_graph = movie_agent_query_builder.compile(name="movie_agent_query")

__all__ = ["movie_agent_query_graph"]
