"""FastAPI entrypoint for the movie Agentic RAG backend."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Literal

import requests
from fastapi import Body, FastAPI
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from neo4j import GraphDatabase
from openai import OpenAI
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from rag_agent.constants import (
    FINAL_ANSWER_NODES,
    MOVIE_GRAPH_SCHEMAS,
    MOVIE_SQL_TABLES,
    get_embedding_api_key,
    get_embedding_endpoint,
    get_embedding_model,
    get_llm_api_key,
    get_llm_endpoint,
    get_llm_model,
    get_movie_vector_table,
)
from rag_agent.context import MemoryContext
from rag_agent.database.connections import (
    get_checkpoint_database_url,
    get_movie_sql_database_url,
    get_movie_vector_database_url,
)
from rag_agent.graph import movie_agent_query_graph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Movie Agentic RAG API",
    version="0.1.0",
    description="Movie backend with SQL, Vector, Graph, smart routing, self-correction, and memory.",
)


class QueryRequest(BaseModel):
    """Request body for the movie query endpoint."""

    query: str = Field(..., description="User movie question")
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = Field(default=False, description="Return immediately and push progress/final answer to webhook")
    chat_id: str | None = None
    user_id: str | None = None
    chat_history_messages: list[str] = Field(default_factory=list)
    target_reply_message: str | None = None
    bot_user_id: str | None = None


class QueryResponse(BaseModel):
    """Response body for wait and stream modes."""

    task_id: str
    thread_id: str | None = None
    status: Literal["running", "completed", "failed"]
    answer: str | None = None
    reasoning: str | None = None
    success: bool | None = None
    routing: dict[str, Any] | None = None


class WebhookPayload(BaseModel):
    """Payload posted to WEBHOOK_URL."""

    task_id: str
    thread_id: str | None = None
    status: Literal["running", "completed", "failed"]
    answer: str | None = None
    reasoning: str | None = None
    success: bool | None = None
    node: str | None = None
    routing: dict[str, Any] | None = None


class RunStatusResponse(BaseModel):
    """Best-effort run status from LangGraph checkpoints."""

    run_id: str
    thread_id: str | None = None
    status: Literal["running", "completed", "failed", "not_found"]
    state: dict[str, Any] | None = None
    answer: str | None = None
    reasoning: str | None = None
    error: str | None = None


class DependencyCheck(BaseModel):
    """Status for one external dependency."""

    status: Literal["ok", "failed"]
    latency_ms: int | None = None
    detail: str | None = None


def _error_message(exc: BaseException) -> str:
    """Return a useful message even for empty ExceptionGroup/CancelledError values."""
    if isinstance(exc, TimeoutError):
        return "Request timed out while waiting for the movie agent graph."
    if hasattr(exc, "exceptions"):
        messages = [str(item) or item.__class__.__name__ for item in getattr(exc, "exceptions", [])]
        if messages:
            return "; ".join(messages)
    return str(exc) or exc.__class__.__name__


def _dependency_timeout_seconds() -> float:
    return float(os.getenv("DEPENDENCY_CHECK_TIMEOUT_SECONDS", "15"))


def _graph_timeout_seconds() -> float:
    return float(os.getenv("GRAPH_TIMEOUT_SECONDS", "90"))


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """Health endpoint."""
    return {
        "status": "healthy",
        "service": "movie-agentic-rag",
        "webhook_configured": bool(os.getenv("WEBHOOK_URL")),
        "graphs": ["movie_agent_query", "movie_rag_add_data", "movie_rag_query"],
    }


async def _run_dependency_check(name: str, check_func) -> tuple[str, DependencyCheck]:
    import time

    start = time.perf_counter()
    try:
        detail = await asyncio.wait_for(
            asyncio.to_thread(check_func),
            timeout=_dependency_timeout_seconds(),
        )
        return name, DependencyCheck(
            status="ok",
            latency_ms=int((time.perf_counter() - start) * 1000),
            detail=detail,
        )
    except Exception as exc:
        return name, DependencyCheck(
            status="failed",
            latency_ms=int((time.perf_counter() - start) * 1000),
            detail=_error_message(exc),
        )


def _check_llm() -> str:
    client = OpenAI(
        base_url=get_llm_endpoint(),
        api_key=get_llm_api_key(),
        timeout=_dependency_timeout_seconds(),
    )
    response = client.chat.completions.create(
        model=get_llm_model(),
        messages=[{"role": "user", "content": "Reply with OK."}],
        max_tokens=4,
        temperature=0,
    )
    return str(response.choices[0].message.content)


def _check_embedding() -> str:
    client = OpenAI(
        base_url=get_embedding_endpoint(),
        api_key=get_embedding_api_key(),
        timeout=_dependency_timeout_seconds(),
    )
    response = client.embeddings.create(model=get_embedding_model(), input=["ping"])
    return f"{len(response.data[0].embedding)} dimensions"


def _check_sql() -> str:
    timeout = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10"))
    engine = create_engine(get_movie_sql_database_url(), connect_args={"connect_timeout": timeout})
    with engine.connect() as conn:
        value = conn.execute(text("SELECT 1")).scalar_one()
    return f"SELECT {value}"


def _check_vector() -> str:
    timeout = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10"))
    engine = create_engine(get_movie_vector_database_url(), connect_args={"connect_timeout": timeout})
    table_name = get_movie_vector_table()
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT to_regclass(:table_name)"), {"table_name": table_name}).scalar_one()
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :table_name"
            ),
            {"table_name": table_name},
        ).all()
    if not exists:
        raise RuntimeError(f"Vector table not found: {table_name}")
    columns = {row[0] for row in rows}
    has_text = bool({"content", "plot"} & columns)
    has_embedding = bool({"embedding", "plot_embedding"} & columns)
    if "movie_id" not in columns or not has_text or not has_embedding:
        raise RuntimeError(
            f"Vector table {table_name} has incompatible columns: {sorted(columns)}. "
            "Expected movie_id plus one text column (content or plot) and one vector column (embedding or plot_embedding)."
        )
    return f"table {table_name} exists with columns {sorted(columns)}"


def _check_neo4j() -> str:
    uri = os.getenv("NEO4J_URL")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE")
    if not all([uri, username, password]):
        raise RuntimeError("NEO4J_URL, NEO4J_USERNAME and NEO4J_PASSWORD are required")
    driver = GraphDatabase.driver(
        uri,
        auth=(username, password),
        connection_timeout=_dependency_timeout_seconds(),
    )
    try:
        with driver.session(database=database) as session:
            value = session.run("RETURN 1 AS ok").single()["ok"]
        return f"RETURN {value}"
    finally:
        driver.close()


@app.get("/health/dependencies", tags=["Health"])
async def dependency_health_check() -> dict[str, DependencyCheck]:
    """Check external services independently so failures are easy to isolate."""
    # Run OpenAI SDK checks sequentially. On Python 3.14 the SDK's lazy imports can
    # deadlock if chat and embedding resources are imported concurrently.
    checks: list[tuple[str, DependencyCheck]] = [
        await _run_dependency_check("llm", _check_llm),
        await _run_dependency_check("embedding", _check_embedding),
    ]
    database_checks = await asyncio.gather(
        _run_dependency_check("sql", _check_sql),
        _run_dependency_check("vector", _check_vector),
        _run_dependency_check("neo4j", _check_neo4j),
    )
    checks.extend(database_checks)
    return dict(checks)


@app.get("/", tags=["Info"])
async def root() -> dict[str, Any]:
    """Return available endpoints."""
    return {
        "name": "Movie Agentic RAG API",
        "custom_endpoints": {
            "query": "/movie/query",
            "health": "/health",
            "dependency_health": "/health/dependencies",
            "run_status": "/runs/{run_id}/status?thread_id=...",
        },
        "langgraph_assistants": {
            "movie_agent_query": "/assistants/movie_agent_query/runs/invoke",
            "movie_agent_query_stream": "/assistants/movie_agent_query/runs/stream",
            "movie_rag_add_data": "/assistants/movie_rag_add_data/runs/invoke",
            "movie_rag_query": "/assistants/movie_rag_query/runs/invoke",
        },
    }


_checkpointer: Any | None = None


def get_checkpointer() -> Any | None:
    """Return a singleton PostgresSaver if available."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    try:
        cm = PostgresSaver.from_conn_string(get_checkpoint_database_url())
        saver = cm.__enter__() if hasattr(cm, "__enter__") else cm
        saver.setup()
        _checkpointer = saver
        return saver
    except Exception as exc:
        logger.error("Failed to initialize checkpointer: %s", exc)
        return None


@app.get("/runs/{run_id}/status", tags=["Runs"])
async def get_run_status(run_id: str, thread_id: str | None = None) -> RunStatusResponse:
    """Read the latest checkpoint for a thread."""
    checkpointer = get_checkpointer()
    if not checkpointer:
        return RunStatusResponse(run_id=run_id, thread_id=thread_id, status="not_found", error="Checkpointer unavailable")
    if not thread_id:
        return RunStatusResponse(run_id=run_id, thread_id=None, status="not_found", error="thread_id is required")
    try:
        config = RunnableConfig(configurable={"thread_id": thread_id})
        checkpoint = checkpointer.get(config)
        if not checkpoint:
            return RunStatusResponse(run_id=run_id, thread_id=thread_id, status="not_found", error="Checkpoint not found")
        state = checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else {}
        status: Literal["running", "completed", "failed"] = "running"
        if isinstance(state, dict) and state.get("answer"):
            status = "completed"
        if isinstance(state, dict) and state.get("error"):
            status = "failed"
        return RunStatusResponse(
            run_id=run_id,
            thread_id=thread_id,
            status=status,
            state=state if isinstance(state, dict) else None,
            answer=state.get("answer") if isinstance(state, dict) else None,
            reasoning=state.get("reasoning") if isinstance(state, dict) else None,
            error=state.get("error") if isinstance(state, dict) else None,
        )
    except Exception as exc:
        return RunStatusResponse(run_id=run_id, thread_id=thread_id, status="failed", error=str(exc))


def _extract_answer_from_output(output: Any) -> tuple[str | None, str | None, bool | None, dict[str, Any] | None]:
    if isinstance(output, AIMessage):
        return str(output.content), None, True, None
    if isinstance(output, dict):
        routing = {
            "tool": output.get("suggested_tools"),
            "query": output.get("routing_query"),
            "confidence": output.get("routing_confidence"),
            "reasoning": output.get("routing_reasoning"),
        }
        return output.get("answer"), output.get("reasoning"), output.get("success"), routing
    return None, None, None, None


async def _post_webhook(payload: WebhookPayload) -> None:
    webhook_url = os.getenv("WEBHOOK_URL")
    if not webhook_url:
        return
    try:
        await asyncio.to_thread(
            lambda: requests.post(
                webhook_url,
                json=payload.model_dump(),
                headers={"Content-Type": "application/json"},
                timeout=10,
            ).raise_for_status()
        )
    except Exception as exc:
        logger.error("Webhook failed: %s", exc)


@app.post("/movie/query", tags=["Movie"])
async def movie_query(request: QueryRequest = Body(...)) -> QueryResponse:
    """Run the global movie agent query graph."""
    run_id = uuid.uuid4()
    thread_id = request.chat_id or str(run_id)
    initial_state: dict[str, Any] = {
        "query": request.query,
        "top_k": request.top_k,
        "sql_table_names": MOVIE_SQL_TABLES,
        "graph_table_names": MOVIE_GRAPH_SCHEMAS,
        "chat_history_messages": request.chat_history_messages,
        "target_reply_message": request.target_reply_message,
        "user_id": request.user_id,
        "chat_id": request.chat_id,
        "reasoning": "",
        "similar_movies": [],
    }
    context = MemoryContext(user_id=request.user_id, chat_id=request.chat_id, top_k=request.top_k)
    config = RunnableConfig(run_id=run_id, configurable={"thread_id": thread_id})

    async def process_invoke_wait() -> QueryResponse:
        """Run the graph via synchronous invoke for wait mode.

        The graph is compiled with sync PostgresSaver checkpointing. LangGraph's
        async event streaming path requires async checkpoint methods, so wait mode
        uses invoke in a worker thread to keep checkpointing enabled.
        """
        try:
            async with asyncio.timeout(_graph_timeout_seconds()):
                result = await asyncio.to_thread(
                    lambda: movie_agent_query_graph.invoke(
                        initial_state,
                        config=config,
                        context=context,
                    )
                )
            answer, reasoning, success, routing = _extract_answer_from_output(result)
            return QueryResponse(
                task_id=str(run_id),
                thread_id=thread_id,
                status="completed" if success is not False else "failed",
                answer=answer,
                reasoning=reasoning,
                success=success,
                routing=routing,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            error = _error_message(exc)
            logger.exception("Movie query failed: %s", error)
            return QueryResponse(
                task_id=str(run_id),
                thread_id=thread_id,
                status="failed",
                answer=error,
                reasoning=error,
                success=False,
            )

    async def process_events() -> QueryResponse:
        final_answer: str | None = None
        final_reasoning: str | None = None
        final_success: bool | None = None
        final_routing: dict[str, Any] | None = None
        final_node: str | None = None

        try:
            async with asyncio.timeout(_graph_timeout_seconds()):
                async for event in movie_agent_query_graph.astream_events(
                    initial_state,
                    config=config,
                    context=context,
                    version="v2",
                ):
                    node = (event.get("metadata") or {}).get("langgraph_node")
                    event_type = event.get("event")
                    output = (event.get("data") or {}).get("output")

                    if event_type == "on_chain_end" and node in FINAL_ANSWER_NODES:
                        answer, reasoning, success, routing = _extract_answer_from_output(output)
                        if answer:
                            final_answer = answer
                            final_reasoning = reasoning or final_reasoning
                            final_success = success
                            final_routing = routing or final_routing
                            final_node = node
                    elif request.stream and event_type == "on_chain_end" and isinstance(output, dict):
                        reasoning = output.get("reasoning")
                        if reasoning:
                            await _post_webhook(
                                WebhookPayload(
                                    task_id=str(run_id),
                                    thread_id=thread_id,
                                    status="running",
                                    reasoning=reasoning,
                                    node=node,
                                )
                            )

            payload = WebhookPayload(
                task_id=str(run_id),
                thread_id=thread_id,
                status="completed",
                answer=final_answer,
                reasoning=final_reasoning,
                success=final_success,
                node=final_node,
                routing=final_routing,
            )
            await _post_webhook(payload)
            return QueryResponse(
                task_id=str(run_id),
                thread_id=thread_id,
                status="completed",
                answer=final_answer,
                reasoning=final_reasoning,
                success=final_success,
                routing=final_routing,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            error = _error_message(exc)
            logger.exception("Movie query failed: %s", error)
            payload = WebhookPayload(
                task_id=str(run_id),
                thread_id=thread_id,
                status="failed",
                answer=error,
                reasoning=error,
                success=False,
            )
            await _post_webhook(payload)
            return QueryResponse(
                task_id=str(run_id),
                thread_id=thread_id,
                status="failed",
                answer=error,
                reasoning=error,
                success=False,
            )

    if request.stream:
        asyncio.create_task(process_events())
        return QueryResponse(task_id=str(run_id), thread_id=thread_id, status="running")

    return await process_invoke_wait()
