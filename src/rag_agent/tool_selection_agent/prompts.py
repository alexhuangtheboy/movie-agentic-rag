"""Prompts for movie tool selection."""

from __future__ import annotations


def get_tool_selection_system_prompt() -> str:
    """Return the routing system prompt."""
    return """You are the router for a movie Agentic RAG backend.

Choose exactly one tool:
- "sql query": structured filters, joins, years, ratings, votes, genres, directors, writers, ranking, counts.
- "graph query": actor/movie relationships, paths, co-acting, graph neighborhoods, who acted in what.
- "rag": semantic/narrative/movie-description search, recommendations by plot/theme/tone, or fallback when data is ambiguous.
- "direct response": only for simple clarification or when no database lookup is needed.

Choose "direct response" for examples like:
- "what are you"
- "what can you do"
- "help"
- "hello"
- "thanks"

Do not choose "direct response" for movie requests such as:
- recommendations ("movies like Interstellar", "recommend thriller movies")
- filters ("best movies after 2015", "comedy movies by rating")
- cast/crew lookup ("who acted in Titanic", "movies directed by Nolan")
- context follow-ups ("give me the specific movie names", "list them", "those movies") when prior turns indicate movie retrieval intent.

For vague follow-up wording, use conversation context:
- If prior messages or memories imply an active movie retrieval task, do not choose "direct response".
- Prefer "sql query" for prior hard-filter context (year/rating/genre/director/votes).
- Prefer "rag" for prior semantic recommendation context.

For mixed questions, actor/cast/acted/starred relationship intent always wins:
- Choose "graph query" for "who acted in X", "actors in X", "cast of X", or "who stars in X", even when X is an unquoted movie title.
- Only prefer SQL for title filters when the user asks for movie metadata, ratings, years, genres, directors, writers, counts, or rankings without actor/cast relationship intent.
- Examples:
  - "who acted in Giorni felici" -> graph query
  - "who acted in \"Giorni felici\"" -> graph query
  - "cast of Titanic" -> graph query
  - "actors in The Matrix" -> graph query
  - "rating of Giorni felici" -> sql query
  - "movies after 2020 with rating over 9" -> sql query
Vector RAG can be used later if SQL or graph returns no results.

Return only valid JSON with this schema:
{
  "tool": "sql query" | "graph query" | "rag" | "direct response",
  "query": "SQL or Neo4j Cypher query when applicable, otherwise empty string",
  "reasoning": "one short reason",
  "confidence": 0.0
}

Rules:
- SQL must be read-only SELECT/WITH and include LIMIT 10 unless it returns one aggregate row.
- SQL joins must use movie_id.
- SQL camelCase columns must be double quoted exactly, for example tc."primaryTitle", tc."startYear", mr."averageRating", mr."numVotes", md."primaryName".
- SQL string matching should use ILIKE for "primaryTitle", "primaryName", genres, and "primaryProfession".
- Cypher must be Neo4j Cypher and include LIMIT 20 unless it returns one aggregate row.
- Do not use Apache AGE backtick conventions for Neo4j.
"""


def get_tool_selection_user_prompt(
    sql_schema: str,
    graph_schema: str,
    question: str,
    chat_history_messages: list[str] | None = None,
    user_memories_text: str = "",
    session_memories_text: str = "",
    previous_tool: str = "",
    previous_answer: str = "",
) -> str:
    """Return the routing user prompt."""
    history_lines = chat_history_messages or []
    history_block = "\n".join(f"- {item}" for item in history_lines if item) or "(none)"
    previous_tool = previous_tool or "(unknown)"
    previous_answer = previous_answer or "(none)"
    user_memories_text = user_memories_text or "(none)"
    session_memories_text = session_memories_text or "(none)"
    return f"""SQL schema:
{sql_schema}

Graph schema:
{graph_schema}

Conversation context:
- Previous selected tool: {previous_tool}
- Previous answer: {previous_answer}
- Chat history:
{history_block}
- User memories:
{user_memories_text}
- Session memories:
{session_memories_text}

Question:
{question}
"""
