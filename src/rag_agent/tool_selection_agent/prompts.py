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

For mixed questions, prefer SQL first when there are hard filters such as year, rating, genre, director, writer, votes, or title. Vector RAG can be used later if SQL returns no results.

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


def get_tool_selection_user_prompt(sql_schema: str, graph_schema: str, question: str) -> str:
    """Return the routing user prompt."""
    return f"""SQL schema:
{sql_schema}

Graph schema:
{graph_schema}

Question:
{question}
"""
