"""Movie SQL and graph schema prompts for routing and query generation."""

MOVIE_SQL_SCHEMA = """
CREATE TABLE title_content (
    movie_id text PRIMARY KEY,
    "titleType" text,
    "primaryTitle" text,
    "originalTitle" text,
    "isAdult" boolean,
    "startYear" integer,
    "endYear" integer,
    "runtimeMinutes" integer,
    genres text
);

CREATE TABLE movie_directors (
    movie_id text REFERENCES title_content(movie_id),
    nconst text,
    "primaryName" text,
    "birthYear" integer,
    "deathYear" integer,
    "primaryProfession" text
);

CREATE TABLE movie_writers (
    movie_id text REFERENCES title_content(movie_id),
    nconst text,
    "primaryName" text,
    "birthYear" integer,
    "deathYear" integer,
    "primaryProfession" text
);

CREATE TABLE movie_ratings (
    movie_id text REFERENCES title_content(movie_id),
    "averageRating" numeric,
    "numVotes" integer
);

SQL rules:
- All joins must use movie_id.
- IMPORTANT: PostgreSQL camelCase columns must always be double quoted:
  "titleType", "primaryTitle", "originalTitle", "isAdult", "startYear",
  "endYear", "runtimeMinutes", "primaryName", "birthYear", "deathYear",
  "primaryProfession", "averageRating", "numVotes".
- Use ILIKE for "primaryTitle", "originalTitle", "primaryName", genres, and "primaryProfession".
- genres is a text field, so genre filters must use ILIKE, for example genres ILIKE '%Action%'.
- "startYear", "endYear", "runtimeMinutes", "averageRating", and "numVotes" are numeric fields.
- Always include LIMIT 10 for non-aggregate result sets.
- Prefer selecting "primaryTitle", "startYear", genres, "averageRating", "numVotes", and relevant people names.
"""

MOVIE_GRAPH_SCHEMA = """
Neo4j movie graph schema.

Nodes:
- (:Actor {nconst, primaryName, birthYear, deathYear, primaryProfession})
- (:Movie {movie_id, primaryTitle})

Relationships:
- (:Actor)-[:ACTED_IN {characters, primaryProfession}]->(:Movie)

Cypher rules:
- Use Neo4j Cypher, not Apache AGE syntax.
- Do not wrap labels or relationship types in backticks unless the name contains special characters.
- Actor optional properties may include primaryName, birthYear, deathYear, primaryProfession.
- Movie optional properties may include primaryTitle.
- Always include LIMIT 20 for non-aggregate result sets.
- Prefer returning actor names, movie titles, character names, and relationship properties.
"""

VECTOR_SCHEMA = """
Vector schema:
- Table: movie_plot_embeddings
- movie_id links to title_content.movie_id.
- content contains primaryTitle, genres, and descriptive narrative text.
- embedding uses vector(1536) for DashScope text-embedding-v4.
"""

SQL_TABLE_SCHEMAS_MAP: dict[str, str] = {
    "title_content": MOVIE_SQL_SCHEMA,
    "movie_directors": "",
    "movie_writers": "",
    "movie_ratings": "",
}

GRAPH_TABLE_SCHEMAS_MAP: dict[str, str] = {
    "movie_graph": MOVIE_GRAPH_SCHEMA,
}


def get_sql_schema_prompts(table_names: list[str]) -> str:
    """Return SQL schemas for requested table names."""
    if not table_names:
        return MOVIE_SQL_SCHEMA
    chunks: list[str] = []
    seen = False
    for table_name in table_names:
        schema = SQL_TABLE_SCHEMAS_MAP.get(table_name)
        if schema and not seen:
            chunks.append(schema)
            seen = True
    return "\n\n".join(chunks) or MOVIE_SQL_SCHEMA


def get_graph_schema_prompts(table_names: list[str]) -> str:
    """Return graph schemas for requested graph names."""
    if not table_names:
        return MOVIE_GRAPH_SCHEMA
    chunks = [GRAPH_TABLE_SCHEMAS_MAP[name] for name in table_names if name in GRAPH_TABLE_SCHEMAS_MAP]
    return "\n\n".join(chunks) or MOVIE_GRAPH_SCHEMA
