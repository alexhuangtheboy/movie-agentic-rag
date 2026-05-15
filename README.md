# Movie Agentic RAG Backend

Standalone FastAPI + LangGraph backend for movie Agentic RAG.

It supports:

- SQL retrieval over Supabase tables: `title_content`, `movie_directors`, `movie_writers`, `movie_ratings`
- Vector retrieval over Supabase pgvector using DashScope `text-embedding-v4`
- Graph retrieval over Neo4j using `Actor`, `Movie`, and `ACTED_IN`
- Smart routing, SQL/Graph self-correction loops, user/session memory, and LangGraph checkpointing
- Webhook streaming from `/movie/query`

## Setup

```powershell
cd C:\Users\alexh\Desktop\movie-agentic-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Copy `.env.example` to `.env` if needed. The local `.env` contains real credentials and is ignored by git.

## Database Initialization

Run these scripts against the appropriate databases:

- `database/init_movie_sql.sql` against the SQL Supabase database
- `database/init_vector.sql` against the Vector Supabase database
- `database/init_memory.sql` against the SQL Supabase database

LangGraph checkpoint tables are created by `PostgresSaver.setup()` at runtime.

Neo4j should contain:

```cypher
(:Actor {nconst, primaryName, birthYear, deathYear, primaryProfession})
(:Movie {movie_id, primaryTitle})
(:Actor)-[:ACTED_IN {characters, primaryProfession}]->(:Movie)
```

## Run

```powershell
uvicorn app:app --app-dir src --reload --port 8000
```

## Test API

Health:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Main endpoint:

```powershell
Invoke-RestMethod http://localhost:8000/movie/query `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"Christopher Nolan 导演的评分高于 8.5 的动作片","top_k":5,"stream":false,"chat_id":"demo-chat","user_id":"demo-user"}'
```

LangGraph assistant endpoint:

```powershell
Invoke-RestMethod http://localhost:8000/assistants/movie_agent_query/runs/invoke `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"input":{"query":"推荐关于时间和亲情的科幻电影","top_k":5}}'
```

## Expected Routing Examples

- SQL: `Christopher Nolan 导演的评分高于 8.5 的动作片`
- Graph: `Leonardo DiCaprio 出演过哪些电影，他和哪些演员通过电影有关联`
- Vector: `推荐剧情像星际穿越、关于时间和亲情的电影`

## Notes

- `/movie/query` treats `generate_answer`, `refine_sql_result`, and `refine_graph_result` as final-answer nodes.
- SQL and Graph self-correction retry at most 3 times before falling back to Vector RAG.
- Memory and checkpointing use the SQL Supabase Postgres connection.
