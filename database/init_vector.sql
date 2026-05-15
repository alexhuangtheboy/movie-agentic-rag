CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS movie_plot_embeddings (
    id bigserial PRIMARY KEY,
    movie_id text NOT NULL,
    content text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    embedding vector(1536),
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_movie_plot_embeddings_movie_id ON movie_plot_embeddings(movie_id);
CREATE INDEX IF NOT EXISTS idx_movie_plot_embeddings_metadata ON movie_plot_embeddings USING gin(metadata);

-- Optional ANN index. Create after the table has data for better build behavior.
-- CREATE INDEX IF NOT EXISTS idx_movie_plot_embeddings_embedding
-- ON movie_plot_embeddings USING hnsw (embedding vector_cosine_ops);
