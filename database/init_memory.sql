CREATE TABLE IF NOT EXISTS user_memories (
    memory_id text PRIMARY KEY,
    user_id text NOT NULL,
    content text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_memories_user_id ON user_memories(user_id);

CREATE TABLE IF NOT EXISTS session_memories (
    memory_id text PRIMARY KEY,
    chat_id text NOT NULL,
    content text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_session_memories_chat_id ON session_memories(chat_id);
