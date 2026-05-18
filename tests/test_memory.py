from rag_agent.context import memory


class _FakeConnection:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), params or {}))


class _FakeEngine:
    def __init__(self) -> None:
        self.connection = _FakeConnection()

    def begin(self):
        return self

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_prune_memories_deletes_rows_over_limit(monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setattr(memory, "_memory_engine", lambda: engine)

    memory._prune_memories("session_memories", "chat_id", "chat-1", 3)

    assert engine.connection.calls
    sql, params = engine.connection.calls[0]
    assert "DELETE FROM session_memories" in sql
    assert params == {"value": "chat-1", "max_items": 3}


def test_prune_memories_skips_non_positive_limit(monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setattr(memory, "_memory_engine", lambda: engine)

    memory._prune_memories("session_memories", "chat_id", "chat-1", 0)

    assert engine.connection.calls == []
