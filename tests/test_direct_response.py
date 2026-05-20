from dataclasses import dataclass

from rag_agent.constants import FINAL_ANSWER_NODES
from rag_agent.context.base import Context
from rag_agent.movie_agent.nodes import generate_direct_response_node


@dataclass
class _FakeResponse:
    content: str


class _FakeLLM:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return _FakeResponse(content="I am a movie assistant.")


@dataclass
class _FakeRuntime:
    context: Context


def test_generate_direct_response_ignores_movie_context(monkeypatch):
    fake_llm = _FakeLLM()
    monkeypatch.setattr("rag_agent.movie_agent.nodes.get_chat_model", lambda _model: fake_llm)

    state = {
        "query": "what are you",
        "context": "Movie context:\nmovie[1].movieId: tt0033943",
        "similar_movies": [{"movie_id": "tt0033943", "content": "some chunk"}],
        "chat_history_messages": ["hello there"],
        "user_memories": [],
        "session_memories": [],
    }
    runtime = _FakeRuntime(context=Context())

    result = generate_direct_response_node(state, runtime)

    assert result["success"] is True
    assert result["answer"] == "I am a movie assistant."
    assert result["reasoning"] == "Generated direct response without movie retrieval."

    prompt = str(fake_llm.messages[0].content)
    assert "Chat history messages:" in prompt
    assert "hello there" in prompt
    assert "movie[1].movieId: tt0033943" not in prompt
    assert "Detailed chunks" not in prompt
    assert "Movie context:" not in prompt


def test_generate_direct_response_is_final_answer_node():
    assert "generate_direct_response" in FINAL_ANSWER_NODES
