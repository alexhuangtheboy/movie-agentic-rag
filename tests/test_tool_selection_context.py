import json

from rag_agent.graph import call_tool_selection_node
from rag_agent.tool_selection_agent.nodes import select_tool_node
from rag_agent.tool_selection_agent.prompts import get_tool_selection_user_prompt


def test_user_prompt_includes_conversation_context():
    prompt = get_tool_selection_user_prompt(
        "sql schema",
        "graph schema",
        "give me the specific movie names",
        chat_history_messages=["tell me movies after 2010 with rating over 9.5"],
        user_memories_text="- likes sci-fi",
        session_memories_text="- asked for post-2010 high rating movies",
        previous_tool="sql query",
        previous_answer="Found 3 matching movies.",
    )

    assert "Conversation context:" in prompt
    assert "Previous selected tool: sql query" in prompt
    assert "tell me movies after 2010 with rating over 9.5" in prompt
    assert "asked for post-2010 high rating movies" in prompt


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeRouterLLM:
    def bind(self, response_format):
        return self

    def invoke(self, messages):
        prompt = str(messages[-1].content)
        has_followup_context = "Previous selected tool: sql query" in prompt and "Chat history:\n- " in prompt
        if has_followup_context:
            payload = {
                "tool": "sql query",
                "query": 'SELECT "primaryTitle" FROM title_content LIMIT 10',
                "reasoning": "Follow-up refers to prior filtered movie list.",
                "confidence": 0.8,
            }
        else:
            payload = {
                "tool": "direct response",
                "query": "",
                "reasoning": "No context for ambiguous request.",
                "confidence": 0.1,
            }
        return _FakeResponse(json.dumps(payload))


def test_select_tool_prefers_sql_for_contextual_followup(monkeypatch):
    monkeypatch.setattr("rag_agent.tool_selection_agent.nodes.get_chat_model", lambda: _FakeRouterLLM())
    state = {
        "question": "give me the specific movie names",
        "sql_table_names": [],
        "graph_table_names": [],
        "chat_history_messages": ["tell me movies after 2010 with rating over 9.5"],
        "user_memories_text": "",
        "session_memories_text": "- asked for movies with rating over 9.5",
        "previous_tool": "sql query",
        "previous_answer": "Found 2 movies.",
    }

    result = select_tool_node(state)
    assert result["tool"] == "sql query"
    assert result["query"]


def test_select_tool_can_direct_response_without_context(monkeypatch):
    monkeypatch.setattr("rag_agent.tool_selection_agent.nodes.get_chat_model", lambda: _FakeRouterLLM())
    state = {
        "question": "give me the specific movie names",
        "sql_table_names": [],
        "graph_table_names": [],
        "chat_history_messages": [],
        "user_memories_text": "",
        "session_memories_text": "",
        "previous_tool": "",
        "previous_answer": "",
    }

    result = select_tool_node(state)
    assert result["tool"] == "direct response"
    assert result["query"] == ""


class _CaptureInvoke:
    def __init__(self) -> None:
        self.payload = None

    def invoke(self, payload):
        self.payload = payload
        return {"tool": "rag", "query": "", "reasoning": "test", "confidence": 0.6}


def test_call_tool_selection_node_forwards_context_fields(monkeypatch):
    capture = _CaptureInvoke()
    monkeypatch.setattr("rag_agent.graph.tool_selection_graph", capture)
    state = {
        "query": "give me the specific movie names",
        "sql_table_names": [],
        "graph_table_names": [],
        "chat_history_messages": ["tell me movies after 2010 with rating over 9.5"],
        "user_memories": [{"content": "prefers highly rated titles"}],
        "session_memories": [{"content": "asked for post-2010 movies"}],
        "suggested_tools": "sql query",
        "answer": "Found 3 movies.",
    }

    call_tool_selection_node(state)

    assert capture.payload is not None
    assert capture.payload["chat_history_messages"] == ["tell me movies after 2010 with rating over 9.5"]
    assert "prefers highly rated titles" in capture.payload["user_memories_text"]
    assert "asked for post-2010 movies" in capture.payload["session_memories_text"]
    assert capture.payload["previous_tool"] == "sql query"
