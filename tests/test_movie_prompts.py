from rag_agent.movie_agent.prompts import build_direct_response_prompt, build_vector_answer_prompt


def test_vector_answer_prompt_contains_required_sections():
    prompt = build_vector_answer_prompt(
        system_prompt="system",
        user_memories_text="- user memory",
        session_memories_text="- session memory",
        movie_context="movie context",
        question="which movies?",
        language_instruction_text="English only",
    )

    assert "system" in prompt
    assert "User memories:" in prompt
    assert "- user memory" in prompt
    assert "Movie context:" in prompt
    assert "movie context" in prompt
    assert "Question: which movies?" in prompt


def test_direct_response_prompt_excludes_movie_context_section():
    prompt = build_direct_response_prompt(
        direct_system_prompt="direct system",
        user_memories_text="",
        session_memories_text="",
        chat_history_text="- hello",
        question="what are you?",
        language_instruction_text="English only",
    )

    assert "direct system" in prompt
    assert "Chat history messages:" in prompt
    assert "- hello" in prompt
    assert "Movie context:" not in prompt
