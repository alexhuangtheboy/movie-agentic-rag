"""Prompt templates for movie agent answer generation."""

from __future__ import annotations


def build_vector_answer_prompt(
    system_prompt: str,
    user_memories_text: str,
    session_memories_text: str,
    movie_context: str,
    question: str,
    language_instruction_text: str,
) -> str:
    """Build the final answer prompt for vector RAG results."""
    return f"""{system_prompt}

User memories:
{user_memories_text}

Session memories:
{session_memories_text}

Movie context:
{movie_context}

Question: {question}

{language_instruction_text}
Include movie titles and relevant evidence from context.
"""


def build_direct_response_prompt(
    direct_system_prompt: str,
    user_memories_text: str,
    session_memories_text: str,
    chat_history_text: str,
    question: str,
    language_instruction_text: str,
) -> str:
    """Build a direct-response prompt without movie database context."""
    return f"""{direct_system_prompt}

User memories:
{user_memories_text}

Session memories:
{session_memories_text}

Chat history messages:
{chat_history_text}

Question: {question}

{language_instruction_text}
Answer naturally and briefly.
"""
