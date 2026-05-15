from rag_agent.utils import detect_answer_language, language_instruction, sanitize_answer_for_query


def test_detect_english_query_language():
    assert detect_answer_language("What are the top 10 highest rated movies after 2010?") == "English"


def test_english_language_instruction_is_strict():
    instruction = language_instruction("What are the top 10 highest rated movies after 2010?")

    assert "English only" in instruction
    assert "Do not answer in German" in instruction


def test_sanitize_english_answer_to_ascii():
    answer = "Only one movie matches \u201cAmerica\u201d: \u00bf... Y el pr\u00f3jimo?"

    sanitized = sanitize_answer_for_query(answer, "Find movies with plot for america drama")

    assert sanitized == 'Only one movie matches "America": ... Y el projimo?'
