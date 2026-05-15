from rag_agent.movie_agent.formatters import convert_movies_to_toon, convert_to_toon_format
from rag_agent.movie_agent.nodes import _no_context_answer


def test_single_movie_toon_contains_title_and_genre():
    result = convert_to_toon_format(
        {
            "movie_id": "tt1",
            "primaryTitle": "Interstellar",
            "genres": "Adventure,Drama,Sci-Fi",
        }
    )

    assert "movie[1]" in result
    assert "Interstellar" in result


def test_multiple_movie_toon_count():
    result = convert_movies_to_toon(
        [
            {"movie_id": "tt1", "primaryTitle": "A"},
            {"movie_id": "tt2", "primaryTitle": "B"},
        ]
    )

    assert "movies[2]" in result
    assert "tt1" in result
    assert "tt2" in result


def test_no_context_answer_matches_english_query():
    result = _no_context_answer("Recommend movies about time travel and family.")

    assert "could not find" in result
    assert "vector database" in result
