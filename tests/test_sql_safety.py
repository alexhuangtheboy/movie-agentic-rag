from rag_agent.movie_agent.tools import _is_read_only_query, normalize_movie_sql_identifiers


def test_read_only_select_allowed():
    assert _is_read_only_query("SELECT * FROM title_content LIMIT 1")


def test_read_only_with_allowed():
    assert _is_read_only_query("WITH x AS (SELECT 1) SELECT * FROM x")


def test_delete_rejected():
    assert not _is_read_only_query("DELETE FROM title_content")


def test_normalize_quotes_camel_case_movie_columns():
    sql = "SELECT tc.primaryTitle, tc.startYear, mr.averageRating FROM title_content tc JOIN movie_ratings mr ON tc.movie_id = mr.movie_id"

    normalized = normalize_movie_sql_identifiers(sql)

    assert 'tc."primaryTitle"' in normalized
    assert 'tc."startYear"' in normalized
    assert 'mr."averageRating"' in normalized
