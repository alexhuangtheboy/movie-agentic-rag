from rag_agent.graph import route_after_graph, route_after_sql, route_decision


def test_route_decision_sql():
    assert route_decision({"suggested_tools": "sql query"}) == "execute_sql_query"


def test_route_decision_graph():
    assert route_decision({"suggested_tools": "graph query"}) == "execute_graph_query"


def test_route_decision_rag_default():
    assert route_decision({"suggested_tools": "rag"}) == "rag_fallback"


def test_sql_refines_empty_result_under_limit():
    assert route_after_sql(
        {
            "success": False,
            "answer": "Query executed successfully. No rows returned.",
            "sql_refinement_attempts": 2,
        }
    ) == "refine_sql_query"


def test_sql_fallback_after_three_refinements():
    assert route_after_sql(
        {
            "success": False,
            "answer": "Query executed successfully. No rows returned.",
            "sql_refinement_attempts": 3,
        }
    ) == "rag_fallback"


def test_graph_refines_empty_result_under_limit():
    assert route_after_graph(
        {
            "success": False,
            "answer": "Query executed successfully. No results returned.",
            "graph_refinement_attempts": 0,
        }
    ) == "refine_graph_query"
