from retrieval.modality_router import route_query_heuristic


def test_route_heuristic_table():
    assert route_query_heuristic("What is in the revenue table?") == "table"


def test_route_heuristic_image():
    assert route_query_heuristic("Describe the chart in figure 3") == "image"


def test_route_heuristic_text():
    assert route_query_heuristic("What is the company mission?") == "text"


def test_route_heuristic_mixed():
    assert route_query_heuristic("Compare the chart and the revenue table") == "mixed"
