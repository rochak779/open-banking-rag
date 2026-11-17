from obrag.app.streamlit_app import drop_routing, summary_rows, used_router


def test_summary_rows_flatten_bands_for_a_table():
    summary = {
        "overall": {"groundedness": 0.9, "correctness": 0.7, "retrieval_relevance": 0.8,
                    "refusal_accuracy": 0.95, "routing_accuracy": 0.9, "n": 40},
        "unanswerable": {"groundedness": 1.0, "correctness": None, "retrieval_relevance": 0.1,
                         "refusal_accuracy": 1.0, "routing_accuracy": None, "n": 10},
    }
    rows = summary_rows(summary)
    assert {r["band"] for r in rows} == {"overall", "unanswerable"}
    assert rows[0]["band"] == "overall"


def test_runs_saved_before_the_router_flag_count_as_routed():
    # Runs 1-4 predate use_router and all called the router.
    assert used_router({"config": {"top_k": 6}}) is True
    assert used_router({"config": {"use_router": False}}) is False


def test_routing_columns_are_dropped_for_a_run_without_the_router():
    # With every question sent to both collections, routing accuracy is just the
    # share of questions that expect both: 33% on the golden set, which reads as a fault.
    rows = [{"band": "overall", "routing_accuracy": 0.333, "correctness": 0.8},
            {"question_id": "reg-01", "routing_correct": 0, "refused": False}]
    assert drop_routing(rows) == [{"band": "overall", "correctness": 0.8},
                                  {"question_id": "reg-01", "refused": False}]
