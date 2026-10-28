from obrag.app.streamlit_app import drop_routing, format_answer, summary_rows, used_router
from obrag.models import Answer, Chunk, RetrievedChunk

HIT = RetrievedChunk(
    Chunk(id="c1", text="body", collection="regulation", citation="PSR 2017, regulation 100",
          source_url="https://www.legislation.gov.uk/uksi/2017/752/regulation/100"),
    0.91,
)


def test_answer_renders_text_and_linked_citations():
    answer = Answer(text="SCA applies [1].", citations=["PSR 2017, regulation 100"],
                    refused=False, retrieved=[HIT])
    rendered = format_answer(answer)
    assert "SCA applies [1]." in rendered
    assert "PSR 2017, regulation 100" in rendered
    assert "https://www.legislation.gov.uk/uksi/2017/752/regulation/100" in rendered


def test_refusal_is_labelled_and_shows_no_citations():
    answer = Answer(text="INSUFFICIENT_CONTEXT: not covered.", citations=[], refused=True, retrieved=[HIT])
    rendered = format_answer(answer)
    assert "No grounded answer" in rendered
    assert "PSR 2017" not in rendered


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


def test_citation_numbers_in_the_text_match_the_source_list():
    spec = RetrievedChunk(
        Chunk(id="c2", text="body", collection="spec", citation="Payment Initiation API v4.0.1 — POST /domestic-payments",
              source_url="https://example.invalid/spec"),
        0.8,
    )
    answer = Answer(text="Rule [1], endpoint [2].",
                    citations=["PSR 2017, regulation 100", "Payment Initiation API v4.0.1 — POST /domestic-payments"],
                    refused=False, retrieved=[spec, HIT])
    rendered = format_answer(answer)
    assert "1. [PSR 2017, regulation 100]" in rendered
    assert "2. [Payment Initiation API v4.0.1 — POST /domestic-payments]" in rendered


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
