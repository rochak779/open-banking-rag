from obrag.app import tiles
from obrag.models import Answer, Chunk, RetrievedChunk

HIT = RetrievedChunk(
    Chunk(id="c1", text="body", collection="regulation", citation="PSR 2017, regulation 100",
          source_url="https://www.legislation.gov.uk/uksi/2017/752/regulation/100"),
    0.91,
)
SPEC = RetrievedChunk(
    Chunk(id="c2", text="body", collection="spec",
          citation="Payment Initiation API v4.0.1 — POST /domestic-payments",
          source_url="https://example.invalid/spec"),
    0.8,
)


def test_sources_are_linked_and_labelled_by_where_they_come_from():
    answer = Answer(text="SCA applies [1].", citations=["PSR 2017, regulation 100"],
                    refused=False, retrieved=[HIT])
    rendered = tiles.sources_html(tiles.source_rows(answer))
    assert "PSR 2017, regulation 100" in rendered
    assert 'href="https://www.legislation.gov.uk/uksi/2017/752/regulation/100"' in rendered
    assert "Law" in rendered


def test_refusal_shows_no_citations():
    answer = Answer(text="INSUFFICIENT_CONTEXT: not covered.", citations=[], refused=True, retrieved=[HIT])
    rendered = tiles.sources_html(tiles.source_rows(answer))
    assert "Nothing was cited" in rendered
    assert "PSR 2017" not in rendered


def test_source_numbers_match_the_citation_markers_in_the_text():
    # Retrieval order differs from citation order; "[1]" must stay the first cited source.
    answer = Answer(text="Rule [1], endpoint [2].",
                    citations=["PSR 2017, regulation 100", "Payment Initiation API v4.0.1 — POST /domestic-payments"],
                    refused=False, retrieved=[SPEC, HIT])
    rows = tiles.source_rows(answer)
    assert [(r["index"], r["citation"], r["collection"]) for r in rows] == [
        (1, "PSR 2017, regulation 100", "regulation"),
        (2, "Payment Initiation API v4.0.1 — POST /domestic-payments", "spec"),
    ]


def test_a_citation_with_no_retrieved_source_is_shown_without_a_link():
    answer = Answer(text="Rule [1].", citations=["PSR 2017, regulation 1"], refused=False, retrieved=[HIT])
    rendered = tiles.sources_html(tiles.source_rows(answer))
    assert "PSR 2017, regulation 1" in rendered
    assert "href" not in rendered


def test_text_from_the_index_cannot_inject_markup():
    rendered = tiles.passage_html("<b>cite</b>", "spec", 0.5, "<script>alert(1)</script>")
    assert "<script>" not in rendered and "<b>cite</b>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_decision_counts_turn_accuracy_back_into_whole_questions():
    summary = {
        "overall": {"refusal_accuracy": 0.875, "n": 40},
        "regulation_only": {"refusal_accuracy": 1, "n": 10},
        "spec_only": {"refusal_accuracy": 0.7, "n": 10},
        "cross_cutting": {"refusal_accuracy": 0.8, "n": 10},
        "unanswerable": {"refusal_accuracy": 1, "n": 10},
    }
    counts = tiles.decision_counts(summary)
    assert [(c["band"], c["correct"], c["n"]) for c in counts] == [
        ("regulation_only", 10, 10), ("spec_only", 7, 10), ("cross_cutting", 8, 10), ("unanswerable", 10, 10),
    ]
    chart = tiles.decision_chart_html(counts)
    assert "35<small>of 40</small>" in chart


def test_score_grid_leaves_routing_out_when_the_run_had_no_router():
    rows = [{"band": "overall", "refusal_accuracy": 0.875, "correctness": None, "n": 40}]
    assert "Routing" not in tiles.score_grid_html(rows, with_routing=False)
    assert "Routing" in tiles.score_grid_html(rows, with_routing=True)
    assert "not scored" in tiles.score_grid_html(rows, with_routing=False)


def test_the_disclaimer_keeps_its_wording():
    rendered = tiles.disclaimer_html("Snapshot taken 2026-09-18 & not advice.")
    assert "Snapshot taken" in rendered and "&amp; not advice." in rendered
    assert '<span class="ob-nowrap">2026-09-18</span>' in rendered
