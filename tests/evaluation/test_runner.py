from obrag.config import Settings
from obrag.evaluation import runner
from obrag.evaluation.storage import GoldenQuestion
from obrag.models import Answer, Chunk, RetrievedChunk


def make_settings(tmp_path):
    return Settings(
        gemini_api_key="ai-x", voyage_api_key="pa-x", eval_db_path=tmp_path / "eval.db"
    )


HIT = RetrievedChunk(
    Chunk(id="c1", text="t", collection="regulation", citation="PSR 2017, regulation 100",
          source_url="https://example.invalid"),
    0.9,
)

GOLDEN = [
    GoldenQuestion(id="reg-01", band="regulation_only", question="when is SCA required?",
                   expected_collections=["regulation"], expected_points=["p"], answerable=True),
    GoldenQuestion(id="unans-01", band="unanswerable", question="barclays rate limit?",
                   expected_collections=[], expected_points=[], answerable=False),
]


def patch_everything(monkeypatch, answer_for, routed=("regulation",)):
    monkeypatch.setattr(runner, "load_golden", lambda path=None: GOLDEN)
    monkeypatch.setattr(
        runner, "ask_with_route", lambda q, settings=None: (list(routed), answer_for(q))
    )
    monkeypatch.setattr(runner, "score_groundedness", lambda a, s, client=None: 1.0)
    monkeypatch.setattr(runner, "score_correctness", lambda a, p, s, client=None: 0.7)
    monkeypatch.setattr(runner, "score_retrieval_relevance", lambda q, a, s, client=None: 0.8)


def rows_for(tmp_path):
    settings = make_settings(tmp_path)
    run_id = runner.run_eval("baseline", settings)
    return {r["question_id"]: r for r in runner.EvalStore(settings).results(run_id)}


def test_run_eval_records_one_row_per_golden_question(tmp_path, monkeypatch):
    patch_everything(
        monkeypatch,
        lambda q: Answer(text="a [1]", citations=["PSR 2017, regulation 100"],
                         refused="barclays" in q, retrieved=[HIT]),
    )
    assert len(rows_for(tmp_path)) == 2


def test_routing_is_scored_from_the_route_the_answer_used(tmp_path, monkeypatch):
    patch_everything(monkeypatch, lambda q: Answer(text="a", citations=[], refused=False, retrieved=[HIT]))
    assert rows_for(tmp_path)["reg-01"]["routing_correct"] == 1

    patch_everything(
        monkeypatch, lambda q: Answer(text="a", citations=[], refused=False, retrieved=[HIT]),
        routed=("regulation", "spec"),
    )
    assert rows_for(tmp_path)["reg-01"]["routing_correct"] == 0


def test_refusal_correctness_is_recorded_for_both_bands(tmp_path, monkeypatch):
    patch_everything(
        monkeypatch,
        lambda q: Answer(text="a", citations=[], refused="barclays" in q, retrieved=[HIT]),
    )
    rows = rows_for(tmp_path)
    assert rows["reg-01"]["refusal_correct"] == 1
    assert rows["unans-01"]["refusal_correct"] == 1


def test_a_failing_judge_records_no_score_instead_of_aborting_the_run(tmp_path, monkeypatch):
    patch_everything(monkeypatch, lambda q: Answer(text="a", citations=[], refused=False, retrieved=[HIT]))

    def exploding(a, s, client=None):
        raise RuntimeError("429")

    monkeypatch.setattr(runner, "score_groundedness", exploding)
    rows = rows_for(tmp_path)
    assert rows["reg-01"]["groundedness"] is None
    assert rows["reg-01"]["correctness"] == 0.7
    assert rows["reg-01"]["refusal_correct"] == 1


def test_summarise_produces_per_band_and_overall_means(tmp_path, monkeypatch):
    patch_everything(monkeypatch, lambda q: Answer(text="a", citations=[], refused=False, retrieved=[HIT]))
    settings = make_settings(tmp_path)
    run_id = runner.run_eval("baseline", settings)
    summary = runner.summarise(runner.EvalStore(settings).results(run_id))
    assert "overall" in summary
    assert "regulation_only" in summary
    assert 0.0 <= summary["overall"]["groundedness"] <= 1.0


def test_summarise_reports_how_many_scores_are_missing():
    rows = [
        {"band": "spec_only", "groundedness": None, "correctness": 1.0,
         "retrieval_relevance": 1.0, "refusal_correct": 1, "routing_correct": 1},
        {"band": "spec_only", "groundedness": 0.5, "correctness": 1.0,
         "retrieval_relevance": 1.0, "refusal_correct": 1, "routing_correct": 1},
    ]
    summary = runner.summarise(rows)
    assert summary["overall"]["groundedness"] == 0.5
    assert summary["overall"]["missing_scores"] == 1
