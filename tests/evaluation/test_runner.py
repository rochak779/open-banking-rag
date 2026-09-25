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


def test_a_transient_pipeline_failure_is_retried(tmp_path, monkeypatch):
    patch_everything(monkeypatch, lambda q: Answer(text="a", citations=[], refused=False, retrieved=[HIT]))
    monkeypatch.setattr(runner, "RETRY_WAIT_SECONDS", 0)
    attempts = []

    def flaky(q, settings=None):
        attempts.append(q)
        if len(attempts) == 1:
            raise ConnectionError("Remote end closed connection without response")
        return ["regulation"], Answer(text="a", citations=[], refused=False, retrieved=[HIT])

    monkeypatch.setattr(runner, "ask_with_route", flaky)
    assert len(rows_for(tmp_path)) == 2
    assert len(attempts) == 3  # one retry for the first question, then one call for the second


def test_a_question_that_keeps_failing_is_skipped_not_recorded(tmp_path, monkeypatch):
    patch_everything(monkeypatch, lambda q: Answer(text="a", citations=[], refused=False, retrieved=[HIT]))
    monkeypatch.setattr(runner, "RETRY_WAIT_SECONDS", 0)

    def broken_for_reg(q, settings=None):
        if "SCA" in q:
            raise ConnectionError("down")
        return [], Answer(text="a", citations=[], refused=True, retrieved=[])

    monkeypatch.setattr(runner, "ask_with_route", broken_for_reg)
    # A made-up row (e.g. a fake refusal) would distort refusal accuracy; a gap is honest.
    assert sorted(rows_for(tmp_path)) == ["unans-01"]


def test_resume_skips_questions_already_recorded_in_that_run(tmp_path, monkeypatch):
    patch_everything(monkeypatch, lambda q: Answer(text="a", citations=[], refused=False, retrieved=[HIT]))
    settings = make_settings(tmp_path)
    store = runner.EvalStore(settings)
    run_id = store.create_run("baseline", {})
    store.record(run_id, {"question_id": "reg-01", "band": "regulation_only"})

    asked = []
    monkeypatch.setattr(
        runner, "ask_with_route",
        lambda q, settings=None: asked.append(q) or ([], Answer(text="a", citations=[], refused=True, retrieved=[])),
    )
    assert runner.run_eval("baseline", settings, resume_run_id=run_id) == run_id
    assert asked == ["barclays rate limit?"]
    assert len(store.results(run_id)) == 2


def test_rescore_rejudges_groundedness_against_the_cited_sources(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    store = runner.EvalStore(settings)
    run_id = store.create_run("baseline", {})
    store.record(run_id, {"question_id": "reg-01", "band": "regulation_only", "answer": "SCA [1].",
                          "refused": False, "citations": ["PSR 2017, regulation 100"],
                          "groundedness": 0.0, "correctness": None})
    store.record(run_id, {"question_id": "unans-01", "band": "unanswerable", "answer": "No.",
                          "refused": True, "citations": [], "groundedness": 1.0, "correctness": 1.0})
    monkeypatch.setattr(runner, "load_golden", lambda path=None: GOLDEN)
    monkeypatch.setattr(runner, "_chunks_by_citation", lambda settings: {"PSR 2017, regulation 100": HIT.chunk})
    seen = []
    monkeypatch.setattr(runner, "score_groundedness",
                        lambda a, s, client=None: seen.append(a) or 0.9)
    monkeypatch.setattr(runner, "score_correctness", lambda a, p, s, client=None: 0.6)

    runner.rescore(run_id, settings)

    rows = {r["question_id"]: r for r in store.results(run_id)}
    assert rows["reg-01"]["groundedness"] == 0.9
    assert rows["reg-01"]["correctness"] == 0.6          # was missing, now filled
    assert rows["unans-01"]["groundedness"] == 1.0       # refusals are not re-judged
    assert [c.chunk.citation for c in seen[0].retrieved] == ["PSR 2017, regulation 100"]


def test_summarise_reports_answerable_questions_separately():
    # Refusals score groundedness and correctness 1.0 by definition, so an
    # "overall" mean that includes the unanswerable band flatters the system.
    rows = [
        {"band": "spec_only", "groundedness": 0.5, "correctness": 0.4,
         "retrieval_relevance": 0.5, "refusal_correct": 1, "routing_correct": 1},
        {"band": "unanswerable", "groundedness": 1.0, "correctness": 1.0,
         "retrieval_relevance": 0.0, "refusal_correct": 1, "routing_correct": None},
    ]
    summary = runner.summarise(rows)
    assert summary["answerable"]["correctness"] == 0.4
    assert summary["answerable"]["n"] == 1
    assert summary["overall"]["correctness"] == 0.7


def _one_answered_row(tmp_path, monkeypatch, groundedness):
    settings = make_settings(tmp_path)
    store = runner.EvalStore(settings)
    run_id = store.create_run("baseline", {})
    store.record(run_id, {"question_id": "reg-01", "band": "regulation_only", "answer": "SCA [1].",
                          "refused": False, "citations": ["PSR 2017, regulation 100"],
                          "groundedness": groundedness, "correctness": 0.5})
    monkeypatch.setattr(runner, "load_golden", lambda path=None: GOLDEN)
    monkeypatch.setattr(runner, "_chunks_by_citation", lambda settings: {"PSR 2017, regulation 100": HIT.chunk})
    return settings, store, run_id


def test_a_failed_rejudge_clears_the_stale_score(tmp_path, monkeypatch):
    settings, store, run_id = _one_answered_row(tmp_path, monkeypatch, groundedness=0.0)

    def exploding(a, s, client=None):
        raise RuntimeError("500")

    monkeypatch.setattr(runner, "score_groundedness", exploding)
    runner.rescore(run_id, settings)
    assert store.results(run_id)[0]["groundedness"] is None


def test_fill_missing_only_judges_gaps(tmp_path, monkeypatch):
    settings, store, run_id = _one_answered_row(tmp_path, monkeypatch, groundedness=0.8)
    calls = []
    monkeypatch.setattr(runner, "score_groundedness", lambda a, s, client=None: calls.append(a) or 0.1)
    runner.rescore(run_id, settings, only_missing=True)
    assert calls == []
    assert store.results(run_id)[0]["groundedness"] == 0.8
