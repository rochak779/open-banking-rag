from obrag.config import Settings
from obrag.evaluation.storage import EvalStore


def make_settings(tmp_path):
    return Settings(
        gemini_api_key="ai-x", voyage_api_key="pa-x", eval_db_path=tmp_path / "eval.db"
    )


def test_create_run_returns_an_increasing_id(tmp_path):
    store = EvalStore(make_settings(tmp_path))
    first = store.create_run("baseline", {"embedding_model": "voyage-4-lite"})
    second = store.create_run("law-embeddings", {"embedding_model": "voyage-law-2"})
    assert second > first


def test_results_round_trip(tmp_path):
    store = EvalStore(make_settings(tmp_path))
    run_id = store.create_run("baseline", {})
    store.record(run_id, {
        "question_id": "reg-01", "band": "regulation_only", "answer": "Yes [1].",
        "refused": False, "citations": ["PSR 2017, regulation 100"],
        "routing_correct": True, "groundedness": 1.0, "correctness": 0.8,
        "retrieval_relevance": 1.0, "refusal_correct": True,
    })
    results = store.results(run_id)
    assert len(results) == 1
    assert results[0]["question_id"] == "reg-01"
    assert results[0]["groundedness"] == 1.0
    assert results[0]["citations"] == ["PSR 2017, regulation 100"]


def test_runs_lists_runs_newest_first_with_their_config(tmp_path):
    store = EvalStore(make_settings(tmp_path))
    store.create_run("baseline", {"top_k": 6})
    store.create_run("wider", {"top_k": 12})
    runs = store.runs()
    assert [r["label"] for r in runs] == ["wider", "baseline"]
    assert runs[0]["config"]["top_k"] == 12


def test_results_are_scoped_to_their_run(tmp_path):
    store = EvalStore(make_settings(tmp_path))
    a = store.create_run("a", {})
    b = store.create_run("b", {})
    store.record(a, {"question_id": "q1", "band": "regulation_only"})
    assert store.results(b) == []


def test_update_changes_named_fields_of_one_result(tmp_path):
    store = EvalStore(make_settings(tmp_path))
    run_id = store.create_run("baseline", {})
    store.record(run_id, {"question_id": "q1", "band": "spec_only", "groundedness": 0.0, "correctness": None})
    store.update(run_id, "q1", {"groundedness": 1.0, "correctness": 0.5})
    row = store.results(run_id)[0]
    assert (row["groundedness"], row["correctness"]) == (1.0, 0.5)
