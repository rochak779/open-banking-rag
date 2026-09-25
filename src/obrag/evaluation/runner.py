"""Run the golden set end to end and persist every score."""

import argparse
import time
from statistics import mean

from obrag.config import Settings, load_settings
from obrag.evaluation.metrics import (
    score_correctness,
    score_groundedness,
    score_refusal,
    score_retrieval_relevance,
)
from obrag.evaluation.storage import EvalStore, load_golden
from obrag.ingest.legislation import parse_all_legislation
from obrag.ingest.obl_spec import parse_spec_dir
from obrag.models import Answer, Chunk, RetrievedChunk
from obrag.query.pipeline import ask_with_route

SCORE_FIELDS = ("retrieval_relevance", "groundedness", "correctness")

ASK_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 30


def _ask_with_retries(question_id: str, question: str, settings: Settings):
    """ask_with_route, retried on failure; None if every attempt fails.

    A network drop to Voyage aborted a 2-hour run on question 6. A question
    that still fails is skipped rather than recorded: a stand-in row (say, a
    refusal) would distort refusal accuracy, while a gap shows up in n.
    """
    for attempt in range(1, ASK_ATTEMPTS + 1):
        try:
            return ask_with_route(question, settings=settings)
        except Exception as error:  # noqa: BLE001 - network, quota or server errors alike
            print(f"    {question_id}: ask failed, attempt {attempt}/{ASK_ATTEMPTS} "
                  f"({type(error).__name__}: {error})")
            if attempt < ASK_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
    print(f"    {question_id}: SKIPPED after {ASK_ATTEMPTS} attempts")
    return None


def _safely(label: str, question_id: str, score, *args):
    """Run one judge; on failure record no score rather than abort the whole run.

    A run takes tens of minutes against free-tier rate limits. Losing it to one
    judge error is worse than a gap, and summarise() reports every gap.
    """
    try:
        return score(*args)
    except Exception as error:  # noqa: BLE001 - any judge failure is a missing score
        print(f"    {question_id}: {label} judge failed ({type(error).__name__}: {error})")
        return None


def run_eval(label: str, settings: Settings, golden_path=None, resume_run_id: int | None = None) -> int:
    """Score every golden question; with resume_run_id, finish an interrupted run."""
    store = EvalStore(settings)
    if resume_run_id is None:
        run_id = store.create_run(
            label,
            {
                "embedding_model": settings.embedding_model,
                "router_model": settings.router_model,
                "generation_model": settings.generation_model,
                "judge_model": settings.judge_model,
                "top_k": settings.top_k,
                "min_score": settings.min_score,
            },
        )
        done: set[str] = set()
    else:
        run_id = resume_run_id
        done = {row["question_id"] for row in store.results(run_id)}
        print(f"resuming run {run_id}: {len(done)} questions already scored")

    for question in load_golden(golden_path):
        if question.id in done:
            continue
        asked = _ask_with_retries(question.id, question.question, settings)
        if asked is None:
            continue
        routed, answer = asked
        routing_correct = (
            set(routed) == set(question.expected_collections)
            if question.expected_collections
            else None
        )
        store.record(run_id, {
            "question_id": question.id,
            "band": question.band,
            "answer": answer.text,
            "refused": answer.refused,
            "citations": answer.citations,
            "routing_correct": routing_correct,
            "retrieval_relevance": _safely(
                "relevance", question.id, score_retrieval_relevance, question.question, answer, settings
            ),
            "groundedness": _safely("groundedness", question.id, score_groundedness, answer, settings),
            "correctness": _safely(
                "correctness", question.id, score_correctness, answer, question.expected_points, settings
            ),
            "refusal_correct": score_refusal(answer, question.answerable),
        })
        print(f"  {question.id}: routed={routed} refused={answer.refused} citations={len(answer.citations)}")

    return run_id


def _chunks_by_citation(settings: Settings) -> dict[str, Chunk]:
    """Every indexed chunk, keyed by citation, parsed from the pinned raw sources."""
    spec_dir = settings.raw_dir / f"obl-specs-{settings.obl_spec_tag}" / "dist" / "openapi"
    chunks = parse_all_legislation(settings.raw_dir) + parse_spec_dir(spec_dir)
    return {chunk.citation: chunk for chunk in chunks}


def rescore(run_id: int, settings: Settings, golden_path=None) -> None:
    """Re-judge groundedness from stored answers and fill in missing correctness.

    Groundedness is judged against the sources the answer cites, rebuilt from
    the raw snapshot, because a run stores citations but not the full retrieved
    set. Refusals keep their automatic 1.0. Retrieval relevance needs the full
    retrieved set, so a missing relevance score stays missing.
    """
    store = EvalStore(settings)
    points = {q.id: q.expected_points for q in load_golden(golden_path)}
    by_citation = _chunks_by_citation(settings)
    for row in store.results(run_id):
        if row["refused"]:
            continue
        cited = [RetrievedChunk(by_citation[c], 1.0) for c in row["citations"] if c in by_citation]
        answer = Answer(text=row["answer"], citations=row["citations"], refused=False, retrieved=cited)
        fields = {}
        grounded = _safely("groundedness", row["question_id"], score_groundedness, answer, settings)
        if grounded is not None:
            fields["groundedness"] = grounded
        if row["correctness"] is None:
            correct = _safely("correctness", row["question_id"], score_correctness,
                              answer, points.get(row["question_id"], []), settings)
            if correct is not None:
                fields["correctness"] = correct
        if fields:
            store.update(run_id, row["question_id"], fields)
        print(f"  {row['question_id']}: {fields}")


def summarise(results: list[dict]) -> dict:
    def block(rows: list[dict]) -> dict:
        out = {}
        missing = 0
        for field in SCORE_FIELDS:
            values = [r[field] for r in rows if r.get(field) is not None]
            missing += len(rows) - len(values)
            out[field] = round(mean(values), 3) if values else None
        flags = [r["refusal_correct"] for r in rows if r.get("refusal_correct") is not None]
        out["refusal_accuracy"] = round(mean(flags), 3) if flags else None
        routing = [r["routing_correct"] for r in rows if r.get("routing_correct") is not None]
        out["routing_accuracy"] = round(mean(routing), 3) if routing else None
        out["n"] = len(rows)
        out["missing_scores"] = missing
        return out

    summary = {"overall": block(results)}
    answerable = [r for r in results if r["band"] != "unanswerable"]
    if answerable and len(answerable) < len(results):
        summary["answerable"] = block(answerable)
    for band in sorted({r["band"] for r in results}):
        summary[band] = block([r for r in results if r["band"] == band])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the obrag golden-set evaluation.")
    parser.add_argument("--label", help="name for this run, e.g. 'baseline'")
    parser.add_argument("--resume", type=int, metavar="RUN_ID", help="finish an interrupted run")
    parser.add_argument("--rescore", type=int, metavar="RUN_ID", help="re-judge a finished run")
    args = parser.parse_args()

    settings = load_settings()
    if args.rescore is not None:
        rescore(args.rescore, settings)
        for band, scores in summarise(EvalStore(settings).results(args.rescore)).items():
            print(f"  {band}: {scores}")
        return
    if not args.label:
        parser.error("--label is required unless --rescore is given")
    print(
        f"running '{args.label}' with {settings.embedding_model} / "
        f"{settings.generation_model} / judge {settings.judge_model}"
    )
    run_id = run_eval(args.label, settings, resume_run_id=args.resume)

    summary = summarise(EvalStore(settings).results(run_id))
    print(f"\nrun {run_id} — {args.label}")
    for band, scores in summary.items():
        print(f"  {band}: {scores}")


if __name__ == "__main__":
    main()
