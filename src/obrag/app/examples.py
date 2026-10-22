"""The example questions and their saved answers, served at launch with no API call.

Answering them live on every cold start took about a minute (measured 2026-10-22)
before the app showed anything. Re-run `python -m obrag.app.examples` after
changing an example, the prompt, the index or the generation model, or the saved
answers go stale.
"""

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from obrag.config import DATA, load_settings
from obrag.models import Answer, Chunk, RetrievedChunk
from obrag.query.generator import API_FAILURE_TEXT
from obrag.query.pipeline import ask

EXAMPLES_PATH = DATA / "examples.json"

# Chosen from the golden-set runs: each answered (or, for the last, refused)
# correctly with full marks, so the saved answers are the system at its best on
# each band. The original headline SCA question is declined (see the case study).
EXAMPLES = [
    "What must a funds confirmation for a card-based payment contain under the PSRs, "
    "and how does the Confirmation of Funds API return it?",
    "How much can a payer be made to pay for an unauthorised payment made with a lost or stolen card?",
    "What happens if a PISP sends the same x-idempotency-key twice when creating a domestic payment?",
    "What is Barclays' rate limit on the accounts endpoint?",
]


def save_examples(answers: dict[str, Answer], path: Path = EXAMPLES_PATH) -> None:
    payload = {question: asdict(answer) for question, answer in answers.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_examples(path: Path = EXAMPLES_PATH) -> dict[str, Answer]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        question: Answer(
            text=answer["text"],
            citations=answer["citations"],
            refused=answer["refused"],
            retrieved=[RetrievedChunk(Chunk(**r["chunk"]), r["score"]) for r in answer["retrieved"]],
        )
        for question, answer in payload.items()
    }


def refresh_examples(ask_fn: Callable[[str], Answer], path: Path = EXAMPLES_PATH) -> None:
    """Answer every example and save them all, or save nothing if any call failed."""
    answers = {question: ask_fn(question) for question in EXAMPLES}
    failed = [question for question, answer in answers.items() if answer.text == API_FAILURE_TEXT]
    if failed:
        raise RuntimeError(f"Answer service unavailable for {len(failed)} example(s), nothing saved: {failed}")
    save_examples(answers, path)


if __name__ == "__main__":
    settings = load_settings()
    refresh_examples(lambda question: ask(question, settings=settings))
    print(f"saved {len(EXAMPLES)} answers to {EXAMPLES_PATH}")
