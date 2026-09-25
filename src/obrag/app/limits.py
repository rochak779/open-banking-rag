"""Keep a public demo from burning its rate-limit quota on casual traffic.

The budget is per browser session and trivially bypassed by opening a new tab —
that is fine and intentional. It stops accidental loops and idle over-use so the
demo still works when it matters; staying on the Gemini free tier is what caps
the downside to throttling rather than a bill.
"""

from collections.abc import Callable
from typing import Literal

from obrag.models import Answer
from obrag.query.generator import API_FAILURE_TEXT

STATE_KEY = "obrag_queries_used"

_CACHE: dict[str, Answer] = {}

Source = Literal["example", "session", "fresh", "over_budget"]


def _normalise(question: str) -> str:
    return " ".join(question.lower().split())


def cache_answer(question: str, answer: Answer) -> None:
    # A failed call is transient; caching it would show "unavailable" to every
    # visitor until the process restarts.
    if answer.text == API_FAILURE_TEXT:
        return
    _CACHE[_normalise(question)] = answer


def cached_answer(question: str) -> Answer | None:
    return _CACHE.get(_normalise(question))


class QueryBudget:
    def __init__(self, max_queries: int = 10):
        self._max = max_queries

    def remaining(self, state) -> int:
        return max(0, self._max - int(state.get(STATE_KEY, 0)))

    def consume(self, state) -> bool:
        used = int(state.get(STATE_KEY, 0))
        if used >= self._max:
            return False
        state[STATE_KEY] = used + 1
        return True


def resolve_answer(
    question: str,
    session_answers: dict,
    state,
    budget: QueryBudget,
    ask_fn: Callable[[str], Answer],
) -> tuple[Answer | None, Source]:
    """Find an answer as cheaply as possible, spending budget only on new questions.

    Order: shared example cache (free), this session's earlier answers (free —
    Streamlit reruns the script on every click, so the same question comes back
    on each tab switch), then a real ask() if the session still has budget.
    """
    example = cached_answer(question)
    if example is not None:
        return example, "example"
    key = _normalise(question)
    if key in session_answers:
        return session_answers[key], "session"
    if not budget.consume(state):
        return None, "over_budget"
    session_answers[key] = ask_fn(question)
    return session_answers[key], "fresh"
