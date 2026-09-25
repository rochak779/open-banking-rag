from obrag.app.limits import QueryBudget, cache_answer, cached_answer, resolve_answer
from obrag.query.generator import API_FAILURE_TEXT
from obrag.models import Answer


def test_budget_starts_full():
    assert QueryBudget(max_queries=5).remaining({}) == 5


def test_consuming_decrements_the_remaining_count():
    budget, state = QueryBudget(max_queries=2), {}
    assert budget.consume(state) is True
    assert budget.remaining(state) == 1


def test_budget_refuses_once_exhausted():
    budget, state = QueryBudget(max_queries=1), {}
    assert budget.consume(state) is True
    assert budget.consume(state) is False
    assert budget.remaining(state) == 0


def test_remaining_never_goes_negative():
    budget, state = QueryBudget(max_queries=1), {}
    budget.consume(state)
    budget.consume(state)
    budget.consume(state)
    assert budget.remaining(state) == 0


def test_cached_answers_are_returned_for_known_questions():
    answer = Answer(text="cached", citations=["c"], refused=False, retrieved=[])
    cache_answer("When must SCA be applied?", answer)
    assert cached_answer("when must sca be applied?").text == "cached"


def test_unknown_questions_are_not_cached():
    assert cached_answer("something nobody has ever asked") is None


def test_an_api_failure_is_never_cached():
    # Otherwise one rate-limited cold start shows "unavailable" to every visitor.
    cache_answer("What is a failing example?", Answer(text=API_FAILURE_TEXT, citations=[], refused=True, retrieved=[]))
    assert cached_answer("What is a failing example?") is None


class Asker:
    def __init__(self):
        self.calls = []

    def __call__(self, question):
        self.calls.append(question)
        return Answer(text=f"answer to {question}", citations=[], refused=False, retrieved=[])


def test_an_example_answer_costs_nothing():
    cache_answer("Example question?", Answer(text="cached", citations=[], refused=False, retrieved=[]))
    ask, state = Asker(), {}
    answer, source = resolve_answer("example question?", {}, state, QueryBudget(1), ask)
    assert (answer.text, source) == ("cached", "example")
    assert ask.calls == [] and QueryBudget(1).remaining(state) == 1


def test_a_new_question_spends_one_query_and_reruns_spend_none():
    ask, state, session = Asker(), {}, {}
    budget = QueryBudget(3)
    first, source = resolve_answer("Fresh question?", session, state, budget, ask)
    assert source == "fresh"
    again, source = resolve_answer(" fresh question? ", session, state, budget, ask)
    assert source == "session" and again is first
    assert ask.calls == ["Fresh question?"]
    assert budget.remaining(state) == 2


def test_an_exhausted_budget_blocks_new_questions_but_not_repeats():
    ask, state, session = Asker(), {}, {}
    budget = QueryBudget(1)
    resolve_answer("First?", session, state, budget, ask)
    answer, source = resolve_answer("Second?", session, state, budget, ask)
    assert (answer, source) == (None, "over_budget")
    assert resolve_answer("First?", session, state, budget, ask)[1] == "session"
    assert ask.calls == ["First?"]
