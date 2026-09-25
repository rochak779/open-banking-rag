import json

from obrag.config import Settings
from obrag.evaluation.metrics import (
    score_correctness,
    score_groundedness,
    score_refusal,
    score_retrieval_relevance,
)
from obrag.models import Answer, Chunk, RetrievedChunk


def make_settings():
    return Settings(gemini_api_key="ai-x", voyage_api_key="pa-x")


class FakeJudge:
    """Stands in for genai.Client(): returns a fixed score as structured JSON."""

    def __init__(self, score):
        self._score = score
        self.calls = []
        self.interactions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.dumps({"reasoning": "because", "score": self._score})
        return type("I", (), {"output_text": payload})()


HIT = RetrievedChunk(
    Chunk(id="c1", text="SCA must be applied when a payer initiates an electronic payment.",
          collection="regulation", citation="PSR 2017, regulation 100",
          source_url="https://example.invalid"),
    0.9,
)

GROUNDED = Answer(text="SCA applies to electronic payments [1].",
                  citations=["PSR 2017, regulation 100"], refused=False, retrieved=[HIT])

REFUSAL = Answer(text="INSUFFICIENT_CONTEXT: not covered.", citations=[], refused=True, retrieved=[HIT])


def test_groundedness_asks_the_judge_and_returns_its_score():
    judge = FakeJudge(0.75)
    assert score_groundedness(GROUNDED, make_settings(), client=judge) == 0.75


def test_groundedness_prompt_contains_the_answer_and_the_sources():
    judge = FakeJudge(1.0)
    score_groundedness(GROUNDED, make_settings(), client=judge)
    prompt = judge.calls[0]["input"]
    assert "SCA applies to electronic payments" in prompt
    assert "SCA must be applied when a payer initiates" in prompt


def test_groundedness_of_a_refusal_is_one_without_judging():
    judge = FakeJudge(0.0)
    assert score_groundedness(REFUSAL, make_settings(), client=judge) == 1.0
    assert judge.calls == []


def test_correctness_scores_against_expected_points():
    judge = FakeJudge(0.66)
    score = score_correctness(GROUNDED, ["SCA applies to electronic payments"],
                              make_settings(), client=judge)
    assert score == 0.66
    assert "SCA applies to electronic payments" in judge.calls[0]["input"]


def test_correctness_of_a_refusal_is_zero_when_points_were_expected():
    judge = FakeJudge(1.0)
    assert score_correctness(REFUSAL, ["a point"], make_settings(), client=judge) == 0.0


def test_retrieval_relevance_uses_the_judge():
    judge = FakeJudge(0.5)
    assert score_retrieval_relevance("when is SCA required?", GROUNDED,
                                     make_settings(), client=judge) == 0.5


def test_retrieval_relevance_with_no_chunks_is_zero():
    empty = Answer(text="x", citations=[], refused=True, retrieved=[])
    judge = FakeJudge(1.0)
    assert score_retrieval_relevance("q", empty, make_settings(), client=judge) == 0.0


def test_refusal_scoring_is_deterministic_and_needs_no_judge():
    assert score_refusal(REFUSAL, answerable=False) is True    # correctly refused
    assert score_refusal(REFUSAL, answerable=True) is False    # wrongly refused
    assert score_refusal(GROUNDED, answerable=True) is True    # correctly answered
    assert score_refusal(GROUNDED, answerable=False) is False  # hallucinated


def test_judge_uses_the_judge_model_and_asks_for_json():
    judge = FakeJudge(1.0)
    score_groundedness(GROUNDED, make_settings(), client=judge)
    assert judge.calls[0]["model"] == "gemini-3.1-pro-preview"
    assert judge.calls[0]["response_format"]["mime_type"] == "application/json"


class FencedJudge(FakeJudge):
    """Gemma sometimes wraps JSON-mode output in a markdown code fence."""

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.dumps({"reasoning": "because", "score": self._score})
        return type("I", (), {"output_text": f" ```json\n{payload}\n```"})()


def test_judge_output_wrapped_in_a_code_fence_still_parses():
    assert score_groundedness(GROUNDED, make_settings(), client=FencedJudge(0.4)) == 0.4


def test_sources_are_given_to_the_judge_with_their_citations():
    judge = FakeJudge(1.0)
    score_retrieval_relevance("q", GROUNDED, make_settings(), client=judge)
    assert "[1] PSR 2017, regulation 100" in judge.calls[0]["input"]
