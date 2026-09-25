"""Four scorers: three LLM judges and one truth table.

Hand-written rather than Ragas — see the note in the plan. Each judge gets the
narrowest possible question and returns a single number, because a judge asked
to assess several things at once returns an average of vibes.

The judge model comes from settings. On the free tier it is Gemma 4 31B (set
via OBRAG_JUDGE_MODEL), a different model family from the Gemini generator,
which softens the self-grading problem without removing it: same vendor, and
an open model smaller than the planned Gemini Pro judge. Session 12 must say
so. score_refusal, the metric that matters most, has no model in it at all.
"""

import re

from google import genai
from pydantic import BaseModel, Field

from obrag.config import Settings
from obrag.models import Answer


class Judgement(BaseModel):
    reasoning: str = Field(description="One or two sentences justifying the score.")
    score: float = Field(ge=0.0, le=1.0, description="The score between 0 and 1.")


def _parse_judgement(text: str) -> Judgement:
    """Parse the judge's JSON, tolerating a markdown code fence around it.

    Gemma wraps JSON-mode replies in ```json fences often enough to matter.
    """
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    return Judgement.model_validate_json(match.group(0) if match else text)


def _judge(prompt: str, system: str, settings: Settings, client=None) -> float:
    client = client or genai.Client(api_key=settings.gemini_api_key)
    interaction = client.interactions.create(
        model=settings.judge_model,
        system_instruction=system,
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": Judgement.model_json_schema(),
        },
    )
    return float(_parse_judgement(interaction.output_text).score)


def _sources_block(answer: Answer) -> str:
    """Number sources the way the answer cites them: cited first, in citation order.

    The generator renumbers its markers so "[1]" is the first source it cited,
    not the first one retrieved. Numbering by retrieval order here made the
    judge check claims against the wrong source and score correct answers 0.
    """
    rank = {citation: i for i, citation in enumerate(answer.citations)}
    ordered = sorted(answer.retrieved, key=lambda item: rank.get(item.chunk.citation, len(rank)))
    return "\n\n".join(
        f"[{i}] {item.chunk.citation}\n{item.chunk.text}"
        for i, item in enumerate(ordered, start=1)
    )


GROUNDEDNESS_SYSTEM = (
    "You check whether an answer is supported by its sources. Score 1.0 if every "
    "factual claim is supported by the numbered sources. Score 0.0 if the answer "
    "asserts things the sources do not contain. Score in between when some claims "
    "are supported and others are not. Judge support only — not whether the answer "
    "is correct in the real world, and not whether it is well written."
)


def score_groundedness(answer: Answer, settings: Settings, client=None) -> float:
    # A refusal asserts nothing, so it cannot be ungrounded.
    if answer.refused:
        return 1.0
    prompt = f"Sources:\n\n{_sources_block(answer)}\n\nAnswer:\n{answer.text}"
    return _judge(prompt, GROUNDEDNESS_SYSTEM, settings, client)


CORRECTNESS_SYSTEM = (
    "You check whether an answer covers a list of expected points. Score the "
    "fraction of expected points the answer makes, allowing for different wording. "
    "Do not penalise extra correct detail. Do not reward fluency."
)


def score_correctness(
    answer: Answer, expected_points: list[str], settings: Settings, client=None
) -> float:
    if not expected_points:
        return 1.0
    if answer.refused:
        return 0.0
    points = "\n".join(f"- {point}" for point in expected_points)
    prompt = f"Expected points:\n{points}\n\nAnswer:\n{answer.text}"
    return _judge(prompt, CORRECTNESS_SYSTEM, settings, client)


RELEVANCE_SYSTEM = (
    "You check whether retrieved sources are relevant to a question. Score the "
    "fraction of the numbered sources that could plausibly contribute to answering "
    "it. Judge the sources, not any answer."
)


def score_retrieval_relevance(
    question: str, answer: Answer, settings: Settings, client=None
) -> float:
    if not answer.retrieved:
        return 0.0
    prompt = f"Question: {question}\n\nRetrieved sources:\n\n{_sources_block(answer)}"
    return _judge(prompt, RELEVANCE_SYSTEM, settings, client)


def score_refusal(answer: Answer, answerable: bool) -> bool:
    """True when the system's decision to answer or refuse was the right one.

    answerable | refused | correct
    -----------|---------|--------
    True       | False   | answered a question it could answer
    True       | True    | wrongly refused — a miss
    False      | True    | correctly declined
    False      | False   | answered something it had no basis for — the worst case
    """
    return answer.refused != answerable
