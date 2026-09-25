"""Synthesise one grounded answer from the retrieved chunks, or refuse.

Refusing is a first-class outcome, not an error path: a compliance-adjacent tool
that always produces an answer is worse than one that admits a gap. There are
three ways to refuse here — nothing retrieved, the model says so, or the API
failed — and all three produce the same Answer shape so callers need no
special-casing.
"""

import re

from google import genai

from obrag.config import Settings
from obrag.models import Answer, RetrievedChunk

REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"

NO_CONTEXT_TEXT = (
    "I don't have anything in the indexed sources that answers this. The index "
    "covers the Payment Services Regulations 2017, the retained SCA-RTS, and the "
    "Open Banking Read/Write API specification."
)

API_FAILURE_TEXT = (
    "The answer service is currently unavailable, so I can't produce a grounded "
    "answer. Please try again."
)

SYSTEM = f"""You answer questions about UK Open Banking using only the numbered sources provided.

Rules:
- Use only the supplied sources. Never use prior knowledge about UK payments law or the Open Banking spec, even if you are confident it is correct.
- Cite with the bracketed number of every source you rely on, inline, like [1] or [2]. Every factual claim needs a marker.
- Where a regulation and the API specification both bear on the question, explain how they relate rather than listing them separately.
- If the sources do not contain enough to answer, reply with exactly "{REFUSAL_SENTINEL}: " followed by one sentence naming what is missing. Do not guess, and do not answer partially from memory.
- Be concise and concrete. No preamble, no restating the question."""

# "[2]" or a grouped "[1, 3]".
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# The sentinel at the start of the reply, tolerating markdown emphasis around it.
_REFUSAL = re.compile(rf"^[\s*_]*{REFUSAL_SENTINEL}[\s*_]*:?\s*")


def _format_sources(retrieved: list[RetrievedChunk]) -> str:
    blocks = []
    for index, item in enumerate(retrieved, start=1):
        blocks.append(f"[{index}] {item.chunk.citation}\n{item.chunk.text}")
    return "\n\n".join(blocks)


def _renumber_markers(text: str, count: int) -> tuple[str, list[int]]:
    """Rewrite markers so they count 1..n in order of first citation.

    Returns the rewritten text and the original source indices in that order.
    Callers list Answer.citations as a numbered 1..n list, so "[3]" in the text
    has to become "[1]" when source 3 is the first one cited. Markers pointing
    at sources that do not exist are dropped rather than shown.
    """
    order: list[int] = []

    def rewrite(match: re.Match) -> str:
        markers = []
        for number in re.split(r"\s*,\s*", match.group(1)):
            index = int(number)
            if not 1 <= index <= count:
                continue
            if index not in order:
                order.append(index)
            markers.append(f"[{order.index(index) + 1}]")
        return "".join(markers)

    return _MARKER.sub(rewrite, text), order


def generate(
    question: str,
    retrieved: list[RetrievedChunk],
    settings: Settings,
    client=None,
) -> Answer:
    if not retrieved:
        return Answer(text=NO_CONTEXT_TEXT, citations=[], refused=True, retrieved=[])

    prompt = f"Sources:\n\n{_format_sources(retrieved)}\n\nQuestion: {question}"

    try:
        client = client or genai.Client(api_key=settings.gemini_api_key)
        interaction = client.interactions.create(
            model=settings.generation_model,
            system_instruction=SYSTEM,
            input=prompt,
        )
    except Exception:
        return Answer(text=API_FAILURE_TEXT, citations=[], refused=True, retrieved=retrieved)

    text = (interaction.output_text or "").strip()
    refusal = _REFUSAL.match(text)
    if refusal:
        reason = text[refusal.end() :].strip()
        reason = reason[:1].upper() + reason[1:]
        return Answer(text=reason or NO_CONTEXT_TEXT, citations=[], refused=True, retrieved=retrieved)

    text, indices = _renumber_markers(text, len(retrieved))
    citations = [retrieved[i - 1].chunk.citation for i in indices]
    return Answer(text=text, citations=citations, refused=False, retrieved=retrieved)
