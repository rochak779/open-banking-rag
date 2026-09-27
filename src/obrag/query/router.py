"""Decide which collections a question needs.

Keyword rules can only widen the route: a question with markers from both sides
goes to both collections without a model call. Everything else goes to one
flash-lite call. Every uncertain path resolves to both collections:
over-retrieving costs a fraction of a penny, under-retrieving produces a
confident answer with the wrong half of the story.

Keywords used to narrow the route too, sending "under the PSRs ... which API
endpoints?" to regulation only. On the baseline golden-set run the keyword
shortcut routed 4 of 9 questions correctly against 17 of 21 for the model, and
caused 5 of the 6 cross-cutting routing misses.
"""

import re


from obrag.config import Settings
from obrag.gemini import client_for
from obrag.models import Collection

BOTH: list[Collection] = ["regulation", "spec"]

SPEC_PATTERNS = (
    r"\b(get|post|put|patch|delete)\s+/",
    r"\bendpoint\b",
    r"\bapi\s+(call|response|request|schema|field)\b",
    r"\bjson\b",
    r"\bschema\b",
    r"\bheader\b",
    r"\bstatus code\b",
    r"\bOB[A-Z]\w+",
)

REGULATION_PATTERNS = (
    r"\bregulation\s+\d",
    r"\barticle\s+\d",
    r"\bPSRs?\b",
    r"\bPSD2\b",
    r"\bRTS\b",
    r"\blegal(ly)?\b",
    r"\brequired by law\b",
    r"\bobligation\b",
)

SYSTEM = (
    "You route questions about UK Open Banking to source collections.\n"
    "'regulation' holds UK payment services law: the Payment Services Regulations 2017 "
    "and the retained SCA-RTS.\n"
    "'spec' holds the Open Banking Limited Read/Write API specification: endpoints, "
    "request and response schemas, headers, and how consent, payment and account flows "
    "work in the API.\n"
    "Answer 'both' when the question touches both sides — for example whether an API "
    "flow, consent or endpoint meets a legal requirement such as SCA — or when you are "
    "unsure. Answer 'regulation' or 'spec' only when the question is clearly about one.\n"
    "Answer with exactly one word: regulation, spec, or both. No punctuation, no explanation."
)


def _matches(patterns: tuple[str, ...], question: str) -> bool:
    return any(re.search(p, question, flags=re.IGNORECASE) for p in patterns)


def route(question: str, settings: Settings, client=None) -> list[Collection]:
    if _matches(SPEC_PATTERNS, question) and _matches(REGULATION_PATTERNS, question):
        return BOTH

    try:
        client = client or client_for(settings)
        interaction = client.interactions.create(
            model=settings.router_model,
            system_instruction=SYSTEM,
            input=question,
        )
        # The prompt asks for no punctuation; tolerate "Spec." anyway.
        reply = re.sub(r"[^a-z]", "", (interaction.output_text or "").lower())
    except Exception:
        return BOTH

    if reply == "spec":
        return ["spec"]
    if reply == "regulation":
        return ["regulation"]
    return BOTH
