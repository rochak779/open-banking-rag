"""The one public entrypoint: question in, grounded Answer out.

The UI, the eval harness and any future API all call ask(). Nothing else should
need to know that a router or a retriever exists.
"""

from obrag.config import Settings, load_settings
from obrag.models import Answer, Collection
from obrag.query.generator import generate
from obrag.query.retriever import Retriever
from obrag.query.router import route


def ask_with_route(
    question: str, settings: Settings | None = None
) -> tuple[list[Collection], Answer]:
    """ask(), plus the collections the router chose.

    The eval scores routing from this, so it grades the decision that actually
    produced the answer rather than a second router call that may differ.
    """
    settings = settings or load_settings()
    collections = route(question, settings)
    retriever = Retriever(settings)
    retrieved = retriever.retrieve(question, collections)
    return collections, generate(question, retrieved, settings)


def ask(question: str, settings: Settings | None = None) -> Answer:
    return ask_with_route(question, settings)[1]
