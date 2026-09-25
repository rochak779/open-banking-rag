"""The one public entrypoint: question in, grounded Answer out.

The UI, the eval harness and any future API all call ask(). Nothing else should
need to know that a router or a retriever exists.
"""

from obrag.config import Settings, load_settings
from obrag.models import Answer
from obrag.query.generator import generate
from obrag.query.retriever import Retriever
from obrag.query.router import route


def ask(question: str, settings: Settings | None = None) -> Answer:
    settings = settings or load_settings()
    collections = route(question, settings)
    retriever = Retriever(settings)
    retrieved = retriever.retrieve(question, collections)
    return generate(question, retrieved, settings)
