from obrag.config import Settings
from obrag.models import Answer, Chunk, RetrievedChunk
from obrag.query import pipeline


def make_settings():
    return Settings(gemini_api_key="ai-x", voyage_api_key="pa-x")


def test_ask_wires_router_retriever_and_generator_in_order(monkeypatch):
    seen = {}
    hit = RetrievedChunk(
        Chunk(id="c", text="t", collection="regulation", citation="PSR 2017, regulation 1",
              source_url="https://example.invalid"),
        0.9,
    )

    monkeypatch.setattr(pipeline, "route", lambda q, s, client=None: seen.setdefault("route", ["regulation"]))

    class FakeRetriever:
        def __init__(self, settings):
            pass

        def retrieve(self, question, collections):
            seen["collections"] = collections
            return [hit]

    monkeypatch.setattr(pipeline, "Retriever", FakeRetriever)
    monkeypatch.setattr(
        pipeline, "generate",
        lambda q, r, s, client=None: Answer(text="grounded [1]", citations=["PSR 2017, regulation 1"],
                                            refused=False, retrieved=r),
    )

    answer = pipeline.ask("when is SCA required?", settings=make_settings())
    assert seen["collections"] == ["regulation"]
    assert answer.citations == ["PSR 2017, regulation 1"]
    assert answer.refused is False


def test_ask_returns_a_refusal_when_nothing_is_retrieved(monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda q, s, client=None: ["regulation", "spec"])

    class EmptyRetriever:
        def __init__(self, settings):
            pass

        def retrieve(self, question, collections):
            return []

    monkeypatch.setattr(pipeline, "Retriever", EmptyRetriever)
    answer = pipeline.ask("what is the capital of France?", settings=make_settings())
    assert answer.refused is True
    assert answer.citations == []


def test_ask_with_route_returns_the_collections_the_answer_used(monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda q, s, client=None: ["spec"])

    class EmptyRetriever:
        def __init__(self, settings):
            pass

        def retrieve(self, question, collections):
            return []

    monkeypatch.setattr(pipeline, "Retriever", EmptyRetriever)
    collections, answer = pipeline.ask_with_route("q", settings=make_settings())
    assert collections == ["spec"]
    assert answer.refused is True
