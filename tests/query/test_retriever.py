from obrag.config import Settings
from obrag.models import Chunk, RetrievedChunk
from obrag.query.retriever import Retriever


def make_settings(**kw):
    base = dict(gemini_api_key="ai-x", voyage_api_key="pa-x")
    base.update(kw)
    return Settings(**base)


def chunk(chunk_id, collection):
    return Chunk(
        id=chunk_id, text=f"text for {chunk_id}", collection=collection,
        citation=chunk_id, source_url="https://example.invalid",
    )


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return [1.0, 0.0]


class FakeStore:
    def __init__(self, by_collection):
        self._by_collection = by_collection
        self.queried = []

    def query(self, name, embedding, top_k):
        self.queried.append(name)
        return self._by_collection.get(name, [])[:top_k]


def test_only_the_routed_collections_are_queried():
    store = FakeStore({
        "regulation": [RetrievedChunk(chunk("r1", "regulation"), 0.9)],
        "spec": [RetrievedChunk(chunk("s1", "spec"), 0.8)],
    })
    Retriever(make_settings(), embedder=FakeEmbedder(), store=store).retrieve("q", ["spec"])
    assert store.queried == ["spec"]


def test_the_question_is_embedded_once_for_both_collections():
    embedder = FakeEmbedder()
    Retriever(make_settings(), embedder=embedder, store=FakeStore({})).retrieve(
        "q", ["regulation", "spec"]
    )
    assert embedder.calls == 1


def test_results_from_both_collections_are_merged_and_sorted_by_score():
    store = FakeStore({
        "regulation": [RetrievedChunk(chunk("r1", "regulation"), 0.5)],
        "spec": [RetrievedChunk(chunk("s1", "spec"), 0.9)],
    })
    results = Retriever(make_settings(), embedder=FakeEmbedder(), store=store).retrieve(
        "q", ["regulation", "spec"]
    )
    assert [r.chunk.id for r in results] == ["s1", "r1"]


def test_results_are_capped_at_top_k():
    store = FakeStore({
        "regulation": [RetrievedChunk(chunk(f"r{i}", "regulation"), 0.9 - i / 100) for i in range(10)],
        "spec": [RetrievedChunk(chunk(f"s{i}", "spec"), 0.8 - i / 100) for i in range(10)],
    })
    results = Retriever(make_settings(top_k=4), embedder=FakeEmbedder(), store=store).retrieve(
        "q", ["regulation", "spec"]
    )
    assert len(results) == 4


def test_low_scoring_chunks_are_dropped():
    store = FakeStore({"regulation": [RetrievedChunk(chunk("r1", "regulation"), 0.05)]})
    results = Retriever(
        make_settings(), embedder=FakeEmbedder(), store=store, min_score=0.2
    ).retrieve("q", ["regulation"])
    assert results == []


def test_no_results_is_an_empty_list_not_an_error():
    results = Retriever(make_settings(), embedder=FakeEmbedder(), store=FakeStore({})).retrieve(
        "q", ["regulation", "spec"]
    )
    assert results == []
