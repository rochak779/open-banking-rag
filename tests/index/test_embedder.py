from obrag.config import Settings
from obrag.index.embedder import Embedder


class FakeVoyage:
    def __init__(self):
        self.calls = []

    def embed(self, texts, model, input_type):
        self.calls.append({"texts": list(texts), "model": model, "input_type": input_type})
        return type("R", (), {"embeddings": [[0.1, 0.2] for _ in texts]})()


def make_settings(**overrides):
    base = dict(gemini_api_key="ai-x", voyage_api_key="pa-x")
    base.update(overrides)
    return Settings(**base)


def test_documents_use_the_document_input_type():
    fake = FakeVoyage()
    Embedder(make_settings(), client=fake).embed_documents(["a", "b"])
    assert fake.calls[0]["input_type"] == "document"


def test_queries_use_the_query_input_type():
    fake = FakeVoyage()
    Embedder(make_settings(), client=fake).embed_query("a question")
    assert fake.calls[0]["input_type"] == "query"


def test_documents_are_batched():
    fake = FakeVoyage()
    embedder = Embedder(make_settings(), client=fake, batch_size=2)
    vectors = embedder.embed_documents(["a", "b", "c", "d", "e"])
    assert len(fake.calls) == 3
    assert len(vectors) == 5


def test_embedding_model_comes_from_settings():
    fake = FakeVoyage()
    Embedder(make_settings(embedding_model="voyage-law-2"), client=fake).embed_query("q")
    assert fake.calls[0]["model"] == "voyage-law-2"
