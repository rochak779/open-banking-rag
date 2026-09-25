"""Voyage embeddings, batched, with the document/query asymmetry made explicit."""

import voyageai

from obrag.config import Settings

DEFAULT_BATCH_SIZE = 64

# The SDK default is no retries, so a single 429 would abort an index build.
MAX_RETRIES = 5


class Embedder:
    def __init__(self, settings: Settings, client=None, batch_size: int = DEFAULT_BATCH_SIZE):
        self._model = settings.embedding_model
        self._batch_size = batch_size
        self._client = client or voyageai.Client(
            api_key=settings.voyage_api_key, max_retries=MAX_RETRIES
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(
                self._client.embed(batch, model=self._model, input_type="document").embeddings
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed([text], model=self._model, input_type="query").embeddings[0]
