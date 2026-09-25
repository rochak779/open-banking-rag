"""Embed the question once, query the routed collections, merge by score."""

from obrag.config import Settings
from obrag.index.embedder import Embedder
from obrag.index.store import ChunkStore
from obrag.models import Collection, RetrievedChunk

# Tuned on voyage-4-lite (2026-09-25): off-topic questions topped out at 0.244,
# the weakest on-topic question scored 0.312. Re-tune if the embedding model changes.
DEFAULT_MIN_SCORE = 0.28


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedder=None,
        store=None,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        self._settings = settings
        self._embedder = embedder or Embedder(settings)
        self._store = store or ChunkStore(settings)
        self._min_score = min_score

    def retrieve(
        self, question: str, collections: list[Collection]
    ) -> list[RetrievedChunk]:
        embedding = self._embedder.embed_query(question)
        top_k = self._settings.top_k

        merged: list[RetrievedChunk] = []
        for name in collections:
            merged.extend(self._store.query(name, embedding, top_k=top_k))

        kept = [r for r in merged if r.score >= self._min_score]
        kept.sort(key=lambda r: r.score, reverse=True)
        return kept[:top_k]
