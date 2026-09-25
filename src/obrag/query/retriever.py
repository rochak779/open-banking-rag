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

        per_collection = [
            [r for r in self._store.query(name, embedding, top_k=top_k) if r.score >= self._min_score]
            for name in collections
        ]

        # Each routed collection is guaranteed an equal share of the slots; a
        # plain merge by score let the spec's near-identical endpoint chunks
        # crowd the law out of mixed questions. Unused share goes to the best
        # of the rest.
        share = -(-top_k // max(len(collections), 1))  # ceiling division
        chosen = [r for results in per_collection for r in results[:share]]
        rest = [r for results in per_collection for r in results[share:]]
        rest.sort(key=lambda r: r.score, reverse=True)
        chosen.sort(key=lambda r: r.score, reverse=True)
        chosen = chosen[:top_k] + rest[: max(top_k - len(chosen), 0)]

        chosen.sort(key=lambda r: r.score, reverse=True)
        return chosen
