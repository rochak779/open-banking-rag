"""Chroma persistence: two collections, cosine space, similarity scores out.

The separate-collections decision from the design lives here and nowhere else.
Callers name a collection; they never see a Chroma object.
"""

import chromadb

from obrag.config import Settings
from obrag.models import Chunk, Collection, RetrievedChunk

COLLECTIONS: tuple[Collection, ...] = ("regulation", "spec")

# Chunk fields stored in Chroma metadata alongside Chunk.metadata.
_RESERVED = ("citation", "source_url")


class ChunkStore:
    def __init__(self, settings: Settings):
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))

    def _collection(self, name: Collection):
        return self._client.get_or_create_collection(
            name=name, configuration={"hnsw": {"space": "cosine"}}
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings — they must correspond"
            )
        for name in COLLECTIONS:
            pairs = [(c, e) for c, e in zip(chunks, embeddings) if c.collection == name]
            if not pairs:
                continue
            self._collection(name).upsert(
                ids=[c.id for c, _ in pairs],
                embeddings=[e for _, e in pairs],
                documents=[c.text for c, _ in pairs],
                metadatas=[
                    {**c.metadata, "citation": c.citation, "source_url": c.source_url}
                    for c, _ in pairs
                ],
            )

    def count(self, name: Collection) -> int:
        return self._collection(name).count()

    def query(
        self, name: Collection, embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        result = self._collection(name).query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        retrieved: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            extra = {k: v for k, v in metadata.items() if k not in _RESERVED}
            retrieved.append(
                RetrievedChunk(
                    chunk=Chunk(
                        id=chunk_id,
                        text=document,
                        collection=name,
                        citation=metadata.get("citation", ""),
                        source_url=metadata.get("source_url", ""),
                        metadata=extra,
                    ),
                    # Chroma returns cosine distance (lower is better). Every consumer
                    # downstream expects "higher is better", so convert exactly once, here.
                    score=1.0 - float(distance),
                )
            )
        return retrieved
