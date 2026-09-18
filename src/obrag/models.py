"""Shared vocabulary for the whole pipeline.

Every stage — ingestion, indexing, retrieval, generation, eval — passes these
three types around. Keeping them in one small module is what lets each stage be
written and tested without importing any other stage.
"""

from dataclasses import dataclass, field
from typing import Literal

Collection = Literal["regulation", "spec"]


@dataclass(frozen=True)
class Chunk:
    """One indexable unit of source material, with everything needed to cite it."""

    id: str
    text: str
    collection: Collection
    citation: str
    source_url: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]
    refused: bool
    retrieved: list[RetrievedChunk]
