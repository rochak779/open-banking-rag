import pytest

from obrag.config import Settings
from obrag.index.store import ChunkStore
from obrag.models import Chunk


def make_settings(tmp_path):
    return Settings(
        gemini_api_key="ai-x",
        voyage_api_key="pa-x",
        chroma_dir=tmp_path / "chroma",
    )


REG = Chunk(
    id="reg:psr_2017:regulation-68",
    text="A payment service provider must provide the information.",
    collection="regulation",
    citation="PSR 2017, regulation 68",
    source_url="https://www.legislation.gov.uk/uksi/2017/752/regulation/68",
    metadata={"instrument": "PSR 2017"},
)
SPEC = Chunk(
    id="spec:payment-initiation-openapi:post:/domestic-payments",
    text="Endpoint: POST /domestic-payments. Create a domestic payment.",
    collection="spec",
    citation="Payment Initiation API v4.0.1 — POST /domestic-payments",
    source_url="https://github.com/OpenBankingUK/read-write-api-specs",
    metadata={"method": "POST"},
)


def test_upsert_then_query_returns_the_chunk_from_its_own_collection(tmp_path):
    store = ChunkStore(make_settings(tmp_path))
    store.upsert([REG, SPEC], [[1.0, 0.0], [0.0, 1.0]])

    results = store.query("regulation", [1.0, 0.0], top_k=5)
    assert [r.chunk.id for r in results] == [REG.id]


def test_collections_are_isolated(tmp_path):
    store = ChunkStore(make_settings(tmp_path))
    store.upsert([REG, SPEC], [[1.0, 0.0], [0.0, 1.0]])
    spec_ids = [r.chunk.id for r in store.query("spec", [0.0, 1.0], top_k=5)]
    assert REG.id not in spec_ids
    assert store.count("regulation") == 1
    assert store.count("spec") == 1


def test_round_trip_preserves_the_whole_chunk(tmp_path):
    store = ChunkStore(make_settings(tmp_path))
    store.upsert([REG], [[1.0, 0.0]])
    assert store.query("regulation", [1.0, 0.0], top_k=1)[0].chunk == REG


def test_upsert_is_idempotent(tmp_path):
    store = ChunkStore(make_settings(tmp_path))
    store.upsert([REG], [[1.0, 0.0]])
    store.upsert([REG], [[1.0, 0.0]])
    assert len(store.query("regulation", [1.0, 0.0], top_k=10)) == 1


def test_scores_are_similarities_not_distances(tmp_path):
    store = ChunkStore(make_settings(tmp_path))
    store.upsert([REG], [[1.0, 0.0]])
    score = store.query("regulation", [1.0, 0.0], top_k=1)[0].score
    assert 0.99 <= score <= 1.0


def test_space_is_cosine(tmp_path):
    # Identical vectors score 1.0 under L2 as well; only an orthogonal pair
    # tells the spaces apart (cosine -> 0.0, squared L2 -> -1.0).
    store = ChunkStore(make_settings(tmp_path))
    store.upsert([REG], [[3.0, 0.0]])
    score = store.query("regulation", [0.0, 1.0], top_k=1)[0].score
    assert score == pytest.approx(0.0, abs=1e-6)


def test_mismatched_lengths_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        ChunkStore(make_settings(tmp_path)).upsert([REG, SPEC], [[1.0, 0.0]])
