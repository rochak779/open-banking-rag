from obrag.models import Chunk, RetrievedChunk, Answer


def test_chunk_carries_citation_and_collection():
    chunk = Chunk(
        id="reg:2017/752/68/1",
        text="A payment service provider must...",
        collection="regulation",
        citation="PSR 2017, reg. 68(1)",
        source_url="https://www.legislation.gov.uk/uksi/2017/752/regulation/68",
    )
    assert chunk.collection == "regulation"
    assert chunk.citation == "PSR 2017, reg. 68(1)"
    assert chunk.metadata == {}


def test_answer_records_refusal_and_citations():
    answer = Answer(text="I don't have...", citations=[], refused=True, retrieved=[])
    assert answer.refused is True
    assert answer.citations == []


def test_retrieved_chunk_pairs_chunk_with_score():
    chunk = Chunk(id="a", text="t", collection="spec", citation="c", source_url="u")
    assert RetrievedChunk(chunk=chunk, score=0.81).score == 0.81
