from obrag.config import Settings
from obrag.models import Chunk, RetrievedChunk
from obrag.query.generator import generate


def make_settings(**kw):
    base = dict(gemini_api_key="ai-x", voyage_api_key="pa-x")
    base.update(kw)
    return Settings(**base)


def retrieved(*specs):
    return [
        RetrievedChunk(
            Chunk(
                id=f"c{i}", text=text, collection=collection,
                citation=citation, source_url="https://example.invalid",
            ),
            score,
        )
        for i, (text, collection, citation, score) in enumerate(specs)
    ]


REG = "PSR 2017, regulation 100"
SPEC = "Payment Initiation API v4.0.1 — POST /domestic-payment-consents"
RTS = "SCA-RTS (EU) 2018/389, article 4"

HITS = retrieved(
    ("SCA must be applied when a payer initiates an electronic payment.", "regulation", REG, 0.82),
    ("Endpoint: POST /domestic-payment-consents. Creates a payment consent.", "spec", SPEC, 0.77),
    ("The authentication code must be one-time use.", "regulation", RTS, 0.70),
)


class FakeGemini:
    def __init__(self, reply):
        self._reply = reply
        self.calls = []
        self.interactions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("I", (), {"output_text": self._reply})()


def test_no_retrieval_refuses_without_calling_the_api():
    fake = FakeGemini("should never be called")
    answer = generate("anything", [], make_settings(), client=fake)
    assert answer.refused is True
    assert answer.citations == []
    assert fake.calls == []


def test_prompt_numbers_the_sources_and_includes_their_citations():
    fake = FakeGemini("Yes, per [1].")
    generate("Does the consent flow satisfy SCA?", HITS, make_settings(), client=fake)
    prompt = fake.calls[0]["input"]
    assert "[1]" in prompt and "[2]" in prompt
    assert REG in prompt
    assert "POST /domestic-payment-consents" in prompt


def test_citations_list_only_the_sources_actually_referenced():
    fake = FakeGemini("SCA applies here [1].")
    answer = generate("q", HITS, make_settings(), client=fake)
    assert answer.citations == [REG]


def test_multiple_markers_produce_multiple_citations_in_order():
    fake = FakeGemini("The rule [1] is implemented by the endpoint [2].")
    answer = generate("q", HITS, make_settings(), client=fake)
    assert answer.citations == [REG, SPEC]


def test_markers_are_renumbered_to_match_the_citation_list():
    # The UI numbers sources 1..n in citation order, so "[3]" in the text must
    # become "[1]" when source 3 is the first one cited.
    fake = FakeGemini("The code is one-time [3], and SCA applies [1].")
    answer = generate("q", HITS, make_settings(), client=fake)
    assert answer.citations == [RTS, REG]
    assert answer.text == "The code is one-time [1], and SCA applies [2]."


def test_grouped_markers_are_recognised():
    fake = FakeGemini("Both apply [1, 3].")
    answer = generate("q", HITS, make_settings(), client=fake)
    assert answer.citations == [REG, RTS]
    assert answer.text == "Both apply [1][2]."


def test_out_of_range_markers_are_dropped():
    fake = FakeGemini("Claim [1] and invented [9].")
    answer = generate("q", HITS, make_settings(), client=fake)
    assert answer.citations == [REG]
    assert "[9]" not in answer.text


def test_refusal_sentinel_sets_refused_and_is_stripped_from_the_text():
    fake = FakeGemini("INSUFFICIENT_CONTEXT: the sources do not cover this.")
    answer = generate("q", HITS, make_settings(), client=fake)
    assert answer.refused is True
    assert answer.citations == []
    assert answer.text == "The sources do not cover this."


def test_refusal_sentinel_is_recognised_inside_markdown():
    fake = FakeGemini("**INSUFFICIENT_CONTEXT**: nothing on mortgages.")
    assert generate("q", HITS, make_settings(), client=fake).refused is True


def test_uses_the_generation_model_and_passes_the_rules_as_a_system_instruction():
    fake = FakeGemini("Answer [1].")
    generate("q", HITS, make_settings(), client=fake)
    call = fake.calls[0]
    assert call["model"] == "gemini-3.8-flash"
    assert "Never use prior knowledge" in call["system_instruction"]


def test_api_failure_refuses_rather_than_raising():
    class Exploding:
        def __init__(self):
            self.interactions = self

        def create(self, **kwargs):
            raise RuntimeError("503")

    answer = generate("q", HITS, make_settings(), client=Exploding())
    assert answer.refused is True
    assert "unavailable" in answer.text.lower()


def test_retrieved_chunks_are_carried_through_for_the_ui():
    fake = FakeGemini("Answer [1].")
    assert generate("q", HITS, make_settings(), client=fake).retrieved == HITS


def test_an_answer_with_no_citation_markers_is_declined():
    # Seen in the golden-set run (cross-09): a fluent answer with no markers at
    # all was shown as a normal answer with an empty source list.
    fake = FakeGemini("An AISP must not access other accounts and must get explicit consent.")
    answer = generate("q", HITS, make_settings(), client=fake)
    assert answer.refused is True
    assert answer.citations == []
    assert "cite" in answer.text.lower()
    assert answer.retrieved == HITS  # still inspectable in the UI


def test_an_answer_citing_only_nonexistent_sources_is_declined():
    fake = FakeGemini("Claim [7].")
    assert generate("q", HITS, make_settings(), client=fake).refused is True
