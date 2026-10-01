import time
from dataclasses import replace

from obrag.config import Settings
from obrag.query.router import route


def make_settings():
    return Settings(gemini_api_key="ai-x", voyage_api_key="pa-x")


class FakeGemini:
    """Stands in for genai.Client(): client.interactions.create(...).output_text."""

    def __init__(self, reply: str):
        self._reply = reply
        self.calls = []
        self.interactions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("I", (), {"output_text": self._reply})()


def test_keywords_from_both_sides_route_to_both_without_an_api_call():
    fake = FakeGemini("spec")
    assert route(
        "Under the PSRs, what does POST /domestic-payments need?", make_settings(), client=fake
    ) == ["regulation", "spec"]
    assert fake.calls == []


def test_one_sided_keywords_do_not_narrow_the_route_on_their_own():
    # Baseline eval: the keyword shortcut routed 4/9 golden questions correctly
    # against 17/21 for the model, and caused 5 of the 6 cross-cutting misses
    # ("under the PSRs ... which API endpoints?" went to regulation only).
    fake = FakeGemini("both")
    assert route(
        "The PSRs say an ASPSP must give a PISP information. Which API endpoints provide it?",
        make_settings(), client=fake,
    ) == ["regulation", "spec"]
    assert len(fake.calls) == 1


def test_the_model_can_still_route_a_keyword_question_to_one_collection():
    fake = FakeGemini("spec")
    assert route("What does POST /domestic-payments return?", make_settings(), client=fake) == ["spec"]
    fake = FakeGemini("regulation")
    assert route("What does regulation 68 of the PSRs require?", make_settings(), client=fake) == [
        "regulation"
    ]


def test_ambiguous_question_asks_the_model():
    fake = FakeGemini("both")
    assert route("Does the consent flow satisfy SCA?", make_settings(), client=fake) == [
        "regulation",
        "spec",
    ]
    assert len(fake.calls) == 1


def test_router_uses_the_cheap_model():
    fake = FakeGemini("spec")
    route("Does the consent flow satisfy SCA?", make_settings(), client=fake)
    assert fake.calls[0]["model"] == "gemini-3.1-flash-lite"


def test_a_stalled_router_call_falls_back_to_both_at_the_deadline():
    # gemini-3.5-flash-lite held about every other free-tier request for 40-60s
    # (2026-09-27), and the SDK's own timeout let a 36s call through. The deadline
    # is wall-clock, so a stall costs the visitor at most router_timeout_seconds.
    class Stalling(FakeGemini):
        def create(self, **kwargs):
            time.sleep(2)
            return super().create(**kwargs)

    settings = replace(make_settings(), router_timeout_seconds=0.1)
    start = time.perf_counter()
    assert route("Does the consent flow satisfy SCA?", settings, client=Stalling("spec")) == [
        "regulation",
        "spec",
    ]
    assert time.perf_counter() - start < 1


def test_single_word_reply_is_tolerant_of_case_and_punctuation():
    fake = FakeGemini(" Spec.\n")
    assert route("Does the consent flow satisfy SCA?", make_settings(), client=fake) == ["spec"]


def test_unparseable_model_reply_falls_back_to_both():
    fake = FakeGemini("I think probably the specification, but honestly")
    assert route("Does the consent flow satisfy SCA?", make_settings(), client=fake) == [
        "regulation",
        "spec",
    ]


def test_api_failure_falls_back_to_both():
    class Exploding:
        def __init__(self):
            self.interactions = self

        def create(self, **kwargs):
            raise RuntimeError("network down")

    assert route("Does the consent flow satisfy SCA?", make_settings(), client=Exploding()) == [
        "regulation",
        "spec",
    ]
