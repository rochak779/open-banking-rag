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


def test_endpoint_keywords_route_to_spec_without_an_api_call():
    fake = FakeGemini("both")
    assert route("What does POST /domestic-payments return?", make_settings(), client=fake) == ["spec"]
    assert fake.calls == []


def test_regulation_keywords_route_to_regulation_without_an_api_call():
    fake = FakeGemini("both")
    assert route("What does regulation 68 of the PSRs require?", make_settings(), client=fake) == [
        "regulation"
    ]
    assert fake.calls == []


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
    assert fake.calls[0]["model"] == "gemini-3.5-flash-lite"


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
