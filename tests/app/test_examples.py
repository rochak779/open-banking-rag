import pytest

from obrag.app.examples import EXAMPLES, load_examples, refresh_examples, save_examples
from obrag.models import Answer, Chunk, RetrievedChunk
from obrag.query.generator import API_FAILURE_TEXT

HIT = RetrievedChunk(
    Chunk(id="c1", text="body", collection="regulation", citation="PSR 2017, regulation 100",
          source_url="https://www.legislation.gov.uk/uksi/2017/752/regulation/100",
          metadata={"number": "100"}),
    0.91,
)


def test_saved_answers_load_back_unchanged(tmp_path):
    path = tmp_path / "examples.json"
    answers = {"When is SCA required?": Answer(text="SCA applies [1].", citations=["PSR 2017, regulation 100"],
                                               refused=False, retrieved=[HIT])}
    save_examples(answers, path)
    assert load_examples(path) == answers


def test_the_committed_file_answers_every_example():
    # The app serves these at launch without an API call. An example added to
    # EXAMPLES without re-running `python -m obrag.app.examples` would instead
    # cost a live call from every visitor's budget.
    answers = load_examples()
    assert set(EXAMPLES) <= set(answers)
    assert all(answer.text != API_FAILURE_TEXT for answer in answers.values())


def test_a_failed_call_is_never_written_to_the_file(tmp_path):
    path = tmp_path / "examples.json"

    def failing(question):
        return Answer(text=API_FAILURE_TEXT, citations=[], refused=True, retrieved=[])

    with pytest.raises(RuntimeError, match="unavailable"):
        refresh_examples(failing, path)
    assert not path.exists()
