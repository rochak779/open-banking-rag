import json

import pytest

from obrag.evaluation.storage import GOLDEN_BANDS, load_golden


def test_golden_set_loads_and_is_the_expected_size():
    questions = load_golden()
    assert len(questions) == 40


def test_every_band_has_ten_questions():
    questions = load_golden()
    for band in GOLDEN_BANDS:
        assert len([q for q in questions if q.band == band]) == 10, band


def test_ids_are_unique():
    ids = [q.id for q in load_golden()]
    assert len(set(ids)) == len(ids)


def test_unanswerable_questions_are_marked_and_have_no_expected_points():
    for q in load_golden():
        if q.band == "unanswerable":
            assert q.answerable is False
            assert q.expected_points == []
        else:
            assert q.answerable is True
            assert q.expected_points, f"{q.id} needs at least one expected point"


def test_expected_collections_are_valid():
    for q in load_golden():
        assert set(q.expected_collections) <= {"regulation", "spec"}
        if q.band == "cross_cutting":
            assert set(q.expected_collections) == {"regulation", "spec"}
