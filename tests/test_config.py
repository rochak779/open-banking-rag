import pytest

from obrag.config import DISCLAIMER, SNAPSHOT_DATE, Settings, load_settings


def test_load_settings_reads_keys_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    s = load_settings()
    assert s.gemini_api_key == "ai-test"
    assert s.voyage_api_key == "pa-test"


def test_model_ids_are_exact(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    s = load_settings()
    assert s.generation_model == "gemini-3.8-flash"
    assert s.router_model == "gemini-3.5-flash-lite"
    assert s.judge_model == "gemini-3.1-pro-preview"


def test_missing_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        load_settings()


def test_disclaimer_and_snapshot_date_are_present():
    assert "not legal" in DISCLAIMER.lower()
    assert SNAPSHOT_DATE.startswith("20")


def test_settings_is_frozen(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    s = load_settings()
    with pytest.raises(Exception):
        s.top_k = 99  # type: ignore[misc]
