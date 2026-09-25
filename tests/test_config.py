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
    monkeypatch.delenv("OBRAG_GENERATION_MODEL", raising=False)
    monkeypatch.delenv("OBRAG_JUDGE_MODEL", raising=False)
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
    # The indexed SCA-RTS predates the FCA's amendments (e.g. article 10A); users must be told.
    assert "retained EU text" in DISCLAIMER
    assert "FCA's later amendments" in DISCLAIMER
    assert SNAPSHOT_DATE.startswith("20")


def test_settings_is_frozen(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    s = load_settings()
    with pytest.raises(Exception):
        s.top_k = 99  # type: ignore[misc]


def test_generation_model_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    monkeypatch.setenv("OBRAG_GENERATION_MODEL", "gemini-3.5-flash-lite")
    assert load_settings().generation_model == "gemini-3.5-flash-lite"


def test_judge_model_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    monkeypatch.setenv("OBRAG_JUDGE_MODEL", "gemma-4-31b-it")
    assert load_settings().judge_model == "gemma-4-31b-it"


def test_chroma_dir_can_be_overridden_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "ai-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    monkeypatch.setenv("OBRAG_CHROMA_DIR", str(tmp_path / "chroma-law"))
    assert load_settings().chroma_dir == tmp_path / "chroma-law"
