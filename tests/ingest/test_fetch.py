from pathlib import Path

import pytest

from obrag.ingest.fetch import LEGISLATION_SOURCES, fetch_legislation, fetch_obl_spec


def test_legislation_sources_cover_psr_and_sca_rts():
    assert "psr_2017" in LEGISLATION_SOURCES
    assert "sca_rts" in LEGISLATION_SOURCES
    for url in LEGISLATION_SOURCES.values():
        assert url.endswith("/data.xml")


def test_fetch_legislation_caches_and_does_not_redownload(tmp_path, monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200
        content = b"<Legislation/>"

        def raise_for_status(self):
            return None

    def fake_get(url, timeout=None, headers=None):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr("obrag.ingest.fetch.requests.get", fake_get)

    first = fetch_legislation("psr_2017", raw_dir=tmp_path)
    second = fetch_legislation("psr_2017", raw_dir=tmp_path)

    assert first == second
    assert first.read_bytes() == b"<Legislation/>"
    assert len(calls) == 1, "second call must hit the cache, not the network"


def test_fetch_legislation_rejects_unknown_name(tmp_path):
    with pytest.raises(KeyError):
        fetch_legislation("not_a_real_source", raw_dir=tmp_path)


@pytest.mark.network
def test_fetch_obl_spec_clones_pinned_tag(tmp_path):
    path = fetch_obl_spec(raw_dir=tmp_path, tag="v4.0.1-Update-1")
    yamls = list(path.glob("*.yaml"))
    assert yamls, f"expected OpenAPI yaml files in {path}"
