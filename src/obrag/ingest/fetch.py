"""Acquire the raw v1 sources and pin them on disk.

Everything downstream reads from data/raw/. Re-running this module is cheap and
idempotent; re-downloading on every parse run is not, and would make the
snapshot date meaningless.
"""

import shutil
import subprocess
from pathlib import Path

import requests

OBL_SPEC_REPO = "https://github.com/OpenBankingUK/read-write-api-specs.git"

LEGISLATION_SOURCES = {
    # The Payment Services Regulations 2017 (SI 2017/752)
    "psr_2017": "https://www.legislation.gov.uk/uksi/2017/752/data.xml",
    # Commission Delegated Regulation (EU) 2018/389 — the SCA-RTS, retained in UK law
    "sca_rts": "https://www.legislation.gov.uk/eur/2018/389/data.xml",
}

USER_AGENT = "obrag/0.1 (open-banking-rag portfolio project)"


def fetch_legislation(name: str, raw_dir: Path) -> Path:
    """Download one legislation.gov.uk CLML document, or return the cached copy."""
    url = LEGISLATION_SOURCES[name]  # KeyError on unknown name is the intended behaviour
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{name}.xml"
    if target.exists():
        return target
    response = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def fetch_obl_spec(raw_dir: Path, tag: str) -> Path:
    """Shallow-clone the OBL spec repo at a pinned tag; return the OpenAPI dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = raw_dir / f"obl-specs-{tag}"
    openapi_dir = repo_dir / "dist" / "openapi"
    if openapi_dir.exists():
        return openapi_dir
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", tag, OBL_SPEC_REPO, str(repo_dir)],
        check=True,
        capture_output=True,
    )
    if not openapi_dir.exists():
        raise FileNotFoundError(
            f"Expected {openapi_dir} after cloning tag {tag}. "
            "The repo layout may have changed — list the clone and adjust the path."
        )
    return openapi_dir
