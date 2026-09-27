"""Single source of truth for keys, model IDs, paths and the legal boilerplate.

Keys are read from the environment and never written to disk. On Streamlit
Community Cloud, secrets declared in the app's secrets manager are injected as
environment variables, so this module works unchanged in both places.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

SNAPSHOT_DATE = "2026-09-18"

DISCLAIMER = (
    "This tool is a technical demonstration. It is not legal, regulatory or "
    f"compliance advice. Answers are drawn from a snapshot taken {SNAPSHOT_DATE} "
    "and may be out of date. SCA-RTS citations are the retained EU text published "
    "on legislation.gov.uk, which does not include the FCA's later amendments. "
    "Always check the cited source."
)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    voyage_api_key: str

    # Exact IDs from ai.google.dev/gemini-api/docs/models. Flash-tier generation is
    # the right call here: synthesis over supplied context is not a hard reasoning
    # task, and the cheap tier keeps a public demo inside the free rate limits.
    generation_model: str = "gemini-3.8-flash"
    router_model: str = "gemini-3.5-flash-lite"
    # Deliberately a stronger model than the generator. It does not solve the
    # self-judging problem — same vendor, same family — but it blunts it. If the
    # preview model proves unstable, fall back to "gemini-2.5-pro".
    judge_model: str = "gemini-3.1-pro-preview"

    embedding_model: str = "voyage-4-lite"

    obl_spec_tag: str = "v4.0.1-Update-1"
    top_k: int = 6
    # Retrieval score floor. Tuned on voyage-4-lite (2026-09-25): off-topic questions
    # topped out at 0.244, the weakest on-topic question scored 0.312. Score scales
    # differ between embedding models, so re-tune it whenever the model changes.
    min_score: float = 0.28
    # Gemini retries per call; None keeps the SDK's default. The demo app sets 0 so a
    # visitor on an exhausted quota gets an answer or an error in seconds.
    gemini_retries: int | None = None

    raw_dir: Path = DATA / "raw"
    chroma_dir: Path = DATA / "chroma"
    golden_path: Path = DATA / "golden" / "questions.jsonl"
    eval_db_path: Path = DATA / "eval.db"


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in, "
            "or set it in the Streamlit Cloud secrets manager."
        )
    return value


def load_settings() -> Settings:
    return Settings(
        gemini_api_key=_require("GEMINI_API_KEY"),
        voyage_api_key=_require("VOYAGE_API_KEY"),
        embedding_model=os.environ.get("OBRAG_EMBEDDING_MODEL", "voyage-4-lite"),
        # Free-tier escape hatch: gemini-3.8-flash allows only 20 requests/day on the
        # free tier, so .env can point generation at a model with more headroom.
        generation_model=os.environ.get("OBRAG_GENERATION_MODEL", "gemini-3.8-flash"),
        # gemini-3.1-pro-preview has no free-tier quota; .env points the judge at Gemma.
        judge_model=os.environ.get("OBRAG_JUDGE_MODEL", "gemini-3.1-pro-preview"),
        min_score=float(os.environ.get("OBRAG_MIN_SCORE", "0.28")),
        # Lets a second embedding model be indexed side by side for an A/B.
        chroma_dir=Path(os.environ.get("OBRAG_CHROMA_DIR", DATA / "chroma")),
    )
