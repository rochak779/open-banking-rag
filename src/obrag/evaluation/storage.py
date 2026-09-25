"""The golden set and the results database.

Results are kept in SQLite rather than files so runs can be compared directly —
the before/after table is the case study's headline artifact, and it should come
out of a query, not out of hand-copied numbers.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from obrag.config import DATA, Settings
from obrag.models import Collection

GOLDEN_BANDS = ("regulation_only", "spec_only", "cross_cutting", "unanswerable")

DEFAULT_GOLDEN_PATH = DATA / "golden" / "questions.jsonl"  # mirrors Settings.golden_path


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    band: str
    question: str
    expected_collections: list[Collection]
    expected_points: list[str]
    answerable: bool


def load_golden(path: Path | None = None) -> list[GoldenQuestion]:
    path = path or DEFAULT_GOLDEN_PATH
    questions: list[GoldenQuestion] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        questions.append(
            GoldenQuestion(
                id=raw["id"],
                band=raw["band"],
                question=raw["question"],
                expected_collections=raw.get("expected_collections", []),
                expected_points=raw.get("expected_points", []),
                answerable=raw.get("answerable", True),
            )
        )
    return questions


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    config TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    question_id TEXT NOT NULL,
    band TEXT NOT NULL,
    answer TEXT,
    refused INTEGER,
    citations TEXT,
    routing_correct INTEGER,
    retrieval_relevance REAL,
    groundedness REAL,
    correctness REAL,
    refusal_correct INTEGER
);
"""

RESULT_FIELDS = (
    "question_id", "band", "answer", "refused", "citations", "routing_correct",
    "retrieval_relevance", "groundedness", "correctness", "refusal_correct",
)


class EvalStore:
    def __init__(self, settings: Settings):
        settings.eval_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = settings.eval_db_path
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Commit on success, roll back on error, and always close.

        sqlite3's own context manager commits but leaves the connection open.
        """
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create_run(self, label: str, config: dict) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO runs (label, config) VALUES (?, ?)",
                (label, json.dumps(config, sort_keys=True)),
            )
            return int(cursor.lastrowid)

    def record(self, run_id: int, result: dict) -> None:
        row = {field: result.get(field) for field in RESULT_FIELDS}
        row["citations"] = json.dumps(result.get("citations", []))
        for flag in ("refused", "routing_correct", "refusal_correct"):
            if row[flag] is not None:
                row[flag] = int(bool(row[flag]))
        columns = ", ".join(RESULT_FIELDS)
        placeholders = ", ".join(f":{field}" for field in RESULT_FIELDS)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO results (run_id, {columns}) VALUES (:run_id, {placeholders})",
                {"run_id": run_id, **row},
            )

    def results(self, run_id: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        out = []
        for row in rows:
            record = dict(row)
            record["citations"] = json.loads(record["citations"] or "[]")
            out.append(record)
        return out

    def runs(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
        return [{**dict(row), "config": json.loads(row["config"])} for row in rows]
