import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class EmbeddingCacheStore:
    """Tracks which chunks have already been embedded and upserted, keyed by
    chunk_id, recording a hash of the text that was embedded.

    Re-running the embed stage skips any chunk whose text hasn't changed
    since the last run - the main cost control called for in
    docs/data-ingestion-plan.md step 7 (embeddings aren't cheap to redo
    across a 30k-paper corpus on every run).
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedded_chunks (
                chunk_id TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                embedded_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def is_up_to_date(self, chunk_id: str, text: str) -> bool:
        row = self._conn.execute(
            "SELECT text_hash FROM embedded_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return row is not None and row[0] == _hash_text(text)

    def mark_embedded(self, chunk_id: str, text: str) -> None:
        self.mark_embedded_many([(chunk_id, text)])

    def mark_embedded_many(self, chunk_id_and_text: list[tuple[str, str]]) -> None:
        # One commit for the whole batch rather than per row - committing
        # ~30k times individually (as the normalize stage originally did)
        # took ~90s just on fsync overhead for that many papers.
        now = datetime.now(timezone.utc).isoformat()
        self._conn.executemany(
            """
            INSERT INTO embedded_chunks (chunk_id, text_hash, embedded_at)
            VALUES (?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                text_hash=excluded.text_hash,
                embedded_at=excluded.embedded_at
            """,
            [(chunk_id, _hash_text(text), now) for chunk_id, text in chunk_id_and_text],
        )
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM embedded_chunks").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EmbeddingCacheStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
