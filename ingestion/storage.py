"""SQLite stores backing the ingestion pipeline.

All three live in the same database file (`data/papers.db` by default), one
table each: raw fetched papers, normalized papers, and the embedding cache.
"""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Self

from ingestion.models import PaperMetadata


class _SqliteStore:
    """Connection lifecycle shared by every store: open (creating parent
    directories), init the schema, count rows, close, and use as a context
    manager. Subclasses set `_table` and implement `_init_schema`.
    """

    _table: str

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        raise NotImplementedError

    def count(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _PaperStore(_SqliteStore):
    """Shared read path for the two stores that hold PaperMetadata rows
    (authors/categories are stored as JSON text).
    """

    def all(self) -> Iterator[PaperMetadata]:
        cursor = self._conn.execute(
            f"""
            SELECT arxiv_id, version, title, abstract, authors, categories,
                   primary_category, published, updated, pdf_url
            FROM {self._table}
            """
        )
        for (
            arxiv_id,
            version,
            title,
            abstract,
            authors_json,
            categories_json,
            primary_category,
            published,
            updated,
            pdf_url,
        ) in cursor:
            yield PaperMetadata(
                arxiv_id=arxiv_id,
                version=version,
                title=title,
                abstract=abstract,
                authors=json.loads(authors_json),
                categories=json.loads(categories_json),
                primary_category=primary_category,
                published=published,
                updated=updated,
                pdf_url=pdf_url,
            )


class RawPaperStore(_PaperStore):
    """SQLite-backed cache of fetched arXiv metadata, keyed by arxiv_id.

    Upserts only move a record forward to a newer paper version, which
    makes re-running a fetch idempotent regardless of API pagination order.
    """

    _table = "papers"

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT NOT NULL,
                categories TEXT NOT NULL,
                primary_category TEXT NOT NULL,
                published TEXT NOT NULL,
                updated TEXT NOT NULL,
                pdf_url TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def upsert(self, paper: PaperMetadata) -> None:
        self._conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, version, title, abstract, authors, categories,
                primary_category, published, updated, pdf_url, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                version=excluded.version,
                title=excluded.title,
                abstract=excluded.abstract,
                authors=excluded.authors,
                categories=excluded.categories,
                primary_category=excluded.primary_category,
                published=excluded.published,
                updated=excluded.updated,
                pdf_url=excluded.pdf_url,
                fetched_at=excluded.fetched_at
            WHERE excluded.version >= papers.version
            """,
            (
                paper.arxiv_id,
                paper.version,
                paper.title,
                paper.abstract,
                json.dumps(paper.authors),
                json.dumps(paper.categories),
                paper.primary_category,
                paper.published,
                paper.updated,
                paper.pdf_url,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()


class NormalizedPaperStore(_PaperStore):
    """SQLite store for papers that passed ingestion.normalize's filter and
    cleaning pass, keyed by arxiv_id.

    Unlike RawPaperStore, upsert is a plain replace: normalization is
    deterministic given the raw record, so the latest run always wins.
    """

    _table = "normalized_papers"

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS normalized_papers (
                arxiv_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT NOT NULL,
                categories TEXT NOT NULL,
                primary_category TEXT NOT NULL,
                published TEXT NOT NULL,
                updated TEXT NOT NULL,
                pdf_url TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def upsert(self, paper: PaperMetadata) -> None:
        self._conn.execute(
            """
            INSERT INTO normalized_papers (
                arxiv_id, version, title, abstract, authors, categories,
                primary_category, published, updated, pdf_url
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                version=excluded.version,
                title=excluded.title,
                abstract=excluded.abstract,
                authors=excluded.authors,
                categories=excluded.categories,
                primary_category=excluded.primary_category,
                published=excluded.published,
                updated=excluded.updated,
                pdf_url=excluded.pdf_url
            """,
            (
                paper.arxiv_id,
                paper.version,
                paper.title,
                paper.abstract,
                json.dumps(paper.authors),
                json.dumps(paper.categories),
                paper.primary_category,
                paper.published,
                paper.updated,
                paper.pdf_url,
            ),
        )
        self._conn.commit()


class EmbeddingCacheStore(_SqliteStore):
    """Tracks which chunks have already been embedded and upserted, keyed by
    chunk_id, recording a hash of the text that was embedded.

    Re-running the embed stage skips any chunk whose text hasn't changed
    since the last run - the main cost control called for in
    docs/data-ingestion-plan.md step 7 (embeddings aren't cheap to redo
    across a 30k-paper corpus on every run).
    """

    _table = "embedded_chunks"

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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
