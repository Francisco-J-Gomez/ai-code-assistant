import json
import sqlite3
from pathlib import Path

from ingestion.models import PaperMetadata


class NormalizedPaperStore:
    """SQLite store for papers that passed ingestion.normalize's filter and
    cleaning pass, keyed by arxiv_id.

    Unlike RawPaperStore, upsert is a plain replace: normalization is
    deterministic given the raw record, so the latest run always wins.
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._init_schema()

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

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM normalized_papers").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "NormalizedPaperStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
