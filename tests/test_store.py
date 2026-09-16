from pathlib import Path

from ingestion.models import PaperMetadata
from ingestion.store import RawPaperStore


def make_paper(**overrides) -> PaperMetadata:
    defaults = dict(
        arxiv_id="2301.12345",
        version=1,
        title="Title",
        abstract="Abstract",
        authors=["A"],
        categories=["cs.SE"],
        primary_category="cs.SE",
        published="2023-01-01T00:00:00Z",
        updated="2023-01-01T00:00:00Z",
        pdf_url="http://arxiv.org/pdf/2301.12345v1",
    )
    defaults.update(overrides)
    return PaperMetadata(**defaults)


def test_upsert_inserts_new_paper(tmp_path: Path):
    with RawPaperStore(tmp_path / "papers.db") as store:
        store.upsert(make_paper())
        assert store.count() == 1


def test_upsert_is_idempotent(tmp_path: Path):
    with RawPaperStore(tmp_path / "papers.db") as store:
        store.upsert(make_paper())
        store.upsert(make_paper())
        assert store.count() == 1


def test_upsert_keeps_latest_version_even_if_older_refetched_later(tmp_path: Path):
    with RawPaperStore(tmp_path / "papers.db") as store:
        store.upsert(make_paper(version=1, title="Old title"))
        store.upsert(make_paper(version=2, title="New title"))
        store.upsert(make_paper(version=1, title="Stale re-fetch"))

        row = store._conn.execute(
            "SELECT version, title FROM papers WHERE arxiv_id = ?",
            ("2301.12345",),
        ).fetchone()

        assert row == (2, "New title")


def test_creates_db_parent_directory(tmp_path: Path):
    nested_path = tmp_path / "nested" / "dir" / "papers.db"
    with RawPaperStore(nested_path) as store:
        store.upsert(make_paper())
        assert nested_path.exists()
