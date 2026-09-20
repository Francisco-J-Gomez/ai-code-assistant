from pathlib import Path

from ingestion.models import PaperMetadata
from ingestion.normalized_store import NormalizedPaperStore


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
    with NormalizedPaperStore(tmp_path / "normalized.db") as store:
        store.upsert(make_paper())
        assert store.count() == 1


def test_upsert_replaces_existing_paper(tmp_path: Path):
    with NormalizedPaperStore(tmp_path / "normalized.db") as store:
        store.upsert(make_paper(title="Old title"))
        store.upsert(make_paper(title="New title"))

        assert store.count() == 1
        row = store._conn.execute(
            "SELECT title FROM normalized_papers WHERE arxiv_id = ?", ("2301.12345",)
        ).fetchone()
        assert row == ("New title",)
