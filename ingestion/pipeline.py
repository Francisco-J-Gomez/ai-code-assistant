import logging

from ingestion.arxiv_client import fetch_papers
from ingestion.config import IngestionConfig
from ingestion.normalize import normalize_paper
from ingestion.normalized_store import NormalizedPaperStore
from ingestion.store import RawPaperStore

logger = logging.getLogger(__name__)


def run_fetch_stage(config: IngestionConfig) -> int:
    """Fetch arXiv metadata into the raw store. Returns papers fetched."""
    fetched_count = 0
    with RawPaperStore(config.db_path) as store:
        for paper in fetch_papers(config):
            store.upsert(paper)
            fetched_count += 1
            if fetched_count % 50 == 0:
                logger.info("Fetched %d papers so far...", fetched_count)

        logger.info(
            "Done. Fetched %d papers this run. Store now has %d papers.",
            fetched_count,
            store.count(),
        )
    return fetched_count


def run_normalize_stage(config: IngestionConfig) -> tuple[int, int]:
    """Filter/clean raw papers into the normalized store. Returns (kept, dropped)."""
    kept, dropped = 0, 0
    with RawPaperStore(config.db_path) as raw_store, NormalizedPaperStore(config.db_path) as norm_store:
        # Materialize before writing: raw_store.all() is a live cursor, and
        # a second connection to the same SQLite file can't commit writes
        # while that read transaction is still open ("database is locked").
        papers = list(raw_store.all())
        for paper in papers:
            normalized = normalize_paper(paper, config)
            if normalized is None:
                dropped += 1
                continue
            norm_store.upsert(normalized)
            kept += 1

        logger.info(
            "Done. Normalized %d papers, dropped %d. Store now has %d normalized papers.",
            kept,
            dropped,
            norm_store.count(),
        )
    return kept, dropped
