import logging

from ingestion.chunking.abstract_chunker import AbstractChunker
from ingestion.chunking.base import Chunk
from ingestion.clients.arxiv_client import fetch_papers
from ingestion.clients.embeddings import Embedder
from ingestion.clients.pinecone_client import PineconeStore
from ingestion.config import EmbeddingConfig, IngestionConfig
from ingestion.normalize import normalize_paper
from ingestion.storage import EmbeddingCacheStore, NormalizedPaperStore, RawPaperStore

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


def run_embed_stage(embedding_config: EmbeddingConfig) -> tuple[int, int]:
    """Chunk normalized papers, embed chunks whose text changed since the
    last run, and upsert them to Pinecone. Returns (embedded, skipped_unchanged).
    """
    chunker = AbstractChunker()
    embedder = Embedder(model=embedding_config.embedding_model)
    pinecone_store = PineconeStore(
        index_name=embedding_config.pinecone_index_name,
        dimension=embedding_config.embedding_dimensions,
        cloud=embedding_config.pinecone_cloud,
        region=embedding_config.pinecone_region,
    )

    with NormalizedPaperStore(embedding_config.db_path) as norm_store:
        papers = list(norm_store.all())  # see run_normalize_stage for why this can't stream
    if embedding_config.limit is not None:
        papers = papers[: embedding_config.limit]

    pending: list[Chunk] = []
    skipped = 0
    with EmbeddingCacheStore(embedding_config.db_path) as cache:
        for paper in papers:
            for chunk in chunker.chunk(paper):
                if cache.is_up_to_date(chunk.chunk_id, chunk.text):
                    skipped += 1
                else:
                    pending.append(chunk)

    embedded = 0
    with EmbeddingCacheStore(embedding_config.db_path) as cache:
        for batch in _batched(pending, embedding_config.embed_batch_size):
            vectors = embedder.embed([c.text for c in batch])
            pinecone_store.upsert(batch, vectors)
            cache.mark_embedded_many([(c.chunk_id, c.text) for c in batch])
            embedded += len(batch)
            logger.info("Embedded %d/%d chunks so far...", embedded, len(pending))

    logger.info(
        "Done. Embedded %d chunks, skipped %d unchanged. Pinecone index now has %d vectors.",
        embedded,
        skipped,
        pinecone_store.count(),
    )
    return embedded, skipped


def _batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
