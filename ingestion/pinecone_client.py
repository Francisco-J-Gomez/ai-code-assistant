import logging
import os
from typing import Sequence

from pinecone import Pinecone, ServerlessSpec

from ingestion.chunking.base import Chunk

logger = logging.getLogger(__name__)


class PineconeStore:
    """Thin wrapper around one Pinecone index: creates it if missing and
    upserts chunk vectors with citation metadata attached.

    Vector id = chunk.chunk_id (arxiv_id:chunk_index) - re-running upsert
    overwrites in place, which is what makes the whole pipeline idempotent
    (docs/data-ingestion-plan.md step 8).
    """

    def __init__(
        self,
        index_name: str,
        dimension: int,
        cloud: str,
        region: str,
        api_key: str | None = None,
    ):
        self._client = Pinecone(api_key=api_key or os.environ.get("PINECONE_API_KEY"))

        if index_name not in self._client.list_indexes().names():
            logger.info("Creating Pinecone index %r (dim=%d, %s/%s)", index_name, dimension, cloud, region)
            self._client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self._index = self._client.Index(index_name)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} != {len(vectors)}")
        if not chunks:
            return

        items = [
            (
                chunk.chunk_id,
                vector,
                {
                    "arxiv_id": chunk.arxiv_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "title": chunk.title,
                    "authors": chunk.authors,
                    "categories": chunk.categories,
                    "published": chunk.published,
                    "source_url": chunk.source_url,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._index.upsert(vectors=items)

    def count(self) -> int:
        return self._index.describe_index_stats().total_vector_count
