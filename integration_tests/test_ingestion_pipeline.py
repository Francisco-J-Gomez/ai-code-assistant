"""End-to-end check of the ingestion pipeline on a ~1K-paper slice.

Runs fetch -> normalize -> embed against the real arXiv, OpenAI and Pinecone
APIs, so we know the whole pipeline works without running the full ~30k
corpus. Skipped by default since it costs a little money (well under a cent
of embeddings), needs credentials, and takes a minute or two (arXiv's ~3s
courtesy delay between requests). Run explicitly with:

    RUN_INGESTION_INTEGRATION_TESTS=1 uv run pytest integration_tests/test_ingestion_pipeline.py -v

Embeds into a throwaway Pinecone index (never the real corpus index), which
is deleted afterwards even if the test fails.
"""
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from pinecone import Pinecone

from ingestion.config import IngestionConfig
from ingestion.embedding_config import EmbeddingConfig
from ingestion.embeddings import Embedder
from ingestion.pinecone_client import PineconeStore
from ingestion.pipeline import run_embed_stage, run_fetch_stage, run_normalize_stage

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INGESTION_INTEGRATION_TESTS") != "1",
    reason=(
        "Hits real arXiv/OpenAI/Pinecone APIs; "
        "set RUN_INGESTION_INTEGRATION_TESTS=1 to run."
    ),
)

SAMPLE_SIZE = 1000
# Pinecone's reported vector count can lag a few seconds behind an upsert.
COUNT_SETTLE_TIMEOUT_SECONDS = 60


@dataclass
class PipelineRun:
    fetched: int
    kept: int
    dropped: int
    embedded: int
    embedding_config: EmbeddingConfig


@pytest.fixture(scope="module")
def scratch_index_name():
    name = f"ingestion-it-{uuid.uuid4().hex[:8]}"
    yield name
    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if name in client.list_indexes().names():
        client.delete_index(name)


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory, scratch_index_name) -> PipelineRun:
    db_path = tmp_path_factory.mktemp("ingestion") / "papers.db"

    ingestion_config = IngestionConfig(max_results=SAMPLE_SIZE, db_path=db_path)
    embedding_config = EmbeddingConfig(
        pinecone_index_name=scratch_index_name, db_path=db_path
    )

    fetched = run_fetch_stage(ingestion_config)
    kept, dropped = run_normalize_stage(ingestion_config)
    embedded, _ = run_embed_stage(embedding_config)

    return PipelineRun(fetched, kept, dropped, embedded, embedding_config)


def _store_for(config: EmbeddingConfig) -> PineconeStore:
    return PineconeStore(
        index_name=config.pinecone_index_name,
        dimension=config.embedding_dimensions,
        cloud=config.pinecone_cloud,
        region=config.pinecone_region,
    )


def _wait_for_count(store: PineconeStore, expected: int) -> int:
    deadline = time.monotonic() + COUNT_SETTLE_TIMEOUT_SECONDS
    count = store.count()
    while count != expected and time.monotonic() < deadline:
        time.sleep(3)
        count = store.count()
    return count


def test_fetch_returns_the_requested_sample(pipeline_run: PipelineRun):
    assert pipeline_run.fetched == SAMPLE_SIZE


def test_normalize_keeps_most_papers(pipeline_run: PipelineRun):
    assert pipeline_run.kept + pipeline_run.dropped == pipeline_run.fetched
    assert pipeline_run.kept >= pipeline_run.fetched * 0.5


def test_every_kept_paper_is_embedded_and_lands_in_pinecone(pipeline_run: PipelineRun):
    # One chunk per paper (abstract) for the MVP.
    assert pipeline_run.embedded == pipeline_run.kept

    store = _store_for(pipeline_run.embedding_config)
    assert _wait_for_count(store, pipeline_run.kept) == pipeline_run.kept


def test_query_returns_chunks_with_citation_metadata(pipeline_run: PipelineRun):
    store = _store_for(pipeline_run.embedding_config)
    _wait_for_count(store, pipeline_run.kept)

    embedder = Embedder(model=pipeline_run.embedding_config.embedding_model)
    [vector] = embedder.embed(["automated program repair with language models"])
    results = store.query(vector, top_k=3)

    assert results
    for result in results:
        assert result["title"]
        assert result["arxiv_id"]
        assert result["source_url"].startswith("http")


def test_rerunning_embed_is_idempotent(pipeline_run: PipelineRun):
    embedded, skipped = run_embed_stage(pipeline_run.embedding_config)

    assert embedded == 0
    assert skipped == pipeline_run.kept
