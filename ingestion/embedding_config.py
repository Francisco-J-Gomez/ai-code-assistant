import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_pinecone_index_name() -> str:
    # A default_factory (evaluated at EmbeddingConfig() construction time,
    # not import time) rather than a fixed literal: a bare EmbeddingConfig()
    # should pick up PINECONE_INDEX_NAME from .env by default, so nothing
    # that constructs one - CLI, agent, an ad hoc script - can silently
    # target the wrong (or a newly auto-created, empty) Pinecone index by
    # forgetting to resolve the env var itself. Requires load_dotenv() to
    # have already run, same as every other env-var read in this project.
    return os.environ.get("PINECONE_INDEX_NAME", "ai-code-assistant")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Knobs for the embed-and-upsert stage. Secrets (OPENAI_API_KEY,
    PINECONE_API_KEY) are read from the environment at the point of use, not
    stored here - see ingestion/embeddings.py and ingestion/pinecone_client.py.
    """

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embed_batch_size: int = 100  # texts per OpenAI embeddings.create call
    limit: int | None = None  # cap papers processed - for smoke-testing before a full run
    pinecone_index_name: str = field(default_factory=_default_pinecone_index_name)
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    max_retries: int = 5
    db_path: Path = Path("data/papers.db")  # same SQLite file as ingestion; adds embedded_chunks table
