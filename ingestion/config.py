import os
from dataclasses import dataclass, field
from pathlib import Path


# AI/ML/NLP papers are only included if their abstract mentions one of these
# software-development terms — cs.AI/cs.LG/cs.CL are too large and too
# generically ML-focused to ingest whole (see docs/data-ingestion-plan.md).
DEFAULT_SOFTWARE_DEV_KEYWORDS: tuple[str, ...] = (
    "code generation",
    "program synthesis",
    "program repair",
    "automated program repair",
    "code completion",
    "software engineering",
    "code review",
    "vulnerability detection",
    "bug detection",
    "code summarization",
    "test generation",
    "code search",
    "software testing",
    "code translation",
)


@dataclass(frozen=True)
class IngestionConfig:
    """Scope and behavior knobs for an arXiv metadata fetch run.

    Scope is two-part:
    - `categories`: included in full, no filtering (software-dev-focused
      categories, low noise).
    - `keyword_filtered_categories`: included only when the abstract matches
      one of `keywords` (broad AI/ML/NLP categories, filtered down to the
      software-development-relevant slice).
    """

    categories: tuple[str, ...] = ("cs.SE", "cs.PL")
    keyword_filtered_categories: tuple[str, ...] = ("cs.AI", "cs.LG", "cs.CL")
    keywords: tuple[str, ...] = DEFAULT_SOFTWARE_DEV_KEYWORDS
    start_date: str = "2020-01-01"  # inclusive, YYYY-MM-DD
    page_size: int = 100
    max_results: int | None = None  # None = no cap
    request_delay_seconds: float = 3.0  # arXiv's requested courtesy delay
    max_retries: int = 5
    db_path: Path = Path("data/papers.db")


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
    stored here - see ingestion/clients/embeddings.py and
    ingestion/clients/pinecone_client.py.
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
