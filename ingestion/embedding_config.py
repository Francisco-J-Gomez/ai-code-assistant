from dataclasses import dataclass
from pathlib import Path


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
    pinecone_index_name: str = "ai-code-assistant"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    max_retries: int = 5
    db_path: Path = Path("data/papers.db")  # same SQLite file as ingestion; adds embedded_chunks table
