from dataclasses import dataclass, field

from ingestion.embedding_config import EmbeddingConfig


@dataclass(frozen=True)
class AgentConfig:
    """Knobs for the retrieval agent. Secrets (OPENAI_API_KEY,
    PINECONE_API_KEY, TAVILY_API_KEY) are read from the environment at the
    point of use, not stored here - see agent/retrieval_tool.py and
    agent/graph.py.
    """

    llm_model: str = "gpt-5-mini"
    llm_temperature: float = 0.0  # factual/grounded answers over creative ones
    retrieval_top_k: int = 5
    web_search_max_results: int = 3

    # Composed rather than duplicated: the retrieval tool must embed
    # queries with the *same* model/dimension/index the ingestion side
    # used to build the corpus, or similarity search is meaningless.
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
