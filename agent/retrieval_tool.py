from langchain_core.tools import tool

from agent.config import AgentConfig
from ingestion.embeddings import Embedder
from ingestion.pinecone_client import PineconeStore


def make_search_papers_tool(config: AgentConfig):
    """Build the search_papers tool bound to a specific AgentConfig.

    A factory rather than a bare @tool function: the tool needs a live
    Embedder/PineconeStore pair, and both must be built from the same
    embedding model/dimension/index the corpus was ingested with (see
    AgentConfig.embedding) or similarity search is meaningless.
    """
    embedder = Embedder(model=config.embedding.embedding_model)
    store = PineconeStore(
        index_name=config.embedding.pinecone_index_name,
        dimension=config.embedding.embedding_dimensions,
        cloud=config.embedding.pinecone_cloud,
        region=config.embedding.pinecone_region,
    )

    @tool
    def search_papers(query: str) -> list[dict]:
        """Search the curated arXiv corpus of AI / software-engineering
        research papers for content relevant to `query`.

        Use this for any question about AI or software-development
        research: findings, methods, techniques, results. Returns the
        most relevant papers as a list of {arxiv_id, title, authors,
        published, source_url, text, score}, most relevant first. Does
        NOT search the general web - use web_search for that.
        """
        vector = embedder.embed([query])[0]
        return store.query(vector, top_k=config.retrieval_top_k)

    return search_papers
