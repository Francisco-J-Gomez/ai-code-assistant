from agent.config import AgentConfig


def test_defaults():
    config = AgentConfig()
    assert config.llm_model == "gpt-5-mini"
    assert config.llm_temperature == 0.0
    assert config.retrieval_top_k == 5
    assert config.web_search_max_results == 3


def test_embedding_config_composed_with_ingestion_defaults():
    config = AgentConfig()
    assert config.embedding.embedding_model == "text-embedding-3-small"
    assert config.embedding.embedding_dimensions == 1536


def test_embedding_config_picks_up_pinecone_index_name_env_override(monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX_NAME", "some-other-index")
    config = AgentConfig()
    assert config.embedding.pinecone_index_name == "some-other-index"


def test_embedding_config_falls_back_without_env_override(monkeypatch):
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
    config = AgentConfig()
    assert config.embedding.pinecone_index_name == "ai-code-assistant"
