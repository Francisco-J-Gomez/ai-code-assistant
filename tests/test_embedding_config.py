from ingestion.embedding_config import EmbeddingConfig


def test_pinecone_index_name_falls_back_to_literal_default(monkeypatch):
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
    assert EmbeddingConfig().pinecone_index_name == "ai-code-assistant"


def test_pinecone_index_name_picks_up_env_override(monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX_NAME", "some-other-index")
    assert EmbeddingConfig().pinecone_index_name == "some-other-index"


def test_explicit_pinecone_index_name_overrides_env(monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX_NAME", "some-other-index")
    assert EmbeddingConfig(pinecone_index_name="explicit-index").pinecone_index_name == "explicit-index"
