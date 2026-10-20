from pathlib import Path

import ingestion.clients.embeddings as embeddings_module
from ingestion.clients.embeddings import Embedder
from ingestion.config import EmbeddingConfig
from ingestion.storage import EmbeddingCacheStore


class _FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeEmbeddingsAPI:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append({"model": model, "input": input})
        return _FakeResponse([_FakeEmbeddingItem([float(len(text))] * 3) for text in input])


class _FakeOpenAIClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.embeddings = _FakeEmbeddingsAPI()


def test_embedder_embed_returns_empty_list_for_no_texts(monkeypatch):
    monkeypatch.setattr(embeddings_module, "OpenAI", _FakeOpenAIClient)
    embedder = Embedder(model="text-embedding-3-small", api_key="fake")

    assert embedder.embed([]) == []


def test_embedder_embed_calls_api_once_and_returns_vectors_in_order(monkeypatch):
    monkeypatch.setattr(embeddings_module, "OpenAI", _FakeOpenAIClient)
    embedder = Embedder(model="text-embedding-3-small", api_key="fake")

    vectors = embedder.embed(["a", "bb", "ccc"])

    assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    assert embedder._client.embeddings.calls == [
        {"model": "text-embedding-3-small", "input": ["a", "bb", "ccc"]}
    ]


def test_cache_unknown_chunk_is_not_up_to_date(tmp_path: Path):
    with EmbeddingCacheStore(tmp_path / "cache.db") as cache:
        assert cache.is_up_to_date("2301.12345:0", "some text") is False


def test_cache_mark_embedded_then_same_text_is_up_to_date(tmp_path: Path):
    with EmbeddingCacheStore(tmp_path / "cache.db") as cache:
        cache.mark_embedded("2301.12345:0", "some text")
        assert cache.is_up_to_date("2301.12345:0", "some text") is True
        assert cache.count() == 1


def test_cache_changed_text_is_not_up_to_date(tmp_path: Path):
    with EmbeddingCacheStore(tmp_path / "cache.db") as cache:
        cache.mark_embedded("2301.12345:0", "original text")
        assert cache.is_up_to_date("2301.12345:0", "edited text") is False


def test_cache_mark_embedded_many_records_all_and_commits_once(tmp_path: Path):
    with EmbeddingCacheStore(tmp_path / "cache.db") as cache:
        cache.mark_embedded_many(
            [("2301.00001:0", "text a"), ("2301.00002:0", "text b")]
        )
        assert cache.count() == 2
        assert cache.is_up_to_date("2301.00001:0", "text a") is True
        assert cache.is_up_to_date("2301.00002:0", "text b") is True


def test_cache_mark_embedded_overwrites_existing_entry(tmp_path: Path):
    with EmbeddingCacheStore(tmp_path / "cache.db") as cache:
        cache.mark_embedded("2301.12345:0", "v1")
        cache.mark_embedded("2301.12345:0", "v2")

        assert cache.count() == 1
        assert cache.is_up_to_date("2301.12345:0", "v1") is False
        assert cache.is_up_to_date("2301.12345:0", "v2") is True


def test_config_pinecone_index_name_falls_back_to_literal_default(monkeypatch):
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
    assert EmbeddingConfig().pinecone_index_name == "ai-code-assistant"


def test_config_pinecone_index_name_picks_up_env_override(monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX_NAME", "some-other-index")
    assert EmbeddingConfig().pinecone_index_name == "some-other-index"


def test_config_explicit_pinecone_index_name_overrides_env(monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX_NAME", "some-other-index")
    assert EmbeddingConfig(pinecone_index_name="explicit-index").pinecone_index_name == "explicit-index"
