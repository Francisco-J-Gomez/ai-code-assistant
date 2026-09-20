import ingestion.embeddings as embeddings_module
from ingestion.embeddings import Embedder


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


def test_embed_returns_empty_list_for_no_texts(monkeypatch):
    monkeypatch.setattr(embeddings_module, "OpenAI", _FakeOpenAIClient)
    embedder = Embedder(model="text-embedding-3-small", api_key="fake")

    assert embedder.embed([]) == []


def test_embed_calls_api_once_and_returns_vectors_in_order(monkeypatch):
    monkeypatch.setattr(embeddings_module, "OpenAI", _FakeOpenAIClient)
    embedder = Embedder(model="text-embedding-3-small", api_key="fake")

    vectors = embedder.embed(["a", "bb", "ccc"])

    assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    assert embedder._client.embeddings.calls == [
        {"model": "text-embedding-3-small", "input": ["a", "bb", "ccc"]}
    ]
