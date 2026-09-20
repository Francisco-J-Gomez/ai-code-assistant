import pytest

import ingestion.pinecone_client as pinecone_client_module
from ingestion.chunking.base import Chunk
from ingestion.pinecone_client import PineconeStore


def make_chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="2301.12345:0",
        arxiv_id="2301.12345",
        chunk_index=0,
        chunk_type="abstract",
        text="An abstract.",
        title="A Paper",
        authors=["Jane Doe"],
        categories=["cs.SE"],
        published="2023-01-01T00:00:00Z",
        source_url="http://arxiv.org/pdf/2301.12345v1",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


class _FakeIndexHandle:
    def __init__(self, name):
        self.name = name
        self.upserted_batches = []

    def upsert(self, vectors):
        self.upserted_batches.append(vectors)

    def describe_index_stats(self):
        total = sum(len(batch) for batch in self.upserted_batches)
        return type("Stats", (), {"total_vector_count": total})()


class _FakeIndexList:
    def __init__(self, names):
        self._names = names

    def names(self):
        return self._names


class _FakePineconeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.existing_indexes = []
        self.created_indexes = []
        self._index_handles = {}

    def list_indexes(self):
        return _FakeIndexList(self.existing_indexes)

    def create_index(self, name, dimension, metric, spec):
        self.created_indexes.append({"name": name, "dimension": dimension, "metric": metric, "spec": spec})
        self.existing_indexes.append(name)

    def Index(self, name):
        return self._index_handles.setdefault(name, _FakeIndexHandle(name))


def _make_store(monkeypatch, fake_client=None) -> tuple[PineconeStore, _FakePineconeClient]:
    fake_client = fake_client or _FakePineconeClient()
    monkeypatch.setattr(pinecone_client_module, "Pinecone", lambda **kwargs: fake_client)
    store = PineconeStore(index_name="test-index", dimension=3, cloud="aws", region="us-east-1", api_key="fake")
    return store, fake_client


def test_creates_index_when_missing(monkeypatch):
    store, fake_client = _make_store(monkeypatch)

    assert len(fake_client.created_indexes) == 1
    created = fake_client.created_indexes[0]
    assert created["name"] == "test-index"
    assert created["dimension"] == 3
    assert created["metric"] == "cosine"


def test_skips_creation_when_index_already_exists(monkeypatch):
    fake_client = _FakePineconeClient()
    fake_client.existing_indexes = ["test-index"]
    store, fake_client = _make_store(monkeypatch, fake_client)

    assert fake_client.created_indexes == []


def test_upsert_sends_vectors_with_citation_metadata(monkeypatch):
    store, fake_client = _make_store(monkeypatch)
    chunk = make_chunk()

    store.upsert([chunk], [[0.1, 0.2, 0.3]])

    batches = fake_client.Index("test-index").upserted_batches
    assert len(batches) == 1
    vector_id, vector, metadata = batches[0][0]
    assert vector_id == "2301.12345:0"
    assert vector == [0.1, 0.2, 0.3]
    assert metadata["arxiv_id"] == "2301.12345"
    assert metadata["title"] == "A Paper"
    assert metadata["text"] == "An abstract."


def test_upsert_raises_on_length_mismatch(monkeypatch):
    store, _ = _make_store(monkeypatch)

    with pytest.raises(ValueError):
        store.upsert([make_chunk()], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def test_upsert_is_noop_for_empty_input(monkeypatch):
    store, fake_client = _make_store(monkeypatch)

    store.upsert([], [])

    assert fake_client.Index("test-index").upserted_batches == []


def test_count_reflects_upserted_vectors(monkeypatch):
    store, _ = _make_store(monkeypatch)
    store.upsert([make_chunk()], [[0.1, 0.2, 0.3]])

    assert store.count() == 1
