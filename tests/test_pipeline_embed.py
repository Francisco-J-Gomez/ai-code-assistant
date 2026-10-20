from pathlib import Path

import ingestion.pipeline as pipeline_module
from ingestion.config import EmbeddingConfig
from ingestion.models import PaperMetadata
from ingestion.pipeline import run_embed_stage
from ingestion.storage import NormalizedPaperStore


def make_paper(**overrides) -> PaperMetadata:
    defaults = dict(
        arxiv_id="2301.12345",
        version=1,
        title="A Paper",
        abstract="An abstract.",
        authors=["Jane Doe"],
        categories=["cs.SE"],
        primary_category="cs.SE",
        published="2023-01-01T00:00:00Z",
        updated="2023-01-01T00:00:00Z",
        pdf_url="http://arxiv.org/pdf/2301.12345v1",
    )
    defaults.update(overrides)
    return PaperMetadata(**defaults)


class _FakeEmbedder:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text))] for text in texts]


class _FakePineconeStore:
    def __init__(self, index_name, dimension, cloud, region):
        self.upserted = []

    def upsert(self, chunks, vectors):
        self.upserted.extend(zip(chunks, vectors))

    def count(self):
        return len(self.upserted)


def _patch_embed_dependencies(monkeypatch):
    monkeypatch.setattr(pipeline_module, "Embedder", _FakeEmbedder)
    monkeypatch.setattr(pipeline_module, "PineconeStore", _FakePineconeStore)


def test_run_embed_stage_respects_limit(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "papers.db"
    with NormalizedPaperStore(db_path) as store:
        for i in range(5):
            store.upsert(make_paper(arxiv_id=f"2301.0000{i}"))

    _patch_embed_dependencies(monkeypatch)
    config = EmbeddingConfig(db_path=db_path, embed_batch_size=10, limit=2)

    embedded, skipped = run_embed_stage(config)

    assert embedded == 2
    assert skipped == 0


def test_run_embed_stage_embeds_all_new_papers(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "papers.db"
    with NormalizedPaperStore(db_path) as store:
        store.upsert(make_paper(arxiv_id="2301.00001"))
        store.upsert(make_paper(arxiv_id="2301.00002"))

    _patch_embed_dependencies(monkeypatch)
    config = EmbeddingConfig(db_path=db_path, embed_batch_size=10)

    embedded, skipped = run_embed_stage(config)

    assert embedded == 2
    assert skipped == 0


def test_run_embed_stage_skips_unchanged_chunks_on_second_run(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "papers.db"
    with NormalizedPaperStore(db_path) as store:
        store.upsert(make_paper(arxiv_id="2301.00001"))

    _patch_embed_dependencies(monkeypatch)
    config = EmbeddingConfig(db_path=db_path, embed_batch_size=10)

    run_embed_stage(config)
    embedded, skipped = run_embed_stage(config)

    assert embedded == 0
    assert skipped == 1


def test_run_embed_stage_reembeds_when_abstract_changes(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "papers.db"
    with NormalizedPaperStore(db_path) as store:
        store.upsert(make_paper(arxiv_id="2301.00001", abstract="original"))

    _patch_embed_dependencies(monkeypatch)
    config = EmbeddingConfig(db_path=db_path, embed_batch_size=10)
    run_embed_stage(config)

    with NormalizedPaperStore(db_path) as store:
        store.upsert(make_paper(arxiv_id="2301.00001", abstract="edited"))

    embedded, skipped = run_embed_stage(config)

    assert embedded == 1
    assert skipped == 0


def test_run_embed_stage_batches_embed_calls(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "papers.db"
    with NormalizedPaperStore(db_path) as store:
        for i in range(5):
            store.upsert(make_paper(arxiv_id=f"2301.0000{i}"))

    created_embedders: list[_FakeEmbedder] = []

    def _embedder_factory(model):
        embedder = _FakeEmbedder(model)
        created_embedders.append(embedder)
        return embedder

    monkeypatch.setattr(pipeline_module, "Embedder", _embedder_factory)
    monkeypatch.setattr(pipeline_module, "PineconeStore", _FakePineconeStore)
    config = EmbeddingConfig(db_path=db_path, embed_batch_size=2)

    embedded, _ = run_embed_stage(config)

    assert embedded == 5
    # batch_size=2 over 5 chunks -> batches of 2, 2, 1
    assert [len(call) for call in created_embedders[0].calls] == [2, 2, 1]
