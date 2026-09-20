from pathlib import Path

from ingestion.embedding_cache import EmbeddingCacheStore


def test_unknown_chunk_is_not_up_to_date(tmp_path: Path):
    with EmbeddingCacheStore(tmp_path / "cache.db") as cache:
        assert cache.is_up_to_date("2301.12345:0", "some text") is False


def test_mark_embedded_then_same_text_is_up_to_date(tmp_path: Path):
    with EmbeddingCacheStore(tmp_path / "cache.db") as cache:
        cache.mark_embedded("2301.12345:0", "some text")
        assert cache.is_up_to_date("2301.12345:0", "some text") is True
        assert cache.count() == 1


def test_changed_text_is_not_up_to_date(tmp_path: Path):
    with EmbeddingCacheStore(tmp_path / "cache.db") as cache:
        cache.mark_embedded("2301.12345:0", "original text")
        assert cache.is_up_to_date("2301.12345:0", "edited text") is False


def test_mark_embedded_many_records_all_and_commits_once(tmp_path: Path):
    with EmbeddingCacheStore(tmp_path / "cache.db") as cache:
        cache.mark_embedded_many(
            [("2301.00001:0", "text a"), ("2301.00002:0", "text b")]
        )
        assert cache.count() == 2
        assert cache.is_up_to_date("2301.00001:0", "text a") is True
        assert cache.is_up_to_date("2301.00002:0", "text b") is True


def test_mark_embedded_overwrites_existing_entry(tmp_path: Path):
    with EmbeddingCacheStore(tmp_path / "cache.db") as cache:
        cache.mark_embedded("2301.12345:0", "v1")
        cache.mark_embedded("2301.12345:0", "v2")

        assert cache.count() == 1
        assert cache.is_up_to_date("2301.12345:0", "v1") is False
        assert cache.is_up_to_date("2301.12345:0", "v2") is True
