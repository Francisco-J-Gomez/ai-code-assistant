from dataclasses import dataclass
from typing import Protocol

from ingestion.models import PaperMetadata


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # stable vector-store id, e.g. "{arxiv_id}:{chunk_index}"
    arxiv_id: str
    chunk_index: int
    chunk_type: str  # "abstract" for the MVP chunker
    text: str  # what actually gets embedded
    title: str
    authors: list[str]
    categories: list[str]
    published: str
    source_url: str


class ChunkerStrategy(Protocol):
    """Splits a normalized paper into embeddable chunks.

    One implementation (AbstractChunker) for the MVP; full-text chunkers
    (fixed-window, section-aware) from the planned benchmarking work
    implement the same interface and plug in without touching
    embed/upsert (docs/data-ingestion-plan.md step 6).
    """

    def chunk(self, paper: PaperMetadata) -> list[Chunk]: ...
