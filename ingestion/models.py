from dataclasses import dataclass


@dataclass(frozen=True)
class PaperMetadata:
    arxiv_id: str  # base id, no version suffix, e.g. "2301.12345"
    version: int
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str  # ISO 8601 timestamp, as returned by arXiv
    updated: str
    pdf_url: str
