from ingestion.chunking.abstract_chunker import AbstractChunker
from ingestion.models import PaperMetadata


def make_paper(**overrides) -> PaperMetadata:
    defaults = dict(
        arxiv_id="2301.12345",
        version=2,
        title="A Great Paper",
        abstract="An abstract about testing.",
        authors=["Jane Doe", "John Smith"],
        categories=["cs.SE", "cs.PL"],
        primary_category="cs.SE",
        published="2023-01-15T00:00:00Z",
        updated="2023-01-15T00:00:00Z",
        pdf_url="http://arxiv.org/pdf/2301.12345v2",
    )
    defaults.update(overrides)
    return PaperMetadata(**defaults)


def test_abstract_chunker_produces_one_chunk_with_the_abstract_as_text():
    paper = make_paper()
    chunks = AbstractChunker().chunk(paper)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == paper.abstract
    assert chunk.chunk_index == 0
    assert chunk.chunk_type == "abstract"


def test_abstract_chunker_chunk_id_is_stable_and_paper_scoped():
    paper = make_paper(arxiv_id="2301.99999")
    chunk = AbstractChunker().chunk(paper)[0]

    assert chunk.chunk_id == "2301.99999:0"
    assert chunk.arxiv_id == "2301.99999"


def test_abstract_chunker_carries_citation_metadata():
    paper = make_paper()
    chunk = AbstractChunker().chunk(paper)[0]

    assert chunk.title == paper.title
    assert chunk.authors == paper.authors
    assert chunk.categories == paper.categories
    assert chunk.published == paper.published
    assert chunk.source_url == paper.pdf_url
