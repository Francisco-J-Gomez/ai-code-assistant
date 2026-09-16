import feedparser
import pytest

from ingestion.arxiv_client import _build_search_query, _parse_entry, _passes_keyword_filter
from ingestion.config import IngestionConfig
from ingestion.models import PaperMetadata

FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v2</id>
    <updated>2023-02-01T00:00:00Z</updated>
    <published>2023-01-15T00:00:00Z</published>
    <title>A Great Paper on Software Engineering</title>
    <summary>  This is the abstract.
    It spans multiple lines.  </summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
    <arxiv:primary_category term="cs.SE" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.SE" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <link title="pdf" href="http://arxiv.org/pdf/2301.12345v2" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


def test_parse_entry_extracts_expected_fields():
    feed = feedparser.parse(FEED_XML)
    paper = _parse_entry(feed.entries[0])

    assert paper is not None
    assert paper.arxiv_id == "2301.12345"
    assert paper.version == 2
    assert paper.title == "A Great Paper on Software Engineering"
    assert paper.abstract == "This is the abstract. It spans multiple lines."
    assert paper.authors == ["Jane Doe", "John Smith"]
    assert paper.categories == ["cs.SE", "cs.AI"]
    assert paper.primary_category == "cs.SE"
    assert paper.pdf_url == "http://arxiv.org/pdf/2301.12345v2"


def test_parse_entry_defaults_to_version_one_without_suffix():
    feed = feedparser.parse(FEED_XML.replace("2301.12345v2", "2301.12345"))
    paper = _parse_entry(feed.entries[0])

    assert paper is not None
    assert paper.arxiv_id == "2301.12345"
    assert paper.version == 1


def test_parse_entry_returns_none_for_non_arxiv_id():
    feed = feedparser.parse(
        FEED_XML.replace(
            "http://arxiv.org/abs/2301.12345v2", "http://example.com/not-arxiv"
        )
    )
    assert _parse_entry(feed.entries[0]) is None


def test_build_search_query_combines_categories_and_date():
    config = IngestionConfig(
        categories=("cs.SE", "cs.PL"),
        keyword_filtered_categories=(),
        keywords=(),
        start_date="2020-01-01",
    )
    query = _build_search_query(config)
    assert query == "((cat:cs.SE OR cat:cs.PL)) AND submittedDate:[20200101000000 TO 99991231235959]"


def test_build_search_query_adds_keyword_filtered_clause():
    config = IngestionConfig(
        categories=("cs.SE",),
        keyword_filtered_categories=("cs.AI", "cs.LG"),
        keywords=("code generation", "program repair"),
        start_date="2020-01-01",
    )
    query = _build_search_query(config)
    assert query == (
        '((cat:cs.SE) OR ((cat:cs.AI OR cat:cs.LG) AND '
        '(abs:"code generation" OR abs:"program repair"))) '
        "AND submittedDate:[20200101000000 TO 99991231235959]"
    )


def test_build_search_query_requires_at_least_one_scope():
    config = IngestionConfig(categories=(), keyword_filtered_categories=(), keywords=())
    with pytest.raises(ValueError):
        _build_search_query(config)


def make_paper(**overrides) -> PaperMetadata:
    defaults = dict(
        arxiv_id="2301.12345",
        version=1,
        title="A Paper",
        abstract="An abstract about something.",
        authors=["A"],
        categories=["cs.AI"],
        primary_category="cs.AI",
        published="2023-01-01T00:00:00Z",
        updated="2023-01-01T00:00:00Z",
        pdf_url="http://arxiv.org/pdf/2301.12345v1",
    )
    defaults.update(overrides)
    return PaperMetadata(**defaults)


def test_keyword_filter_always_keeps_papers_in_full_scope_categories():
    config = IngestionConfig(
        categories=("cs.SE",), keyword_filtered_categories=("cs.AI",), keywords=("code generation",)
    )
    paper = make_paper(primary_category="cs.SE", categories=["cs.SE"], abstract="Nothing relevant here.")
    assert _passes_keyword_filter(paper, config) is True


def test_keyword_filter_rejects_keyword_category_paper_without_real_phrase_match():
    # Reproduces arXiv's loose abs: search: the paper only surfaced because
    # arXiv matched the words individually, not as the literal phrase.
    config = IngestionConfig(
        categories=("cs.SE",), keyword_filtered_categories=("cs.AI",), keywords=("code generation",)
    )
    paper = make_paper(
        primary_category="cs.AI",
        categories=["cs.AI"],
        abstract="We propose a new generation of models trained on source code.",
    )
    assert _passes_keyword_filter(paper, config) is False


def test_keyword_filter_accepts_keyword_category_paper_with_real_phrase_match():
    config = IngestionConfig(
        categories=("cs.SE",), keyword_filtered_categories=("cs.AI",), keywords=("code generation",)
    )
    paper = make_paper(
        primary_category="cs.AI",
        categories=["cs.AI"],
        abstract="We study code generation with large language models.",
    )
    assert _passes_keyword_filter(paper, config) is True


def test_keyword_filter_rejects_paper_outside_all_configured_categories():
    config = IngestionConfig(
        categories=("cs.SE",), keyword_filtered_categories=("cs.AI",), keywords=("code generation",)
    )
    paper = make_paper(primary_category="cs.CV", categories=["cs.CV"], abstract="Code generation for images.")
    assert _passes_keyword_filter(paper, config) is False
