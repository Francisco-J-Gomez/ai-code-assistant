from datetime import datetime

import feedparser
import pytest

import ingestion.clients.arxiv_client as arxiv_client
from ingestion.clients.arxiv_client import _fetch_scope_entries, _parse_entry, _passes_keyword_filter, _scope_queries
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


def test_scope_queries_one_clause_per_full_scope_category():
    config = IngestionConfig(
        categories=("cs.SE", "cs.PL"), keyword_filtered_categories=(), keywords=(),
    )
    assert _scope_queries(config) == ["cat:cs.SE", "cat:cs.PL"]


def test_scope_queries_adds_keyword_filtered_clause():
    config = IngestionConfig(
        categories=("cs.SE",),
        keyword_filtered_categories=("cs.AI", "cs.LG"),
        keywords=("code generation", "program repair"),
    )
    assert _scope_queries(config) == [
        "cat:cs.SE",
        '(cat:cs.AI OR cat:cs.LG) AND (abs:"code generation" OR abs:"program repair")',
    ]


def test_scope_queries_requires_at_least_one_scope():
    config = IngestionConfig(categories=(), keyword_filtered_categories=(), keywords=())
    with pytest.raises(ValueError):
        _scope_queries(config)


class _FakeFeed:
    def __init__(self, entries, total):
        self.entries = entries
        self.feed = {"opensearch_totalresults": str(total)}


def test_fetch_scope_entries_paginates_without_bisecting_small_window(monkeypatch):
    monkeypatch.setattr(arxiv_client.time, "sleep", lambda *_: None)

    items = list(range(7))  # fewer than ARXIV_MAX_START and than one page

    def fake_get_with_retries(params, max_retries):
        page = items[params["start"]: params["start"] + params["max_results"]]
        return _FakeFeed(page, total=len(items))

    monkeypatch.setattr(arxiv_client, "_get_with_retries", fake_get_with_retries)

    config = IngestionConfig(page_size=3, max_retries=1)
    results = list(_fetch_scope_entries("cat:cs.SE", datetime(2020, 1, 1), datetime(2020, 1, 2), config))

    assert results == items


def test_fetch_scope_entries_bisects_oversized_window_without_duplicates(monkeypatch):
    # ARXIV_MAX_START patched small so the test doesn't need thousands of
    # fake entries to exercise the bisection path.
    monkeypatch.setattr(arxiv_client, "ARXIV_MAX_START", 10)
    monkeypatch.setattr(arxiv_client.time, "sleep", lambda *_: None)

    date_from = datetime(2020, 1, 1)
    date_to = datetime(2020, 1, 2)
    total_items = 25
    span = (date_to - date_from) / total_items
    all_items = [(i, date_from + span * i) for i in range(total_items)]

    requested_starts = []

    def fake_get_with_retries(params, max_retries):
        window = params["search_query"].split("submittedDate:[")[1].rstrip("]")
        from_str, to_str = (s.strip() for s in window.split(" TO "))
        win_from = datetime.strptime(from_str, "%Y%m%d%H%M%S")
        win_to = datetime.strptime(to_str, "%Y%m%d%H%M%S")
        matching = [i for i, ts in all_items if win_from <= ts < win_to]

        requested_starts.append(params["start"])
        page = matching[params["start"]: params["start"] + params["max_results"]]
        return _FakeFeed(page, total=len(matching))

    monkeypatch.setattr(arxiv_client, "_get_with_retries", fake_get_with_retries)

    config = IngestionConfig(page_size=5, max_retries=1)
    results = list(_fetch_scope_entries("cat:cs.SE", date_from, date_to, config))

    assert sorted(results) == list(range(total_items))  # every item exactly once
    assert max(requested_starts) < 10  # never paginates past the (patched) wall


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
