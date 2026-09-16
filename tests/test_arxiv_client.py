import feedparser

from ingestion.arxiv_client import _build_search_query, _parse_entry

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
    query = _build_search_query(("cs.SE", "cs.AI"), "2020-01-01")
    assert query == "(cat:cs.SE OR cat:cs.AI) AND submittedDate:[20200101000000 TO 99991231235959]"
