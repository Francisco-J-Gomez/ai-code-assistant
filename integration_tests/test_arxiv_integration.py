"""Integration tests against the real arXiv API.

Skipped by default (plan step 10: "marked slow/opt-in, not run by
default") since they hit a live external service and take longer than the
mocked unit tests. Run explicitly with:

    RUN_ARXIV_INTEGRATION_TESTS=1 uv run pytest integration_tests/test_arxiv_integration.py -v
"""
import itertools
import os
from datetime import datetime, timedelta

import feedparser
import pytest
import requests

from ingestion.arxiv_client import _fetch_scope_entries, _parse_entry, fetch_papers
from ingestion.config import IngestionConfig
from ingestion.normalize import normalize_paper

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ARXIV_INTEGRATION_TESTS") != "1",
    reason="Hits the real arXiv API; set RUN_ARXIV_INTEGRATION_TESTS=1 to run.",
)


def test_fetch_papers_returns_real_papers_in_scope():
    config = IngestionConfig(
        categories=("cs.SE",), keyword_filtered_categories=(), keywords=(), max_results=5,
    )
    papers = list(fetch_papers(config))

    assert len(papers) == 5
    for paper in papers:
        assert paper.arxiv_id
        assert paper.title
        assert paper.abstract
        assert "cs.SE" in paper.categories or paper.primary_category == "cs.SE"


def test_normalize_drops_a_real_withdrawn_paper():
    # 0705.1442 "Does P=NP?" is a real, permanently withdrawn arXiv paper -
    # a stable fixture for exercising the withdrawn-paper filter end-to-end
    # (arXiv keeps withdrawn papers' metadata, replacing the abstract with a
    # withdrawal notice, so this id won't change out from under the test).
    resp = requests.get(
        "http://export.arxiv.org/api/query", params={"id_list": "0705.1442"}, timeout=30
    )
    feed = feedparser.parse(resp.content)
    paper = _parse_entry(feed.entries[0])
    assert paper is not None
    assert "withdrawn" in paper.abstract.lower()

    config = IngestionConfig(categories=paper.categories, start_date="2007-01-01")
    assert normalize_paper(paper, config) is None


def test_deep_pagination_bisection_handles_oversized_category():
    # cs.SE has 20k+ papers since 2020, well past ARXIV_MAX_START, so this
    # exercises the real bisection path (see arxiv_client.ARXIV_MAX_START)
    # against the live API - just enough entries to prove it works, not the
    # whole category.
    config = IngestionConfig(page_size=100, max_retries=5, request_delay_seconds=3.0)
    date_from = datetime(2020, 1, 1)
    date_to = datetime.now() + timedelta(days=1)

    entries = list(itertools.islice(_fetch_scope_entries("cat:cs.SE", date_from, date_to, config), 5))

    assert len(entries) == 5
