import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

import feedparser
import requests

from ingestion.config import IngestionConfig
from ingestion.models import PaperMetadata

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_ABS_PREFIX = "http://arxiv.org/abs/"

# arXiv's legacy API returns a 500 for ANY query once `start` reaches ~10000,
# regardless of query complexity - confirmed against a trivial single-category
# query. It's a hard wall, not something retries can work around. A scope
# query whose total result count exceeds this has its date range bisected
# up front (see _fetch_scope_entries) instead of paginating into the wall.
# Margin below the observed 10000 failure point in case it's not exact under
# load.
ARXIV_MAX_START = 9000


class ArxivClientError(Exception):
    """Raised when the arXiv API cannot be queried successfully."""


def fetch_papers(config: IngestionConfig) -> Iterator[PaperMetadata]:
    """Yield PaperMetadata for every paper matching the configured scope.

    Each scope clause (see `_scope_queries`) is fetched independently so
    that no single query's result count risks the deep-pagination 500
    described above. Sleeps `config.request_delay_seconds` between requests
    per arXiv's courtesy rate-limit guidance, and retries transient failures
    with exponential backoff.
    """
    date_from = datetime.strptime(config.start_date, "%Y-%m-%d")
    date_to = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)  # open-ended upper bound

    fetched = 0
    for scope_query in _scope_queries(config):
        for entry in _fetch_scope_entries(scope_query, date_from, date_to, config):
            if config.max_results is not None and fetched >= config.max_results:
                return

            paper = _parse_entry(entry)
            if paper is None:
                logger.warning("Skipping entry with unexpected id format: %s", entry.get("id"))
                continue
            if not _passes_keyword_filter(paper, config):
                continue

            yield paper
            fetched += 1


def _scope_queries(config: IngestionConfig) -> list[str]:
    """Build one search_query clause per full-scope category, plus one for
    the keyword-filtered categories - kept separate (rather than OR'd into a
    single query) so each stays under arXiv's deep-pagination wall on its
    own. A cross-listed paper matching more than one clause is simply
    fetched more than once; the raw store's upsert-by-arxiv_id absorbs that.
    """
    queries = [f"cat:{category}" for category in config.categories]

    if config.keyword_filtered_categories and config.keywords:
        cat_clause = "(" + " OR ".join(f"cat:{c}" for c in config.keyword_filtered_categories) + ")"
        keyword_clause = "(" + " OR ".join(f'abs:"{k}"' for k in config.keywords) + ")"
        queries.append(f"{cat_clause} AND {keyword_clause}")

    if not queries:
        raise ValueError("IngestionConfig must define at least one category scope")

    return queries


def _fetch_scope_entries(scope_query: str, date_from: datetime, date_to: datetime, config: IngestionConfig):
    """Paginate one scope query across [date_from, date_to).

    Probes the window's total result count first (a cheap max_results=1
    request) and bisects the date range up front when that total would need
    start >= ARXIV_MAX_START to page through - rather than paginating partway
    and bisecting reactively, which would re-fetch (and re-yield) whatever
    was already pulled from the un-split window once it split.
    """
    if date_from >= date_to:
        return

    date_query = f"({scope_query}) AND submittedDate:[{_format_date(date_from)} TO {_format_date(date_to)}]"

    probe = _request(date_query, start=0, max_results=1, config=config)
    total = _total_results(probe)
    if total == 0:
        return

    if total > ARXIV_MAX_START:
        midpoint = date_from + (date_to - date_from) / 2
        if midpoint <= date_from:
            logger.warning(
                "Cannot bisect further for %r in [%s, %s) (%d results) - some may be missed",
                scope_query,
                date_from,
                date_to,
                total,
            )
            return
        yield from _fetch_scope_entries(scope_query, date_from, midpoint, config)
        yield from _fetch_scope_entries(scope_query, midpoint, date_to, config)
        return

    start = 0
    while True:
        feed = _request(date_query, start=start, max_results=config.page_size, config=config)
        entries = feed.entries
        if not entries:
            return

        yield from entries

        start += len(entries)
        if len(entries) < config.page_size:
            return  # reached the last page


def _request(date_query: str, start: int, max_results: int, config: IngestionConfig):
    params = {
        "search_query": date_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    feed = _get_with_retries(params, config.max_retries)
    time.sleep(config.request_delay_seconds)
    return feed


def _total_results(feed) -> int:
    try:
        return int(feed.feed.get("opensearch_totalresults", 0))
    except (TypeError, ValueError):
        return 0


def _format_date(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _passes_keyword_filter(paper: PaperMetadata, config: IngestionConfig) -> bool:
    """Re-check keyword-gated papers client-side.

    arXiv's `abs:"..."` field search does not enforce exact phrase matching
    (a query for a nonsense reversed phrase still returns hundreds of hits),
    so papers reached only through `keyword_filtered_categories` are
    re-verified here with a real substring check on title+abstract. Papers
    in `config.categories` are always kept regardless of keywords.
    """
    paper_categories = set(paper.categories) | {paper.primary_category}

    if paper_categories & set(config.categories):
        return True

    if paper_categories & set(config.keyword_filtered_categories):
        haystack = f"{paper.title} {paper.abstract}".lower()
        return any(keyword.lower() in haystack for keyword in config.keywords)

    return False


def _parse_entry(entry) -> Optional[PaperMetadata]:
    raw_id = entry.get("id", "")
    if not raw_id.startswith(ARXIV_ABS_PREFIX):
        return None

    id_with_version = raw_id[len(ARXIV_ABS_PREFIX):]
    if "v" in id_with_version:
        base_id, _, version_str = id_with_version.rpartition("v")
        version = int(version_str)
    else:
        base_id, version = id_with_version, 1

    categories = [tag["term"] for tag in entry.get("tags", [])]
    primary_category = entry.get("arxiv_primary_category", {}).get(
        "term", categories[0] if categories else ""
    )
    pdf_url = next(
        (link["href"] for link in entry.get("links", []) if link.get("title") == "pdf"),
        "",
    )

    return PaperMetadata(
        arxiv_id=base_id,
        version=version,
        title=" ".join(entry.get("title", "").split()),
        abstract=" ".join(entry.get("summary", "").split()),
        authors=[author["name"] for author in entry.get("authors", [])],
        categories=categories,
        primary_category=primary_category,
        published=entry.get("published", ""),
        updated=entry.get("updated", ""),
        pdf_url=pdf_url,
    )


def _get_with_retries(params: dict, max_retries: int):
    delay = 3.0
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(ARXIV_API_URL, params=params, timeout=30)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            if feed.bozo and not feed.entries:
                raise ArxivClientError(f"Malformed feed: {feed.bozo_exception}")
            return feed
        except (requests.RequestException, ArxivClientError) as exc:
            last_error = exc
            logger.warning("arXiv request failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2

    raise ArxivClientError(f"arXiv API request failed after {max_retries} attempts") from last_error
