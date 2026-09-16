import logging
import time
from datetime import datetime
from typing import Iterator, Optional

import feedparser
import requests

from ingestion.config import IngestionConfig
from ingestion.models import PaperMetadata

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_ABS_PREFIX = "http://arxiv.org/abs/"


class ArxivClientError(Exception):
    """Raised when the arXiv API cannot be queried successfully."""


def fetch_papers(config: IngestionConfig) -> Iterator[PaperMetadata]:
    """Yield PaperMetadata for every paper matching the configured scope.

    Paginates through the arXiv API, sleeping `config.request_delay_seconds`
    between requests per arXiv's courtesy rate-limit guidance, and retrying
    transient failures with exponential backoff.
    """
    query = _build_search_query(config.categories, config.start_date)
    start = 0
    fetched = 0

    while config.max_results is None or fetched < config.max_results:
        page_size = config.page_size
        if config.max_results is not None:
            page_size = min(page_size, config.max_results - fetched)

        params = {
            "search_query": query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }

        feed = _get_with_retries(params, config.max_retries)
        entries = feed.entries
        if not entries:
            break

        for entry in entries:
            paper = _parse_entry(entry)
            if paper is None:
                logger.warning("Skipping entry with unexpected id format: %s", entry.get("id"))
                continue
            yield paper

        fetched += len(entries)
        start += len(entries)

        if len(entries) < page_size:
            break  # reached the last page

        time.sleep(config.request_delay_seconds)


def _build_search_query(categories: tuple[str, ...], start_date: str) -> str:
    category_clause = " OR ".join(f"cat:{c}" for c in categories)
    start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d%H%M%S")
    return f"({category_clause}) AND submittedDate:[{start} TO 99991231235959]"


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
