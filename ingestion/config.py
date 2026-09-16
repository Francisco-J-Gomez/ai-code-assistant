from dataclasses import dataclass
from pathlib import Path


# AI/ML/NLP papers are only included if their abstract mentions one of these
# software-development terms — cs.AI/cs.LG/cs.CL are too large and too
# generically ML-focused to ingest whole (see docs/data-ingestion-plan.md).
DEFAULT_SOFTWARE_DEV_KEYWORDS: tuple[str, ...] = (
    "code generation",
    "program synthesis",
    "program repair",
    "automated program repair",
    "code completion",
    "software engineering",
    "code review",
    "vulnerability detection",
    "bug detection",
    "code summarization",
    "test generation",
    "code search",
    "software testing",
    "code translation",
)


@dataclass(frozen=True)
class IngestionConfig:
    """Scope and behavior knobs for an arXiv metadata fetch run.

    Scope is two-part:
    - `categories`: included in full, no filtering (software-dev-focused
      categories, low noise).
    - `keyword_filtered_categories`: included only when the abstract matches
      one of `keywords` (broad AI/ML/NLP categories, filtered down to the
      software-development-relevant slice).
    """

    categories: tuple[str, ...] = ("cs.SE", "cs.PL")
    keyword_filtered_categories: tuple[str, ...] = ("cs.AI", "cs.LG", "cs.CL")
    keywords: tuple[str, ...] = DEFAULT_SOFTWARE_DEV_KEYWORDS
    start_date: str = "2020-01-01"  # inclusive, YYYY-MM-DD
    page_size: int = 100
    max_results: int | None = None  # None = no cap
    request_delay_seconds: float = 3.0  # arXiv's requested courtesy delay
    max_retries: int = 5
    db_path: Path = Path("data/papers.db")
