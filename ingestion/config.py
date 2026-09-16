from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestionConfig:
    """Scope and behavior knobs for an arXiv metadata fetch run.

    Defaults match ingestion milestone 1 (docs/data-ingestion-plan.md):
    cs.SE only, no cap on results. Widen `categories` once the narrow
    scope is verified.
    """

    categories: tuple[str, ...] = ("cs.SE",)
    start_date: str = "2020-01-01"  # inclusive, YYYY-MM-DD
    page_size: int = 100
    max_results: int | None = None  # None = no cap
    request_delay_seconds: float = 3.0  # arXiv's requested courtesy delay
    max_retries: int = 5
    db_path: Path = Path("data/papers.db")
