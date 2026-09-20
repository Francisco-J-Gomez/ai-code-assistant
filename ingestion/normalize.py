import logging
import re
import unicodedata
from dataclasses import replace
from datetime import datetime
from typing import Optional

from ingestion.config import IngestionConfig
from ingestion.models import PaperMetadata

logger = logging.getLogger(__name__)

_WITHDRAWN_MARKERS = ("this paper has been withdrawn", "paper has been withdrawn")

# Not a full LaTeX parser - just strips the handful of constructs that show
# up routinely in arXiv abstracts (citation keys, common text-formatting
# commands, math delimiters, escaped special characters).
_LATEX_CITE_RE = re.compile(r"\\(?:cite|citep|citet)[a-zA-Z]*\{[^{}]*\}")
_LATEX_WRAP_RE = re.compile(r"\\(?:emph|textbf|textit|text|mathrm)\{([^{}]*)\}")
_MATH_DELIM_RE = re.compile(r"\$([^$]*)\$")
_LATEX_ESCAPED_CHAR_RE = re.compile(r"\\([%&_#$])")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_paper(paper: PaperMetadata, config: IngestionConfig) -> Optional[PaperMetadata]:
    """Clean and validate a raw fetched paper; return None to drop it.

    Checks, in order: category/date scope (defense in depth against a stale
    raw store or a changed config), withdrawn-paper removal, text cleaning,
    then required-field validation on the cleaned result. Every drop is
    logged with its reason so ingestion runs are auditable.
    """
    if not _in_configured_scope(paper, config):
        logger.debug("Dropping %s: outside configured category scope", paper.arxiv_id)
        return None

    if not _published_on_or_after(paper.published, config.start_date):
        logger.debug("Dropping %s: published before %s", paper.arxiv_id, config.start_date)
        return None

    if _is_withdrawn(paper):
        logger.info("Dropping %s: withdrawn", paper.arxiv_id)
        return None

    title = _clean_text(paper.title)
    abstract = _clean_text(paper.abstract)
    authors = [name for name in (_clean_text(a) for a in paper.authors) if name]

    if not title or not abstract or not authors or not paper.pdf_url:
        logger.warning("Dropping %s: missing required field after cleaning", paper.arxiv_id)
        return None

    return replace(paper, title=title, abstract=abstract, authors=authors)


def _in_configured_scope(paper: PaperMetadata, config: IngestionConfig) -> bool:
    allowed = set(config.categories) | set(config.keyword_filtered_categories)
    paper_categories = set(paper.categories) | {paper.primary_category}
    return bool(paper_categories & allowed)


def _published_on_or_after(published: str, start_date: str) -> bool:
    try:
        published_date = datetime.strptime(published[:10], "%Y-%m-%d")
    except ValueError:
        return False
    return published_date >= datetime.strptime(start_date, "%Y-%m-%d")


def _is_withdrawn(paper: PaperMetadata) -> bool:
    haystack = f"{paper.title} {paper.abstract}".lower()
    return any(marker in haystack for marker in _WITHDRAWN_MARKERS)


def _clean_text(text: str) -> str:
    text = _LATEX_CITE_RE.sub("", text)
    text = _LATEX_WRAP_RE.sub(r"\1", text)
    text = _MATH_DELIM_RE.sub(r"\1", text)
    text = _LATEX_ESCAPED_CHAR_RE.sub(r"\1", text)
    text = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
