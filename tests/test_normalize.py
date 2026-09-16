from ingestion.config import IngestionConfig
from ingestion.models import PaperMetadata
from ingestion.normalize import normalize_paper

CONFIG = IngestionConfig(
    categories=("cs.SE",),
    keyword_filtered_categories=("cs.AI",),
    keywords=("code generation",),
    start_date="2020-01-01",
)


def make_paper(**overrides) -> PaperMetadata:
    defaults = dict(
        arxiv_id="2301.12345",
        version=1,
        title="A Great Paper",
        abstract="An abstract about testing.",
        authors=["Jane Doe", "John Smith"],
        categories=["cs.SE"],
        primary_category="cs.SE",
        published="2023-01-15T00:00:00Z",
        updated="2023-01-15T00:00:00Z",
        pdf_url="http://arxiv.org/pdf/2301.12345v1",
    )
    defaults.update(overrides)
    return PaperMetadata(**defaults)


def test_keeps_valid_paper_and_returns_cleaned_copy():
    paper = make_paper()
    result = normalize_paper(paper, CONFIG)
    assert result is not None
    assert result.arxiv_id == paper.arxiv_id
    assert result.title == "A Great Paper"


def test_collapses_whitespace_in_title_and_abstract():
    paper = make_paper(title="  A   Great\nPaper  ", abstract="Multiple   spaces.\nAnd newlines.")
    result = normalize_paper(paper, CONFIG)
    assert result is not None
    assert result.title == "A Great Paper"
    assert result.abstract == "Multiple spaces. And newlines."


def test_strips_latex_wrapping_and_citation_commands():
    paper = make_paper(abstract="We use \\emph{fine-tuning} as shown in \\cite{smith2020}.")
    result = normalize_paper(paper, CONFIG)
    assert result is not None
    assert "\\emph" not in result.abstract
    assert "\\cite" not in result.abstract
    assert "fine-tuning" in result.abstract


def test_strips_math_delimiters_and_escaped_chars():
    paper = make_paper(abstract="Complexity is $O(n^2)$ and accuracy improved by 5\\% overall.")
    result = normalize_paper(paper, CONFIG)
    assert result is not None
    assert result.abstract == "Complexity is O(n^2) and accuracy improved by 5% overall."


def test_drops_withdrawn_paper():
    paper = make_paper(abstract="This paper has been withdrawn by the author due to a critical error.")
    assert normalize_paper(paper, CONFIG) is None


def test_drops_paper_outside_configured_categories():
    paper = make_paper(categories=["cs.CV"], primary_category="cs.CV")
    assert normalize_paper(paper, CONFIG) is None


def test_drops_paper_published_before_start_date():
    paper = make_paper(published="2019-06-01T00:00:00Z")
    assert normalize_paper(paper, CONFIG) is None


def test_drops_paper_with_blank_title():
    paper = make_paper(title="   ")
    assert normalize_paper(paper, CONFIG) is None


def test_drops_paper_with_no_authors_after_cleaning():
    paper = make_paper(authors=["   ", ""])
    assert normalize_paper(paper, CONFIG) is None


def test_keeps_paper_reached_only_via_keyword_category():
    # Category scope re-check should accept keyword_filtered_categories too,
    # not just the full-scope categories.
    paper = make_paper(categories=["cs.AI"], primary_category="cs.AI")
    result = normalize_paper(paper, CONFIG)
    assert result is not None
