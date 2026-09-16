import argparse
import logging
from pathlib import Path

from ingestion.arxiv_client import fetch_papers
from ingestion.config import IngestionConfig
from ingestion.normalize import normalize_paper
from ingestion.normalized_store import NormalizedPaperStore
from ingestion.store import RawPaperStore


def _add_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Categories included in full, no keyword filter (default: cs.SE cs.PL)",
    )
    parser.add_argument(
        "--keyword-categories", nargs="+", default=None,
        help="Categories only included when abstract matches --keywords (default: cs.AI cs.LG cs.CL)",
    )
    parser.add_argument(
        "--keywords", nargs="+", default=None,
        help="Abstract keywords that gate --keyword-categories (default: software-dev term list)",
    )
    parser.add_argument("--start-date", default="2020-01-01")


def _config_overrides_from_args(args: argparse.Namespace) -> dict:
    overrides: dict = {"start_date": args.start_date}
    if args.categories is not None:
        overrides["categories"] = tuple(args.categories)
    if args.keyword_categories is not None:
        overrides["keyword_filtered_categories"] = tuple(args.keyword_categories)
    if args.keywords is not None:
        overrides["keywords"] = tuple(args.keywords)
    return overrides


def _run_fetch(args: argparse.Namespace) -> None:
    overrides = _config_overrides_from_args(args)
    overrides["max_results"] = args.max_results
    overrides["db_path"] = Path(args.db_path)
    config = IngestionConfig(**overrides)

    fetched_count = 0
    with RawPaperStore(config.db_path) as store:
        for paper in fetch_papers(config):
            store.upsert(paper)
            fetched_count += 1
            if fetched_count % 50 == 0:
                logging.info("Fetched %d papers so far...", fetched_count)

        logging.info(
            "Done. Fetched %d papers this run. Store now has %d papers.",
            fetched_count,
            store.count(),
        )


def _run_normalize(args: argparse.Namespace) -> None:
    overrides = _config_overrides_from_args(args)
    overrides["db_path"] = Path(args.db_path)
    config = IngestionConfig(**overrides)

    kept, dropped = 0, 0
    with RawPaperStore(config.db_path) as raw_store, NormalizedPaperStore(config.db_path) as norm_store:
        # Materialize before writing: raw_store.all() is a live cursor, and
        # a second connection to the same SQLite file can't commit writes
        # while that read transaction is still open ("database is locked").
        papers = list(raw_store.all())
        for paper in papers:
            normalized = normalize_paper(paper, config)
            if normalized is None:
                dropped += 1
                continue
            norm_store.upsert(normalized)
            kept += 1

        logging.info(
            "Done. Normalized %d papers, dropped %d. Store now has %d normalized papers.",
            kept,
            dropped,
            norm_store.count(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv ingestion pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch arXiv metadata into the raw store.")
    _add_scope_args(fetch_parser)
    fetch_parser.add_argument("--max-results", type=int, default=None)
    fetch_parser.add_argument("--db-path", default="data/papers.db")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Filter and clean raw papers into the normalized store."
    )
    _add_scope_args(normalize_parser)
    normalize_parser.add_argument("--db-path", default="data/papers.db")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "fetch":
        _run_fetch(args)
    elif args.command == "normalize":
        _run_normalize(args)


if __name__ == "__main__":
    main()
