import argparse
import logging
from pathlib import Path

from ingestion.arxiv_client import fetch_papers
from ingestion.config import IngestionConfig
from ingestion.store import RawPaperStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch arXiv paper metadata into the local raw store."
    )
    parser.add_argument("--categories", nargs="+", default=["cs.SE"])
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--db-path", default="data/papers.db")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = IngestionConfig(
        categories=tuple(args.categories),
        start_date=args.start_date,
        max_results=args.max_results,
        db_path=Path(args.db_path),
    )

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


if __name__ == "__main__":
    main()
