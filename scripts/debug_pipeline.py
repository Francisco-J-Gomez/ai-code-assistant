#!/usr/bin/env python3
"""Debug entry point: runs fetch + normalize back-to-back in one process.

In PyCharm, right-click this file -> Debug 'debug_pipeline' to step through
the whole ingestion pipeline (ingestion/arxiv_client.py -> ingestion/store.py
-> ingestion/normalize.py -> ingestion/normalized_store.py) with breakpoints,
without configuring a Run/Debug Configuration's arguments by hand - the
`ingestion.cli` module needs a subcommand (fetch/normalize) and won't run
directly from a plain "Run" click.

MAX_RESULTS is capped by default so a debug session doesn't sit through the
full rate-limited fetch (~33.5k papers, ~1 request/3s). Set it to None once
you're done stepping through and want the full dataset - or just use the
CLI directly at that point:

    .venv/bin/python -m ingestion.cli fetch --db-path data/papers.db
    .venv/bin/python -m ingestion.cli normalize --db-path data/papers.db
    .venv/bin/python -m ingestion.cli embed --db-path data/papers.db

RUN_EMBED is off by default since it needs OPENAI_API_KEY/PINECONE_API_KEY
in .env and spends real (if tiny) money - flip it on once those are set.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from ingestion.config import IngestionConfig
from ingestion.embedding_config import EmbeddingConfig
from ingestion.pipeline import run_embed_stage, run_fetch_stage, run_normalize_stage

MAX_RESULTS = 20  # None for the full scope
DB_PATH = Path("data/papers.db")
RUN_EMBED = False


def main() -> None:
    load_dotenv()  # before reading any PINECONE_INDEX_NAME-style env overrides below
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = IngestionConfig(max_results=MAX_RESULTS, db_path=DB_PATH)

    run_fetch_stage(config)
    run_normalize_stage(config)

    if RUN_EMBED:
        embedding_config = EmbeddingConfig(
            db_path=DB_PATH,
            pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME", "ai-code-assistant"),
        )
        run_embed_stage(embedding_config)


if __name__ == "__main__":
    main()
