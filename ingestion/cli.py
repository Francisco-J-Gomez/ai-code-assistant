import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from ingestion.config import EmbeddingConfig, IngestionConfig
from ingestion.pipeline import run_embed_stage, run_fetch_stage, run_normalize_stage


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
    run_fetch_stage(IngestionConfig(**overrides))


def _run_normalize(args: argparse.Namespace) -> None:
    overrides = _config_overrides_from_args(args)
    overrides["db_path"] = Path(args.db_path)
    run_normalize_stage(IngestionConfig(**overrides))


def _run_embed(args: argparse.Namespace) -> None:
    config = EmbeddingConfig(
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
        embed_batch_size=args.embed_batch_size,
        pinecone_index_name=args.pinecone_index,
        pinecone_cloud=args.pinecone_cloud,
        pinecone_region=args.pinecone_region,
        db_path=Path(args.db_path),
        limit=args.limit,
    )
    run_embed_stage(config)


def main() -> None:
    load_dotenv()  # before building the parser, so env-based defaults below can see it

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

    embed_parser = subparsers.add_parser(
        "embed", help="Chunk normalized papers, embed them, and upsert to Pinecone."
    )
    embed_parser.add_argument("--db-path", default="data/papers.db")
    embed_parser.add_argument("--embedding-model", default="text-embedding-3-small")
    embed_parser.add_argument("--embedding-dimensions", type=int, default=1536)
    embed_parser.add_argument("--embed-batch-size", type=int, default=100)
    embed_parser.add_argument(
        "--pinecone-index", default=os.environ.get("PINECONE_INDEX_NAME", "ai-code-assistant")
    )
    embed_parser.add_argument("--pinecone-cloud", default="aws")
    embed_parser.add_argument("--pinecone-region", default="us-east-1")
    embed_parser.add_argument(
        "--limit", type=int, default=None, help="Cap papers processed, for smoke-testing"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "fetch":
        _run_fetch(args)
    elif args.command == "normalize":
        _run_normalize(args)
    elif args.command == "embed":
        _run_embed(args)


if __name__ == "__main__":
    main()
