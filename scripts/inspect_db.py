#!/usr/bin/env python3
"""Local inspection tool for the ingestion SQLite store.

Stdlib only (no dependency on requirements.txt), so it works even before
the venv is set up, and lets you sanity-check the pipeline's output without
installing a separate SQLite GUI. Usage:

    python scripts/inspect_db.py summary
    python scripts/inspect_db.py sample --table normalized --limit 10
    python scripts/inspect_db.py search "code generation" --table normalized
"""
import argparse
import sqlite3
import textwrap
from collections import Counter
from pathlib import Path

TABLES = {"raw": "papers", "normalized": "normalized_papers"}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def cmd_summary(conn: sqlite3.Connection) -> None:
    for label, table in TABLES.items():
        if not _table_exists(conn, table):
            print(f"{label.upper():10s} table '{table}' not found "
                  f"(run the corresponding CLI step first)")
            continue

        total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"\n{label.upper()} ({table}): {total} papers")
        if total == 0:
            continue

        min_pub, max_pub = conn.execute(f"SELECT MIN(published), MAX(published) FROM {table}").fetchone()
        print(f"  published range: {min_pub} .. {max_pub}")

        categories = Counter(
            row[0] for row in conn.execute(f"SELECT primary_category FROM {table}")
        )
        print("  top primary categories:")
        for category, count in categories.most_common(10):
            print(f"    {category:10s} {count}")


def cmd_sample(conn: sqlite3.Connection, table_key: str, limit: int) -> None:
    table = TABLES[table_key]
    if not _table_exists(conn, table):
        print(f"table '{table}' not found")
        return

    rows = conn.execute(
        f"SELECT arxiv_id, primary_category, published, title FROM {table} "
        f"ORDER BY published DESC LIMIT ?",
        (limit,),
    ).fetchall()
    for arxiv_id, primary_category, published, title in rows:
        print(f"{arxiv_id:14s} {primary_category:8s} {published[:10]}  {textwrap.shorten(title, 90)}")


def cmd_search(conn: sqlite3.Connection, table_key: str, term: str, limit: int) -> None:
    table = TABLES[table_key]
    if not _table_exists(conn, table):
        print(f"table '{table}' not found")
        return

    like = f"%{term}%"
    rows = conn.execute(
        f"SELECT arxiv_id, primary_category, title FROM {table} "
        f"WHERE title LIKE ? OR abstract LIKE ? LIMIT ?",
        (like, like, limit),
    ).fetchall()
    print(f"{len(rows)} match(es) for {term!r}:")
    for arxiv_id, primary_category, title in rows:
        print(f"  {arxiv_id:14s} {primary_category:8s} {textwrap.shorten(title, 90)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the local ingestion SQLite store.")
    parser.add_argument("--db-path", default="data/papers.db")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Counts, date range, category breakdown.")

    sample_parser = subparsers.add_parser("sample", help="Print sample rows.")
    sample_parser.add_argument("--table", choices=TABLES, default="normalized")
    sample_parser.add_argument("--limit", type=int, default=10)

    search_parser = subparsers.add_parser("search", help="Search title/abstract for a term.")
    search_parser.add_argument("term")
    search_parser.add_argument("--table", choices=TABLES, default="normalized")
    search_parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        parser.error(f"no such database: {db_path} (run `python -m ingestion.cli fetch` first)")

    conn = sqlite3.connect(db_path)
    try:
        if args.command == "summary":
            cmd_summary(conn)
        elif args.command == "sample":
            cmd_sample(conn, args.table, args.limit)
        elif args.command == "search":
            cmd_search(conn, args.table, args.term, args.limit)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
