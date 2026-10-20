# Ingestion pipeline (offline)

Builds the knowledge base the agent searches. It fetches paper metadata from
the arXiv API, cleans it, and embeds it into a Pinecone index. Run it once
before using the chat app, and again whenever you want to refresh the corpus.

**Status: implemented** and run end-to-end against the full ~30k-paper corpus.
For the full design see [../docs/data-ingestion-plan.md](../docs/data-ingestion-plan.md);
for deferred work see [../TODO.md](../TODO.md). Every knob mentioned below
lives in `config.py` / `embedding_config.py`.

## Quick start: build the full corpus

This gets you from a fresh clone to a populated Pinecone index, which is all
the chat app needs. It fetches paper **metadata and abstracts only** (~30k
papers) — no PDFs or full text are downloaded.

**Prerequisites**

- [uv](https://docs.astral.sh/uv/), then `uv sync` from the repo root.
- An **OpenAI** API key with billing enabled (embeddings only).
- A **Pinecone** account and API key. The index is created automatically on
  first run as a serverless index on AWS `us-east-1`; the free plan should be
  enough for ~30k vectors.
- Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and
  `PINECONE_API_KEY` (`.env` is gitignored). Fetch and normalize need no keys.

**Expected cost and time:** about $0.12 of OpenAI embeddings; the fetch takes
tens of minutes (arXiv asks for ~1 request per 3 s), the embed a few minutes.

**Run these three commands, in order, from the repo root:**

```bash
uv run python -m ingestion.cli fetch       # 1. arXiv -> data/papers.db
uv run python -m ingestion.cli normalize   # 2. clean and filter
uv run python -m ingestion.cli embed       # 3. embed + upsert to Pinecone
```

You'll know each step worked when:

1. `fetch` logs `Fetched N papers` with N around 30k.
2. `normalize` logs `Normalized N, dropped N`; most papers should be kept.
3. The vector count on your index at https://app.pinecone.io matches the
   normalized paper count, with dimension 1536.

If a step is interrupted (or you hit a rate limit or network error), just
re-run it: every stage is idempotent and picks up without duplicating work,
and `embed` skips chunks it has already embedded. Then start the chat app —
see [../agent/README.md](../agent/README.md).

Want to try it cheaply first? Add `--max-results 15` to `fetch` and
`--limit 3` to `embed` for a tiny end-to-end smoke test (see below). To
verify the whole pipeline on a ~1K-paper slice (under a cent of embeddings,
throwaway Pinecone index, deleted afterwards):

```bash
RUN_INGESTION_INTEGRATION_TESTS=1 uv run pytest integration_tests/test_ingestion_pipeline.py -v
```

## How it works

1. **Fetch metadata** — query the arXiv API for `cs.SE` + `cs.PL` in full,
   plus `cs.AI`/`cs.LG`/`cs.CL` filtered to abstracts mentioning a
   software-dev term (too large/generic to take in full — see `config.py`),
   with submission date ≥ 2020-01-01. Stores raw metadata (id, title,
   abstract, authors, categories, published date, PDF URL), currently ~30k
   papers.
2. **Normalize** — drop out-of-scope, withdrawn, and malformed papers; clean
   title/abstract text.
3. **Chunk** — one chunk per paper (its abstract) behind a `ChunkerStrategy`
   interface. Full-text chunking is deferred (see [../TODO.md](../TODO.md)).
4. **Embed** — OpenAI `text-embedding-3-small` (1536-dim), batched.
5. **Upsert to Pinecone** — vectors plus the metadata needed for citations:
   paper id, title, authors, categories, published date, source URL.
6. **Idempotent** — chunks are keyed by paper id + chunk index, so re-running
   upserts in place. A content-hash cache also skips re-embedding chunks that
   haven't changed.

## Running it, step by step (reference)

The detail behind the quick start. All commands run from the repo root; `--db-path`
defaults to `data/papers.db` for all three, so it is omitted below.

### 1. Fetch raw metadata from arXiv

```bash
# Optional: small capped run, good for a first sanity check / debugging:
uv run python -m ingestion.cli fetch --max-results 15

# Full scope (cs.SE + cs.PL, plus keyword-filtered cs.AI/cs.LG/cs.CL, since 2020-01-01):
uv run python -m ingestion.cli fetch
```

Useful flags: `--categories`, `--keyword-categories`, `--keywords`,
`--start-date`, `--max-results`. Rate-limited to arXiv's courtesy
~3s/request, so an uncapped full run over ~30k papers takes a while (tens
of minutes) — start with `--max-results` while debugging.

Data lands in the `papers` table of the SQLite file at `--db-path`
(default `data/papers.db`; `data/` is gitignored — this is local scratch data, not committed).

### 2. Filter & normalize

```bash
uv run python -m ingestion.cli normalize
```

Reads everything in the raw store, drops out-of-scope/withdrawn/malformed
papers, cleans title/abstract text, and writes the result to a separate
`normalized_papers` table in the same file. Re-run anytime after a fetch —
it's idempotent and safe to re-run after tweaking scope/keywords without
re-fetching.

### 3. Chunk, embed, and upsert to Pinecone

```bash
# Optional: smoke-test with a handful of papers before the full run:
uv run python -m ingestion.cli embed --limit 3

# Full run (~30k papers, a few minutes):
uv run python -m ingestion.cli embed
```

One chunk per paper (its abstract) for the MVP, embedded with OpenAI
`text-embedding-3-small` (1536-dim) and upserted into a Pinecone index
(created automatically on first run if it doesn't exist, named whatever
`PINECONE_INDEX_NAME` resolves to, default `ai-code-assistant`).
Re-running skips any chunk whose text hasn't changed since the last run —
tracked via a content hash in an `embedded_chunks` table in the same
SQLite file — so it's both idempotent and cheap to re-run.

### 4. Verify it landed in Pinecone

Check https://app.pinecone.io — open your index and its dashboard shows
the vector count and dimension directly (should match the normalized
paper count and 1536); use the console's query/browse tool to inspect
individual vectors and metadata.

Then start the chat app (see [../agent/README.md](../agent/README.md)).

## Debugging tips

- Both CLI commands log counts (`Fetched N papers`, `Normalized N, dropped
  N`) — a big gap between fetched and normalized counts is worth checking
  with `scripts/inspect_db.py`.
- Run `fetch` with a small `--max-results` first; only drop the cap once a
  small batch looks right end-to-end.
- Logging is INFO by default; category/date/withdrawn drops during
  `normalize` log at DEBUG/INFO — see `normalize.py` if you need to see
  exactly why a specific paper was dropped.
- To step through the pipeline in an IDE debugger, run the module `ingestion.cli`
  with arguments, e.g. `fetch --max-results 20`.

## Visualizing the dataset locally

`scripts/inspect_db.py` is a small stdlib-only tool for looking at the data
without installing a SQLite GUI:

```bash
# Row counts, published-date range, and category breakdown for both tables:
uv run python scripts/inspect_db.py summary

# Most recent N rows (default: normalized table):
uv run python scripts/inspect_db.py sample --table normalized --limit 20
uv run python scripts/inspect_db.py sample --table raw --limit 20

# Full-text-ish search over title/abstract:
uv run python scripts/inspect_db.py search "program repair" --table normalized
```

Other options if you want something more visual:

- **DB Browser for SQLite** (GUI, spreadsheet-like) — a free download if you
  want one locally.
- **pandas in a notebook/REPL** for quick stats or plots:
  ```python
  import sqlite3, pandas as pd
  conn = sqlite3.connect("data/papers.db")
  df = pd.read_sql("SELECT * FROM normalized_papers", conn)
  df["primary_category"].value_counts()
  ```
  (pandas isn't a direct dependency in `pyproject.toml` — it's only needed
  for ad hoc exploration, not the pipeline itself. Run it with
  `uv run --with pandas python`.)
