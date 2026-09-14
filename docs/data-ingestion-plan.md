# Data Ingestion Pipeline — Step-by-Step Plan

Scope: MVP ingestion of arXiv paper **abstracts** for `cs.SE` + a slice of
`cs.AI`/`cs.LG`, since 2020, into Pinecone. Full-text ingestion and chunking
benchmarks are explicitly deferred (see [README.md](../README.md)).

## Design principles

- **Resumable & idempotent** — re-running any stage should not duplicate work
  or data. A crash mid-run should lose at most the current batch.
- **Stages decoupled** — fetch, chunk, embed, upsert are separable steps with
  a persisted artifact between each, so any stage can be re-run, debugged, or
  swapped independently (e.g. swapping the chunker later without touching
  fetch/embed/upsert).
- **Cheap to fail** — validate and log bad records rather than crashing the
  whole run; one malformed abstract shouldn't kill a 5,000-paper batch.
- **Cost-aware** — cache embeddings by content hash so edits/re-runs don't
  re-embed unchanged chunks.

## High-level flow

```
arXiv API → raw metadata store → filter/normalize → chunk → embed → upsert (Pinecone)
```

## Steps

### 1. Config & project setup

- `.env` for secrets: `PINECONE_API_KEY`, `EMBEDDING_API_KEY`.
- `config.yaml` (or similar) for non-secret knobs: categories, date range,
  batch sizes, Pinecone index/namespace — keeps re-scoping (e.g. adding a
  category later) a config change, not a code change.
- `.env.example` committed; `.env` gitignored.

### 2. Data model

`PaperMetadata`: `arxiv_id`, `version`, `title`, `abstract`, `authors`,
`categories`, `primary_category`, `published`, `updated`, `pdf_url`.

### 3. Raw metadata store

Persist fetched metadata locally (SQLite, keyed by `arxiv_id`) **before** any
chunking/embedding. This is the cache/audit trail that lets later stages
re-run without re-hitting the arXiv API, and gives a resumability checkpoint
if a fetch run is interrupted.

### 4. arXiv fetcher

- Query: `cat:cs.SE OR cat:cs.AI OR cat:cs.LG` (start narrow, widen later via
  config), filtered by `submittedDate:[20200101 TO *]`.
- Paginate with `start`/`max_results` (batches of ~100–1000); **respect
  arXiv's rate-limit guidance (~1 request per 3s)** — this is a courtesy
  limit on a free public API, not a hard block, so violating it risks
  IP-level throttling.
- Exponential backoff + capped retries on transient HTTP errors (5xx,
  timeouts).
- arXiv returns one entry per version — dedupe to the latest version per
  base id, but keep the version number in the stored record.
- Write to the raw store incrementally per page/batch, upserted by
  `arxiv_id`, so an interrupted fetch resumes rather than restarts.

### 5. Filter & normalize

- Re-assert category + date scope after fetch (defense in depth).
- Drop withdrawn papers.
- Clean abstract text (strip LaTeX artifacts, collapse whitespace), normalize
  author name formatting for later citation display.
- Validate required fields; log and skip malformed records instead of
  failing the batch.

### 6. Chunking (MVP: abstract-only)

- One chunk per paper = the abstract (no splitting needed at this length).
- Attach citation metadata to every chunk: `arxiv_id`, `title`, `authors`,
  `categories`, `published`, `source_url`, `chunk_index=0`,
  `chunk_type="abstract"`.
- Implement as a `ChunkerStrategy` interface even though the MVP has one
  implementation — full-text chunkers (fixed-window, section-aware) from the
  planned benchmarking work plug in later without touching fetch/embed/upsert.

### 7. Embedding

- **Open decision**: embedding provider/model (affects Pinecone index
  dimension — must be chosen before index creation).
- Batch requests (e.g. 100 chunks/call).
- Cache embeddings keyed by a hash of chunk text, so unchanged chunks are
  never re-embedded on subsequent runs — this is the main cost control.
- Retry on API errors; surface persistent failures rather than silently
  dropping chunks.

### 8. Upsert to Pinecone

- One-time index setup: dimension = embedding model's output size, metric =
  cosine.
- Vector id = `arxiv_id` (+ `chunk_index` once multi-chunk) — re-ingestion
  overwrites in place, which is what makes the whole pipeline idempotent.
- Batch upserts (~100 vectors/call).
- Keep metadata payload within Pinecone's per-vector metadata limit; fine for
  abstracts, will need truncation/pointer-only once full text is added.

### 9. Orchestration

- CLI with separate subcommands (`fetch`, `embed`, `upsert`, or a `run` that
  chains them) so each stage is independently runnable and debuggable.
- A small state/checkpoint file (or a status column in the raw store) tracks
  which papers are fetched vs. embedded vs. upserted, so `run` only processes
  what's new.

### 10. Observability & testing

- Structured logging per stage: counts in/out, counts skipped, errors.
- Unit tests: fetcher (mocked HTTP), normalizer, chunker — no network calls.
- Integration test against a small real arXiv query + a scratch Pinecone
  namespace, marked slow/opt-in (not run by default).
- Smoke test: full pipeline on ~10 papers before scaling to the full corpus.

### 11. Cost & rate-limit notes

- arXiv API: free, no auth, but rate-limited by convention (~1 req/3s).
- Embedding cost scales with corpus size — abstracts-only for `cs.SE` + a
  slice of `cs.AI`/`cs.LG` since 2020 is a few thousand papers, so get a
  rough count from a dry-run fetch before committing to an embedding budget.
- Pinecone cost driven by vector count × dimension.

## Proposed project layout

```
ai-code-assistant/
  ingestion/
    __init__.py
    config.py
    arxiv_client.py
    models.py              # PaperMetadata
    store.py               # raw metadata persistence (SQLite)
    chunking/
      base.py              # ChunkerStrategy interface
      abstract_chunker.py
    embeddings.py
    pinecone_client.py
    cli.py
  tests/
    test_arxiv_client.py
    test_chunking.py
    test_store.py
  docs/
    data-ingestion-plan.md   # this file
  .env.example
```

## Suggested implementation order

1. `config.py` + `arxiv_client.py` + `store.py` — fetch `cs.SE` only into
   the raw store, no embedding yet. Verify counts look sane.
2. Extend to full MVP category/date scope.
3. Chunking (abstract) + embedding, with the content-hash cache.
4. Pinecone index creation + upsert; verify with a manual query.
5. Wire the CLI, checkpointing, and logging end-to-end.
6. Unit tests + smoke test.
7. (Separate milestone) full-text parsing and chunking benchmarks.

## Open decisions before implementation starts

- Embedding provider/model (drives Pinecone index dimension).
- Raw store engine: SQLite vs. JSONL (leaning SQLite for query/checkpoint
  convenience).
- Pinecone deployment: serverless vs. pod-based, and region.
