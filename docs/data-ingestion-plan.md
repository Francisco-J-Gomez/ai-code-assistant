# Data Ingestion Pipeline — Step-by-Step Plan

**Status: implemented and run end-to-end** — ~30k papers fetched,
normalized, embedded, and upserted to Pinecone. This doc is kept as the
detailed design reference; see [README.md](../README.md) for exact
commands to run it yourself.

Scope: MVP ingestion of arXiv paper **abstracts** for `cs.SE` + `cs.PL` in
full, plus `cs.AI`/`cs.LG`/`cs.CL` filtered to abstracts mentioning a
software-dev term, since 2020, into Pinecone. Full-text ingestion and
chunking benchmarks are explicitly deferred (see [README.md](../README.md)).

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

- `.env` for secrets: `OPENAI_API_KEY`, `PINECONE_API_KEY` (plus
  `GROQ_API_KEY`/`TAVILY_API_KEY`/`LANGSMITH_*` for the not-yet-built
  serving/agent side — see `.env.example`).
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

- Each full-scope category (`cs.SE`, `cs.PL`) and the keyword-filtered
  clause (`cs.AI`/`cs.LG`/`cs.CL` + software-dev terms) is queried
  **separately** rather than OR'd into one query, filtered by
  `submittedDate:[20200101 TO *]`.
- **Real production issue found and fixed**: arXiv's API 500s on *any*
  query once `start` reaches ~10000, regardless of query complexity (not
  documented, found by hitting it during the full run). A scope query
  whose result count would exceed that has its date range bisected
  up front (probed via a cheap `max_results=1` request first) rather than
  paginating into the wall — see `ARXIV_MAX_START` in
  `ingestion/arxiv_client.py`.
- Paginate with `start`/`max_results` (batches of ~100); **respect
  arXiv's rate-limit guidance (~1 request per 3s)** — this is a courtesy
  limit on a free public API, not a hard block, so violating it risks
  IP-level throttling.
- Exponential backoff + capped retries on transient HTTP errors (5xx,
  timeouts, DNS failures — survived an overnight network drop during the
  real full run without manual intervention).
- arXiv returns one entry per version — dedupe to the latest version per
  base id, but keep the version number in the stored record.
- arXiv's `abs:"..."` field search doesn't enforce exact phrase matching
  (confirmed: a nonsense reversed phrase still returned hundreds of hits),
  so the keyword-filtered clause is re-checked client-side with a real
  substring match rather than trusted as-is.
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

- **Decided**: OpenAI `text-embedding-3-small` (1536-dim) — cheap (~$0.12
  for the full ~30k-paper corpus), fast, best LangChain support.
  Alternatives to benchmark later, see [TODO.md](../TODO.md).
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

- CLI with separate subcommands (`fetch`, `normalize`, `embed`) so each
  stage is independently runnable and debuggable.
- **Implemented differently than originally planned**: no separate
  state/checkpoint file. The SQLite tables themselves are the checkpoint —
  `papers` (raw), `normalized_papers`, and `embedded_chunks` (content-hash
  cache) each upsert by id, so re-running any stage naturally only (re-)
  processes what's new or changed, without a separate tracking file.

### 10. Observability & testing

- Structured logging per stage: counts in/out, counts skipped, errors.
- Unit tests: fetcher (mocked HTTP), normalizer, chunker, embedder,
  Pinecone client (all mocked) — no network calls; 50 tests total.
- Integration test against the real arXiv API, opt-in via
  `RUN_ARXIV_INTEGRATION_TESTS=1` (`tests/test_arxiv_integration.py`).
  **Not yet done**: an equivalent opt-in test against real OpenAI/Pinecone
  for the embed stage — verified manually instead so far (see TODO.md).
- Smoke test: full pipeline run on a handful of papers (`--limit`/
  `--max-results`) before scaling to the full corpus — done for every
  stage before each full run.

### 11. Cost & rate-limit notes

- arXiv API: free, no auth, but rate-limited by convention (~1 req/3s).
- Embedding cost scales with corpus size — the actual full scope (`cs.SE` +
  `cs.PL` in full, plus keyword-filtered `cs.AI`/`cs.LG`/`cs.CL`) since
  2020 turned out to be **~30k papers**, not "a few thousand" as
  originally guessed — real cost with `text-embedding-3-small` is ~$0.12
  for the whole corpus, confirmed by an actual full run.
- Pinecone cost driven by vector count × dimension. Free tier comfortably
  covers 30k vectors at 1536-dim.

## Project layout

The layout below was the original proposal; the real `ingestion/`
directory is the source of truth and has grown a few files beyond this
(`normalize.py`, `normalized_store.py`, `pipeline.py`,
`embedding_config.py`, `embedding_cache.py`, a fuller `tests/` — see the
repo directly rather than expecting this block to stay in sync):

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

## Implementation order

1. ✅ `config.py` + `arxiv_client.py` + `store.py` — fetch `cs.SE` only into
   the raw store, no embedding yet. Verify counts look sane.
2. ✅ Extend to full MVP category/date scope.
3. ✅ Chunking (abstract) + embedding, with the content-hash cache.
4. ✅ Pinecone index creation + upsert; verified with a manual query and a
   full ~30k-paper run.
5. ✅ Wire the CLI and logging end-to-end (checkpointing via SQLite upserts
   rather than a separate file — see step 9 above).
6. ✅ Unit tests + smoke test (50 tests; opt-in real-API integration tests
   for the fetch stage only so far).
7. **Not started** (separate milestone): full-text parsing and chunking
   benchmarks.

## Decisions that were open before implementation, now resolved

- ~~Embedding provider/model~~ → OpenAI `text-embedding-3-small`, 1536-dim.
- ~~Raw store engine: SQLite vs. JSONL~~ → SQLite.
- ~~Pinecone deployment: serverless vs. pod-based, and region~~ →
  serverless, AWS `us-east-1` (configurable via `EmbeddingConfig`).
