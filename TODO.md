# TODO / Future Enhancements

Deliberately deferred while getting a working end-to-end pipeline in place
first. Not blocking current progress — revisit when relevant.

## Embedding & retrieval quality

- **Benchmark alternative embedding models** once we can measure retrieval
  quality (need eval questions + expected source papers first): OpenAI
  `text-embedding-3-large`, Cohere `embed-english-v3.0` (asymmetric
  search_document/search_query modes are interesting for RAG specifically),
  local `sentence-transformers` (bge-base/large).
- **Chunking strategy benchmarks** (already flagged in
  docs/data-ingestion-plan.md) once full-text PDF parsing exists — compare
  abstract-only vs. fixed-window vs. section-aware chunking.
- **Pinecone namespace strategy**: currently everything upserts into the
  default (empty) namespace. Consider namespacing by embedding model
  version so alternate models can be trialed without clobbering the
  "production" vectors, or so old vectors can be cleanly retired.

## Ingestion pipeline hardening

- Only `PINECONE_INDEX_NAME` currently gets an env-var override in the CLI
  (`ingestion/cli.py`); the other embed knobs (model, dimensions, batch
  size, cloud/region) are CLI-flag-only. Extend if they turn out to need
  changing without a flag.
- No automated integration test for the embed stage against real
  OpenAI/Pinecone (unlike `tests/test_arxiv_integration.py` for the fetch
  stage) — verified manually this session instead, since it costs real
  (if tiny) money per run and needs a scratch Pinecone index. Worth adding
  one, opt-in, if this pipeline gets touched often enough for regressions
  to be a real risk.
- No concurrency in the embed stage — batches are embedded and upserted
  serially. Fine at ~30k papers / ~300 batches, worth revisiting if the
  corpus grows an order of magnitude.
- No real cost/token tracking during a run — cost estimates so far are
  upfront approximations (~$0.12 for the full corpus), not measured from
  actual API responses.

## CI/CD

- `.github/workflows/ci.yml` runs unit + arXiv integration tests on every
  PR/push to `main`, but **merges aren't actually blocked on it failing**:
  GitHub's required-status-checks / branch protection (classic API and the
  newer rulesets API both) needs either a public repo or GitHub Pro on the
  free plan — this repo is private. Revisit if the repo goes public or the
  account upgrades; until then, check the PR's CI status manually before
  merging.

## Agent / serving side (in progress, see docs/agent-plan.md)

- **LLM provider decided: OpenAI-only** (`gpt-5-mini` default), for a
  single-provider stack (same account/key already used for embeddings).
  `GROQ_API_KEY` in `.env` is unused as a result — Groq's current models
  (`openai/gpt-oss-120b`/`20b`) were evaluated and confirmed working with
  tool-calling, so it's a viable option to revisit later if cost/latency
  comparison against OpenAI ever becomes worth doing.
- `TAVILY_API_KEY` is in `.env` but unused — web-search tool is explicitly
  deferred past the MVP (single-tool: corpus search only), see
  docs/agent-plan.md.
- `LANGSMITH_*` vars are in `.env` but tracing isn't wired into any code
  yet — wire in when building the agent, for visibility into
  retrieval + generation steps.
