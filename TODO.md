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
- Neither `tests/test_pipeline_embed.py` (mocked, always runs) nor the
  real-API opt-in suites (`test_arxiv_integration.py`,
  `test_agent_integration.py`) exercise real OpenAI/Pinecone/Tavily in CI
  — the opt-in ones need secrets CI doesn't have configured. Verified
  manually each time so far. Worth adding as GitHub Actions secrets +
  a scheduled (not every-push) CI job if this ever needs to run
  unattended.

## Agent / serving side (MVP done, see docs/agent-plan.md)

Built: `agent/config.py` (`AgentConfig`), `agent/retrieval_tool.py`
(`search_papers`), `agent/graph.py` (the agent itself — both tools, system
prompt, `InMemorySaver` checkpointer), `ui/chat_app.py` (Streamlit chat).
Verified end-to-end against real OpenAI + Pinecone + Tavily across all
three cases (corpus hit, web fallback, out-of-scope decline) — both via
`tests/test_agent_integration.py` and by hand in a real browser through
the actual chat UI.

- **LLM provider decided: OpenAI-only** (`gpt-5-mini` default), for a
  single-provider stack (same account/key already used for embeddings).
  `GROQ_API_KEY` in `.env` is unused as a result — Groq's current models
  (`openai/gpt-oss-120b`/`20b`) were evaluated and confirmed working with
  tool-calling, so it's a viable option to revisit later if cost/latency
  comparison against OpenAI ever becomes worth doing.
- **Web search fallback decided: in scope for MVP**, not deferred —
  `TAVILY_API_KEY` is used by a second tool (`web_search`, via
  `langchain-tavily`'s ready-made `TavilySearch`), triggered when
  `search_papers` finds nothing relevant. Confirmed working with a real
  call. See docs/agent-plan.md for how the system prompt keeps this
  scoped to AI/software-dev questions rather than a general web assistant.
- **Real gap found and fixed while testing**: an early system-prompt
  version let the agent answer a genuinely out-of-scope question (a
  cookie recipe) directly from general knowledge, calling no tool at all
  — exactly the failure mode the plan flagged as "hard to guarantee from
  the prompt alone." Fixed by making the scope check an explicit Step 0
  in the prompt, before any tool-use instructions; now codified as a
  regression test.
- **Real gap found and fixed via browser testing** (things the REPL-only
  verification couldn't catch): `streamlit run ui/chat_app.py` put `ui/`
  rather than the project root on `sys.path`, breaking `import agent` —
  fixed with an explicit path insert. Found the identical latent bug in
  `scripts/debug_pipeline.py` while checking for it elsewhere (masked
  there because it's normally run from PyCharm, which sets `PYTHONPATH`
  itself) and fixed it the same way. Also deduped the Sources list by
  URL — the agent sometimes calls `search_papers` twice in one turn
  (e.g. a refined retry), which surfaced the same paper twice.
- `LANGSMITH_*` vars are in `.env` but tracing isn't wired into any code
  yet — should already auto-instrument once these are set (no code change
  expected), but not yet verified that traces actually show up in the
  LangSmith UI.
- Conversation memory is `InMemorySaver` — process-local, lost whenever
  the Streamlit server restarts. Fine for local single-user use; durable
  storage is a later concern if this ever needs to survive restarts or
  serve multiple users.
