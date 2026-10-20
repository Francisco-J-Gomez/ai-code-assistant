# RAG Agent — Step-by-Step Plan

Scope: a conversational agent, exposed as a chat UI, that answers questions
about AI and software development grounded in the Pinecone corpus the
ingestion pipeline built (~30k arXiv abstracts), falling back to a live
Tavily web search — still scoped to AI/software-dev topics — when the
corpus has nothing relevant. This is the "Serving API" half of the
project (see [agent/README.md](../agent/README.md)),
picking up where `docs/data-ingestion-plan.md` left off.

## Design principles

- **Grounded first, honest about gaps** — always search before answering a
  factual question (corpus, then web as a fallback — see below); if
  neither turns up anything relevant, say so rather than hallucinate
  (extends the original "Decisions so far" table from the old README, which
  predates the web-search-fallback decision).
- **Real tool-calling agent, not a fixed chain** — the LLM decides whether
  and when to call the retrieval tool, via LangChain's agent framework.
  This is the actual "agents" learning goal of the project, not just RAG.
- **Reuse the ingestion code, don't fork it** — the retrieval tool must
  embed queries with the *same* model/dimension used to build the index
  (`text-embedding-3-small`, 1536-dim). Reusing `ingestion.embeddings.Embedder`
  and `ingestion.pinecone_client.PineconeStore` directly guarantees that by
  construction, instead of a second implementation that could drift.
- **Corpus first, web search as an honest fallback** — always try
  `search_papers` before answering; only fall back to a live Tavily web
  search when the corpus has nothing relevant, so the assistant can still
  answer rather than flatly refuse. The system prompt keeps the fallback
  scoped to AI/software-development questions (the project's actual
  purpose), not a general web assistant — and the answer clearly marks
  which claims are grounded in the paper corpus vs. sourced from the web.
- **Start narrow otherwise**: one model, no memory persistence beyond a
  chat session. Alternate models and durable multi-session memory are
  explicitly deferred (see Open decisions and TODO.md).

## High-level flow

```
User question (Streamlit chat)
  -> LangChain agent (langchain.agents.create_agent, OpenAI LLM)
       -> tool: search_papers(query) -> embeds query, queries Pinecone top-k
       -> if relevant results: LLM answers from them, with paper citations
       -> if nothing relevant: tool: web_search(query) (Tavily), scoped by
          the system prompt to stay on AI/software-dev topics
            -> LLM answers from web results, clearly marked as
               web-sourced (not the curated corpus), with links
  -> answer + sources rendered in the chat UI
```

Single-provider by design: OpenAI for both embeddings (already decided,
`text-embedding-3-small`) and the agent LLM — one API key, one bill, no
second provider account to manage. (Groq was evaluated as an alternative
and its models confirmed working with tool-calling, but OpenAI-only is
simpler and was the explicit choice — see TODO.md if Groq/other providers
are ever worth revisiting, e.g. for cost or latency comparison.)

## Verified before writing this plan

Checked directly rather than assumed, since APIs/models drift fast:

- **LangChain is now v1.x** (installed: 1.4.2) and has a new unified agent
  API, `langchain.agents.create_agent(model, tools, system_prompt,
  checkpointer=...)` — built on LangGraph underneath, this is the modern
  entry point (vs. reaching into `langgraph.prebuilt.create_react_agent`
  directly, which still works but is the lower-level API).
- **`langchain-openai` installs cleanly** here — no numpy dependency at
  all (it needs `tiktoken`, which has a Python 3.14 wheel). The earlier
  numpy/no-compiler install failure was from `langchain-pinecone`
  specifically, not `langchain-openai` — and we don't need
  `langchain-pinecone` anyway, since the retrieval tool reuses
  `ingestion.pinecone_client.PineconeStore` (raw `pinecone` SDK) directly.
- **Current OpenAI models confirmed working with tool-calling** via a
  real `ChatOpenAI(...).bind_tools([...])` call: `gpt-5`, `gpt-5-mini`,
  `gpt-5-nano`, plus the still-available `gpt-4.1-mini`/`gpt-4o-mini`.
- **`langchain-tavily`'s `TavilySearch` confirmed working** with a real
  call using the `.env` key — tool name `tavily_search`, returns a dict
  with a `results` list of `{title, url, content, ...}`. It's a ready-made
  LangChain tool (no custom wrapper code needed, unlike `search_papers`)
  — just instantiate `TavilySearch(max_results=3)` directly.
- **`langchain`, `langgraph`, `langchain-openai`, `langchain-tavily`,
  `streamlit`** all install cleanly (no numpy dependency).

## Steps

### 1. Config

New `AgentConfig` dataclass (mirrors `IngestionConfig`/`EmbeddingConfig`):
`llm_model` (default `gpt-5-mini`), `llm_temperature` (default 0,
factual/grounded answers over creative ones), `retrieval_top_k` (default
5), `web_search_max_results` (default 3), `embedding_model`/
`embedding_dimensions` (must match the ingestion side — reuse
`EmbeddingConfig`'s defaults rather than duplicating), `pinecone_index_name`.

### 2. Tools

- **`search_papers(query: str) -> list[dict]`** (primary, custom):
  embeds `query` with `Embedder`, queries `PineconeStore`'s underlying
  index for top-k matches, returns `[{title, authors, published,
  source_url, text, score}, ...]`. Wrapped via `@tool` (from
  `langchain_core.tools`) with a docstring making clear it searches *this
  specific* AI/software-engineering arXiv corpus, not the general web.
- **`web_search`** (fallback, ready-made): `TavilySearch(max_results=
  config.web_search_max_results)` from `langchain-tavily` — no custom
  code needed. Given to the agent as a second tool.
- Unit tests: `search_papers` with mocked `Embedder`/`PineconeStore` (same
  pattern as `tests/test_pipeline_embed.py`). `web_search` needs no unit
  test of our own — it's a third-party tool, not our logic.
- Opt-in real-API integration test (mirrors `integration_tests/test_arxiv_integration.py`):
  a known in-corpus question resolves via `search_papers` (expected paper
  id appears), and a known out-of-corpus-but-in-scope question triggers
  `web_search` instead.

### 3. Agent

- `langchain.agents.create_agent(model=ChatOpenAI(...), tools=[search_papers,
  web_search], system_prompt=..., checkpointer=InMemorySaver())`.
- System prompt instructs, in order: (1) always call `search_papers` first
  for questions about AI/software-development research; (2) only call
  `web_search` if `search_papers` returned nothing relevant, and only for
  questions still in the AI/software-dev scope — not a general web
  assistant; (3) cite sources either way (paper title + link, or web
  title + URL) and clearly mark whether an answer is grounded in the
  curated corpus or supplemented from a live web search; (4) if neither
  tool turns up anything relevant, say so plainly rather than guessing.
- `InMemorySaver` checkpointer for MVP conversation memory — process-local,
  lost on restart. Fine for a single-user local chat session; durable
  memory is a later concern if this ever needs multiple users/sessions.

### 4. Citations

Pull citation info from each tool's *structured return value*
(`search_papers`: title/authors/source_url; `web_search`: title/url from
Tavily's `results`), not by asking the LLM to format links in free text —
more reliable, and lets the UI render a proper "Sources" list — tagged by
which tool it came from — from the agent's intermediate tool-call results
rather than parsing prose.

### 5. Chat UI (Streamlit)

- `ui/chat_app.py`, run via `streamlit run ui/chat_app.py`.
- `st.chat_message`/`st.chat_input` for the conversation.
- The compiled agent graph and a per-session `thread_id` live in
  `st.session_state` — Streamlit reruns the whole script on every
  interaction, so the agent must be created once and reused, not rebuilt
  per message.
- A "Sources" expander under each answer, populated from that turn's tool
  call results — labeled by tool (📄 corpus paper vs. 🌐 web result) so
  it's visually obvious which kind of source backs the answer.

### 6. Observability

`LANGSMITH_TRACING`/`LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` are already in
`.env` (added earlier, unused until now) — LangChain auto-instruments
when these are set, no code changes needed. Verify traces actually show
up in the LangSmith UI after the first real run.

### 7. Testing

- Unit tests: `search_papers` (mocked), config defaults.
- Opt-in integration test: real OpenAI + real Pinecone + real Tavily —
  (a) a question whose answer is known to come from a specific paper,
  assert that paper's id appears in the tool results; (b) a question
  in-scope but not covered by the corpus, assert `web_search` gets called
  and a real URL comes back; (c) a question genuinely out of scope
  (nothing to do with AI/software-dev), assert the agent declines rather
  than answering from `web_search` anyway — this is the one behavior
  that's hard to guarantee from the system prompt alone and worth
  actually checking.
- Manual: run the Streamlit app, ask a handful of real questions across
  all three cases above, check citations render correctly labeled
  (corpus vs. web) and match what was actually used.

## Project layout

```
ai-code-assistant/
  agent/
    __init__.py
    config.py            # AgentConfig
    retrieval_tool.py     # search_papers tool (reuses ingestion.embeddings/pinecone_client)
    graph.py               # create_agent(...) wiring, system prompt, web_search (TavilySearch)
  ui/
    chat_app.py             # Streamlit chat interface
  tests/
    test_retrieval_tool.py
  integration_tests/
    test_agent_integration.py   # opt-in, real APIs (corpus hit / web fallback / out-of-scope)
```

## Implementation order

1. `agent/config.py` + `agent/retrieval_tool.py` — verify `search_papers`
   against the real index with a couple of known-answer queries before
   wiring up the LLM at all.
2. `agent/graph.py` — the agent with both tools; verify with a plain
   Python REPL loop (no UI yet), testing all three cases (corpus hit,
   web fallback, genuinely out of scope).
3. `ui/chat_app.py` — Streamlit wrapper around the working agent.
4. LangSmith trace verification.
5. Unit tests + the opt-in integration test.

## Open decisions to confirm before implementation

- **OpenAI model**: `gpt-5-mini` (recommended default — good
  quality/cost balance) vs. `gpt-5` (best quality, pricier) vs.
  `gpt-5-nano` (cheapest/fastest) — all three confirmed working with
  tool-calling.
- **`retrieval_top_k`**: default 5, adjust after seeing real answer
  quality.
- **`web_search_max_results`**: default 3, adjust after seeing real
  answer quality/cost.
