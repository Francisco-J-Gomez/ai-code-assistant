# RAG Agent — Step-by-Step Plan

Scope: a conversational agent, exposed as a chat UI, that answers questions
about AI and software development grounded in the Pinecone corpus the
ingestion pipeline built (~30k arXiv abstracts). This is the "Serving API"
half of the project (see [README.md](../README.md) "Serving API
(online, agentic)"), picking up where `docs/data-ingestion-plan.md` left off.

## Design principles

- **Grounded first, honest about gaps** — always search the corpus before
  answering a factual question; if retrieval doesn't return anything
  relevant, say so rather than hallucinate (already decided in the
  README's "Decisions so far" table).
- **Real tool-calling agent, not a fixed chain** — the LLM decides whether
  and when to call the retrieval tool, via LangChain's agent framework.
  This is the actual "agents" learning goal of the project, not just RAG.
- **Reuse the ingestion code, don't fork it** — the retrieval tool must
  embed queries with the *same* model/dimension used to build the index
  (`text-embedding-3-small`, 1536-dim). Reusing `ingestion.embeddings.Embedder`
  and `ingestion.pinecone_client.PineconeStore` directly guarantees that by
  construction, instead of a second implementation that could drift.
- **Start narrow**: one tool (corpus search), one model, no memory
  persistence beyond a chat session. Web search, alternate models, and
  durable multi-session memory are explicitly deferred (see Open decisions
  and TODO.md).

## High-level flow

```
User question (Streamlit chat)
  -> LangChain agent (langchain.agents.create_agent, OpenAI LLM)
       -> tool: search_papers(query) -> embeds query, queries Pinecone top-k
       -> LLM synthesizes an answer from returned chunks, with citations
       -> if nothing relevant came back: says so explicitly
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
- **`langchain`, `langgraph`, `langchain-openai`, `langchain-tavily`,
  `streamlit`** all install cleanly (no numpy dependency).

## Steps

### 1. Config

New `AgentConfig` dataclass (mirrors `IngestionConfig`/`EmbeddingConfig`):
`llm_model` (default `gpt-5-mini`), `llm_temperature` (default 0,
factual/grounded answers over creative ones), `retrieval_top_k` (default
5), `embedding_model`/`embedding_dimensions` (must match the ingestion
side — reuse `EmbeddingConfig`'s defaults rather than duplicating),
`pinecone_index_name`.

### 2. Retrieval tool

- `search_papers(query: str) -> list[dict]`: embeds `query` with
  `Embedder`, queries `PineconeStore`'s underlying index for top-k
  matches, returns `[{title, authors, published, source_url,
  text, score}, ...]`.
- Wrapped as a LangChain tool via `@tool` (from `langchain_core.tools`)
  with a docstring the LLM uses to decide when to call it — needs to make
  clear it searches *this specific* AI/software-engineering arXiv corpus,
  not the general web.
- Unit tested with mocked `Embedder`/`PineconeStore` (same mocking pattern
  as `tests/test_pipeline_embed.py`), plus one opt-in real-API test
  (asks a known in-corpus question, checks an expected paper id comes
  back) mirroring `tests/test_arxiv_integration.py`.

### 3. Agent

- `langchain.agents.create_agent(model=ChatOpenAI(...), tools=[search_papers],
  system_prompt=..., checkpointer=InMemorySaver())`.
- System prompt instructs: always search before answering questions about
  AI/software-development research; answer only from what the tool
  returns; cite paper titles + links; if nothing relevant comes back, say
  plainly that the corpus doesn't cover it rather than guessing (open
  question below: whether to still offer a clearly-labeled non-grounded
  general answer as a fallback, or refuse outright).
- `InMemorySaver` checkpointer for MVP conversation memory — process-local,
  lost on restart. Fine for a single-user local chat session; durable
  memory is a later concern if this ever needs multiple users/sessions.

### 4. Citations

Pull citation info from the tool's *structured return value*
(title/authors/source_url), not by asking the LLM to format links in
free text — more reliable, and lets the UI render a proper "Sources"
list from the agent's intermediate tool-call results rather than parsing
prose.

### 5. Chat UI (Streamlit)

- `ui/chat_app.py`, run via `streamlit run ui/chat_app.py`.
- `st.chat_message`/`st.chat_input` for the conversation.
- The compiled agent graph and a per-session `thread_id` live in
  `st.session_state` — Streamlit reruns the whole script on every
  interaction, so the agent must be created once and reused, not rebuilt
  per message.
- A "Sources" expander under each answer, populated from that turn's tool
  call results.

### 6. Observability

`LANGSMITH_TRACING`/`LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` are already in
`.env` (added earlier, unused until now) — LangChain auto-instruments
when these are set, no code changes needed. Verify traces actually show
up in the LangSmith UI after the first real run.

### 7. Testing

- Unit tests: retrieval tool (mocked), config defaults.
- Opt-in integration test: real OpenAI + real Pinecone, ask a question
  whose answer is known to come from a specific paper in the corpus,
  assert that paper's id appears in the tool-call results and/or the
  answer.
- Manual: run the Streamlit app, ask a handful of real in-scope and
  out-of-scope questions, check citations render and the "not grounded"
  path is honest rather than a hallucinated answer.

## Project layout

```
ai-code-assistant/
  agent/
    __init__.py
    config.py            # AgentConfig
    retrieval_tool.py     # search_papers tool (reuses ingestion.embeddings/pinecone_client)
    graph.py               # create_agent(...) wiring + system prompt
  ui/
    chat_app.py             # Streamlit chat interface
  tests/
    test_retrieval_tool.py
    test_agent_integration.py   # opt-in, real APIs
```

## Implementation order

1. `agent/config.py` + `agent/retrieval_tool.py` — verify `search_papers`
   against the real index with a couple of known-answer queries before
   wiring up the LLM at all.
2. `agent/graph.py` — the agent itself; verify with a plain Python REPL
   loop (no UI yet) asking a few questions.
3. `ui/chat_app.py` — Streamlit wrapper around the working agent.
4. LangSmith trace verification.
5. Unit tests + the opt-in integration test.
6. (Deferred, see TODO.md) Tavily web-search tool for in-scope-but-
   not-in-corpus questions.

## Open decisions to confirm before implementation

- **OpenAI model**: `gpt-5-mini` (recommended default — good
  quality/cost balance) vs. `gpt-5` (best quality, pricier) vs.
  `gpt-5-nano` (cheapest/fastest) — all three confirmed working with
  tool-calling.
- **Out-of-corpus fallback behavior**: refuse outright ("I don't have
  grounded info on this") vs. offer a clearly-labeled general-knowledge
  answer anyway. The README's existing decision leans toward the former,
  but worth confirming for chat UX specifically.
- **`retrieval_top_k`**: default 5, adjust after seeing real answer
  quality.
