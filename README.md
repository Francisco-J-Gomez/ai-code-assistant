# AI Code Assistant — an agentic RAG assistant over arXiv

A research assistant for AI and software-development questions, grounded in
~30k arXiv papers (2020–present). Ask a question in a chat UI and an
LLM agent decides how to answer it, with sources attached to every answer.

Built as a learning project to get hands-on with LangChain, LangGraph, RAG,
and tool-using agents.

## What the agent does

For every question, a tool-calling agent picks one of three paths:

| Situation | What happens |
|---|---|
| **In scope, found in the paper corpus** | Retrieves the top matches from Pinecone and answers from them, citing each paper (title + arXiv link). |
| **In scope, not in the corpus** | Falls back to a live web search (Tavily) and says clearly that the answer came from the web, not the curated corpus. |
| **Out of scope** (cooking, trivia, …) | Declines politely without calling any tool. If nothing relevant turns up anywhere, it says so rather than guessing. |

Each answer in the chat UI has a collapsible "Sources" list, labeled by
whether it came from the corpus or the web.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it will fetch Python 3.12+ if needed).

```bash
uv sync                                  # create .venv, install locked deps
cp .env.example .env                     # then fill in your own keys
uv run streamlit run ui/chat_app.py      # open the chat UI
```

You need `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `TAVILY_API_KEY`. The app
queries an existing Pinecone index, so **populate it first** by running the
ingestion pipeline — see [ingestion/README.md](ingestion/README.md).

Run the tests (fast, mocked, no network):

```bash
uv run pytest -q
```

Opt-in tests against the real OpenAI / Pinecone / Tavily / arXiv APIs live in
[integration_tests/](integration_tests/) and are skipped unless their env flag
is set — see [agent/README.md](agent/README.md#tests).

## How it works

```mermaid
graph TD;
	__start__([<p>__start__</p>]):::first
	model(model)
	tools(tools)
	__end__([<p>__end__</p>]):::last
	__start__ --> model;
	model -.-> __end__;
	model -.-> tools;
	tools -.-> model;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

The LangGraph loop is simple: the `model` node (OpenAI `gpt-5-mini`) decides
what to do, the `tools` node runs `search_papers` (Pinecone) or `web_search`
(Tavily), and the loop repeats until the model stops asking for tools. The
routing logic — scope check, corpus first, web only as a fallback — is
**prompt-level behavior**, not graph edges. Details in
[agent/README.md](agent/README.md).

Two independent systems, built and tested separately:

- **Online agent** (`agent/`, `ui/`) — question → retrieve → answer with sources.
- **Offline ingestion** (`ingestion/`) — fetch → normalize → chunk → embed →
  upsert into Pinecone. See [ingestion/README.md](ingestion/README.md).

## Key decisions

| Question | Decision |
|---|---|
| Agent LLM | **OpenAI `gpt-5-mini`** — single-provider stack (same account as embeddings); tool-calling confirmed working. |
| Relevance gating | No separate classifier — the agent searches the corpus first and the **LLM decides** whether the results actually answer the question, falling back to web search if not. |
| Web search fallback | **Tavily** via `langchain-tavily`, a second agent tool, still scoped to AI/software-dev topics by the system prompt. |
| Knowledge base | Official **arXiv API** (no scraping), abstracts only for now, embedded with **`text-embedding-3-small`** and stored in **Pinecone**. |

Full design notes: [docs/agent-plan.md](docs/agent-plan.md) and
[docs/data-ingestion-plan.md](docs/data-ingestion-plan.md). Deferred work and
open questions (chunking benchmarks, alternative embedding models, …) are in
[TODO.md](TODO.md).

## Tech stack

- **LangChain / LangGraph** — agent, tools, conversation memory.
- **OpenAI** — chat model (`gpt-5-mini`) and embeddings (`text-embedding-3-small`).
- **Pinecone** — vector store.
- **Tavily** — live web search fallback.
- **Streamlit** — chat UI.
- **arXiv API** — paper metadata source.
- **uv** — environment and dependency management.

## Repo layout

| Path | What's in it |
|---|---|
| `agent/` | The LangGraph agent, its tools, and config. |
| `ui/` | Streamlit chat app. |
| `ingestion/` | Offline pipeline that builds the Pinecone corpus. |
| `scripts/` | Dev helper `inspect_db.py` for browsing the ingestion SQLite store. |
| `tests/` | Fast mocked unit tests. |
| `integration_tests/` | Opt-in tests against real external APIs. |
| `docs/` | Design plans for the agent and the ingestion pipeline. |
