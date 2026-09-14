# AI Code Assistant — RAG over arXiv (AI / Software Development)

Learning project to get hands-on with LangChain, RAG, and agents by building an
assistant that answers questions about AI and software development, grounded in
arXiv papers (2020–present).

## Idea

1. Pull paper metadata (and eventually full text) from arXiv for AI / software
   development categories since 2020.
2. Chunk and embed the content, upsert into Pinecone as the knowledge base.
3. Serve an API where a user asks a question; if it's in-scope, we retrieve
   from Pinecone as ground truth and answer with an LLM, citing sources.

Two independent systems, built and tested separately:

- **Offline ingestion pipeline** — fetch → parse → chunk → embed → upsert.
- **Online serving API** — query → retrieve → generate.

## Decisions so far

| Question | Decision |
|---|---|
| Scraping vs. arXiv API | Use the official **arXiv API** — no HTML scraping. Clean metadata (title, abstract, authors, categories) for free; full text still needs PDF parsing. |
| MVP scope | Start with **abstracts only**, narrower category filter (e.g. `cs.SE` + a slice of `cs.AI`/`cs.LG`) rather than the full corpus. Expand to full-text chunking later. |
| Relevance gating | No separate classifier — retrieve, then let the **LLM decide** post-retrieval whether context is sufficient ("I don't have grounded info on this"). LLM choice still open. |
| Pipeline structure | Split ingestion (offline) from serving (online) from day one. |
| Chunking strategy | **Open** — decide via benchmarks (see below) rather than upfront. |

## Open decisions

- **LLM for generation / relevance gating** — not yet chosen.
- **Chunking strategy** — to be settled empirically (see Benchmarking below).

## Ingestion pipeline (offline)

1. **Fetch metadata** — query the arXiv API for `cs.SE`, `cs.AI`, `cs.LG` (start
   narrow, expand later), filtered to submission date ≥ 2020-01-01. Store raw
   metadata (id, title, abstract, authors, categories, published date, PDF URL).
2. **(Later) Fetch & parse full text** — download PDFs for papers that pass the
   MVP scope, extract text with PyMuPDF or GROBID. Deferred until abstract-only
   pipeline works end-to-end.
3. **Chunk** — split abstracts (and later, full text) into embeddable units.
   Strategy TBD (see Benchmarking).
4. **Embed** — generate vector embeddings for each chunk.
5. **Upsert to Pinecone** — store vectors with metadata needed for citations:
   paper id, title, authors, section (once full-text), source URL.
6. **Idempotency** — re-running ingestion should not duplicate vectors; key by
   paper id (+ chunk index) so re-ingestion upserts in place.

## Serving API (online, agentic)

1. User submits a question via the API.
2. Embed the query, retrieve top-k chunks from Pinecone.
3. LLM evaluates retrieved context:
   - Sufficient → answer grounded in the retrieved chunks, with citations
     (paper title + link).
   - Insufficient / out of scope → respond that there's no grounded info,
     rather than hallucinating.
4. Return answer + sources to the user.
5. This is the first "agentic" decision point (accept/reject retrieved
   context); more agent behavior (e.g. query rewriting, multi-hop retrieval)
   can be layered on once the basic RAG loop works.

## Benchmarking chunking strategies

Before committing to one approach, test candidates against a small labeled set
of questions + expected source papers, comparing retrieval precision/recall
and answer quality:

- Abstract-only chunks (one chunk per paper).
- Fixed-size token windows with overlap, over full text.
- Section-aware chunking (intro/method/results/etc.) once full-text parsing
  is in place.

Track: retrieval hit rate, chunk count / index size, embedding cost, and
qualitative answer grounding.

## Tech stack

- **LangChain** — orchestration, retrieval chain, agent logic.
- **arXiv API** — paper metadata source.
- **PyMuPDF / GROBID** — PDF → text parsing (once full-text is in scope).
- **Pinecone** — vector store.
- **LLM** — provider/model TBD.

## Roadmap

1. Ingestion MVP: arXiv API fetch (metadata only, narrow categories) → embed
   abstracts → upsert to Pinecone.
2. Serving MVP: query → retrieve → LLM answer with citations, or "not grounded"
   fallback.
3. Chunking benchmarks: compare strategies once full-text parsing is added.
4. Expand category/date coverage based on MVP learnings.
5. Layer in agent behavior beyond the basic accept/reject retrieval gate.
