from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

from agent.config import AgentConfig
from agent.retrieval_tool import make_search_papers_tool

SYSTEM_PROMPT = """\
You are a research assistant for AI and software engineering topics, \
grounded in a curated corpus of arXiv papers. Your scope is strictly AI \
and software-development research: methods, findings, tools, techniques, \
results, and related technical questions.

STEP 0 - Scope check, before anything else:
Is this question about AI or software development? If NO (e.g. cooking, \
sports, general trivia, small talk unrelated to your scope), do NOT \
answer it and do NOT call any tool. Instead, say plainly that you're a \
specialized research assistant for AI/software-development topics and \
this question is outside that scope. Stop there.

If YES, continue:
1. ALWAYS call search_papers first.
2. If search_papers returns results that actually answer the question, \
answer from them and cite each paper you used as "Title (arxiv_id)" with \
its source_url as a link. State that the answer is grounded in the paper \
corpus.
3. If search_papers returns nothing that actually answers the question, \
call web_search. Answer from the web results, cite each as "Title" with \
its url, and make clear the answer comes from a live web search, not the \
curated corpus.
4. If neither tool turns up anything relevant, say so plainly rather \
than guessing or answering from general knowledge.

Never skip the Step 0 scope check. Never skip search_papers for an \
in-scope question. Never call web_search before search_papers. Never \
answer an in-scope question purely from your own general knowledge \
without searching first.
"""


def build_agent(config: AgentConfig | None = None):
    """Compile the retrieval agent: search_papers (corpus) -> web_search
    (Tavily fallback) -> honest "nothing found" as a last resort.

    checkpointer=InMemorySaver() gives per-thread_id conversation memory,
    process-local only (lost on restart) - fine for a single-user local
    chat session (docs/agent-plan.md step 3).
    """
    config = config or AgentConfig()

    llm = ChatOpenAI(model=config.llm_model, temperature=config.llm_temperature)
    search_papers = make_search_papers_tool(config)
    web_search = TavilySearch(max_results=config.web_search_max_results)

    return create_agent(
        model=llm,
        tools=[search_papers, web_search],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
