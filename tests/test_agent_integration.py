"""Integration tests against the real agent stack: OpenAI, Pinecone, Tavily.

Skipped by default (mirrors tests/test_arxiv_integration.py) since they
hit live external services, cost real (if tiny) money, and take longer
than the mocked unit tests. Run explicitly with:

    RUN_AGENT_INTEGRATION_TESTS=1 uv run pytest tests/test_agent_integration.py -v

These three cases were found to matter in practice: an early version of
the system prompt let the agent answer a genuinely out-of-scope question
(a cookie recipe) directly from general knowledge, calling no tool at
all - exactly the failure mode docs/agent-plan.md flagged as "hard to
guarantee from the system prompt alone." Fixed by making the scope check
an explicit first step in the prompt; these tests guard against regressing
that fix silently.
"""
import os

import pytest

from agent.config import AgentConfig
from agent.graph import build_agent

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENT_INTEGRATION_TESTS") != "1",
    reason="Hits real OpenAI/Pinecone/Tavily APIs; set RUN_AGENT_INTEGRATION_TESTS=1 to run.",
)


def _ask(agent, question: str, thread_id: str) -> dict:
    return agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )


def _tool_names_used(result: dict) -> list[str]:
    return [m.name for m in result["messages"] if getattr(m, "type", None) == "tool"]


def test_in_corpus_question_answers_from_search_papers_only():
    agent = build_agent(AgentConfig())
    result = _ask(
        agent,
        "What techniques exist for automated program repair using large language models?",
        "integration-test-corpus-hit",
    )

    tools_used = _tool_names_used(result)
    assert "search_papers" in tools_used
    assert "tavily_search" not in tools_used

    answer = result["messages"][-1].content.lower()
    assert "arxiv" in answer or "corpus" in answer or "paper" in answer


def test_in_scope_question_not_in_corpus_falls_back_to_web_search():
    agent = build_agent(AgentConfig())
    result = _ask(
        agent,
        "What is the most recently released OpenAI model as of today?",
        "integration-test-web-fallback",
    )

    tools_used = _tool_names_used(result)
    assert "search_papers" in tools_used
    assert "tavily_search" in tools_used


def test_out_of_scope_question_is_declined_without_calling_any_tool():
    agent = build_agent(AgentConfig())
    result = _ask(
        agent,
        "What's a good recipe for chocolate chip cookies?",
        "integration-test-out-of-scope",
    )

    assert _tool_names_used(result) == []

    answer = result["messages"][-1].content.lower()
    assert "chocolate" not in answer and "flour" not in answer  # didn't just answer it
    assert "scope" in answer or "ai" in answer or "software" in answer
