import json

from langchain_core.messages import AIMessage, ToolMessage

from agent.results import classify_answer_source, extract_sources


def _search_papers_message(*items: dict) -> ToolMessage:
    return ToolMessage(content=json.dumps(list(items)), name="search_papers", tool_call_id="c1")


def _web_message(*results: dict) -> ToolMessage:
    return ToolMessage(content=json.dumps({"results": list(results)}), name="tavily_search", tool_call_id="c2")


PAPER = {"title": "A Paper", "source_url": "http://arxiv.org/abs/2301.12345", "arxiv_id": "2301.12345"}


def test_extract_sources_reads_corpus_results():
    sources = extract_sources([_search_papers_message(PAPER)])

    assert sources == [
        {"kind": "corpus", "title": "A Paper", "url": "http://arxiv.org/abs/2301.12345", "arxiv_id": "2301.12345"}
    ]


def test_extract_sources_reads_web_results():
    sources = extract_sources([_web_message({"title": "A Blog", "url": "https://example.com/post"})])

    assert sources == [{"kind": "web", "title": "A Blog", "url": "https://example.com/post"}]


def test_extract_sources_dedupes_by_url_across_repeated_tool_calls():
    sources = extract_sources([_search_papers_message(PAPER), _search_papers_message(PAPER)])

    assert len(sources) == 1


def test_extract_sources_ignores_non_tool_messages_and_unparseable_content():
    messages = [
        AIMessage(content="thinking"),
        ToolMessage(content="not json", name="search_papers", tool_call_id="c3"),
    ]

    assert extract_sources(messages) == []


def test_classify_prefers_web_when_both_tools_fired():
    badge = classify_answer_source([_search_papers_message(PAPER), _web_message()])

    assert badge["kind"] == "web"


def test_classify_corpus_when_only_search_papers_fired():
    assert classify_answer_source([_search_papers_message(PAPER)])["kind"] == "corpus"


def test_classify_none_when_no_tool_fired():
    assert classify_answer_source([AIMessage(content="out of scope")])["kind"] == "none"
