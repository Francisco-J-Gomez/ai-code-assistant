import json


def extract_sources(new_messages: list) -> list[dict]:
    """Collect citations from this turn's tool results, de-duplicated by URL.

    `new_messages` is the slice of the agent's messages added during the
    current turn (the full thread history is cumulative). De-duplicating by
    url matters in practice: the agent sometimes calls search_papers more
    than once per turn (e.g. retrying with a refined query), which can
    surface the same paper twice.
    """
    sources = []
    for message in new_messages:
        if getattr(message, "type", None) != "tool":
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            continue

        if message.name == "search_papers":
            for item in payload:
                sources.append(
                    {
                        "kind": "corpus",
                        "title": item.get("title"),
                        "url": item.get("source_url"),
                        "arxiv_id": item.get("arxiv_id"),
                    }
                )
        elif message.name == "tavily_search":
            for item in payload.get("results", []):
                sources.append({"kind": "web", "title": item.get("title"), "url": item.get("url")})

    seen_urls = set()
    deduped = []
    for source in sources:
        if source["url"] in seen_urls:
            continue
        seen_urls.add(source["url"])
        deduped.append(source)
    return deduped


def classify_answer_source(new_messages: list) -> dict:
    """Which tool actually fired this turn, not what the answer's prose
    claims - search_papers always returns its top-k regardless of whether
    the results are actually relevant, so "was a tool called" is the
    reliable signal, not "did it return something non-empty". If
    web_search fired at all, the agent judged the corpus insufficient and
    escalated, regardless of what search_papers happened to return.
    """
    tool_names = {getattr(m, "name", None) for m in new_messages if getattr(m, "type", None) == "tool"}
    if "tavily_search" in tool_names:
        return {"icon": "🌐", "label": "Includes live web search", "kind": "web"}
    if "search_papers" in tool_names:
        return {"icon": "📄", "label": "Grounded in the paper corpus", "kind": "corpus"}
    return {"icon": "❌", "label": "Not grounded — no relevant source found", "kind": "none"}
