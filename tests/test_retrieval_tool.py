import agent.retrieval_tool as retrieval_tool_module
from agent.config import AgentConfig
from agent.retrieval_tool import make_search_papers_tool


class _FakeEmbedder:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakePineconeStore:
    def __init__(self, index_name, dimension, cloud, region):
        self.index_name = index_name
        self.query_calls = []
        self.results_to_return = []

    def query(self, vector, top_k):
        self.query_calls.append({"vector": vector, "top_k": top_k})
        return self.results_to_return


def _patch(monkeypatch):
    monkeypatch.setattr(retrieval_tool_module, "Embedder", _FakeEmbedder)
    monkeypatch.setattr(retrieval_tool_module, "PineconeStore", _FakePineconeStore)


def test_tool_name_and_description(monkeypatch):
    _patch(monkeypatch)
    tool = make_search_papers_tool(AgentConfig())

    assert tool.name == "search_papers"
    assert "arxiv" in tool.description.lower()


def test_tool_returns_pinecone_results_unmodified(monkeypatch):
    captured = {}

    def embedder_factory(model):
        e = _FakeEmbedder(model)
        captured["embedder"] = e
        return e

    def store_factory(index_name, dimension, cloud, region):
        s = _FakePineconeStore(index_name, dimension, cloud, region)
        s.results_to_return = [
            {"chunk_id": "2301.12345:0", "score": 0.9, "title": "A Paper", "arxiv_id": "2301.12345"}
        ]
        captured["store"] = s
        return s

    monkeypatch.setattr(retrieval_tool_module, "Embedder", embedder_factory)
    monkeypatch.setattr(retrieval_tool_module, "PineconeStore", store_factory)

    config = AgentConfig(retrieval_top_k=7)
    tool = make_search_papers_tool(config)

    result = tool.invoke({"query": "program repair with LLMs"})

    assert result == [{"chunk_id": "2301.12345:0", "score": 0.9, "title": "A Paper", "arxiv_id": "2301.12345"}]
    assert captured["embedder"].calls == [["program repair with LLMs"]]
    assert captured["store"].query_calls == [{"vector": [0.1, 0.2, 0.3], "top_k": 7}]
    assert captured["store"].index_name == config.embedding.pinecone_index_name
