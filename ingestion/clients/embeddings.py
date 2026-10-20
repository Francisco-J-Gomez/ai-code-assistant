import os
from typing import Sequence

from openai import OpenAI


class Embedder:
    """Thin wrapper around the OpenAI embeddings API.

    Batching is the caller's responsibility (see pipeline.run_embed_stage) -
    this just embeds whatever list of texts it's given in one API call and
    returns vectors in the same order (guaranteed by the API).
    """

    def __init__(self, model: str, api_key: str | None = None):
        self._model = model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        return [item.embedding for item in response.data]
