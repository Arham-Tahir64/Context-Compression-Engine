from __future__ import annotations

import numpy as np
import httpx

from cce.embeddings.base import EmbeddingProvider

_MODEL_NAME = "nomic-embed-text"
_DIMENSION = 768


class LMStudioEmbeddingProvider(EmbeddingProvider):
    """Swap-in embedding provider using nomic-embed-text via LM Studio.

    Requires LM Studio to be running with an embeddings-capable model loaded.
    Activate by setting EMBEDDING_PROVIDER=lm_studio in your .env.
    """

    def __init__(self, base_url: str, model: str = _MODEL_NAME, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def dimension(self) -> int:
        return _DIMENSION

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()

        vectors = [item["embedding"] for item in data["data"]]
        # Normalise to unit length for cosine similarity via dot product
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).tolist()

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]
