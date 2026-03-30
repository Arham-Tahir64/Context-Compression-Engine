from __future__ import annotations

import asyncio
from functools import lru_cache

import numpy as np

from cce.embeddings.base import EmbeddingProvider

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIMENSION = 384


@lru_cache(maxsize=1)
def _load_model():
    # Deferred import so the process starts fast; model loads once and is cached.
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(_MODEL_NAME, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Embedding model 'all-MiniLM-L6-v2' is not available locally. "
            "Download it once in a connected environment or switch to a different "
            "embedding provider before starting the server."
        ) from exc


class SentenceTransformerProvider(EmbeddingProvider):
    """Default embedding provider using all-MiniLM-L6-v2.

    Runs synchronous encode() in a thread pool so it doesn't block the
    asyncio event loop.
    """

    @property
    def dimension(self) -> int:
        return _DIMENSION

    @property
    def model_name(self) -> str:
        return _MODEL_NAME

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        vectors: np.ndarray = await loop.run_in_executor(
            None, self._encode_sync, texts
        )
        return vectors.tolist()

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]

    async def check_ready(self) -> tuple[bool, str | None]:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _load_model)
        except Exception as exc:
            return False, str(exc)
        return True, None

    def _encode_sync(self, texts: list[str]) -> np.ndarray:
        model = _load_model()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.astype(np.float32)
