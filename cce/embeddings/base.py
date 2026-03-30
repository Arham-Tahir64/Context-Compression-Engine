from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers.

    All providers must produce L2-normalised vectors so cosine similarity
    can be computed as a plain dot product (required by FAISS IndexFlatIP).
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text."""
        ...

    @abstractmethod
    async def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for single-text embedding."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier, stored in project metadata."""
        ...
