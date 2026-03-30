from __future__ import annotations

from abc import ABC, abstractmethod

from cce.models.types import MemoryRecord


class MemoryStore(ABC):
    """Abstract interface for all memory tier implementations."""

    @abstractmethod
    async def write(self, record: MemoryRecord) -> None:
        """Persist a memory record to this tier."""
        ...

    @abstractmethod
    async def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[MemoryRecord]:
        """Retrieve the top-k most relevant records for a query embedding."""
        ...

    @abstractmethod
    async def evict(self) -> int:
        """Enforce capacity limits. Returns number of records evicted."""
        ...

    @abstractmethod
    async def clear(self) -> int:
        """Delete all records. Returns number of records deleted."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return number of records currently stored."""
        ...
