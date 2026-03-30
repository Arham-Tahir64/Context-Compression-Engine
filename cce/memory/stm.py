from __future__ import annotations

from collections import deque

import numpy as np

from cce.memory.base import MemoryStore
from cce.models.types import MemoryRecord


class ShortTermMemory(MemoryStore):
    """In-process ring buffer for the most recent N turns.

    Verbatim — no compression. Lost on process restart by design;
    the client always resends recent context on the next request.
    One instance per project, keyed by project_id in the MemoryManager.
    """

    def __init__(self, max_turns: int = 20):
        self._max_turns = max_turns
        self._buffer: deque[MemoryRecord] = deque(maxlen=max_turns)

    async def write(self, record: MemoryRecord) -> None:
        self._buffer.append(record)

    async def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[MemoryRecord]:
        """Return all STM records newest-first — no vector search needed.

        STM is always small enough to return in full; the assembler
        handles budget trimming.
        """
        records = list(self._buffer)
        records.reverse()  # newest first
        return records[:top_k]

    async def all(self) -> list[MemoryRecord]:
        """Return all records newest-first (used by assembler)."""
        records = list(self._buffer)
        records.reverse()
        return records

    async def evict(self) -> int:
        # deque(maxlen=N) auto-evicts; nothing to do explicitly
        return 0

    async def clear(self) -> int:
        n = len(self._buffer)
        self._buffer.clear()
        return n

    async def count(self) -> int:
        return len(self._buffer)

    async def token_estimate(self) -> int:
        return sum(record.compressed_token_count for record in self._buffer)

    async def oldest_record_timestamp(self) -> float | None:
        if not self._buffer:
            return None
        return min(record.created_at for record in self._buffer)
