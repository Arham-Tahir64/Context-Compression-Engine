from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from cce.memory.ltm import LongTermMemory
from cce.memory.stm import ShortTermMemory
from cce.memory.wm import WorkingMemory
from cce.models.types import MemoryRecord, MemoryTier, ScoredChunk
from cce.settings import Settings
from cce.storage.db import Database


@dataclass
class RoutingResult:
    stm: int = 0
    wm: int = 0
    ltm: int = 0
    discarded: int = 0
    record_ids_by_chunk_id: dict[str, str] = field(default_factory=dict)


class MemoryRouter:
    """Routes ScoredChunks into the appropriate memory tier.

    Routing thresholds (all configurable via Settings):
      score >= stm_threshold  → STM  (verbatim, high-signal)
      score >= wm_threshold   → WM   (light compression queued)
      score >= discard_threshold → LTM (heavy compression queued)
      score <  discard_threshold → DISCARD
    """

    def __init__(
        self,
        stm: ShortTermMemory,
        wm: WorkingMemory,
        ltm: LongTermMemory,
        settings: Settings,
        db: Database | None = None,
    ):
        self._stm = stm
        self._wm = wm
        self._ltm = ltm
        self._stm_threshold = settings.router_stm_threshold
        self._wm_threshold = settings.router_wm_threshold
        self._discard_threshold = settings.router_discard_threshold
        self._db = db

    async def route(self, scored_chunks: list[ScoredChunk]) -> RoutingResult:
        """Route each chunk to the appropriate tier based on composite score."""
        result = RoutingResult()

        for sc in scored_chunks:
            tier = self._classify(sc.composite_score)
            sc.chunk.importance_score = sc.composite_score

            if tier == MemoryTier.DISCARD:
                result.discarded += 1
                continue

            record = self._to_record(sc, tier)

            if tier == MemoryTier.STM:
                await self._stm.write(record)
                result.stm += 1
            elif tier == MemoryTier.WM:
                await self._wm.write(record)
                result.wm += 1
            elif tier == MemoryTier.LTM:
                await self._ltm.write(record)
                result.ltm += 1

            await self._log_chunk(sc, tier)
            result.record_ids_by_chunk_id[sc.chunk.chunk_id] = record.record_id

        return result

    def _classify(self, score: float) -> MemoryTier:
        if score >= self._stm_threshold:
            return MemoryTier.STM
        if score >= self._wm_threshold:
            return MemoryTier.WM
        if score >= self._discard_threshold:
            return MemoryTier.LTM
        return MemoryTier.DISCARD

    def _to_record(self, sc: ScoredChunk, tier: MemoryTier) -> MemoryRecord:
        now = time.time()
        return MemoryRecord(
            record_id=str(uuid.uuid4()),
            project_id=sc.chunk.project_id,
            content=sc.chunk.content,
            original_token_count=sc.chunk.token_count,
            compressed_token_count=sc.chunk.token_count,  # uncompressed at write time
            tier=tier,
            source_chunk_ids=[sc.chunk.chunk_id],
            embedding=sc.chunk.embedding or [],
            importance_score=sc.composite_score,
            created_at=sc.chunk.created_at,
            last_accessed_at=now,
            metadata=sc.chunk.metadata,
        )

    async def _log_chunk(self, sc: ScoredChunk, tier: MemoryTier) -> None:
        if self._db is None:
            return
        await self._db.log_chunk(
            chunk_id=sc.chunk.chunk_id,
            project_id=sc.chunk.project_id,
            content=sc.chunk.content,
            content_type=sc.chunk.content_type.value,
            token_count=sc.chunk.token_count,
            turn_index=sc.chunk.turn_index,
            created_at=sc.chunk.created_at,
            tier_assigned=tier.value,
            metadata=sc.chunk.metadata,
        )
