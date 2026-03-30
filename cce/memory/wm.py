from __future__ import annotations

import json
import time

import numpy as np

from cce.memory.base import MemoryStore
from cce.models.types import MemoryRecord, MemoryTier
from cce.storage.db import Database, decode_embedding, decode_json, encode_embedding, encode_json


class WorkingMemory(MemoryStore):
    """Top-K scored chunks, lightly compressed, persisted in SQLite.

    Capacity is enforced by evicting the lowest-scored record whenever
    a new write would exceed max_records.
    Query re-ranks all WM records by cosine similarity to the query
    embedding using numpy (50 vectors — no FAISS needed).
    """

    def __init__(self, db: Database, project_id: str, max_records: int = 50):
        self._db = db
        self._project_id = project_id
        self._max_records = max_records

    async def write(self, record: MemoryRecord) -> None:
        current = await self.count()
        if current >= self._max_records:
            await self._evict_lowest()

        async with self._db.transaction():
            await self._db.execute(
                """
                INSERT INTO wm_records (
                    record_id, project_id, content,
                    original_token_count, compressed_token_count,
                    source_chunk_ids, embedding, importance_score,
                    created_at, last_accessed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    content = excluded.content,
                    compressed_token_count = excluded.compressed_token_count,
                    importance_score = excluded.importance_score,
                    last_accessed_at = excluded.last_accessed_at,
                    metadata = excluded.metadata
                """,
                (
                    record.record_id,
                    self._project_id,
                    record.content,
                    record.original_token_count,
                    record.compressed_token_count,
                    encode_json(record.source_chunk_ids),
                    encode_embedding(record.embedding),
                    record.importance_score,
                    record.created_at,
                    record.last_accessed_at,
                    encode_json(record.metadata),
                ),
            )

    async def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[MemoryRecord]:
        rows = await self._db.fetchall(
            "SELECT * FROM wm_records WHERE project_id = ?",
            (self._project_id,),
        )
        if not rows:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        scored: list[tuple[float, MemoryRecord]] = []

        for row in rows:
            record = self._row_to_record(row)
            c_vec = np.array(record.embedding, dtype=np.float32)
            score = float(np.dot(c_vec, q_vec))
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [r for _, r in scored[:top_k]]

        # Update last_accessed_at for retrieved records
        now = time.time()
        ids = [r.record_id for r in results]
        if ids:
            placeholders = ",".join("?" * len(ids))
            await self._db.execute(
                f"UPDATE wm_records SET last_accessed_at = ? WHERE record_id IN ({placeholders})",
                (now, *ids),
            )
            await self._db.execute("SELECT 1")  # flush

        return results

    async def evict(self) -> int:
        current = await self.count()
        if current <= self._max_records:
            return 0
        excess = current - self._max_records
        await self._evict_lowest(n=excess)
        return excess

    async def clear(self) -> int:
        n = await self.count()
        async with self._db.transaction():
            await self._db.execute(
                "DELETE FROM wm_records WHERE project_id = ?",
                (self._project_id,),
            )
        return n

    async def count(self) -> int:
        row = await self._db.fetchone(
            "SELECT COUNT(*) FROM wm_records WHERE project_id = ?",
            (self._project_id,),
        )
        return row[0] if row else 0

    # ------------------------------------------------------------------

    async def _evict_lowest(self, n: int = 1) -> None:
        async with self._db.transaction():
            await self._db.execute(
                """
                DELETE FROM wm_records
                WHERE record_id IN (
                    SELECT record_id FROM wm_records
                    WHERE project_id = ?
                    ORDER BY importance_score ASC
                    LIMIT ?
                )
                """,
                (self._project_id, n),
            )

    def _row_to_record(self, row) -> MemoryRecord:
        return MemoryRecord(
            record_id=row["record_id"],
            project_id=row["project_id"],
            content=row["content"],
            original_token_count=row["original_token_count"],
            compressed_token_count=row["compressed_token_count"],
            tier=MemoryTier.WM,
            source_chunk_ids=decode_json(row["source_chunk_ids"]),
            embedding=decode_embedding(row["embedding"]),
            importance_score=row["importance_score"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            metadata=decode_json(row["metadata"]),
        )
