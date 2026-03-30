from __future__ import annotations

import time

from cce.memory.base import MemoryStore
from cce.models.types import MemoryRecord, MemoryTier
from cce.storage.db import Database, decode_json, encode_json
from cce.storage.faiss_store import FaissStore


class LongTermMemory(MemoryStore):
    """Heavily compressed summaries stored in FAISS + SQLite.

    FAISS handles vector search; SQLite holds the full content and metadata.
    The FAISS index is loaded from disk on init and saved atomically on write.
    One instance per project.
    """

    def __init__(
        self,
        db: Database,
        faiss: FaissStore,
        project_id: str,
        max_records: int = 10_000,
    ):
        self._db = db
        self._faiss = faiss
        self._project_id = project_id
        self._max_records = max_records

    async def write(self, record: MemoryRecord) -> None:
        # Assign a FAISS integer ID by adding the vector
        faiss_ids = self._faiss.add([record.embedding])
        faiss_id = faiss_ids[0]

        async with self._db.transaction():
            await self._db.execute(
                """
                INSERT INTO ltm_records (
                    record_id, project_id, faiss_id, content,
                    original_token_count, compressed_token_count,
                    source_chunk_ids, importance_score,
                    created_at, last_accessed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    self._project_id,
                    faiss_id,
                    record.content,
                    record.original_token_count,
                    record.compressed_token_count,
                    encode_json(record.source_chunk_ids),
                    record.importance_score,
                    record.created_at,
                    record.last_accessed_at,
                    encode_json(record.metadata),
                ),
            )

        # Persist FAISS index atomically after every write
        self._faiss.save()

    async def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[MemoryRecord]:
        hits = self._faiss.search(query_embedding, top_k)
        if not hits:
            return []

        faiss_ids = [fid for fid, _ in hits]
        placeholders = ",".join("?" * len(faiss_ids))
        rows = await self._db.fetchall(
            f"""
            SELECT * FROM ltm_records
            WHERE project_id = ? AND faiss_id IN ({placeholders})
            """,
            (self._project_id, *faiss_ids),
        )

        # Preserve FAISS ranking order
        id_to_row = {row["faiss_id"]: row for row in rows}
        results: list[MemoryRecord] = []
        now = time.time()

        for fid, _ in hits:
            row = id_to_row.get(fid)
            if row:
                results.append(self._row_to_record(row))

        # Update last_accessed_at
        if results:
            record_ids = [r.record_id for r in results]
            placeholders2 = ",".join("?" * len(record_ids))
            await self._db.execute(
                f"UPDATE ltm_records SET last_accessed_at = ? WHERE record_id IN ({placeholders2})",
                (now, *record_ids),
            )

        return results

    async def evict(self) -> int:
        # LTM eviction: remove oldest low-importance records beyond capacity
        current = await self.count()
        if current <= self._max_records:
            return 0
        excess = current - self._max_records
        rows = await self._db.fetchall(
            """
            SELECT record_id FROM ltm_records
            WHERE project_id = ?
            ORDER BY importance_score ASC, last_accessed_at ASC
            LIMIT ?
            """,
            (self._project_id, excess),
        )
        if rows:
            ids = [r["record_id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            async with self._db.transaction():
                await self._db.execute(
                    f"DELETE FROM ltm_records WHERE record_id IN ({placeholders})",
                    tuple(ids),
                )
        return len(rows)

    async def clear(self) -> int:
        n = await self.count()
        async with self._db.transaction():
            await self._db.execute(
                "DELETE FROM ltm_records WHERE project_id = ?",
                (self._project_id,),
            )
        # Rebuild empty FAISS index
        from cce.storage.faiss_store import FaissStore
        self._faiss._index = None
        self._faiss.load()
        self._faiss.save()
        return n

    async def count(self) -> int:
        row = await self._db.fetchone(
            "SELECT COUNT(*) FROM ltm_records WHERE project_id = ?",
            (self._project_id,),
        )
        return row[0] if row else 0

    # ------------------------------------------------------------------

    def _row_to_record(self, row) -> MemoryRecord:
        # Embedding is not stored in SQLite for LTM — it lives in FAISS.
        # Return a placeholder; callers that need the vector use FAISS directly.
        return MemoryRecord(
            record_id=row["record_id"],
            project_id=row["project_id"],
            content=row["content"],
            original_token_count=row["original_token_count"],
            compressed_token_count=row["compressed_token_count"],
            tier=MemoryTier.LTM,
            source_chunk_ids=decode_json(row["source_chunk_ids"]),
            embedding=[],  # not stored in SQLite; lives in FAISS
            importance_score=row["importance_score"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            metadata=decode_json(row["metadata"]),
        )
