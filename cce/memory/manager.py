from __future__ import annotations

import asyncio
import time
from typing import NamedTuple

from cce.compression.compressor import Compressor
from cce.compression.queue import CompressionQueue
from cce.embeddings.base import EmbeddingProvider
from cce.memory.ltm import LongTermMemory
from cce.memory.stm import ShortTermMemory
from cce.memory.wm import WorkingMemory
from cce.models.types import CompressionJob, CompressionLevel, MemoryTier
from cce.pipeline.router import MemoryRouter
from cce.settings import Settings
from cce.storage.db import Database
from cce.storage.faiss_store import FaissStore


class ProjectMemory(NamedTuple):
    stm: ShortTermMemory
    wm: WorkingMemory
    ltm: LongTermMemory
    router: MemoryRouter
    queue: CompressionQueue


class MemoryManager:
    """Creates and owns per-project memory tier instances.

    Lazily initialises a full memory stack (DB, FAISS, STM, WM, LTM,
    Router, CompressionQueue) on first access for a given project_id.
    Also registers each project in the SQLite projects table.
    """

    def __init__(self, settings: Settings, embedding_provider: EmbeddingProvider):
        self._settings = settings
        self._embedding_provider = embedding_provider
        self._projects: dict[str, ProjectMemory] = {}
        self._dbs: dict[str, Database] = {}
        self._lock = asyncio.Lock()

    async def get(self, project_id: str) -> ProjectMemory:
        if project_id not in self._projects:
            async with self._lock:
                if project_id not in self._projects:
                    await self._init_project(project_id)
        return self._projects[project_id]

    async def close_all(self) -> None:
        for db in self._dbs.values():
            await db.close()
        self._projects.clear()
        self._dbs.clear()

    async def project_ids(self) -> list[str]:
        return list(self._projects.keys())

    # ------------------------------------------------------------------

    async def _init_project(self, project_id: str) -> None:
        s = self._settings

        db = Database(s.db_path(project_id))
        await db.connect()
        self._dbs[project_id] = db

        # Register project (idempotent)
        await db.execute(
            """
            INSERT OR IGNORE INTO projects
                (project_id, display_name, embedding_model, embedding_dim,
                 created_at, last_accessed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                project_id,
                self._embedding_provider.model_name,
                self._embedding_provider.dimension,
                time.time(),
                time.time(),
            ),
        )
        await db._conn.commit()

        faiss = FaissStore(s.faiss_index_path(project_id), self._embedding_provider.dimension)
        faiss.load()

        stm = ShortTermMemory(max_turns=s.stm_max_turns)
        wm = WorkingMemory(db, project_id, max_records=s.wm_max_records)
        ltm = LongTermMemory(db, faiss, project_id, max_records=s.ltm_max_records)
        router = MemoryRouter(stm, wm, ltm, s)
        compressor = Compressor(s, wm, ltm)
        queue = CompressionQueue(compressor, maxsize=s.compression_queue_maxsize)

        self._projects[project_id] = ProjectMemory(
            stm=stm, wm=wm, ltm=ltm, router=router, queue=queue
        )
