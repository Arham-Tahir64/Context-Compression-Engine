from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from cce.compression.compressor import Compressor
from cce.compression.queue import CompressionQueue
from cce.embeddings.base import EmbeddingProvider
from cce.memory.ltm import LongTermMemory
from cce.memory.stm import ShortTermMemory
from cce.memory.wm import WorkingMemory
from cce.pipeline.router import MemoryRouter
from cce.settings import Settings
from cce.storage.db import Database
from cce.storage.faiss_store import FaissStore

logger = logging.getLogger(__name__)


@dataclass
class ProjectMemory:
    stm: ShortTermMemory
    wm: WorkingMemory
    ltm: LongTermMemory
    router: MemoryRouter
    queue: CompressionQueue
    worker_task: asyncio.Task[None] | None = None


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
        else:
            self._ensure_worker(project_id)
        return self._projects[project_id]

    async def shutdown(self) -> None:
        for mem in self._projects.values():
            await mem.queue.stop()

        tasks = [mem.worker_task for mem in self._projects.values() if mem.worker_task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        await self.close_all()

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

        # Integrity check: warn if FAISS and SQLite counts diverge
        ltm_count_row = await db.fetchone(
            "SELECT COUNT(*) FROM ltm_records WHERE project_id = ?", (project_id,)
        )
        ltm_count = ltm_count_row[0] if ltm_count_row else 0
        ok, msg = faiss.verify_integrity(ltm_count)
        if not ok:
            logger.warning("FAISS integrity check failed for project %s: %s", project_id, msg)

        stm = ShortTermMemory(max_turns=s.stm_max_turns)
        wm = WorkingMemory(db, project_id, max_records=s.wm_max_records)
        ltm = LongTermMemory(db, faiss, project_id, max_records=s.ltm_max_records)
        router = MemoryRouter(stm, wm, ltm, s, db=db)
        compressor = Compressor(s, wm, ltm)
        queue = CompressionQueue(compressor, maxsize=s.compression_queue_maxsize)

        self._projects[project_id] = ProjectMemory(
            stm=stm, wm=wm, ltm=ltm, router=router, queue=queue
        )
        self._ensure_worker(project_id)

    def _ensure_worker(self, project_id: str) -> None:
        mem = self._projects[project_id]
        if mem.worker_task is not None and not mem.worker_task.done():
            return

        task = asyncio.create_task(
            mem.queue.drain_worker(),
            name=f"cce-compression-worker-{project_id}",
        )
        task.add_done_callback(lambda t, pid=project_id: self._on_worker_done(pid, t))
        mem.worker_task = task

    def _on_worker_done(self, project_id: str, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return

        try:
            exc = task.exception()
        except Exception:
            logger.exception("Failed to inspect compression worker state for project %s", project_id)
            return

        if exc is not None:
            logger.warning(
                "Compression worker for project %s stopped unexpectedly: %s. "
                "It will be restarted on the next request.",
                project_id,
                exc,
            )
