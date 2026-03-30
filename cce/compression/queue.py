from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from cce.models.types import CompressionJob

if TYPE_CHECKING:
    from cce.compression.compressor import Compressor

logger = logging.getLogger(__name__)


class CompressionQueue:
    """Async queue that decouples compression from the request critical path.

    The drain_worker coroutine runs as a background asyncio task (started in
    the FastAPI lifespan). The main request handler enqueues jobs and returns
    immediately — compression never blocks an LLM call.
    """

    def __init__(self, compressor: "Compressor", maxsize: int = 500):
        self._compressor = compressor
        self._queue: asyncio.Queue[CompressionJob] = asyncio.Queue(maxsize=maxsize)
        self._running = False

    async def enqueue(self, job: CompressionJob) -> None:
        """Add a job. Drops silently if queue is full to protect latency."""
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.warning(
                "Compression queue full (%d items) — dropping job %s",
                self._queue.maxsize,
                job.job_id,
            )

    async def drain_worker(self) -> None:
        """Background coroutine: process jobs one at a time until stopped."""
        self._running = True
        logger.info("Compression worker started")
        while self._running:
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                await self._compressor.compress(job)
            except Exception:
                logger.exception("Compression job %s failed", job.job_id)
            finally:
                self._queue.task_done()

    async def stop(self) -> None:
        """Signal the worker to stop after finishing the current job."""
        self._running = False

    @property
    def qsize(self) -> int:
        return self._queue.qsize()
