from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cce.dependencies import get_settings

logger = logging.getLogger(__name__)
_start_time = time.time()

# Module-level handle so routes can access the queue after startup
_compression_queue = None


def get_compression_queue():
    return _compression_queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _compression_queue

    settings = get_settings()

    # Lazy imports to keep startup fast when deps aren't all wired yet
    from cce.compression.compressor import Compressor
    from cce.compression.queue import CompressionQueue

    # Placeholder memory stores — Phase 4 wires real per-project instances.
    # For now, compressor is constructed but the queue runs idle until
    # the memory manager is initialized per request.
    _compression_queue = None  # will be set once MemoryManager is ready (Phase 4)

    logger.info("Context Compression Engine starting on %s:%d", settings.host, settings.port)

    yield

    logger.info("Context Compression Engine shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Context Compression Engine",
        version="0.1.0",
        description="Local sidecar for LLM context compression and memory management",
        lifespan=lifespan,
    )

    from cce.api.routes import router
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("cce.main:app", host=settings.host, port=settings.port, reload=True)
