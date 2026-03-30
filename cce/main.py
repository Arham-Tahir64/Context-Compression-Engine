from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cce.dependencies import get_memory_manager, get_settings

logger = logging.getLogger(__name__)
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Context Compression Engine starting on %s:%d", settings.host, settings.port)

    # Pre-load the memory manager (creates data dir, etc.)
    get_memory_manager()

    yield

    # Shutdown: stop all per-project compression workers and close DBs
    manager = get_memory_manager()
    for project_id in await manager.project_ids():
        mem = await manager.get(project_id)
        await mem.queue.stop()

    await manager.close_all()
    logger.info("Context Compression Engine shut down cleanly")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Context Compression Engine",
        version="0.1.0",
        description="Local sidecar for LLM context compression and memory management",
        lifespan=lifespan,
    )

    from cce.api.routes import router
    from cce.api.errors import CCEError, cce_error_handler, generic_error_handler
    app.include_router(router)
    app.add_exception_handler(CCEError, cce_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("cce.main:app", host=settings.host, port=settings.port, reload=True)
