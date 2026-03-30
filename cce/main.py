from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cce.dependencies import get_settings

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Phase 0: nothing to warm up yet.
    # Phase 2+: start compression background worker here.
    yield
    # Shutdown: flush FAISS indices, close DB connections.


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
