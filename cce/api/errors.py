from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class CCEError(Exception):
    """Base class for application errors returned as structured HTTP responses."""
    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class ProjectInitError(CCEError):
    status_code = 503
    error_code = "project_init_failed"


class EmbeddingError(CCEError):
    status_code = 503
    error_code = "embedding_failed"


async def cce_error_handler(request: Request, exc: CCEError) -> JSONResponse:
    logger.error("%s: %s", exc.error_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "detail": exc.detail},
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred."},
    )
