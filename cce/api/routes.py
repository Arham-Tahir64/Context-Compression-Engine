from __future__ import annotations

import time
import uuid

from fastapi import APIRouter

from cce.api.schemas import (
    CompressRequest, CompressResponse,
    RecallRequest, RecallResponse,
    HealthResponse,
    ProjectStatsResponse,
    ClearMemoryResponse,
    LatencyBreakdown, MemoryHits, MemorySources,
)
from cce.dependencies import SettingsDep

router = APIRouter()

_START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep):
    return HealthResponse(
        status="ok",
        version="0.1.0",
        lm_studio_reachable=False,  # Phase 3: probe LM Studio
        projects_loaded=0,           # Phase 2: count from memory manager
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.post("/compress", response_model=CompressResponse)
async def compress(request: CompressRequest, settings: SettingsDep):
    # Phase 1+: wire to pipeline. Stub returns identity (passthrough) for now.
    t0 = time.time()
    request_id = request.metadata.request_id or str(uuid.uuid4())
    original = request.current_message
    elapsed = (time.time() - t0) * 1000

    return CompressResponse(
        request_id=request_id,
        project_id="__stub__",
        optimized_prompt=original,
        original_token_estimate=_estimate_tokens(original),
        compressed_token_estimate=_estimate_tokens(original),
        compression_ratio=1.0,
        memory_hits=MemoryHits(stm=0, wm=0, ltm=0),
        latency_ms=LatencyBreakdown(total_ms=elapsed, retrieval_ms=0.0, assembly_ms=0.0),
        warnings=["pipeline not yet wired — passthrough mode"],
    )


@router.post("/recall", response_model=RecallResponse)
async def recall(request: RecallRequest, settings: SettingsDep):
    # Phase 2+: wire to memory manager + assembler.
    t0 = time.time()
    request_id = str(uuid.uuid4())
    elapsed = (time.time() - t0) * 1000

    return RecallResponse(
        request_id=request_id,
        project_id="__stub__",
        briefing="[memory not yet loaded — pipeline not wired]",
        token_estimate=0,
        memory_sources=MemorySources(stm_records=0, wm_records=0, ltm_records=0),
        latency_ms=elapsed,
    )


@router.get("/project/{project_id}/stats", response_model=ProjectStatsResponse)
async def project_stats(project_id: str, settings: SettingsDep):
    return ProjectStatsResponse(
        project_id=project_id,
        stm_records=0,
        wm_records=0,
        ltm_records=0,
        total_token_estimate=0,
        faiss_index_size=0,
        oldest_record_ts=None,
    )


@router.delete("/project/{project_id}/memory", response_model=ClearMemoryResponse)
async def clear_memory(project_id: str, settings: SettingsDep):
    # Phase 2+: delegate to memory manager.
    return ClearMemoryResponse(cleared=True, records_deleted=0)


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)
