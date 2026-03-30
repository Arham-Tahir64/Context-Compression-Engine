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
from cce.api.errors import EmbeddingError, ProjectInitError
from cce.dependencies import SettingsDep, get_assembler, get_embedding_provider, get_memory_manager
from cce.identity.resolver import resolve_project_id
from cce.pipeline.assembler import _GENERIC_RECALL_QUERY

router = APIRouter()
_START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep):
    from cce.compression.compressor import Compressor
    compressor = Compressor(settings, None, None)  # type: ignore[arg-type]
    lm_reachable = await compressor.probe()
    manager = get_memory_manager()

    return HealthResponse(
        status="ok",
        version="0.1.0",
        lm_studio_reachable=lm_reachable,
        projects_loaded=len(await manager.project_ids()),
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.post("/compress", response_model=CompressResponse)
async def compress(request: CompressRequest, settings: SettingsDep):
    t_start = time.time()
    request_id = request.metadata.request_id or str(uuid.uuid4())
    warnings: list[str] = []

    # Resolve project identity
    project_id = resolve_project_id(request.project_hint)

    # Get per-project memory stack
    try:
        manager = get_memory_manager()
        mem = await manager.get(project_id)
    except Exception as exc:
        raise ProjectInitError(f"Failed to initialise project {project_id}: {exc}") from exc

    # Embed the current message for scoring and retrieval
    embedding_provider = get_embedding_provider()
    # True original size = all incoming context turns + current message
    from cce.pipeline.chunker import estimate_tokens as _est
    raw_original_tokens = _est(request.current_message) + sum(
        _est(t.content) for t in request.recent_context
    )

    t_retrieval_start = time.time()
    try:
        query_embedding = await embedding_provider.embed_one(request.current_message)
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed: {exc}") from exc

    # Chunk and score incoming context, then route to memory tiers
    if request.recent_context:
        from cce.models.types import ConversationTurn, Role
        from cce.pipeline.chunker import Chunker
        from cce.pipeline.scorer import ImportanceScorer

        chunker = Chunker(settings)
        scorer = ImportanceScorer(settings)

        turns = [
            ConversationTurn(
                role=Role(t.role),
                content=t.content,
                turn_index=t.turn_index,
                timestamp=t.timestamp,
            )
            for t in request.recent_context
        ]
        chunks = chunker.chunk_turns(turns, project_id=project_id)

        # Embed all chunks in one batch
        embeddings = await embedding_provider.embed([c.content for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        current_turn = max((t.turn_index for t in turns), default=0)
        scored = scorer.score(chunks, query_embedding, request.current_message, current_turn)

        await mem.router.route(scored)

        # Enqueue compression jobs for WM/LTM chunks (async, off critical path)
        from cce.models.types import CompressionJob, CompressionLevel, MemoryTier
        for sc in scored:
            if sc.composite_score >= settings.router_wm_threshold and \
               sc.composite_score < settings.router_stm_threshold:
                job = CompressionJob(
                    job_id=str(uuid.uuid4()),
                    project_id=project_id,
                    chunks=[sc.chunk],
                    target_tier=MemoryTier.WM,
                    compression_level=CompressionLevel.LIGHT,
                )
                await mem.queue.enqueue(job)

    t_retrieval_ms = (time.time() - t_retrieval_start) * 1000

    # Assemble the optimized prompt
    t_assembly_start = time.time()
    assembler = get_assembler()
    package = await assembler.assemble(
        stm=mem.stm,
        wm=mem.wm,
        ltm=mem.ltm,
        query_embedding=query_embedding,
        current_message=request.current_message,
        max_context_tokens=request.metadata.max_context_tokens,
        original_token_count=raw_original_tokens,
    )
    t_assembly_ms = (time.time() - t_assembly_start) * 1000
    t_total_ms = (time.time() - t_start) * 1000

    return CompressResponse(
        request_id=request_id,
        project_id=project_id,
        optimized_prompt=package.optimized_prompt,
        original_token_estimate=package.original_token_estimate,
        compressed_token_estimate=package.compressed_token_estimate,
        compression_ratio=package.compression_ratio,
        memory_hits=MemoryHits(stm=package.stm_hits, wm=package.wm_hits, ltm=package.ltm_hits),
        latency_ms=LatencyBreakdown(
            total_ms=round(t_total_ms, 2),
            retrieval_ms=round(t_retrieval_ms, 2),
            assembly_ms=round(t_assembly_ms, 2),
        ),
        warnings=warnings,
    )


@router.post("/recall", response_model=RecallResponse)
async def recall(request: RecallRequest, settings: SettingsDep):
    t_start = time.time()
    request_id = str(uuid.uuid4())

    project_id = resolve_project_id(request.project_hint)
    try:
        manager = get_memory_manager()
        mem = await manager.get(project_id)
    except Exception as exc:
        raise ProjectInitError(f"Failed to initialise project {project_id}: {exc}") from exc

    embedding_provider = get_embedding_provider()
    query_text = request.query or _GENERIC_RECALL_QUERY
    try:
        query_embedding = await embedding_provider.embed_one(query_text)
    except Exception as exc:
        raise EmbeddingError(f"Embedding failed: {exc}") from exc

    assembler = get_assembler()
    briefing = await assembler.recall(
        stm=mem.stm,
        wm=mem.wm,
        ltm=mem.ltm,
        query_embedding=query_embedding,
        max_tokens=request.max_tokens,
    )

    from cce.pipeline.chunker import estimate_tokens
    wm_count = await mem.wm.count()
    ltm_count = await mem.ltm.count()
    stm_count = await mem.stm.count()

    return RecallResponse(
        request_id=request_id,
        project_id=project_id,
        briefing=briefing,
        token_estimate=estimate_tokens(briefing),
        memory_sources=MemorySources(
            stm_records=stm_count,
            wm_records=wm_count,
            ltm_records=ltm_count,
        ),
        latency_ms=round((time.time() - t_start) * 1000, 2),
    )


@router.get("/project/{project_id}/stats", response_model=ProjectStatsResponse)
async def project_stats(project_id: str, settings: SettingsDep):
    manager = get_memory_manager()
    mem = await manager.get(project_id)

    stm_count = await mem.stm.count()
    wm_count = await mem.wm.count()
    ltm_count = await mem.ltm.count()

    return ProjectStatsResponse(
        project_id=project_id,
        stm_records=stm_count,
        wm_records=wm_count,
        ltm_records=ltm_count,
        total_token_estimate=0,
        faiss_index_size=ltm_count,
        oldest_record_ts=None,
    )


@router.delete("/project/{project_id}/memory", response_model=ClearMemoryResponse)
async def clear_memory(project_id: str, settings: SettingsDep):
    manager = get_memory_manager()
    mem = await manager.get(project_id)

    stm_del = await mem.stm.clear()
    wm_del = await mem.wm.clear()
    ltm_del = await mem.ltm.clear()

    return ClearMemoryResponse(cleared=True, records_deleted=stm_del + wm_del + ltm_del)


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)
