from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# --- Shared sub-models ---

class ConversationTurnSchema(BaseModel):
    role: str
    content: str
    turn_index: int
    timestamp: float | None = None


class RequestMetadata(BaseModel):
    tool: str = "unknown"           # cursor | claude_code | codex | unknown
    model_target: str = "unknown"   # claude-3-5-sonnet | gpt-4o | ...
    max_context_tokens: int = 8192
    request_id: str | None = None


# --- /compress ---

class CompressRequest(BaseModel):
    project_hint: str = Field(..., description="Absolute path or git root hint")
    current_message: str = Field(..., description="The user's current message/prompt")
    recent_context: list[ConversationTurnSchema] = Field(default_factory=list)
    metadata: RequestMetadata = Field(default_factory=RequestMetadata)


class LatencyBreakdown(BaseModel):
    total_ms: float
    retrieval_ms: float
    assembly_ms: float


class MemoryHits(BaseModel):
    stm: int
    wm: int
    ltm: int


class CompressResponse(BaseModel):
    request_id: str
    project_id: str
    optimized_prompt: str
    original_token_estimate: int
    compressed_token_estimate: int
    compression_ratio: float
    memory_hits: MemoryHits
    latency_ms: LatencyBreakdown
    warnings: list[str] = Field(default_factory=list)


# --- /recall ---

class RecallRequest(BaseModel):
    project_hint: str
    query: str | None = None
    max_tokens: int = 2048
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySources(BaseModel):
    stm_records: int
    wm_records: int
    ltm_records: int


class RecallResponse(BaseModel):
    request_id: str
    project_id: str
    briefing: str
    token_estimate: int
    memory_sources: MemorySources
    latency_ms: float


# --- /health ---

class HealthResponse(BaseModel):
    status: str
    version: str
    lm_studio_reachable: bool
    embedding_ready: bool = True
    embedding_model: str = "unknown"
    embedding_error: str | None = None
    projects_loaded: int
    uptime_seconds: float


# --- /project/{project_id}/stats ---

class ProjectStatsResponse(BaseModel):
    project_id: str
    stm_records: int
    wm_records: int
    ltm_records: int
    total_token_estimate: int
    faiss_index_size: int
    oldest_record_ts: float | None


# --- /project/{project_id}/memory (DELETE) ---

class ClearMemoryResponse(BaseModel):
    cleared: bool
    records_deleted: int
