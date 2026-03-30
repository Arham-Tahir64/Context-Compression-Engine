from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkType(str, Enum):
    CODE = "code"
    PROSE = "prose"
    TOOL_CALL = "tool_call"
    SYSTEM = "system"


class MemoryTier(str, Enum):
    STM = "stm"
    WM = "wm"
    LTM = "ltm"
    DISCARD = "discard"


class CompressionLevel(str, Enum):
    NONE = "none"
    LIGHT = "light"
    HEAVY = "heavy"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ConversationTurn:
    role: Role
    content: str
    turn_index: int
    timestamp: float | None = None


@dataclass
class Chunk:
    chunk_id: str
    project_id: str
    content: str
    content_type: ChunkType
    token_count: int
    char_offset: int
    turn_index: int
    created_at: float
    embedding: list[float] | None = None
    importance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    recency_score: float
    relevance_score: float
    keyword_score: float
    composite_score: float


@dataclass
class MemoryRecord:
    record_id: str
    project_id: str
    content: str
    original_token_count: int
    compressed_token_count: int
    tier: MemoryTier
    source_chunk_ids: list[str]
    embedding: list[float]
    importance_score: float
    created_at: float
    last_accessed_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionJob:
    job_id: str
    project_id: str
    chunks: list[Chunk]
    target_tier: MemoryTier
    compression_level: CompressionLevel


@dataclass
class PromptPackage:
    project_id: str
    optimized_prompt: str
    original_token_estimate: int
    compressed_token_estimate: int
    compression_ratio: float
    stm_hits: int
    wm_hits: int
    ltm_hits: int


@dataclass
class RetrievalLatency:
    total_ms: float
    retrieval_ms: float
    assembly_ms: float
