import time
from cce.models.types import (
    Chunk, ChunkType, MemoryRecord, MemoryTier,
    ScoredChunk, PromptPackage, ConversationTurn, Role,
)


def test_chunk_instantiation():
    c = Chunk(
        chunk_id="abc",
        project_id="proj1",
        content="def foo(): pass",
        content_type=ChunkType.CODE,
        token_count=5,
        char_offset=0,
        turn_index=0,
        created_at=time.time(),
    )
    assert c.importance_score == 0.0
    assert c.embedding is None
    assert c.metadata == {}


def test_scored_chunk():
    c = Chunk("id", "p", "text", ChunkType.PROSE, 10, 0, 0, time.time())
    sc = ScoredChunk(
        chunk=c,
        recency_score=0.8,
        relevance_score=0.7,
        keyword_score=0.5,
        composite_score=0.72,
    )
    assert sc.composite_score == 0.72


def test_memory_record():
    r = MemoryRecord(
        record_id="r1",
        project_id="proj1",
        content="summary text",
        original_token_count=200,
        compressed_token_count=40,
        tier=MemoryTier.LTM,
        source_chunk_ids=["c1", "c2"],
        embedding=[0.1] * 384,
        importance_score=0.6,
        created_at=time.time(),
        last_accessed_at=time.time(),
    )
    assert r.tier == MemoryTier.LTM
    assert len(r.source_chunk_ids) == 2


def test_conversation_turn():
    t = ConversationTurn(role=Role.USER, content="hello", turn_index=0)
    assert t.role == Role.USER
