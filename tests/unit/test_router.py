from __future__ import annotations

import time
import uuid

import numpy as np
import pytest

import time

from cce.memory.stm import ShortTermMemory
from cce.memory.wm import WorkingMemory
from cce.memory.ltm import LongTermMemory
from cce.models.types import Chunk, ChunkType, MemoryTier, ScoredChunk
from cce.pipeline.router import MemoryRouter
from cce.settings import Settings
from cce.storage.db import Database
from cce.storage.faiss_store import FaissStore


def _unit_vec(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(384).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _scored_chunk(score: float, content: str = "text") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=str(uuid.uuid4()),
        project_id="proj",
        content=content,
        content_type=ChunkType.PROSE,
        token_count=10,
        char_offset=0,
        turn_index=0,
        created_at=time.time(),
        embedding=_unit_vec(),
    )
    return ScoredChunk(
        chunk=chunk,
        recency_score=score,
        relevance_score=score,
        keyword_score=score,
        composite_score=score,
    )


@pytest.fixture
async def router(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    await db.connect()
    await db.execute(
        """
        INSERT OR IGNORE INTO projects
            (project_id, display_name, embedding_model, embedding_dim, created_at, last_accessed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("proj", "Test Project", "all-MiniLM-L6-v2", 384, time.time(), time.time()),
    )
    await db._conn.commit()

    faiss = FaissStore(tmp_path / "ltm.faiss", dimension=384)
    faiss.load()

    settings = Settings()
    stm = ShortTermMemory(max_turns=settings.stm_max_turns)
    wm = WorkingMemory(db, "proj", max_records=settings.wm_max_records)
    ltm = LongTermMemory(db, faiss, "proj", max_records=settings.ltm_max_records)

    yield MemoryRouter(stm, wm, ltm, settings, db=db)
    await db.close()


@pytest.mark.asyncio
async def test_router_high_score_goes_to_stm(router):
    sc = _scored_chunk(score=0.90)  # above stm_threshold=0.85
    result = await router.route([sc])
    assert result.stm == 1
    assert result.wm == 0
    assert result.ltm == 0


@pytest.mark.asyncio
async def test_router_mid_score_goes_to_wm(router):
    sc = _scored_chunk(score=0.65)  # between wm_threshold=0.50 and stm=0.85
    result = await router.route([sc])
    assert result.wm == 1
    assert result.stm == 0


@pytest.mark.asyncio
async def test_router_low_score_goes_to_ltm(router):
    sc = _scored_chunk(score=0.30)  # between discard=0.15 and wm=0.50
    result = await router.route([sc])
    assert result.ltm == 1


@pytest.mark.asyncio
async def test_router_very_low_score_discarded(router):
    sc = _scored_chunk(score=0.05)  # below discard_threshold=0.15
    result = await router.route([sc])
    assert result.discarded == 1
    assert result.stm == 0
    assert result.wm == 0
    assert result.ltm == 0


@pytest.mark.asyncio
async def test_router_mixed_batch(router):
    chunks = [
        _scored_chunk(0.90),  # STM
        _scored_chunk(0.65),  # WM
        _scored_chunk(0.30),  # LTM
        _scored_chunk(0.05),  # DISCARD
    ]
    result = await router.route(chunks)
    assert result.stm == 1
    assert result.wm == 1
    assert result.ltm == 1
    assert result.discarded == 1


@pytest.mark.asyncio
async def test_router_sets_importance_score_on_chunk(router):
    sc = _scored_chunk(score=0.77)
    await router.route([sc])
    assert sc.chunk.importance_score == pytest.approx(0.77)


@pytest.mark.asyncio
async def test_router_logs_chunks_to_database(router):
    sc = _scored_chunk(score=0.65, content="remember this chunk")
    result = await router.route([sc])
    assert result.wm == 1

    row = await router._db.fetchone("SELECT content, tier_assigned FROM chunks WHERE chunk_id = ?", (sc.chunk.chunk_id,))
    assert row is not None
    assert row["content"] == "remember this chunk"
    assert row["tier_assigned"] == "wm"
