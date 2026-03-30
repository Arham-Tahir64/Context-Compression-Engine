from __future__ import annotations

import asyncio
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from cce.memory.stm import ShortTermMemory
from cce.memory.wm import WorkingMemory
from cce.memory.ltm import LongTermMemory
from cce.models.types import MemoryRecord, MemoryTier
from cce.storage.db import Database
from cce.storage.faiss_store import FaissStore


# --- Fixtures ---

def _record(project_id: str = "proj", score: float = 0.5, content: str = "test content", embedding: list[float] | None = None) -> MemoryRecord:
    vec = embedding or _unit_vec()
    return MemoryRecord(
        record_id=str(uuid.uuid4()),
        project_id=project_id,
        content=content,
        original_token_count=20,
        compressed_token_count=10,
        tier=MemoryTier.WM,
        source_chunk_ids=["c1"],
        embedding=vec,
        importance_score=score,
        created_at=time.time(),
        last_accessed_at=time.time(),
    )


def _unit_vec(dim: int = 384, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
async def db(tmp_path):
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    await database.connect()
    # Insert a project row so FK constraints on wm_records/ltm_records pass
    await database.execute(
        """
        INSERT OR IGNORE INTO projects
            (project_id, display_name, embedding_model, embedding_dim, created_at, last_accessed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("proj", "Test Project", "all-MiniLM-L6-v2", 384, time.time(), time.time()),
    )
    await database._conn.commit()
    yield database
    await database.close()


@pytest.fixture
def faiss_store(tmp_path):
    store = FaissStore(tmp_path / "test.faiss", dimension=384)
    store.load()
    return store


# ===== STM =====

@pytest.mark.asyncio
async def test_stm_write_and_query():
    stm = ShortTermMemory(max_turns=5)
    r = _record()
    await stm.write(r)
    results = await stm.query(_unit_vec(), top_k=10)
    assert len(results) == 1
    assert results[0].record_id == r.record_id


@pytest.mark.asyncio
async def test_stm_respects_max_turns():
    stm = ShortTermMemory(max_turns=3)
    for _ in range(5):
        await stm.write(_record())
    assert await stm.count() == 3


@pytest.mark.asyncio
async def test_stm_query_newest_first():
    stm = ShortTermMemory(max_turns=10)
    ids = []
    for i in range(3):
        r = _record(content=f"turn {i}")
        ids.append(r.record_id)
        await stm.write(r)
    results = await stm.query(_unit_vec(), top_k=10)
    # Newest written should be first
    assert results[0].record_id == ids[-1]


@pytest.mark.asyncio
async def test_stm_clear():
    stm = ShortTermMemory()
    await stm.write(_record())
    await stm.write(_record())
    deleted = await stm.clear()
    assert deleted == 2
    assert await stm.count() == 0


# ===== WM =====

@pytest.mark.asyncio
async def test_wm_write_and_query(db):
    wm = WorkingMemory(db, project_id="proj", max_records=10)
    r = _record(embedding=_unit_vec(seed=1))
    await wm.write(r)
    results = await wm.query(_unit_vec(seed=1), top_k=5)
    assert len(results) == 1
    assert results[0].record_id == r.record_id


@pytest.mark.asyncio
async def test_wm_evicts_lowest_score(db):
    wm = WorkingMemory(db, project_id="proj", max_records=2)
    low = _record(score=0.1, content="low score")
    high1 = _record(score=0.9, content="high score 1")
    high2 = _record(score=0.8, content="high score 2")
    await wm.write(low)
    await wm.write(high1)
    # This write triggers eviction of lowest (low)
    await wm.write(high2)
    assert await wm.count() == 2
    results = await wm.query(_unit_vec(), top_k=10)
    contents = [r.content for r in results]
    assert "low score" not in contents


@pytest.mark.asyncio
async def test_wm_upsert_same_id(db):
    wm = WorkingMemory(db, project_id="proj", max_records=10)
    r = _record()
    await wm.write(r)
    # Write same record_id with updated content
    r2 = MemoryRecord(
        record_id=r.record_id,
        project_id=r.project_id,
        content="updated content",
        original_token_count=r.original_token_count,
        compressed_token_count=5,
        tier=MemoryTier.WM,
        source_chunk_ids=r.source_chunk_ids,
        embedding=r.embedding,
        importance_score=0.9,
        created_at=r.created_at,
        last_accessed_at=time.time(),
    )
    await wm.write(r2)
    assert await wm.count() == 1
    results = await wm.query(_unit_vec(), top_k=1)
    assert results[0].content == "updated content"


@pytest.mark.asyncio
async def test_wm_clear(db):
    wm = WorkingMemory(db, project_id="proj", max_records=10)
    for _ in range(3):
        await wm.write(_record())
    deleted = await wm.clear()
    assert deleted == 3
    assert await wm.count() == 0


@pytest.mark.asyncio
async def test_wm_query_ranks_by_similarity(db):
    wm = WorkingMemory(db, project_id="proj", max_records=10)
    query_vec = _unit_vec(seed=99)
    # Record with same vector as query should rank first
    matching = _record(content="matching", embedding=query_vec)
    unrelated = _record(content="unrelated", embedding=_unit_vec(seed=7))
    await wm.write(matching)
    await wm.write(unrelated)
    results = await wm.query(query_vec, top_k=2)
    assert results[0].content == "matching"


# ===== LTM =====

@pytest.mark.asyncio
async def test_ltm_write_and_query(db, faiss_store):
    ltm = LongTermMemory(db, faiss_store, project_id="proj")
    vec = _unit_vec(seed=5)
    r = _record(embedding=vec)
    await ltm.write(r)
    results = await ltm.query(vec, top_k=5)
    assert len(results) == 1
    assert results[0].record_id == r.record_id


@pytest.mark.asyncio
async def test_ltm_faiss_index_persists(tmp_path, db):
    index_path = tmp_path / "ltm.faiss"
    store1 = FaissStore(index_path, dimension=384)
    store1.load()
    ltm1 = LongTermMemory(db, store1, project_id="proj")
    vec = _unit_vec(seed=10)
    r = _record(embedding=vec)
    await ltm1.write(r)

    # Load a fresh store from the same path
    store2 = FaissStore(index_path, dimension=384)
    store2.load()
    assert store2.ntotal == 1


@pytest.mark.asyncio
async def test_ltm_empty_query_returns_empty(db, faiss_store):
    ltm = LongTermMemory(db, faiss_store, project_id="proj")
    results = await ltm.query(_unit_vec(), top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_ltm_clear(db, faiss_store):
    ltm = LongTermMemory(db, faiss_store, project_id="proj")
    await ltm.write(_record(embedding=_unit_vec(seed=1)))
    await ltm.write(_record(embedding=_unit_vec(seed=2)))
    deleted = await ltm.clear()
    assert deleted == 2
    assert await ltm.count() == 0
