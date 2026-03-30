from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from cce.compression.compressor import Compressor
from cce.compression.queue import CompressionQueue
from cce.models.types import Chunk, ChunkType, CompressionJob, CompressionLevel, MemoryRecord, MemoryTier
from cce.settings import Settings
from cce.storage.db import Database
from cce.storage.faiss_store import FaissStore
from cce.memory.wm import WorkingMemory
from cce.memory.ltm import LongTermMemory


# --- Helpers ---

def _unit_vec(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(384).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _chunk(content: str = "def foo(): return 42") -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        project_id="proj",
        content=content,
        content_type=ChunkType.CODE,
        token_count=10,
        char_offset=0,
        turn_index=0,
        created_at=time.time(),
        embedding=_unit_vec(),
        importance_score=0.6,
    )


def _job(level: CompressionLevel = CompressionLevel.LIGHT, tier: MemoryTier = MemoryTier.WM) -> CompressionJob:
    return CompressionJob(
        job_id=str(uuid.uuid4()),
        project_id="proj",
        chunks=[_chunk("def calculate(x): return x * 2")],
        target_tier=tier,
        compression_level=level,
    )


def _memory_record(
    *,
    tier: MemoryTier,
    content: str,
    embedding: list[float] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        record_id=str(uuid.uuid4()),
        project_id="proj",
        content=content,
        original_token_count=20,
        compressed_token_count=10,
        tier=tier,
        source_chunk_ids=["seed"],
        embedding=embedding or _unit_vec(seed=5),
        importance_score=0.7,
        created_at=time.time(),
        last_accessed_at=time.time(),
    )


@pytest.fixture
async def stores(tmp_path):
    db = Database(tmp_path / "test.db")
    await db.connect()
    await db.execute(
        "INSERT OR IGNORE INTO projects (project_id, display_name, embedding_model, embedding_dim, created_at, last_accessed_at) VALUES (?,?,?,?,?,?)",
        ("proj", "Test", "all-MiniLM-L6-v2", 384, time.time(), time.time()),
    )
    await db._conn.commit()
    faiss = FaissStore(tmp_path / "ltm.faiss", dimension=384)
    faiss.load()
    wm = WorkingMemory(db, "proj")
    ltm = LongTermMemory(db, faiss, "proj")
    yield wm, ltm
    await db.close()


# ===== Compressor: LM Studio happy path =====

@pytest.mark.asyncio
async def test_compressor_calls_lm_studio_and_writes_wm(stores):
    wm, ltm = stores
    settings = Settings()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Compressed: calculates double of x."}}]
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        compressor = Compressor(settings, wm, ltm)
        record = await compressor.compress(_job(level=CompressionLevel.LIGHT, tier=MemoryTier.WM))

    assert record is not None
    assert record.content == "Compressed: calculates double of x."
    assert await wm.count() == 1


@pytest.mark.asyncio
async def test_compressor_writes_ltm_when_tier_is_ltm(stores):
    wm, ltm = stores
    settings = Settings()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Heavy compressed summary."}}]
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        compressor = Compressor(settings, wm, ltm)
        record = await compressor.compress(_job(level=CompressionLevel.HEAVY, tier=MemoryTier.LTM))

    assert record is not None
    assert await ltm.count() == 1
    assert await wm.count() == 0


@pytest.mark.asyncio
async def test_compressor_updates_existing_wm_record_in_place(stores):
    wm, ltm = stores
    settings = Settings()

    existing = _memory_record(tier=MemoryTier.WM, content="raw wm content")
    await wm.write(existing)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "compressed wm update"}}]
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        compressor = Compressor(settings, wm, ltm)
        job = _job(level=CompressionLevel.LIGHT, tier=MemoryTier.WM)
        job.target_record_id = existing.record_id
        await compressor.compress(job)

    assert await wm.count() == 1
    results = await wm.query(_unit_vec(), top_k=1)
    assert results[0].record_id == existing.record_id
    assert results[0].content == "compressed wm update"


@pytest.mark.asyncio
async def test_compressor_updates_existing_ltm_record_in_place(stores):
    wm, ltm = stores
    settings = Settings()

    existing = _memory_record(
        tier=MemoryTier.LTM,
        content="raw ltm content",
        embedding=_unit_vec(seed=9),
    )
    await ltm.write(existing)
    original_index_size = ltm.faiss_index_size

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "compressed ltm update"}}]
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        compressor = Compressor(settings, wm, ltm)
        job = _job(level=CompressionLevel.HEAVY, tier=MemoryTier.LTM)
        job.target_record_id = existing.record_id
        await compressor.compress(job)

    assert await ltm.count() == 1
    assert ltm.faiss_index_size == original_index_size
    results = await ltm.query(existing.embedding, top_k=1)
    assert results[0].record_id == existing.record_id
    assert results[0].content == "compressed ltm update"


# ===== Compressor: fallback when LM Studio unreachable =====

@pytest.mark.asyncio
async def test_compressor_falls_back_on_connect_error(stores):
    import httpx as _httpx
    wm, ltm = stores
    settings = Settings()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=_httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        compressor = Compressor(settings, wm, ltm)
        record = await compressor.compress(_job())

    # Fallback should still produce a record (truncated content)
    assert record is not None
    assert len(record.content) > 0
    assert await wm.count() == 1


@pytest.mark.asyncio
async def test_compressor_falls_back_on_timeout(stores):
    import httpx as _httpx
    wm, ltm = stores
    settings = Settings()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=_httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        compressor = Compressor(settings, wm, ltm)
        record = await compressor.compress(_job())

    assert record is not None


# ===== Extractive fallback =====

@pytest.mark.asyncio
async def test_extractive_fallback_truncates(stores):
    wm, ltm = stores
    settings = Settings()
    compressor = Compressor(settings, wm, ltm)
    long_text = "word " * 500
    result = compressor._extractive_fallback(long_text, max_tokens=50)
    assert len(result.split()) < 60  # 50/1.3 ≈ 38 words + ellipsis
    assert result.endswith("…")


@pytest.mark.asyncio
async def test_extractive_fallback_no_ellipsis_when_short(stores):
    wm, ltm = stores
    settings = Settings()
    compressor = Compressor(settings, wm, ltm)
    result = compressor._extractive_fallback("short text", max_tokens=200)
    assert not result.endswith("…")


# ===== Compression Queue =====

@pytest.mark.asyncio
async def test_queue_enqueue_and_drain(stores):
    wm, ltm = stores
    settings = Settings()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "summary"}}]
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        compressor = Compressor(settings, wm, ltm)
        queue = CompressionQueue(compressor, maxsize=10)

        await queue.enqueue(_job())

        # Run the drain worker briefly — process one job then stop
        worker_task = asyncio.create_task(queue.drain_worker())
        await asyncio.sleep(0.05)
        await queue.stop()
        await asyncio.wait_for(worker_task, timeout=2.0)

    assert await wm.count() == 1


@pytest.mark.asyncio
async def test_queue_drops_when_full(stores):
    wm, ltm = stores
    settings = Settings()
    compressor = Compressor(settings, wm, ltm)
    queue = CompressionQueue(compressor, maxsize=2)

    # Fill the queue without draining
    await queue.enqueue(_job())
    await queue.enqueue(_job())
    await queue.enqueue(_job())  # should drop silently

    assert queue.qsize == 2
