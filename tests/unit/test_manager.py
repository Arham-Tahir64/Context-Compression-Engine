from __future__ import annotations

from pathlib import Path

import pytest

from cce.memory.manager import MemoryManager
from cce.settings import Settings


class FakeEmbeddingProvider:
    dimension = 384
    model_name = "fake"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]

    async def embed_one(self, text: str) -> list[float]:
        return [0.0] * self.dimension

    async def check_ready(self) -> tuple[bool, str | None]:
        return True, None


@pytest.mark.asyncio
async def test_manager_starts_single_worker_per_project(tmp_path):
    settings = Settings(data_dir=Path(tmp_path / "data"))
    manager = MemoryManager(settings, FakeEmbeddingProvider())

    first = await manager.get("proj")
    second = await manager.get("proj")

    assert first.worker_task is not None
    assert first.worker_task is second.worker_task
    assert not first.worker_task.done()

    await manager.shutdown()
    assert first.worker_task.done()
