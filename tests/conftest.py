from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest
from fastapi.testclient import TestClient

import cce.dependencies as dependencies
from cce.compression.compressor import Compressor
from cce.dependencies import get_assembler, get_embedding_provider, get_memory_manager, get_settings
from cce.main import app


class FakeEmbeddingProvider:
    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "fake-all-MiniLM-L6-v2"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    async def embed_one(self, text: str) -> list[float]:
        return self._vectorize(text)

    async def check_ready(self) -> tuple[bool, str | None]:
        return True, None

    def _vectorize(self, text: str) -> list[float]:
        digest = sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "cce-data"))
    monkeypatch.setenv("LM_STUDIO_MODEL", "test-model")
    monkeypatch.setattr(dependencies, "SentenceTransformerProvider", FakeEmbeddingProvider)

    async def fake_probe(self) -> bool:
        return True

    async def fake_call(self, content: str, level) -> str:
        prefix = "light" if level.value == "light" else "heavy"
        summary = content.strip().splitlines()[0][:120] if content.strip() else "empty"
        return f"{prefix} summary: {summary}"

    monkeypatch.setattr(Compressor, "probe", fake_probe)
    monkeypatch.setattr(Compressor, "_call_lm_studio", fake_call)

    get_settings.cache_clear()
    get_embedding_provider.cache_clear()
    get_memory_manager.cache_clear()
    get_assembler.cache_clear()

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    get_embedding_provider.cache_clear()
    get_memory_manager.cache_clear()
    get_assembler.cache_clear()
