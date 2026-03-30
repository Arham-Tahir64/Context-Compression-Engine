from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from cce.settings import Settings
from cce.embeddings.base import EmbeddingProvider
from cce.embeddings.sentence_transformer import SentenceTransformerProvider


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    # Swap to LMStudioEmbeddingProvider here when ready
    return SentenceTransformerProvider()


@lru_cache(maxsize=1)
def get_memory_manager():
    from cce.memory.manager import MemoryManager
    return MemoryManager(get_settings(), get_embedding_provider())


@lru_cache(maxsize=1)
def get_assembler():
    from cce.pipeline.assembler import PromptAssembler
    return PromptAssembler(get_settings())


SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingDep = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
