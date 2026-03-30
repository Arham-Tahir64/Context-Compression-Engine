from __future__ import annotations

import logging
import time
import uuid

import httpx

from cce.memory.ltm import LongTermMemory
from cce.memory.wm import WorkingMemory
from cce.models.types import (
    CompressionJob,
    CompressionLevel,
    MemoryRecord,
    MemoryTier,
)
from cce.pipeline.chunker import estimate_tokens
from cce.settings import Settings

logger = logging.getLogger(__name__)

_LIGHT_PROMPT = (
    "Summarize the following in 1-2 sentences. "
    "Preserve all function names, variable names, file paths, and technical decisions. "
    "Output only the summary, no preamble.\n\n"
    "{content}"
)

_HEAVY_PROMPT = (
    "Compress the following into a concise technical memory. "
    "Preserve: key decisions, function/class names, file paths, error messages, constraints. "
    "Discard: conversational filler, examples, repetition. "
    "Output only the compressed memory, no preamble.\n\n"
    "{content}"
)


class Compressor:
    """Calls LM Studio to summarize chunks and writes results back to memory.

    Falls back to extractive truncation if LM Studio is unreachable, so
    the system degrades gracefully when the local model is not running.
    """

    def __init__(
        self,
        settings: Settings,
        wm: WorkingMemory,
        ltm: LongTermMemory,
    ):
        self._base_url = settings.lm_studio_base_url.rstrip("/")
        self._model = settings.lm_studio_model
        self._timeout = settings.lm_studio_timeout_seconds
        self._wm = wm
        self._ltm = ltm

    async def compress(self, job: CompressionJob) -> MemoryRecord | None:
        """Compress a job's chunks into a single MemoryRecord and write to tier."""
        combined = "\n\n".join(c.content for c in job.chunks)
        original_tokens = sum(c.token_count for c in job.chunks)

        compressed_text = await self._call_lm_studio(combined, job.compression_level)
        compressed_tokens = estimate_tokens(compressed_text)

        # Build the embedding from the first chunk that has one
        embedding = next(
            (c.embedding for c in job.chunks if c.embedding),
            [],
        )

        record = MemoryRecord(
            record_id=str(uuid.uuid4()),
            project_id=job.project_id,
            content=compressed_text,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            tier=job.target_tier,
            source_chunk_ids=[c.chunk_id for c in job.chunks],
            embedding=embedding,
            importance_score=max((c.importance_score for c in job.chunks), default=0.0),
            created_at=time.time(),
            last_accessed_at=time.time(),
        )

        if job.target_tier == MemoryTier.WM:
            await self._wm.write(record)
        elif job.target_tier == MemoryTier.LTM:
            await self._ltm.write(record)

        ratio = original_tokens / max(compressed_tokens, 1)
        logger.debug(
            "Compressed %d chunks → %d tokens (%.1fx) for project %s",
            len(job.chunks),
            compressed_tokens,
            ratio,
            job.project_id,
        )
        return record

    async def probe(self) -> bool:
        """Check whether LM Studio is reachable. Used by /health."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self._base_url}/models")
                return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------

    async def _call_lm_studio(self, content: str, level: CompressionLevel) -> str:
        template = _LIGHT_PROMPT if level == CompressionLevel.LIGHT else _HEAVY_PROMPT
        prompt = template.format(content=content)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 512,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()

        except httpx.ConnectError:
            logger.warning("LM Studio unreachable — falling back to truncation")
            return self._extractive_fallback(content)
        except httpx.TimeoutException:
            logger.warning("LM Studio timed out — falling back to truncation")
            return self._extractive_fallback(content)
        except Exception as exc:
            logger.warning("LM Studio error (%s) — falling back to truncation", exc)
            return self._extractive_fallback(content)

    def _extractive_fallback(self, content: str, max_tokens: int = 200) -> str:
        """Keep the first max_tokens worth of content when LM Studio is down."""
        words = content.split()
        kept = int(max_tokens / 1.3)
        return " ".join(words[:kept]) + (" …" if len(words) > kept else "")
