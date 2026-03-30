from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass

from cce.models.types import Chunk, ChunkType, ConversationTurn, Role
from cce.settings import Settings


# Matches opening code fence: ```optional-lang
_CODE_FENCE_OPEN = re.compile(r"^```[\w]*\s*$", re.MULTILINE)
_CODE_FENCE_CLOSE = re.compile(r"^```\s*$", re.MULTILINE)

# Simple sentence boundary: end of sentence followed by whitespace + capital letter
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Stopwords for keyword extraction (compact list sufficient for scoring)
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "i", "you", "we",
    "he", "she", "they", "me", "him", "her", "us", "them", "my", "your",
    "our", "their", "what", "which", "who", "how", "when", "where", "why",
    "if", "then", "else", "not", "no", "so", "up", "out", "about", "into",
    "just", "also", "more", "than", "other", "some", "any", "all", "each",
})


def estimate_tokens(text: str) -> int:
    """Fast token estimate: word count * 1.3 (skews higher for code)."""
    return max(1, int(len(text.split()) * 1.3))


def extract_keywords(text: str) -> frozenset[str]:
    """Extract lowercase alphanum tokens > 3 chars, minus stopwords."""
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text)
    return frozenset(t.lower() for t in tokens if t.lower() not in _STOPWORDS)


class Chunker:
    """Splits raw context (conversation turns or plain text) into Chunks.

    Two-pass strategy:
      Pass 1 — structural split: code fences → paragraph breaks → turn boundaries
      Pass 2 — size enforcement: oversized chunks are split on sentence boundaries
    """

    def __init__(self, settings: Settings):
        self._max_prose = settings.max_chunk_tokens_prose
        self._max_code = settings.max_chunk_tokens_code

    def chunk_turns(
        self,
        turns: list[ConversationTurn],
        project_id: str,
    ) -> list[Chunk]:
        """Primary entry point: chunk a list of conversation turns."""
        chunks: list[Chunk] = []
        for turn in turns:
            turn_chunks = self._chunk_text(
                text=turn.content,
                project_id=project_id,
                turn_index=turn.turn_index,
                created_at=turn.timestamp or time.time(),
                role=turn.role,
            )
            chunks.extend(turn_chunks)
        return chunks

    def chunk_text(
        self,
        text: str,
        project_id: str,
        turn_index: int = 0,
        created_at: float | None = None,
    ) -> list[Chunk]:
        """Chunk a raw string (no role context)."""
        return self._chunk_text(
            text=text,
            project_id=project_id,
            turn_index=turn_index,
            created_at=created_at or time.time(),
            role=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_text(
        self,
        text: str,
        project_id: str,
        turn_index: int,
        created_at: float,
        role: Role | None,
    ) -> list[Chunk]:
        segments = self._structural_split(text)
        chunks: list[Chunk] = []
        char_offset = 0

        for content, content_type in segments:
            max_tokens = (
                self._max_code if content_type == ChunkType.CODE else self._max_prose
            )
            sub_segments = self._enforce_size(content, max_tokens, content_type)
            for sub in sub_segments:
                if not sub.strip():
                    char_offset += len(sub)
                    continue
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    project_id=project_id,
                    content=sub.strip(),
                    content_type=content_type,
                    token_count=estimate_tokens(sub),
                    char_offset=char_offset,
                    turn_index=turn_index,
                    created_at=created_at,
                    metadata={"role": role.value if role else None},
                )
                chunks.append(chunk)
                char_offset += len(sub)

        return chunks

    def _structural_split(self, text: str) -> list[tuple[str, ChunkType]]:
        """Split text into (content, type) segments by detecting code fences."""
        segments: list[tuple[str, ChunkType]] = []
        remaining = text
        pos = 0

        while pos < len(remaining):
            open_match = _CODE_FENCE_OPEN.search(remaining, pos)
            if open_match is None:
                # No more code blocks — rest is prose
                prose = remaining[pos:]
                if prose.strip():
                    segments.extend(self._split_prose(prose))
                break

            # Prose before the code fence
            prose_before = remaining[pos:open_match.start()]
            if prose_before.strip():
                segments.extend(self._split_prose(prose_before))

            # Find the closing fence; if absent, treat everything to EOF as code
            close_match = _CODE_FENCE_CLOSE.search(remaining, open_match.end())
            if close_match is None:
                code_content = remaining[open_match.end():]
                segments.append((code_content, ChunkType.CODE))
                pos = len(remaining)
            else:
                code_content = remaining[open_match.end():close_match.start()]
                segments.append((code_content, ChunkType.CODE))
                pos = close_match.end()

        return segments

    def _split_prose(self, text: str) -> list[tuple[str, ChunkType]]:
        """Split prose on double-newline paragraph boundaries."""
        paragraphs = re.split(r"\n{2,}", text)
        return [(p, ChunkType.PROSE) for p in paragraphs if p.strip()]

    def _enforce_size(
        self,
        text: str,
        max_tokens: int,
        content_type: ChunkType,
    ) -> list[str]:
        """If text exceeds max_tokens, split on sentence boundaries."""
        if estimate_tokens(text) <= max_tokens:
            return [text]

        if content_type == ChunkType.CODE:
            # Split code on blank lines rather than sentences
            lines = text.split("\n")
            return self._pack_lines(lines, max_tokens)

        sentences = _SENTENCE_SPLIT.split(text)
        return self._pack_lines(sentences, max_tokens)

    def _pack_lines(self, lines: list[str], max_tokens: int) -> list[str]:
        """Greedily pack lines into chunks up to max_tokens each."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = estimate_tokens(line)
            if current_tokens + line_tokens > max_tokens and current:
                chunks.append("\n".join(current))
                current = [line]
                current_tokens = line_tokens
            else:
                current.append(line)
                current_tokens += line_tokens

        if current:
            chunks.append("\n".join(current))

        return chunks
